// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use crate::{
    common::FlexibleKind,
    problem::{
        berth::{Berth, BerthIdentifier},
        builder::ProblemBuilder,
        err::{ProblemLoaderError, RequestError},
        prob::Problem,
        req::{Request, RequestIdentifier},
    },
};
use berth_alloc_core::prelude::*;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProblemLoader {
    forbid_at_least: i64,
    fail_on_unassignable: bool,
}

impl Default for ProblemLoader {
    fn default() -> Self {
        Self {
            forbid_at_least: 99999,
            fail_on_unassignable: true,
        }
    }
}

impl ProblemLoader {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn forbid_at_least(mut self, v: i64) -> Self {
        self.forbid_at_least = v;
        self
    }

    #[inline]
    pub fn fail_on_unassignable(mut self, yes: bool) -> Self {
        self.fail_on_unassignable = yes;
        self
    }

    pub fn from_bufread<R: BufRead>(&self, mut br: R) -> Result<Problem<i64>, ProblemLoaderError> {
        let mut sc = Scanner::new(&mut br);
        let n = sc.next_i64()? as usize;
        let m = sc.next_i64()? as usize;
        if n == 0 || m == 0 {
            return Err(ProblemLoaderError::NonPositiveCounts);
        }

        let mut ta = Vec::with_capacity(n);
        for _ in 0..n {
            ta.push(sc.next_i64()?);
        }

        let mut s = Vec::with_capacity(m);
        for _ in 0..m {
            s.push(sc.next_i64()?);
        }

        let mut h = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(m);
            for _ in 0..m {
                row.push(sc.next_i64()?);
            }
            h.push(row);
        }

        let mut e = Vec::with_capacity(m);
        for _ in 0..m {
            e.push(sc.next_i64()?);
        }

        let mut tmax = Vec::with_capacity(n);
        for _ in 0..n {
            tmax.push(sc.next_i64()?);
        }

        let mut builder = ProblemBuilder::new();

        for j in 0..m {
            let id = BerthIdentifier::new(j + 1);
            let sj = TimePoint::new(s[j]);
            let ej = TimePoint::new(e[j]);
            builder.add_berth(Berth::from_windows(id, [TimeInterval::new(sj, ej)]));
        }

        for i in 0..n {
            let rid = RequestIdentifier::new(i + 1);
            let a = TimePoint::new(ta[i]);
            let dmax = TimePoint::new(tmax[i]);
            let window = TimeInterval::new(a, dmax);

            let mut pt = BTreeMap::new();
            for j in 0..m {
                let hij = h[i][j];
                if hij >= self.forbid_at_least {
                    continue;
                }
                pt.insert(BerthIdentifier::new(j + 1), TimeDelta::new(hij));
            }

            // No weights given in the input format, so we use a uniform weight of 1.
            match Request::<FlexibleKind, i64>::new_flexible(rid, window, 1, pt) {
                Ok(req) => {
                    builder.add_flexible(req);
                }
                Err(RequestError::NoFeasibleAssignment(_)) if !self.fail_on_unassignable => {}
                Err(RequestError::NoFeasibleAssignment(_)) => {
                    return Err(ProblemLoaderError::NoFeasibleRequest(rid));
                }
                Err(e) => return Err(ProblemLoaderError::Request(e)),
            }
        }

        Ok(builder.build()?)
    }

    #[inline]
    pub fn from_path(&self, path: impl AsRef<Path>) -> Result<Problem<i64>, ProblemLoaderError> {
        let file = File::open(path).map_err(ProblemLoaderError::Io)?;
        let br = BufReader::new(file);
        self.from_bufread(br)
    }

    #[inline]
    pub fn from_reader<R: Read>(&self, r: R) -> Result<Problem<i64>, ProblemLoaderError> {
        self.from_bufread(BufReader::new(r))
    }

    #[inline]
    pub fn from_str(&self, s: &str) -> Result<Problem<i64>, ProblemLoaderError> {
        self.from_reader(s.as_bytes())
    }
}

#[derive(Debug)]
struct Scanner<R: BufRead> {
    rdr: R,
    buf: String,
    pos: usize,
}

impl<R: BufRead> Scanner<R> {
    fn new(rdr: R) -> Self {
        Self {
            rdr,
            buf: String::new(),
            pos: 0,
        }
    }

    #[inline]
    fn fill_line(&mut self) -> Result<(), ProblemLoaderError> {
        self.buf.clear();
        self.pos = 0;
        let n = self
            .rdr
            .read_line(&mut self.buf)
            .map_err(ProblemLoaderError::Io)?;
        if n == 0 {
            return Err(ProblemLoaderError::UnexpectedEof);
        }
        Ok(())
    }

    #[inline]
    fn skip_ws(&mut self) -> Result<(), ProblemLoaderError> {
        loop {
            if self.pos >= self.buf.len() {
                self.fill_line()?;
                continue;
            }
            while self.pos < self.buf.len() && self.buf.as_bytes()[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos >= self.buf.len() {
                continue;
            }
            return Ok(());
        }
    }

    #[inline]
    fn next_i64(&mut self) -> Result<i64, ProblemLoaderError> {
        self.skip_ws()?;
        let start = self.pos;
        while self.pos < self.buf.len() && !self.buf.as_bytes()[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        let tok = &self.buf[start..self.pos];
        tok.parse::<i64>().map_err(ProblemLoaderError::ParseInt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL_OK: &str = r#"
        2
        2
        0  10
        0  5
        3  1000000
        4  6
        10 20
        10  20
    "#;

    #[test]
    fn test_loads_minimal_and_prunes_forbidden() {
        let loader = ProblemLoader::new().forbid_at_least(1_000_000);
        let p: Problem<i64> = loader.from_str(SMALL_OK).unwrap();

        // 2 berths defined
        assert_eq!(p.berths().len(), 2);

        // Requests: vessel 1 -> only berth 1 feasible (3 < forbid), berth 2 is forbidden (1e6)
        // vessel 2 -> both feasible (4,6)
        assert_eq!(p.flexible_requests().len(), 2);
    }

    #[test]
    fn test_load_all_instances_from_workspace_root_instances_folder() {
        use std::fs;
        use std::path::{Path, PathBuf};

        // Find the nearest ancestor that contains an `instances/` directory.
        fn find_instances_dir() -> Option<PathBuf> {
            let mut cur: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
            while let Some(p) = cur {
                let cand = p.join("instances");
                if cand.is_dir() {
                    return Some(cand);
                }
                cur = p.parent();
            }
            None
        }

        let inst_dir = find_instances_dir().expect(
            "Could not find an `instances/` directory in any ancestor of CARGO_MANIFEST_DIR",
        );

        // Gather all .txt files (ignore subdirs/other files).
        let mut files: Vec<PathBuf> = fs::read_dir(&inst_dir)
            .expect("read_dir(instances) failed")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                    && e.path().extension().map(|x| x == "txt").unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        files.sort();

        assert!(
            !files.is_empty(),
            "No .txt instance files found in {}",
            inst_dir.display()
        );

        let loader = ProblemLoader::default();

        for path in files {
            eprintln!("Loading instance: {}", path.display());
            let problem = loader
                .from_path(&path)
                .unwrap_or_else(|e| panic!("Failed to load {}: {e}", path.display()));

            // Sanity checks: there should be at least one berth and one request in real instances.
            assert!(
                !problem.berths().is_empty(),
                "No berths parsed in {}",
                path.display()
            );
            assert!(
                !problem.flexible_requests().is_empty(),
                "No flexible requests parsed in {}",
                path.display()
            );
        }
    }
}
