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

use berth_alloc_model::prelude::{Problem, SolutionView};
use berth_alloc_model::problem::loader::ProblemLoader;
use berth_alloc_solver::engine::solver_engine::SolverEngineBuilder;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

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

#[allow(dead_code)]
fn instances() -> impl Iterator<Item = (Problem<i64>, String)> {
    let inst_dir = find_instances_dir()
        .expect("Could not find an `instances/` directory in any ancestor of CARGO_MANIFEST_DIR");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&inst_dir)
        .expect("read_dir(instances) failed")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && e.path().extension().map(|x| x == "txt").unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    files.sort();
    files.into_iter().filter_map(|f| {
        let loader = ProblemLoader::default();
        match loader.from_path(&f) {
            Ok(problem) => {
                let name = f
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| f.to_string_lossy().into_owned());
                Some((problem, name))
            }
            Err(_) => None,
        }
    })
}

#[allow(dead_code)]
fn enable_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE)
        .init();
}

fn main() {
    enable_tracing();

    for (problem, file) in instances().take(1) {
        tracing::info!(
            "Solving problem {} with {} berths and {} vessels",
            file,
            problem.berths().len(),
            problem.flexible_requests().len()
        );

        let mut solver = SolverEngineBuilder::<i64>::default().build();
        match solver.solve(&problem) {
            Ok(Some(solution)) => {
                tracing::info!(
                    "Solver finished on problem {}: cost={}",
                    file,
                    solution.cost()
                );
            }
            _ => {
                tracing::error!("Solver failed on problem {}", file);
            }
        }
    }
}
