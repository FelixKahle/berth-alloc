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
    model::{
        err::{MissingRequestError, SolverModelBuildError},
        index::{BerthIndex, RequestIndex},
        index_manager::SolverIndexManager,
        neighborhood::{ProximityMap, ProximityMapConfig},
    },
    state::berth::berthocc::{BerthOccupancy, BerthWrite},
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use berth_alloc_model::{
    prelude::{Berth, Problem},
    problem::{asg::AssignmentView, req::RequestView},
};
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverModel<'problem, T: Copy + Ord> {
    index_manager: SolverIndexManager,
    weights: Vec<Cost>,                          // len = R
    feasible_intervals: Vec<TimeInterval<T>>,    // len = R
    processing_times: Vec<Option<TimeDelta<T>>>, // len = R * B
    berths_len: usize,                           // B
    requests_len: usize,                         // R
    berths: Vec<Berth<T>>,                       // len = B
    allowed_berth_indices: Vec<Vec<BerthIndex>>, // len = R
    proximity_map: ProximityMap,                 // Neighborhood info
    problem: &'problem Problem<T>,               // reference to original problem
}

impl<'problem, T: Copy + Ord + CheckedAdd + CheckedSub> SolverModel<'problem, T> {
    #[inline]
    pub fn weights(&self) -> &[Cost] {
        &self.weights
    }

    #[inline]
    pub fn weight(&self, req: RequestIndex) -> Cost {
        let index = req.get();
        debug_assert!(index < self.weights.len());

        self.weights[index]
    }

    #[inline]
    pub fn feasible_intervals(&self) -> &[TimeInterval<T>] {
        &self.feasible_intervals
    }

    #[inline]
    pub fn feasible_interval(&self, req: RequestIndex) -> TimeInterval<T> {
        let index = req.get();
        debug_assert!(index < self.feasible_intervals.len());

        self.feasible_intervals[index]
    }

    #[inline]
    pub fn arrival_time(&self, req: RequestIndex) -> TimePoint<T> {
        self.feasible_interval(req).start()
    }

    #[inline]
    pub fn processing_times(&self) -> &[Option<TimeDelta<T>>] {
        &self.processing_times
    }

    #[inline]
    pub fn processing_time(&self, req: RequestIndex, berth: BerthIndex) -> Option<TimeDelta<T>> {
        let index = self.flat_index(req, berth);
        debug_assert!(index < self.processing_times.len());

        self.processing_times[index]
    }

    #[inline(always)]
    fn flat_index(&self, req: RequestIndex, berth: BerthIndex) -> usize {
        debug_assert!(req.0 < self.requests_len);
        debug_assert!(berth.0 < self.berths_len);

        req.0 * self.berths_len + berth.0
    }

    #[inline]
    pub fn berths_len(&self) -> usize {
        self.berths_len
    }

    #[inline]
    pub fn flexible_requests_len(&self) -> usize {
        self.requests_len
    }

    #[inline]
    pub fn index_manager(&self) -> &SolverIndexManager {
        &self.index_manager
    }

    #[inline]
    pub fn berths(&self) -> &[Berth<T>] {
        &self.berths
    }

    #[inline]
    pub fn berth(&self, berth: BerthIndex) -> &Berth<T> {
        let index = berth.get();
        debug_assert!(index < self.berths_len());

        &self.berths[index]
    }

    #[inline]
    pub fn interval(
        &self,
        req: RequestIndex,
        berth: BerthIndex,
        start_time: TimePoint<T>,
    ) -> Option<TimeInterval<T>> {
        let processing_time = self.processing_time(req, berth)?;
        Some(TimeInterval::new(start_time, start_time + processing_time))
    }

    #[inline]
    pub fn allowed_berth_indices(&self, req: RequestIndex) -> &[BerthIndex] {
        let index = req.get();
        debug_assert!(index < self.allowed_berth_indices.len());

        self.allowed_berth_indices[index].as_slice()
    }

    #[inline]
    pub fn waiting_time(
        &self,
        req: RequestIndex,
        start_time: TimePoint<T>,
    ) -> Option<TimeDelta<T>> {
        let arrival_time = self.arrival_time(req);

        if start_time < arrival_time {
            return None;
        }

        Some(start_time - arrival_time)
    }

    pub fn turnaround_time(
        &self,
        req: RequestIndex,
        berth: BerthIndex,
        start_time: TimePoint<T>,
    ) -> Option<TimeDelta<T>> {
        let processing_time = self.processing_time(req, berth)?;
        Some(processing_time + self.waiting_time(req, start_time)?)
    }

    #[inline]
    pub fn cost_of_assignment(
        &self,
        req: RequestIndex,
        berth: BerthIndex,
        start_time: TimePoint<T>,
    ) -> Option<Cost>
    where
        T: Into<Cost>,
    {
        let weight = self.weight(req);
        let turnaround_time = self.turnaround_time(req, berth, start_time)?;
        Some(weight.saturating_mul(turnaround_time.value().into()))
    }

    #[inline]
    pub fn problem(&self) -> &'problem Problem<T> {
        self.problem
    }

    #[inline]
    pub fn proximity_map(&self) -> &ProximityMap {
        &self.proximity_map
    }

    pub fn from_problem_with_proximity_map_config(
        p: &'problem Problem<T>,
        proximity_map_config: ProximityMapConfig,
    ) -> Result<Self, SolverModelBuildError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug,
    {
        let index_manager = SolverIndexManager::from(p);
        let berths_len = index_manager.berths_len();
        let requests_len = index_manager.requests_len();

        let mut weights = Vec::with_capacity(requests_len);
        let mut feasible_intervals = Vec::with_capacity(requests_len);
        let mut processing_times = vec![None; requests_len * berths_len];
        let mut allowed_berth_indices: Vec<Vec<BerthIndex>> = Vec::with_capacity(requests_len);
        let mut berths = Vec::with_capacity(berths_len);

        for ri_u in 0..requests_len {
            let ri = RequestIndex(ri_u);
            let rid = index_manager
                .request_id(ri)
                .expect("request_id must exist for 0..requests_len");

            let rq = p.flexible_requests().get(rid).ok_or_else(|| {
                SolverModelBuildError::MissingRequest(MissingRequestError::new(rid))
            })?;

            weights.push(rq.weight());
            feasible_intervals.push(rq.feasible_window());

            // collect allowed berth indices for this request
            let mut allowed_for_req: Vec<BerthIndex> = Vec::new();

            for (&bid, &dt) in rq.processing_times().iter() {
                let bi = index_manager.berth_index(bid).ok_or_else(|| {
                    SolverModelBuildError::BerthNotFound(
                        berth_alloc_model::problem::err::BerthNotFoundError::new(rq.id(), bid),
                    )
                })?;
                let flat = ri_u * berths_len + bi.0;
                processing_times[flat] = Some(dt);
                allowed_for_req.push(bi);
            }

            // keep indices in ascending order for determinism
            allowed_for_req.sort_by_key(|b| b.0);
            allowed_berth_indices.push(allowed_for_req);
        }

        let mut baseline_occupancies = Vec::with_capacity(berths_len);
        for bi_u in 0..berths_len {
            let bid = index_manager
                .berth_id(BerthIndex(bi_u))
                .expect("berth_id must exist for 0..berths_len");
            let berth_ref = p
                .berths()
                .get(bid)
                .expect("berth id from manager must exist in Problem");
            baseline_occupancies.push(BerthOccupancy::new(berth_ref));
            berths.push(berth_ref.clone());
        }

        for a in p.iter_fixed_assignments() {
            let bid = a.berth_id();
            let bi = index_manager
                .berth_index(bid)
                .expect("every fixed assignment berth must be indexable");
            let iv = TimeInterval::new(a.start_time(), a.end_time());
            baseline_occupancies[bi.0]
                .occupy(iv)
                .expect("seeding fixed assignment should not fail");
        }

        let proximity_map = ProximityMap::from_lists(
            &feasible_intervals,
            &allowed_berth_indices,
            berths_len,
            proximity_map_config,
        );

        Ok(SolverModel {
            index_manager,
            allowed_berth_indices,
            weights,
            feasible_intervals,
            processing_times,
            berths_len,
            requests_len,
            berths,
            proximity_map,
            problem: p,
        })
    }

    pub fn from_problem(p: &'problem Problem<T>) -> Result<Self, SolverModelBuildError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug,
    {
        Self::from_problem_with_proximity_map_config(p, ProximityMapConfig::default())
    }
}

impl<'problem, T> TryFrom<&'problem Problem<T>> for SolverModel<'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug,
{
    type Error = SolverModelBuildError;

    fn try_from(p: &'problem Problem<T>) -> Result<Self, Self::Error> {
        Self::from_problem(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::loader::ProblemLoader;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }

    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }

    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }

    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }

    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex(n)
    }

    #[inline]
    fn ri(n: usize) -> RequestIndex {
        RequestIndex(n)
    }

    fn make_problem_basic() -> Problem<i64> {
        // Two berths with full windows, intentionally added in reverse id order
        // to verify that SolverModel sorting puts them in ascending id order.
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);

        // Two flexible requests, intentionally added in reverse id order.
        // rid 10: only berth 2 is feasible
        let mut pt_r10 = BTreeMap::new();
        pt_r10.insert(bid(2), td(4));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 3, pt_r10)
            .expect("r10 should be feasible");

        // rid 20: both berth 1 and 2 are feasible
        let mut pt_r20 = BTreeMap::new();
        pt_r20.insert(bid(1), td(5));
        pt_r20.insert(bid(2), td(9));
        let r20 = Request::<FlexibleKind, i64>::new(rid(20), iv(0, 100), 7, pt_r20)
            .expect("r20 should be feasible");

        let mut builder = ProblemBuilder::new();
        // Add in descending id order on purpose
        builder.add_berth(b2.clone());
        builder.add_berth(b1.clone());
        // Add requests in descending id order on purpose
        builder.add_flexible(r20.clone());
        builder.add_flexible(r10.clone());
        builder
            .build()
            .expect("builder should produce a valid problem")
    }

    #[test]
    fn test_try_from_problem_builds_consistent_indices_and_arrays() {
        let p = make_problem_basic();
        let m = SolverModel::try_from(&p).expect("conversion should succeed");

        // Dimensions
        assert_eq!(m.berths_len(), 2);
        assert_eq!(m.flexible_requests_len(), 2);

        // Index manager orders by ascending ids
        let im = m.index_manager();
        assert_eq!(im.berth_id(bi(0)), Some(bid(1)));
        assert_eq!(im.berth_id(bi(1)), Some(bid(2)));
        assert_eq!(im.berth_index(bid(1)), Some(bi(0)));
        assert_eq!(im.berth_index(bid(2)), Some(bi(1)));

        assert_eq!(im.request_id(ri(0)), Some(rid(10)));
        assert_eq!(im.request_id(ri(1)), Some(rid(20)));
        assert_eq!(im.request_index(rid(10)), Some(ri(0)));
        assert_eq!(im.request_index(rid(20)), Some(ri(1)));

        // Weights follow sorted request-id order: rid 10 -> 3, rid 20 -> 7
        assert_eq!(m.weights(), &[3, 7]);

        // Feasible windows follow the same order
        assert_eq!(m.feasible_intervals(), &[iv(0, 100), iv(0, 100)]);

        // Processing times matrix (R x B) via accessor:
        // rid 10: [None, Some(4)] because only berth 2 is present
        assert_eq!(m.processing_time(ri(0), bi(0)), None);
        assert_eq!(m.processing_time(ri(0), bi(1)), Some(td(4)));

        // rid 20: [Some(5), Some(9)]
        assert_eq!(m.processing_time(ri(1), bi(0)), Some(td(5)));
        assert_eq!(m.processing_time(ri(1), bi(1)), Some(td(9)));
    }

    #[test]
    fn test_try_from_problem_errors_when_request_refers_to_unknown_berth() {
        // One berth in the problem
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        // Request references an unknown berth id=99
        let mut pt = BTreeMap::new();
        pt.insert(bid(99), td(3));
        let r = Request::<FlexibleKind, i64>::new(rid(7), iv(0, 50), 1, pt)
            .expect("request remains syntactically valid");

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r);

        let err = builder
            .build()
            .expect_err("builder should reject unknown berths");

        match err {
            berth_alloc_model::problem::err::ProblemError::FixedAssignmentOverlap(
                assignment_overlap_error,
            ) => {
                panic!(
                    "expected BerthNotFound, got overlap: {}",
                    assignment_overlap_error
                );
            }
            berth_alloc_model::problem::err::ProblemError::BerthNotFound(berth_not_found_error) => {
                assert_eq!(berth_not_found_error.request(), rid(7));
                assert_eq!(berth_not_found_error.requested_berth(), bid(99));
            }
        }
    }

    #[test]
    fn test_load_all_instances_from_workspace_root_instances_folder_create_model() {
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

            let model = SolverModel::try_from(&problem);
            assert!(
                model.is_ok(),
                "SolverModel creation failed for {}",
                path.display()
            );
        }
    }
}
