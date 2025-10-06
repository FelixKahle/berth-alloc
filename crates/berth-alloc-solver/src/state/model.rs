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

use crate::state::{
    berth::{berthocc::BerthOccupancy, traits::BerthWrite},
    err::{MissingRequestError, SolverModelBuildError},
    index::{BerthIndex, RequestIndex},
    index_manager::SolverIndexManager,
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval};
use berth_alloc_model::{
    prelude::Problem,
    problem::{asg::AssignmentView, req::RequestView},
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone)]
pub struct SolverModel<'problem, T: Copy + Ord> {
    index_manager: SolverIndexManager,
    weights: Vec<Cost>,                                     // len = R
    feasible_intervals: Vec<TimeInterval<T>>,               // len = R
    processing_times: Vec<Option<TimeDelta<T>>>,            // len = R * B
    berths_len: usize,                                      // B
    requests_len: usize,                                    // R
    baseline_occupancies: Vec<BerthOccupancy<'problem, T>>, // len = B
    problem: &'problem Problem<T>,                          // reference to original problem
}

impl<'problem, T: Copy + Ord + CheckedAdd + CheckedSub> SolverModel<'problem, T> {
    #[inline]
    pub fn weights(&self) -> &[Cost] {
        &self.weights
    }

    #[inline]
    pub fn feasible_intervals(&self) -> &[TimeInterval<T>] {
        &self.feasible_intervals
    }

    #[inline]
    pub fn processing_times(&self) -> &[Option<TimeDelta<T>>] {
        &self.processing_times
    }

    #[inline(always)]
    fn flat_index(&self, req: RequestIndex, berth: BerthIndex) -> usize {
        debug_assert!(req.0 < self.requests_len);
        debug_assert!(berth.0 < self.berths_len);

        req.0 * self.berths_len + berth.0
    }

    #[inline]
    pub fn processing_time(
        &self,
        req: RequestIndex,
        berth: BerthIndex,
    ) -> Option<Option<TimeDelta<T>>> {
        self.processing_times
            .get(self.flat_index(req, berth))
            .copied()
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
    pub fn baseline_occupancies(&self) -> &[BerthOccupancy<'problem, T>] {
        &self.baseline_occupancies
    }

    #[inline]
    pub fn baseline_occupancy_for_berth(
        &self,
        berth: BerthIndex,
    ) -> Option<&BerthOccupancy<'problem, T>> {
        self.baseline_occupancies.get(berth.0)
    }

    #[inline]
    pub fn problem(&self) -> &'problem Problem<T> {
        self.problem
    }

    pub fn from_problem(p: &'problem Problem<T>) -> Result<Self, SolverModelBuildError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub + std::fmt::Debug,
    {
        let index_manager = SolverIndexManager::from(p);
        let berths_len = index_manager.berths_len();
        let requests_len = index_manager.requests_len();

        let mut weights = Vec::with_capacity(requests_len);
        let mut feasible_intervals = Vec::with_capacity(requests_len);
        let mut processing_times = vec![None; requests_len * berths_len];

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

            for (&bid, &dt) in rq.processing_times().iter() {
                let bi = index_manager.berth_index(bid).ok_or_else(|| {
                    SolverModelBuildError::BerthNotFound(
                        berth_alloc_model::problem::err::BerthNotFoundError::new(rq.id(), bid),
                    )
                })?;
                let flat = ri_u * berths_len + bi.0;
                processing_times[flat] = Some(dt);
            }
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

        Ok(SolverModel {
            index_manager,
            weights,
            feasible_intervals,
            processing_times,
            berths_len,
            requests_len,
            problem: p,
            baseline_occupancies,
        })
    }
}

impl<'problem, T> TryFrom<&'problem Problem<T>> for SolverModel<'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + std::fmt::Debug,
{
    type Error = SolverModelBuildError;

    fn try_from(p: &'problem Problem<T>) -> Result<Self, Self::Error> {
        Self::from_problem(p)
    }
}

#[cfg(test)]
mod tests {
    use crate::state::berth::traits::BerthRead;

    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::{FixedKind, FlexibleKind};
    use berth_alloc_model::prelude::{
        Assignment, Berth, BerthIdentifier, Problem, RequestIdentifier,
    };
    use berth_alloc_model::problem::builder::ProblemBuilder;
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
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }

    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
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

    fn req_fixed(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn req_flex(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn berth(id: usize, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn asg_fixed(
        req: &Request<FixedKind, i64>,
        berth: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new(req.clone(), berth.clone(), tp(start)).unwrap()
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
        assert_eq!(m.processing_time(ri(0), bi(0)), Some(None));
        assert_eq!(m.processing_time(ri(0), bi(1)), Some(Some(td(4))));

        // rid 20: [Some(5), Some(9)]
        assert_eq!(m.processing_time(ri(1), bi(0)), Some(Some(td(5))));
        assert_eq!(m.processing_time(ri(1), bi(1)), Some(Some(td(9))));
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
    fn test_baseline_occupancies_seed_fixed_and_gaps_correctly() {
        // b1 availability [0,50), b2 availability [0,40)
        let b1 = berth(1, 0, 50);
        let b2 = berth(2, 0, 40);

        // Fixed on b1: rF1 [10,20), rF2 [30,35)
        let rf1 = req_fixed(900, (0, 50), &[(1, 10)]);
        let rf2 = req_fixed(901, (0, 50), &[(1, 5)]);
        let af1 = asg_fixed(&rf1, &b1, 10);
        let af2 = asg_fixed(&rf2, &b1, 30);

        // One flex just to ensure R×B arrays still get built
        let flex = req_flex(1000, (0, 50), &[(1, 7), (2, 7)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1.clone());
        builder.add_berth(b2.clone());
        builder.add_fixed(af1);
        builder.add_fixed(af2);
        builder.add_flexible(flex);

        let p = builder.build().expect("problem should be valid");

        let m = SolverModel::try_from(&p).expect("model build should succeed");

        // Ensure index order is (1)->bi(0), (2)->bi(1)
        let im = m.index_manager();
        assert_eq!(im.berth_id(bi(0)), Some(bid(1)));
        assert_eq!(im.berth_id(bi(1)), Some(bid(2)));

        // b1 baseline free should be [0,10), [20,30), [35,50)
        let occ_b1 = &m.baseline_occupancies[0];
        let free_b1: Vec<_> = occ_b1.iter_free_intervals_in(iv(0, 100)).collect();
        assert_eq!(free_b1, vec![iv(0, 10), iv(20, 30), iv(35, 50)]);

        // b2 baseline free should be [0,40) (no fixed)
        let occ_b2 = &m.baseline_occupancies[1];
        let free_b2: Vec<_> = occ_b2.iter_free_intervals_in(iv(0, 100)).collect();
        assert_eq!(free_b2, vec![iv(0, 40)]);
    }

    #[test]
    fn test_baseline_occupancies_touching_fixed_are_ok() {
        // b1: availability [0,30)
        let b1 = berth(1, 0, 30);
        // fixed: [5,10) and [10,15) — touching, not overlapping
        let rf_a = req_fixed(910, (0, 30), &[(1, 5)]);
        let rf_b = req_fixed(911, (0, 30), &[(1, 5)]);
        let af_a = asg_fixed(&rf_a, &b1, 5);
        let af_b = asg_fixed(&rf_b, &b1, 10);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1.clone());
        builder.add_fixed(af_a);
        builder.add_fixed(af_b);
        // add 1 flex so R×B exists
        builder.add_flexible(req_flex(1001, (0, 30), &[(1, 3)]));

        let p = builder.build().unwrap();
        let m = SolverModel::try_from(&p).unwrap();

        // Free should be [0,5), [15,30)
        let occ = &m.baseline_occupancies[0];
        let free: Vec<_> = occ.iter_free_intervals_in(iv(0, 30)).collect();
        assert_eq!(free, vec![iv(0, 5), iv(15, 30)]);
    }

    #[test]
    fn test_baseline_occupancies_multiple_berths_and_insertion_order() {
        // Make sure insertion order of fixed doesn't matter (domain already validates)
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);

        // Fixed on b2 at [20,40) and [60,80)
        let rf1 = req_fixed(920, (0, 100), &[(2, 20)]);
        let rf2 = req_fixed(921, (0, 100), &[(2, 20)]);
        let a1 = asg_fixed(&rf1, &b2, 20);
        let a2 = asg_fixed(&rf2, &b2, 60);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1.clone());
        builder.add_berth(b2.clone());

        // Insert second first to jumble order
        builder.add_fixed(a2);
        builder.add_fixed(a1);

        builder.add_flexible(req_flex(1002, (0, 100), &[(1, 5), (2, 5)]));

        let p = builder.build().unwrap();
        let m = SolverModel::try_from(&p).unwrap();

        // Verify berth-index mapping is 1->0, 2->1 again
        let im = m.index_manager();
        assert_eq!(im.berth_id(bi(0)), Some(bid(1)));
        assert_eq!(im.berth_id(bi(1)), Some(bid(2)));

        // b1 has no fixed
        let free_b1: Vec<_> = m.baseline_occupancies[0]
            .iter_free_intervals_in(iv(0, 100))
            .collect();
        assert_eq!(free_b1, vec![iv(0, 100)]);

        // b2 free = [0,20), [40,60), [80,100)
        let free_b2: Vec<_> = m.baseline_occupancies[1]
            .iter_free_intervals_in(iv(0, 100))
            .collect();
        assert_eq!(free_b2, vec![iv(0, 20), iv(40, 60), iv(80, 100)]);
    }

    #[test]
    fn test_processing_matrix_and_indices_unchanged_with_fixed_present() {
        // Ensure adding fixed doesn’t disturb R×B arrays or index order.
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);

        // Flexible: r10 allowed only on b2 (4), r20 allowed on b1 (5) and b2 (9)
        let mut pt_r10 = BTreeMap::new();
        pt_r10.insert(bid(2), td(4));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 3, pt_r10).unwrap();

        let mut pt_r20 = BTreeMap::new();
        pt_r20.insert(bid(1), td(5));
        pt_r20.insert(bid(2), td(9));
        let r20 = Request::<FlexibleKind, i64>::new(rid(20), iv(0, 100), 7, pt_r20).unwrap();

        // Fixed on b1: [10,15)
        let rf = req_fixed(800, (0, 100), &[(1, 5)]);
        let af = asg_fixed(&rf, &b1, 10);

        let mut builder = ProblemBuilder::new();
        // Add berths in reverse to ensure manager sorts to ascending id
        builder.add_berth(b2.clone());
        builder.add_berth(b1.clone());

        // Add fixed first; then flex in reverse id order to test sorting
        builder.add_fixed(af);
        builder.add_flexible(r20.clone());
        builder.add_flexible(r10.clone());

        let p = builder.build().unwrap();
        let m = SolverModel::try_from(&p).unwrap();

        // Dimensions
        assert_eq!(m.berths_len(), 2);
        assert_eq!(m.flexible_requests_len(), 2);

        // Index manager orders by ascending ids
        let im = m.index_manager();
        assert_eq!(im.berth_id(bi(0)), Some(bid(1)));
        assert_eq!(im.berth_id(bi(1)), Some(bid(2)));
        assert_eq!(im.request_id(ri(0)), Some(rid(10)));
        assert_eq!(im.request_id(ri(1)), Some(rid(20)));

        // Weights follow sorted request-id order: rid 10 -> 3, rid 20 -> 7
        assert_eq!(m.weights(), &[3, 7]);

        // Feasible windows follow the same order
        assert_eq!(m.feasible_intervals(), &[iv(0, 100), iv(0, 100)]);

        // Processing times matrix (R x B)
        assert_eq!(m.processing_time(ri(0), bi(0)), Some(None)); // r10 on b1
        assert_eq!(m.processing_time(ri(0), bi(1)), Some(Some(td(4)))); // r10 on b2

        assert_eq!(m.processing_time(ri(1), bi(0)), Some(Some(td(5)))); // r20 on b1
        assert_eq!(m.processing_time(ri(1), bi(1)), Some(Some(td(9)))); // r20 on b2

        // Baseline free for b1 should reflect fixed [10,15)
        let free_b1: Vec<_> = m.baseline_occupancies[0]
            .iter_free_intervals_in(iv(0, 100))
            .collect();
        assert_eq!(free_b1, vec![iv(0, 10), iv(15, 100)]);

        // Baseline free for b2 is entire [0,100)
        let free_b2: Vec<_> = m.baseline_occupancies[1]
            .iter_free_intervals_in(iv(0, 100))
            .collect();
        assert_eq!(free_b2, vec![iv(0, 100)]);
    }
}
