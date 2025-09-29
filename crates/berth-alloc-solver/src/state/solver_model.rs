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

use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval};
use berth_alloc_model::{
    prelude::{BerthIdentifier, Problem, RequestIdentifier},
    problem::{err::BerthNotFoundError, req::RequestView},
};
use num_traits::{CheckedAdd, CheckedSub};
use std::collections::BTreeMap;

use crate::state::err::{MissingRequestError, SolverModelBuildError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BerthIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestIndex(pub usize);

#[derive(Debug, Clone)]
pub struct IdMapper {
    berth_to_index: BTreeMap<BerthIdentifier, BerthIndex>,
    req_to_index: BTreeMap<RequestIdentifier, RequestIndex>,
    index_to_berth: Vec<BerthIdentifier>,
    index_to_request: Vec<RequestIdentifier>,
}

impl IdMapper {
    #[inline]
    fn new(
        berth_to_index: BTreeMap<BerthIdentifier, BerthIndex>,
        req_to_index: BTreeMap<RequestIdentifier, RequestIndex>,
        index_to_berth: Vec<BerthIdentifier>,
        index_to_request: Vec<RequestIdentifier>,
    ) -> Self {
        Self {
            berth_to_index,
            req_to_index,
            index_to_berth,
            index_to_request,
        }
    }

    #[inline]
    pub fn berth_index(&self, id: BerthIdentifier) -> Option<BerthIndex> {
        self.berth_to_index.get(&id).copied()
    }

    #[inline]
    pub fn request_index(&self, id: RequestIdentifier) -> Option<RequestIndex> {
        self.req_to_index.get(&id).copied()
    }

    #[inline]
    pub fn berth_id(&self, i: BerthIndex) -> Option<BerthIdentifier> {
        self.index_to_berth.get(i.0).copied()
    }

    #[inline]
    pub fn request_id(&self, i: RequestIndex) -> Option<RequestIdentifier> {
        self.index_to_request.get(i.0).copied()
    }

    #[inline]
    pub fn berths_len(&self) -> usize {
        self.index_to_berth.len()
    }

    #[inline]
    pub fn requests_len(&self) -> usize {
        self.index_to_request.len()
    }
}

#[derive(Debug, Clone)]
pub struct SolverModel<T: Copy + Ord> {
    berths: Vec<usize>,                          // len = B
    requests: Vec<usize>,                        // len = R
    weights: Vec<Cost>,                          // len = R
    feasible_intervals: Vec<TimeInterval<T>>,    // len = R
    processing_times: Vec<Option<TimeDelta<T>>>, // len = R * B
    mappings: IdMapper,
    berths_len: usize,
    requests_len: usize,
}

impl<T: Copy + Ord> SolverModel<T> {
    #[inline]
    pub fn berths(&self) -> &[usize] {
        &self.berths
    }

    #[inline]
    pub fn requests(&self) -> &[usize] {
        &self.requests
    }

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
    pub fn requests_len(&self) -> usize {
        self.requests_len
    }

    #[inline]
    pub fn mappings(&self) -> &IdMapper {
        &self.mappings
    }
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub> TryFrom<&Problem<T>> for SolverModel<T> {
    type Error = SolverModelBuildError;

    fn try_from(p: &Problem<T>) -> Result<Self, Self::Error> {
        let mut index_to_berth: Vec<BerthIdentifier> = p.berths().iter().map(|b| b.id()).collect();
        index_to_berth.sort_unstable();
        let berths_len = index_to_berth.len();

        let berth_to_index: BTreeMap<_, _> = index_to_berth
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, BerthIndex(i)))
            .collect();

        let mut index_to_request: Vec<RequestIdentifier> =
            p.iter_flexible_requests().map(|r| r.id()).collect();
        index_to_request.sort_unstable();
        let requests_len = index_to_request.len();

        let req_to_index: BTreeMap<_, _> = index_to_request
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, RequestIndex(i)))
            .collect();

        let mut weights = Vec::with_capacity(requests_len);
        let mut feasible_intervals = Vec::with_capacity(requests_len);
        let mut processing_times = vec![None; requests_len * berths_len];

        for (ri, rid) in index_to_request.iter().enumerate() {
            let rq = p.flexible_requests().get(*rid).ok_or_else(|| {
                SolverModelBuildError::MissingRequest(MissingRequestError::new(*rid))
            })?;

            weights.push(rq.weight());
            feasible_intervals.push(rq.feasible_window());

            for (&bid, &dt) in rq.processing_times().iter() {
                let bi = berth_to_index
                    .get(&bid)
                    .ok_or_else(|| {
                        SolverModelBuildError::BerthNotFound(BerthNotFoundError::new(rq.id(), bid))
                    })?
                    .0;
                let flat = ri * berths_len + bi;
                processing_times[flat] = Some(dt);
            }
        }

        let berths: Vec<usize> = (0..berths_len).collect();
        let requests: Vec<usize> = (0..requests_len).collect();

        let mappings = IdMapper::new(
            berth_to_index,
            req_to_index,
            index_to_berth,
            index_to_request,
        );

        Ok(SolverModel {
            berths,
            requests,
            weights,
            feasible_intervals,
            processing_times,
            mappings,
            berths_len,
            requests_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::{FixedKind, FlexibleKind};
    use berth_alloc_model::prelude::{BerthIdentifier, RequestIdentifier};
    use berth_alloc_model::problem::asg::{Assignment, AssignmentContainer};
    use berth_alloc_model::problem::berth::{Berth, BerthContainer};
    use berth_alloc_model::problem::req::{Request, RequestContainer};

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
    fn berth(id: usize, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    #[inline]
    fn req_flex(
        id: usize,
        window: (i64, i64),
        weight: Cost,
        pts: &[(usize, i64)],
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new_flexible(rid(id), iv(window.0, window.1), weight, m)
            .expect("flex request must be valid")
    }

    #[inline]
    fn req_fixed(
        id: usize,
        window: (i64, i64),
        weight: Cost,
        pts: &[(usize, i64)],
    ) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new_fixed(rid(id), iv(window.0, window.1), weight, m)
            .expect("fixed request must be valid")
    }

    #[inline]
    fn asg_fixed(
        req: &Request<FixedKind, i64>,
        berth: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new(req.clone(), berth.clone(), tp(start))
            .expect("fixed assignment must be valid")
    }

    #[test]
    fn from_empty_problem_produces_empty_model() {
        let berths = BerthContainer::<i64>::new();
        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let p = Problem::new(berths, fixed, flex).expect("empty problem is valid");
        let m: SolverModel<i64> = (&p).try_into().expect("empty model is valid");

        assert_eq!(m.berths_len(), 0);
        assert_eq!(m.requests_len(), 0);
        assert!(m.berths().is_empty());
        assert!(m.requests().is_empty());
        assert!(m.weights().is_empty());
        assert!(m.feasible_intervals().is_empty());
        assert!(m.processing_times().is_empty());
    }

    #[test]
    fn single_berth_single_request_populates_everything() {
        // One berth, one flex request allowed on it.
        let b = berth(1, 0, 100);
        let r = req_flex(10, (0, 50), 7, &[(1, 5)]);

        let mut berths = BerthContainer::new();
        berths.insert(b.clone());

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(r.clone());

        let p = Problem::new(berths, fixed, flex).unwrap();
        let m: SolverModel<i64> = (&p).try_into().unwrap();

        // sizes
        assert_eq!(m.berths_len(), 1);
        assert_eq!(m.requests_len(), 1);
        assert_eq!(m.berths(), &[0]);
        assert_eq!(m.requests(), &[0]);

        // mapper round-trip
        let bi = m.mappings().berth_index(b.id()).unwrap();
        let ri = m.mappings().request_index(r.id()).unwrap();
        assert_eq!(bi.0, 0);
        assert_eq!(ri.0, 0);
        assert_eq!(m.mappings().berth_id(bi), Some(b.id()));
        assert_eq!(m.mappings().request_id(ri), Some(r.id()));

        // attributes
        assert_eq!(m.weights()[0], 7);
        assert_eq!(m.feasible_intervals()[0], iv(0, 50));

        // processing time: row-major [R x B], here only one slot
        assert_eq!(m.processing_times().len(), 1);
        assert_eq!(m.processing_time(ri, bi), Some(Some(td(5))));
    }

    #[test]
    fn two_berths_one_request_only_one_allowed_and_sorted_indices() {
        // Insert berths out-of-order to verify stable sort by ID.
        let b_big = berth(5, 0, 100);
        let b_small = berth(2, 0, 100);

        let req = req_flex(20, (0, 50), 3, &[(5, 11)]); // allowed only on berth 5

        let mut berths = BerthContainer::new();
        berths.insert(b_big.clone());
        berths.insert(b_small.clone());

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(req.clone());

        let p = Problem::new(berths, fixed, flex).unwrap();
        let m: SolverModel<i64> = (&p).try_into().unwrap();

        // Berths sorted: [2, 5] -> indices 0 for 2, 1 for 5
        let bi2 = m.mappings().berth_index(b_small.id()).unwrap();
        let bi5 = m.mappings().berth_index(b_big.id()).unwrap();
        assert_eq!(bi2.0, 0);
        assert_eq!(bi5.0, 1);

        let ri = m.mappings().request_index(req.id()).unwrap();
        assert_eq!(m.processing_times().len(), 1 * 2);

        // Allowed only on 5 => (ri, bi2) = None, (ri, bi5) = Some(11)
        assert_eq!(m.processing_time(ri, bi2), Some(None));
        assert_eq!(m.processing_time(ri, bi5), Some(Some(td(11))));
    }

    #[test]
    fn multiple_requests_multiple_berths_row_major_layout() {
        // Berths [1,2,3]
        let mut berths = BerthContainer::new();
        let b1 = berth(1, 0, 1000);
        let b2 = berth(2, 0, 1000);
        let b3 = berth(3, 0, 1000);
        berths.insert(b2.clone()); // out-of-order on purpose
        berths.insert(b1.clone());
        berths.insert(b3.clone());

        // Requests (sorted by id in model)
        // rA (id 30): allowed on 1->2, 3->7
        // rB (id 31): allowed on 2->4
        let r_a = req_flex(30, (0, 100), 1, &[(1, 2), (3, 7)]);
        let r_b = req_flex(31, (10, 200), 5, &[(2, 4)]);

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(r_b.clone()); // insert out-of-order
        flex.insert(r_a.clone());

        let p = Problem::new(berths, fixed, flex).unwrap();
        let m: SolverModel<i64> = (&p).try_into().unwrap();

        assert_eq!(m.berths_len(), 3);
        assert_eq!(m.requests_len(), 2);
        assert_eq!(m.processing_times().len(), 2 * 3);

        // Mapped indices: berths sorted -> [1,2,3] => (b1=0, b2=1, b3=2)
        let b1i = m.mappings().berth_index(b1.id()).unwrap();
        let b2i = m.mappings().berth_index(b2.id()).unwrap();
        let b3i = m.mappings().berth_index(b3.id()).unwrap();
        assert_eq!((b1i.0, b2i.0, b3i.0), (0, 1, 2));

        // Requests sorted: [30, 31] => r_a=0, r_b=1
        let ra_i = m.mappings().request_index(r_a.id()).unwrap();
        let rb_i = m.mappings().request_index(r_b.id()).unwrap();
        assert_eq!((ra_i.0, rb_i.0), (0, 1));

        // Row 0 (rA): [pt(1)=2, None, pt(3)=7]
        assert_eq!(m.processing_time(ra_i, b1i), Some(Some(td(2))));
        assert_eq!(m.processing_time(ra_i, b2i), Some(None));
        assert_eq!(m.processing_time(ra_i, b3i), Some(Some(td(7))));

        // Row 1 (rB): [None, pt(2)=4, None]
        assert_eq!(m.processing_time(rb_i, b1i), Some(None));
        assert_eq!(m.processing_time(rb_i, b2i), Some(Some(td(4))));
        assert_eq!(m.processing_time(rb_i, b3i), Some(None));

        // Weights & windows aligned with request row indices
        assert_eq!(m.weights()[0], 1);
        assert_eq!(m.weights()[1], 5);
        assert_eq!(m.feasible_intervals()[0], iv(0, 100));
        assert_eq!(m.feasible_intervals()[1], iv(10, 200));
    }

    #[test]
    fn fixed_assignments_are_ignored_in_solver_model() {
        // One berth
        let b = berth(1, 0, 1000);

        // One fixed assignment on that berth
        let rf = req_fixed(100, (0, 1000), 1, &[(1, 10)]);
        let af = asg_fixed(&rf, &b, 0);

        // One flexible request too
        let rflex = req_flex(200, (0, 50), 42, &[(1, 3)]);

        let mut berths = BerthContainer::new();
        berths.insert(b);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(rflex.clone());

        let p = Problem::new(berths, fixed, flex).unwrap();
        let m: SolverModel<i64> = (&p).try_into().unwrap();

        // Only the flexible request should appear
        assert_eq!(m.requests_len(), 1);
        let r_idx = m.mappings().request_index(rflex.id()).unwrap();
        assert_eq!(r_idx.0, 0);
        assert_eq!(m.weights()[0], 42);
    }

    #[test]
    fn mapper_roundtrip_and_lengths_match() {
        // 3 berths, 4 flex requests with mixed allowances
        let b1 = berth(10, 0, 500);
        let b2 = berth(2, 0, 500);
        let b3 = berth(7, 0, 500);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());
        berths.insert(b2.clone());
        berths.insert(b3.clone());

        let r1 = req_flex(3, (0, 100), 1, &[(2, 5)]);
        let r2 = req_flex(1, (0, 100), 2, &[(10, 7), (7, 9)]);
        let r3 = req_flex(4, (0, 100), 3, &[(7, 11)]);
        let r4 = req_flex(2, (0, 100), 4, &[(10, 1), (2, 2), (7, 3)]);

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        for r in [r1.clone(), r2.clone(), r3.clone(), r4.clone()] {
            flex.insert(r);
        }

        let p = Problem::new(berths, fixed, flex).unwrap();
        let m: SolverModel<i64> = (&p).try_into().unwrap();

        // lengths
        assert_eq!(m.berths_len(), 3);
        assert_eq!(m.requests_len(), 4);
        assert_eq!(m.berths().len(), 3);
        assert_eq!(m.requests().len(), 4);
        assert_eq!(m.processing_times().len(), 12);

        // round-trip mapping
        for &b in &[b1.id(), b2.id(), b3.id()] {
            let bi = m.mappings().berth_index(b).unwrap();
            assert_eq!(m.mappings().berth_id(bi), Some(b));
        }
        for &r in &[r1.id(), r2.id(), r3.id(), r4.id()] {
            let ri = m.mappings().request_index(r).unwrap();
            assert_eq!(m.mappings().request_id(ri), Some(r));
        }
    }
}
