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

use crate::model::index::{BerthIndex, RequestIndex};
use berth_alloc_model::{
    prelude::{BerthIdentifier, Problem, RequestIdentifier},
    problem::req::RequestView,
};
use num_traits::{CheckedAdd, CheckedSub};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SolverIndexManager {
    berth_to_index: HashMap<BerthIdentifier, BerthIndex>,
    request_to_index: HashMap<RequestIdentifier, RequestIndex>,
    index_to_berth: Vec<BerthIdentifier>,
    index_to_request: Vec<RequestIdentifier>,
}

impl SolverIndexManager {
    #[inline]
    pub fn new(
        berth_to_index: HashMap<BerthIdentifier, BerthIndex>,
        req_to_index: HashMap<RequestIdentifier, RequestIndex>,
        index_to_berth: Vec<BerthIdentifier>,
        index_to_request: Vec<RequestIdentifier>,
    ) -> Self {
        Self {
            berth_to_index,
            request_to_index: req_to_index,
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
        self.request_to_index.get(&id).copied()
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

impl<T: Copy + Ord + CheckedAdd + CheckedSub> From<&Problem<T>> for SolverIndexManager {
    fn from(problem: &Problem<T>) -> Self {
        let mut index_to_berth: Vec<BerthIdentifier> =
            problem.berths().iter().map(|b| b.id()).collect();
        index_to_berth.sort_unstable();

        let mut index_to_request: Vec<RequestIdentifier> =
            problem.iter_flexible_requests().map(|r| r.id()).collect();
        index_to_request.sort_unstable();

        let berth_to_index: HashMap<_, _> = index_to_berth
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, BerthIndex(i)))
            .collect();
        let request_to_index: HashMap<_, _> = index_to_request
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, RequestIndex(i)))
            .collect();

        Self::new(
            berth_to_index,
            request_to_index,
            index_to_berth,
            index_to_request,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;

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

    fn make_problem_basic() -> Problem<i64> {
        // Berths in reverse order to verify sorting
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);

        // Two flexible requests in reverse order to verify sorting
        let mut pt_r10 = std::collections::BTreeMap::new();
        pt_r10.insert(bid(2), td(4));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 3, pt_r10).unwrap();

        let mut pt_r20 = std::collections::BTreeMap::new();
        pt_r20.insert(bid(1), td(5));
        pt_r20.insert(bid(2), td(9));
        let r20 = Request::<FlexibleKind, i64>::new(rid(20), iv(0, 100), 7, pt_r20).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b2);
        builder.add_berth(b1);
        builder.add_flexible(r20);
        builder.add_flexible(r10);
        builder.build().unwrap()
    }

    #[test]
    fn test_roundtrip_indices_and_ids() {
        // Previous manual construction still works
        let berths = vec![bid(10), bid(20), bid(30)];
        let reqs = vec![rid(101), rid(202)];

        let mut berth_to_index = HashMap::new();
        for (i, id) in berths.iter().copied().enumerate() {
            berth_to_index.insert(id, bi(i));
        }

        let mut req_to_index = HashMap::new();
        for (i, id) in reqs.iter().copied().enumerate() {
            req_to_index.insert(id, ri(i));
        }

        let m = SolverIndexManager::new(berth_to_index, req_to_index, berths.clone(), reqs.clone());

        assert_eq!(m.berths_len(), 3);
        assert_eq!(m.requests_len(), 2);
        assert_eq!(m.berth_id(bi(1)), Some(bid(20)));
        assert_eq!(m.request_id(ri(1)), Some(rid(202)));
        assert_eq!(m.berth_index(bid(30)), Some(bi(2)));
        assert_eq!(m.request_index(rid(101)), Some(ri(0)));
    }

    #[test]
    fn test_from_problem_sorts_and_builds_indices() {
        let p = make_problem_basic();
        let m: SolverIndexManager = SolverIndexManager::from(&p);

        // Sorted berths: [1,2]
        assert_eq!(m.berths_len(), 2);
        assert_eq!(m.berth_id(bi(0)), Some(bid(1)));
        assert_eq!(m.berth_id(bi(1)), Some(bid(2)));

        // Sorted flex requests: [10,20]
        assert_eq!(m.requests_len(), 2);
        assert_eq!(m.request_id(ri(0)), Some(rid(10)));
        assert_eq!(m.request_id(ri(1)), Some(rid(20)));

        // Reverse lookups
        assert_eq!(m.berth_index(bid(2)), Some(bi(1)));
        assert_eq!(m.request_index(rid(10)), Some(ri(0)));
    }

    #[test]
    fn test_unknown_ids_and_out_of_bounds_indices_return_none() {
        let p = make_problem_basic();
        let m = SolverIndexManager::from(&p);

        assert_eq!(m.berth_index(bid(999)), None);
        assert_eq!(m.request_index(rid(999)), None);
        assert_eq!(m.berth_id(bi(99)), None);
        assert_eq!(m.request_id(ri(99)), None);
    }

    #[test]
    fn test_empty_manager_behaves_safely() {
        let m = SolverIndexManager::new(HashMap::new(), HashMap::new(), Vec::new(), Vec::new());
        assert_eq!(m.berths_len(), 0);
        assert_eq!(m.requests_len(), 0);
        assert_eq!(m.berth_index(bid(1)), None);
        assert_eq!(m.request_index(rid(1)), None);
        assert_eq!(m.berth_id(bi(0)), None);
        assert_eq!(m.request_id(ri(0)), None);
    }
}
