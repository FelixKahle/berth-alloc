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
    chain::delta::ChainDelta,
    filter::filter_trait::Filter,
    index::{BerthIndex, RequestIndex},
    solver::solver_state::SolverState,
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone, Default)]
pub struct FeasibleBerthFilter;

impl<T: Copy + Ord + CheckedAdd + CheckedSub> Filter<T> for FeasibleBerthFilter {
    #[inline]
    fn check(&self, delta: &ChainDelta, state: &SolverState<T>) -> bool {
        let chain = state.chain();
        let model = state.model();

        let next = chain.next_slice();
        let starts = chain.start_slice();
        let ends = chain.end_slice();

        for b in 0..model.berths_len() {
            let s = starts[b];
            let e = ends[b];

            let mut cur = delta.next_after(next, s);

            while cur != e {
                let ri = RequestIndex(cur);
                let bi = BerthIndex(b);

                match model.processing_time(ri, bi) {
                    Some(Some(_)) => {}
                    _ => return false,
                }

                cur = delta.next_after(next, cur);
            }
        }

        true
    }

    #[inline]
    fn complexity(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        chain::delta::ChainDelta, chain::double_chain::DoubleChain, index::RequestIndex,
        solver::solver_state::SolverState,
    };
    use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use berth_alloc_model::problem::{
        asg::AssignmentContainer, berth::BerthContainer, req::RequestContainer,
    };

    type T = i64;

    fn tp(v: T) -> TimePoint<T> {
        TimePoint::new(v)
    }
    fn iv(a: T, b: T) -> TimeInterval<T> {
        TimeInterval::new(tp(a), tp(b))
    }
    fn td(v: T) -> TimeDelta<T> {
        TimeDelta::new(v)
    }
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn mk_problem_allowed() -> Problem<T> {
        // 1 berth (id 1), 1 flexible request (id 10) allowed on berth 1
        let mut berths = BerthContainer::new();
        berths.insert(Berth::from_windows(bid(1), [iv(0, 100)]));

        let fixed = AssignmentContainer::new();

        let mut flex = RequestContainer::new();
        let mut pt = std::collections::BTreeMap::new();
        pt.insert(bid(1), td(5));
        let r = Request::<berth_alloc_model::common::FlexibleKind, T>::new_flexible(
            rid(10),
            iv(0, 50),
            1 as Cost,
            pt,
        )
        .unwrap();
        flex.insert(r);

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn mk_problem_forbidden() -> Problem<T> {
        // 2 berths: {1, 2}. Request allowed only on berth 1.
        let mut berths = BerthContainer::new();
        berths.insert(Berth::from_windows(bid(1), [iv(0, 100)]));
        berths.insert(Berth::from_windows(bid(2), [iv(0, 100)]));

        let fixed = AssignmentContainer::new();

        let mut flex = RequestContainer::new();
        let mut pt = std::collections::BTreeMap::new();
        pt.insert(bid(1), td(5)); // NOT allowed on berth 2
        let r = Request::<berth_alloc_model::common::FlexibleKind, T>::new_flexible(
            rid(10),
            iv(0, 50),
            1 as Cost,
            pt,
        )
        .unwrap();
        flex.insert(r);

        Problem::new(berths, fixed, flex).unwrap()
    }

    // Build a delta that inserts node n right after berth b's start, and connects it to end.
    fn delta_insert_front(chain: &DoubleChain, node: usize, berth: usize) -> ChainDelta {
        let s = chain.start_of(berth);
        let e = chain.end_of(berth);

        let mut d = ChainDelta::new();
        // s: expected_head currently e (empty), new_head -> node
        d.push(s, chain.succ(s), node);
        // node: expected_head currently node (skipped), new_head -> e
        d.push(node, node, e);
        d
    }

    #[test]
    fn feasible_berth_accepts_allowed_insertion() {
        let p = mk_problem_allowed();
        let st = SolverState::new(&p).unwrap();

        // chain: 1 request, 1 berth
        let chain = st.chain().clone();
        let node = RequestIndex(0).0;
        let b = 0usize;

        let d = delta_insert_front(&chain, node, b);
        let f = FeasibleBerthFilter::default();
        assert!(f.check(&d, &st));
    }

    #[test]
    fn feasible_berth_rejects_forbidden_insertion() {
        let p = mk_problem_forbidden();
        let st = SolverState::new(&p).unwrap();

        // model sorted berths => berth id 1 -> index 0, id 2 -> index 1
        let chain = st.chain().clone();
        let node = RequestIndex(0).0;

        // Try to insert into berth index 1 (id 2), which is forbidden
        let d = delta_insert_front(&chain, node, 1);
        let f = FeasibleBerthFilter::default();
        assert!(!f.check(&d, &st));
    }
}
