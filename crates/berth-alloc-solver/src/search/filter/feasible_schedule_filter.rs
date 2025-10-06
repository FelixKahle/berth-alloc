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
    search::{filter::traits::FeasibilityFilter, scheduling::scheduler::Scheduler},
    state::{
        chain_set::{delta::ChainSetDelta, overlay::ChainSetOverlay},
        cost_policy::CostPolicy,
        search_state::SolverSearchState,
    },
};
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeasibleScheduleFilter<T: Copy + Ord + CheckedAdd + CheckedSub + Zero, S: Scheduler<T>> {
    scheduler: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<'model, 'problem, T, P, S> FeasibilityFilter<'model, 'problem, T, P>
    for FeasibleScheduleFilter<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero,
    P: CostPolicy<T>,
    S: Scheduler<T>,
{
    fn is_feasible(
        &self,
        delta: &ChainSetDelta,
        search_state: &SolverSearchState<'model, 'problem, T, P>,
    ) -> bool {
        let overlay = ChainSetOverlay::new(search_state.chain_set(), delta);
        self.scheduler
            .check_schedule(search_state, &overlay)
            .is_ok()
    }

    #[inline]
    fn complexity(&self) -> usize {
        10 // larger than berth filter, as this does full scheduling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        search::scheduling::greedy::GreedyEarliest,
        state::{
            chain_set::{
                base::ChainSet,
                delta_builder::ChainSetDeltaBuilder,
                index::{ChainIndex, NodeIndex},
                view::ChainSetView,
            },
            cost_policy::WeightedFlowTime,
            model::SolverModel,
            search_state::SolverSearchState,
        },
    };
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

    fn req_flex(
        id: usize,
        window: (i64, i64),
        pts: &[(usize, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn req_fixed(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn asg_fixed(
        req: &Request<FixedKind, i64>,
        berth: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new(req.clone(), berth.clone(), tp(start)).unwrap()
    }

    #[inline]
    fn make_search_state<'p>(
        model: &'p SolverModel<'p, i64>,
    ) -> SolverSearchState<'p, 'p, i64, WeightedFlowTime<'p, 'p, i64>> {
        let policy = WeightedFlowTime::new(model);
        SolverSearchState::new(model, policy)
    }

    // Build a ChainSetDelta that links node indices in order on the given chain.
    fn build_delta_linking(
        base: &ChainSet,
        chain: ChainIndex,
        nodes: &[usize],
    ) -> crate::state::chain_set::delta::ChainSetDelta {
        let mut builder = ChainSetDeltaBuilder::new(base);
        let mut prev = base.start_of_chain(chain);
        for &node in nodes {
            let n = NodeIndex(node);
            builder.insert_after(prev, n);
            prev = n;
        }
        builder.build()
    }

    #[test]
    fn test_empty_delta_is_feasible() {
        // No requests scheduled. Should be feasible.
        let b = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r = req_flex(10, (0, 100), &[(1, 5)], 1);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b);
        pb.add_flexible(r);
        let p: Problem<i64> = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let search_state = make_search_state(&model);

        let delta = ChainSetDelta::new(); // empty
        let filter = FeasibleScheduleFilter::<i64, _> {
            scheduler: GreedyEarliest::default(),
            _phantom: std::marker::PhantomData,
        };

        assert!(filter.is_feasible(&delta, &search_state));
    }

    #[test]
    fn test_feasible_simple_two_on_one_chain() {
        // One berth; two feasible requests on that berth.
        let b = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r10 = req_flex(10, (0, 100), &[(1, 5)], 1);
        let r20 = req_flex(20, (0, 100), &[(1, 7)], 1);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b);
        pb.add_flexible(r10);
        pb.add_flexible(r20);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let search_state = make_search_state(&model);

        // indices: rid(10)->0, rid(20)->1
        let delta = build_delta_linking(search_state.chain_set(), ChainIndex(0), &[0, 1]);

        let filter = FeasibleScheduleFilter::<i64, _> {
            scheduler: GreedyEarliest::default(),
            _phantom: std::marker::PhantomData,
        };
        assert!(filter.is_feasible(&delta, &search_state));
    }

    #[test]
    fn test_infeasible_not_allowed_on_berth() {
        // Two berths; request feasible only on berth 2 but placed on chain for berth 1.
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        let r = req_flex(10, (0, 100), &[(2, 5)], 1); // only on berth 2

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let search_state = make_search_state(&model);

        // Put that single request (index 0) on chain 0 (berth 1) -> infeasible for GreedyEarliest
        let delta = build_delta_linking(search_state.chain_set(), ChainIndex(0), &[0]);

        let filter = FeasibleScheduleFilter::<i64, _> {
            scheduler: GreedyEarliest::default(),
            _phantom: std::marker::PhantomData,
        };
        assert!(!filter.is_feasible(&delta, &search_state));
    }

    #[test]
    fn test_infeasible_window_violation() {
        // Two berths (to keep berths_len == requests_len for the internal ChainSet).
        // Both requests are feasible only on berth 1.
        // rA length 8, rB length 5 with window [0,10]; order [A,B] forces rB end=13>10 -> infeasible.
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        let r_a = req_flex(10, (0, 100), &[(1, 8)], 1); // only berth 1
        let r_b = req_flex(20, (0, 10), &[(1, 5)], 1); // only berth 1

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r_a);
        pb.add_flexible(r_b);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let search_state = make_search_state(&model);

        // Link both requests onto chain 0 (berth 1)
        let delta = build_delta_linking(search_state.chain_set(), ChainIndex(0), &[0, 1]);

        let filter = FeasibleScheduleFilter::<i64, _> {
            scheduler: GreedyEarliest::default(),
            _phantom: std::marker::PhantomData,
        };
        assert!(
            !filter.is_feasible(&delta, &search_state),
            "Expected infeasible due to rB window violation after scheduling rA first"
        );
    }

    #[test]
    fn test_respects_fixed_and_finds_earliest_fit() {
        // b1 availability [0,100), fixed [10,20).
        // Two flex jobs of 7 each on b1, both [0,100).
        // Should be feasible: place [0,7) then [20,27).
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let rf = req_fixed(900, (0, 100), &[(1, 10)]);
        let fixed = asg_fixed(&rf, &b1, 10);

        let r_a = req_flex(10, (0, 100), &[(1, 7)], 1);
        let r_b = req_flex(20, (0, 100), &[(1, 7)], 1);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_fixed(fixed);
        pb.add_flexible(r_a);
        pb.add_flexible(r_b);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let search_state = make_search_state(&model);

        let delta = build_delta_linking(search_state.chain_set(), ChainIndex(0), &[0, 1]);

        let filter = FeasibleScheduleFilter::<i64, _> {
            scheduler: GreedyEarliest::default(),
            _phantom: std::marker::PhantomData,
        };
        assert!(filter.is_feasible(&delta, &search_state));
    }
}
