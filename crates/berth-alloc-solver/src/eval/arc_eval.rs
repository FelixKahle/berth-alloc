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
    chain_set::index::{ChainIndex, NodeIndex},
    cost_policy::CostPolicy,
    index::{BerthIndex, RequestIndex},
    search_state::SolverSearchState,
};
use berth_alloc_core::prelude::{Cost, TimeDelta};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait ArcCostEvaluator<T: Copy + Ord, P: CostPolicy<T>> {
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Compute a **lower bound** on the incremental objective cost of adding
    /// the arc `(from → to)` on a given `chain` (one chain -> one berth).
    ///
    /// # Intent
    /// This is a *fast*, *occupancy-agnostic* estimate used during search.
    /// It must not mutate state or perform full scheduling. The intent is to
    /// approximate the cost of placing `to` immediately after `from` on the
    /// specified berth, using only per-request feasible windows and processing
    /// times, **not** the detailed free/occupied slices.
    fn evaluate_arc_cost(
        &self,
        chain: ChainIndex,
        from: NodeIndex,
        to: NodeIndex,
        search_state: &SolverSearchState<T, P>,
    ) -> Option<Cost>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DefaultArcCostEvaluator;

impl<T, P> ArcCostEvaluator<T, P> for DefaultArcCostEvaluator
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug + Into<Cost>,
    P: CostPolicy<T>,
{
    fn evaluate_arc_cost(
        &self,
        chain: ChainIndex,
        from_node: NodeIndex,
        to_node: NodeIndex,
        search_state: &SolverSearchState<T, P>,
    ) -> Option<Cost> {
        let model = search_state.model();
        let policy = search_state.cost_policy();

        let berth_idx = BerthIndex(chain.get());
        let to_req_idx = RequestIndex(to_node.get());
        let proc_cost = policy.scheduled_cost(to_req_idx, berth_idx)?;

        let to_win = model.feasible_intervals()[to_req_idx.get()];
        let earliest_arrival = if from_node == to_node {
            to_win.start()
        } else {
            let from_idx = RequestIndex(from_node.get());
            match model.processing_time(from_idx, berth_idx) {
                Some(Some(proc_from)) => {
                    let from_win = model.feasible_intervals()[from_idx.get()];
                    from_win
                        .start()
                        .checked_add(proc_from)
                        .unwrap_or(to_win.end())
                }
                _ => return None,
            }
        };

        let zero = TimeDelta::zero();
        let wait_lb = {
            let d = earliest_arrival - to_win.start();
            if d < zero { zero } else { d }
        };
        let wait_cost = policy.wait_cost(to_req_idx, wait_lb);
        Some(proc_cost.saturating_add(wait_cost))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        cost_policy::WeightedFlowTime, model::SolverModel, search_state::SolverSearchState,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    // Short-hands
    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
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
        weight: i64,
        pts: &[(usize, i64)],
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    // Helper to get indices via the model's index manager
    fn require_indices_for(
        model: &SolverModel<i64>,
        berth_id: usize,
        req_id: usize,
    ) -> (ChainIndex, NodeIndex) {
        let im = model.index_manager();
        let bi = im
            .berth_index(bid(berth_id))
            .expect("berth id must be in model");
        let ri = im
            .request_index(rid(req_id))
            .expect("request id must be in model");
        (ChainIndex::new(bi.get()), NodeIndex::new(ri.get()))
    }

    #[test]
    fn test_self_arc_cost_basic() {
        // One berth (1), one request r20 on berth 1 with proc 5, weight 7.
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r20 = req_flex(20, (0, 100), 7, &[(1, 5)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r20);
        let p: Problem<i64> = builder.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let policy = WeightedFlowTime::new(&model);
        let ss = SolverSearchState::new(&model, policy);

        let (chain_idx, to_node) = require_indices_for(&model, 1, 20);
        let from_node = to_node; // self-arc

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, from_node, to_node, &ss)
            .expect("cost should exist");

        // Self-arc: wait = 0; cost = 7 * 5 = 35
        assert_eq!(cost, 35);
    }

    #[test]
    fn test_sequential_arc_positive_wait() {
        // Berth 1
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        // r10: proc 10 on berth 1, weight 3
        let r10 = req_flex(10, (0, 100), 3, &[(1, 10)]);
        // r20: proc 5 on berth 1, weight 7
        let r20 = req_flex(20, (0, 100), 7, &[(1, 5)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.add_flexible(r20);
        let p: Problem<i64> = builder.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let policy = WeightedFlowTime::new(&model);
        let ss = SolverSearchState::new(&model, policy);

        let (chain_idx, r10_node) = require_indices_for(&model, 1, 10);
        let (_, r20_node) = require_indices_for(&model, 1, 20);

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, r10_node, r20_node, &ss)
            .expect("cost should exist");

        // from.start=0 + 10 = 10; to.start=0 => wait=10; proc_to=5; weight=7
        // 7*(10+5) = 105
        assert_eq!(cost, 105);
    }

    #[test]
    fn test_zero_wait_case() {
        // Berth 1
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        // r1: from, proc 2 on berth 1 (w=1)
        let r1 = req_flex(1, (0, 100), 1, &[(1, 2)]);
        // r2: to, window [5,100), proc 3 on berth 1, weight 11
        let r2 = req_flex(2, (5, 100), 11, &[(1, 3)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        let p: Problem<i64> = builder.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let policy = WeightedFlowTime::new(&model);
        let ss = SolverSearchState::new(&model, policy);

        let (chain_idx, r1_node) = require_indices_for(&model, 1, 1);
        let (_, r2_node) = require_indices_for(&model, 1, 2);

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, r1_node, r2_node, &ss)
            .expect("cost should exist");

        // earliest arrival = 2; to.start=5 => wait=0; proc_to=3; weight=11 => 33
        assert_eq!(cost, 33);
    }

    #[test]
    fn test_to_not_feasible_on_chain_returns_none() {
        // Two berths
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        // from (feasible on b1)
        let r1 = req_flex(1, (0, 100), 1, &[(1, 2)]);
        // to (feasible only on b2)
        let r2 = req_flex(2, (0, 100), 5, &[(2, 3)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        let p: Problem<i64> = builder.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let policy = WeightedFlowTime::new(&model);
        let ss = SolverSearchState::new(&model, policy);

        let im = model.index_manager();
        let chain_b1 = ChainIndex::new(im.berth_index(bid(1)).unwrap().get());
        let r1_node = NodeIndex::new(im.request_index(rid(1)).unwrap().get());
        let r2_node = NodeIndex::new(im.request_index(rid(2)).unwrap().get());

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval.evaluate_arc_cost(chain_b1, r1_node, r2_node, &ss);
        assert!(
            cost.is_none(),
            "expected None when 'to' is infeasible on chain"
        );
    }

    #[test]
    fn test_from_not_feasible_on_chain_returns_none() {
        // Two berths so the problem is valid
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        // from is feasible only on b2
        let r_from = req_flex(10, (0, 100), 1, &[(2, 7)]);
        // to is feasible only on b1
        let r_to = req_flex(20, (0, 100), 9, &[(1, 3)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r_from);
        builder.add_flexible(r_to);
        let p: Problem<i64> = builder.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let policy = WeightedFlowTime::new(&model);
        let ss = SolverSearchState::new(&model, policy);

        let im = model.index_manager();
        let chain_b1 = ChainIndex::new(im.berth_index(bid(1)).unwrap().get());
        let n_from = NodeIndex::new(im.request_index(rid(10)).unwrap().get());
        let n_to = NodeIndex::new(im.request_index(rid(20)).unwrap().get());

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval.evaluate_arc_cost(chain_b1, n_from, n_to, &ss);
        assert!(
            cost.is_none(),
            "expected None when 'from' is infeasible on chain"
        );
    }

    #[test]
    fn test_weight_saturating_mul_on_large_values() {
        // Self-arc with proc_to small but weight very large
        let b1 = Berth::from_windows(bid(1), [iv(0, 10)]);
        let r = req_flex(99, (0, 10), i64::MAX, &[(1, 2)]); // weight = MAX, proc_to = 2

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r);
        let p: Problem<i64> = builder.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let policy = WeightedFlowTime::new(&model);
        let ss = SolverSearchState::new(&model, policy);

        let (chain_idx, n) = require_indices_for(&model, 1, 99);

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, n, n, &ss)
            .expect("cost should exist");

        // time_cost = 2; i64::MAX * 2 saturates to i64::MAX
        assert_eq!(cost, i64::MAX);
    }
}
