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
    index::{BerthIndex, RequestIndex},
    model::SolverModel,
};
use berth_alloc_core::prelude::{Cost, TimeDelta};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait ArcCostEvaluator<T: Copy + Ord> {
    fn name(&self) -> &'static str {
        // Default implementation returns the type name
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
        model: &SolverModel<T>,
    ) -> Option<Cost>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DefaultArcCostEvaluator;

impl<T> ArcCostEvaluator<T> for DefaultArcCostEvaluator
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug + Into<Cost>,
{
    fn evaluate_arc_cost(
        &self,
        chain: ChainIndex,
        from_node: NodeIndex,
        to_node: NodeIndex,
        model: &SolverModel<T>,
    ) -> Option<Cost> {
        let berth_idx = BerthIndex(chain.get());
        let to_req_idx = RequestIndex(to_node.get());

        // Use the cached scheduled cost = weight * processing_time(on berth)
        let scheduled_cost = model.cost(to_req_idx, berth_idx)?;

        // We still need proc_from to build an earliest-arrival LB.
        let to_window = model.feasible_intervals()[to_req_idx.get()];
        let earliest_arrival_lb = if from_node == to_node {
            to_window.start()
        } else {
            let from_req_idx = RequestIndex(from_node.get());
            match model.processing_time(from_req_idx, berth_idx) {
                Some(Some(proc_from)) => {
                    let from_window = model.feasible_intervals()[from_req_idx.get()];
                    from_window
                        .start()
                        .checked_add(proc_from)
                        .unwrap_or(to_window.end())
                }
                _ => return None, // predecessor infeasible on this berth
            }
        };

        // wait_lb = max(0, earliest_arrival_lb - to_window.start())
        let zero = TimeDelta::zero();
        let wait_lb = {
            let d = earliest_arrival_lb - to_window.start();
            if d < zero { zero } else { d }
        };

        // Wait cost uses the same per-request weight; convert domain Δ to Cost.
        let weight = model.weights()[to_req_idx.get()];
        let wait_cost: Cost = weight.saturating_mul(wait_lb.value().into());

        // LB = cached scheduled_cost (weight*proc_to) + weight*wait_lb
        Some(wait_cost.saturating_add(scheduled_cost))
    }
}

#[cfg(test)]
mod tests {
    use crate::state::cost_policy::WeightedFlowTime;

    use super::*;
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

    // Helper to get indices robustly via the model's index manager
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
    fn test_name_contains_type() {
        let eval = DefaultArcCostEvaluator::default();
        let name = <DefaultArcCostEvaluator as ArcCostEvaluator<i64>>::name(&eval);
        assert!(
            name.contains("DefaultArcCostEvaluator"),
            "name was {}",
            name
        );
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
        let m = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();

        let (chain_idx, to_node) = require_indices_for(&m, 1, 20);
        let from_node = to_node; // self-arc

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, from_node, to_node, &m)
            .expect("cost should exist");

        // Self-arc uses wait = 0 and proc_to = 5, weight = 7 => 7 * 5 = 35
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
        let m = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();

        let (chain_idx, r10_node) = require_indices_for(&m, 1, 10);
        let (_, r20_node) = require_indices_for(&m, 1, 20);

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, r10_node, r20_node, &m)
            .expect("cost should exist");

        // from_window.start = 0, proc_from = 10 => earliest arrival LB = 10
        // to_window.start = 0 => wait = 10
        // proc_to = 5, total_time = 15
        // weight(to) = 7 => 7 * 15 = 105
        assert_eq!(cost, 105);
    }

    #[test]
    fn test_zero_wait_case() {
        // Berth 1
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        // r1: from, proc 2 on berth 1
        let r1 = req_flex(1, (0, 100), 1, &[(1, 2)]);
        // r2: to, window [5,100), proc 3 on berth 1, weight 11
        let r2 = req_flex(2, (5, 100), 11, &[(1, 3)]);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        let p: Problem<i64> = builder.build().unwrap();
        let m = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();

        let (chain_idx, r1_node) = require_indices_for(&m, 1, 1);
        let (_, r2_node) = require_indices_for(&m, 1, 2);

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, r1_node, r2_node, &m)
            .expect("cost should exist");

        // earliest arrival = from.start(0) + 2 = 2; to.start = 5 => wait LB = max(0, 2-5)=0
        // total_time = 0 + proc_to(3) = 3; weight(to) = 11 => 33
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
        let m = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();

        let im = m.index_manager();
        let chain_b1 = ChainIndex::new(im.berth_index(bid(1)).unwrap().get());
        let r1_node = NodeIndex::new(im.request_index(rid(1)).unwrap().get());
        let r2_node = NodeIndex::new(im.request_index(rid(2)).unwrap().get());

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval.evaluate_arc_cost(chain_b1, r1_node, r2_node, &m);
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
        let m = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();

        let im = m.index_manager();
        let chain_b1 = ChainIndex::new(im.berth_index(bid(1)).unwrap().get());
        let n_from = NodeIndex::new(im.request_index(rid(10)).unwrap().get());
        let n_to = NodeIndex::new(im.request_index(rid(20)).unwrap().get());

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval.evaluate_arc_cost(chain_b1, n_from, n_to, &m);
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
        let m = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();

        let (chain_idx, n) = require_indices_for(&m, 1, 99);

        let eval = DefaultArcCostEvaluator::default();
        let cost = eval
            .evaluate_arc_cost(chain_idx, n, n, &m)
            .expect("cost should exist");

        // time_cost = 2; i64::MAX * 2 saturates to i64::MAX
        assert_eq!(cost, i64::MAX);
    }
}
