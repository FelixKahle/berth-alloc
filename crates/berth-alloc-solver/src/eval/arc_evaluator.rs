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

use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};

use crate::{
    eval::objective::Objective,
    state::{
        chain_set::{
            index::NodeIndex,
            view::{ChainRef, ChainSetView},
        },
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
    },
};

pub trait ArcEvaluator<T: Copy + Ord> {
    fn evaluate<V>(
        &self,
        model: &SolverModel<T>,
        chain: ChainRef<'_, V>,
        from: NodeIndex,
        to: NodeIndex,
    ) -> Option<Cost>
    where
        V: ChainSetView;
}

#[derive(Debug, Clone)]
pub struct ObjectiveArcEvaluator<'objective, T, O>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    O: Objective<T>,
{
    objective: &'objective O,
    _phantom: std::marker::PhantomData<T>,
}

impl<'objective, T, O> ObjectiveArcEvaluator<'objective, T, O>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    O: Objective<T>,
{
    pub fn new(objective: &'objective O) -> Self {
        Self {
            objective,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn objective(&self) -> &O {
        self.objective
    }
}

impl<'objective, T, O> ArcEvaluator<T> for ObjectiveArcEvaluator<'objective, T, O>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    O: Objective<T>,
{
    fn evaluate<V>(
        &self,
        model: &SolverModel<T>,
        chain: ChainRef<'_, V>,
        from: NodeIndex,
        to: NodeIndex,
    ) -> Option<Cost>
    where
        V: ChainSetView,
    {
        // Chain ↔ berth: chain index is the berth index.
        let bi = BerthIndex(chain.chain_index().get());
        if bi.0 >= model.berths_len() {
            return None;
        }

        // We want the arc [from, to) — exclusive on `to`.
        // resolve_slice gives us the first real node to process and an exclusive bound.
        let (cur_opt, end_exclusive) = chain.resolve_slice(from, Some(to));

        // Empty slice is fine (e.g., from == to or sentinel directly before `to`).
        let Some(mut cur) = cur_opt else {
            return Some(0);
        };

        let mut acc: Cost = 0;

        // Defensive guard to avoid infinite walks on malformed chains.
        let mut steps_left = model.flexible_requests_len();

        while cur != end_exclusive {
            if steps_left == 0 {
                // Didn't reach `to` within the number of requests → likely not reachable/cycle.
                return None;
            }
            steps_left -= 1;

            // Map node -> request index; bounds check against model arrays.
            let ri = RequestIndex(cur.get());
            if ri.0 >= model.flexible_requests_len() {
                return None;
            }

            // Use a consistent surrogate start time (window start). If you have actual
            // scheduled start times available elsewhere, you can plug them in here.
            let start_time = model.feasible_intervals()[ri.0].start();

            // Delegate to objective; if the objective can’t price it, treat it as unreachable.
            let c = self.objective.assignment_cost(model, ri, bi, start_time)?;
            acc += c;

            // Advance; if we fall off the chain before seeing `to`, it’s not reachable.
            cur = chain.next(cur)?;
        }

        Some(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::wtt::WeightedTurnaroundTimeObjective;
    use crate::state::chain_set::{
        base::ChainSet,
        delta::{ChainNextRewire, ChainSetDelta},
        index::ChainIndex,
        view::ChainSetView,
    };
    use crate::state::model::SolverModel;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    // ---------- helpers ----------
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

    // Build a Problem with explicit weights.
    // berths_windows[b] = vec![(s,e), ...] availability windows (ids 0..B-1)
    // request_windows[r] = (s,e) feasible window (ids 0..R-1)
    // weights[r] = weight
    // processing[r][b] = Some(dur) if allowed on berth b; None otherwise
    fn build_problem(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        weights: &[u64],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let b_len = berths_windows.len();
        let r_len = request_windows.len();
        assert_eq!(weights.len(), r_len);
        assert_eq!(processing.len(), r_len);
        for row in processing {
            assert_eq!(row.len(), b_len, "processing rows must match #berths");
        }

        let mut builder = ProblemBuilder::new();

        for (i, windows) in berths_windows.iter().enumerate() {
            let b = Berth::from_windows(bid(i), windows.iter().map(|&(s, e)| iv(s, e)));
            builder.add_berth(b);
        }

        for (i, &(ws, we)) in request_windows.iter().enumerate() {
            let mut map = BTreeMap::new();
            for (j, p) in processing[i].iter().copied().enumerate() {
                if let Some(dur) = p {
                    map.insert(bid(j), td(dur));
                }
            }
            let req = Request::<FlexibleKind, i64>::new(rid(i), iv(ws, we), weights[i] as i64, map)
                .unwrap();
            builder.add_flexible(req);
        }

        builder.build().expect("problem should build")
    }

    // Link request nodes onto chain c: start -> nodes[0] -> ... -> nodes[k] -> end
    fn link_chain(cs: &mut ChainSet, c: usize, nodes: &[usize]) {
        let s = cs.start_of_chain(ChainIndex(c));
        let e = cs.end_of_chain(ChainIndex(c));
        if nodes.is_empty() {
            return;
        }
        let mut delta = ChainSetDelta::new();
        delta.push_rewire(ChainNextRewire::new(s, NodeIndex(nodes[0])));
        for w in nodes.windows(2) {
            delta.push_rewire(ChainNextRewire::new(NodeIndex(w[0]), NodeIndex(w[1])));
        }
        delta.push_rewire(ChainNextRewire::new(NodeIndex(*nodes.last().unwrap()), e));
        cs.apply_delta(delta);
    }

    fn make_eval<'a, T, O>(objective: &'a O) -> ObjectiveArcEvaluator<'a, T, O>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        O: Objective<T>,
    {
        ObjectiveArcEvaluator {
            objective,
            _phantom: std::marker::PhantomData,
        }
    }

    // ---------- tests ----------

    #[test]
    fn self_arc_is_empty_slice_and_cost_is_zero() {
        // One berth, one request with weight=3 and pt=5.
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[3], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        // Chain with node 0
        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let obj = WeightedTurnaroundTimeObjective;
        let eval = make_eval::<i64, _>(&obj);

        // from == to => empty slice => cost 0
        let cost = eval.evaluate(&m, c0, NodeIndex(0), NodeIndex(0));
        assert_eq!(cost, Some(0));
    }

    #[test]
    fn reachable_arc_single_node_returns_weight_times_pt() {
        // weight=4, pt=7 => cost = 28
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[4], &[vec![Some(7)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);
        let c0 = cs.chain(ChainIndex(0));

        let obj = WeightedTurnaroundTimeObjective;
        let eval = make_eval::<i64, _>(&obj);

        // Evaluate from head sentinel to tail sentinel (sums node 0)
        let from = cs.start_of_chain(ChainIndex(0));
        let to = cs.end_of_chain(ChainIndex(0));
        let cost = eval.evaluate(&m, c0, from, to);
        assert_eq!(cost, Some(4 * 7));
    }

    #[test]
    fn reachable_arc_multiple_nodes_sums_costs() {
        // Request 0: w=2, pt=3 => 6
        // Request 1: w=5, pt=4 => 20
        // Total => 26
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[2, 5],
            &[vec![Some(3)], vec![Some(4)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]);
        let c0 = cs.chain(ChainIndex(0));

        let obj = WeightedTurnaroundTimeObjective;
        let eval = make_eval::<i64, _>(&obj);

        let from = cs.start_of_chain(ChainIndex(0));
        let to = cs.end_of_chain(ChainIndex(0));
        let cost = eval.evaluate(&m, c0, from, to);
        assert_eq!(cost, Some(26));
    }

    #[test]
    fn unreachable_arc_returns_none() {
        // Chain order is 0 -> 1; asking for arc from 1 to 0 is not reachable in forward traversal.
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[1, 1],
            &[vec![Some(2)], vec![Some(2)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]);
        let c0 = cs.chain(ChainIndex(0));

        let obj = WeightedTurnaroundTimeObjective;
        let eval = make_eval::<i64, _>(&obj);

        let cost = eval.evaluate(&m, c0, NodeIndex(1), NodeIndex(0));
        assert_eq!(cost, None);
    }

    #[test]
    fn not_allowed_on_chain_berth_returns_none() {
        // Two berths; request allowed only on berth 1. Evaluating on chain 0 should return None.
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100)],
            &[7],
            &[vec![None, Some(5)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]); // chain 0 ↔ berth 0
        let c0 = cs.chain(ChainIndex(0));

        let obj = WeightedTurnaroundTimeObjective;
        let eval = make_eval::<i64, _>(&obj);

        let from = cs.start_of_chain(ChainIndex(0));
        let to = cs.end_of_chain(ChainIndex(0));
        let cost = eval.evaluate(&m, c0, from, to);
        assert_eq!(cost, None);
    }

    #[test]
    fn chain_index_drives_berth_for_costs_in_multi_berth_case() {
        // Single request r0 with w=2.
        // pt on berth 0 = 10 => cost 20
        // pt on berth 1 = 1  => cost 2
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100)],
            &[2],
            &[vec![Some(10), Some(1)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let obj = WeightedTurnaroundTimeObjective;
        let eval = make_eval::<i64, _>(&obj);

        // Evaluate on chain 0 (berth 0) using a fresh ChainSet
        {
            let mut cs0 = ChainSet::new(m.flexible_requests_len(), m.berths_len());
            link_chain(&mut cs0, 0, &[0]); // chain 0 ↔ [0]
            let c0 = cs0.chain(ChainIndex(0));
            let from0 = cs0.start_of_chain(ChainIndex(0));
            let to0 = cs0.end_of_chain(ChainIndex(0));
            let cost0 = eval.evaluate(&m, c0, from0, to0);
            assert_eq!(cost0, Some(2 * 10));
        }

        // Evaluate on chain 1 (berth 1) using another fresh ChainSet
        {
            let mut cs1 = ChainSet::new(m.flexible_requests_len(), m.berths_len());
            link_chain(&mut cs1, 1, &[0]); // chain 1 ↔ [0]
            let c1 = cs1.chain(ChainIndex(1));
            let from1 = cs1.start_of_chain(ChainIndex(1));
            let to1 = cs1.end_of_chain(ChainIndex(1));
            let cost1 = eval.evaluate(&m, c1, from1, to1);
            assert_eq!(cost1, Some(2 * 1));
        }
    }
}
