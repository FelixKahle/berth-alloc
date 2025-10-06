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
    search::scheduling::{err::SchedulingError, scheduler::Scheduler},
    state::{
        chain_set::view::ChainSetView, cost_policy::CostPolicy, index::RequestIndex,
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait ScheduleScorer<T: Copy + Ord + CheckedAdd + CheckedSub + Zero, P: CostPolicy<T>> {
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn score_exact<C: ChainSetView, S: Scheduler<T>>(
        &self,
        solver_state: &SolverSearchState<T, P>,
        chains: &C,
        scheduler: &S,
    ) -> Result<Cost, SchedulingError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefaultScheduleScorer;

impl<T, P> ScheduleScorer<T, P> for DefaultScheduleScorer
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    P: CostPolicy<T>,
{
    fn score_exact<C: ChainSetView, S: Scheduler<T>>(
        &self,
        state: &SolverSearchState<T, P>,
        chains: &C,
        scheduler: &S,
    ) -> Result<Cost, crate::search::scheduling::err::SchedulingError> {
        let model = state.model();
        let policy = state.cost_policy();
        let zero = TimeDelta::zero();

        let mut total: Cost = 0.into();
        for i in 0..model.flexible_requests_len() {
            total = total.saturating_add(policy.unperformed_penalty(RequestIndex(i)));
        }

        scheduler.process_schedule(state, chains, |sched| {
            let r = sched.request_index();
            let iv = sched.assigned_time_interval();
            let win = model.feasible_intervals()[r.get()];
            let wgt = model.weights()[r.get()];
            let penalty = policy.unperformed_penalty(r);
            let bi = sched.berth_index();
            let proc_cost = policy
                .scheduled_cost(r, bi)
                .expect("scheduler produced (req, berth) that is infeasible");

            let wait = {
                let d = iv.start() - win.start();
                if d < zero { zero } else { d }
            };
            let wait_cost: Cost = wgt.saturating_mul(wait.value().into());

            let scheduled_total = proc_cost.saturating_add(wait_cost);
            total = total
                .saturating_add(scheduled_total)
                .saturating_sub(penalty);
        })?;

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        search::scheduling::{err::SchedulingError, greedy::GreedyEarliest},
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
        weight: i64,
        pts: &[(usize, i64)],
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
    fn make_state<'p>(
        model: &'p SolverModel<'p, i64>,
    ) -> SolverSearchState<'p, 'p, i64, WeightedFlowTime<'p, 'p, i64>> {
        SolverSearchState::new(model, WeightedFlowTime::new(model))
    }

    // Link a sequence into a given chain: head -> n0 -> n1 -> ... -> tail
    fn link_sequence(cs: &mut ChainSet, chain: ChainIndex, nodes: &[usize]) {
        let mut builder = ChainSetDeltaBuilder::new(cs);
        let start_node = cs.start_of_chain(chain);
        let mut current_tail = start_node;
        for &node_id in nodes {
            let node_to_link = NodeIndex(node_id);
            builder.insert_after(current_tail, node_to_link);
            current_tail = node_to_link;
        }
        cs.apply_delta(&builder.build());
    }

    #[test]
    fn test_score_exact_simple_two_on_one_chain() {
        // One berth, two requests with windows [0,100)
        // r10: d1=5, w1=3
        // r20: d2=7, w2=11
        // Chain: [r10, r20] → schedule [0,5) then [5,12).
        // Score = w1*(0 + d1) + w2*(5 + d2) = 3*5 + 11*(5+7) = 147
        let b = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r10 = req_flex(10, (0, 100), 3, &[(1, 5)]);
        let r20 = req_flex(20, (0, 100), 11, &[(1, 7)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b);
        pb.add_flexible(r10);
        pb.add_flexible(r20);
        let p: Problem<i64> = pb.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let state = make_state(&model);

        // indices: rid(10)->0, rid(20)->1
        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let scorer = DefaultScheduleScorer;
        let cost = scorer
            .score_exact(&state, &cs, &GreedyEarliest)
            .expect("schedule should be feasible");

        assert_eq!(cost, 147);
    }

    #[test]
    fn test_score_exact_unperformed_penalty_applied() {
        // One berth, two flexible requests; only one is linked into the chain.
        // r10: d=5, w=3
        // r20: window [0,100), w=11, d=7 -> not scheduled -> penalty = 100*11 = 1100
        // Score = 3*(0+5) + 1100 = 1115
        let b = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r10 = req_flex(10, (0, 100), 3, &[(1, 5)]);
        let r20 = req_flex(20, (0, 100), 11, &[(1, 7)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b);
        pb.add_flexible(r10);
        pb.add_flexible(r20);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        // Only schedule the first request (index 0)
        link_sequence(&mut cs, ChainIndex(0), &[0]);

        let scorer = DefaultScheduleScorer;
        let cost = scorer
            .score_exact(&state, &cs, &GreedyEarliest)
            .expect("schedule should be feasible");

        assert_eq!(cost, 1115);
    }

    #[test]
    fn test_score_exact_not_allowed_on_berth_error() {
        // Two berths; request feasible only on berth 2 but placed on chain for berth 1.
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        let r = req_flex(10, (0, 100), 1, &[(2, 5)]); // only on berth 2

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        // Link the single request onto chain 0 (berth 1)
        link_sequence(&mut cs, ChainIndex(0), &[0]);

        let scorer = DefaultScheduleScorer;
        let res = scorer.score_exact(&state, &cs, &GreedyEarliest);
        assert!(matches!(res, Err(SchedulingError::NotAllowedOnBerth(_))));
    }

    #[test]
    fn test_score_exact_feasible_window_violation_error() {
        // One berth; rA length 8, rB length 5 with window [0,10].
        // Order [rA, rB] forces rB earliest start at 8, end=13>10 -> violation.
        let b = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r_a = req_flex(10, (0, 100), 1, &[(1, 8)]);
        let r_b = req_flex(20, (0, 10), 1, &[(1, 5)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b);
        pb.add_flexible(r_a);
        pb.add_flexible(r_b);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let scorer = DefaultScheduleScorer;
        let res = scorer.score_exact(&state, &cs, &GreedyEarliest);
        assert!(matches!(
            res,
            Err(SchedulingError::FeasiblyWindowViolation(_))
        ));
    }

    #[test]
    fn test_score_exact_multiple_chains_sum_costs() {
        // Two berths, each with one request starting at 0.
        // Costs sum: w1*proc1 + w2*proc2 = 3*5 + 7*4 = 43
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        let r1 = req_flex(10, (0, 100), 3, &[(1, 5)]);
        let r2 = req_flex(20, (0, 100), 7, &[(2, 4)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        pb.add_flexible(r2);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        // ids sorted: rid(10)->0 on chain 0 (berth 1), rid(20)->1 on chain 1 (berth 2)
        link_sequence(&mut cs, ChainIndex(0), &[0]);
        link_sequence(&mut cs, ChainIndex(1), &[1]);

        let scorer = DefaultScheduleScorer;
        let cost = scorer
            .score_exact(&state, &cs, &GreedyEarliest)
            .expect("schedule should be feasible");

        assert_eq!(cost, 43);
    }

    #[test]
    fn test_score_exact_respects_fixed_and_finds_earliest_fit() {
        // b1 availability [0,100), fixed [10,20).
        // Two flex jobs of 7 each on b1, both [0,100).
        // Expected schedule: [0,7) then [20,27) due to fixed blocking [10,20).
        // Score: rA wait=0, proc=7, w=2 => 14
        //        rB wait=20, proc=7, w=3 => 81
        // Total = 95
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let rf = req_fixed(900, (0, 100), &[(1, 10)]);
        let fixed = asg_fixed(&rf, &b1, 10); // occupies [10,20)

        let r_a = req_flex(10, (0, 100), 2, &[(1, 7)]);
        let r_b = req_flex(20, (0, 100), 3, &[(1, 7)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_fixed(fixed);
        pb.add_flexible(r_a);
        pb.add_flexible(r_b);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        // order [r_a, r_b] by indices [0,1] on chain 0
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let scorer = DefaultScheduleScorer;
        let cost = scorer
            .score_exact(&state, &cs, &GreedyEarliest)
            .expect("schedule should be feasible");

        assert_eq!(cost, 95);
    }
}
