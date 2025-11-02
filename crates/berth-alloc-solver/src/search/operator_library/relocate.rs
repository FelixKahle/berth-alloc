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
    model::index::RequestIndex,
    search::{
        eval::CostEvaluator,
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::Plan,
        solver_state::SolverStateView,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

/// RelocateOp: moves a single assigned request to another feasible slot
/// across its allowed berths (including same berth, different time).
///
/// Strategy:
/// - Iterate requests in index order.
/// - For the first assigned request:
///   - Unassign it to open the slot.
///   - Enumerate free intervals on the union of its allowed berths
///     within its feasible window using the builder’s iterators.
///   - Choose the earliest feasible start that fits the processing time,
///     skipping the exact (berth,start) it previously had.
///   - Reassign and yield a plan (unassign + assign → delta_unassigned = 0).
#[derive(Debug, Default)]
pub struct RelocateOp {
    i: usize,
}

impl RelocateOp {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, C, R> LocalSearchOperator<T, C, R> for RelocateOp
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RelocateOp"
    }

    fn reset(&mut self) {
        self.i = 0;
    }

    fn has_fragments(&self) -> bool {
        false
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let dvars = ctx.state().decision_variables();
        let n = dvars.len();

        while self.i < n {
            let r = RequestIndex::new(self.i);
            self.i += 1;

            let (b_old, s_old) = match dvars.get(r.get()).copied() {
                Some(DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                })) => (berth_index, start_time),
                _ => continue,
            };

            let mut pb = ctx.builder();
            let sp = pb.savepoint();

            if pb.propose_unassignment(r).is_err() {
                pb.undo_to(sp);
                continue;
            }

            let mut candidates: Vec<_> = pb
                .iter_free_for(r)
                .filter_map(|fb| {
                    let start = fb.interval().start();
                    if fb.berth_index() == b_old && start == s_old {
                        return None;
                    }
                    Some((start, fb.clone()))
                })
                .collect();
            candidates.sort_by_key(|(start, _)| *start);

            for (start, fb) in candidates {
                if pb.propose_assignment(r, start, &fb).is_ok() {
                    return Some(pb.finalize());
                }
            }

            pb.undo_to(sp);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::{eval::DefaultCostEvaluator, operator::OperatorContext},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            solver_state::{SolverState, SolverStateView},
            terminal::terminalocc::{TerminalOccupancy, TerminalWrite},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    type TT = i64;

    #[inline]
    fn tp(v: TT) -> TimePoint<TT> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: TT, b: TT) -> TimeInterval<TT> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: TT) -> TimeDelta<TT> {
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

    fn berth(id: u32, s: TT, e: TT) -> Berth<TT> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (TT, TT),
        pts: &[(u32, TT)],
        weight: TT,
    ) -> Request<FlexibleKind, TT> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, TT>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    // Two berths, one flexible request allowed on both.
    fn problem_two_berths_one_req() -> Problem<TT> {
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        let r1 = flex_req(10, (0, 100), &[(1, 10), (2, 10)], 1);
        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        pb.build().expect("valid problem")
    }

    #[test]
    fn test_relocate_moves_request() {
        let problem = problem_two_berths_one_req();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Initial state: r1 assigned on b1 at t=20.
        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1.get()] = DecisionVar::assigned(b1, tp(20));

        let mut term = TerminalOccupancy::new(model.berths());
        let iv1 = model.interval(r1, b1, tp(20)).unwrap();
        term.occupy(b1, iv1).unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = RelocateOp::new();
        let plan = op
            .make_next_neighbor(&mut ctx)
            .expect("relocate should produce a neighbor");

        // Unassign + assign => delta_unassigned == 0, at least 2 patches
        assert_eq!(plan.fitness_delta.delta_unassigned, 0);
        assert!(plan.decision_var_patches.len() >= 2);

        // Apply and check that the new assignment is either on b2, or on b1 at a different start.
        let mut st = state.clone();
        st.apply_plan(plan);
        let dv = st.decision_variables()[r1.get()];
        let asg = dv.as_assigned().expect("still assigned");
        assert!(
            asg.berth_index == b2 || (asg.berth_index == b1 && asg.start_time != tp(20)),
            "relocation must change berth or start time"
        );
    }

    #[test]
    fn test_relocate_skips_when_unassigned() {
        let problem = problem_two_berths_one_req();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Initial state: unassigned
        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();

        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let term = TerminalOccupancy::new(model.berths());
        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = RelocateOp::new();
        let plan = op.make_next_neighbor(&mut ctx);
        assert!(plan.is_none(), "no relocation when request is unassigned");

        // Ensure the DV remains unassigned in the original state
        assert!(matches!(
            state.decision_variables()[r1.get()],
            DecisionVar::Unassigned
        ));
    }

    #[test]
    fn test_relocate_noop_not_emitted_when_same_slot_only_free() {
        // Arrange problem with 1 request allowed on both berths but we'll build terminal such that
        // for the assigned berth the only free slot after unassign is exactly the previous slot.
        let problem = problem_two_berths_one_req();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        // r1 assigned on b1@[10,20)
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1.get()] = DecisionVar::assigned(b1, tp(10));

        // Occupancy: everything except [10,20) is occupied on b1, so after unassign the only free
        // interval will be exactly [10,20). Also fully occupy b2 to prevent cross-berth relocation.
        let mut term = TerminalOccupancy::new(model.berths());
        let iv_curr = model.interval(r1, b1, tp(10)).unwrap();
        // First occupy the current slot (to match DV state)
        term.occupy(b1, iv_curr).unwrap();
        // Then occupy [0,10) and [20,100) to remove alternatives on b1
        term.occupy(b1, iv(0, 10)).unwrap();
        term.occupy(b1, iv(20, 100)).unwrap();
        // Block b2 completely to avoid relocation to the other allowed berth
        term.occupy(b2, iv(0, 100)).unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = RelocateOp::new();
        let plan = op.make_next_neighbor(&mut ctx);
        assert!(
            plan.is_none(),
            "relocate must not emit a no-op when only exact same slot is available"
        );
    }

    #[test]
    fn test_relocate_picks_earliest_across_allowed_berths() {
        let problem = problem_two_berths_one_req();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        // r1 assigned on b1@t=30
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1.get()] = DecisionVar::assigned(b1, tp(30));

        // Terminal: create free gap on b2 starting at 5, and on b1 earliest at 12
        // Occupy b1 so earliest free becomes >= 12 (e.g., occupy [0,12) and [40,100))
        let mut term = TerminalOccupancy::new(model.berths());
        term.occupy(b1, iv(0, 12)).unwrap();
        term.occupy(b1, iv(40, 100)).unwrap();
        // Mark the current assignment on b1@30
        let iv_curr = model.interval(r1, b1, tp(30)).unwrap();
        term.occupy(b1, iv_curr).unwrap();
        // On b2, leave earliest gap at [5, 100) (so earliest candidate is 5)
        term.occupy(b2, iv(0, 5)).unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = RelocateOp::new();
        let plan = op
            .make_next_neighbor(&mut ctx)
            .expect("relocate should produce a plan");

        let mut st = state.clone();
        st.apply_plan(plan);
        let asg = st.decision_variables()[r1.get()]
            .as_assigned()
            .unwrap()
            .clone();

        assert_eq!(asg.start_time, tp(5), "should pick earliest start");
        assert_eq!(asg.berth_index, b2, "earliest was on b2");
    }

    #[test]
    fn test_relocate_respects_feasible_window() {
        // r1 window [20,100), assigned at 40 initially
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        let r1 = flex_req(10, (20, 100), &[(1, 10), (2, 10)], 1);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        let problem = pb.build().expect("valid");
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let im = model.index_manager();
        let r1_ix = im.request_index(rid(10)).unwrap();
        let b1_ix = im.berth_index(bid(1)).unwrap();
        let b2_ix = im.berth_index(bid(2)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1_ix.get()] = DecisionVar::assigned(b1_ix, tp(40));

        // Occupancy:
        // - On b1: block [0,20) and [30,40) and [50,100). Mark current [40,50).
        //          This leaves [20,30) as the earliest free segment inside the feasible window.
        // - On b2: fully occupied to force staying on b1.
        let mut term = TerminalOccupancy::new(model.berths());
        term.occupy(b1_ix, iv(0, 20)).unwrap();
        let iv_curr = model.interval(r1_ix, b1_ix, tp(40)).unwrap(); // [40,50)
        term.occupy(b1_ix, iv_curr).unwrap();
        term.occupy(b1_ix, iv(30, 40)).unwrap();
        term.occupy(b1_ix, iv(50, 100)).unwrap();
        term.occupy(b2_ix, iv(0, 100)).unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(14);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = RelocateOp::new();
        let plan = op
            .make_next_neighbor(&mut ctx)
            .expect("relocate should produce a plan");

        let mut st = state.clone();
        st.apply_plan(plan);
        let asg = st.decision_variables()[r1_ix.get()]
            .as_assigned()
            .unwrap()
            .clone();

        assert_eq!(asg.start_time, tp(20), "must respect feasible window start");
        assert_eq!(
            asg.berth_index, b1_ix,
            "forced onto b1 due to b2 being blocked"
        );
    }

    #[test]
    fn test_relocate_prunes_not_allowed_berths() {
        // r1 only allowed on b1; ensure we never relocate to b2 even if it has an earlier slot.
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        // Allowed only on b1:
        let r1 = flex_req(10, (0, 100), &[(1, 10)], 1);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        let problem = pb.build().expect("valid");

        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let im = model.index_manager();
        let r1_ix = im.request_index(rid(10)).unwrap();
        let b1_ix = im.berth_index(bid(1)).unwrap();
        let b2_ix = im.berth_index(bid(2)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1_ix.get()] = DecisionVar::assigned(b1_ix, tp(50));

        // Terminal: b2 has early gap at 10, b1 earliest gap at 30
        let mut term = TerminalOccupancy::new(model.berths());
        // mark current on b1
        let iv_curr = model.interval(r1_ix, b1_ix, tp(50)).unwrap();
        term.occupy(b1_ix, iv_curr).unwrap();
        // on b1, block [0,30) so earliest becomes 30 after unassign
        term.occupy(b1_ix, iv(0, 30)).unwrap();
        // on b2, early gap at 10..100 (making 10 earlier than 30)
        term.occupy(b2_ix, iv(0, 10)).unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(15);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = RelocateOp::new();
        let plan = op
            .make_next_neighbor(&mut ctx)
            .expect("relocate should still produce a plan on allowed berth");

        let mut st = state.clone();
        st.apply_plan(plan);
        let asg = st.decision_variables()[r1_ix.get()]
            .as_assigned()
            .unwrap()
            .clone();

        assert_eq!(
            asg.berth_index, b1_ix,
            "must not relocate to disallowed berth"
        );
        assert_eq!(
            asg.start_time,
            tp(30),
            "must pick earliest on allowed berth"
        );
    }
}
