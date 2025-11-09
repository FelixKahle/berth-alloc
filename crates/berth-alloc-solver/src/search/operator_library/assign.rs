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
    state::{plan::Plan, solver_state::SolverStateView, terminal::terminalocc::FreeBerth},
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

/// Tries to assign an unassigned request to its earliest feasible slot.
///
/// This operator iterates through all requests. If a request is unassigned,
/// it queries the `PlanBuilder` for all feasible free slots (respecting
/// allowed berths and time windows) and attempts to create a plan
/// that assigns the request to the *earliest* possible start time found.
#[derive(Debug, Default)]
pub struct AssignOp {
    /// The index of the next request to check.
    i: usize,
}

impl AssignOp {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, C, R> LocalSearchOperator<T, C, R> for AssignOp
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "AssignOp"
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

            // --- Key logic: only consider unassigned requests ---
            if dvars.get(r.get()).is_some_and(|dv| dv.is_assigned()) {
                continue;
            }

            let mut pb = ctx.builder();

            // Find the earliest possible start time across all allowed berths/windows
            let best_slot: Option<FreeBerth<T>> =
                pb.iter_free_for(r).min_by_key(|fb| fb.interval().start());

            if let Some(fb) = best_slot {
                let sp = pb.savepoint();
                let start = fb.interval().start();

                // Propose the assignment
                if pb.propose_assignment(r, start, &fb).is_ok() {
                    // Yield the plan (unassigned -> assigned)
                    return Some(pb.finalize());
                } else {
                    // This should ideally not fail if iter_free_for is correct,
                    // but we roll back just in case.
                    pb.undo_to(sp);
                }
            }
            // If no slot was found, or assignment failed,
            // pb is dropped (discarding changes) and we continue to the next request.
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

    // One berth [0, 100]
    // r10: pt=10, window [0, 100]
    // r20: pt=5, window [15, 100] (note the later start)
    fn problem_two_reqs_one_berth() -> Problem<TT> {
        let b1 = berth(1, 0, 100);
        let r10 = flex_req(10, (0, 100), &[(1, 10)], 1);
        let r20 = flex_req(20, (15, 100), &[(1, 5)], 1);
        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r10);
        pb.add_flexible(r20);
        pb.build().expect("valid problem")
    }

    // State with r10 unassigned, r20 assigned at t=20..25
    fn state_r10_unassigned_r20_assigned<'p>(
        model: &'p SolverModel<'p, TT>,
        eval: &impl CostEvaluator<TT>,
    ) -> SolverState<'p, TT> {
        let im = model.index_manager();
        let r20 = im.request_index(rid(20)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r20.get()] = DecisionVar::assigned(b1, tp(20)); // r10 (index 0) remains unassigned

        let mut term = TerminalOccupancy::new(model.berths());
        let iv20 = model.interval(r20, b1, tp(20)).unwrap(); // [20, 25]
        term.occupy(b1, iv20).unwrap();

        let fit = eval.eval_fitness(model, &dvars);
        SolverState::new(DecisionVarVec::from(dvars), term, fit)
    }

    #[test]
    fn test_assign_op_finds_unassigned_and_assigns_earliest() {
        let problem = problem_two_reqs_one_berth();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // State: r10 (idx 0) is unassigned. r20 (idx 1) is assigned at [20, 25].
        // Earliest slot for r10 (pt=10, win=[0,100]) is [0, 10], which is free.
        let mut state = state_r10_unassigned_r20_assigned(&model, &evaluator);

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut op = AssignOp::new();

        // Scope the context so we can drop the immutable borrow of `state` before mutating it.
        let plan = {
            let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);
            // make_next_neighbor should find r10 (index 0) first.
            let plan = op
                .make_next_neighbor(&mut ctx)
                .expect("AssignOp should find a plan for r10");

            // Check plan delta
            assert_eq!(plan.fitness_delta.delta_unassigned, -1);
            assert!(plan.fitness_delta.delta_cost > 0);
            assert_eq!(plan.decision_var_patches.len(), 1);

            plan
        };

        // Apply and verify
        state.apply_plan(plan);
        let r10_idx = model.index_manager().request_index(rid(10)).unwrap();
        let dv10 = state.decision_variables()[r10_idx.get()];
        let asg10 = dv10.as_assigned().expect("r10 should now be assigned");

        assert_eq!(
            asg10.start_time,
            tp(0),
            "should be assigned at earliest slot"
        );

        // Next call should find no more unassigned requests; recreate context with updated state.
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);
        let plan2 = op.make_next_neighbor(&mut ctx);
        assert!(plan2.is_none(), "op should be exhausted (r20 was assigned)");
    }

    #[test]
    fn test_assign_op_respects_window_and_occupancy() {
        let problem = problem_two_reqs_one_berth();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // State: r20 (idx 1) is unassigned. r10 (idx 0) is assigned at [0, 10].
        // r20 has window [15, 100] and pt=5.
        // Earliest slot is [15, 20] (since [0,10] is blocked and window starts at 15).
        let im = model.index_manager();
        let r10 = im.request_index(rid(10)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r10.get()] = DecisionVar::assigned(b1, tp(0)); // r20 (index 1) remains unassigned

        let mut term = TerminalOccupancy::new(model.berths());
        let iv10 = model.interval(r10, b1, tp(0)).unwrap(); // [0, 10]
        term.occupy(b1, iv10).unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let mut state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut op = AssignOp::new();

        // First call checks r10 (idx 0), which is assigned, so it's skipped.
        // Second call (internal loop) checks r20 (idx 1), which is unassigned.
        let plan = {
            let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);
            op.make_next_neighbor(&mut ctx)
                .expect("AssignOp should find a plan for r20")
        };

        // Apply and verify
        state.apply_plan(plan);
        let r20_idx = model.index_manager().request_index(rid(20)).unwrap();
        let dv20 = state.decision_variables()[r20_idx.get()];
        let asg20 = dv20.as_assigned().expect("r20 should now be assigned");

        assert_eq!(
            asg20.start_time,
            tp(15),
            "should be assigned at earliest slot respecting occupancy and window"
        );

        // Next call should be exhausted; recreate context with updated state.
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);
        assert!(op.make_next_neighbor(&mut ctx).is_none());
    }

    #[test]
    fn test_assign_op_yields_none_if_all_assigned() {
        let problem = problem_two_reqs_one_berth();
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // State: Both assigned
        let im = model.index_manager();
        let r10 = im.request_index(rid(10)).unwrap();
        let r20 = im.request_index(rid(20)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r10.get()] = DecisionVar::assigned(b1, tp(0));
        dvars[r20.get()] = DecisionVar::assigned(b1, tp(20));

        let mut term = TerminalOccupancy::new(model.berths());
        term.occupy(b1, model.interval(r10, b1, tp(0)).unwrap())
            .unwrap();
        term.occupy(b1, model.interval(r20, b1, tp(20)).unwrap())
            .unwrap();

        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fit);

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext::new(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = AssignOp::new();
        assert!(
            op.make_next_neighbor(&mut ctx).is_none(),
            "should yield nothing if all requests are assigned"
        );
    }
}
