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

use std::ops::Mul;

use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};

use crate::{
    model::solver_model::SolverModel,
    search::{eval::CostEvaluator, planner::PlanBuilder},
    state::{
        decisionvar::DecisionVar,
        plan::Plan,
        solver_state::{SolverState, SolverStateView},
    },
};

pub struct OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    model: &'m SolverModel<'p, T>,
    state: &'s SolverState<'p, T>,
    evaluator: &'c C,

    rng: &'r mut R,
    buffer: &'b mut [DecisionVar<T>],
}

impl<'b, 'r, 'c, 's, 'm, 'p, T, C, R> OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub fn state(&self) -> &'s SolverState<'p, T> {
        self.state
    }

    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    pub fn cost_evaluator(&self) -> &'c C {
        self.evaluator
    }

    pub fn rng(&mut self) -> &mut R {
        self.rng
    }

    // Build a plan using a closure to configure the builder.
    #[inline]
    pub fn with_builder<F>(&mut self, f: F) -> Plan<'p, T>
    where
        F: FnOnce(&mut PlanBuilder<'_, 'c, 's, 'm, 'p, T, C>),
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        // Seed the work buffer with the current decision variables.
        self.buffer.copy_from_slice(self.state.decision_variables());

        // For now we do not use specialized overlays. Just the cloned ledger and terminal,
        // so proposals can be made on them independently from the master state.
        let mut pb = PlanBuilder::new(
            self.model,
            self.state.terminal_occupancy(),
            self.evaluator,
            self.buffer,
        );

        f(&mut pb);
        pb.finalize()
    }

    #[inline]
    pub fn builder(&mut self) -> PlanBuilder<'_, 'c, 's, 'm, 'p, T, C>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        // Seed the work buffer with the current decision variables.
        self.buffer.copy_from_slice(self.state.decision_variables());

        PlanBuilder::new(
            self.model,
            self.state.terminal_occupancy(),
            self.evaluator,
            self.buffer,
        )
    }
}

pub trait LocalSearchOperator<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;
    fn reset(&mut self);
    fn has_fragments(&self) -> bool {
        false
    }
    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>>;
}

pub struct NeighborOperatorSession<'o, 'ctx, 'b, 'r, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    context: &'ctx mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    operator: &'o mut dyn LocalSearchOperator<T, C, R>,
}

impl<'o, 'ctx, 'b, 'r, 'c, 's, 'm, 'p, T, C, R>
    NeighborOperatorSession<'o, 'ctx, 'b, 'r, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub fn new(
        context: &'ctx mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        operator: &'o mut dyn LocalSearchOperator<T, C, R>,
    ) -> Self {
        operator.reset();
        Self { context, operator }
    }

    pub fn make_next_neighbor(&mut self) -> Option<Plan<'p, T>> {
        self.operator.make_next_neighbor(self.context)
    }
}

impl<'o, 'ctx, 'b, 'r, 'c, 's, 'm, 'p, T, C, R> Iterator
    for NeighborOperatorSession<'o, 'ctx, 'b, 'r, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    type Item = Plan<'p, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.make_next_neighbor()
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use crate::search::eval::DefaultCostEvaluator;
    use ::static_assertions::{assert_impl_all, assert_obj_safe};
    use rand_chacha::ChaCha8Rng;

    macro_rules! test_integer_types {
        ($($t:ty),* $(,)?) => {
            $(
                assert_obj_safe!(LocalSearchOperator<$t, DefaultCostEvaluator, ChaCha8Rng>);
                assert_impl_all!(dyn LocalSearchOperator<$t, DefaultCostEvaluator, ChaCha8Rng> + Send + Sync: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::{eval::DefaultCostEvaluator, planner::PlanBuilder},
        state::{
            berth::berthocc::BerthRead,
            decisionvar::DecisionVar,
            solver_state::SolverState,
            terminal::terminalocc::{TerminalOccupancy, TerminalRead},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
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
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pts: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_one_berth_one_flex() -> Problem<i64> {
        let b1 = berth(1, 0, 100);
        let r1 = flex_req(10, (0, 100), &[(1, 10)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.build().expect("valid problem")
    }

    // A simple operator that assigns the first unassigned request exactly once.
    struct TestAssignOp {
        yielded: bool,
    }

    impl TestAssignOp {
        fn new() -> Self {
            Self { yielded: false }
        }
    }

    impl LocalSearchOperator<i64, DefaultCostEvaluator, ChaCha8Rng> for TestAssignOp {
        fn name(&self) -> &str {
            "TestAssignOp"
        }

        fn reset(&mut self) {
            self.yielded = false;
        }

        fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            ctx: &mut OperatorContext<
                'b,
                'r,
                'c,
                's,
                'm,
                'p,
                i64,
                DefaultCostEvaluator,
                ChaCha8Rng,
            >,
        ) -> Option<Plan<'p, i64>> {
            if self.yielded {
                return None;
            }

            // Touch RNG before borrowing ctx mutably for the builder
            let _sample: u32 = ctx.rng().random();

            // Build a plan: assign the first unassigned request to the first available berth slot
            let mut pb: PlanBuilder<'_, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator> = ctx.builder();

            // Use the explorer only; do not access ctx while pb is alive
            let some = pb.with_explorer(|ex| {
                let r_ix = ex.iter_unassigned().next()?;

                let allowed = ex.model().allowed_berth_indices(r_ix).to_vec();
                let window = ex.model().feasible_interval(r_ix);

                let free = ex
                    .sandbox()
                    .inner()
                    .iter_free_intervals_for_berths_in_slice(&allowed, window)
                    .next()?;

                let start = free.interval().start();
                Some((r_ix, free, start))
            });

            let (r_ix, free, start) = match some {
                Some(v) => v,
                None => return None,
            };

            pb.propose_assignment(r_ix, start, &free)
                .expect("assign ok");

            let plan = pb.finalize();
            self.yielded = true;
            Some(plan)
        }
    }

    #[test]
    fn test_operator_context_builds_and_applies_plan() {
        // Arrange problem/model/state
        let problem = problem_one_berth_one_flex();
        let model = SolverModel::try_from(&problem).expect("model ok");

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let evaluator = DefaultCostEvaluator;

        let base = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fit0 = evaluator.eval_fitness(&model, &dvars);
        let mut state = SolverState::new(dvars.clone().into(), base, fit0);

        // Prepare operator context with a work buffer
        let mut work = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext {
            model: &model,
            state: &state,
            evaluator: &evaluator,
            rng: &mut rng,
            buffer: work.as_mut_slice(),
        };

        let mut op = TestAssignOp::new();

        // Act
        let plan = op
            .make_next_neighbor(&mut ctx)
            .expect("operator should yield a plan");

        // Assert plan shape
        assert!(plan.delta_cost > 0, "plan must add positive cost");
        assert_eq!(plan.delta_unassigned, -1, "one request assigned");
        assert_eq!(plan.decision_var_patches.len(), 1, "exactly one DV patch");
        assert!(
            !plan.terminal_delta.is_empty(),
            "terminal delta must not be empty"
        );

        // Apply to state and verify effect
        let prev_cost = state.fitness().cost;
        let prev_unassigned = state.fitness().unassigned_requests;
        state.apply_plan(plan);

        assert_eq!(state.fitness().unassigned_requests, prev_unassigned - 1);
        assert!(state.fitness().cost > prev_cost);

        // Terminal reflects occupancy at some valid interval on berth 1
        let r_ix = model.index_manager().request_index(rid(10)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let pt = model.processing_time(r_ix, b_ix).unwrap();
        let start = model.feasible_interval(r_ix).start();
        let iv_assigned = TimeInterval::new(start, start + pt);
        let occ = state.terminal_occupancy().berth(b_ix).unwrap();
        assert!(
            occ.is_occupied(iv_assigned),
            "expected occupied interval after apply"
        );
    }

    #[test]
    fn test_neighbor_operator_session_iterates_once() {
        // Arrange problem/model/state
        let problem = problem_one_berth_one_flex();
        let model = SolverModel::try_from(&problem).expect("model ok");

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let evaluator = DefaultCostEvaluator;

        let base = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fit0 = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(dvars.into(), base, fit0);

        let mut work = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = OperatorContext {
            model: &model,
            state: &state,
            evaluator: &evaluator,
            rng: &mut rng,
            buffer: work.as_mut_slice(),
        };

        let mut op = TestAssignOp::new();
        let mut sess = NeighborOperatorSession::new(&mut ctx, &mut op);

        let first = sess.next();
        let second = sess.next();

        assert!(first.is_some(), "first neighbor should exist");
        assert!(second.is_none(), "operator should yield exactly once");
    }
}
