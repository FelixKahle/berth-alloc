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
    model::{index::RequestIndex, solver_model::SolverModel},
    search::{eval::CostEvaluator, planner::PlanBuilder},
    state::{
        decisionvar::DecisionVar,
        plan::Plan,
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::{no_op, prelude::Cost};
use num_traits::{CheckedAdd, CheckedSub};
use std::{ops::Mul, sync::Arc};

pub type NeighborFnVec = dyn Fn(RequestIndex) -> Vec<RequestIndex> + Send + Sync;
pub type NeighborFnSlice<'a> = dyn Fn(RequestIndex) -> &'a [RequestIndex] + Send + Sync + 'a;

pub enum NeighborFn<'a> {
    Vec(Arc<NeighborFnVec>),
    Slice(Arc<NeighborFnSlice<'a>>),
}

#[derive(Debug, PartialEq)]
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
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        state: &'s SolverState<'p, T>,
        evaluator: &'c C,
        rng: &'r mut R,
        buffer: &'b mut [DecisionVar<T>],
    ) -> Self {
        Self {
            model,
            state,
            evaluator,
            rng,
            buffer,
        }
    }

    #[inline]
    pub fn state(&self) -> &'s SolverState<'p, T> {
        self.state
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn cost_evaluator(&self) -> &'c C {
        self.evaluator
    }

    #[inline]
    pub fn rng(&mut self) -> &mut R {
        self.rng
    }

    #[inline]
    pub fn with_builder<F>(&mut self, f: F) -> Plan<'p, T>
    where
        F: FnOnce(&mut PlanBuilder<'_, 'c, 's, 'm, 'p, T, C>),
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        self.buffer.copy_from_slice(self.state.decision_variables());

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
        self.buffer.copy_from_slice(self.state.decision_variables());

        PlanBuilder::new(
            self.model,
            self.state.terminal_occupancy(),
            self.evaluator,
            self.buffer,
        )
    }
}

/// Local search operator interface for the berth-alloc solver.
///
/// An operator defines a neighborhood over the current solution by
/// proposing small, incremental changes. Each proposal is returned as a
/// `Plan`, which encapsulates:
/// - decision variable patches (assignment edits),
/// - a `TerminalDelta` with occupancy updates,
/// - and the incremental fitness deltas.
///
/// Lifecycle:
/// - Create or obtain an `OperatorContext` from the search loop. The context
///   grants read access to the `SolverState`, the `SolverModel`, the
///   `CostEvaluator`, a fast RNG, and a scratch buffer for decision variables.
/// - Call `make_next_neighbor()` repeatedly to iterate candidate moves. Each
///   call may return `Some(Plan)` or `None` when the operator has exhausted
///   its neighborhood for the current synchronization point.
/// - The search loop applies/accepts plans as it sees fit. The operator can be
///   reset via `reset()` when the global state changes substantially, so it can
///   restart its neighborhood enumeration.
///
/// Notes:
/// - Operators typically build plans through `OperatorContext::builder()` or
///   `with_builder()`. This yields a fresh `PlanBuilder` seeded from the current
///   state and using a writable scratch buffer, avoiding allocations.
/// - `PlanBuilder` provides a `TerminalSandbox` internally, so occupancy edits
///   are isolated and emitted as a minimal `TerminalDelta`.
/// - `has_fragments()` can be overridden to signal that an operator yields
///   dependent fragments (useful for composite or chained neighborhoods).
pub trait LocalSearchOperator<T, C, R>: Send
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    /// Returns a human-readable, stable name for this operator.
    ///
    /// This is used in logs, metrics, and benchmarks to identify the
    /// neighborhood strategy (e.g., "SwapAdjacent", "ReassignGreedy").
    fn name(&self) -> &str;

    /// Resets the operator's internal iteration state.
    ///
    /// Call this when the global solver state changes in a way that invalidates
    /// the current neighborhood enumeration (e.g., after applying a plan).
    /// Implementations must not mutate the global solver state; they should
    /// only clear internal cursors/caches so that the next call to
    /// `make_next_neighbor()` starts enumerating from the current `SolverState`.
    fn reset(&mut self);

    /// Synchronize the operator's internal state with the current search context.
    ///
    /// Default implementation is a no-op.
    fn synchronize<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) {
        no_op!()
    }

    /// Indicates whether the operator yields related "fragments" of a move.
    ///
    /// Return `true` if successive calls to `make_next_neighbor()` are expected
    /// to produce a sequence of logically related neighbors (e.g., a chained or
    /// composite neighborhood), which can help the search orchestration decide
    /// how to consume candidates. Return `false` for independent neighbors.
    fn has_fragments(&self) -> bool;

    /// Produces the next neighbor as a `Plan`, or `None` if the neighborhood
    /// is exhausted for the current synchronization point.
    ///
    /// The returned `Plan` encapsulates decision-variable patches and a
    /// `TerminalDelta` describing occupancy changes; its fitness deltas are
    /// computed by the builder/evaluator.
    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>>;
}

impl<T, C, R> std::fmt::Debug for dyn LocalSearchOperator<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchOperator {{ name: {} }}", self.name())
    }
}

impl<T, C, R> std::fmt::Display for dyn LocalSearchOperator<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
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

        fn has_fragments(&self) -> bool {
            false
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
        assert!(
            plan.fitness_delta.delta_cost > 0,
            "plan must add positive cost"
        );
        assert_eq!(
            plan.fitness_delta.delta_unassigned, -1,
            "one request assigned"
        );
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
}
