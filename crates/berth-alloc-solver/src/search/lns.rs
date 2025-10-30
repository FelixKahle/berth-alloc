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
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

#[derive(Debug, PartialEq)]
pub struct RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
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

impl<'b, 'r, 'c, 's, 'm, 'p, T, C, R> RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuinOutcome<'p, T>
where
    T: Copy + Ord,
{
    pub ruined_plan: Plan<'p, T>,
    pub ruined: Vec<RequestIndex>,
}

impl<'p, T> RuinOutcome<'p, T>
where
    T: Copy + Ord,
{
    #[inline]
    pub fn new(ruined_plan: Plan<'p, T>, ruined: Vec<RequestIndex>) -> Self {
        Self {
            ruined_plan,
            ruined,
        }
    }
}

pub trait RuinProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T>;
}

impl<T, C, R> std::fmt::Debug for dyn RuinProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RuinProcedure({})", self.name())
    }
}

impl<T, C, R> std::fmt::Display for dyn RuinProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RuinProcedure({})", self.name())
    }
}

#[derive(Debug, PartialEq)]
pub struct RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
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

impl<'b, 'r, 'c, 's, 'm, 'p, T, C, R> RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
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
    pub fn with_builder<F>(&mut self, plan: Plan<'p, T>, f: F) -> Plan<'p, T>
    where
        F: FnOnce(&mut PlanBuilder<'_, 'c, 's, 'm, 'p, T, C>),
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        self.buffer.copy_from_slice(self.state.decision_variables());

        let mut pb = PlanBuilder::from_plan(
            self.model,
            plan,
            self.state.terminal_occupancy(),
            self.evaluator,
            self.buffer,
        )
        .expect("Failed to create PlanBuilder from given plan");

        f(&mut pb);
        pb.finalize()
    }

    #[inline]
    pub fn builder(&mut self, plan: Plan<'p, T>) -> PlanBuilder<'_, 'c, 's, 'm, 'p, T, C>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        self.buffer.copy_from_slice(self.state.decision_variables());

        PlanBuilder::from_plan(
            self.model,
            plan,
            self.state.terminal_occupancy(),
            self.evaluator,
            self.buffer,
        )
        .expect("Failed to create PlanBuilder from given plan")
    }
}

pub trait RepairProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn repair<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ruined_outcome: RuinOutcome<'p, T>,
    ) -> Plan<'p, T>;
}

impl<T, C, R> std::fmt::Debug for dyn RepairProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RepairProcedure({})", self.name())
    }
}

impl<T, C, R> std::fmt::Display for dyn RepairProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RepairProcedure({})", self.name())
    }
}

#[derive(Debug, PartialEq)]
pub struct PerturbationProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
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

impl<'b, 'r, 'c, 's, 'm, 'p, T, C, R> PerturbationProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
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
}

pub trait PerturbationProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn perturb<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut PerturbationProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Plan<'p, T>;
}

impl<T, C, R> std::fmt::Debug for dyn PerturbationProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PerturbationProcedure({})", self.name())
    }
}

impl<T, C, R> std::fmt::Display for dyn PerturbationProcedure<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PerturbationProcedure({})", self.name())
    }
}

pub struct RandomRuinRepairPerturbPair<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    ruin: Vec<Box<dyn RuinProcedure<T, C, R>>>,
    repair: Vec<Box<dyn RepairProcedure<T, C, R>>>,
}

impl<T, C, R> RandomRuinRepairPerturbPair<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        ruin: Vec<Box<dyn RuinProcedure<T, C, R>>>,
        repair: Vec<Box<dyn RepairProcedure<T, C, R>>>,
    ) -> Self {
        Self { ruin, repair }
    }

    #[inline]
    pub fn random_ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T> {
        let idx = context.rng().random_range(0..self.ruin.len());
        self.ruin[idx].ruin(context)
    }

    #[inline]
    pub fn random_repair<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ruined_outcome: RuinOutcome<'p, T>,
    ) -> Plan<'p, T> {
        let idx = context.rng().random_range(0..self.repair.len());
        self.repair[idx].repair(context, ruined_outcome)
    }
}

impl<T, C, R> PerturbationProcedure<T, C, R> for RandomRuinRepairPerturbPair<T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomRuinRepairPerturbPair"
    }

    fn perturb<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut PerturbationProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Plan<'p, T> {
        let ruined_outcome = {
            let mut ruin_ctx = RuinProcedureContext::new(
                context.model,
                context.state,
                context.evaluator,
                context.rng,
                context.buffer,
            );
            self.random_ruin(&mut ruin_ctx)
        };

        let mut repair_ctx = RepairProcedureContext::new(
            context.model,
            context.state,
            context.evaluator,
            context.rng,
            context.buffer,
        );
        self.random_repair(&mut repair_ctx, ruined_outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::{CostEvaluator, DefaultCostEvaluator},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            terminal::{
                delta::TerminalDelta,
                terminalocc::{TerminalOccupancy, TerminalRead},
            },
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{AssignmentContainer, Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{asg::Assignment, req::Request},
    };
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use std::collections::BTreeMap;

    type T = i64;

    #[inline]
    fn tp(v: i64) -> TimePoint<T> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<T> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<T> {
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

    fn problem_one_berth_two_flex() -> Problem<T> {
        // One berth with broad window
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);
        // Two flexible requests, both allowed on berth 1
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, T>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(5));
        let r2 = Request::<FlexibleKind, T>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(b1);

        let fixed = AssignmentContainer::<_, T, Assignment<_, T>>::new();
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<T, Request<FlexibleKind, T>>::new();
        flex.insert(r1);
        flex.insert(r2);

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn make_state(
        problem: &Problem<T>,
    ) -> (SolverModel<'_, T>, SolverState<'_, T>, DefaultCostEvaluator) {
        let model = SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let eval = DefaultCostEvaluator;
        // Fitness for state here is initial arbitrary; contexts only copy DV and sandbox
        let fitness = Fitness::new(0, model.flexible_requests_len());
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    struct DummyRuin;
    impl<T, C, R> RuinProcedure<T, C, R> for DummyRuin
    where
        T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "DummyRuin"
        }

        fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ) -> RuinOutcome<'p, T> {
            // Emit an empty plan and empty ruined list
            ctx.with_builder(|_pb| { /* no-op */ });
            RuinOutcome::new(
                Plan::new_delta(
                    Vec::new(),
                    TerminalDelta::empty(),
                    crate::state::fitness::FitnessDelta::zero(),
                ),
                Vec::new(),
            )
        }
    }

    struct DummyRepair;
    impl<T, C, R> RepairProcedure<T, C, R> for DummyRepair
    where
        T: Copy + Ord + std::fmt::Debug,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "DummyRepair"
        }
        fn repair<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
            ruined_outcome: RuinOutcome<'p, T>,
        ) -> Plan<'p, T> {
            ruined_outcome.ruined_plan
        }
    }

    struct DummyPerturb;
    impl<T, C, R> PerturbationProcedure<T, C, R> for DummyPerturb
    where
        T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "DummyPerturb"
        }
        fn perturb<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut PerturbationProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ) -> Plan<'p, T> {
            Plan::empty()
        }
    }

    #[test]
    fn test_ruin_context_with_builder_produces_plan_and_keeps_state_immutable() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(1);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut ctx = RuinProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);

        // Build a plan: assign request 1 at time 0
        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let free = ctx
            .state()
            .terminal_occupancy()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .expect("free interval exists");

        let plan = ctx.with_builder(|pb| {
            pb.propose_assignment(r_ix, tp(0), &free)
                .expect("assign ok");
        });

        // Plan has 1 patch, non-empty terminal delta, and fitness delta cost > 0
        assert_eq!(plan.decision_var_patches.len(), 1);
        assert!(!plan.terminal_delta.is_empty());
        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -1);

        // Original state remains unchanged (all Unassigned in initial build)
        for dv in state.decision_variables().iter() {
            assert!(matches!(dv, DecisionVar::Unassigned));
        }
    }

    #[test]
    fn test_ruin_context_builder_roundtrip_assignment_unassignment_yields_zero_delta() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(2);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut ctx = RuinProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);

        let r_ix = model.index_manager().request_index(rid(2)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let free = ctx
            .state()
            .terminal_occupancy()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();

        let mut pb = ctx.builder();
        pb.propose_assignment(r_ix, tp(0), &free)
            .expect("assign ok");
        let fb = pb.propose_unassignment(r_ix).expect("unassign ok");

        // Unassignment returns the same interval/berth
        let pt = model.processing_time(r_ix, b_ix).unwrap();
        assert_eq!(fb.interval(), iv(0, pt.value()));
        assert_eq!(fb.berth_index(), b_ix);

        let plan = pb.finalize();
        // Net zero when we assign then unassign
        assert_eq!(plan.fitness_delta.delta_cost, 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, 0);
    }

    #[test]
    fn test_repair_context_with_builder_from_plan_starts_from_baseline() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(3);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Build a baseline plan that assigns r1 at t=0
        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let r2 = model.index_manager().request_index(rid(2)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();
        let free_r1 = state
            .terminal_occupancy()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();

        // Use a builder to create the baseline plan
        let mut baseline_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut priming_ctx =
            RuinProcedureContext::new(&model, &state, &eval, &mut rng, &mut baseline_buf);
        let baseline = priming_ctx.with_builder(|pb| {
            pb.propose_assignment(r1, tp(0), &free_r1)
                .expect("assign r1 ok");
        });

        // Now repair context starts from that baseline and assigns r2; the resulting plan must
        // only include the new change (r2), not r1 again.
        let mut repair_ctx =
            RepairProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);
        let plan2 = repair_ctx.with_builder(baseline, |pb| {
            let free_r2 = pb
                .sandbox()
                .inner()
                .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r2))
                .next()
                .expect("free for r2 exists");

            assert!(!free_r2.interval().contains(tp(0)));

            let start2 = free_r2.interval().start();
            pb.propose_assignment(r2, start2, &free_r2)
                .expect("assign r2 ok");
        });

        // Expect exactly one patch (for r2), and non-zero fitness delta; r1 not present
        assert_eq!(plan2.decision_var_patches.len(), 1);
        assert_eq!(plan2.decision_var_patches[0].index, r2);
        assert!(plan2.fitness_delta.delta_cost > 0);
        assert_eq!(plan2.fitness_delta.delta_unassigned, -1);
    }

    #[test]
    fn test_trait_debug_and_display() {
        let mut ruin = DummyRuin;
        let mut repair = DummyRepair;
        let mut perturb = DummyPerturb;

        let rp: &mut dyn RuinProcedure<T, DefaultCostEvaluator, StdRng> = &mut ruin;
        let ap: &mut dyn RepairProcedure<T, DefaultCostEvaluator, StdRng> = &mut repair;
        let pp: &mut dyn PerturbationProcedure<T, DefaultCostEvaluator, StdRng> = &mut perturb;

        assert_eq!(format!("{:?}", rp), "RuinProcedure(DummyRuin)");
        assert_eq!(format!("{}", rp), "RuinProcedure(DummyRuin)");
        assert_eq!(format!("{:?}", ap), "RepairProcedure(DummyRepair)");
        assert_eq!(format!("{}", ap), "RepairProcedure(DummyRepair)");
        assert_eq!(format!("{:?}", pp), "PerturbationProcedure(DummyPerturb)");
        assert_eq!(format!("{}", pp), "PerturbationProcedure(DummyPerturb)");
    }

    #[test]
    fn test_ruin_outcome_new_sets_fields() {
        let empty: Plan<'_, i64> = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::zero(),
        );
        let ruined = vec![RequestIndex::new(0), RequestIndex::new(1)];
        let ro = RuinOutcome::new(empty.clone(), ruined.clone());
        assert_eq!(ro.ruined_plan, empty);
        assert_eq!(ro.ruined, ruined);
    }

    #[test]
    fn test_context_accessors_and_rng() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(42);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut ruin_ctx = RuinProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);
        assert!(std::ptr::eq(ruin_ctx.model(), &model));
        assert!(std::ptr::eq(ruin_ctx.state(), &state));
        assert!(std::ptr::eq(ruin_ctx.cost_evaluator(), &eval));
        let _sample = ruin_ctx.rng().next_u32();

        let mut repair_ctx =
            RepairProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);
        assert!(std::ptr::eq(repair_ctx.model(), &model));
        assert!(std::ptr::eq(repair_ctx.state(), &state));
        assert!(std::ptr::eq(repair_ctx.cost_evaluator(), &eval));
        let _sample2 = repair_ctx.rng().next_u32();
    }

    // Additional tests for RandomRuinRepairPerturbPair

    struct CountingRuin {
        calls: usize,
    }
    impl CountingRuin {
        fn new() -> Self {
            Self { calls: 0 }
        }
    }
    impl<T, C, R> RuinProcedure<T, C, R> for CountingRuin
    where
        T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "CountingRuin"
        }

        fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ) -> RuinOutcome<'p, T> {
            self.calls += 1;

            // Build a baseline ruin plan that assigns r1 at t=0 (if available).
            let r1 = ctx
                .model()
                .index_manager()
                .request_index(rid(1))
                .expect("r1 index");
            let b1 = ctx
                .model()
                .index_manager()
                .berth_index(bid(1))
                .expect("b1 index");

            let free_r1 = ctx
                .state()
                .terminal_occupancy()
                .iter_free_intervals_for_berths_in([b1], ctx.model().feasible_interval(r1))
                .next()
                .expect("free interval for r1 exists");

            let ruined_plan = ctx.with_builder(|pb| {
                let start = free_r1.interval().start();
                pb.propose_assignment(r1, start, &free_r1)
                    .expect("assign r1 in ruin");
            });

            RuinOutcome::new(ruined_plan, Vec::new())
        }
    }

    struct CountingRepair {
        calls: usize,
    }
    impl CountingRepair {
        fn new() -> Self {
            Self { calls: 0 }
        }
    }
    impl<T, C, R> RepairProcedure<T, C, R> for CountingRepair
    where
        T: Copy + Ord + std::fmt::Debug,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "CountingRepair"
        }

        fn repair<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
            ruined_outcome: RuinOutcome<'p, T>,
        ) -> Plan<'p, T> {
            self.calls += 1;
            // Return the baseline ruined plan unchanged
            ruined_outcome.ruined_plan
        }
    }

    #[test]
    fn test_random_ruin_with_singleton_calls_ruin_and_returns_plan() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(1234);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut pair: RandomRuinRepairPerturbPair<T, DefaultCostEvaluator, StdRng> = {
            let ruin: Box<dyn RuinProcedure<T, _, _>> = Box::new(CountingRuin::new());
            let repair: Box<dyn RepairProcedure<T, _, _>> = Box::new(CountingRepair::new());
            RandomRuinRepairPerturbPair::new(vec![ruin], vec![repair])
        };

        let mut ruin_ctx = RuinProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);
        let outcome = pair.random_ruin(&mut ruin_ctx);

        // Expect a non-empty plan (one assignment patch), and empty ruined list.
        assert_eq!(outcome.ruined.len(), 0);
        assert_eq!(outcome.ruined_plan.decision_var_patches.len(), 1);
        assert!(!outcome.ruined_plan.terminal_delta.is_empty());
        assert_eq!(outcome.ruined_plan.fitness_delta.delta_unassigned, -1);
    }

    #[test]
    fn test_random_repair_with_singleton_calls_repair_and_returns_input_baseline() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(5678);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Build a baseline plan (assign r2 at earliest feasible time).
        let r2 = model.index_manager().request_index(rid(2)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();
        let free_r2 = state
            .terminal_occupancy()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r2))
            .next()
            .unwrap();

        let mut priming_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut priming_ctx =
            RuinProcedureContext::new(&model, &state, &eval, &mut rng, &mut priming_buf);
        let baseline = priming_ctx.with_builder(|pb| {
            pb.propose_assignment(r2, tp(0), &free_r2)
                .expect("assign r2 ok");
        });

        let mut pair: RandomRuinRepairPerturbPair<T, DefaultCostEvaluator, StdRng> = {
            let ruin: Box<dyn RuinProcedure<T, _, _>> = Box::new(CountingRuin::new());
            let repair: Box<dyn RepairProcedure<T, _, _>> = Box::new(CountingRepair::new());
            RandomRuinRepairPerturbPair::new(vec![ruin], vec![repair])
        };

        let mut repair_ctx =
            RepairProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);
        let plan = pair.random_repair(
            &mut repair_ctx,
            RuinOutcome::new(baseline.clone(), Vec::new()),
        );

        // CountingRepair returns the baseline unchanged.
        assert_eq!(plan, baseline);
    }

    #[test]
    fn test_perturb_singleton_invokes_both_and_returns_repair_result() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(9999);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Singleton ruin + repair
        let mut pair: RandomRuinRepairPerturbPair<T, DefaultCostEvaluator, StdRng> = {
            let ruin: Box<dyn RuinProcedure<T, _, _>> = Box::new(CountingRuin::new());
            let repair: Box<dyn RepairProcedure<T, _, _>> = Box::new(CountingRepair::new());
            RandomRuinRepairPerturbPair::new(vec![ruin], vec![repair])
        };

        let mut ctx =
            PerturbationProcedureContext::new(&model, &state, &eval, &mut rng, &mut buffer);
        let plan = pair.perturb(&mut ctx);

        // Because repair returns the ruined plan unchanged, expect a plan with exactly one patch.
        assert_eq!(plan.decision_var_patches.len(), 1);
        assert!(!plan.terminal_delta.is_empty());
        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -1);
    }

    #[test]
    fn test_pair_name() {
        let pair: RandomRuinRepairPerturbPair<T, DefaultCostEvaluator, StdRng> =
            RandomRuinRepairPerturbPair::new(
                vec![Box::new(CountingRuin::new())],
                vec![Box::new(CountingRepair::new())],
            );

        let pp: &dyn PerturbationProcedure<T, DefaultCostEvaluator, StdRng> = &pair;
        assert_eq!(pp.name(), "RandomRuinRepairPerturbPair");
    }
}
