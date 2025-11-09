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
    core::numeric::SolveNumeric,
    engine::strategy::{Strategy, StrategyContext},
    model::solver_model::SolverModel,
    monitor::{
        search_monitor::{LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck},
        step::{PlanLimitMonitor, StagnationMonitor},
    },
    search::{
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
    },
    state::{
        decisionvar::DecisionVar,
        plan::Plan,
        solver_state::{SolverState, SolverStateView},
    },
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug)]
pub struct IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub model: &'m SolverModel<'p, T>,
    pub solver_state: &'s SolverState<'p, T>,
    pub evaluator: &'e C,
    pub rng: &'r mut R,
}

impl<'e, 'r, 's, 'm, 'p, T, C, R> IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        solver_state: &'s SolverState<'p, T>,
        evaluator: &'e C,
        rng: &'r mut R,
    ) -> Self {
        Self {
            model,
            solver_state,
            evaluator,
            rng,
        }
    }
}

/// Trait for acceptance criteria layered on top of a decision builder.
/// These decide whether a candidate local plan is accepted.
pub trait IlsAcceptanceCriterion<T, C, R>: Send
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        new_state: &SolverState<'p, T>,
    ) -> bool;
}

impl<'a, T, C, R> std::fmt::Debug for dyn IlsAcceptanceCriterion<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IlsAcceptanceCriterion({})", self.name())
    }
}

impl<'a, T, C, R> std::fmt::Display for dyn IlsAcceptanceCriterion<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IlsAcceptanceCriterion({})", self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

impl<T, C, R> Default for GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R> GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, R> IlsAcceptanceCriterion<T, C, R> for GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "GreedyDescentAcceptanceCriterion"
    }

    #[inline]
    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        new_state: &SolverState<'p, T>,
    ) -> bool {
        new_state.fitness() < context.solver_state.fitness()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

impl<T, C, R> Default for AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R> AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, R> IlsAcceptanceCriterion<T, C, R> for AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "AlwaysAcceptAcceptanceCriterion"
    }

    #[inline]
    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        _context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        _new_state: &SolverState<'p, T>,
    ) -> bool {
        true
    }
}

pub struct IteratedLocalSearchConfig<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    pub max_local_stagnation_steps: u64,
    pub max_local_steps: Option<u64>,
    pub acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
    pub improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    pub perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    pub evaluator: C,
    pub rng: R,
}

impl<'n, T, C, R> IteratedLocalSearchConfig<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn new(
        max_local_stagnation_steps: u64,
        max_local_steps: Option<u64>,
        acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
        improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
        perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
        evaluator: C,
        rng: R,
    ) -> Self {
        Self {
            max_local_stagnation_steps,
            max_local_steps,
            acceptance_criterion,
            improving_decision_builder,
            perturbing_decision_builder,
            evaluator,
            rng,
        }
    }
}

pub struct LocalSearchMonitor {
    stagnation_monitor: StagnationMonitor,
    plan_generation_monitor: Option<PlanLimitMonitor>,
}

impl LocalSearchMonitor {
    #[inline(always)]
    pub fn new(stagnation_budget: u64, plan_generation_limit: Option<u64>) -> Self {
        Self {
            stagnation_monitor: StagnationMonitor::new(stagnation_budget),
            plan_generation_monitor: plan_generation_limit.map(PlanLimitMonitor::new),
        }
    }
}

impl TerminationCheck for LocalSearchMonitor {
    fn should_terminate_search(&self) -> bool {
        if self.stagnation_monitor.should_terminate_search() {
            return true;
        }
        if let Some(plan_limit_monitor) = &self.plan_generation_monitor
            && plan_limit_monitor.should_terminate_search()
        {
            return true;
        }
        false
    }
}

impl<T> PlanEventMonitor<T> for LocalSearchMonitor
where
    T: Copy + Ord,
{
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>) {
        self.stagnation_monitor.on_plan_generated(plan);
        if let Some(plan_limit_monitor) = &mut self.plan_generation_monitor {
            plan_limit_monitor.on_plan_generated(plan);
        }
    }

    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>) {
        self.stagnation_monitor.on_plan_rejected(plan);
        if let Some(plan_limit_monitor) = &mut self.plan_generation_monitor {
            plan_limit_monitor.on_plan_rejected(plan);
        }
    }

    fn on_plan_accepted<'p>(&mut self, plan: &Plan<'p, T>) {
        self.stagnation_monitor.on_plan_accepted(plan);
        if let Some(plan_limit_monitor) = &mut self.plan_generation_monitor {
            plan_limit_monitor.on_plan_accepted(plan);
        }
    }
}

impl LifecycleMonitor for LocalSearchMonitor {
    fn on_search_start(&mut self) {
        self.stagnation_monitor.on_search_start();
        if let Some(plan_limit_monitor) = &mut self.plan_generation_monitor {
            plan_limit_monitor.on_search_start();
        }
    }

    fn on_search_end(&mut self) {
        self.stagnation_monitor.on_search_end();
        if let Some(plan_limit_monitor) = &mut self.plan_generation_monitor {
            plan_limit_monitor.on_search_end();
        }
    }
}

impl<T> SearchMonitor<T> for LocalSearchMonitor
where
    T: Copy + Ord,
{
    fn name(&self) -> &str {
        "LocalSearchMonitor"
    }
}

pub struct LocalSearchForwardMonitor<'a, T, M>
where
    T: Copy + Ord,
{
    base_monitor: &'a mut dyn SearchMonitor<T>,
    local_search_monitor: &'a mut M,
}

impl<'a, T, M> LocalSearchForwardMonitor<'a, T, M>
where
    T: Copy + Ord,
{
    #[inline(always)]
    pub fn new(
        base_monitor: &'a mut dyn SearchMonitor<T>,
        local_search_monitor: &'a mut M,
    ) -> Self {
        Self {
            base_monitor,
            local_search_monitor,
        }
    }
}

impl<'a, T, M> TerminationCheck for LocalSearchForwardMonitor<'a, T, M>
where
    T: Copy + Ord,
    M: TerminationCheck,
{
    fn should_terminate_search(&self) -> bool {
        self.base_monitor.should_terminate_search()
            || self.local_search_monitor.should_terminate_search()
    }
}

impl<'a, T, M> PlanEventMonitor<T> for LocalSearchForwardMonitor<'a, T, M>
where
    T: Copy + Ord,
    M: PlanEventMonitor<T>,
{
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>) {
        self.base_monitor.on_plan_generated(plan);
        self.local_search_monitor.on_plan_generated(plan);
    }

    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>) {
        self.base_monitor.on_plan_rejected(plan);
        self.local_search_monitor.on_plan_rejected(plan);
    }

    fn on_plan_accepted<'p>(&mut self, plan: &Plan<'p, T>) {
        self.base_monitor.on_plan_accepted(plan);
        self.local_search_monitor.on_plan_accepted(plan);
    }
}

impl<'a, T, M> LifecycleMonitor for LocalSearchForwardMonitor<'a, T, M>
where
    T: Copy + Ord,
    M: LifecycleMonitor,
{
    fn on_search_start(&mut self) {
        self.base_monitor.on_search_start();
        self.local_search_monitor.on_search_start();
    }

    fn on_search_end(&mut self) {
        self.base_monitor.on_search_end();
        self.local_search_monitor.on_search_end();
    }
}

impl<'a, T, M> SearchMonitor<T> for LocalSearchForwardMonitor<'a, T, M>
where
    T: Copy + Ord,
    M: SearchMonitor<T>,
{
    fn name(&self) -> &str {
        "LocalSearchForwardMonitor"
    }
}

pub struct IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    max_local_stagnation_steps: u64,
    max_local_steps: Option<u64>,
    acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
    improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    evaluator: C,
    rng: R,
}

impl<'n, T, C, R> IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new(config: IteratedLocalSearchConfig<'n, T, C, R>) -> Self {
        Self {
            max_local_stagnation_steps: config.max_local_stagnation_steps,
            max_local_steps: config.max_local_steps,
            acceptance_criterion: config.acceptance_criterion,
            improving_decision_builder: config.improving_decision_builder,
            perturbing_decision_builder: config.perturbing_decision_builder,
            evaluator: config.evaluator,
            rng: config.rng,
        }
    }

    #[inline(always)]
    fn allocate_work_buffer(&self, context: &StrategyContext<'_, '_, '_, T>) -> Vec<DecisionVar<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        vec![DecisionVar::unassigned(); context.model().flexible_requests_len()]
    }

    fn run_improving_phase<'e, 'm, 'p>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
        state: &mut SolverState<'p, T>,
        work_buf: &mut [DecisionVar<T>],
    ) -> u64 {
        let model = context.model();
        let mut local_search_monitor =
            LocalSearchMonitor::new(self.max_local_stagnation_steps, self.max_local_steps);

        let mut applied = 0;

        //self.improving_decision_builder.reset();

        loop {
            if LocalSearchForwardMonitor::new(context.monitor_mut(), &mut local_search_monitor)
                .should_terminate_search()
            {
                break;
            }

            let current_fitness = *state.fitness();

            let next_plan = {
                let mut forward_monitor = LocalSearchForwardMonitor::new(
                    context.monitor_mut(),
                    &mut local_search_monitor,
                );

                let mut ctx = SearchContext::new(
                    model,
                    state,
                    &self.evaluator,
                    &mut self.rng,
                    work_buf,
                    current_fitness,
                    &mut forward_monitor,
                );
                self.improving_decision_builder.next(&mut ctx)
            };

            match next_plan {
                Some(plan) => {
                    state.apply_plan(plan);
                    applied += 1;
                    // Shared incumbent update is optional inside the local phase; keep if desired:
                    let _ = context.shared_incumbent().try_update(state, model);
                }
                None => break,
            }
        }

        applied
    }

    fn run_perturbing_phase<'e, 'm, 'p>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
        state: &mut SolverState<'p, T>,
        work_buf: &mut [DecisionVar<T>],
    ) -> u64 {
        let model = context.model();
        let mut plan_limit_monitor = PlanLimitMonitor::new(1);

        let next_plan = {
            let mut forward_monitor =
                LocalSearchForwardMonitor::new(context.monitor_mut(), &mut plan_limit_monitor);

            let mut ctx = SearchContext::new(
                model,
                state,
                &self.evaluator,
                &mut self.rng,
                work_buf,
                *state.fitness(),
                &mut forward_monitor,
            );
            self.perturbing_decision_builder.next(&mut ctx)
        };

        if let Some(plan) = next_plan {
            state.apply_plan(plan);
            1
        } else {
            0
        }
    }
}

impl<'n, T, C, R> Strategy<T> for IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "IteratedLocalSearchStrategy"
    }

    fn run<'p, 'e, 'm>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>> {
        let mut work_buf = self.allocate_work_buffer(context);
        let mut state = context.state().clone();

        context.monitor_mut().on_search_start();

        // Initial descent to local optimum
        let _ = self.run_improving_phase(context, &mut state, &mut work_buf);

        loop {
            if context.monitor().should_terminate_search() {
                break;
            }

            // Candidate begins as incumbent clone
            let mut candidate = state.clone();

            let perturb_count = self.run_perturbing_phase(context, &mut candidate, &mut work_buf);
            let improve_count = self.run_improving_phase(context, &mut candidate, &mut work_buf);
            let total_progress = perturb_count + improve_count;

            if total_progress == 0 {
                // No change produced by perturb+improve: terminate outer loop
                break;
            }

            // Acceptance only meaningful if candidate changed
            let model = context.model();
            let mut ils_ctx =
                IlsAcceptanceCriterionContext::new(model, &state, &self.evaluator, &mut self.rng);

            if self.acceptance_criterion.accept(&mut ils_ctx, &candidate) {
                state = candidate;
                let _ = context.shared_incumbent().try_update(&state, model);
            }

            if context.monitor().should_terminate_search() {
                break;
            }
        }

        context.monitor_mut().on_search_end();
        Some(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{shared_incumbent::SharedIncumbent, strategy::StrategyContext},
        monitor::search_monitor::{
            CompositeSearchMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
        },
        search::eval::{CostEvaluator, DefaultCostEvaluator},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::{Fitness, FitnessDelta},
            plan::{DecisionVarPatch, Plan},
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{builder::ProblemBuilder, req::Request},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    type T = i64;

    #[inline]
    fn tp(v: T) -> TimePoint<T> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: T, b: T) -> TimeInterval<T> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: T) -> TimeDelta<T> {
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

    fn make_plan_budget_monitor<T: Copy + Ord>(limit: u64) -> CompositeSearchMonitor<T> {
        let mut comp: CompositeSearchMonitor<T> = CompositeSearchMonitor::new();
        comp.add_monitor(Box::new(PlanLimitMonitor::new(limit)));
        comp
    }

    fn problem_one_berth_two_flex() -> Problem<T> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let mut pt1 = std::collections::BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, T>::new(rid(1), iv(0, 100), 1, pt1).unwrap();

        let mut pt2 = std::collections::BTreeMap::new();
        pt2.insert(bid(1), td(20));
        let r2 = Request::<FlexibleKind, T>::new(rid(2), iv(0, 100), 1, pt2).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        builder.build().unwrap()
    }

    fn make_model_state_eval(
        problem: &Problem<T>,
    ) -> (
        crate::model::solver_model::SolverModel<'_, T>,
        SolverState<'_, T>,
        DefaultCostEvaluator,
    ) {
        let model = crate::model::solver_model::SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        // Positive cost (debug assertion safeguard) & max unassigned initially.
        let fitness = Fitness::new(1, model.flexible_requests_len());
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, DefaultCostEvaluator)
    }

    // Monitor controlling outer iterations (like CountingMonitor style).
    struct IterationStopMonitor {
        iterations_left: usize,
        started: bool,
        ended: bool,
        generated: usize,
        accepted: usize,
        rejected: usize,
    }
    impl IterationStopMonitor {
        fn new(iters: usize) -> Self {
            Self {
                iterations_left: iters,
                started: false,
                ended: false,
                generated: 0,
                accepted: 0,
                rejected: 0,
            }
        }
    }
    impl TerminationCheck for IterationStopMonitor {
        fn should_terminate_search(&self) -> bool {
            self.iterations_left == 0
        }
    }
    impl<Tv: Copy + Ord> PlanEventMonitor<Tv> for IterationStopMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, Tv>) {
            self.generated += 1;
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, Tv>) {
            self.rejected += 1;
        }
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, Tv>) {
            self.accepted += 1;
        }
    }
    impl LifecycleMonitor for IterationStopMonitor {
        fn on_search_start(&mut self) {
            self.started = true;
        }
        fn on_search_end(&mut self) {
            self.ended = true;
        }
    }
    impl<Tv: Copy + Ord> SearchMonitor<Tv> for IterationStopMonitor {
        fn name(&self) -> &str {
            "IterationStopMonitor"
        }
    }

    // DecisionBuilder that assigns r1 (improves unassigned by -1) exactly once.
    struct AssignR1Once {
        done: bool,
    }
    impl AssignR1Once {
        fn new() -> Self {
            Self { done: false }
        }
    }
    impl<C: CostEvaluator<T>, R: rand::Rng> DecisionBuilder<T, C, R> for AssignR1Once {
        fn name(&self) -> &str {
            "AssignR1Once"
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            context: &mut crate::search::decision_builder::SearchContext<
                'b,
                'sm,
                'c,
                's,
                'm,
                'p,
                T,
                C,
                R,
            >,
        ) -> Option<Plan<'p, T>> {
            if self.done {
                return None;
            }
            self.done = true;

            let model = context.model;
            let r_ix = model.index_manager().request_index(rid(1)).unwrap();
            let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

            // Fitness delta via evaluator (note: pass &Decision)
            let assign_dv = DecisionVar::assigned(b_ix, tp(0));
            let delta_cost = context
                .evaluator
                .eval_decision(model, r_ix, assign_dv.as_assigned().unwrap())
                .unwrap();

            let fit_delta = FitnessDelta {
                delta_cost,
                delta_unassigned: -1,
            };
            let patch = DecisionVarPatch::new(r_ix, assign_dv);
            let plan = Plan::new_delta(
                vec![patch],
                crate::state::terminal::delta::TerminalDelta::empty(),
                fit_delta,
            );

            // Emit plan events so monitors (PlanLimitMonitor) can terminate the run
            context.monitor.on_plan_generated(&plan);
            context.monitor.on_plan_accepted(&plan);

            Some(plan)
        }

        fn reset(&mut self) {
            todo!()
        }
    }

    // Perturbation builder that unassigns r1 once (worsens unassigned by +1).
    struct UnassignR1Once {
        done: bool,
    }
    impl UnassignR1Once {
        fn new() -> Self {
            Self { done: false }
        }
    }
    impl<C: CostEvaluator<T>, R: rand::Rng> DecisionBuilder<T, C, R> for UnassignR1Once {
        fn name(&self) -> &str {
            "UnassignR1Once"
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            context: &mut crate::search::decision_builder::SearchContext<
                'b,
                'sm,
                'c,
                's,
                'm,
                'p,
                T,
                C,
                R,
            >,
        ) -> Option<Plan<'p, T>> {
            if self.done {
                return None;
            }
            self.done = true;

            let model = context.model;
            let state = context.state;
            let r_ix = model.index_manager().request_index(rid(1)).unwrap();

            match state.decision_variables()[r_ix.get()] {
                DecisionVar::Unassigned => None,
                DecisionVar::Assigned(dec) => {
                    let prev_cost = context.evaluator.eval_decision(model, r_ix, &dec).unwrap();

                    let fit_delta = FitnessDelta {
                        delta_cost: -prev_cost,
                        delta_unassigned: 1,
                    };
                    let patch = DecisionVarPatch::new(r_ix, DecisionVar::unassigned());
                    let plan = Plan::new_delta(
                        vec![patch],
                        crate::state::terminal::delta::TerminalDelta::empty(),
                        fit_delta,
                    );

                    // Emit plan events so monitors (PlanLimitMonitor) can terminate the run
                    context.monitor.on_plan_generated(&plan);
                    context.monitor.on_plan_accepted(&plan);

                    Some(plan)
                }
            }
        }

        fn reset(&mut self) {
            todo!()
        }
    }

    // No-op builder (never returns a plan).
    struct NoopBuilder;
    impl<C: CostEvaluator<T>, R: rand::Rng> DecisionBuilder<T, C, R> for NoopBuilder {
        fn name(&self) -> &str {
            "NoopBuilder"
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut crate::search::decision_builder::SearchContext<
                'b,
                'sm,
                'c,
                's,
                'm,
                'p,
                T,
                C,
                R,
            >,
        ) -> Option<Plan<'p, T>> {
            None
        }

        fn reset(&mut self) {
            todo!()
        }
    }

    fn make_context<'e, 'm, 'p>(
        model: &'m crate::model::solver_model::SolverModel<'p, T>,
        shared: &'e SharedIncumbent<'p, T>,
        monitor: &'e mut dyn SearchMonitor<T>,
        base_state: &'e SolverState<'p, T>,
    ) -> StrategyContext<'e, 'm, 'p, T> {
        StrategyContext::new(model, shared, monitor, base_state)
    }

    #[test]
    fn test_greedy_rejects_worse_candidate() {
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

        let config = IteratedLocalSearchConfig::new(
            10,
            Some(10),
            Box::new(GreedyDescentAcceptanceCriterion::<T, _, _>::new()),
            Box::new(AssignR1Once::new()),
            Box::new(UnassignR1Once::new()),
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        // Initial improve (1) + perturb (1) => budget 2
        let mut comp = make_plan_budget_monitor::<T>(2);
        let mut ctx = make_context(&model, &shared, &mut comp, &base_state);

        let final_state = strategy.run(&mut ctx).unwrap();
        assert_eq!(final_state.fitness().unassigned_requests, 1);
    }

    #[test]
    fn test_always_accept_worse_candidate() {
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);

        let config = IteratedLocalSearchConfig::new(
            10,
            Some(10),
            Box::new(AlwaysAcceptAcceptanceCriterion::<T, _, _>::new()),
            Box::new(AssignR1Once::new()),
            Box::new(UnassignR1Once::new()),
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        // Initial improve (1) + perturb (1) => budget 2
        let mut comp = make_plan_budget_monitor::<T>(2);
        let mut ctx = make_context(&model, &shared, &mut comp, &base_state);

        let final_state = strategy.run(&mut ctx).unwrap();
        assert_eq!(final_state.fitness().unassigned_requests, 2);
    }

    #[test]
    fn test_greedy_accepts_better_candidate() {
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        let config = IteratedLocalSearchConfig::new(
            5,
            Some(5),
            Box::new(GreedyDescentAcceptanceCriterion::<T, _, _>::new()),
            Box::new(AssignR1Once::new()),
            Box::new(NoopBuilder),
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        // Only initial improvement needed => budget 1
        let mut comp = make_plan_budget_monitor::<T>(1);
        let mut ctx = make_context(&model, &shared, &mut comp, &base_state);

        let final_state = strategy.run(&mut ctx).unwrap();
        assert_eq!(final_state.fitness().unassigned_requests, 1);
    }

    #[test]
    fn test_lifecycle_start_end_called() {
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = ChaCha8Rng::seed_from_u64(4);

        let config = IteratedLocalSearchConfig::new(
            1,
            Some(1),
            Box::new(AlwaysAcceptAcceptanceCriterion::<T, _, _>::new()),
            Box::new(NoopBuilder),
            Box::new(NoopBuilder),
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        let mut monitor = IterationStopMonitor::new(0); // terminate immediately after initial improve
        let mut ctx = make_context(&model, &shared, &mut monitor, &base_state);

        let _ = strategy.run(&mut ctx);
        assert!(monitor.started, "lifecycle start");
        assert!(monitor.ended, "lifecycle end");
    }

    #[test]
    fn test_ils_terminates_when_no_progress() {
        // improving = Noop, perturbing = Noop => total_progress == 0 => outer loop breaks
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);

        let config = IteratedLocalSearchConfig::new(
            /*max_local_stagnation_steps*/ 5,
            /*max_local_steps*/ Some(5),
            Box::new(AlwaysAcceptAcceptanceCriterion::<T, _, _>::new()),
            Box::new(NoopBuilder),
            Box::new(NoopBuilder),
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        let mut monitor = crate::monitor::search_monitor::NullSearchMonitor;
        let mut ctx = make_context(&model, &shared, &mut monitor, &base_state);

        let final_state = strategy.run(&mut ctx).unwrap();

        // No progress => state unchanged (both requests unassigned)
        assert_eq!(final_state.fitness().unassigned_requests, 2);
    }

    #[test]
    fn test_ils_stops_after_initial_improvement_when_no_later_progress() {
        // Initial run_improving_phase assigns r1 once; subsequent perturb+improve do nothing => total_progress == 0 => break
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(12);

        let config = IteratedLocalSearchConfig::new(
            5,
            Some(5),
            Box::new(AlwaysAcceptAcceptanceCriterion::<T, _, _>::new()),
            Box::new(AssignR1Once::new()), // initial improvement
            Box::new(NoopBuilder),         // later: no perturbing progress
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        let mut monitor = crate::monitor::search_monitor::NullSearchMonitor;
        let mut ctx = make_context(&model, &shared, &mut monitor, &base_state);

        let final_state = strategy.run(&mut ctx).unwrap();

        // Initial improvement applied; later 0-progress iteration stops outer loop
        assert_eq!(final_state.fitness().unassigned_requests, 1);
    }

    #[test]
    fn test_ils_accepts_when_only_perturb_makes_progress() {
        // improving = Noop, perturbing = AssignOnce => total_progress > 0 => acceptance runs (Greedy accepts)
        let problem = problem_one_berth_two_flex();
        let (model, base_state, eval) = make_model_state_eval(&problem);
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(13);

        let config = IteratedLocalSearchConfig::new(
            5,
            Some(5),
            Box::new(GreedyDescentAcceptanceCriterion::<T, _, _>::new()),
            Box::new(NoopBuilder),         // improving does nothing
            Box::new(AssignR1Once::new()), // perturb assigns r1 once
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let shared = SharedIncumbent::new(base_state.clone());
        let mut monitor = crate::monitor::search_monitor::NullSearchMonitor;
        let mut ctx = make_context(&model, &shared, &mut monitor, &base_state);

        let final_state = strategy.run(&mut ctx).unwrap();

        // Candidate improved by perturb-only; Greedy accepts since fewer unassigned
        assert_eq!(final_state.fitness().unassigned_requests, 1);
    }
}
