// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to do so, subject to the following conditions:
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
    search::{decision_builder::DecisionBuilder, eval::CostEvaluator},
    state::decisionvar::DecisionVar,
    state::solver_state::SolverState,
};

#[derive(Debug, Clone)]
pub struct IteratedLocalSearchConfig {
    local_stagnation_limit: usize,
    max_local_steps: Option<usize>, // If none, run until no improvement
    max_pertubation_steps: Option<usize>, // If none, run until success
}

impl IteratedLocalSearchConfig {
    #[inline]
    pub fn new(
        local_stagnation_limit: usize,
        max_local_steps: Option<usize>,
        max_pertubation_steps: Option<usize>,
    ) -> Self {
        Self {
            local_stagnation_limit,
            max_local_steps,
            max_pertubation_steps,
        }
    }

    #[inline]
    pub fn local_stagnation_limit(&self) -> usize {
        self.local_stagnation_limit
    }

    #[inline]
    pub fn max_local_steps(&self) -> Option<usize> {
        self.max_local_steps
    }

    #[inline]
    pub fn max_pertubation_steps(&self) -> Option<usize> {
        self.max_pertubation_steps
    }
}

impl Default for IteratedLocalSearchConfig {
    fn default() -> Self {
        Self {
            local_stagnation_limit: 100,
            max_local_steps: None,
            max_pertubation_steps: Some(100),
        }
    }
}

impl std::fmt::Display for IteratedLocalSearchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IteratedLocalSearchConfig {{ \
            local_stagnation_limit: {},
            max_local_steps: {:?}, \
            max_pertubation_steps: {:?} }}",
            self.local_stagnation_limit, self.max_local_steps, self.max_pertubation_steps
        )
    }
}

#[derive(Debug)]
pub struct IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + 'n>,
    perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + 'n>,
    evaluator: C,
    rng: R,
    name: String,
}

impl<'n, T, C, R> IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + 'n>,
        perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + 'n>,
        evaluator: C,
        rng: R,
    ) -> Self {
        let name = format!(
            "IteratedLocalSearchStrategy<{}, {}>",
            improving_decision_builder.name(),
            perturbing_decision_builder.name()
        );

        Self {
            improving_decision_builder,
            perturbing_decision_builder,
            evaluator,
            rng,
            name,
        }
    }

    #[inline]
    pub fn improving_decision_builder(&self) -> &dyn DecisionBuilder<T, C, R> {
        self.improving_decision_builder.as_ref()
    }

    #[inline]
    pub fn perturbing_decision_builder(&self) -> &dyn DecisionBuilder<T, C, R> {
        self.perturbing_decision_builder.as_ref()
    }

    #[inline]
    pub fn evaluator(&self) -> &C {
        &self.evaluator
    }
}

impl<'n, T, C, R> Strategy<T> for IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn run<'p, 'e, 'm>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>> {
        let cfg = IteratedLocalSearchConfig::default();
        let model = context.model();

        {
            // Borrow monitor only for the lifecycle start.
            let monitor = context.monitor();
            monitor.on_search_start();
        }

        let mut state = context.state().clone();
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut stagnation_counter = 0;

        let is_improvement = |du: i32, dc: i64| du < 0 || (du == 0 && dc < 0);

        'outer: loop {
            // Termination check before local phase.
            if context.monitor().should_terminate_search() {
                break 'outer;
            }

            // -------- Local Improvement Phase --------
            let mut local_steps_phase = 0usize;
            loop {
                if context.monitor().should_terminate_search() {
                    break 'outer;
                }
                if let Some(max_l) = cfg.max_local_steps()
                    && local_steps_phase >= max_l {
                        break;
                    }
                if stagnation_counter >= cfg.local_stagnation_limit() {
                    break;
                }

                let current_fitness = *state.fitness();
                let next_plan_opt = {
                    let monitor = context.monitor();
                    let mut search_ctx = crate::search::decision_builder::SearchContext::new(
                        model,
                        &state,
                        &self.evaluator,
                        &mut self.rng,
                        &mut work_buf,
                        current_fitness,
                        monitor,
                    );
                    self.improving_decision_builder.next(&mut search_ctx)
                };

                let Some(plan) = next_plan_opt else {
                    // No improving neighbor; move to perturbation.
                    break;
                };

                state.apply_plan(plan.clone());

                if is_improvement(
                    plan.fitness_delta.delta_unassigned,
                    plan.fitness_delta.delta_cost,
                ) {
                    stagnation_counter = 0;
                    let _ = context.shared_incumbent().try_update(&state, model);
                } else {
                    stagnation_counter = stagnation_counter.saturating_add(1);
                }

                local_steps_phase = local_steps_phase.saturating_add(1);

                if stagnation_counter >= cfg.local_stagnation_limit() {
                    break;
                }
            }

            if context.monitor().should_terminate_search() {
                break 'outer;
            }

            // -------- Perturbation Phase --------
            let target_perturb_steps = cfg.max_pertubation_steps().unwrap_or(1).max(1);
            let mut performed_perturbation = false;
            let mut perturb_attempts = 0usize;

            while perturb_attempts < target_perturb_steps {
                if context.monitor().should_terminate_search() {
                    break 'outer;
                }

                let current_fitness = *state.fitness();
                let plan_opt = {
                    let monitor = context.monitor();
                    let mut search_ctx = crate::search::decision_builder::SearchContext::new(
                        model,
                        &state,
                        &self.evaluator,
                        &mut self.rng,
                        &mut work_buf,
                        current_fitness,
                        monitor,
                    );
                    self.perturbing_decision_builder.next(&mut search_ctx)
                };

                let Some(plan) = plan_opt else {
                    perturb_attempts = perturb_attempts.saturating_add(1);
                    continue;
                };

                state.apply_plan(plan.clone());
                if is_improvement(
                    plan.fitness_delta.delta_unassigned,
                    plan.fitness_delta.delta_cost,
                ) {
                    let _ = context.shared_incumbent().try_update(&state, model);
                }
                performed_perturbation = true;
                stagnation_counter = 0;
                break;
            }

            if context.monitor().should_terminate_search() {
                break 'outer;
            }

            // Stopping condition: no perturbation and stagnated.
            if !performed_perturbation && stagnation_counter >= cfg.local_stagnation_limit() {
                break 'outer;
            }

            // Another stopping heuristic: no perturbation produced any plan and local produced none.
            if !performed_perturbation && local_steps_phase == 0 {
                break 'outer;
            }
        }

        {
            let monitor = context.monitor();
            monitor.on_search_end();
        }

        Some(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::shared_incumbent::SharedIncumbent,
        model::{
            index::{BerthIndex, RequestIndex},
            solver_model::SolverModel,
        },
        monitor::search_monitor::{
            LifecycleMonitor, NullSearchMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
        },
        search::{decision_builder::SearchContext, eval::DefaultCostEvaluator},
        state::{
            decisionvar::DecisionVar,
            fitness::{Fitness, FitnessDelta},
            plan::{DecisionVarPatch, Plan},
            solver_state::SolverState,
            terminal::{delta::TerminalDelta, terminalocc::TerminalOccupancy},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::builder::ProblemBuilder,
        problem::req::Request,
    };
    use rand::{SeedableRng, rngs::StdRng};
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

    fn problem_one_berth_two_flex() -> Problem<i64> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, i64>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(5));
        let r2 = Request::<FlexibleKind, i64>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        builder.build().expect("valid problem")
    }

    fn make_initial_state(
        problem: &Problem<i64>,
    ) -> (
        SolverModel<'_, i64>,
        SolverState<'_, i64>,
        DefaultCostEvaluator,
    ) {
        let model = SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        // Positive cost to satisfy debug assertion after apply_plan()
        let fitness = Fitness::new(1, model.flexible_requests_len());
        let state = SolverState::new(dvars.into(), term, fitness);
        (model, state, DefaultCostEvaluator)
    }

    // Builder that returns exactly one improving plan (unassigned -1).
    #[derive(Debug)]
    struct SingleImprovingBuilder {
        emitted: bool,
        name: &'static str,
    }
    impl SingleImprovingBuilder {
        fn new(name: &'static str) -> Self {
            Self {
                emitted: false,
                name,
            }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for SingleImprovingBuilder {
        fn name(&self) -> &str {
            self.name
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;

            let patch = DecisionVarPatch::new(
                RequestIndex::new(0),
                DecisionVar::assigned(BerthIndex::new(0), tp(0)),
            );
            let delta = FitnessDelta::new(0, -1);
            let plan = Plan::new_delta(vec![patch], TerminalDelta::empty(), delta);

            // Emit monitor events so tests relying on generated/accepted counters work
            ctx.monitor.on_plan_generated(&plan);
            ctx.monitor.on_plan_accepted(&plan);

            Some(plan)
        }
    }

    // Builder that always returns None (no improvement).
    #[derive(Debug)]
    struct NoOpBuilder(&'static str);
    impl NoOpBuilder {
        fn new(name: &'static str) -> Self {
            Self(name)
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for NoOpBuilder {
        fn name(&self) -> &str {
            self.0
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            None
        }
    }

    // Perturbing builder that produces one improving plan if called.
    #[derive(Debug)]
    struct SinglePerturbBuilder {
        emitted: bool,
        name: &'static str,
    }
    impl SinglePerturbBuilder {
        fn new(name: &'static str) -> Self {
            Self {
                emitted: false,
                name,
            }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for SinglePerturbBuilder {
        fn name(&self) -> &str {
            self.name
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;
            let patch = DecisionVarPatch::new(
                RequestIndex::new(0),
                DecisionVar::assigned(BerthIndex::new(0), tp(0)),
            );
            let delta = FitnessDelta::new(0, -1);
            let plan = Plan::new_delta(vec![patch], TerminalDelta::empty(), delta);

            // Emit monitor events for generated & accepted
            ctx.monitor.on_plan_generated(&plan);
            ctx.monitor.on_plan_accepted(&plan);

            Some(plan)
        }
    }

    // Counting monitor similar to other tests
    #[derive(Debug, Default)]
    struct CountingMonitor {
        started: usize,
        ended: usize,
        generated: usize,
        accepted: usize,
        terminated: bool,
        terminate_after_gen: Option<usize>,
    }
    impl CountingMonitor {
        fn new() -> Self {
            Self {
                ..Default::default()
            }
        }
        fn with_terminate_after_gen(n: usize) -> Self {
            Self {
                terminate_after_gen: Some(n),
                ..Default::default()
            }
        }
    }
    impl TerminationCheck for CountingMonitor {
        fn should_terminate_search(&self) -> bool {
            self.terminated
        }
    }
    impl PlanEventMonitor<i64> for CountingMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.generated = self.generated.saturating_add(1);
            if let Some(limit) = self.terminate_after_gen {
                if self.generated >= limit {
                    self.terminated = true;
                }
            }
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, i64>) {}
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.accepted = self.accepted.saturating_add(1);
        }
    }
    impl LifecycleMonitor for CountingMonitor {
        fn on_search_start(&mut self) {
            self.started = self.started.saturating_add(1);
        }
        fn on_search_end(&mut self) {
            self.ended = self.ended.saturating_add(1);
        }
    }
    impl SearchMonitor<i64> for CountingMonitor {
        fn name(&self) -> &str {
            "CountingMonitor"
        }
    }

    // Immediate terminate monitor (never lets strategy enter loops)
    #[derive(Debug, Default)]
    struct ImmediateTerminateMonitor;
    impl TerminationCheck for ImmediateTerminateMonitor {
        fn should_terminate_search(&self) -> bool {
            true
        }
    }
    impl PlanEventMonitor<i64> for ImmediateTerminateMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, i64>) {}
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, i64>) {}
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, i64>) {}
    }
    impl LifecycleMonitor for ImmediateTerminateMonitor {
        fn on_search_start(&mut self) {}
        fn on_search_end(&mut self) {}
    }
    impl SearchMonitor<i64> for ImmediateTerminateMonitor {
        fn name(&self) -> &str {
            "ImmediateTerminateMonitor"
        }
    }

    #[test]
    fn test_ils_strategy_name_includes_builder_names() {
        let improving = Box::new(NoOpBuilder::new("ImproveB"));
        let perturb = Box::new(NoOpBuilder::new("PerturbB"));
        let rng = StdRng::seed_from_u64(0);
        let strategy: IteratedLocalSearchStrategy<'_, i64, DefaultCostEvaluator, StdRng> =
            IteratedLocalSearchStrategy::new(improving, perturb, DefaultCostEvaluator, rng);

        assert_eq!(
            strategy.name(),
            "IteratedLocalSearchStrategy<ImproveB, PerturbB>"
        );
    }

    #[test]
    fn test_ils_run_applies_improving_plan_and_updates_incumbent() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_initial_state(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let improving = Box::new(SingleImprovingBuilder::new("ImpOne"));
        let perturb = Box::new(NoOpBuilder::new("NoPerturb")); // never used effectively
        let rng = StdRng::seed_from_u64(1);
        let mut strategy = IteratedLocalSearchStrategy::new(improving, perturb, eval, rng);

        let mut monitor = NullSearchMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let before = shared.peek();
        assert_eq!(before.unassigned_requests, model.flexible_requests_len());

        let res = strategy.run(&mut ctx);
        assert!(res.is_some());

        let after = shared.peek();
        assert_eq!(
            after.unassigned_requests,
            model.flexible_requests_len() - 1,
            "incumbent should update after improving plan"
        );
    }

    #[test]
    fn test_ils_run_applies_perturbation_when_no_improvement_available() {
        // Improving builder yields None; perturbing builder yields one improving plan.
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_initial_state(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let improving = Box::new(NoOpBuilder::new("NoImprove"));
        let perturb = Box::new(SinglePerturbBuilder::new("PerturbOnce"));
        let rng = StdRng::seed_from_u64(7);
        let mut strategy = IteratedLocalSearchStrategy::new(improving, perturb, eval, rng);

        let mut monitor = NullSearchMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let before = shared.peek();
        assert_eq!(before.unassigned_requests, model.flexible_requests_len());

        let res = strategy.run(&mut ctx);
        assert!(res.is_some());

        let after = shared.peek();
        assert_eq!(
            after.unassigned_requests,
            model.flexible_requests_len() - 1,
            "perturbation plan should apply when no improvement plans exist"
        );
    }

    #[test]
    fn test_ils_run_respects_termination_before_any_plan() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_initial_state(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let improving = Box::new(SingleImprovingBuilder::new("Imp"));
        let perturb = Box::new(SinglePerturbBuilder::new("Pert"));
        let rng = StdRng::seed_from_u64(11);
        let mut strategy = IteratedLocalSearchStrategy::new(improving, perturb, eval, rng);

        let mut monitor = ImmediateTerminateMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let before = shared.peek();
        let res = strategy.run(&mut ctx).expect("strategy returns a state");
        let after = shared.peek();

        assert_eq!(
            before.unassigned_requests, after.unassigned_requests,
            "incumbent unchanged when monitor terminates immediately"
        );
        assert_eq!(
            res.fitness().unassigned_requests,
            before.unassigned_requests,
            "returned state should match initial when terminated before search loop"
        );
    }

    #[test]
    fn test_ils_monitor_counts_lifecycle_events() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_initial_state(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let improving = Box::new(SingleImprovingBuilder::new("Imp"));
        let perturb = Box::new(NoOpBuilder::new("Pert"));
        let rng = StdRng::seed_from_u64(22);
        let mut strategy = IteratedLocalSearchStrategy::new(improving, perturb, eval, rng);

        let mut monitor = CountingMonitor::new();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);
        let _ = strategy.run(&mut ctx);

        assert_eq!(monitor.started, 1);
        assert_eq!(monitor.ended, 1);
        assert!(
            monitor.generated >= 1,
            "at least one plan should be generated by improving builder"
        );
    }

    #[test]
    fn test_ils_monitor_termination_after_first_generated_plan() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_initial_state(&problem);
        let shared = SharedIncumbent::new(state.clone());

        // Builder emits one improving plan; monitor terminates once generated count reaches 1.
        let improving = Box::new(SingleImprovingBuilder::new("ImpOnce"));
        let perturb = Box::new(SinglePerturbBuilder::new("PertOnce"));
        let rng = StdRng::seed_from_u64(33);
        let mut strategy = IteratedLocalSearchStrategy::new(improving, perturb, eval, rng);

        let mut monitor = CountingMonitor::with_terminate_after_gen(1);
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);
        let _ = strategy.run(&mut ctx);

        assert_eq!(monitor.generated, 1, "termination after first generated");
        assert!(monitor.should_terminate_search(), "monitor terminated");
    }
}
