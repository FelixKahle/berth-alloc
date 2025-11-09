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
    engine::shared_incumbent::SharedIncumbent,
    model::solver_model::SolverModel,
    monitor::search_monitor::SearchMonitor,
    search::{
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
    },
    state::{decisionvar::DecisionVar, solver_state::SolverState},
};
use num_traits::{CheckedAdd, CheckedSub};

pub struct StrategyContext<'e, 'm, 'p, T: Copy + Ord> {
    model: &'m SolverModel<'p, T>,
    shared_incumbent: &'e SharedIncumbent<'p, T>,
    monitor: &'e mut dyn SearchMonitor<T>,
    state: &'e SolverState<'p, T>,
}

impl<'e, 'm, 'p, T: Copy + Ord> StrategyContext<'e, 'm, 'p, T> {
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        shared_incumbent: &'e SharedIncumbent<'p, T>,
        monitor: &'e mut dyn SearchMonitor<T>,
        state: &'e SolverState<'p, T>,
    ) -> Self {
        Self {
            model,
            shared_incumbent,
            monitor,
            state,
        }
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn shared_incumbent(&self) -> &'e SharedIncumbent<'p, T> {
        self.shared_incumbent
    }

    #[inline]
    pub fn monitor(&self) -> &dyn SearchMonitor<T> {
        self.monitor
    }

    #[inline]
    pub fn monitor_mut(&mut self) -> &mut dyn SearchMonitor<T> {
        self.monitor
    }

    #[inline]
    pub fn state(&self) -> &'e SolverState<'p, T> {
        self.state
    }
}

pub trait Strategy<T>: Send
where
    T: Copy + Ord,
{
    fn name(&self) -> &str;

    fn run<'p, 'e, 'm>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>>;
}

impl<'a, T> std::fmt::Debug for dyn Strategy<T> + 'a
where
    T: Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Strategy {{ name: {} }}", self.name())
    }
}

impl<'a, T> std::fmt::Display for dyn Strategy<T> + 'a
where
    T: Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub struct ImprovingStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    R: rand::Rng,
{
    evaluator: C,
    rng: R,
    decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    name: String,
}

impl<'n, T, C, R> std::fmt::Debug for ImprovingStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ImprovingStrategy {{ name: {}, decision_builder: {} }}",
            self.name,
            self.decision_builder.name()
        )
    }
}

impl<'n, T, C, R> std::fmt::Display for ImprovingStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<'n, T, C, R> ImprovingStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    #[inline]
    pub fn new(
        evaluator: C,
        rng: R,
        decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    ) -> Self {
        let name = format!("ImprovingStrategy<{}>", decision_builder.name());
        Self {
            evaluator,
            rng,
            decision_builder,
            name,
        }
    }

    #[inline(always)]
    fn allocate_work_buffer(&self, context: &StrategyContext<'_, '_, '_, T>) -> Vec<DecisionVar<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        vec![DecisionVar::unassigned(); context.model().flexible_requests_len()]
    }
}

impl<'n, T, C, R> Strategy<T> for ImprovingStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        &self.name
    }

    #[tracing::instrument(skip(self, context))]
    fn run<'e, 'm, 'p>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>> {
        context.monitor_mut().on_search_start();

        let mut work_buf = self.allocate_work_buffer(context);
        let mut state = context.state().clone();

        loop {
            if context.monitor_mut().should_terminate_search() {
                break;
            }

            let current_fitness = *state.fitness();
            let model = context.model();

            let next_plan = {
                let monitor = context.monitor_mut();
                let mut ctx = SearchContext::new(
                    model,
                    &state,
                    &self.evaluator,
                    &mut self.rng,
                    &mut work_buf,
                    current_fitness,
                    monitor,
                );
                self.decision_builder.next(&mut ctx)
            };

            match next_plan {
                Some(plan) => {
                    state.apply_plan(plan);
                    let _ = context.shared_incumbent().try_update(&state, model);
                }
                None => break,
            }
        }

        context.monitor_mut().on_search_end();
        Some(state)
    }
}

pub trait StrategyFactory<T>: Send + Sync
where
    T: Copy + Ord,
{
    fn make<'m, 'p>(&self, model: &'m SolverModel<'p, T>) -> Box<dyn Strategy<T> + Send + 'm>;
}

pub struct FnStrategyFactory<T, F>(F, std::marker::PhantomData<T>);

impl<T, F> FnStrategyFactory<T, F>
where
    T: Copy + Ord,
    F: for<'m, 'p> Fn(&'m SolverModel<'p, T>) -> Box<dyn Strategy<T> + Send + 'm> + Send + Sync,
{
    #[inline]
    pub fn new(func: F) -> Self {
        Self(func, std::marker::PhantomData)
    }
}

impl<T, F> StrategyFactory<T> for FnStrategyFactory<T, F>
where
    T: Copy + Ord + Send + Sync,
    F: for<'m, 'p> Fn(&'m SolverModel<'p, T>) -> Box<dyn Strategy<T> + Send + 'm> + Send + Sync,
{
    #[inline]
    fn make<'m, 'p>(&self, model: &'m SolverModel<'p, T>) -> Box<dyn Strategy<T> + Send + 'm> {
        (self.0)(model)
    }
}

impl<T, F> StrategyFactory<T> for F
where
    T: Copy + Ord,
    F: for<'m, 'p> Fn(&'m SolverModel<'p, T>) -> Box<dyn Strategy<T> + Send + 'm> + Send + Sync,
{
    #[inline]
    fn make<'m, 'p>(&self, model: &'m SolverModel<'p, T>) -> Box<dyn Strategy<T> + Send + 'm> {
        (self)(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::index::{BerthIndex, RequestIndex};
    use crate::state::{
        fitness::FitnessDelta,
        plan::{DecisionVarPatch, Plan},
        terminal::delta::TerminalDelta,
    };
    use crate::{
        model::solver_model::SolverModel,
        monitor::search_monitor::{
            LifecycleMonitor, NullSearchMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
        },
        search::eval::DefaultCostEvaluator,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
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

    // A tiny problem: one berth, two flexible requests both allowed on the berth.
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

    // Build (model, state, evaluator) with all-unassigned initial state.
    fn make_model_state_eval(
        problem: &Problem<i64>,
    ) -> (
        SolverModel<'_, i64>,
        SolverState<'_, i64>,
        DefaultCostEvaluator,
    ) {
        let model = SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let eval = DefaultCostEvaluator;

        // Start with positive cost to satisfy debug assertion in apply_plan()
        let fitness = Fitness::new(1, model.flexible_requests_len());
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    // A minimal DecisionBuilder that yields N empty plans, then returns None.
    #[derive(Debug)]
    struct DummyBuilder {
        remaining: usize,
        name: &'static str,
        call_count: usize,
        emit_monitor_events: bool,
    }

    impl DummyBuilder {
        fn new(remaining: usize, name: &'static str) -> Self {
            Self {
                remaining,
                name,
                call_count: 0,
                emit_monitor_events: false,
            }
        }

        fn with_monitor(mut self) -> Self {
            self.emit_monitor_events = true;
            self
        }
    }

    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for DummyBuilder {
        fn name(&self) -> &str {
            self.name
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            context: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<crate::state::plan::Plan<'p, i64>> {
            self.call_count = self.call_count.saturating_add(1);

            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;

            // Optionally emit a generated event, which some monitors use to decide termination.
            if self.emit_monitor_events {
                context
                    .monitor
                    .on_plan_generated(&crate::state::plan::Plan::empty());
            }

            // Strategy will apply this (no-op) plan and continue the loop.
            Some(crate::state::plan::Plan::empty())
        }

        fn reset(&mut self) {
            todo!()
        }
    }

    // A simple monitor that counts lifecycle and plan events and can request termination
    // after N generated candidates.
    #[derive(Debug, Default)]
    struct CountingMonitor {
        started: usize,
        ended: usize,
        generated: usize,
        rejected: usize,
        accepted: usize,
        terminate_after: Option<usize>,
        terminated: bool,
        name: &'static str,
    }

    impl CountingMonitor {
        fn new() -> Self {
            Self {
                name: "CountingMonitor",
                ..Default::default()
            }
        }
        fn with_terminate_after(n: usize) -> Self {
            Self {
                name: "CountingMonitor",
                terminate_after: Some(n),
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
        fn on_plan_generated<'p>(&mut self, _plan: &crate::state::plan::Plan<'p, i64>) {
            self.generated = self.generated.saturating_add(1);
            if let Some(limit) = self.terminate_after {
                if self.generated >= limit {
                    self.terminated = true;
                }
            }
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &crate::state::plan::Plan<'p, i64>) {
            self.rejected = self.rejected.saturating_add(1);
        }
        fn on_plan_accepted<'p>(&mut self, _plan: &crate::state::plan::Plan<'p, i64>) {
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
            self.name
        }
    }

    #[test]
    fn test_strategy_name_includes_builder_name() {
        let problem = problem_one_berth_two_flex();
        let (_, _, eval) = make_model_state_eval(&problem);

        let rng = StdRng::seed_from_u64(0);
        let builder = Box::new(DummyBuilder::new(0, "DummyDB"));

        let strategy = ImprovingStrategy::new(eval, rng, builder);
        assert_eq!(strategy.name(), "ImprovingStrategy<DummyDB>");
        // Ensure trait object printing goes through our Strategy::name impl
        let trait_obj: &dyn Strategy<i64> = &strategy;
        assert_eq!(format!("{}", trait_obj), "ImprovingStrategy<DummyDB>");
    }

    #[test]
    fn test_run_calls_builder_until_none_and_invokes_lifecycle() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);

        // Make a shared incumbent with the initial state.
        let shared = SharedIncumbent::new(state.clone());

        // Builder will yield two plans (empty) then stop.
        let builder = Box::new(DummyBuilder::new(2, "Dummy"));
        let rng = StdRng::seed_from_u64(1);
        let mut strategy = ImprovingStrategy::new(eval, rng, builder);

        let mut monitor = CountingMonitor::new();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let res = strategy.run(&mut ctx);
        assert!(
            res.is_some(),
            "strategy returns final state when search ends"
        );

        // Lifecycle calls
        assert_eq!(monitor.started, 1, "search should start exactly once");
        assert_eq!(monitor.ended, 1, "search should end exactly once");
    }

    #[test]
    fn test_run_respects_monitor_termination_after_first_generated() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        // Builder emits a generated event on each plan.
        let builder = Box::new(DummyBuilder::new(10, "Dummy").with_monitor());
        let rng = StdRng::seed_from_u64(2);
        let mut strategy = ImprovingStrategy::new(eval, rng, builder);

        // Monitor terminates after the first generated plan.
        let mut monitor = CountingMonitor::with_terminate_after(1);
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let res = strategy.run(&mut ctx);
        assert!(res.is_some());

        assert_eq!(monitor.started, 1);
        assert_eq!(monitor.ended, 1);
        assert_eq!(monitor.generated, 1, "should generate exactly one plan");
        assert!(monitor.terminated, "monitor should request termination");
    }

    #[test]
    fn test_run_with_null_monitor_is_noop_and_finishes() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let builder = Box::new(DummyBuilder::new(0, "NoneBuilder")); // returns None immediately
        let rng = StdRng::seed_from_u64(3);
        let mut strategy = ImprovingStrategy::new(eval, rng, builder);

        let mut monitor = NullSearchMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let res = strategy.run(&mut ctx);
        assert!(
            res.is_some(),
            "no candidates => immediate end yields final state"
        );

        // For a no-op builder the returned state should equal the input state.
        let final_state = res.unwrap();
        assert_eq!(final_state, state);
    }

    // A DecisionBuilder that emits exactly one improving plan: assigns the first request and
    // decreases unassigned by 1 (cost delta 0).
    #[derive(Debug)]
    struct ImproveOneBuilder {
        name: &'static str,
        emitted: bool,
    }
    impl ImproveOneBuilder {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                emitted: false,
            }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for ImproveOneBuilder {
        fn name(&self) -> &str {
            self.name
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _context: &mut SearchContext<
                'b,
                'sm,
                'c,
                's,
                'm,
                'p,
                i64,
                DefaultCostEvaluator,
                StdRng,
            >,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;

            // Assign request 0 to berth 0 at time 0
            let patch = DecisionVarPatch::new(
                RequestIndex::new(0),
                DecisionVar::assigned(BerthIndex::new(0), tp(0)),
            );

            // Decrease unassigned by 1; keep cost unchanged (0 delta)
            let delta = FitnessDelta::new(0, -1);

            Some(Plan::new_delta(vec![patch], TerminalDelta::empty(), delta))
        }

        fn reset(&mut self) {
            todo!()
        }
    }

    // A monitor that is already terminated before run() starts.
    #[derive(Debug, Default)]
    struct ImmediateTerminateMonitor {
        started: usize,
        ended: usize,
    }
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
        fn on_search_start(&mut self) {
            self.started += 1;
        }
        fn on_search_end(&mut self) {
            self.ended += 1;
        }
    }
    impl SearchMonitor<i64> for ImmediateTerminateMonitor {
        fn name(&self) -> &str {
            "ImmediateTerminateMonitor"
        }
    }

    // A DecisionBuilder that asserts the SearchContext is sane (model pointer + work_buf length).
    #[derive(Debug)]
    struct ContextAssertingBuilder {
        expected_model_addr: usize, // store address as usize so it is Send
        called: bool,
    }
    impl ContextAssertingBuilder {
        fn new<T>(model_ref: &T) -> Self {
            Self {
                expected_model_addr: (model_ref as *const T) as usize,
                called: false,
            }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for ContextAssertingBuilder {
        fn name(&self) -> &str {
            "ContextAssertingBuilder"
        }

        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            context: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            self.called = true;

            // Check model pointer identity (compare addresses)
            let got_addr = (context.model() as *const _) as usize;
            assert_eq!(
                self.expected_model_addr, got_addr,
                "SearchContext model pointer mismatch"
            );

            // Check work buffer length
            assert_eq!(
                context.work_buf().len(),
                context.model().flexible_requests_len(),
                "work_buf length should equal number of flexible requests"
            );

            None
        }

        fn reset(&mut self) {
            todo!()
        }
    }

    #[test]
    fn test_run_updates_shared_incumbent_on_improvement() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let builder = Box::new(ImproveOneBuilder::new("ImproveOne"));
        let rng = StdRng::seed_from_u64(123);
        let mut strategy = ImprovingStrategy::new(eval, rng, builder);

        let mut monitor = NullSearchMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        // Before: unassigned = 2
        let before = shared.peek();
        assert_eq!(before.unassigned_requests, 2);

        let _ = strategy.run(&mut ctx);

        // After one improving plan: unassigned should be 1
        let after = shared.peek();
        assert_eq!(
            after.unassigned_requests, 1,
            "incumbent should reflect improved unassigned"
        );
    }

    #[test]
    fn test_run_skips_builder_when_monitor_already_terminated() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        // If builder is called, it would return a plan and increment version.
        let builder = Box::new(DummyBuilder::new(1, "WouldRun"));
        let rng = StdRng::seed_from_u64(999);
        let mut strategy = ImprovingStrategy::new(eval, rng, builder);

        let mut monitor = ImmediateTerminateMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let _ = strategy.run(&mut ctx);
        assert_eq!(monitor.started, 1, "lifecycle start called");
        assert_eq!(monitor.ended, 1, "lifecycle end called");
    }

    #[test]
    fn test_builder_receives_expected_context() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        // Builder asserts context invariants and returns None (no plan applied).
        let builder = Box::new(ContextAssertingBuilder::new(&model));
        let rng = StdRng::seed_from_u64(42);
        let mut strategy = ImprovingStrategy::new(eval, rng, builder);

        let mut monitor = NullSearchMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);

        let _ = strategy.run(&mut ctx);
    }
}
