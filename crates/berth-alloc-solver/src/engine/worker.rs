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
    engine::{
        shared_incumbent::SharedIncumbent,
        strategy::{Strategy, StrategyContext},
    },
    model::solver_model::SolverModel,
    monitor::search_monitor::SearchMonitor,
};
use std::thread;

pub struct SolverWorker<'e, 'm, 'p, T>
where
    T: Copy + Ord + Send,
{
    id: usize,
    model: &'m SolverModel<'p, T>,
    shared_incumbent: &'e SharedIncumbent<'p, T>,
    strategy: Box<dyn Strategy<'p, T> + Send>,
    monitor: Box<dyn SearchMonitor<T> + Send>,
}

impl<'e, 'm, 'p, T> SolverWorker<'e, 'm, 'p, T>
where
    T: SolveNumeric,
{
    #[inline]
    pub fn new(
        id: usize,
        model: &'m SolverModel<'p, T>,
        shared_incumbent: &'e SharedIncumbent<'p, T>,
        strategy: Box<dyn Strategy<'p, T> + Send>,
        monitor: Box<dyn SearchMonitor<T> + Send>,
    ) -> Self {
        Self {
            id,
            model,
            shared_incumbent,
            strategy,
            monitor,
        }
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn run(mut self) {
        let mut ctx =
            StrategyContext::new(self.model, self.shared_incumbent, self.monitor.as_mut());
        let _ = self.strategy.run(&mut ctx);
    }
}

pub struct WorkerPool<'e, 'm, 'p, T>
where
    T: Copy + Ord + Send + Sync + 'p,
{
    workers: Vec<SolverWorker<'e, 'm, 'p, T>>,
}

impl<'e, 'm, 'p, T> WorkerPool<'e, 'm, 'p, T>
where
    T: SolveNumeric,
{
    #[inline]
    pub fn new(workers: Vec<SolverWorker<'e, 'm, 'p, T>>) -> Self {
        Self { workers }
    }

    #[inline]
    pub fn run_scoped(self) {
        thread::scope(|scope| {
            for w in self.workers {
                scope.spawn(move || w.run());
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::strategy::StrategyContext,
        model::solver_model::SolverModel,
        monitor::search_monitor::{
            LifecycleMonitor, NullSearchMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
        },
        search::eval::{CostEvaluator, DefaultCostEvaluator},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
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
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    // Minimal problem: one berth, two flexible requests on the same berth.
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

    // Build a model and an initial state (all unassigned) and a shared incumbent from that state.
    fn make_model_and_incumbent() -> (SolverModel<'static, i64>, SharedIncumbent<'static, i64>) {
        // Leak the problem to make it 'static for simplicity in tests
        let problem = problem_one_berth_two_flex();
        let problem_static: &'static Problem<i64> = Box::leak(Box::new(problem));

        let model = SolverModel::try_from(problem_static).expect("model ok");

        // Compute initial state fitness using the default evaluator.
        let evaluator = DefaultCostEvaluator;
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let term = TerminalOccupancy::new(problem_static.berths().iter());
        let fitness = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);

        let inc = SharedIncumbent::new(state);
        (model, inc)
    }

    // A simple monitor that counts lifecycle events via atomics.
    #[derive(Debug)]
    struct CountingMonitor {
        started: Arc<AtomicUsize>,
        ended: Arc<AtomicUsize>,
        name: &'static str,
    }

    impl CountingMonitor {
        fn new(started: Arc<AtomicUsize>, ended: Arc<AtomicUsize>) -> Self {
            Self {
                started,
                ended,
                name: "CountingMonitor",
            }
        }
    }

    impl TerminationCheck for CountingMonitor {
        fn should_terminate_search(&self) -> bool {
            false
        }
    }
    impl PlanEventMonitor<i64> for CountingMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &crate::state::plan::Plan<'p, i64>) {}
        fn on_plan_rejected<'p>(&mut self, _plan: &crate::state::plan::Plan<'p, i64>) {}
        fn on_plan_accepted<'p>(&mut self, _plan: &crate::state::plan::Plan<'p, i64>) {}
    }
    impl LifecycleMonitor for CountingMonitor {
        fn on_search_start(&mut self) {
            self.started.fetch_add(1, Ordering::SeqCst);
        }
        fn on_search_end(&mut self) {
            self.ended.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl SearchMonitor<i64> for CountingMonitor {
        fn name(&self) -> &str {
            self.name
        }
    }

    // A dummy strategy that records being run, and calls lifecycle on the monitor.
    #[derive(Debug)]
    struct DummyStrategy {
        ran: Arc<AtomicUsize>,
        name: &'static str,
    }

    impl DummyStrategy {
        fn new(ran: Arc<AtomicUsize>, name: &'static str) -> Self {
            Self { ran, name }
        }
    }

    impl<'p> Strategy<'p, i64> for DummyStrategy {
        fn name(&self) -> &str {
            self.name
        }

        fn run<'e, 'm>(
            &mut self,
            context: &mut StrategyContext<'e, 'm, 'p, i64>,
        ) -> Option<SolverState<'p, i64>> {
            context.monitor().on_search_start();
            self.ran.fetch_add(1, Ordering::SeqCst);
            context.monitor().on_search_end();
            None
        }
    }

    #[test]
    fn test_worker_run_invokes_strategy_and_lifecycle() {
        let (model, incumbent) = make_model_and_incumbent();

        let ran = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(AtomicUsize::new(0));
        let ended = Arc::new(AtomicUsize::new(0));

        let strategy = Box::new(DummyStrategy::new(ran.clone(), "DummyStrategy"));
        let monitor = Box::new(CountingMonitor::new(started.clone(), ended.clone()));

        let worker = super::SolverWorker::new(7, &model, &incumbent, strategy, monitor);
        worker.run();

        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "strategy.run should be called exactly once"
        );
        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "monitor.on_search_start should be called"
        );
        assert_eq!(
            ended.load(Ordering::SeqCst),
            1,
            "monitor.on_search_end should be called"
        );
    }

    #[test]
    fn test_worker_id_returns_configured_id() {
        let (model, incumbent) = make_model_and_incumbent();

        let ran = Arc::new(AtomicUsize::new(0));
        let strategy = Box::new(DummyStrategy::new(ran, "Dummy"));
        let monitor = Box::new(NullSearchMonitor::default());

        let worker = super::SolverWorker::new(42, &model, &incumbent, strategy, monitor);
        assert_eq!(worker.id(), 42, "worker.id() must return the configured id");
        // do not run; this test only checks id
    }

    #[test]
    fn test_search_pool_runs_all_workers() {
        let (model, incumbent) = make_model_and_incumbent();

        // Build several workers, each recording its own run count
        let mut counters = Vec::new();
        let mut workers = Vec::new();

        for i in 0..4usize {
            let ran = Arc::new(AtomicUsize::new(0));
            let started = Arc::new(AtomicUsize::new(0));
            let ended = Arc::new(AtomicUsize::new(0));

            let strategy = Box::new(DummyStrategy::new(ran.clone(), "PoolStrategy"));
            let monitor = Box::new(CountingMonitor::new(started.clone(), ended.clone()));

            let worker = super::SolverWorker::new(i, &model, &incumbent, strategy, monitor);
            workers.push(worker);
            counters.push((ran, started, ended));
        }

        let pool = super::WorkerPool::new(workers);
        pool.run_scoped();

        for (idx, (ran, started, ended)) in counters.into_iter().enumerate() {
            assert_eq!(
                ran.load(Ordering::SeqCst),
                1,
                "worker {} strategy should have run exactly once",
                idx
            );
            assert_eq!(
                started.load(Ordering::SeqCst),
                1,
                "worker {} monitor should have on_search_start exactly once",
                idx
            );
            assert_eq!(
                ended.load(Ordering::SeqCst),
                1,
                "worker {} monitor should have on_search_end exactly once",
                idx
            );
        }
    }
}
