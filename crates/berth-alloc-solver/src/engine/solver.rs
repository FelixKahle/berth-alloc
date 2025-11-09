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
        strategy::{Strategy, StrategyFactory},
        worker::{SolverWorker, WorkerPool},
    },
    model::{err::SolverModelBuildError, solver_model::SolverModel},
    monitor::{
        search_monitor::{CompositeSearchMonitor, SearchMonitor},
        step::StagnationMonitor,
        time::TimeLimitMonitor,
    },
    opening::{greedy::GreedyOpening, opening_strategy::OpeningStrategy},
    state::solver_state::SolverStateView,
};
use berth_alloc_model::{
    prelude::{Problem, SolutionRef},
    solution::SolutionError,
};

#[derive(Copy, Clone, Debug)]
pub struct SolverConfig {
    pub num_workers: usize,
    pub time_limit: std::time::Duration,
    pub stagnation_generated_without_accept: Option<u64>,
}

impl std::fmt::Display for SolverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SolverConfig {{ num_workers: {}, time_limit: {:?}, stagnation_generated_without_accept: {:?} }}",
            self.num_workers, self.time_limit, self.stagnation_generated_without_accept
        )
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            num_workers: 2,
            time_limit: std::time::Duration::from_secs(60),
            stagnation_generated_without_accept: None,
        }
    }
}

#[derive(Debug)]
pub enum EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    SolverModel(SolverModelBuildError),
    Solution(SolutionError),
    NoStrategies,
    OpeningFailed(S::Error),
}

impl<T, S> From<SolutionError> for EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    #[inline]
    fn from(err: SolutionError) -> Self {
        EngineError::Solution(err)
    }
}

impl<T, S> std::fmt::Display for EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Solution(err) => write!(f, "Solution error: {}", err),
            EngineError::NoStrategies => write!(f, "No search strategies provided"),
            EngineError::OpeningFailed(err) => write!(f, "Opening strategy failed: {:?}", err),
            EngineError::SolverModel(err) => write!(f, "Solver model build error: {}", err),
        }
    }
}

impl<T, S> std::error::Error for EngineError<T, S>
where
    T: Copy + Ord + std::fmt::Debug + std::fmt::Display,
    S: OpeningStrategy<T> + std::fmt::Debug,
{
}

impl<T, S> From<SolverModelBuildError> for EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    #[inline]
    fn from(err: SolverModelBuildError) -> Self {
        EngineError::SolverModel(err)
    }
}

pub struct Solver<T>
where
    T: Copy + Ord,
{
    config: SolverConfig,
    strategy_factories: Vec<Box<dyn StrategyFactory<T> + Send>>,
}

impl<T> Default for Solver<T>
where
    T: SolveNumeric,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Solver<T>
where
    T: SolveNumeric,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            config: SolverConfig::default(),
            strategy_factories: Vec::new(),
        }
    }

    #[inline]
    pub fn with_config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self
    }

    #[inline]
    pub fn with_strategy(mut self, strategy: Box<dyn StrategyFactory<T> + Send>) -> Self {
        self.strategy_factories.push(strategy);
        self
    }

    #[inline]
    pub fn with_strategy_fn<F>(mut self, f: F) -> Self
    where
        F: for<'m, 'pm> Fn(&'m SolverModel<'pm, T>) -> Box<dyn Strategy<T> + Send + 'm>
            + Send
            + Sync
            + 'static,
    {
        self.strategy_factories.push(Box::new(f));
        self
    }

    #[inline]
    pub fn with_strategies(mut self, strategies: Vec<Box<dyn StrategyFactory<T> + Send>>) -> Self {
        self.strategy_factories.extend(strategies);
        self
    }

    #[inline]
    pub fn with_time_limit(mut self, time_limit: std::time::Duration) -> Self {
        self.config.time_limit = time_limit;
        self
    }

    #[inline]
    pub fn with_stagnation_budget(mut self, budget_without_accept: u64) -> Self {
        self.config.stagnation_generated_without_accept = Some(budget_without_accept);
        self
    }

    fn build_engine_monitor(&self) -> Box<dyn SearchMonitor<T> + Send> {
        let mut comp: CompositeSearchMonitor<T> = CompositeSearchMonitor::new();
        comp.add_monitor(Box::new(TimeLimitMonitor::new(self.config.time_limit)));
        if let Some(budget) = self.config.stagnation_generated_without_accept {
            comp.add_monitor(Box::new(StagnationMonitor::new(budget)));
        }
        Box::new(comp)
    }

    pub fn solve<'p>(
        &mut self,
        problem: &'p Problem<T>,
    ) -> Result<Option<SolutionRef<'p, T>>, EngineError<T, GreedyOpening<T>>>
    where
        T: std::fmt::Display + std::fmt::Debug,
    {
        let model: SolverModel<'p, T> = SolverModel::try_from(problem)?;
        let mut strategies: Vec<Box<dyn Strategy<T> + Send>> =
            Vec::with_capacity(self.strategy_factories.len());
        for factory in &self.strategy_factories {
            strategies.push(factory.make(&model));
        }

        if strategies.is_empty() {
            return Err(EngineError::NoStrategies);
        }

        let opening_strategy = GreedyOpening::new();
        let initial_state = opening_strategy
            .build(&model)
            .map_err(EngineError::OpeningFailed)?;

        // Install the state from the opening strategy as initial incumbent
        let incumbent = SharedIncumbent::new(initial_state.clone());

        let mut workers = Vec::with_capacity(strategies.len());
        for (i, strat) in strategies.into_iter().enumerate() {
            let mon = self.build_engine_monitor();
            let worker = SolverWorker::new(i, &model, &incumbent, strat, mon);
            workers.push(worker);
        }

        let pool = WorkerPool::new(workers);
        pool.run_scoped(&initial_state);

        let best_state = incumbent.snapshot();
        if !best_state.is_feasible() {
            return Ok(None);
        }

        match best_state.into_solution(&model) {
            Ok(sol) => Ok(Some(sol)),
            Err(err) => Err(EngineError::from(err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::strategy::FnStrategyFactory;
    use crate::model::index::BerthIndex;
    use crate::search::eval::CostEvaluator;
    use crate::{
        engine::strategy::StrategyContext,
        state::{decisionvar::DecisionVar, solver_state::SolverState},
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier, SolutionView},
        problem::{builder::ProblemBuilder, req::Request},
    };
    use std::collections::BTreeMap;
    use std::time::Duration;

    // Helpers
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

    fn make_problem() -> Problem<i64> {
        // One berth, two flex requests allowed on berth 1
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

    // Assign both requests; produce a feasible incumbent with consistent fitness.
    #[derive(Debug)]
    struct AssignAll;
    impl Strategy<i64> for AssignAll {
        fn name(&self) -> &str {
            "AssignAll"
        }
        fn run<'e, 'm, 'p>(
            &mut self,
            ctx: &mut StrategyContext<'e, 'm, 'p, i64>,
        ) -> Option<SolverState<'p, i64>> {
            use crate::search::eval::DefaultCostEvaluator;
            use crate::state::decisionvar::DecisionVarVec;
            use crate::state::terminal::terminalocc::TerminalOccupancy;

            ctx.monitor_mut().on_search_start();

            let model = ctx.model();

            // Build decision variables with full assignment:
            // R0 -> B0 @ t=0, R1 -> B0 @ t=20 (non-overlapping on the single berth)
            let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
            dvars[0] = DecisionVar::assigned(BerthIndex::new(0), tp(0));
            dvars[1] = DecisionVar::assigned(BerthIndex::new(0), tp(20));

            // Compute true fitness to keep it consistent with SolutionRef recomputation
            let fitness = DefaultCostEvaluator.eval_fitness(model, &dvars);

            // Terminal occupancy can be constructed from the base problem; it does not affect SolutionRef cost checks.
            let term = TerminalOccupancy::new(model.problem().berths().iter());

            let candidate = SolverState::new(DecisionVarVec::from(dvars), term, fitness);

            let _ = ctx.shared_incumbent().try_update(&candidate, model);

            ctx.monitor_mut().on_search_end();
            None
        }
    }

    #[test]
    fn test_solve_no_strategies_returns_error() {
        let problem = make_problem();

        let mut solver = super::Solver::new().with_time_limit(Duration::from_millis(20));

        match solver.solve(&problem) {
            Err(super::EngineError::NoStrategies) => {}
            other => panic!("expected NoStrategies, got {:?}", other),
        }
    }

    #[test]
    fn test_solve_full_assignment_returns_solution() {
        let problem = make_problem();

        let mut solver = super::Solver::new()
            .with_time_limit(Duration::from_millis(50))
            .with_stagnation_budget(1000)
            .with_strategy(Box::new(FnStrategyFactory::new(|_model| {
                Box::new(AssignAll)
            })));

        let res = solver.solve(&problem).expect("solver should not error");
        assert!(
            res.is_some(),
            "feasible incumbent should yield Some(SolutionRef)"
        );

        let sol = res.unwrap();
        // Sanity check: cost should be non-negative
        assert!(sol.cost() >= 0);
        // The solution should contain assignments for all flexible requests
        let n_flex = problem.flexible_requests().len();
        let n_sol = sol.flexible_assignments().len();
        assert_eq!(
            n_sol, n_flex,
            "solution should contain all flexible assignments"
        );
    }
}
