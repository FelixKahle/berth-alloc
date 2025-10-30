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
    engine::{
        shared_incumbent::SharedIncumbent,
        strategy::{SearchContext, SearchStrategy},
    },
    model::solver_model::SolverModel,
};
use berth_alloc_model::prelude::Problem;
use std::thread;

pub struct SearchWorkerContext<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    problem: &'p Problem<T>,
    model: &'m SolverModel<'p, T>,
    incumbent: &'e SharedIncumbent<'p, T>,
    stop: &'e std::sync::atomic::AtomicBool,
    rng: R,
}

impl<'e, 'm, 'p, T, R> SearchWorkerContext<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        problem: &'p Problem<T>,
        model: &'m SolverModel<'p, T>,
        incumbent: &'e SharedIncumbent<'p, T>,
        stop: &'e std::sync::atomic::AtomicBool,
        rng: R,
    ) -> Self {
        Self {
            problem,
            model,
            incumbent,
            stop,
            rng,
        }
    }

    #[inline]
    pub fn rng(&self) -> &R {
        &self.rng
    }

    #[inline]
    pub fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    #[inline]
    pub fn problem(&self) -> &'p Problem<T> {
        self.problem
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn incumbent(&self) -> &'e SharedIncumbent<'p, T> {
        self.incumbent
    }

    #[inline]
    pub fn stop(&self) -> &'e std::sync::atomic::AtomicBool {
        self.stop
    }
}

impl<'e, 'm, 'p, T, R> From<SearchWorkerContext<'e, 'm, 'p, T, R>>
    for SearchContext<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    #[inline]
    fn from(val: SearchWorkerContext<'e, 'm, 'p, T, R>) -> Self {
        SearchContext::new(val.problem, val.model, val.incumbent, val.stop, val.rng)
    }
}

pub struct SearchWorker<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    id: usize,
    context: SearchWorkerContext<'e, 'm, 'p, T, R>,
    strategy: Box<dyn SearchStrategy<T, R> + Send>,
}

impl<'e, 'm, 'p, T, R> SearchWorker<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    pub fn new(
        id: usize,
        strategy: Box<dyn SearchStrategy<T, R> + Send>,
        context: SearchWorkerContext<'e, 'm, 'p, T, R>,
    ) -> Self {
        Self {
            id,
            context,
            strategy,
        }
    }

    #[inline]
    fn run_once(self) {
        let mut ctx = self.context.into();
        let mut strat = self.strategy;
        strat.run(&mut ctx);
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }
}

impl<'e, 'm, 'p, T, R> std::fmt::Debug for SearchWorker<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchWorker")
            .field("id", &self.id)
            .field("problem", &"Problem { .. }")
            .field("model", &"SolverModel { .. }")
            .field("incumbent", &"SharedIncumbent { .. }")
            .field("stop", &self.context.stop)
            .field("strategy", &self.strategy.name())
            .field("rng", &"Rng { .. }")
            .finish()
    }
}

impl<'e, 'm, 'p, T, R> std::fmt::Display for SearchWorker<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchWorker #{}", self.id)
    }
}

pub struct SearchPool<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + 'p,
    R: rand::Rng + Send + 'static,
{
    workers: Vec<SearchWorker<'e, 'm, 'p, T, R>>,
}

impl<'e, 'm, 'p, T, R> SearchPool<'e, 'm, 'p, T, R>
where
    T: Copy + Ord + Send + Sync + 'p,
    R: rand::Rng + Send + 'static,
{
    pub fn new(workers: Vec<SearchWorker<'e, 'm, 'p, T, R>>) -> Self {
        Self { workers }
    }

    pub fn run_scoped(self) {
        thread::scope(|scope| {
            for w in self.workers {
                scope.spawn(move || {
                    w.run_once();
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::strategy::{SearchContext, SearchStrategy},
        model::solver_model::SolverModel,
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
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::{
        collections::BTreeMap,
        fmt::Write as _,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    // ------------- Helpers -------------

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

    fn make_basic_problem() -> Problem<i64> {
        // One berth [0,100)
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        // One flexible request with weight 1 and pt(10) on berth 1
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 1, pt).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.build().unwrap()
    }

    fn make_model_and_incumbent(
        problem: &Problem<i64>,
    ) -> (SolverModel<'_, i64>, SharedIncumbent<'_, i64>) {
        let model = SolverModel::try_from(problem).expect("model should build");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fitness = DefaultCostEvaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        let incumbent = SharedIncumbent::new(state);
        (model, incumbent)
    }

    #[derive(Clone)]
    struct DummyStrategy {
        name: &'static str,
        counter: Arc<AtomicUsize>,
    }

    impl DummyStrategy {
        fn new(name: &'static str) -> (Self, Arc<AtomicUsize>) {
            let counter = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    name,
                    counter: counter.clone(),
                },
                counter,
            )
        }
    }

    impl SearchStrategy<i64, ChaCha8Rng> for DummyStrategy {
        fn name(&self) -> &str {
            self.name
        }

        fn run<'e, 'm, 'p>(&mut self, ctx: &mut SearchContext<'e, 'm, 'p, i64, ChaCha8Rng>) {
            let _ = ctx.problem();
            let _ = ctx.model();
            let _ = ctx.shared_incumbent();
            let _ = ctx.stop();
            let _sample: u32 = ctx.rng().next_u32();

            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_worker_context_accessors_and_into() {
        let problem = make_basic_problem();
        let (model, incumbent) = make_model_and_incumbent(&problem);
        let stop = AtomicBool::new(false);
        let rng = ChaCha8Rng::seed_from_u64(123);

        let mut ctx = SearchWorkerContext::new(&problem, &model, &incumbent, &stop, rng);

        // Accessors
        let _ = ctx.rng(); // existence
        let _ = ctx.rng_mut(); // mutability
        assert!(std::ptr::eq(ctx.problem(), &problem));
        assert!(std::ptr::eq(ctx.model(), &model));
        assert!(std::ptr::eq(ctx.incumbent(), &incumbent));
        assert!(std::ptr::eq(ctx.stop(), &stop));

        // Into SearchContext moves rng but keeps references the same
        let mut scx: SearchContext<'_, '_, '_, i64, ChaCha8Rng> = ctx.into();
        assert!(std::ptr::eq(scx.problem(), &problem));
        assert!(std::ptr::eq(scx.model(), &model));
        assert!(std::ptr::eq(scx.shared_incumbent(), &incumbent));
        assert!(std::ptr::eq(scx.stop(), &stop));

        // RNG works post-move
        let _ = scx.rng().next_u32();
    }

    #[test]
    fn test_search_worker_run_once_calls_strategy() {
        let problem = make_basic_problem();
        let (model, incumbent) = make_model_and_incumbent(&problem);
        let stop = AtomicBool::new(false);
        let rng = ChaCha8Rng::seed_from_u64(1);

        let ctx = SearchWorkerContext::new(&problem, &model, &incumbent, &stop, rng);
        let (strategy, counter) = DummyStrategy::new("Dummy");

        let worker = SearchWorker::new(7, Box::new(strategy), ctx);
        assert_eq!(worker.id(), 7);

        // Debug/Display formatting sanity
        let mut dbg_s = String::new();
        let _ = write!(&mut dbg_s, "{:?}", &worker);
        assert!(dbg_s.contains("SearchWorker"));
        assert!(dbg_s.contains("id"));
        assert!(dbg_s.contains("Dummy"));

        let mut disp_s = String::new();
        let _ = write!(&mut disp_s, "{}", &worker);
        assert!(disp_s.contains("SearchWorker #7"));

        // run_once consumes worker and should call strategy.run
        worker.run_once();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_search_pool_runs_all_workers() {
        let problem = make_basic_problem();
        let (model, incumbent) = make_model_and_incumbent(&problem);
        let stop = AtomicBool::new(false);

        let mut counters = Vec::new();
        let mut workers = Vec::new();

        for i in 0..4 {
            let rng = ChaCha8Rng::seed_from_u64(i as u64);
            let ctx = SearchWorkerContext::new(&problem, &model, &incumbent, &stop, rng);
            let (strategy, counter) = DummyStrategy::new("PoolDummy");
            counters.push(counter);
            let w = SearchWorker::new(i, Box::new(strategy), ctx);
            workers.push(w);
        }

        let pool = SearchPool::new(workers);
        pool.run_scoped();

        for (i, c) in counters.iter().enumerate() {
            assert_eq!(
                c.load(Ordering::SeqCst),
                1,
                "worker {i} should be run exactly once"
            );
        }
    }
}
