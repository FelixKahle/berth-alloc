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
    engine::shared_incumbent::SharedIncumbent,
    model::solver_model::SolverModel,
    monitor::{controller::GlobalController, observer::SearchObserver, termination::Termination},
    search::{
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
    },
    state::{decisionvar::DecisionVar, solver_state::SolverState},
};
use std::sync::Arc;
use std::thread;

pub struct SearchWorker<'e, 'm, 'p, T, C, R, Obs>
where
    T: Copy + Ord + Send + 'p,
    C: CostEvaluator<T> + Send + Sync + 'static,
    R: rand::Rng + Send + 'static,
    Obs: SearchObserver,
{
    id: usize,
    model: &'m SolverModel<'p, T>,
    evaluator: &'e C,
    shared_incumbent: &'e SharedIncumbent<'p, T>,

    state: SolverState<'p, T>,
    db: Box<dyn DecisionBuilder<T, C, R> + Send>,

    rng: R,
    ctrl: Arc<GlobalController>,
    observer: Obs,
}
impl<'e, 'm, 'p, T, C, R, Obs> SearchWorker<'e, 'm, 'p, T, C, R, Obs>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync + 'static,
    R: rand::Rng + Send + 'static,
    Obs: SearchObserver,
{
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        id: usize,
        model: &'m SolverModel<'p, T>,
        evaluator: &'e C,
        shared_incumbent: &'e SharedIncumbent<'p, T>,
        initial_state: SolverState<'p, T>,
        db: Box<dyn DecisionBuilder<T, C, R> + Send>,
        rng: R,
        ctrl: Arc<GlobalController>,
        observer: Obs,
    ) -> Self {
        /* assign fields */
        Self {
            id,
            model,
            evaluator,
            shared_incumbent,
            state: initial_state,
            db,
            rng,
            ctrl,
            observer,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn run(mut self) {
        let mut work_buf = vec![DecisionVar::unassigned(); self.model.flexible_requests_len()];
        let mut iter_id: u64 = 0;
        self.observer.on_search_start();

        // Build one termination policy per worker.
        let mut term = Termination::from_controller(self.ctrl.clone());

        loop {
            // Very cheap check (iter budget, sampling, external stop)
            if term.tick_iteration() {
                break;
            }

            let current_fitness = *self.state.fitness();
            let mut ctx = SearchContext::new(
                self.model,
                &self.state,
                self.evaluator,
                &mut self.rng,
                &mut work_buf,
                current_fitness,
                &mut term,
            );

            match self.db.next(&mut ctx) {
                Some(plan) => {
                    let old_cost = self.state.fitness().cost;
                    self.state.apply_plan(plan);
                    let new_cost = self.state.fitness().cost;

                    // Rare atomics (accepted, maybe incumbent)
                    self.ctrl.on_accepted();
                    if self.shared_incumbent.try_update(&self.state, self.model) {
                        self.ctrl.on_incumbent_improvement();
                        self.observer.on_new_incumbent(iter_id, new_cost);
                    }
                    self.observer
                        .on_iteration_accepted(iter_id, new_cost, old_cost);
                    iter_id += 1;
                    continue;
                }
                None => break,
            }
        }

        self.observer.on_search_end();
    }
}

pub struct SearchPool<'e, 'm, 'p, T, C, R, Obs>
where
    T: Copy + Ord + Send + Sync + 'p,
    C: CostEvaluator<T> + Send + Sync + 'static,
    R: rand::Rng + Send + 'static,
    Obs: SearchObserver + Send + 'static,
{
    workers: Vec<SearchWorker<'e, 'm, 'p, T, C, R, Obs>>,
}

impl<'e, 'm, 'p, T, C, R, Obs> SearchPool<'e, 'm, 'p, T, C, R, Obs>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync + 'static,
    R: rand::Rng + Send + 'static,
    Obs: SearchObserver + Send + 'static,
{
    #[inline]
    pub fn new(workers: Vec<SearchWorker<'e, 'm, 'p, T, C, R, Obs>>) -> Self {
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
