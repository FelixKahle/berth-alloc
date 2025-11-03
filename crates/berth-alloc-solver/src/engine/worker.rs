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
    search::{
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
    },
    state::{decisionvar::DecisionVar, solver_state::SolverState},
};
use std::thread;

pub struct SearchWorker<'e, 'm, 'p, T, C, R>
where
    T: Copy + Ord + Send + 'p,
    C: CostEvaluator<T> + Send + Sync + 'static,
    R: rand::Rng + Send + 'static,
{
    id: usize,
    model: &'m SolverModel<'p, T>,
    evaluator: &'e C,
    shared_incumbent: &'e SharedIncumbent<'p, T>,
    state: SolverState<'p, T>,
    decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send>,
    rng: R,
}
impl<'e, 'm, 'p, T, C, R> SearchWorker<'e, 'm, 'p, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
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
    ) -> Self {
        /* assign fields */
        Self {
            id,
            model,
            evaluator,
            shared_incumbent,
            state: initial_state,
            decision_builder: db,
            rng,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn run(mut self) {
        let mut work_buf = vec![DecisionVar::unassigned(); self.model.flexible_requests_len()];

        loop {
            let current_fitness = *self.state.fitness();
            let mut ctx = SearchContext::new(
                self.model,
                &self.state,
                self.evaluator,
                &mut self.rng,
                &mut work_buf,
                current_fitness,
            );

            match self.decision_builder.next(&mut ctx) {
                Some(plan) => {
                    self.state.apply_plan(plan);
                    let _ = self.shared_incumbent.try_update(&self.state, self.model);
                    continue;
                }
                None => break,
            }
        }
    }
}

pub struct SearchPool<'e, 'm, 'p, T, C, R>
where
    T: Copy + Ord + Send + Sync + 'p,
    C: CostEvaluator<T> + Send + Sync + 'static,
    R: rand::Rng + Send + 'static,
{
    workers: Vec<SearchWorker<'e, 'm, 'p, T, C, R>>,
}

impl<'e, 'm, 'p, T, C, R> SearchPool<'e, 'm, 'p, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    #[inline]
    pub fn new(workers: Vec<SearchWorker<'e, 'm, 'p, T, C, R>>) -> Self {
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
