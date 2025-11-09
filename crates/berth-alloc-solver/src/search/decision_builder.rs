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
    model::solver_model::SolverModel,
    monitor::search_monitor::PlanEventMonitor,
    search::eval::CostEvaluator,
    state::{decisionvar::DecisionVar, fitness::Fitness, plan::Plan, solver_state::SolverState},
};

pub struct SearchContext<'b, 'sm, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub model: &'m SolverModel<'p, T>,
    pub state: &'s SolverState<'p, T>,
    pub evaluator: &'c C,
    pub rng: &'b mut R,
    pub work_buf: &'b mut [DecisionVar<T>],
    pub current_fitness: Fitness,
    pub monitor: &'sm mut dyn PlanEventMonitor<T>,
}

impl<'b, 'sm, 'c, 's, 'm, 'p, T, C, R> SearchContext<'b, 'sm, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        state: &'s SolverState<'p, T>,
        evaluator: &'c C,
        rng: &'b mut R,
        work_buf: &'b mut [DecisionVar<T>],
        current_fitness: Fitness,
        monitor: &'sm mut dyn PlanEventMonitor<T>,
    ) -> SearchContext<'b, 'sm, 'c, 's, 'm, 'p, T, C, R> {
        SearchContext {
            model,
            state,
            evaluator,
            rng,
            work_buf,
            current_fitness,
            monitor,
        }
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn state(&self) -> &'s SolverState<'p, T> {
        self.state
    }

    #[inline]
    pub fn evaluator(&self) -> &'c C {
        self.evaluator
    }

    #[inline]
    pub fn rng(&mut self) -> &mut R {
        self.rng
    }

    #[inline]
    pub fn work_buf(&mut self) -> &mut [DecisionVar<T>] {
        self.work_buf
    }

    #[inline]
    pub fn current_fitness(&self) -> Fitness {
        self.current_fitness
    }
}

pub trait DecisionBuilder<T, C, R>: Send
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn reset(&mut self);

    fn next<'b, 'sm, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>>;
}

impl<'a, T, C, R> std::fmt::Debug for dyn DecisionBuilder<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DecisionBuilder({})", self.name())
    }
}

impl<'a, T, C, R> std::fmt::Display for dyn DecisionBuilder<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DecisionBuilder({})", self.name())
    }
}
