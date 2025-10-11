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

use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};

use crate::{
    eval::{
        arc_evaluator::ObjectiveArcEvaluator, search::SearchObjective,
        wtt::WeightedTurnaroundTimeObjective,
    },
    scheduling::{greedy::GreedyCalendar, pipeline::PipelineScheduler, traits::CalendarScheduler},
    search::filter::{filter_stack::FilterStack, traits::FeasibilityFilter},
    state::model::SolverModel,
};

pub struct EngineContext<'model, 'problem, T, S = PipelineScheduler<GreedyCalendar, T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: CalendarScheduler<T>,
{
    model: &'model SolverModel<'problem, T>,
    scheduler: S,
    filters: FilterStack<'model, 'problem, T>,
}

impl<'model, 'problem, T, S> EngineContext<'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: CalendarScheduler<T>,
{
    pub fn new(model: &'model SolverModel<'problem, T>, scheduler: S) -> Self {
        Self {
            model,
            scheduler,
            filters: FilterStack::new(),
        }
    }

    #[inline]
    pub fn with_filter(mut self, f: Box<dyn FeasibilityFilter<'model, 'problem, T>>) -> Self {
        self.filters.add_filter(f);
        self
    }

    #[inline]
    pub fn model(&self) -> &SolverModel<'problem, T> {
        self.model
    }

    #[inline]
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }

    #[inline]
    pub fn filters(&self) -> &FilterStack<'model, 'problem, T> {
        &self.filters
    }
}

pub struct SearchContext<'engine, 'model, 'problem, T, S = PipelineScheduler<GreedyCalendar, T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: CalendarScheduler<T>,
{
    engine_context: &'engine EngineContext<'model, 'problem, T, S>,
    objective: WeightedTurnaroundTimeObjective,
    search_objective: SearchObjective<WeightedTurnaroundTimeObjective>,
}

impl<'engine, 'model, 'problem, T, S> SearchContext<'engine, 'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: CalendarScheduler<T>,
{
    pub fn new(
        engine_context: &'engine EngineContext<'model, 'problem, T, S>,
        lambda: f64,
    ) -> Self {
        Self {
            engine_context,
            objective: WeightedTurnaroundTimeObjective,
            search_objective: SearchObjective::new(WeightedTurnaroundTimeObjective, lambda),
        }
    }

    #[inline]
    pub fn engine_context(&self) -> &'engine EngineContext<'model, 'problem, T, S> {
        self.engine_context
    }

    #[inline]
    pub fn objective(&self) -> &WeightedTurnaroundTimeObjective {
        &self.objective
    }

    #[inline]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.search_objective.set_lambda(lambda);
    }

    #[inline]
    pub fn search_objective(&self) -> &SearchObjective<WeightedTurnaroundTimeObjective> {
        &self.search_objective
    }

    #[inline]
    pub fn scheduler(&self) -> &S {
        self.engine_context.scheduler()
    }

    #[inline]
    pub fn filters(&self) -> &FilterStack<'model, 'problem, T> {
        self.engine_context.filters()
    }

    #[inline]
    pub fn search_arc_evaluator(
        &self,
    ) -> ObjectiveArcEvaluator<'_, T, SearchObjective<WeightedTurnaroundTimeObjective>>
    where
        T: CheckedAdd + CheckedSub + Into<Cost>,
    {
        ObjectiveArcEvaluator::new(self.search_objective())
    }

    #[inline]
    pub fn true_arc_evaluator(
        &self,
    ) -> ObjectiveArcEvaluator<'_, T, WeightedTurnaroundTimeObjective>
    where
        T: CheckedAdd + CheckedSub + Into<Cost>,
    {
        ObjectiveArcEvaluator::new(self.objective())
    }
}
