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
    engine::operators::OperatorPool,
    eval::{search::SearchObjective, wtt::WeightedTurnaroundTimeObjective},
    model::{neighborhood::ProximityMap, solver_model::SolverModel},
    scheduling::{pipeline::SchedulingPipeline, traits::Scheduler},
    search::{
        candidate::NeighborhoodCandidate, filter::filter_stack::FilterStack,
        perturbation::Perturbation,
    },
    state::search_state::SolverSearchState,
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug)]
pub struct EngineContext<'engine, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: Scheduler<T>,
{
    model: &'engine SolverModel<'problem, T>,
    proximity_map: &'engine ProximityMap,
    pipeline: &'engine SchedulingPipeline<T, S>,
    filters: &'engine FilterStack<T>,
}

impl<'model, 'problem, T, S> EngineContext<'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: Scheduler<T>,
{
    pub fn new(
        model: &'model SolverModel<'problem, T>,
        close_model: &'model ProximityMap,
        scheduler: &'model SchedulingPipeline<T, S>,
        filters: &'model FilterStack<T>,
    ) -> Self {
        Self {
            model,
            proximity_map: close_model,
            pipeline: scheduler,
            filters,
        }
    }

    #[inline]
    pub fn model(&self) -> &SolverModel<'problem, T> {
        self.model
    }

    #[inline]
    pub fn proximity_map(&self) -> &ProximityMap {
        self.proximity_map
    }

    #[inline]
    pub fn pipeline(&self) -> &SchedulingPipeline<T, S> {
        self.pipeline
    }

    #[inline]
    pub fn filters(&self) -> &'model FilterStack<T> {
        self.filters
    }
}

pub struct SearchContext<'engine, 'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: Scheduler<T>,
{
    engine_context: &'engine EngineContext<'model, 'problem, T, S>,
    state: SolverSearchState<'model, 'problem, T>,
    objective: WeightedTurnaroundTimeObjective,
    search_objective: SearchObjective<WeightedTurnaroundTimeObjective>,
    operators: OperatorPool<'engine, T>,
    pertubations: Vec<Box<dyn Perturbation<T> + 'engine>>,
}

impl<'engine, 'model, 'problem, T, S> SearchContext<'engine, 'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
    S: Scheduler<T>,
{
    pub fn new(
        engine_context: &'engine EngineContext<'model, 'problem, T, S>,
        mut state: SolverSearchState<'model, 'problem, T>,
        lambda: f64,
    ) -> Self {
        let weighted_objective = WeightedTurnaroundTimeObjective;
        let search_objective = SearchObjective::new(weighted_objective.clone(), lambda);
        state.recompute_costs(&weighted_objective, &search_objective);

        Self {
            engine_context,
            objective: weighted_objective,
            state,
            search_objective: search_objective,
            operators: OperatorPool::new(),
            pertubations: Vec::new(),
        }
    }

    #[inline]
    pub fn model(&self) -> &SolverModel<'problem, T> {
        self.engine_context.model()
    }

    #[inline]
    pub fn proximity_map(&self) -> &ProximityMap {
        self.engine_context.proximity_map()
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
    pub fn pipeline(&self) -> &'engine SchedulingPipeline<T, S> {
        self.engine_context.pipeline()
    }

    #[inline]
    pub fn filters(&self) -> &'engine FilterStack<T> {
        self.engine_context.filters()
    }

    pub fn operators(&self) -> &OperatorPool<'engine, T> {
        &self.operators
    }

    pub fn operators_mut(&mut self) -> &mut OperatorPool<'engine, T> {
        &mut self.operators
    }

    pub fn pertubations(&self) -> &Vec<Box<dyn Perturbation<T> + 'engine>> {
        &self.pertubations
    }

    pub fn pertubations_mut(&mut self) -> &mut Vec<Box<dyn Perturbation<T> + 'engine>> {
        &mut self.pertubations
    }

    #[inline]
    pub fn accept_candidate(&mut self, candidate: NeighborhoodCandidate<T>) {
        self.state.apply_candidate(candidate);
    }

    #[inline]
    pub fn state(&self) -> &SolverSearchState<'model, 'problem, T> {
        &self.state
    }

    #[inline]
    pub fn state_mut(&mut self) -> &mut SolverSearchState<'model, 'problem, T> {
        &mut self.state
    }

    #[inline]
    pub fn into_state(self) -> SolverSearchState<'model, 'problem, T> {
        self.state
    }
}
