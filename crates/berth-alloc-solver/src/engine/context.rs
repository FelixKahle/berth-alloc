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
    eval::{objective::Objective, search::SearchObjective, wtt::WeightedTurnaroundTimeObjective},
    model::{
        index::{BerthIndex, RequestIndex},
        neighborhood::ProximityMap,
        solver_model::SolverModel,
    },
    scheduling::{pipeline::SchedulingPipeline, traits::Scheduler},
    search::{filter::filter_stack::FilterStack, operator::runner::NeighborhoodCandidate},
    state::{
        chain_set::{
            index::NodeIndex,
            view::{ChainRef, ChainSetView},
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
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

        println!("Initial Cost: {}", state.current_true_cost());

        Self {
            engine_context,
            objective: weighted_objective,
            state,
            search_objective: search_objective,
            operators: OperatorPool::new(),
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

    #[inline]
    pub fn accept_candidate(&mut self, candidate: NeighborhoodCandidate<T>) {
        self.state.apply_candidate(candidate);
    }

    #[inline]
    pub fn state(&self) -> &SolverSearchState<'model, 'problem, T> {
        &self.state
    }

    #[inline]
    pub fn make_search_arc_eval<V>(
        &self,
        chain: ChainRef<'_, V>,
    ) -> impl Fn(NodeIndex, NodeIndex) -> Option<Cost>
    where
        V: ChainSetView,
        T: CheckedAdd + CheckedSub + Into<Cost>,
    {
        let model: &SolverModel<'problem, T> = self.engine_context.model();
        let objective: &SearchObjective<WeightedTurnaroundTimeObjective> = self.search_objective();
        move |from, to| eval_arc_with_objective::<T, _, V>(model, chain, objective, from, to)
    }

    #[inline]
    pub fn make_true_arc_eval<V>(
        &self,
        chain: ChainRef<V>,
    ) -> impl Fn(NodeIndex, NodeIndex) -> Option<Cost>
    where
        V: ChainSetView,
        T: CheckedAdd + CheckedSub + Into<Cost>,
    {
        let model: &SolverModel<'problem, T> = self.engine_context.model();
        let objective: &WeightedTurnaroundTimeObjective = self.objective();

        move |from, to| eval_arc_with_objective::<T, _, V>(model, chain, objective, from, to)
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

#[inline]
fn eval_arc_with_objective<'problem, T, O, V>(
    model: &SolverModel<'problem, T>,
    chain: ChainRef<'_, V>,
    objective: &O,
    from: NodeIndex,
    to: NodeIndex,
) -> Option<Cost>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
    O: Objective<T>,
    V: ChainSetView,
{
    let bi = BerthIndex(chain.chain_index().get());
    if bi.get() >= model.berths_len() {
        return None;
    }

    // Resolve [from, to) against this chain, skipping sentinels if needed.
    let (cur_opt, end_exclusive) = chain.resolve_slice(from, Some(to));
    let Some(mut cur) = cur_opt else {
        return Some(0); // empty slice
    };

    // ---------- 1) Build cursor by scheduling START..from (exclusive)
    // If `from` is the sentinel START, this stays None (use each req TW start).
    let mut cursor: Option<_> = None;
    if chain.start() != from {
        // First real node after START
        let mut n = match chain.first_real_node(chain.start()) {
            Some(x) => x,
            None => return Some(0), // empty chain
        };

        let mut t_opt: Option<_> = None;
        let mut steps_left = model.flexible_requests_len();

        while n != from {
            if steps_left == 0 {
                return None;
            }
            steps_left -= 1;

            // Only request nodes are valid for model indexing.
            let ridx = n.get();
            if ridx >= model.flexible_requests_len() {
                return None; // not a request node (sentinel or foreign)
            }
            let ri = RequestIndex(ridx);

            let req_tw = model.feasible_intervals()[ri.get()];
            let dur = match model.processing_time(ri, bi) {
                Some(Some(d)) => d,
                _ => return None,
            };

            let base = t_opt.unwrap_or(req_tw.start());
            let start = earliest_fit_after_in_calendar(model, bi, req_tw, dur, base)?;
            t_opt = Some(start.checked_add(dur)?);

            // advance to next real node
            n = match chain.next_real(n) {
                Some(x) => x,
                None => break,
            };
        }
        cursor = t_opt;
    }

    // ---------- 2) Schedule [from, to) and accumulate objective
    let mut acc: Cost = 0;
    let mut steps_left = model.flexible_requests_len();

    while cur != end_exclusive {
        if steps_left == 0 {
            return None;
        }
        steps_left -= 1;

        // Guard: current must be a request node
        let ridx = cur.get();
        if ridx >= model.flexible_requests_len() {
            return None;
        }
        let ri = RequestIndex(ridx);

        let req_tw = model.feasible_intervals()[ri.get()];
        let dur = match model.processing_time(ri, bi) {
            Some(Some(d)) => d,
            _ => return None,
        };

        let base = cursor.unwrap_or(req_tw.start());
        let start = earliest_fit_after_in_calendar(model, bi, req_tw, dur, base)?;
        acc = acc.saturating_add(objective.assignment_cost(model, ri, bi, start)?);
        cursor = Some(start.checked_add(dur)?);

        cur = match chain.next_real(cur) {
            Some(x) => x,
            None => break,
        };
    }

    Some(acc)
}

fn earliest_fit_after_in_calendar<T>(
    model: &SolverModel<'_, T>,
    berth: BerthIndex,
    req_tw: TimeInterval<T>,
    dur: TimeDelta<T>,
    cursor: TimePoint<T>,
) -> Option<TimePoint<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let cal = model.calendar_for_berth(berth)?;
    for slot in cal.free_intervals() {
        // intersection of [req.start, req.end) ∩ [slot.start, slot.end) ∩ [cursor, +∞)
        let lo = max_tp(max_tp(req_tw.start(), slot.start()), cursor);
        let hi = min_tp(req_tw.end(), slot.end());
        let latest = hi.checked_sub(dur)?;
        if lo <= latest {
            return Some(lo);
        }
    }
    None
}

#[inline]
fn max_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a >= b { a } else { b }
}

#[inline]
fn min_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a <= b { a } else { b }
}
