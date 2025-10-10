// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
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
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    engine::context::EngineContext,
    eval::objective::Objective,
    scheduling::traits::CalendarScheduler,
    search::{filter::traits::FeasibilityFilter, operator::patch::VarPatch},
    state::{
        chain_set::{delta::ChainSetDelta, overlay::ChainSetOverlay, view::ChainSetView},
        index::RequestIndex,
        model::SolverModel,
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone)]
struct Scratch<T> {
    iv: Vec<IntervalVar<T>>,
    dv: Vec<DecisionVar<T>>,
}

impl<T: Copy + Ord> Scratch<T> {
    fn new(iv_base: &[IntervalVar<T>], dv_base: &[DecisionVar<T>]) -> Self {
        Self {
            iv: iv_base.to_vec(),
            dv: dv_base.to_vec(),
        }
    }

    fn reset_from_base(
        &mut self,
        iv_base: &[IntervalVar<T>],
        dv_base: &[DecisionVar<T>],
        touched: &[usize],
    ) {
        for &i in touched {
            self.iv[i] = iv_base[i];
            self.dv[i] = dv_base[i];
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeighborhoodCandidate<T> {
    pub delta: ChainSetDelta,
    pub interval_var_patch: Vec<VarPatch<IntervalVar<T>>>,
    pub decision_vars_patch: Vec<VarPatch<DecisionVar<T>>>,
    pub true_delta_cost: Cost,
    pub search_delta_cost: Cost,
}

impl<T> NeighborhoodCandidate<T> {
    #[inline]
    pub fn new(
        delta: ChainSetDelta,
        interval_var_patch: Vec<VarPatch<IntervalVar<T>>>,
        decision_vars_patch: Vec<VarPatch<DecisionVar<T>>>,
        true_delta_cost: Cost,
        search_delta_cost: Cost,
    ) -> Self {
        Self {
            delta,
            interval_var_patch,
            decision_vars_patch,
            true_delta_cost,
            search_delta_cost,
        }
    }
}

/// CandidateEvaluator: given a pure structural delta, check feasibility, repair/schedule,
/// compute sparse patches, and score incremental cost. No RNG, no ArcEvaluator, no operator call.
pub struct CandidateEvaluator<T>
where
    T: Copy + Ord,
{
    scratch: Scratch<T>,
}

impl<T> CandidateEvaluator<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    #[inline]
    pub fn new<'model, 'problem>(state: &SolverSearchState<'model, 'problem, T>) -> Self {
        Self {
            scratch: Scratch::new(state.interval_vars(), state.decision_vars()),
        }
    }

    /// Evaluate a candidate delta against the current search state.
    /// Returns None if any filter rejects it or the scheduler cannot repair.
    pub fn evaluate<'model, 'problem, S>(
        &mut self,
        engine_context: &EngineContext<'model, 'problem, T, S>,
        state: &SolverSearchState<'model, 'problem, T>,
        delta: ChainSetDelta,
    ) -> Option<NeighborhoodCandidate<T>>
    where
        S: CalendarScheduler<T>,
    {
        // 1) fast feasibility via filters
        if !engine_context.filters().is_feasible(&delta, state) {
            return None;
        }

        // 2) overlay + touched nodes
        let base_cs = state.chain_set();
        let overlay = ChainSetOverlay::new(base_cs, &delta);

        let mut touched: Vec<usize> = Vec::new();
        for &cid in delta.affected_chains() {
            for n in overlay.iter_chain(cid) {
                if overlay.is_sentinel_node(n) {
                    continue;
                }
                touched.push(n.get());
            }
        }
        touched.sort_unstable();
        touched.dedup();

        // 3) reset scratch rows from base
        self.scratch
            .reset_from_base(state.interval_vars(), state.decision_vars(), &touched);

        // 4) schedule only impacted slices per affected chain
        for &cid in delta.affected_chains() {
            if let Some(start) = overlay.earliest_impacted_on_chain(cid) {
                let chain = overlay.chain(cid);
                let end_excl = chain.end();
                if engine_context
                    .scheduler()
                    .schedule_chain_slice(
                        state.model(),
                        chain,
                        start,
                        Some(end_excl),
                        &mut self.scratch.iv,
                        &mut self.scratch.dv,
                    )
                    .is_err()
                {
                    // could not repair ⇒ infeasible
                    return None;
                }
            }
        }

        // 5) sparse patches by diffing touched indices
        let mut iv_patch = Vec::with_capacity(touched.len());
        let mut dv_patch = Vec::with_capacity(touched.len());
        let iv_base = state.interval_vars();
        let dv_base = state.decision_vars();

        for i in touched {
            if self.scratch.iv[i] != iv_base[i] {
                iv_patch.push(VarPatch::new(self.scratch.iv[i], i));
            }
            if self.scratch.dv[i] != dv_base[i] {
                dv_patch.push(VarPatch::new(self.scratch.dv[i], i));
            }
        }

        // 6) incremental scoring using the Objective (assignment/unassignment deltas)
        let true_delta_cost = Self::incremental_cost_with_against(
            state.model(),
            engine_context.objective(),
            dv_base,
            &dv_patch,
        )?;

        let search_delta_cost = Self::incremental_cost_with_against(
            state.model(),
            engine_context.search_objective(),
            dv_base,
            &dv_patch,
        )?;

        Some(NeighborhoodCandidate::new(
            delta,
            iv_patch,
            dv_patch,
            true_delta_cost,
            search_delta_cost,
        ))
    }

    #[inline]
    fn incremental_cost_with_against<O>(
        model: &SolverModel<T>,
        obj: &O,
        dv_before: &[DecisionVar<T>],
        dv_patch: &[VarPatch<DecisionVar<T>>],
    ) -> Option<Cost>
    where
        O: Objective<T>,
    {
        let mut acc: Cost = 0;
        for p in dv_patch {
            let idx = p.index();
            let before = dv_before[idx];
            let after = *p.patch();

            let cb = Self::cost_of_dv(model, obj, idx, before)?;
            let ca = Self::cost_of_dv(model, obj, idx, after)?;
            acc = acc.saturating_add(ca.saturating_sub(cb));
        }
        Some(acc)
    }

    #[inline]
    fn cost_of_dv<O>(
        model: &SolverModel<T>,
        objective: &O,
        req_idx_usize: usize,
        dv: DecisionVar<T>,
    ) -> Option<Cost>
    where
        O: Objective<T>,
    {
        let ri = RequestIndex(req_idx_usize);
        match dv {
            DecisionVar::Unassigned => Some(objective.unassignment_cost(model, ri)),
            DecisionVar::Assigned(dec) => {
                objective.assignment_cost(model, ri, dec.berth_index, dec.start_time)
            }
        }
    }
}
