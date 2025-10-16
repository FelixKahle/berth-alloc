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
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    engine::context::SearchContext,
    eval::objective::Objective,
    model::{index::RequestIndex, solver_model::SolverModel},
    scheduling::traits::Scheduler,
    search::{
        candidate::NeighborhoodCandidate, filter::traits::FeasibilityFilter, patch::VarPatch,
    },
    state::{
        chain_set::{delta::ChainSetDelta, overlay::ChainSetOverlay, view::ChainSetView},
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};

/// A reusable, temporary workspace for evaluating a candidate move.
///
/// This is a critical performance optimization. Instead of allocating new vectors
/// for every "what-if" scenario, the `CandidateEvaluator` uses this pre-allocated
/// buffer. It copies in the relevant parts of the current state, mutates them during
/// evaluation, and then is reset for the next candidate.
#[derive(Debug)]
struct ScratchBuffer<T>
where
    T: Copy + Ord,
{
    /// A mutable copy of the solver's interval variables.
    interval_vars: Vec<IntervalVar<T>>,
    /// A mutable copy of the solver's decision variables.
    decision_vars: Vec<DecisionVar<T>>,
    /// A list of indices that have been affected by a delta.
    touched: Vec<usize>,
}

impl<T> ScratchBuffer<T>
where
    T: Copy + Ord,
{
    /// Creates a new scratch buffer, cloning the initial state. This is typically done
    /// once per thread at the start of the search.
    #[inline]
    fn new(interval_vars_base: &[IntervalVar<T>], decision_vars_base: &[DecisionVar<T>]) -> Self {
        let num_nodes = interval_vars_base.len();
        Self {
            interval_vars: interval_vars_base.to_vec(),
            decision_vars: decision_vars_base.to_vec(),
            touched: Vec::with_capacity(num_nodes),
        }
    }

    /// Clears the list of affected indices, ready for a new evaluation.
    #[inline]
    fn clear_touched(&mut self) {
        self.touched.clear();
    }

    /// Sorts and removes duplicates from the list of touched indices.
    #[inline]
    fn sort_and_dedup_touched(&mut self) {
        self.touched.sort_unstable();
        self.touched.dedup();
    }

    /// Records a request index as being affected by a move.
    #[inline]
    fn push_touched(&mut self, index: usize) {
        self.touched.push(index);
    }

    /// Resets the scratch variables for the affected indices back to the base state.
    /// This provides a clean slate for the next "what-if" evaluation.
    #[inline]
    fn reset_from_base_touched(&mut self, iv: &[IntervalVar<T>], dv: &[DecisionVar<T>]) {
        debug_assert_eq!(iv.len(), self.interval_vars.len());
        debug_assert_eq!(dv.len(), self.decision_vars.len());
        for &i in &self.touched {
            debug_assert!(i < self.interval_vars.len());
            self.interval_vars[i] = iv[i];
            self.decision_vars[i] = dv[i];
        }
    }
}

/// A core engine component responsible for evaluating a proposed `ChainSetDelta`.
///
/// This struct takes a purely structural change (e.g., "move this block of requests"),
/// and performs the full evaluation pipeline:
/// 1. Checks feasibility filters.
/// 2. Re-schedules the affected parts of the solution to repair it.
/// 3. Diffs the result to create sparse "patches" of what changed.
/// 4. Scores the change incrementally.
///
/// It is deterministic and does not contain any operator-specific logic.
pub struct CandidateEvaluator<T>
where
    T: Copy + Ord,
{
    buffer: ScratchBuffer<T>,
}

impl<T> CandidateEvaluator<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    /// Creates a new evaluator, which should be done once per search thread.
    #[inline]
    pub fn new<'model, 'problem>(state: &SolverSearchState<'model, 'problem, T>) -> Self {
        Self {
            buffer: ScratchBuffer::new(state.interval_vars(), state.decision_vars()),
        }
    }

    pub fn evaluate<'search, 'engine, 'model, 'problem, S>(
        &mut self,
        search_context: &'search SearchContext<'engine, 'model, 'problem, T, S>,
        state: &SolverSearchState<'model, 'problem, T>,
        delta: ChainSetDelta,
    ) -> Option<NeighborhoodCandidate<T>>
    where
        S: Scheduler<T>,
    {
        if !search_context.filters().is_feasible(&delta, state) {
            return None;
        }

        // --- NEW: reset using *previous* touched set ---
        let iv_base = state.interval_vars();
        let dv_base = state.decision_vars();

        // Take the previous touched set, reset those indices, then clear.
        let prev_touched = std::mem::take(&mut self.buffer.touched);
        if !prev_touched.is_empty() {
            // temporarily reuse buffer.touched to call reset
            self.buffer.touched = prev_touched;
            self.buffer.reset_from_base_touched(iv_base, dv_base);
            self.buffer.clear_touched();
        }

        // 1) BUILD current touched from overlay
        let overlay = ChainSetOverlay::new(state.chain_set(), &delta);
        for &chain_id in delta.affected_chains() {
            for node in overlay.iter_chain(chain_id) {
                if !overlay.is_sentinel_node(node) {
                    self.buffer.push_touched(node.get());
                }
            }
        }
        self.buffer.sort_and_dedup_touched();

        // 2) Reset these indices to base (now safe)
        self.buffer.reset_from_base_touched(iv_base, dv_base);

        // 3) Repair/schedule
        for &chain_id in delta.affected_chains() {
            if let Some(start_node) = overlay.earliest_impacted_on_chain(chain_id) {
                let chain_view = overlay.chain(chain_id);
                if search_context
                    .pipeline()
                    .run_slice_overlay(
                        state.model(),
                        chain_view,
                        start_node,
                        Some(chain_view.end()),
                        &mut self.buffer.interval_vars,
                        &mut self.buffer.decision_vars,
                    )
                    .is_err()
                {
                    return None;
                }
            }
        }

        // 4) Diff patches (same as you have)...
        let mut interval_vars_patch = Vec::with_capacity(self.buffer.touched.len());
        let mut decision_var_patch = Vec::with_capacity(self.buffer.touched.len());
        for &index in &self.buffer.touched {
            if self.buffer.interval_vars[index] != iv_base[index] {
                interval_vars_patch.push(VarPatch::new(self.buffer.interval_vars[index], index));
            }
            if self.buffer.decision_vars[index] != dv_base[index] {
                decision_var_patch.push(VarPatch::new(self.buffer.decision_vars[index], index));
            }
        }

        let true_delta_cost = Self::incremental_cost_with_against(
            state.model(),
            search_context.objective(),
            dv_base,
            &decision_var_patch,
        )?;
        let search_delta_cost = Self::incremental_cost_with_against(
            state.model(),
            search_context.search_objective(),
            dv_base,
            &decision_var_patch,
        )?;

        Some(NeighborhoodCandidate::new(
            delta,
            interval_vars_patch,
            decision_var_patch,
            true_delta_cost,
            search_delta_cost,
        ))
    }

    /// Helper function to perform incremental cost calculation (delta evaluation).
    /// It calculates `(cost_after - cost_before)` only for the variables that changed.
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
        for patch in dv_patch {
            let index = patch.index();
            let before = dv_before[index];
            let after = *patch.patch();

            let cost_before = Self::cost_of_dv(model, obj, index, before)?;
            let cost_after = Self::cost_of_dv(model, obj, index, after)?;
            acc = acc.saturating_add(cost_after.saturating_sub(cost_before));
        }
        Some(acc)
    }

    /// Helper to get the cost of a single decision variable.
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
        let request_index = RequestIndex(req_idx_usize);
        match dv {
            DecisionVar::Unassigned => Some(objective.unassignment_cost(model, request_index)),
            DecisionVar::Assigned(decision) => objective.assignment_cost(
                model,
                request_index,
                decision.berth_index,
                decision.start_time,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::index::BerthIndex, state::chain_set::delta::ChainSetDelta};
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::builder::ProblemBuilder,
        problem::req::Request,
    };
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex(n)
    }

    // Build a Problem:
    // - berths_windows[b] = vec![(s,e), ...] availability windows for berth b (ids 0..B-1).
    // - request_windows[r] = (s,e) feasible window for request r (ids 0..R-1).
    // - weights[r] = weight for request r
    // - processing[r][b] = Some(dur) if r allowed on berth b with PT=dur; None otherwise.
    fn build_problem_with_weights(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        weights: &[i64],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let b_len = berths_windows.len();
        let r_len = request_windows.len();
        assert_eq!(weights.len(), r_len);
        assert_eq!(processing.len(), r_len);
        for row in processing {
            assert_eq!(
                row.len(),
                b_len,
                "processing times per request must match number of berths"
            );
        }

        let mut builder = ProblemBuilder::new();

        for (i, windows) in berths_windows.iter().enumerate() {
            let b = Berth::from_windows(bid(i), windows.iter().map(|&(s, e)| iv(s, e)));
            builder.add_berth(b);
        }

        for (i, &(ws, we)) in request_windows.iter().enumerate() {
            let mut map = BTreeMap::new();
            for (j, p) in processing[i].iter().copied().enumerate() {
                if let Some(dur) = p {
                    map.insert(bid(j), td(dur));
                }
            }
            let req =
                Request::<FlexibleKind, i64>::new(rid(i), iv(ws, we), weights[i], map).unwrap();
            builder.add_flexible(req);
        }

        builder.build().expect("problem should build")
    }

    #[test]
    fn test_incremental_cost_and_cost_of_dv_helpers_work() {
        // 1 berth with [0,10), 1 request weight=2, PT=5
        let p = build_problem_with_weights(&[vec![(0, 10)]], &[(0, 10)], &[2], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        // before: Unassigned; patch: Assigned on berth 0 at t=0
        let dv_before = vec![DecisionVar::Unassigned];
        let dv_after = DecisionVar::assigned(bi(0), tp(0));
        let dv_patch = vec![VarPatch::new(dv_after, 0)];

        // true objective = WeightedTurnaroundTimeObjective
        let true_obj = crate::eval::wtt::WeightedTurnaroundTimeObjective;
        let delta_true = super::CandidateEvaluator::<i64>::incremental_cost_with_against(
            &m, &true_obj, &dv_before, &dv_patch,
        )
        .expect("delta should compute");
        // Assignment = 2*5=10, unassignment=2*10=20 => delta = -10
        assert_eq!(delta_true, 10 - 20);

        // cost_of_dv for assigned and unassigned
        let c_assigned = super::CandidateEvaluator::<i64>::cost_of_dv(&m, &true_obj, 0, dv_after)
            .expect("assigned cost");
        let c_unassigned =
            super::CandidateEvaluator::<i64>::cost_of_dv(&m, &true_obj, 0, DecisionVar::Unassigned)
                .expect("unassigned cost");
        assert_eq!(c_assigned, 10);
        assert_eq!(c_unassigned, 20);
    }

    #[test]
    fn test_scratch_buffer_tracks_and_resets_touched_indices() {
        let iv_base = vec![
            IntervalVar::new(tp(0), tp(10)),
            IntervalVar::new(tp(5), tp(15)),
        ];
        let dv_base = vec![DecisionVar::Unassigned, DecisionVar::Unassigned];

        let mut buf = super::ScratchBuffer::new(&iv_base, &dv_base);

        // Touch out-of-order with duplicates
        buf.push_touched(1);
        buf.push_touched(0);
        buf.push_touched(1);
        buf.sort_and_dedup_touched();
        assert_eq!(buf.touched, vec![0, 1]);

        // Mutate buffer and then reset from base only for touched
        buf.interval_vars[0].start_time_lower_bound = tp(99);
        buf.decision_vars[1] = DecisionVar::assigned(bi(0), tp(3));
        buf.reset_from_base_touched(&iv_base, &dv_base);
        assert_eq!(buf.interval_vars, iv_base);
        assert_eq!(buf.decision_vars, dv_base);

        // Clear and ensure touched is empty
        buf.clear_touched();
        assert!(buf.touched.is_empty());
    }

    #[test]
    fn test_neighborhood_candidate_new_packs_fields() {
        let cand = NeighborhoodCandidate::new(
            ChainSetDelta::new(),
            vec![VarPatch::new(IntervalVar::new(tp(0), tp(1)), 0)],
            vec![VarPatch::new(DecisionVar::assigned(bi(0), tp(0)), 0)],
            -7,
            -9,
        );
        assert_eq!(cand.interval_var_patch.len(), 1);
        assert_eq!(cand.decision_vars_patch.len(), 1);
        assert_eq!(cand.true_delta_cost, -7);
        assert_eq!(cand.search_delta_cost, -9);
    }
}
