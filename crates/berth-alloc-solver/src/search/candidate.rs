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
    search::patch::VarPatch,
    state::chain_set::delta::ChainSetDelta,
};
use berth_alloc_core::prelude::Cost;

/// Represents a fully evaluated, feasible "move" that a search algorithm can apply.
///
/// This struct is the output of the `CandidateEvaluator`. It contains everything the
/// main search loop needs to accept a change: the structural `delta`, the resulting
/// changes to decision variables, and the calculated change in cost (for both the
/// true and search objectives).
#[derive(Debug, Clone)]
pub struct NeighborhoodCandidate<T> {
    /// The structural change to the chain set (e.g., "move request 5 after request 10").
    pub delta: ChainSetDelta,
    /// A sparse list of changes to the `IntervalVar`s, resulting from re-scheduling.
    pub interval_var_patch: Vec<VarPatch<IntervalVar<T>>>,
    /// A sparse list of changes to the `DecisionVar`s, resulting from re-scheduling.
    pub decision_vars_patch: Vec<VarPatch<DecisionVar<T>>>,
    /// The incremental change in cost according to the "true" objective function.
    pub true_delta_cost: Cost,
    /// The incremental change in cost according to the "search" objective function.
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
