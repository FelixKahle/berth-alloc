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
    eval::ArcEvaluator,
    scheduling::traits::Scheduler,
    search::perturbation::Perturbation,
    state::{
        chain_set::{
            base::ChainSet,
            delta::ChainSetDelta,
            delta_builder::ChainSetDeltaBuilder,
            index::{ChainIndex, NodeIndex},
            overlay::ChainSetOverlay,
            view::ChainSetView,
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

/// Greedy (first-feasible) ruin & repair:
/// - Randomly pick up to `remove_k` real nodes across all chains.
/// - RUIN: detach them (node -> node, bypass in chain).
/// - REPAIR: scan chains/gaps in a fixed order; the first gap that schedules successfully is accepted.
/// - If a node can't be placed anywhere, it remains isolated (left unassigned).
#[derive(Debug, Clone)]
pub struct GreedyRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub remove_k: usize,
    /// Optional hard cap on the number of (chain,gap) positions scanned per removed node.
    pub scan_cap: Option<usize>,
    /// Scheduler to validate placements.
    pub scheduler: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, S> GreedyRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub fn new(remove_k: usize, scan_cap: Option<usize>, scheduler: S) -> Self {
        Self {
            remove_k: remove_k.max(1),
            scan_cap,
            scheduler,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S> Perturbation<T> for GreedyRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    fn name(&self) -> &str {
        "GreedyRuinRepair(RandomFirstFeasible)"
    }

    fn apply<'s, 'm, 'p>(
        &self,
        search_state: &'s SolverSearchState<'m, 'p, T>,
        _arc_eval: &ArcEvaluator,
        ivars: &'m mut [IntervalVar<T>],
        dvars: &'m mut [DecisionVar<T>],
        rng: &mut ChaCha8Rng,
    ) -> ChainSetDelta {
        let cs: &ChainSet = search_state.chain_set();
        let model = search_state.model();

        // -----------------------------
        // 1) Gather all real nodes; shuffle and pick K
        // -----------------------------
        let mut real_nodes: Vec<NodeIndex> = Vec::with_capacity(model.flexible_requests_len());
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let mut u = cs.start_of_chain(ci);
            while let Some(v) = cs.next_node(u) {
                if v == cs.end_of_chain(ci) {
                    break;
                }
                real_nodes.push(v);
                u = v;
            }
        }

        if real_nodes.is_empty() {
            // Nothing to do → return empty delta
            return ChainSetDelta::new();
        }

        real_nodes.shuffle(rng);
        let k = self.remove_k.min(real_nodes.len());
        let picked = &real_nodes[..k];

        // -----------------------------
        // 2) RUIN: detach picked nodes
        // -----------------------------
        let mut builder = ChainSetDeltaBuilder::new(cs);
        for &x in picked {
            builder.detach(x);
        }

        // -----------------------------
        // 3) REPAIR: for each x, scan chains/gaps; first-feasible commit
        // -----------------------------
        for &x in picked {
            // fresh overlay on the *current* working delta
            let ov_now = ChainSetOverlay::new(cs, builder.delta());

            let mut scanned = 0usize;
            let mut placed = false;

            'chains: for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let (start, end) = (cs.start_of_chain(ci), cs.end_of_chain(ci));

                let mut d = start;
                loop {
                    // Optional per-node scan cap
                    if let Some(cap) = self.scan_cap {
                        if scanned >= cap {
                            break 'chains;
                        }
                    }
                    scanned += 1;

                    let Some(succ) = ov_now.next_node(d) else {
                        break;
                    };
                    if succ == d {
                        break; // self-loop guard
                    }

                    // Don't try "after x" while x may still sit somewhere
                    if d != x {
                        // trial = current committed delta + one insertion
                        let cur_delta = builder.delta().clone();
                        let mut trial_b = ChainSetDeltaBuilder::new_with_delta(cs, cur_delta);
                        if trial_b.insert_after(d, x) {
                            // Build view and reset chain variables
                            let ov_trial = ChainSetOverlay::new(cs, trial_b.delta());
                            let cview = ov_trial.chain(ci);

                            // Fresh copies for scheduling
                            let mut iv_s = ivars.to_vec();
                            let mut dv_s = dvars.to_vec();

                            // Reset all requests in this chain view
                            {
                                let mut n_opt = cview.first_real_node(cview.start());
                                let mut guard = model.flexible_requests_len().saturating_add(2);
                                while let Some(n) = n_opt {
                                    if guard == 0 || n == cview.end() {
                                        break;
                                    }
                                    let i = crate::model::index::RequestIndex(n.get()).get();
                                    dv_s[i] = DecisionVar::Unassigned;
                                    let w = model.feasible_intervals()[i];
                                    iv_s[i].start_time_lower_bound = w.start();
                                    iv_s[i].start_time_upper_bound = w.end();

                                    let next = cview.next_real(n);
                                    if next == Some(n) {
                                        break;
                                    }
                                    n_opt = next;
                                    guard = guard.saturating_sub(1);
                                }
                            }

                            // Also reset the moved node explicitly
                            {
                                let xi = crate::model::index::RequestIndex(x.get()).get();
                                let w = model.feasible_intervals()[xi];
                                dv_s[xi] = DecisionVar::Unassigned;
                                iv_s[xi].start_time_lower_bound = w.start();
                                iv_s[xi].start_time_upper_bound = w.end();
                            }

                            // Validate with the scheduler
                            let ok = self
                                .scheduler
                                .schedule_chain_slice(
                                    model,
                                    cview,
                                    cview.start(),
                                    None,
                                    &mut iv_s,
                                    &mut dv_s,
                                )
                                .is_ok();

                            if ok {
                                builder.replace_delta(trial_b.build());
                                placed = true;
                                break 'chains;
                            }
                        }
                    }

                    if succ == end {
                        break;
                    }
                    d = succ;
                }
            }

            // If not placed, keep x isolated (x->x) on purpose.
            if !placed {
                // already isolated due to RUIN step
            }
        }

        // Always return a delta (could be empty, or with isolated nodes, or with committed insertions).
        builder.build()
    }
}
