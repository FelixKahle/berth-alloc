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

/// Best-insertion ruin & repair perturbation.
///
/// This perturbation follows a classic ruin-and-repair pattern:
/// 1.  **RUIN**: Randomly select `remove_k` assigned requests (nodes) from the solution
///     and detach them, placing them in an "unassigned" pool.
/// 2.  **REPAIR**: For each unassigned request, evaluate every possible insertion point
///     across all chains. The insertion point that results in the lowest objective cost
///     increase (i.e., the "best" insertion) is chosen.
///
/// This is generally more effective than a greedy first-fit repair as it makes more
/// informed decisions, leading to higher-quality neighboring solutions.
#[derive(Debug, Clone)]
pub struct BestInsertionRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub remove_k: usize,
    /// Optional hard cap on the number of (chain, gap) positions scanned per removed node.
    pub scan_cap: Option<usize>,
    /// Scheduler to validate placements.
    pub scheduler: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, S> BestInsertionRuinRepair<T, S>
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

impl<T, S> Perturbation<T> for BestInsertionRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    fn name(&self) -> &str {
        "BestInsertionRuinRepair"
    }

    fn apply<'s, 'm, 'p>(
        &self,
        search_state: &'s SolverSearchState<'m, 'p, T>,
        arc_eval: &ArcEvaluator,
        ivars: &'m mut [IntervalVar<T>],
        dvars: &'m mut [DecisionVar<T>],
        rng: &mut ChaCha8Rng,
    ) -> ChainSetDelta {
        let cs: &ChainSet = search_state.chain_set();
        let model = search_state.model();

        // 1) RUIN: Gather all assigned nodes, shuffle, and pick K to remove.
        // CORRECTED: Manually iterate through chains to collect real nodes.
        let mut real_nodes: Vec<NodeIndex> = Vec::with_capacity(model.flexible_requests_len());
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let mut u = cs.start_of_chain(ci);
            let end_node = cs.end_of_chain(ci);

            // Use a guard to prevent infinite loops on broken chain structures
            let mut guard = cs.num_nodes() + 2;
            while let Some(v) = cs.next_node(u) {
                if v == end_node || guard == 0 {
                    break;
                }
                real_nodes.push(v);
                u = v;
                guard -= 1;
            }
        }

        if real_nodes.is_empty() {
            return ChainSetDelta::new();
        }

        real_nodes.shuffle(rng);
        let k = self.remove_k.min(real_nodes.len());
        let picked = &real_nodes[..k];

        // Detach all picked nodes first.
        let mut builder = ChainSetDeltaBuilder::new(cs);
        for &x in picked {
            builder.detach(x);
        }

        // 2) REPAIR: For each detached node, find its best possible insertion point.
        for &x in picked {
            let mut best_pos: Option<(NodeIndex, NodeIndex)> = None;
            let mut min_cost = Cost::MAX;
            let mut scanned = 0usize;

            // Get a view of the solution *with all previous insertions committed*.
            let ov_now = ChainSetOverlay::new(cs, builder.delta());

            'chains: for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let mut u = cs.start_of_chain(ci);

                loop {
                    // Check scan cap
                    if let Some(cap) = self.scan_cap {
                        if scanned >= cap {
                            break 'chains;
                        }
                    }
                    scanned += 1;

                    let v = ov_now.next_node(u).unwrap_or(cs.end_of_chain(ci));

                    // Evaluate inserting 'x' between 'u' and 'v'
                    let insertion_cost = arc_eval(u, x).unwrap_or(Cost::MAX);

                    if insertion_cost < min_cost {
                        // This position is promising, now check if it's feasible
                        let mut trial_b =
                            ChainSetDeltaBuilder::new_with_delta(cs, builder.delta().clone());
                        if trial_b.insert_after(u, x) {
                            let ov_trial = ChainSetOverlay::new(cs, trial_b.delta());
                            let cview = ov_trial.chain(ci);

                            // Schedule the modified chain to check for validity.
                            if self
                                .scheduler
                                .schedule_chain_slice(
                                    model,
                                    cview,
                                    cview.start(),
                                    None,
                                    ivars,
                                    dvars,
                                )
                                .is_ok()
                            {
                                // Feasible and better than the best-so-far!
                                min_cost = insertion_cost;
                                best_pos = Some((u, x));
                            }
                        }
                    }

                    // Move to the next position in the chain
                    if v == cs.end_of_chain(ci) {
                        break;
                    }
                    u = v;
                }
            }

            // After checking all positions, commit the best one found for node 'x'.
            if let Some((u, x_node)) = best_pos {
                builder.insert_after(u, x_node);
            }
            // If no feasible position was found, 'x' remains detached (unassigned).
        }

        builder.build()
    }
}
