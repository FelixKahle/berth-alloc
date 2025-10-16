// Copyright (c) 2025 Felix Kahle.
// MIT License

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
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// A large-scale perturbation that "nukes" a single chain and repairs it.
///
/// This perturbation follows a more disruptive ruin-and-repair pattern:
/// 1.  **RUIN**: Randomly select one non-empty chain from the solution. Detach all
///     of its assigned requests, effectively emptying it and creating a large pool
///     of unassigned nodes.
/// 2.  **REPAIR**: Use a "best-insertion" strategy to find the optimal new position
///     for each of the removed requests. These requests can be re-inserted anywhere
///     in the solution, including other chains or back into the original chain.
///
/// This provides a powerful "kick" to escape deep local optima by completely
/// restructuring a significant part of the solution.
#[derive(Debug, Clone)]
pub struct NukeChainRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    /// Optional hard cap on the number of (chain, gap) positions scanned per removed node.
    pub scan_cap: Option<usize>,
    /// Scheduler to validate placements.
    pub scheduler: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, S> NukeChainRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub fn new(scan_cap: Option<usize>, scheduler: S) -> Self {
        Self {
            scan_cap,
            scheduler,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S> Perturbation<T> for NukeChainRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    fn name(&self) -> &str {
        "NukeChainRuinRepair"
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

        // 1) RUIN: Find all non-empty chains and pick one to nuke.
        let non_empty_chains: Vec<ChainIndex> = (0..cs.num_chains())
            .map(ChainIndex)
            .filter(|&ci| !cs.is_chain_empty(ci))
            .collect();

        if non_empty_chains.is_empty() {
            return ChainSetDelta::new(); // Nothing to do
        }

        let chain_to_nuke_idx = rng.random_range(0..non_empty_chains.len());
        let chain_to_nuke = non_empty_chains[chain_to_nuke_idx];

        // Collect all real nodes from the selected chain.
        let mut nodes_to_remove = Vec::new();
        let mut u = cs.start_of_chain(chain_to_nuke);
        let end_node = cs.end_of_chain(chain_to_nuke);
        while let Some(v) = cs.next_node(u) {
            if v == end_node {
                break;
            }
            nodes_to_remove.push(v);
            u = v;
        }

        if nodes_to_remove.is_empty() {
            return ChainSetDelta::new(); // Should not happen due to filter, but as a guard.
        }

        // Detach all nodes from the nuked chain.
        let mut builder = ChainSetDeltaBuilder::new(cs);
        for &node in &nodes_to_remove {
            builder.detach(node);
        }

        // 2) REPAIR: For each detached node, find its best possible insertion point.
        for &x in &nodes_to_remove {
            let mut best_pos: Option<(NodeIndex, NodeIndex)> = None;
            let mut min_cost = Cost::MAX;
            let mut scanned = 0usize;

            let ov_now = ChainSetOverlay::new(cs, builder.delta());

            'chains: for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let mut u = cs.start_of_chain(ci);

                loop {
                    if let Some(cap) = self.scan_cap {
                        if scanned >= cap {
                            break 'chains;
                        }
                    }
                    scanned += 1;

                    let v = ov_now.next_node(u).unwrap_or(cs.end_of_chain(ci));
                    let insertion_cost = arc_eval(u, x).unwrap_or(Cost::MAX);

                    if insertion_cost < min_cost {
                        let mut trial_b =
                            ChainSetDeltaBuilder::new_with_delta(cs, builder.delta().clone());
                        if trial_b.insert_after(u, x) {
                            let ov_trial = ChainSetOverlay::new(cs, trial_b.delta());
                            let cview = ov_trial.chain(ci);

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
                                min_cost = insertion_cost;
                                best_pos = Some((u, x));
                            }
                        }
                    }

                    if v == cs.end_of_chain(ci) {
                        break;
                    }
                    u = v;
                }
            }

            if let Some((u, x_node)) = best_pos {
                builder.insert_after(u, x_node);
            }
        }

        builder.build()
    }
}
