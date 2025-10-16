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

/// A perturbation that walks all chains and removes nodes with a given probability.
///
/// This perturbation provides a balanced approach to ruin and repair:
/// 1.  **RUIN**: It iterates through every single assigned request (node) in the entire
///     solution. For each node, it performs a random check. With a probability of
///     `removal_prob`, the node is detached and added to an "unassigned" pool.
/// 2.  **REPAIR**: After the walk is complete, it uses a "best-insertion" strategy
///     to find the optimal new position for each of the removed requests.
///
/// This method is less disruptive than nuking an entire chain but more extensive than
/// removing a small, fixed number of random nodes. It's effective at creating diverse
/// neighboring solutions by partially deconstructing multiple chains simultaneously.
#[derive(Debug, Clone)]
pub struct RandomWalkRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    /// The probability (0.0 to 1.0) of removing any given node during the walk.
    pub removal_prob: f64,
    /// Optional hard cap on the number of (chain, gap) positions scanned per removed node.
    pub scan_cap: Option<usize>,
    /// Scheduler to validate placements.
    pub scheduler: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, S> RandomWalkRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub fn new(removal_prob: f64, scan_cap: Option<usize>, scheduler: S) -> Self {
        Self {
            removal_prob: removal_prob.clamp(0.0, 1.0),
            scan_cap,
            scheduler,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S> Perturbation<T> for RandomWalkRuinRepair<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    fn name(&self) -> &str {
        "RandomWalkRuinRepair"
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

        // 1) RUIN: Walk all chains and remove nodes based on probability.
        let mut nodes_to_remove = Vec::new();
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let mut u = cs.start_of_chain(ci);
            let end_node = cs.end_of_chain(ci);

            while let Some(v) = cs.next_node(u) {
                if v == end_node {
                    break;
                }
                // For each real node, check if we should remove it.
                if rng.random_bool(self.removal_prob) {
                    nodes_to_remove.push(v);
                }
                u = v;
            }
        }

        if nodes_to_remove.is_empty() {
            return ChainSetDelta::new();
        }

        // Detach all selected nodes.
        let mut builder = ChainSetDeltaBuilder::new(cs);
        for &node in &nodes_to_remove {
            builder.detach(node);
        }

        // 2) REPAIR: For each detached node, find its best possible insertion point.
        // This repair logic is identical to the other advanced perturbations.
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
