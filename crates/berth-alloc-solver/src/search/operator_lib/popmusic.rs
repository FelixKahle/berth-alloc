// Copyright (c) 2025 Felix Kahle.
//
// MIT license header…

use crate::{
    eval::arc_evaluator::ArcEvaluator,
    search::operator::{
        context::OperatorContext,
        traits::{NeighborhoodOperator, OperatorTask},
    },
    state::chain_set::{
        delta::ChainSetDelta,
        delta_builder::ChainSetDeltaBuilder,
        index::{ChainIndex, NodeIndex},
        overlay::ChainSetOverlay,
        view::ChainSetView,
    },
    state::search_state::SolverSearchState,
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};

/// A powerful local search operator that improves a solution using the POPMUSIC heuristic.
///
/// This operator works by relocating a contiguous block of requests from a "source"
/// position to a "destination" position to find a better overall schedule. It is:
/// - **Deterministic**: Given the same input, it will always produce the same output.
/// - **Intensifying**: It focuses on a small part of the problem (a "window" of chains)
///   and tries to improve it greedily before moving on.
/// - **Pruned**: It uses a clever "cap rule" inspired by Google's OR-Tools to avoid
///   evaluating moves that are unlikely to be beneficial, drastically speeding up the search.
///
/// ## Parameters
/// - `window_size`: The number of adjacent chains to consider in a single optimization task.
///   A larger window allows for more complex moves but is computationally slower.
/// - `max_greedy_iterations`: The maximum number of improving moves to make within a single window
///   before stopping.
/// - `sample_per_chain`: To speed up the search, this can be set to a non-zero value to only
///   consider a subset of all possible start/end points for a block move. `0` means all points are considered.
#[derive(Debug, Clone)]
pub struct PopmusicRelocate {
    pub window_size: usize,
    pub max_greedy_iterations: usize,
    pub sample_per_chain: usize,
}

impl Default for PopmusicRelocate {
    fn default() -> Self {
        Self {
            window_size: 2,
            max_greedy_iterations: 32,
            sample_per_chain: 0,
        }
    }
}

impl<'eval, 'state, 'model, 'problem, T, A>
    NeighborhoodOperator<'eval, 'state, 'model, 'problem, T, A> for PopmusicRelocate
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
    A: ArcEvaluator<T>,
{
    fn name(&self) -> &str {
        "PopmusicRelocate"
    }

    /// Creates a "plan" of work by dividing the problem into smaller, independent tasks.
    ///
    /// This method slides a window across all the chains in the problem. For each
    /// position of the window, it creates a `PopmusicTask`. This design is ideal
    /// for parallel execution, as each task can be run on a separate CPU core.
    fn plan(
        &self,
        state: &'state SolverSearchState<'model, 'problem, T>,
    ) -> Vec<Box<dyn OperatorTask<'eval, 'state, 'model, 'problem, T, A>>> {
        let num_chains = state.chain_set().num_chains();
        if num_chains == 0 {
            return Vec::new();
        }
        let window_size = self.window_size.max(1).min(num_chains);
        let mut tasks: Vec<Box<dyn OperatorTask<'eval, 'state, 'model, 'problem, T, A>>> =
            Vec::with_capacity(num_chains);

        // Create one task for each possible starting point of the sliding window.
        for start_chain_index in 0..num_chains {
            tasks.push(Box::new(PopmusicTask {
                start_chain_index,
                window_size,
                max_greedy_iterations: self.max_greedy_iterations,
                sample_per_chain: self.sample_per_chain,
            }));
        }
        tasks
    }
}

/// Represents a single, self-contained optimization task for a specific window of chains.
#[derive(Debug, Clone)]
struct PopmusicTask {
    start_chain_index: usize,
    window_size: usize,
    max_greedy_iterations: usize,
    sample_per_chain: usize,
}

impl<'eval, 'state, 'model, 'problem, T, A> OperatorTask<'eval, 'state, 'model, 'problem, T, A>
    for PopmusicTask
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
    A: ArcEvaluator<T>,
{
    fn name(&self) -> &str {
        "PopmusicRelocate::Task"
    }

    fn key(&self) -> u64 {
        ((self.start_chain_index as u64) << 32)
            ^ ((self.window_size as u64) << 16)
            ^ (self.max_greedy_iterations as u64)
    }

    /// Executes the POPMUSIC heuristic for the task's assigned window.
    fn run(
        &self,
        ctx: &OperatorContext<'eval, 'state, 'model, 'problem, T, A>,
    ) -> Option<ChainSetDelta> {
        let search_state = ctx.search_state();
        let base_chain_set = search_state.chain_set();
        let num_chains = base_chain_set.num_chains();
        if num_chains == 0 {
            return None;
        }

        let window_size = self.window_size.min(num_chains);
        // Define the set of chains to operate on, wrapping around if necessary.
        let chains_in_window: Vec<ChainIndex> = (0..window_size)
            .map(|offset| ChainIndex((self.start_chain_index + offset) % num_chains))
            .collect();

        // This builder will accumulate all the moves we decide to make.
        let mut delta_builder = ChainSetDeltaBuilder::new(base_chain_set);
        let mut any_improvement_found = false;

        // --- Main Greedy Improvement Loop ---
        // In each iteration, we find the single best move within the window and apply it
        // to our temporary delta. The next iteration then works from this improved state.
        for _ in 0..self.max_greedy_iterations {
            // A temporary struct to hold information about the best move found in one full scan.
            #[derive(Clone, Copy)]
            struct BestMove {
                gain: Cost,
                source_predecessor: NodeIndex,
                source_block_end: NodeIndex,
                destination_predecessor: NodeIndex,
            }
            let mut best_move_in_iteration: Option<BestMove> = None;

            // The overlay view allows us to see the "current state" of our solution,
            // including all the changes we've made so far in the delta_builder.
            // We use a scoped block to ensure the overlay is dropped before we mutate the delta_builder again.
            {
                let overlay_view = ChainSetOverlay::new(base_chain_set, delta_builder.delta());

                // Helper to downsample the number of positions we check, for performance.
                let thin_out_nodes =
                    |nodes: Vec<NodeIndex>, max_nodes_to_keep: usize| -> Vec<NodeIndex> {
                        if max_nodes_to_keep == 0 || nodes.len() <= max_nodes_to_keep {
                            return nodes;
                        }
                        // Select a subset of nodes, always including the first, and then
                        // picking others at evenly spaced intervals.
                        let mut sampled_nodes = Vec::with_capacity(max_nodes_to_keep);
                        sampled_nodes.push(nodes[0]);
                        let span = nodes.len().saturating_sub(1);
                        for i in 1..max_nodes_to_keep {
                            let position = (i * span) / (max_nodes_to_keep - 1);
                            sampled_nodes.push(nodes[position]);
                        }
                        sampled_nodes.sort_unstable_by_key(|node| node.get());
                        sampled_nodes.dedup();
                        sampled_nodes
                    };

                // --- Nested Search for the Best Relocation ---
                // 1. Iterate through all chains in our window as potential sources.
                for &source_chain_index in &chains_in_window {
                    let mut source_predecessor_nodes: Vec<NodeIndex> = Vec::with_capacity(16);
                    source_predecessor_nodes
                        .push(base_chain_set.start_of_chain(source_chain_index));
                    for node in overlay_view.iter_chain(source_chain_index) {
                        source_predecessor_nodes.push(node);
                    }
                    if self.sample_per_chain > 0 {
                        source_predecessor_nodes =
                            thin_out_nodes(source_predecessor_nodes, self.sample_per_chain.max(2));
                    }

                    // 2. Iterate through all possible start points for a block to be moved.
                    for &source_predecessor_node in &source_predecessor_nodes {
                        // The block to be moved starts with the node immediately after the predecessor.
                        let Some(first_node_in_block) = overlay_view
                            .next_node(source_predecessor_node)
                            .filter(|&n| !base_chain_set.is_sentinel_node(n))
                        else {
                            continue;
                        };

                        // 3. Iterate through all chains as potential destinations.
                        for &destination_chain_index in &chains_in_window {
                            let mut destination_predecessor_nodes: Vec<NodeIndex> =
                                Vec::with_capacity(16);
                            destination_predecessor_nodes
                                .push(base_chain_set.start_of_chain(destination_chain_index));
                            for node in overlay_view.iter_chain(destination_chain_index) {
                                destination_predecessor_nodes.push(node);
                            }
                            if self.sample_per_chain > 0 {
                                destination_predecessor_nodes = thin_out_nodes(
                                    destination_predecessor_nodes,
                                    self.sample_per_chain.max(2),
                                );
                            }

                            // 4. Iterate through all possible insertion points.
                            for &destination_predecessor_node in &destination_predecessor_nodes {
                                if destination_predecessor_node == source_predecessor_node {
                                    continue;
                                }
                                let Some(destination_insertion_node) =
                                    overlay_view.next_node(destination_predecessor_node)
                                else {
                                    continue;
                                };
                                if destination_insertion_node == first_node_in_block {
                                    continue;
                                }

                                // --- The "Cap Rule" Heuristic ---
                                // To prune the search, we calculate the cost of the connection we would break at the
                                // destination. This cost becomes a "cap" that limits the size of the block we consider moving.
                                let Some(cost_cap) = ctx.eval().evaluate(
                                    ctx.model(),
                                    overlay_view.chain(destination_chain_index),
                                    destination_predecessor_node,
                                    first_node_in_block,
                                ) else {
                                    continue;
                                };

                                // Greedily extend the block to be moved as long as the connections within it are cheaper than the cap.
                                let mut last_node_in_block = first_node_in_block;
                                loop {
                                    let next_node_after_block =
                                        overlay_view.next_node(last_node_in_block).unwrap();
                                    if base_chain_set.is_sentinel_node(next_node_after_block)
                                        || next_node_after_block == destination_insertion_node
                                    {
                                        break;
                                    }

                                    match ctx.eval().evaluate(
                                        ctx.model(),
                                        overlay_view.chain(source_chain_index),
                                        last_node_in_block,
                                        next_node_after_block,
                                    ) {
                                        Some(arc_cost) if arc_cost <= cost_cap => {
                                            last_node_in_block = next_node_after_block; // Extend the block
                                        }
                                        _ => break, // Stop extending
                                    }
                                }

                                // A block cannot be moved in a way that its destination is one of the nodes within itself.
                                let mut block_crosses_destination = false;
                                let mut node_in_block = first_node_in_block;
                                while node_in_block != last_node_in_block {
                                    if node_in_block == destination_predecessor_node {
                                        block_crosses_destination = true;
                                        break;
                                    }
                                    node_in_block = overlay_view.next_node(node_in_block).unwrap();
                                }
                                if block_crosses_destination {
                                    continue;
                                }

                                // This is a no-op (moving a block to the same place it already is).
                                if source_chain_index == destination_chain_index
                                    && destination_predecessor_node == last_node_in_block
                                {
                                    continue;
                                }

                                // --- Calculate the Cost Change (Gain) of the Move ---
                                // We calculate the cost of the 4 affected connections *before* the move
                                // and compare it to the cost of the 3 new connections *after* the move.
                                //
                                // Before: ... -> src_pred -> [block] -> after_block -> ...
                                //         ... -> dst_pred -> dest_node -> ...
                                // After:  ... -> src_pred -> after_block -> ...
                                //         ... -> dst_pred -> [block] -> dest_node -> ...

                                let node_after_moved_block =
                                    overlay_view.next_node(last_node_in_block).unwrap();
                                let Some(cost_src_old_1) = ctx.eval().evaluate(
                                    ctx.model(),
                                    overlay_view.chain(source_chain_index),
                                    source_predecessor_node,
                                    first_node_in_block,
                                ) else {
                                    continue;
                                };
                                let cost_src_old_2 =
                                    if base_chain_set.is_sentinel_node(node_after_moved_block) {
                                        0
                                    } else {
                                        ctx.eval()
                                            .evaluate(
                                                ctx.model(),
                                                overlay_view.chain(source_chain_index),
                                                last_node_in_block,
                                                node_after_moved_block,
                                            )
                                            .unwrap_or(Cost::MAX)
                                    };
                                let cost_src_new =
                                    if base_chain_set.is_sentinel_node(node_after_moved_block) {
                                        0
                                    } else {
                                        ctx.eval()
                                            .evaluate(
                                                ctx.model(),
                                                overlay_view.chain(source_chain_index),
                                                source_predecessor_node,
                                                node_after_moved_block,
                                            )
                                            .unwrap_or(Cost::MAX)
                                    };
                                let cost_dst_old = if base_chain_set
                                    .is_sentinel_node(destination_insertion_node)
                                {
                                    0
                                } else {
                                    ctx.eval()
                                        .evaluate(
                                            ctx.model(),
                                            overlay_view.chain(destination_chain_index),
                                            destination_predecessor_node,
                                            destination_insertion_node,
                                        )
                                        .unwrap_or(Cost::MAX)
                                };
                                let Some(cost_dst_new_1) = ctx.eval().evaluate(
                                    ctx.model(),
                                    overlay_view.chain(destination_chain_index),
                                    destination_predecessor_node,
                                    first_node_in_block,
                                ) else {
                                    continue;
                                };
                                let cost_dst_new_2 = if base_chain_set
                                    .is_sentinel_node(destination_insertion_node)
                                {
                                    0
                                } else {
                                    ctx.eval()
                                        .evaluate(
                                            ctx.model(),
                                            overlay_view.chain(destination_chain_index),
                                            last_node_in_block,
                                            destination_insertion_node,
                                        )
                                        .unwrap_or(Cost::MAX)
                                };

                                // We want to minimize cost, so a negative gain is an improvement.
                                let gain = (cost_src_new + cost_dst_new_1 + cost_dst_new_2)
                                    - (cost_src_old_1 + cost_src_old_2 + cost_dst_old);

                                // If this move is the best improvement we've seen so far, record it.
                                if gain < 0 && best_move_in_iteration.is_none_or(|b| gain < b.gain)
                                {
                                    best_move_in_iteration = Some(BestMove {
                                        gain,
                                        source_predecessor: source_predecessor_node,
                                        source_block_end: last_node_in_block,
                                        destination_predecessor: destination_predecessor_node,
                                    });
                                }
                            }
                        }
                    }
                }
            } // The overlay_view is dropped here, releasing its borrow on the delta_builder.

            // After scanning all possibilities, if we found an improving move, apply it.
            if let Some(best_move) = best_move_in_iteration {
                delta_builder.move_block_after(
                    best_move.destination_predecessor,
                    best_move.source_predecessor,
                    best_move.source_block_end,
                );
                any_improvement_found = true;
            } else {
                // No improvement found in this iteration, so further iterations won't help.
                break;
            }
        }

        if any_improvement_found {
            Some(delta_builder.build())
        } else {
            None
        }
    }
}
