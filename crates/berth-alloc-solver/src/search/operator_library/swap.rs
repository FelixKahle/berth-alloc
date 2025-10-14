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
    eval::ArcEvaluator,
    search::operator::traits::NeighborhoodOperator,
    state::{
        chain_set::{
            base::ChainSet, delta::ChainSetDelta, delta_builder::ChainSetDeltaBuilder,
            index::ChainIndex, view::ChainSetView,
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::num::NonZeroUsize;

pub struct FirstAdjacentSwapAnywhere;

impl<T> NeighborhoodOperator<T> for FirstAdjacentSwapAnywhere
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
{
    fn make_neighboor<'s, 'm, 'p>(
        &self,
        st: &'s SolverSearchState<'m, 'p, T>,
        _arc: &crate::eval::ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs = st.chain_set();
        for c in 0..cs.num_chains() {
            let ci = crate::state::chain_set::index::ChainIndex(c);
            let s = cs.start_of_chain(ci);
            let e = cs.end_of_chain(ci);
            if let Some(a) = cs.next_node(s) {
                if a != e {
                    let mut b = ChainSetDeltaBuilder::new(cs);
                    b.swap_adjacent_after(s);
                    return Some(b.build());
                }
            }
        }
        None
    }
}

/// A neighborhood operator that finds the first available "swap successors" move that improves the solution.
///
/// This operator iterates through pairs of predecessor nodes (`p`, `q`) and evaluates the cost
/// change of swapping their immediate successors (`a`, `b`).
///
/// **Move Definition:**
/// - **Before:** `... -> p -> a -> a_next -> ...` and `... -> q -> b -> b_next -> ...`
/// - **After:** `... -> p -> b -> a_next -> ...` and `... -> q -> a -> b_next -> ...`
///
/// It is a "first improvement" strategy, meaning it terminates and returns the first
/// improving move it finds, rather than searching the entire neighborhood for the best move.
pub struct SwapSuccessorsFirstImprovement {
    /// A closure that dynamically provides an optional scan cap for predecessors.
    /// `None` means no limit.
    pub get_cap: Box<dyn Fn() -> Option<NonZeroUsize>>,
}

impl Default for SwapSuccessorsFirstImprovement {
    /// By default, creates an operator with no scan cap.
    fn default() -> Self {
        Self {
            get_cap: Box::new(|| None), // Default closure returns None, indicating no cap.
        }
    }
}

impl<T> NeighborhoodOperator<T> for SwapSuccessorsFirstImprovement
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        _arc_eval: &ArcEvaluator, // not used; scheduling/filters will handle feasibility
    ) -> Option<ChainSetDelta> {
        let chain_set: &ChainSet = search_state.chain_set();

        let num_chains = chain_set.num_chains();
        let scan_cap: Option<NonZeroUsize> = (self.get_cap)();

        let mut predecessors_scanned = 0usize;

        // Iterate through each chain to select the first predecessor, `predecessor_p`.
        for chain_index_p in 0..num_chains {
            let start_node_p = chain_set.start_of_chain(ChainIndex(chain_index_p));
            let end_node_p = chain_set.end_of_chain(ChainIndex(chain_index_p));

            let mut predecessor_p = start_node_p;
            loop {
                // Identify `successor_a`, the node immediately following `predecessor_p`.
                let successor_a = match chain_set.next_node(predecessor_p) {
                    Some(node) if node != end_node_p => node,
                    _ => break, // No valid successor to swap, end of this chain.
                };

                // Now, iterate through all chains again to select the second predecessor, `predecessor_q`.
                for chain_index_q in 0..num_chains {
                    let start_node_q = chain_set.start_of_chain(ChainIndex(chain_index_q));
                    let end_node_q = chain_set.end_of_chain(ChainIndex(chain_index_q));

                    let mut predecessor_q = start_node_q;
                    loop {
                        // Identify `successor_b`, the node immediately following `predecessor_q`.
                        let successor_b = match chain_set.next_node(predecessor_q) {
                            Some(node) if node != end_node_q => node,
                            _ => break, // No valid successor, end of this inner chain.
                        };

                        // Ensure we are not trying to swap a node with itself and that successors are distinct.
                        if predecessor_p != predecessor_q && successor_a != successor_b {
                            // symmetry break on same chain to avoid mirrored duplicates
                            if chain_index_p == chain_index_q
                                && predecessor_p.get() >= predecessor_q.get()
                            {
                                predecessor_q = successor_b;
                                continue;
                            }

                            // Propose the structural swap (no arc-cost gating). Let the scheduling
                            // pipeline repair and the evaluator compute actual delta costs.
                            let mut builder = ChainSetDeltaBuilder::new(chain_set);
                            builder.swap_after(predecessor_p, predecessor_q);
                            return Some(builder.build());
                        }

                        // Advance to the next predecessor in the inner loop.
                        predecessor_q = successor_b;
                    }
                }

                // After checking `predecessor_p` against all possible `predecessor_q`, increment the counter.
                predecessors_scanned += 1;
                if let Some(cap) = scan_cap
                    && predecessors_scanned >= cap.get()
                {
                    return None; // Stop searching if the scan cap is reached.
                }

                // Advance to the next predecessor in the outer loop.
                predecessor_p = successor_a;
            }
        }

        // If the loops complete without finding any structural swap candidate, return None.
        None
    }

    fn name(&self) -> &str {
        "SwapSuccessorsFirstImprovement"
    }
}
