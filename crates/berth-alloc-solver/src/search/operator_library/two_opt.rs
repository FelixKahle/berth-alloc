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
            base::ChainSet,
            delta::ChainSetDelta,
            delta_builder::ChainSetDeltaBuilder,
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::num::NonZeroUsize;

/// A neighborhood operator that performs a 2-Opt move to improve a solution.
///
/// This is a classic local search operator primarily for intra-chain optimization. It works
/// by selecting two non-adjacent edges in a chain, removing them, and reconnecting the
/// two resulting paths in the only other possible way. This is equivalent to reversing
/// the segment of the chain between the two broken edges.
///
/// **Move Definition:**
/// - **Before:** `... -> p -> a -> ... -> q -> b -> ...`
/// - **After:** `... -> p -> q -> [reversed segment from a to q] -> a -> b -> ...`
///
/// It is a "first improvement" strategy, returning immediately once an improving move is found.
#[allow(dead_code, clippy::type_complexity)]
pub struct TwoOpt<T: Copy + Ord + CheckedAdd + CheckedSub> {
    /// A closure that selects which chains are promising candidates for untangling.
    /// If `None`, all chains will be considered.
    pub chain_selector: Option<Box<dyn Fn(&SolverSearchState<'_, '_, T>) -> Vec<ChainIndex>>>,

    /// An optional cap on the number of 2-Opt moves to evaluate per chain.
    pub get_cap_per_chain: Box<dyn Fn() -> Option<NonZeroUsize>>,
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub> Default for TwoOpt<T> {
    fn default() -> Self {
        Self {
            chain_selector: None,
            get_cap_per_chain: Box::new(|| None),
        }
    }
}

impl<T> TwoOpt<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    /// Creates a new `TwoOpt` operator with the specified chain selector and evaluation cap.
    ///
    /// # Arguments
    /// * `chain_selector` - An optional closure to select which chains to consider.
    /// * `get_cap_per_chain` - A closure that provides an optional cap on evaluations per chain.
    pub fn new(
        chain_selector: Option<Box<dyn Fn(&SolverSearchState<'_, '_, T>) -> Vec<ChainIndex>>>,
        get_cap_per_chain: Box<dyn Fn() -> Option<NonZeroUsize>>,
    ) -> Self {
        Self {
            chain_selector,
            get_cap_per_chain,
        }
    }
}

impl<T> NeighborhoodOperator<T> for TwoOpt<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let chain_set: &ChainSet = search_state.chain_set();
        let arc_cost = |u: NodeIndex, v: NodeIndex| -> Option<Cost> { arc_eval(u, v) };
        let cap_per_chain = (self.get_cap_per_chain)();

        // Determine which chains to scan.
        let chains_to_scan = match &self.chain_selector {
            Some(selector) => selector(search_state),
            None => (0..chain_set.num_chains()).map(ChainIndex).collect(),
        };

        // Iterate through each candidate chain.
        for chain_index in chains_to_scan {
            let mut moves_evaluated = 0;
            let start_node = chain_set.start_of_chain(chain_index);
            let end_node = chain_set.end_of_chain(chain_index);

            // Select the first edge to break: (predecessor_p -> successor_a)
            let mut predecessor_p = start_node;
            while let Some(successor_a) = chain_set.next_node(predecessor_p) {
                if successor_a == end_node {
                    break;
                }

                // Select the second edge to break: (predecessor_q -> successor_b)
                // It must come after the first edge.
                let mut predecessor_q = successor_a;
                while let Some(successor_b) = chain_set.next_node(predecessor_q) {
                    if successor_b == end_node {
                        break;
                    }

                    // We have two edges to break: (p -> a) and (q -> b)
                    // Cost before: cost(p -> a) + cost(q -> b)
                    // Cost after:  cost(p -> q) + cost(a -> b)
                    let old_cost = match (
                        arc_cost(predecessor_p, successor_a),
                        arc_cost(predecessor_q, successor_b),
                    ) {
                        (Some(c1), Some(c2)) => c1.saturating_add(c2),
                        _ => {
                            // If an original arc is invalid, we can't perform this move.
                            predecessor_q = successor_b;
                            continue;
                        }
                    };

                    if let (Some(new_c1), Some(new_c2)) = (
                        arc_cost(predecessor_p, predecessor_q),
                        arc_cost(successor_a, successor_b),
                    ) {
                        let new_cost = new_c1.saturating_add(new_c2);

                        // If we found an improvement, build the delta and return.
                        if new_cost < old_cost {
                            let mut builder = ChainSetDeltaBuilder::new(chain_set);
                            // This operation reconnects p->q, reverses the segment a..q, then reconnects a->b.
                            builder.two_opt(predecessor_p, predecessor_q);
                            return Some(builder.build());
                        }
                    }

                    // Check if we have exceeded the evaluation cap for this chain.
                    moves_evaluated += 1;
                    if let Some(cap) = cap_per_chain
                        && moves_evaluated >= cap.get()
                    {
                        break;
                    }

                    predecessor_q = successor_b;
                }
                if let Some(cap) = cap_per_chain
                    && moves_evaluated >= cap.get()
                {
                    break;
                }
                predecessor_p = successor_a;
            }
        }

        None
    }

    fn name(&self) -> &str {
        "TwoOpt"
    }
}
