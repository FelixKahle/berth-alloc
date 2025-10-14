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

// src/search/operator/two_opt_intra_first_improvement.rs

use crate::{
    eval::ArcEvaluator,
    search::operator::traits::NeighborhoodOperator,
    state::chain_set::{
        base::ChainSet,
        delta::ChainSetDelta,
        delta_builder::ChainSetDeltaBuilder,
        index::{ChainIndex, NodeIndex},
        view::ChainSetView,
    },
    state::search_state::SolverSearchState,
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::num::NonZeroUsize;

/// 2-opt on a single chain; first-improvement.
/// Reverses (a..q) by breaking (p->a) and (q->b), adding (p->q) and (a->b).
///
/// Hooks:
/// - `get_cap`: early stop after scanning this many (p,q) predecessor-pairs
/// - `get_q_neighbors`: given p (NodeIndex), return candidate q’s to try (neighborhood pruning).
pub struct IntraChainTwoOptFirstImprovement {
    pub get_cap: Box<dyn Fn() -> Option<NonZeroUsize> + Send + Sync>,
    /// Given `p`, return a candidate list of `q` on *the same chain* (owned, for easy lifetimes).
    pub get_q_neighbors: Box<dyn Fn(NodeIndex, &ChainSet) -> Vec<NodeIndex> + Send + Sync>,
}

impl Default for IntraChainTwoOptFirstImprovement {
    fn default() -> Self {
        Self {
            get_cap: Box::new(|| None),
            get_q_neighbors: Box::new(|_p, _cs| Vec::new()), // empty => fallback to full scan
        }
    }
}

impl<T> NeighborhoodOperator<T> for IntraChainTwoOptFirstImprovement
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        _arc_eval: &ArcEvaluator, // not used; scheduling/filters handle feasibility
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        let cap = (self.get_cap)();
        let mut scanned = 0usize;

        // Walk each chain; try 2-opt pairs on that chain only.
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);

            // Enumerate predecessors p along the chain
            let mut p = start;
            loop {
                let a = match cs.next_node(p) {
                    Some(n) if n != end => n,
                    _ => break, // no a
                };

                // Neighborhood for q: callback first; if empty, full scan over same chain.
                let mut q_candidates = (self.get_q_neighbors)(p, cs);
                if q_candidates.is_empty() {
                    // Full scan: q must be farther than a (so we reverse a non-empty segment)
                    let mut q = a;
                    loop {
                        let nxt = match cs.next_node(q) {
                            Some(n) if n != end => n,
                            _ => break,
                        };
                        q_candidates.push(q);
                        q = nxt;
                    }
                } else {
                    // keep only same-chain, valid “predecessors” that have a non-sentinel successor
                    q_candidates.retain(|&q| {
                        if cs.chain_of_node(q) != cs.chain_of_node(p) {
                            return false;
                        }
                        if let Some(b) = cs.next_node(q) {
                            b != end
                        } else {
                            false
                        }
                    });
                }

                for &q in &q_candidates {
                    if q == p {
                        continue;
                    }
                    let a = cs.next_node(p).unwrap(); // safe from above

                    // Disallow degenerate case where a == q (adjacent meeting) or a sentinel path
                    if a == q {
                        continue;
                    }

                    // Propose the structural 2-opt (reverse segment (a..=q)) without arc-cost gating.
                    let mut bld = ChainSetDeltaBuilder::new(cs);
                    bld.two_opt(p, q);
                    return Some(bld.build());
                }

                scanned += 1;
                if let Some(cap) = cap {
                    if scanned >= cap.get() {
                        return None;
                    }
                }

                p = a;
            }
        }

        None
    }

    fn name(&self) -> &str {
        "IntraChainTwoOptFirstImprovement"
    }
}
