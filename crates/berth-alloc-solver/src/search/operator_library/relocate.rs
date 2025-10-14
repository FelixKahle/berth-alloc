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

// src/search/operator/relocate1_first_improvement.rs

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

/// Move a single node x from after src_prev to after dst_prev (Or-Opt-1), first-improvement.
/// If `same_chain_only = true`, keeps moves intra-berth; else, allows cross-berth relocations.
///
/// Hooks:
/// - `get_cap`: early stop
/// - `get_destinations`: for a given node x, propose destination predecessors (Vec<NodeIndex>)
pub struct Relocate1FirstImprovement {
    pub same_chain_only: bool,
    pub get_cap: Box<dyn Fn() -> Option<NonZeroUsize> + Send + Sync>,
    pub get_destinations: Box<dyn Fn(NodeIndex, &ChainSet) -> Vec<NodeIndex> + Send + Sync>,
}

impl Default for Relocate1FirstImprovement {
    fn default() -> Self {
        Self {
            same_chain_only: true,
            get_cap: Box::new(|| None),
            // Fallback: full scan of all predecessors across all chains
            get_destinations: Box::new(|_x, cs| {
                let mut v = Vec::new();
                for c in 0..cs.num_chains() {
                    let ci = ChainIndex(c);
                    let start = cs.start_of_chain(ci);
                    let end = cs.end_of_chain(ci);
                    // iterate predecessors (all nodes incl. head)
                    let mut p = start;
                    loop {
                        v.push(p);
                        let nxt = cs.next_node(p).unwrap();
                        if nxt == end {
                            break;
                        }
                        p = nxt;
                    }
                }
                v
            }),
        }
    }
}

impl<T> NeighborhoodOperator<T> for Relocate1FirstImprovement
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        _arc_eval: &ArcEvaluator, // unused on purpose
    ) -> Option<ChainSetDelta> {
        // println!("Called Relocate1FirstImprovement");

        let cs: &ChainSet = search_state.chain_set();
        let cap = (self.get_cap)();
        let mut scanned = 0usize;

        // Enumerate each chain to choose src_prev (predecessor of x)
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);

            let mut src_prev = start;
            loop {
                let x = match cs.next_node(src_prev) {
                    Some(n) if n != end => n,
                    _ => break,
                };

                // Candidate destinations (predecessors)
                let mut dst_list = (self.get_destinations)(x, cs);

                // Fallback if callback returns empty: scan all predecessors across all chains
                if dst_list.is_empty() {
                    for c2 in 0..cs.num_chains() {
                        let ci2 = ChainIndex(c2);
                        let mut p2 = cs.start_of_chain(ci2);
                        loop {
                            dst_list.push(p2);
                            let nx = cs.next_node(p2).unwrap();
                            if nx == cs.end_of_chain(ci2) {
                                break;
                            }
                            p2 = nx;
                        }
                    }
                }

                for &dst_prev in &dst_list {
                    // Skip illegal/self placements and degenerate cases
                    if dst_prev == src_prev {
                        continue; // no-op
                    }
                    if self.same_chain_only
                        && cs.chain_of_node(dst_prev) != cs.chain_of_node(src_prev)
                    {
                        continue;
                    }
                    // Don’t insert x immediately after its own predecessor (no change)
                    if dst_prev == cs.prev_node(x).unwrap_or(dst_prev) {
                        continue;
                    }

                    // Avoid inserting right before itself (would create 1-cycle)
                    let dst_next = cs.next_node(dst_prev).unwrap();
                    if dst_next == x {
                        continue;
                    }

                    // Propose the structural move without arc-cost gating.
                    let mut bld = ChainSetDeltaBuilder::new(cs);
                    bld.move_after(dst_prev, src_prev);
                    return Some(bld.build());
                }

                scanned += 1;
                if let Some(cap) = cap {
                    if scanned >= cap.get() {
                        return None;
                    }
                }

                src_prev = x;
            }
        }

        None
    }

    fn name(&self) -> &str {
        "Relocate1FirstImprovement"
    }
}
