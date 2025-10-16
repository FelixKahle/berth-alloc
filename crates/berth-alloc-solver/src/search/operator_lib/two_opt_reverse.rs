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
    eval::ArcEvaluator, search::operator::NeighborhoodOperator, state::{
        chain_set::{
            base::ChainSet,
            delta::ChainSetDelta,
            delta_builder::ChainSetDeltaBuilder,
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
        },
        search_state::SolverSearchState,
    }
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub struct TwoOptReverseBestImprovement {
    pub same_chain_only: bool,
    pub scan_cap: Option<usize>,
}

impl TwoOptReverseBestImprovement {
    pub fn new(same_chain_only: bool, scan_cap: Option<usize>) -> Self {
        Self {
            same_chain_only,
            scan_cap,
        }
    }
}

impl<T> NeighborhoodOperator<T> for TwoOptReverseBestImprovement
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'s, 'm, 'p>(
        &self,
        search_state: &'s SolverSearchState<'m, 'p, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        let mut best: Option<(Cost, NodeIndex, NodeIndex)> = None; // (delta, a, b)
        let mut scanned = 0usize;

        // Try each chain independently.
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);

            // iterate edges (a -> a_next) where a_next is real
            let mut a = start;
            while let Some(a_next) = cs.next_node(a) {
                if a_next == end {
                    break;
                }

                // Pick b >= a_next (ensure non-empty segment)
                let mut b = a_next;
                loop {
                    // Need b_next to form (b, b_next) edge
                    let Some(b_next) = cs.next_node(b) else {
                        break;
                    };
                    if b == end || b_next == end {
                        break;
                    }

                    // Δ = new - old = (a,b) + (a_next,b_next) - (a,a_next) - (b,b_next)
                    if let (Some(old1), Some(old2), Some(n1), Some(n2)) = (
                        arc_eval(a, a_next),
                        arc_eval(b, b_next),
                        arc_eval(a, b),
                        arc_eval(a_next, b_next),
                    ) {
                        let old = old1.saturating_add(old2);
                        let newc = n1.saturating_add(n2);
                        let delta = newc.saturating_sub(old);
                        if delta < 0 {
                            match best {
                                Some((cur, _, _)) if delta >= cur => {}
                                _ => best = Some((delta, a, b)),
                            }
                        }
                    }

                    scanned += 1;
                    if let Some(cap) = self.scan_cap {
                        if scanned >= cap {
                            break;
                        }
                    }

                    // advance b
                    let Some(nextb) = cs.next_node(b) else {
                        break;
                    };
                    if nextb == end {
                        break;
                    }
                    b = nextb;
                }

                // advance a
                a = a_next;
            }
        }

        let Some((_delta, a, b)) = best else {
            return None;
        };

        // Safety guards (cheap; avoid builder panics if the situation changed)
        let a_next = match cs.next_node(a) {
            Some(n) => n,
            None => return None,
        };
        let b_next = match cs.next_node(b) {
            Some(n) => n,
            None => return None,
        };
        // must be intra-chain 2-opt: both a and b are in same chain and both successors exist & are real
        let ci_a = match cs.chain_of_node(a) {
            Some(ci) => ci,
            None => return None,
        };
        let ci_b = match cs.chain_of_node(b) {
            Some(ci) => ci,
            None => return None,
        };
        if ci_a != ci_b {
            return None;
        }
        let end = cs.end_of_chain(ci_a);
        if a_next == end || b_next == end {
            return None;
        }
        if a_next == b {
            return None;
        } // adjacent → no-op for reverse

        // Apply with the builder's dedicated primitive (overlay-safe)
        let mut bld = ChainSetDeltaBuilder::new(cs);
        bld.two_opt(a, b);
        Some(bld.build())
    }

    fn name(&self) -> &str {
        "TwoOptReverseBestImprovement"
    }
}
