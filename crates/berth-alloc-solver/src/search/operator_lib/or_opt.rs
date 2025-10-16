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
    search::operator::NeighborhoodOperator,
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

#[inline]
fn advance_k(
    cs: &ChainSet,
    mut n: NodeIndex,
    k_minus_1: usize,
    end: NodeIndex,
) -> Option<NodeIndex> {
    // Return the k-th node starting at n (k-1 steps), stopping before hitting 'end'
    for _ in 0..k_minus_1 {
        let nxt = cs.next_node(n)?;
        if nxt == end {
            return None;
        }
        n = nxt;
    }
    Some(n)
}

/// Δ for relocating a block [x..y] from after p to after d.
// Replace your existing delta_estimate_oropt_k with this one.

#[inline]
fn delta_estimate_oropt_k(
    cs: &ChainSet,
    arc_eval: &ArcEvaluator,
    p: NodeIndex, // Predecessor of the block
    x: NodeIndex, // First node in the block
    y: NodeIndex, // Last node in the block
    d: NodeIndex, // New predecessor for the block
) -> Option<Cost> {
    // Helper to get the cost contribution of a single node `n` scheduled after `pred`.
    let cost_of_node_after = |pred: NodeIndex, n: NodeIndex| -> Option<Cost> {
        let succ_of_n = cs.next_node(n)?;
        arc_eval(pred, succ_of_n)
    };

    let y2 = cs.next_node(y)?; // Node after the block's original position
    let e = cs.next_node(d)?; // Node after the block's new position

    // --- Delta for REMOVING the block [x..y] ---
    // Change is: (p -> y2) - (p -> x ... y -> y2)
    // Heuristic: We only look at the boundaries.
    // Cost change = cost(p->y2) - [cost(p->x) + cost(y->y2)]
    let removal_delta = {
        let cost_p_y2_new = cost_of_node_after(p, y2)?;
        let cost_p_x_old = cost_of_node_after(p, x)?;
        let cost_y_y2_old = cost_of_node_after(y, y2)?;
        cost_p_y2_new
            .saturating_sub(cost_p_x_old)
            .saturating_sub(cost_y_y2_old)
    };

    // --- Delta for INSERTING the block [x..y] ---
    // Change is: (d -> x ... y -> e) - (d -> e)
    // Heuristic: We only look at the boundaries.
    // Cost change = [cost(d->x) + cost(y->e)] - cost(d->e)
    let insertion_delta = {
        let cost_d_x_new = cost_of_node_after(d, x)?;
        let cost_y_e_new = cost_of_node_after(y, e)?;
        let cost_d_e_old = cost_of_node_after(d, e)?;
        cost_d_x_new
            .saturating_add(cost_y_e_new)
            .saturating_sub(cost_d_e_old)
    };

    Some(removal_delta.saturating_add(insertion_delta))
}

pub struct OrOptKNeighborsBestImprovement<'n, T> {
    pub same_chain_only: bool,
    pub max_k: usize,
    pub get_cap: Box<dyn Fn() -> Option<NonZeroUsize> + Send + Sync>,
    pub get_outgoing:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub get_incoming:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub allow:
        Option<Box<dyn Fn(NodeIndex, NodeIndex, NodeIndex, &ChainSet) -> bool + Send + Sync>>,
    _marker: std::marker::PhantomData<T>,
}

impl<'n, T> OrOptKNeighborsBestImprovement<'n, T> {
    pub fn new(
        same_chain_only: bool,
        max_k: usize,
        get_cap: Box<dyn Fn() -> Option<NonZeroUsize> + Send + Sync>,
        get_outgoing: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        get_incoming: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        allow: Option<
            Box<dyn Fn(NodeIndex, NodeIndex, NodeIndex, &ChainSet) -> bool + Send + Sync>,
        >,
    ) -> Self {
        Self {
            same_chain_only,
            max_k: max_k.max(1),
            get_cap,
            get_outgoing,
            get_incoming,
            allow,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'n, T> NeighborhoodOperator<T> for OrOptKNeighborsBestImprovement<'n, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        let cap = (self.get_cap)();
        let mut scanned = 0usize;

        // store y as well so we don't have to recompute it later
        let mut best: Option<(Cost, NodeIndex, NodeIndex, NodeIndex, NodeIndex)> = None;
        //                                           Δ      p        x         y         d

        let mut consider = |p: NodeIndex, x: NodeIndex, y: NodeIndex, d: NodeIndex| {
            if d == p {
                return;
            }
            if self.same_chain_only && cs.chain_of_node(d) != cs.chain_of_node(p) {
                return;
            }
            // Disallow inserting inside the moved block [x..=y] (would create a loop).
            let mut w = x;
            loop {
                if w == d {
                    return;
                }
                if w == y {
                    break;
                }
                w = match cs.next_node(w) {
                    Some(nn) => nn,
                    None => return,
                }
            }
            if let Some(ref allow) = self.allow {
                if !allow(p, x, d, cs) {
                    return;
                }
            }
            if let Some(delta) = delta_estimate_oropt_k(cs, arc_eval, p, x, y, d) {
                if delta < 0 {
                    match best {
                        Some((b, ..)) if delta >= b => {}
                        _ => best = Some((delta, p, x, y, d)),
                    }
                }
            }
        };

        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);

            let mut p = start;
            while let Some(x) = cs.next_node(p) {
                if x == end {
                    break;
                }

                let mut used_any = false;

                for k in 1..=self.max_k {
                    let Some(y) = advance_k(cs, x, k - 1, end) else {
                        break;
                    };

                    // 1) outgoing neighbors of x
                    if let Some(ref acc) = self.get_outgoing {
                        for &d in acc(x, start) {
                            used_any = true;
                            consider(p, x, y, d);
                        }
                    }
                    // 2) incoming neighbors of x
                    if let Some(ref acc) = self.get_incoming {
                        for &d in acc(x, start) {
                            used_any = true;
                            consider(p, x, y, d);
                        }
                    }
                    // 3) fallback: scan all predecessors if no neighbor lists
                    if !used_any {
                        for c2 in 0..cs.num_chains() {
                            let ci2 = ChainIndex(c2);
                            let mut d = cs.start_of_chain(ci2);
                            loop {
                                consider(p, x, y, d);
                                // safe unwrap: start_of_chain has a next (possibly end)
                                let nxt = cs.next_node(d).unwrap();
                                if nxt == cs.end_of_chain(ci2) {
                                    break;
                                }
                                d = nxt;
                            }
                        }
                    }
                }

                scanned += 1;
                if let Some(limit) = cap {
                    if scanned >= limit.get() {
                        break;
                    }
                }

                p = x;
            }
        }

        if let Some((_, p, x, y, d)) = best {
            // IMPORTANT: infer the chain from a real node (x), not from `p` (which can be a sentinel).
            let ci = cs
                .chain_of_node(x)
                .or_else(|| cs.chain_of_node(d))
                .expect("x/d must belong to some chain");
            let _end = cs.end_of_chain(ci);

            // Build the sequence [x ..= y]
            let mut block: Vec<NodeIndex> = Vec::new();
            {
                let mut cur = x;
                loop {
                    block.push(cur);
                    if cur == y {
                        break;
                    }
                    cur = cs
                        .next_node(cur)
                        .expect("block traversal must stay within chain");
                }
            }

            let mut b = ChainSetDeltaBuilder::new(cs);
            b.move_block_after(d, p, y);
            return Some(b.build());
        }

        None
    }

    fn name(&self) -> &str {
        "OrOptKNeighborsBestImprovement"
    }
}
