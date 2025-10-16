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
fn delta_estimate_relocate1(
    cs: &ChainSet,
    arc_eval: &ArcEvaluator,
    p: NodeIndex, // predecessor of x
    x: NodeIndex, // node to move
    d: NodeIndex, // new predecessor of x
) -> Option<Cost> {
    // Current successors we need.
    let x2 = cs.next_node(x)?; // successor of x (could be tail)
    let e = cs.next_node(d)?; // successor of d (could be tail)

    // If x is isolated (x->x), or would be inserted immediately before itself: bail.
    if x2 == x || d == p {
        return None;
    }
    // Moving right before itself is a no-op: d_next == x means inserting x where it already is.
    if e == x {
        return None;
    }

    // Old cost contribution of affected arcs.
    let old_p_x = arc_eval(p, x)?;
    let old_x_x2 = arc_eval(x, x2)?;
    let old_d_e = arc_eval(d, e)?;

    // New cost contribution after relocating x after d.
    let new_p_x2 = arc_eval(p, x2)?;
    let new_d_x = arc_eval(d, x)?;
    let new_x_e = arc_eval(x, e)?;

    // Δ = new - old
    let new_sum = new_p_x2.saturating_add(new_d_x).saturating_add(new_x_e);
    let old_sum = old_p_x.saturating_add(old_x_x2).saturating_add(old_d_e);

    Some(new_sum.saturating_sub(old_sum))
}

/// Relocate-1 **Best-Improvement** guided by neighbor lists.
pub struct RelocateNeighborsBestImprovement<'n, T> {
    pub same_chain_only: bool,
    pub get_cap: Box<dyn Fn() -> Option<NonZeroUsize> + Send + Sync>,
    pub get_outgoing:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub get_incoming:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub allow:
        Option<Box<dyn Fn(NodeIndex, NodeIndex, NodeIndex, &ChainSet) -> bool + Send + Sync>>,
    _marker: std::marker::PhantomData<T>,
}

impl<'n, T> RelocateNeighborsBestImprovement<'n, T> {
    pub fn new(
        same_chain_only: bool,
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
            get_cap,
            get_outgoing,
            get_incoming,
            allow,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'n, T> NeighborhoodOperator<T> for RelocateNeighborsBestImprovement<'n, T>
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

        let mut best_improving: Option<(Cost, NodeIndex, NodeIndex)> = None; // (delta, p, d)
        let mut best_any: Option<(Cost, NodeIndex, NodeIndex)> = None;

        let mut consider = |p: NodeIndex, x: NodeIndex, d: NodeIndex| {
            if d == p {
                return;
            }
            let Some(d_next) = cs.next_node(d) else {
                return;
            };
            if d_next == x {
                return;
            }
            if self.same_chain_only && cs.chain_of_node(d) != cs.chain_of_node(p) {
                return;
            }
            if let Some(ref allow) = self.allow {
                if !allow(p, x, d, cs) {
                    return;
                }
            }
            if let Some(delta) = delta_estimate_relocate1(cs, arc_eval, p, x, d) {
                // Track best overall
                match best_any {
                    Some((cur, _, _)) if delta >= cur => {}
                    _ => best_any = Some((delta, p, d)),
                }
                // Track best improving
                if delta < 0 {
                    match best_improving {
                        Some((cur, _, _)) if delta >= cur => {}
                        _ => best_improving = Some((delta, p, d)),
                    }
                }
            }
        };

        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);

            let mut p = start;
            loop {
                let x = match cs.next_node(p) {
                    Some(n) if n != end => n,
                    _ => break,
                };

                let mut used_any = false;

                // 1) outgoing list (preferred)
                if let Some(ref acc) = self.get_outgoing {
                    for &d in acc(x, start) {
                        used_any = true;
                        consider(p, x, d);
                    }
                }

                // 2) incoming list (alternate)
                if let Some(ref acc) = self.get_incoming {
                    for &d in acc(x, start) {
                        used_any = true;
                        consider(p, x, d);
                    }
                }

                // 3) fallback: scan all predecessors if no neighbor lists
                if !used_any {
                    for c2 in 0..cs.num_chains() {
                        let ci2 = ChainIndex(c2);
                        let start2 = cs.start_of_chain(ci2);
                        let end2 = cs.end_of_chain(ci2);

                        // Skip empty chain quickly.
                        if matches!(cs.next_node(start2), Some(n) if n == end2) {
                            continue;
                        }

                        let mut d = start2;
                        loop {
                            consider(p, x, d);

                            // Safe advance; terminate on None or reaching tail.
                            let Some(nxt) = cs.next_node(d) else {
                                break;
                            };
                            if nxt == end2 {
                                break;
                            }
                            d = nxt;
                        }
                    }
                }

                scanned += 1;
                if let Some(limit) = cap {
                    if scanned >= limit.get() {
                        break; // stop early; we’ll apply best-so-far
                    }
                }

                p = x;
            }
        }

        // Prefer improving; else return least-worsening
        let pick = best_improving.or(best_any)?;
        let (_delta, p, d) = pick;
        let mut b = ChainSetDeltaBuilder::new(cs);
        b.move_after(d, p);
        Some(b.build())
    }

    fn name(&self) -> &str {
        "RelocateNeighborsBestImprovement"
    }
}

pub struct RelocateMostExpensiveArc<'n> {
    pub get_outgoing:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub scan_cap: Option<usize>,
}

impl<'n> RelocateMostExpensiveArc<'n> {
    pub fn new(
        get_outgoing: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        scan_cap: Option<usize>,
    ) -> Self {
        Self {
            get_outgoing,
            scan_cap,
        }
    }
}

impl<T> NeighborhoodOperator<T> for RelocateMostExpensiveArc<'_>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'s, 'm, 'p>(
        &self,
        search_state: &'s SolverSearchState<'m, 'p, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();

        // Track worst arc (p -> x) while carrying the chain index to avoid chain_of_node on sentinels.
        let mut worst: Option<(Cost, ChainIndex, NodeIndex, NodeIndex)> = None; // (cost, ci, p, x)

        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);

            let mut p = start;
            while let Some(x) = cs.next_node(p) {
                if x == end {
                    break;
                }
                if let Some(cst) = arc_eval(p, x) {
                    match worst {
                        Some((w, _, _, _)) if cst <= w => {}
                        _ => worst = Some((cst, ci, p, x)),
                    }
                }
                p = x;
            }
        }

        let Some((_w_cost, ci_p, p, x)) = worst else {
            return None;
        };

        let start_p = cs.start_of_chain(ci_p);

        // We'll need x2 for delta computations; if it's missing something is inconsistent—bail out.
        let Some(x2) = cs.next_node(x) else {
            return None;
        };

        // Search best destination 'd' (predecessor to insert x after).
        let mut best: Option<(Cost, NodeIndex)> = None; // (delta, d)
        let mut scan = 0usize;

        let mut consider_d = |d: NodeIndex| {
            // Skip degenerate placements.
            if d == p {
                return;
            }
            let Some(d_next) = cs.next_node(d) else {
                return;
            };
            if d_next == x {
                return;
            }

            // Δ = [ (p->x2) + (d->x) + (x->d_next) ] - [ (p->x) + (x->x2) + (d->d_next) ]
            let old = arc_eval(p, x)
                .and_then(|v| arc_eval(x, x2).map(|w| v.saturating_add(w)))
                .and_then(|v| arc_eval(d, d_next).map(|w| v.saturating_add(w)));
            let new = arc_eval(p, x2)
                .and_then(|v| arc_eval(d, x).map(|w| v.saturating_add(w)))
                .and_then(|v| arc_eval(x, d_next).map(|w| v.saturating_add(w)));

            if let (Some(o), Some(n)) = (old, new) {
                let delta = n.saturating_sub(o);
                if delta < 0 {
                    match best {
                        Some((cur, _)) if delta >= cur => {}
                        _ => best = Some((delta, d)),
                    }
                }
            }
        };

        if let Some(ref out) = self.get_outgoing {
            for &d in out(x, start_p) {
                consider_d(d);
                scan += 1;
                if let Some(cap) = self.scan_cap {
                    if scan >= cap {
                        break;
                    }
                }
            }
        } else {
            // Fallback: scan all predecessors across all chains safely (no unwraps).
            for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let start2 = cs.start_of_chain(ci);
                let end2 = cs.end_of_chain(ci);

                // Quick skip if chain empty (start->end directly).
                if matches!(cs.next_node(start2), Some(n) if n == end2) {
                    continue;
                }

                let mut d = start2;
                loop {
                    consider_d(d);

                    let Some(nxt) = cs.next_node(d) else {
                        break; // defensive; shouldn't happen
                    };
                    if nxt == end2 {
                        break;
                    }
                    d = nxt;
                }
            }
        }

        let Some((_, d)) = best else {
            return None;
        };

        // Build delta: move successor of p (which is x) to be successor of d.
        let mut b = ChainSetDeltaBuilder::new(cs);
        b.move_after(d, p);
        Some(b.build())
    }

    fn name(&self) -> &str {
        "RelocateMostExpensiveArc"
    }
}
