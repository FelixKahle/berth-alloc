// crates/berth-alloc-solver/src/search/operator_library/relocate_best_neighbors.rs

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

/// Δ for relocating node `x` from after `p` to after `d`.
#[inline]
fn delta_estimate_relocate1(
    cs: &ChainSet,
    arc_eval: &ArcEvaluator,
    p: NodeIndex,
    x: NodeIndex,
    d: NodeIndex,
) -> Option<Cost> {
    let x2 = cs.next_node(x)?;
    let e = cs.next_node(d)?;
    let old = arc_eval(p, x)?
        .saturating_add(arc_eval(x, x2)?)
        .saturating_add(arc_eval(d, e)?);
    let new = arc_eval(p, x2)?
        .saturating_add(arc_eval(d, x)?)
        .saturating_add(arc_eval(x, e)?);
    Some(new.saturating_sub(old))
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

        // Track best improving candidate
        let mut best_delta: Option<(Cost, NodeIndex, NodeIndex)> = None; // (delta, p, d)

        // Helper: consider candidate d for relocating successor x of p.
        let mut consider = |p: NodeIndex, x: NodeIndex, d: NodeIndex| {
            if d == p {
                return;
            }
            let Some(d_next) = cs.next_node(d) else {
                return;
            };
            if d_next == x {
                return;
            } // trivial no-op cycle

            if self.same_chain_only && cs.chain_of_node(d) != cs.chain_of_node(p) {
                return;
            }
            if let Some(ref allow) = self.allow {
                if !allow(p, x, d, cs) {
                    return;
                }
            }
            if let Some(delta) = delta_estimate_relocate1(cs, arc_eval, p, x, d) {
                if delta < 0 {
                    match best_delta {
                        Some((best, _, _)) if delta >= best => {}
                        _ => best_delta = Some((delta, p, d)),
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
                        let mut d = cs.start_of_chain(ci2);
                        loop {
                            consider(p, x, d);
                            let nxt = cs.next_node(d).unwrap();
                            if nxt == cs.end_of_chain(ci2) {
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

        // Apply best improvement, if any
        if let Some((_delta, p, d)) = best_delta {
            let mut b = ChainSetDeltaBuilder::new(cs);
            // Move successor of p to become successor of d.
            b.move_after(d, p);
            Some(b.build())
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        "RelocateNeighborsBestImprovement"
    }
}
