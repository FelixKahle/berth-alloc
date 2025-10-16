// crates/berth-alloc-solver/src/search/operator_library/swap_best_neighbors.rs

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

// Replace your existing delta_estimate_swap with this one.

#[inline]
fn delta_estimate_swap(
    cs: &ChainSet,
    arc_eval: &ArcEvaluator,
    p: NodeIndex, // Predecessor of x
    x: NodeIndex, // First node to swap
    q: NodeIndex, // Predecessor of y
    y: NodeIndex, // Second node to swap
) -> Option<Cost> {
    // A helper to get the cost contribution of a single node `n` scheduled after `pred`.
    let cost_of_node_after = |pred: NodeIndex, n: NodeIndex| -> Option<Cost> {
        let succ_of_n = cs.next_node(n)?;
        arc_eval(pred, succ_of_n)
    };

    let x2 = cs.next_node(x)?; // Node after x
    let y2 = cs.next_node(y)?; // Node after y

    // Handle adjacent case: p -> x -> y -> y2 (where q = x)
    if x2 == y {
        // Old arcs: (p->x), (x->y), (y->y2)
        // New arcs: (p->y), (y->x), (x->y2)
        let old = cost_of_node_after(p, x)?
            .saturating_add(cost_of_node_after(x, y)?)
            .saturating_add(cost_of_node_after(y, y2)?);
        let new = cost_of_node_after(p, y)?
            .saturating_add(cost_of_node_after(y, x)?)
            .saturating_add(cost_of_node_after(x, y2)?);
        return Some(new.saturating_sub(old));
    }

    // Handle adjacent case: q -> y -> x -> x2 (where p = y)
    if y2 == x {
        // Symmetric to the case above, just call it with swapped args
        return delta_estimate_swap(cs, arc_eval, q, y, p, x);
    }

    // --- Non-adjacent case ---
    // Old arcs: (p->x), (x->x2), (q->y), (y->y2)
    // New arcs: (p->y), (y->x2), (q->x), (x->y2)
    let old = cost_of_node_after(p, x)?
        .saturating_add(cost_of_node_after(x, x2)?)
        .saturating_add(cost_of_node_after(q, y)?)
        .saturating_add(cost_of_node_after(y, y2)?);

    let new = cost_of_node_after(p, y)?
        .saturating_add(cost_of_node_after(y, x2)?)
        .saturating_add(cost_of_node_after(q, x)?)
        .saturating_add(cost_of_node_after(x, y2)?);

    Some(new.saturating_sub(old))
}

pub struct SwapNeighborsBestImprovement<'n, T> {
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

impl<'n, T> SwapNeighborsBestImprovement<'n, T> {
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

impl<'n, T> NeighborhoodOperator<T> for SwapNeighborsBestImprovement<'n, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    // Replace your existing make_neighboor implementation

    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        let cap = (self.get_cap)();
        let mut scanned = 0usize;

        let mut best: Option<(Cost, NodeIndex, NodeIndex)> = None; // (Δ, p, q)

        // Iterate through all pairs of predecessors (p, q)
        for c1 in 0..cs.num_chains() {
            let ci1 = ChainIndex(c1);
            let mut p = cs.start_of_chain(ci1);
            while let Some(x) = cs.next_node(p) {
                if p == cs.end_of_chain(ci1) || x == cs.end_of_chain(ci1) {
                    break;
                }

                let start_c2 = if self.same_chain_only { c1 } else { 0 };
                for c2 in start_c2..cs.num_chains() {
                    let ci2 = ChainIndex(c2);
                    let mut q = cs.start_of_chain(ci2);

                    while let Some(y) = cs.next_node(q) {
                        if q == cs.end_of_chain(ci2) || y == cs.end_of_chain(ci2) {
                            break;
                        }

                        if p == q {
                            q = y;
                            continue;
                        } // Skip self-swap

                        if let Some(delta) = delta_estimate_swap(cs, arc_eval, p, x, q, y) {
                            if best.is_none() || delta < best.unwrap().0 {
                                best = Some((delta, p, q));
                            }
                        }

                        scanned += 1;
                        if let Some(limit) = cap {
                            if scanned >= limit.get() {
                                break;
                            }
                        }
                        q = y;
                    }
                    if let Some(limit) = cap {
                        if scanned >= limit.get() {
                            break;
                        }
                    }
                }
                if let Some(limit) = cap {
                    if scanned >= limit.get() {
                        break;
                    }
                }
                p = x;
            }
        }

        if let Some((delta, p, q)) = best {
            if delta < 0 {
                // Use the robust swap_after primitive from your builder
                let mut bld = ChainSetDeltaBuilder::new(cs);
                bld.swap_after(p, q);
                return Some(bld.build());
            }
        }

        None
    }
    fn name(&self) -> &str {
        "SwapNeighborsBestImprovement"
    }
}

/// Blind/random swap: pick random predecessors `p` and `q` (optionally same-chain),
/// then perform `swap_after(p, q)` without consulting costs.
/// Randomness is supplied by an immutable callback `Fn(usize) -> usize`.
pub struct RandomSwapNeighborsBlind<'n, T> {
    /// Restrict swaps to within the same chain.
    pub same_chain_only: bool,

    /// Hard cap on random attempts per call.
    pub max_tries: NonZeroUsize,

    /// Optional proximity helpers (bias y choice given x).
    pub get_outgoing:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub get_incoming:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,

    /// Optional structural guard: (p, q, x, cs) -> allowed?
    pub allow:
        Option<Box<dyn Fn(NodeIndex, NodeIndex, NodeIndex, &ChainSet) -> bool + Send + Sync>>,

    /// Random index sampler: given `n`, return an index in [0, n).
    index_sampler: Box<dyn Fn(usize) -> usize + Send + Sync + 'n>,

    _marker: std::marker::PhantomData<T>,
}

impl<'n, T> RandomSwapNeighborsBlind<'n, T> {
    pub fn new(
        same_chain_only: bool,
        max_tries: NonZeroUsize,
        get_outgoing: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        get_incoming: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        allow: Option<
            Box<dyn Fn(NodeIndex, NodeIndex, NodeIndex, &ChainSet) -> bool + Send + Sync>,
        >,
        index_sampler: Box<dyn Fn(usize) -> usize + Send + Sync + 'n>,
    ) -> Self {
        Self {
            same_chain_only,
            max_tries,
            get_outgoing,
            get_incoming,
            allow,
            index_sampler,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn sample_idx(&self, n: usize) -> usize {
        if n <= 1 { 0 } else { (self.index_sampler)(n) }
    }

    /// Random predecessor `p` in chain `ci` whose successor is real.
    fn sample_pred_in_chain(&self, cs: &ChainSet, ci: ChainIndex) -> Option<NodeIndex> {
        let start = cs.start_of_chain(ci);
        let end = cs.end_of_chain(ci);

        let mut preds: Vec<NodeIndex> = Vec::with_capacity(16);
        let mut p = start;
        while let Some(x) = cs.next_node(p) {
            if x == end {
                break;
            }
            preds.push(p); // p->x where x is real
            p = x;
        }
        if preds.is_empty() {
            None
        } else {
            Some(preds[self.sample_idx(preds.len())])
        }
    }

    /// Choose q given x, prefer proximity if provided; else random (respecting same_chain_only).
    fn pick_q_given_x(
        &self,
        cs: &ChainSet,
        x: NodeIndex,
        ci_hint: Option<ChainIndex>,
    ) -> Option<NodeIndex> {
        if let Some(ref get_out) = self.get_outgoing {
            let start_for_list = ci_hint
                .or_else(|| cs.chain_of_node(x))
                .map(|ci| cs.start_of_chain(ci))
                .unwrap_or(x);

            let outs = get_out(x, start_for_list);
            if !outs.is_empty() {
                let y = outs[self.sample_idx(outs.len())];
                if let Some(q) = cs.prev_node(y) {
                    if !cs.is_tail_node(q) && q != y {
                        return Some(q);
                    }
                }
            }
        }

        // Fallback: pick random chain & predecessor
        let ci = if self.same_chain_only {
            cs.chain_of_node(x)?
        } else {
            ChainIndex(self.sample_idx(cs.num_chains()))
        };
        self.sample_pred_in_chain(cs, ci)
    }
}

impl<'n, T> NeighborhoodOperator<T> for RandomSwapNeighborsBlind<'n, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        _arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        if cs.num_chains() == 0 {
            return None;
        }

        // Quick check: do we have any real node at all?
        let mut has_real = false;
        'outer: for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let p = cs.start_of_chain(ci);
            while let Some(x) = cs.next_node(p) {
                if x == cs.end_of_chain(ci) {
                    break;
                }
                has_real = true;
                break 'outer;
            }
        }
        if !has_real {
            return None;
        }

        let tries = self.max_tries.get();
        for _ in 0..tries {
            // 1) pick chain for p
            let ci_p = if self.same_chain_only {
                // try a few chains until one has a real node
                let mut picked = None;
                for _ in 0..cs.num_chains().max(1) {
                    let ci = ChainIndex(self.sample_idx(cs.num_chains()));
                    if let Some(x) = cs.next_node(cs.start_of_chain(ci)) {
                        if x != cs.end_of_chain(ci) {
                            picked = Some(ci);
                            break;
                        }
                    }
                }
                picked.unwrap_or(ChainIndex(0))
            } else {
                ChainIndex(self.sample_idx(cs.num_chains()))
            };

            // 2) sample predecessor p with real successor x
            let p = match self.sample_pred_in_chain(cs, ci_p) {
                Some(pp) => pp,
                None => continue,
            };
            let x = cs.next_node(p)?;
            if cs.is_sentinel_node(x) {
                continue;
            }

            // 3) choose q (predecessor of y)
            let q = match self.pick_q_given_x(cs, x, self.same_chain_only.then_some(ci_p)) {
                Some(qq) => qq,
                None => continue,
            };

            if p == q {
                continue;
            } // no-op
            let y = cs.next_node(q)?;
            if cs.is_sentinel_node(y) || x == y {
                continue;
            }

            // Optional structural guard
            if let Some(ref allow) = self.allow {
                if !allow(p, q, x, cs) {
                    continue;
                }
            }

            // Build delta: swap successors of p and q
            let mut b = ChainSetDeltaBuilder::new(cs);
            b.swap_after(p, q);
            return Some(b.build());
        }

        None
    }

    fn name(&self) -> &str {
        "RandomSwapNeighborsBlind"
    }
}
