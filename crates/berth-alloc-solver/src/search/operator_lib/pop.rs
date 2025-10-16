// crates/berth-alloc-solver/src/search/operator_library/popmusic.rs

// Copyright (c) 2025 Felix Kahle.
// MIT License (see project root)

use crate::{
    eval::ArcEvaluator,
    model::solver_model::SolverModel,
    search::operator::NeighborhoodOperator,
    state::{
        chain_set::{
            base::ChainSet,
            delta::ChainSetDelta,
            delta_builder::ChainSetDeltaBuilder,
            index::{ChainIndex, NodeIndex},
            overlay::ChainSetOverlay,
            view::{ChainRef, ChainSetView},
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::num::NonZeroUsize;

/// A small, plain description of the local POPMUSIC subproblem that you can solve in a callback.
/// You get:
/// - immutable model, chain set & an overlay delta (empty when passed)
/// - the chosen chain indices (seed + neighbors)
/// - the current node order per chosen chain (excluding sentinels)
///
/// Your callback should return either:
/// - `Some(new_orders)` with the **full** new order for *each* chosen chain (excluding sentinels)
/// - or `None` if you didn't find an improvement / you want to skip.
///
/// The operator will build a delta that reorders those chains to match your new orders.
#[derive(Debug, Clone)]
pub struct PopSubproblem<'a, 'm, T>
where
    T: Ord + Copy + CheckedAdd + CheckedSub + Zero + Into<Cost>,
{
    pub model: &'m SolverModel<'m, T>,
    pub chains: &'a ChainSet,
    pub overlay: ChainSetOverlay<'a, 'm>,
    pub chosen: Vec<ChainIndex>,
    pub current_orders: Vec<(ChainIndex, Vec<NodeIndex>)>,
}

/// The response expected from the subproblem solver:
/// a complete new order (no sentinels) for each chain listed in `chosen`.
pub type PopSolution = Vec<(ChainIndex, Vec<NodeIndex>)>;

/// POPMUSIC-style operator:
/// - pick a seed chain,
/// - pick up to `r` neighbor chains (policy = callback),
/// - expose the subproblem to `solve_subproblem`,
/// - if callback returns a new per-chain order, build a delta to realize it.
pub struct PopmusicOperator<'n, T>
where
    T: Ord + Copy + CheckedAdd + CheckedSub + Zero + Into<Cost>,
{
    /// Size of the neighborhood (number of neighbor chains to aggregate with the seed).
    pub r: NonZeroUsize,

    /// Select a seed chain. You decide how (round-robin, random, most loaded, …).
    pub select_seed: Box<dyn Fn(&ChainSet) -> ChainIndex + Send + Sync + 'n>,

    /// Given (seed, r), return the neighbor chains to include (do NOT include the seed itself).
    /// You can compute “closeness” via proximity, shared arcs, etc.
    pub neighbors_of:
        Box<dyn Fn(&ChainSet, ChainIndex, usize) -> Vec<ChainIndex> + Send + Sync + 'n>,

    /// Optional guard to veto a subproblem before solving (e.g., too small, trivial, forbidden).
    pub allow_subproblem:
        Option<Box<dyn Fn(&ChainSet, ChainIndex, &[ChainIndex]) -> bool + Send + Sync + 'n>>,

    /// Subproblem solver: receives a `PopSubproblem` and may return a **complete reordering**
    /// for each chosen chain. You can run anything here (DP, CP, MILP, greedy, …).
    pub solve_subproblem:
        Box<dyn Fn(PopSubproblem<'_, '_, T>) -> Option<PopSolution> + Send + Sync + 'n>,

    _marker: std::marker::PhantomData<T>,
}

impl<'n, T> PopmusicOperator<'n, T>
where
    T: Ord + Copy + CheckedAdd + CheckedSub + Zero + Into<Cost>,
{
    pub fn new(
        r: NonZeroUsize,
        select_seed: impl Fn(&ChainSet) -> ChainIndex + Send + Sync + 'n,
        neighbors_of: impl Fn(&ChainSet, ChainIndex, usize) -> Vec<ChainIndex> + Send + Sync + 'n,
        solve_subproblem: impl Fn(PopSubproblem<'_, '_, T>) -> Option<PopSolution> + Send + Sync + 'n,
    ) -> Self {
        Self {
            r,
            select_seed: Box::new(select_seed),
            neighbors_of: Box::new(neighbors_of),
            allow_subproblem: None,
            solve_subproblem: Box::new(solve_subproblem),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_allow(
        mut self,
        allow: impl Fn(&ChainSet, ChainIndex, &[ChainIndex]) -> bool + Send + Sync + 'n,
    ) -> Self {
        self.allow_subproblem = Some(Box::new(allow));
        self
    }
}

impl<'n, T> NeighborhoodOperator<T> for PopmusicOperator<'n, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        _arc_eval: &ArcEvaluator, // not needed here; sub-solver can ignore or derive its own
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        if cs.num_chains() == 0 {
            return None;
        }

        // 1) Select seed and neighbors according to your policies.
        let seed = (self.select_seed)(cs);
        let neigh = (self.neighbors_of)(cs, seed, self.r.get().saturating_sub(1));
        // Build the subproblem chain set = seed ∪ neighbors (unique, keep deterministic order).
        let mut chosen: Vec<ChainIndex> = Vec::with_capacity(1 + neigh.len());
        chosen.push(seed);
        for ci in neigh {
            if !chosen.contains(&ci) {
                chosen.push(ci);
            }
        }
        if chosen.is_empty() {
            return None;
        }

        if let Some(allow) = &self.allow_subproblem {
            if !allow(cs, seed, &chosen[1..]) {
                return None;
            }
        }

        // 2) Capture current orders (exclude sentinels).
        let mut current_orders: Vec<(ChainIndex, Vec<NodeIndex>)> =
            Vec::with_capacity(chosen.len());
        for &ci in &chosen {
            let cr = ChainRef::new(cs, ci);
            let mut order: Vec<NodeIndex> = Vec::new();
            let mut n_opt = cr.first_real_node(cr.start());
            let mut guard = 0usize;
            let max = 2 * cs.num_nodes(); // safety walk cap
            while let Some(n) = n_opt {
                if n == cr.end() || guard >= max {
                    break;
                }
                order.push(n);
                n_opt = cr.next_real(n);
                guard += 1;
            }
            current_orders.push((ci, order));
        }

        // Early out if all selected chains are empty (no work to do).
        if current_orders.iter().all(|(_, v)| v.is_empty()) {
            return None;
        }

        // 3) Hand the subproblem to the user’s solver.
        let model = search_state.model();
        let delta = ChainSetDelta::new();
        let overlay = ChainSetOverlay::new(cs, &delta);
        let sub = PopSubproblem {
            model,
            chains: cs,
            overlay,
            chosen: chosen.clone(),
            current_orders,
        };

        let Some(new_orders) = (self.solve_subproblem)(sub.clone()) else {
            return None;
        };

        // 4) Validate: every chosen chain must appear once with a full new order (no duplicate nodes).
        //    We keep it minimal and defensive; you can strengthen inside your solver.
        if new_orders.len() != chosen.len() {
            return None;
        }

        // Map chain -> new order
        // Also detect “no-op” (all chains unchanged) to skip empty deltas.
        let mut all_unchanged = true;
        for (i, _) in chosen.iter().enumerate() {
            let (_, ref current) = sub.current_orders[i];
            let (_, ref newv) = new_orders[i];
            if current.len() != newv.len() || current.iter().zip(newv.iter()).any(|(a, b)| a != b) {
                all_unchanged = false;
                break;
            }
        }
        if all_unchanged {
            return None;
        }

        // 5) Build a delta to realize the new per-chain orders:
        //    - detach all real nodes of each chosen chain,
        //    - reinsert them in the provided order immediately after the start sentinel.
        let mut bld = ChainSetDeltaBuilder::new(cs);

        // Detach phase (safe even if already isolated).
        for &ci in &chosen {
            let cr = ChainRef::new(cs, ci);
            let mut n_opt = cr.first_real_node(cr.start());
            // detach all real nodes of this chain
            while let Some(n) = n_opt {
                if n == cr.end() {
                    break;
                }
                bld.detach(n);
                n_opt = cr.next_real(n); // walk original view; detached nodes are isolated anyway
            }
        }

        // Insert phase
        for (ci, order) in new_orders.iter() {
            let cr = ChainRef::new(cs, *ci);
            let mut prev = cr.start();
            for &n in order {
                // Note: insert_after returns false only on structural misuse (e.g., sentinel misuse).
                if !bld.insert_after(prev, n) {
                    // If a single insert fails, abandon the move.
                    return None;
                }
                prev = n;
            }
        }

        let delta = bld.build();
        if delta.is_empty() { None } else { Some(delta) }
    }

    fn name(&self) -> &str {
        "POPMUSIC(Local Re-optimization)"
    }
}
