// src/search/operator/random_swap_anywhere.rs

use crate::{
    eval::ArcEvaluator,
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
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
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Picks two random non-sentinel nodes and swaps them (via swap_after on their predecessors).
/// Use this to sanity-check whether the engine/evaluator accepts feasible moves after the first one.
pub struct RandomSwapAnywhere<T> {
    /// Maximum random trials to find a usable pair in one call.
    pub max_trials: usize,
    /// If true, only sample pairs from the same chain (avoids cross-berth by construction).
    pub same_chain_only: bool,
    /// If false, we pre-check berth permission for cross-berth swaps; if true we skip and let the evaluator reject.
    pub permit_cross_berth: bool,
    /// RNG
    rng: ChaCha8Rng,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub> RandomSwapAnywhere<T> {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u8; 32];
        s[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            max_trials: 128,
            same_chain_only: false,
            permit_cross_berth: false,
            rng: ChaCha8Rng::from_seed(s),
            _phantom: std::marker::PhantomData,
        }
    }

    fn collect_predecessors(cs: &ChainSet) -> Vec<(ChainIndex, NodeIndex, NodeIndex)> {
        // (chain, predecessor, node) for all non-sentinel nodes
        let mut v = Vec::new();
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);
            let mut p = start;
            while let Some(x) = cs.next_node(p) {
                if x == end {
                    break;
                }
                v.push((ci, p, x));
                p = x;
            }
        }
        v
    }

    #[inline]
    fn model_allows_on_berth(
        model: &SolverModel<'_, T>,
        req: RequestIndex,
        berth: BerthIndex,
    ) -> bool {
        matches!(model.processing_time(req, berth), Some(Some(_)))
    }
}

impl<T> NeighborhoodOperator<T> for RandomSwapAnywhere<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
{
    fn make_neighboor<'s, 'm, 'p>(
        &self,
        st: &'s SolverSearchState<'m, 'p, T>,
        _arc: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = st.chain_set();
        let preds = Self::collect_predecessors(cs);
        if preds.len() < 2 {
            return None;
        }

        // We need interior mutability for RNG; keep operator as &self but use a local copy each call
        // to avoid lifetime issues; this is fine for testing. For production, wrap rng in a Mutex/RefCell.
        let mut rng = self.rng.clone();

        for _ in 0..self.max_trials {
            let i = rng.random_range(0..preds.len());
            let j = rng.random_range(0..preds.len());
            if i == j {
                continue;
            }

            let (ci_p, p, a) = preds[i];
            let (ci_q, q, b) = preds[j];

            // skip degenerate: same predecessor (no-op), or adjacency that would create 1-cycles
            if p == q {
                continue;
            }
            if a == b {
                continue;
            }

            if self.same_chain_only && ci_p != ci_q {
                continue;
            }

            if !self.permit_cross_berth && ci_p != ci_q {
                // quick permission check: a must be allowed on berth(ci_q), b on berth(ci_p)
                let model = st.model();
                let bi_p = BerthIndex(ci_p.get());
                let bi_q = BerthIndex(ci_q.get());
                let a_req = RequestIndex(a.get());
                let b_req = RequestIndex(b.get());
                if !Self::model_allows_on_berth(model, a_req, bi_q) {
                    continue;
                }
                if !Self::model_allows_on_berth(model, b_req, bi_p) {
                    continue;
                }
            }

            // Avoid inserting node immediately before itself (rare corner when p is predecessor of b and q of a and nodes are adjacent across chains).
            if let Some(p_next) = cs.next_node(p) {
                if p_next == b {
                    continue;
                }
            }
            if let Some(q_next) = cs.next_node(q) {
                if q_next == a {
                    continue;
                }
            }

            // Build swap: swap successors after p and q (i.e., swap nodes a and b)
            let mut bld = ChainSetDeltaBuilder::new(cs);
            bld.swap_after(p, q);
            return Some(bld.build());
        }
        None
    }

    fn name(&self) -> &str {
        "RandomSwapAnywhere"
    }
}
