// crates/berth-alloc-solver/src/search/operator_lib/ruin.rs
use crate::{
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
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct RuinRandomSegment {
    pub min_k: usize,
    pub max_k: usize,
    pub seed: u64,
}

impl RuinRandomSegment {
    pub fn new(min_k: usize, max_k: usize, seed: u64) -> Self {
        Self {
            min_k,
            max_k: max_k.max(min_k),
            seed,
        }
    }
}

impl<T> NeighborhoodOperator<T> for RuinRandomSegment
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'s, 'm, 'p>(
        &self,
        search_state: &'s SolverSearchState<'m, 'p, T>,
        _arc_eval: &crate::eval::ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        if cs.num_chains() == 0 {
            return None;
        }

        // RNG
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&self.seed.to_le_bytes());
        let mut rng = ChaCha8Rng::from_seed(seed);

        // Pick a random non-empty chain
        let chain_choices: Vec<ChainIndex> = (0..cs.num_chains()).map(ChainIndex).collect();
        // find first with at least 2 real nodes
        let ci = chain_choices.into_iter().find(|&ci| {
            let mut cnt = 0usize;
            let start = cs.start_of_chain(ci);
            let end = cs.end_of_chain(ci);
            let mut n = start;
            while let Some(nx) = cs.next_node(n) {
                if nx == end {
                    break;
                }
                cnt += 1;
                n = nx;
                if cnt >= 2 {
                    return true;
                }
            }
            false
        })?;

        let start = cs.start_of_chain(ci);
        let end = cs.end_of_chain(ci);

        // gather nodes
        let mut nodes: Vec<NodeIndex> = Vec::new();
        {
            let mut n = cs.next_node(start)?;
            while n != end {
                nodes.push(n);
                n = cs.next_node(n)?;
            }
        }
        if nodes.is_empty() {
            return None;
        }

        // pick k and a starting index
        let range = (self.max_k - self.min_k + 1) as u64;
        let k = self.min_k + ((rng.next_u64() % range) as usize);
        let k = k.min(nodes.len());
        let from = (rng.next_u64() as usize) % (nodes.len() - k + 1);
        let segment = &nodes[from..from + k];

        // Build delta: remove the contiguous block by repeatedly moving successor of its predecessor after the predecessor of segment end. Easiest: splice segment out by bypassing it.
        // With only `move_after(dst, src_prev)`, we can remove the segment by “pulling” successors of segment predecessors to just before end boundary. Instead, simpler:
        // Reconnect: prev(seg[0]) -> next(seg[k-1]).
        let next = cs.next_node(*segment.last().unwrap()).unwrap();

        let mut b = ChainSetDeltaBuilder::new(cs);
        let dst = cs.prev_node(next).unwrap();
        for &node in segment.iter().rev() {
            let src_prev = cs.prev_node(node).unwrap();
            b.move_after(dst, src_prev);
        }
        Some(b.build())
    }

    fn name(&self) -> &str {
        "RuinRandomSegment"
    }
}
