// crates/berth-alloc-solver/src/search/operator_lib/alns.rs

use crate::{
    eval::ArcEvaluator,
    scheduling::traits::Scheduler,
    search::operator::NeighborhoodOperator,
    state::{
        chain_set::{
            base::ChainSet,
            delta::ChainSetDelta,
            delta_builder::ChainSetDeltaBuilder,
            index::{ChainIndex, NodeIndex},
            overlay::ChainSetOverlay,
            view::ChainSetView,
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Saturating, Zero};

#[derive(Clone)]
pub struct ShawPackParams {
    /// fraction of real nodes to destroy (0,1], clamped internally
    pub rho: f64,
    /// randomized greediness α in the paper’s Eq.(21). 1.0 = fully random
    pub alpha: f64,
    /// max candidates to scan per reinsertion
    pub scan_cap: Option<usize>,
}

impl Default for ShawPackParams {
    fn default() -> Self {
        Self {
            rho: 0.15,
            alpha: 0.25,
            scan_cap: Some(20_000),
        }
    }
}

/// Random source injected from the engine (no RefCell, no global).
pub type RandIndexFn = dyn Fn(usize) -> usize + Send + Sync;
pub type RandUnitFn = dyn Fn() -> f64 + Send + Sync;

/// Combined ALNS operator: Shaw removal + "packing greedy" insertion (Section 4.2.1 & 4.2.6)
pub struct ShawDestroyPackInsert<'n, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub params: ShawPackParams,
    /// Optional bias lists (like your proximity map) to accelerate candidate gaps.
    pub get_outgoing:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,
    pub get_incoming:
        Option<Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>>,

    /// RNG hooks provided by the engine per call.
    pub rand_idx: Box<RandIndexFn>,
    pub rand_u: Box<RandUnitFn>,
    pub scheduler: S,

    _phantom: std::marker::PhantomData<(T, S)>,
}

impl<'n, T, S> ShawDestroyPackInsert<'n, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
    S: Scheduler<T>,
{
    pub fn new(
        params: ShawPackParams,
        scheduler: S,
        get_outgoing: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        get_incoming: Option<
            Box<dyn Fn(NodeIndex, NodeIndex) -> &'n [NodeIndex] + Send + Sync + 'n>,
        >,
        rand_idx: Box<RandIndexFn>,
        rand_u: Box<RandUnitFn>,
    ) -> Self {
        Self {
            params,
            get_outgoing,
            get_incoming,
            rand_idx,
            rand_u,
            scheduler,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    fn alpha_pick(&self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let p = (self.rand_u)();
        let a = self.params.alpha.max(1e-9);
        let idx = ((len as f64) * p.powf(a)).ceil() as usize;
        idx.saturating_sub(1).min(len - 1)
    }
}

impl<'n, T, S> NeighborhoodOperator<T> for ShawDestroyPackInsert<'n, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
    S: Scheduler<T>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        //println!("ShawDestroyPackInsert");

        let cs: &ChainSet = search_state.chain_set();
        let model = search_state.model();

        // ------- collect all real nodes -------
        let mut real: Vec<NodeIndex> = Vec::with_capacity(model.flexible_requests_len());
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let mut u = cs.start_of_chain(ci);
            while let Some(v) = cs.next_node(u) {
                if v == cs.end_of_chain(ci) {
                    break;
                }
                real.push(v);
                u = v;
            }
        }
        if real.is_empty() {
            return None;
        }

        // ------- destroy size -------
        let k = ((self.params.rho.max(0.0).min(1.0) * (real.len() as f64)).round() as usize)
            .clamp(1, real.len());

        // ------- relatedness metric M(i,j) in time/space (paper Eq.20 spirit) -------
        // Here we approximate with arc proxy around successors/predecessors (fast & local).
        // Build seed list by "badness": nodes with largest local cost contribution.
        let mut scored: Vec<(NodeIndex, Cost)> = Vec::with_capacity(real.len());
        for &x in &real {
            let p = cs.prev_node(x).unwrap_or(x);
            let s = cs.next_node(x).unwrap_or(x);
            // local arc contribution: (p->x) + (x->s)
            let mut loc = 0;
            if let Some(a) = arc_eval(p, x) {
                loc = loc.saturating_add(a);
            }
            if let Some(b) = arc_eval(x, s) {
                loc = loc.saturating_add(b);
            }
            scored.push((x, loc));
        }
        scored.sort_by(|a, b| b.1.cmp(&a.1)); // descending badness
        let base_ix = self.alpha_pick(scored.len());
        let seed = scored[base_ix].0;

        // Build Ω: candidate pairs sorted by "relatedness" ≈ |Δ around seed|
        // For speed, use incoming/outgoing neighbor lists if provided; else scan all.
        let mut pool: Vec<NodeIndex> = if let Some(go) = &self.get_outgoing {
            (go)(seed, seed).to_vec()
        } else {
            real.clone()
        };
        // Remove the seed itself from pool
        pool.retain(|&n| n != seed);

        // Shuffle-like randomized pick with α-bias; collect K nodes (may include seed)
        let mut to_remove: Vec<NodeIndex> = Vec::with_capacity(k);
        to_remove.push(seed);
        while to_remove.len() < k && !pool.is_empty() {
            let pick = self.alpha_pick(pool.len());
            let n = pool.swap_remove(pick);
            to_remove.push(n);
        }

        // ------- RUIN: detach the chosen nodes -------
        let mut bld = ChainSetDeltaBuilder::new(cs);
        for &x in &to_remove {
            bld.detach(x);
        }

        // ------- REPAIR: "packing" greedy (strictly adjacent to borders/gaps) -------
        let mut ivars = search_state.interval_vars().to_vec();
        let mut dvars = search_state.decision_vars().to_vec();
        let mut placed = 0usize;
        let scan_cap = self.params.scan_cap.unwrap_or(usize::MAX);
        let mut scans = 0usize;

        'each_removed: for &x in &to_remove {
            // refresh overlay
            let ov = ChainSetOverlay::new(cs, bld.delta());

            // try all chains; try to insert x after gaps bordering either start, real nodes, or near end
            for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let (start, end) = (cs.start_of_chain(ci), cs.end_of_chain(ci));

                // walk (d -> succ) pairs, but prefer “packing” positions:
                let mut d = start;
                while let Some(succ) = ov.next_node(d) {
                    if scans >= scan_cap {
                        break 'each_removed;
                    }
                    scans += 1;

                    // Strict adjacency notion in ordering: insert immediately after d
                    // (true temporal/space adjacency will be validated by scheduler)
                    if d != x {
                        // trial = current committed + insertion
                        let cur = bld.delta().clone();
                        let mut tb = ChainSetDeltaBuilder::new_with_delta(cs, cur);

                        if tb.insert_after(d, x) {
                            // reset decision/interval vars for chain slice, like you already do elsewhere
                            let ov2 = ChainSetOverlay::new(cs, tb.delta());
                            let cview = ov2.chain(ci);
                            let mut iv = ivars.clone();
                            let mut dv = dvars.clone();

                            // naive reset of the chain slice to give scheduler freedom
                            {
                                let mut n_opt = cview.first_real_node(cview.start());
                                let mut guard = search_state
                                    .model()
                                    .flexible_requests_len()
                                    .saturating_add(2);
                                while let Some(n) = n_opt {
                                    if guard == 0 || n == cview.end() {
                                        break;
                                    }
                                    let i = crate::model::index::RequestIndex(n.get()).get();
                                    dv[i] = crate::core::decisionvar::DecisionVar::Unassigned;
                                    let w = search_state.model().feasible_intervals()[i];
                                    iv[i].start_time_lower_bound = w.start();
                                    iv[i].start_time_upper_bound = w.end();
                                    n_opt = cview.next_real(n);
                                    guard = guard.saturating_sub(1);
                                }
                            }

                            // validate: first-feasible schedule of the slice
                            let ok = self
                                .scheduler
                                .schedule_chain_slice(
                                    search_state.model(),
                                    cview,
                                    cview.start(),
                                    None,
                                    &mut iv,
                                    &mut dv,
                                )
                                .is_ok();

                            if ok {
                                // commit this insertion
                                bld.replace_delta(tb.build());
                                // keep the “best known reset” for next node — cheap win
                                ivars = iv;
                                dvars = dv;
                                placed += 1;
                                continue 'each_removed;
                            }
                        }
                    }

                    if succ == end {
                        break;
                    }
                    d = succ;
                }
            }
            // not placed? leave unassigned (intentional).
        }

        let delta = bld.build();
        if placed == 0 { None } else { Some(delta) }
    }

    fn name(&self) -> &str {
        "ALNS(ShawDestroy+PackingInsert)"
    }
}
