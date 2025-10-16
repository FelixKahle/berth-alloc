// crates/berth-alloc-solver/src/search/operator_lib/regret.rs

use crate::{
    eval::ArcEvaluator,
    scheduling::traits::Scheduler,
    search::operator::NeighborhoodOperator,
    state::{
        chain_set::{
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
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::num::NonZeroUsize;

pub struct RegretKInsert<T, S> {
    pub k: NonZeroUsize,
    pub scan_cap: Option<usize>,
    pub scheduler: S,
    _p: std::marker::PhantomData<(T, S)>,
}

impl<T, S> RegretKInsert<T, S> {
    pub fn new(k: NonZeroUsize, scan_cap: Option<usize>, scheduler: S) -> Self {
        Self {
            k,
            scan_cap,
            scheduler,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T, S> NeighborhoodOperator<T> for RegretKInsert<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
    S: Scheduler<T>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        ss: &'state SolverSearchState<'model, 'problem, T>,
        _arc: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        //println!("RegretKInsert");

        let cs = ss.chain_set();
        // collect unassigned nodes (isolated x->x after previous ruins)
        let mut unassigned: Vec<NodeIndex> = Vec::new();
        for c in 0..cs.num_chains() {
            let ci = ChainIndex(c);
            let mut u = cs.start_of_chain(ci);
            while let Some(v) = cs.next_node(u) {
                if v == cs.end_of_chain(ci) {
                    break;
                }
                if cs.next_node(v) == Some(v) && cs.prev_node(v) == Some(v) {
                    unassigned.push(v);
                }
                u = v;
            }
        }
        if unassigned.is_empty() {
            return None;
        }

        let bld = ChainSetDeltaBuilder::new(cs);
        let ivars = ss.interval_vars().to_vec();
        let dvars = ss.decision_vars().to_vec();

        let mut scans = 0usize;
        let scan_cap = self.scan_cap.unwrap_or(usize::MAX);

        // One outer pass: insert one node by regret; return that partial delta.
        // (Keeping it light-weight makes it play nicely with SA.)
        let mut best_pick: Option<(NodeIndex, ChainIndex, NodeIndex, Cost, Cost)> = None;
        // (x, chain, pred, best_cost, k_cost)

        for &x in &unassigned {
            // enumerate (chain, pred) positions and compute incremental objective via scheduler
            let mut candidates: Vec<(ChainIndex, NodeIndex, Cost)> = Vec::new();

            'scan: for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let (start, end) = (cs.start_of_chain(ci), cs.end_of_chain(ci));
                let mut d = start;
                while let Some(succ) = cs.next_node(d) {
                    if scans >= scan_cap {
                        break 'scan;
                    }
                    scans += 1;

                    if d != x {
                        let cur = bld.delta().clone();
                        let mut tb = ChainSetDeltaBuilder::new_with_delta(cs, cur);
                        if tb.insert_after(d, x) {
                            // schedule slice
                            let ov2 = ChainSetOverlay::new(cs, tb.delta());
                            let cview = ov2.chain(ci);
                            let mut iv = ivars.clone();
                            let mut dv = dvars.clone();
                            // cheap reset as earlier…
                            let ok = self
                                .scheduler
                                .schedule_chain_slice(
                                    ss.model(),
                                    cview,
                                    cview.start(),
                                    None,
                                    &mut iv,
                                    &mut dv,
                                )
                                .is_ok();
                            if ok {
                                // use true objective delta available from your Context Evaluator later;
                                // here we approximate with 0 to pick any feasible position (blind regret).
                                candidates.push((ci, d, 0));
                            }
                        }
                    }

                    if succ == end {
                        break;
                    }
                    d = succ;
                }
            }

            if candidates.len() >= self.k.get() {
                // sort by (approx) cost; take best and k-th
                candidates.sort_by(|a, b| a.2.cmp(&b.2));
                let best = candidates[0];
                let kth = candidates[self.k.get() - 1];
                let regret = kth.2.saturating_sub(best.2);
                match &mut best_pick {
                    None => best_pick = Some((x, best.0, best.1, best.2, kth.2)),
                    Some((_, _, _, best_best, best_k)) => {
                        let cur_reg = best_k.saturating_sub(*best_best);
                        if regret > cur_reg {
                            best_pick = Some((x, best.0, best.1, best.2, kth.2));
                        }
                    }
                }
            }
        }

        if let Some((x, _, pred, _b, _k)) = best_pick {
            // commit chosen insertion
            let mut tb = ChainSetDeltaBuilder::new_with_delta(cs, bld.build());
            if tb.insert_after(pred, x) {
                return Some(tb.build());
            }
        }
        None
    }

    fn name(&self) -> &str {
        "ALNS(RegretKInsert)"
    }
}
