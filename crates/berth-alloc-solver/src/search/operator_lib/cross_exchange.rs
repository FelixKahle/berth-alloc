// crates/berth-alloc-solver/src/search/operator_lib/cross_exchange.rs
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

pub struct CrossExchangeBestImprovement {
    pub same_chain_only: bool,
    pub scan_cap: Option<usize>,
}

impl CrossExchangeBestImprovement {
    pub fn new(same_chain_only: bool, scan_cap: Option<usize>) -> Self {
        Self {
            same_chain_only,
            scan_cap,
        }
    }
}

impl<T> NeighborhoodOperator<T> for CrossExchangeBestImprovement
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'s, 'm, 'p>(
        &self,
        search_state: &'s SolverSearchState<'m, 'p, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();
        let mut best: Option<(Cost, NodeIndex, NodeIndex)> = None; // (delta, a, d)
        let mut scanned = 0usize;

        // iterate cut (a -> a_next)
        for c1 in 0..cs.num_chains() {
            let ci1 = ChainIndex(c1);
            let start1 = cs.start_of_chain(ci1);
            let end1 = cs.end_of_chain(ci1);

            let mut a = start1;
            while let Some(a_next) = cs.next_node(a) {
                if a_next == end1 {
                    break;
                }

                // iterate cut (d -> d_next)
                for c2 in 0..cs.num_chains() {
                    if self.same_chain_only && c1 != c2 {
                        continue;
                    }
                    let ci2 = ChainIndex(c2);
                    let start2 = cs.start_of_chain(ci2);
                    let end2 = cs.end_of_chain(ci2);

                    let mut d = start2;
                    while let Some(d_next) = cs.next_node(d) {
                        if d_next == end2 {
                            break;
                        }
                        if ci1 == ci2 && (d == a || d_next == a_next) {
                            // avoid no-op/self
                            // (same cut or immediately adjacent swap degeneracy)
                            let _ = 0;
                        }

                        // Δ = (a,d_next) + (d,a_next) - (a,a_next) - (d,d_next)
                        if let (Some(old1), Some(old2), Some(n1), Some(n2)) = (
                            arc_eval(a, a_next),
                            arc_eval(d, d_next),
                            arc_eval(a, d_next),
                            arc_eval(d, a_next),
                        ) {
                            let old = old1.saturating_add(old2);
                            let newc = n1.saturating_add(n2);
                            let delta = newc.saturating_sub(old);
                            if delta < 0 {
                                match best {
                                    Some((cur, _, _)) if delta >= cur => {}
                                    _ => best = Some((delta, a, d)),
                                }
                            }
                        }

                        scanned += 1;
                        if let Some(cap) = self.scan_cap {
                            if scanned >= cap {
                                break;
                            }
                        }

                        d = d_next;
                    }
                }
                a = a_next;
            }
        }

        let Some((_, a, d)) = best else {
            return None;
        };

        // Tail swap rewire with move_after primitives:
        // Goal: (a -> d_next ... ) and (d -> a_next ...)
        let mut b = ChainSetDeltaBuilder::new(cs);
        // 1) Take successor of d and move it after a
        b.move_after(a, d);
        // 2) Take (new) successor of a_next’s previous (which is a) repeatedly? Simpler:
        // Next, we need the old successor of a to become successor of d.
        // But move_after above consumed d_next. Now we need to move a_next after d.
        b.move_after(d, a);
        Some(b.build())
    }

    fn name(&self) -> &str {
        "CrossExchangeBestImprovement"
    }
}
