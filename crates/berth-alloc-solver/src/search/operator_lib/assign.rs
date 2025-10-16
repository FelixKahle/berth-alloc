// crates/berth-alloc-solver/src/search/operator_library/assign_unassigned_first_feasible.rs

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

/// Tries to (re)insert any isolated node (x->x) back into the solution.
/// Strategy:
///   * Collect all real nodes x with next(x) == x (unassigned/isolated).
///   * For each x, scan all chains and gaps (p -> succ(p)) and compute a proxy delta:
///       Δ ≈ cost(p->x) + cost(x->succ) - cost(p->succ)
///     using the provided `ArcEvaluator`.
///   * Pick the best (lowest Δ) slot over all x and return a single insertion delta.
/// Notes:
///   * This operator returns a delta even if Δ >= 0 (worsening). SA / acceptance
///     criteria can still take it, and it helps recover from perturbations.
///   * Feasibility is *not* hard-checked here; the CandidateEvaluator/scheduler
///     will validate when scoring the candidate.
pub struct AssignUnassignedFirstFeasible;

impl AssignUnassignedFirstFeasible {
    pub fn new() -> Self {
        Self
    }
}

impl<T> NeighborhoodOperator<T> for AssignUnassignedFirstFeasible
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
{
    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        arc_eval: &ArcEvaluator,
    ) -> Option<ChainSetDelta> {
        let cs: &ChainSet = search_state.chain_set();

        // 1) Gather isolated (unassigned) real nodes: next(x) == x (and not sentinel)
        let mut isolated = Vec::<NodeIndex>::new();
        let total = cs.num_total_nodes();
        for idx in 0..total {
            let x = NodeIndex(idx);
            if cs.is_sentinel_node(x) {
                continue;
            }
            if let Some(nx) = cs.next_node(x) {
                if nx == x {
                    isolated.push(x);
                }
            }
        }
        if isolated.is_empty() {
            return None; // nothing to (re)assign
        }

        // Helper to compute proxy delta for inserting `x` after `p`
        let proxy_delta = |p: NodeIndex, x: NodeIndex| -> Option<Cost> {
            let succ = cs.next_node(p)?;
            // avoid p == x (inserting after itself while it's isolated is a no-op)
            if p == x {
                return None;
            }
            let ax = arc_eval(p, x)?;
            let xb = arc_eval(x, succ)?;
            let dd = arc_eval(p, succ)?;
            Some(ax.saturating_add(xb).saturating_sub(dd))
        };

        // 2) Scan all chains/gaps for the best (p, x)
        let mut best: Option<(Cost, NodeIndex, NodeIndex)> = None; // (Δ, p, x)
        for &x in &isolated {
            for c in 0..cs.num_chains() {
                let ci = ChainIndex(c);
                let (start, end) = (cs.start_of_chain(ci), cs.end_of_chain(ci));

                let mut p = start;
                loop {
                    let Some(succ) = cs.next_node(p) else {
                        break;
                    };
                    if succ == p {
                        break; // safety
                    }

                    // Try inserting `x` after `p` (before `succ`)
                    if let Some(delta) = proxy_delta(p, x) {
                        if best.is_none() || delta < best.as_ref().unwrap().0 {
                            best = Some((delta, p, x));
                        }
                    }

                    if succ == end {
                        break;
                    }
                    p = succ;
                }
            }
        }

        // 3) Build the delta for the best slot (even if Δ >= 0)
        if let Some((_d, p, x)) = best {
            let mut b = ChainSetDeltaBuilder::new(cs);
            // `insert_after` will detach `x` first if needed (x is isolated anyway).
            if b.insert_after(p, x) {
                return Some(b.build());
            }
        }

        None
    }

    fn name(&self) -> &str {
        "AssignUnassignedFirstFeasible"
    }
}
