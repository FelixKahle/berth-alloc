use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, Zero};

use crate::{
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    state::chain_set::{
        index::NodeIndex,
        view::{ChainRef, ChainSetView},
    },
};

#[derive(Debug)]
pub struct SimpleChainArcEval<'m, T> {
    service: Vec<Option<TimeDelta<T>>>, // PT[i] on this chain's berth
    tw: Vec<TimeInterval<T>>,           // feasible windows per request (global)
    _m: std::marker::PhantomData<&'m T>,
}

impl<'m, T> SimpleChainArcEval<'m, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Zero,
{
    // Change: accept any ChainSetView
    pub fn build<V>(model: &'m SolverModel<'m, T>, chain: ChainRef<'_, V>) -> Self
    where
        V: ChainSetView,
    {
        let berth = BerthIndex(chain.chain_index().get()); // 1:1 chain↔berth by construction
        let n = model.flexible_requests_len();

        let mut service = Vec::with_capacity(n);
        for i in 0..n {
            let ri = RequestIndex(i);
            let dt = match model.processing_time(ri, berth) {
                Some(Some(d)) => Some(d),
                _ => None,
            };
            service.push(dt);
        }

        let tw = model.feasible_intervals().to_vec();
        Self {
            service,
            tw,
            _m: std::marker::PhantomData,
        }
    }

    #[inline]
    fn e(&self, i: usize) -> TimePoint<T> {
        self.tw[i].start()
    }
    #[inline]
    fn lstart(&self, i: usize) -> Option<TimePoint<T>> {
        let s = self.service[i]?;
        self.tw[i].end().checked_sub(s)
    }

    /// Returns:
    /// - Some(cost) for arcs that are structurally valid and temporally feasible.
    /// - None for infeasible arcs (e.g., incompatible berth, TW violation, overflow).
    ///
    /// Semantics:
    /// - v == head: invalid => None
    /// - u == head && v == tail: neutral => Some(0)
    /// - x -> tail: neutral close => Some(0)
    /// - u == head && v == j: small regularizer (tie-breaker) if j is allowed on berth; else None
    /// - regular u == i, v == j: cost = local wait surrogate + small regularizer; None on infeasibility
    fn cost_uv(
        &self,
        head: NodeIndex,
        tail: NodeIndex,
        u: NodeIndex,
        v: NodeIndex,
    ) -> Option<Cost> {
        // Disallow arcs that “enter” the head sentinel.
        if v == head {
            return None;
        }
        // Sentinels:
        if u == head && v == tail {
            return Some(0); // neutral
        }
        if v == tail {
            return Some(0); // neutral close
        }

        let j = v.get();

        // head->j: require berth compatibility
        if u == head {
            if self.service[j].is_none() {
                return None; // incompatible berth
            }
            // small tie-breaker to avoid 0-ties
            let reg: Cost = 1;
            return Some(reg);
        }

        // regular i->j
        let i = u.get();
        let s_i = self.service[i]?; // incompatible berth for i -> None
        let s_j = self.service[j]?; // incompatible berth for j -> None

        // Compute surrogate times from global feasible windows (not current DV).
        let fin_i = self.e(i).checked_add(s_i)?;
        let e_j = self.e(j);
        let l_j = self.lstart(j)?; // latest feasible start for j

        // Proposed start for j if linked after i.
        let start_j = if fin_i >= e_j { fin_i } else { e_j };

        // Waiting cost (non-negative)
        let wait: Cost = if start_j > e_j {
            let d: TimeDelta<T> = start_j - e_j;
            d.value().into()
        } else {
            0
        };

        // Lateness penalty if we’d start past latest; keep finite instead of None
        let late: Cost = if start_j > l_j {
            let d: TimeDelta<T> = start_j - l_j;
            d.value().into()
        } else {
            0
        };

        // Tiny structure tie-breaker
        let reg: Cost = if i != j { 1 } else { 0 };

        Some(wait.saturating_add(late).saturating_add(reg))
    }
}

pub fn make_simple_chain_arc_evaluator<'m, T, V>(
    model: &'m SolverModel<'m, T>,
    chain: ChainRef<'_, V>,
) -> impl Fn(NodeIndex, NodeIndex) -> Option<Cost> + 'm
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Zero + 'm,
    V: ChainSetView,
{
    let eval = SimpleChainArcEval::<T>::build(model, chain);
    let head = chain.start();
    let tail = chain.end();
    // Accept sentinels or real nodes; allow v that is not in this chain (it will be “moved”).
    move |from: NodeIndex, to: NodeIndex| eval.cost_uv(head, tail, from, to)
}
