// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};

use crate::{
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    engine::traits::Opening,
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    state::{
        chain_set::{
            base::ChainSet,
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
        },
        search_state::SolverSearchState,
    },
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GreedyOpening;

impl<'model, 'problem, T> Opening<'model, 'problem, T> for GreedyOpening
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Send + Sync + Into<Cost>,
{
    fn build(
        &self,
        model: &'model SolverModel<'problem, T>,
    ) -> SolverSearchState<'model, 'problem, T> {
        let r_len = model.flexible_requests_len();
        let b_len = model.berths_len();

        // 1) Interval vars mirror feasible windows
        let interval_vars: Vec<IntervalVar<T>> = model
            .feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect();

        // 2) Start with everything unassigned
        let mut decision_vars = vec![DecisionVar::Unassigned; r_len];

        // 3) Greedy: earliest start first, try shortest processing time berth
        let mut order: Vec<usize> = (0..r_len).collect();
        order.sort_by_key(|&i| model.feasible_intervals()[i].start());

        for i in order {
            let ri = RequestIndex(i);
            let req_tw = model.feasible_intervals()[i];

            // feasible (berth, duration)
            let mut options: Vec<(BerthIndex, TimeDelta<T>)> = (0..b_len)
                .filter_map(|b| {
                    let bi = BerthIndex(b);
                    match model.processing_time(ri, bi) {
                        Some(Some(dt)) => Some((bi, dt)),
                        _ => None,
                    }
                })
                .collect();
            options.sort_by_key(|&(_, dt)| dt);

            // try to place
            for (bi, dur) in options {
                if let Some(start) = earliest_fit_in_calendar(model, bi, req_tw, dur) {
                    decision_vars[i] = DecisionVar::assigned(bi, start);
                    break;
                }
            }
        }

        // 4) Build chains directly from the assignments
        let mut per_berth: Vec<Vec<(NodeIndex, _ /*TimePoint<T>*/)>> = vec![Vec::new(); b_len];
        for (i, dv) in decision_vars.iter().enumerate() {
            if let DecisionVar::Assigned(a) = dv {
                per_berth[a.berth_index.get()].push((NodeIndex(i), a.start_time));
            }
        }
        for seq in &mut per_berth {
            seq.sort_by_key(|&(_, t)| t);
        }

        let mut chain_set = ChainSet::new(r_len, b_len);
        for (b, _) in per_berth.iter().enumerate().take(b_len) {
            let start = chain_set.start_of_chain(ChainIndex(b));
            let end = chain_set.end_of_chain(ChainIndex(b));

            // link: start -> n0 -> n1 -> ... -> end
            let mut tail = start;
            for (ni, _) in per_berth[b].iter().copied() {
                chain_set.set_next(tail, ni);
                tail = ni;
            }
            chain_set.set_next(tail, end);
        }

        // 5) Seed state (engine will recompute costs)
        SolverSearchState::new(model, chain_set, interval_vars, decision_vars, 0, 0)
    }
}

/// Finds the earliest feasible start time inside a berth’s calendar that also respects the request TW.
/// Returns None if it can’t fit anywhere.
fn earliest_fit_in_calendar<T>(
    model: &SolverModel<'_, T>,
    berth: BerthIndex,
    req_tw: TimeInterval<T>,
    dur: TimeDelta<T>,
) -> Option<TimePoint<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let cal = model.calendar_for_berth(berth)?;
    for slot in cal.free_intervals() {
        if let Some(start) = earliest_fit_in_slot(req_tw, *slot, dur) {
            return Some(start);
        }
    }
    None
}

/// Earliest fit inside a single free slot intersected with request TW.
/// Returns None if duration doesn’t fit.
fn earliest_fit_in_slot<T>(
    req_tw: TimeInterval<T>,
    free: TimeInterval<T>,
    dur: TimeDelta<T>,
) -> Option<TimePoint<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    // Intersection [lo, hi)
    let lo = max_tp(req_tw.start(), free.start());
    let hi = min_tp(req_tw.end(), free.end());

    // Latest feasible start = hi - dur
    let latest = hi.checked_sub(dur)?;
    if lo <= latest { Some(lo) } else { None }
}

#[inline]
fn max_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a >= b { a } else { b }
}
#[inline]
fn min_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a <= b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::traits::Opening;
    use berth_alloc_model::{
        common::{FixedKind, FlexibleKind},
        prelude::{Assignment, Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::builder::ProblemBuilder,
        problem::req::Request,
    };
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex(n)
    }

    fn build_problem_two_berths_two_flex() -> Problem<i64> {
        // Two berths [0,100)
        let b0 = Berth::from_windows(bid(0), [iv(0, 100)]);
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);

        // r0: allowed on both, shorter on berth 1
        let mut pt0 = BTreeMap::new();
        pt0.insert(bid(0), td(10));
        pt0.insert(bid(1), td(5));
        let r0 = Request::<FlexibleKind, i64>::new(rid(0), iv(0, 100), 1, pt0).unwrap();

        // r1: allowed on both, shorter on berth 0
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(0), td(3));
        pt1.insert(bid(1), td(7));
        let r1 = Request::<FlexibleKind, i64>::new(rid(1), iv(0, 100), 1, pt1).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b0);
        builder.add_berth(b1);
        builder.add_flexible(r0);
        builder.add_flexible(r1);

        builder.build().expect("problem should build")
    }

    #[test]
    fn test_greedy_assigns_shortest_earliest() {
        let p = build_problem_two_berths_two_flex();
        let m = SolverModel::from_problem(&p).unwrap();

        let state = GreedyOpening.build(&m);

        // r0 -> berth 1 at t=0 (shortest 5)
        // r1 -> berth 0 at t=0 (shortest 3)
        let dv = state.decision_vars();
        assert!(dv[0].is_assigned());
        assert!(dv[1].is_assigned());

        let a0 = dv[0].as_assigned().unwrap();
        let a1 = dv[1].as_assigned().unwrap();
        assert_eq!(a0.berth_index, bi(1));
        assert_eq!(a1.berth_index, bi(0));
        assert_eq!(a0.start_time, tp(0));
        assert_eq!(a1.start_time, tp(0));

        // Chains contain the expected nodes on each berth
        let chain0: Vec<_> = state.chain_set().iter_chain(ChainIndex(0)).collect();
        let chain1: Vec<_> = state.chain_set().iter_chain(ChainIndex(1)).collect();
        // On berth 0: request 1
        assert_eq!(chain0, vec![NodeIndex(1)]);
        // On berth 1: request 0
        assert_eq!(chain1, vec![NodeIndex(0)]);
    }

    #[test]
    fn test_greedy_respects_fixed_calendar() {
        // One berth [0,50), fixed occupies [10,20)
        // One flexible request with TW [15,40), dur 5 => should be placed at 20 (after fixed)
        let mut builder = ProblemBuilder::new();
        let b = Berth::from_windows(bid(0), [iv(0, 50)]);
        builder.add_berth(b.clone());

        // Fixed request (dur 10) starting at 10
        let mut pt_fixed = BTreeMap::new();
        pt_fixed.insert(bid(0), td(10));
        let r_fixed = Request::<FixedKind, i64>::new(rid(1000), iv(0, 50), 0, pt_fixed).unwrap();
        let a_fixed = Assignment::<FixedKind, i64>::new(r_fixed, b.clone(), tp(10)).unwrap();
        builder.add_fixed(a_fixed);

        // Flexible request allowed on berth 0, dur 5, window starts at 15
        let mut pt_flex = BTreeMap::new();
        pt_flex.insert(bid(0), td(5));
        let r_flex = Request::<FlexibleKind, i64>::new(rid(0), iv(15, 40), 1, pt_flex).unwrap();
        builder.add_flexible(r_flex);

        let p = builder.build().expect("problem should build");
        let m = SolverModel::from_problem(&p).unwrap();

        let state = GreedyOpening.build(&m);

        let dv = state.decision_vars();
        assert!(dv[0].is_assigned());
        let a = dv[0].as_assigned().unwrap();
        assert_eq!(a.berth_index, bi(0));
        // Must start at 20 due to fixed [10,20)
        assert_eq!(a.start_time, tp(20));

        // Chain for berth 0 contains the single node 0
        let chain: Vec<_> = state.chain_set().iter_chain(ChainIndex(0)).collect();
        assert_eq!(chain, vec![NodeIndex(0)]);
    }

    #[test]
    fn test_greedy_leaves_unassignable_unassigned() {
        // One berth [0,30)
        // Fixed occupy [0,10) and [15,30) -> free gap = [10,15) of length 5
        // One flexible request needs dur 10 => cannot fit into the remaining free gap
        let mut builder = ProblemBuilder::new();
        let b = Berth::from_windows(bid(0), [iv(0, 30)]);
        builder.add_berth(b.clone());

        // Fixed #1: dur 10 at t=0 -> [0,10)
        let mut pt_fx1 = BTreeMap::new();
        pt_fx1.insert(bid(0), td(10));
        let r_fx1 = Request::<FixedKind, i64>::new(rid(1000), iv(0, 30), 0, pt_fx1).unwrap();
        let a_fx1 = Assignment::<FixedKind, i64>::new(r_fx1, b.clone(), tp(0)).unwrap();
        builder.add_fixed(a_fx1);

        // Fixed #2: dur 15 at t=15 -> [15,30)
        let mut pt_fx2 = BTreeMap::new();
        pt_fx2.insert(bid(0), td(15));
        let r_fx2 = Request::<FixedKind, i64>::new(rid(1001), iv(0, 30), 0, pt_fx2).unwrap();
        let a_fx2 = Assignment::<FixedKind, i64>::new(r_fx2, b.clone(), tp(15)).unwrap();
        builder.add_fixed(a_fx2);

        // Flexible needs dur 10 within [0,30), allowed on berth 0
        // This request is feasible w.r.t. its own window, but cannot fit due to fixed blocks.
        let mut pt = BTreeMap::new();
        pt.insert(bid(0), td(10));
        let r = Request::<FlexibleKind, i64>::new(rid(0), iv(0, 30), 1, pt).unwrap();
        builder.add_flexible(r);

        let p = builder.build().expect("problem should build");
        let m = SolverModel::from_problem(&p).unwrap();

        let state = GreedyOpening.build(&m);

        // Unassignable due to calendar => remains Unassigned
        assert!(matches!(state.decision_vars()[0], DecisionVar::Unassigned));

        // No nodes in the only chain
        let nodes_in_chain0: Vec<_> = state.chain_set().iter_chain(ChainIndex(0)).collect();
        assert!(nodes_in_chain0.is_empty());
    }
}
