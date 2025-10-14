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
use berth_alloc_core::prelude::{Cost, TimeDelta, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, SaturatingSub, Zero};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GreedyOpening;

impl<'model, 'problem, T> Opening<'model, 'problem, T> for GreedyOpening
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Send + Sync + Into<Cost> + Zero + SaturatingSub,
{
    fn build(
        &self,
        model: &'model SolverModel<'problem, T>,
    ) -> SolverSearchState<'model, 'problem, T> {
        let r_len = model.flexible_requests_len();
        let b_len = model.berths_len();

        let interval_vars: Vec<IntervalVar<T>> = model
            .feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect();

        let mut decision_vars = vec![DecisionVar::Unassigned; r_len];

        // --- NEW: Simplified per-berth state ---
        // The cursor is the only state we need. It tracks the end time of the last-placed job.
        let mut cursors: Vec<TimePoint<T>> = vec![TimePoint::zero(); b_len];

        // --- Same greedy order ---
        let mut order: Vec<usize> = (0..r_len).collect();
        order.sort_by_key(|&i| model.feasible_intervals()[i].start());

        for i in order {
            let ri = RequestIndex(i);
            let req_tw = model.feasible_intervals()[i];

            // Find best berth (shortest duration)
            let mut options: Vec<(BerthIndex, TimeDelta<T>)> = (0..b_len)
                .filter_map(|b| {
                    let bi = BerthIndex(b);
                    model.processing_time(ri, bi).flatten().map(|dt| (bi, dt))
                })
                .collect();
            options.sort_by_key(|&(_, dt)| dt);

            // Try to place on the best option
            for (bi, dur) in options {
                // The candidate start time is the later of the request's own start time
                // and the current end-time cursor of the berth.
                let candidate_start = max_tp(req_tw.start(), cursors[bi.get()]);

                // Now, find the actual earliest placement time, respecting the fixed calendar.
                if let Some(assigned_start) =
                    find_earliest_fit_in_berth_calendar(model, bi, candidate_start, dur)
                {
                    // Check if this placement is still valid for the request's time window.
                    if assigned_start <= req_tw.end().saturating_sub(dur) {
                        // Success! Assign and update the cursor.
                        decision_vars[i] = DecisionVar::assigned(bi, assigned_start);
                        cursors[bi.get()] =
                            assigned_start.checked_add(dur).expect("end time must fit");
                        break; // Move to the next request
                    }
                }
            }
        }

        // 4) build chains from assigned starts (already non-overlapping per berth)
        let mut per_berth: Vec<Vec<(NodeIndex, TimePoint<T>)>> = vec![Vec::new(); b_len];
        for (i, dv) in decision_vars.iter().enumerate() {
            if let DecisionVar::Assigned(a) = dv {
                per_berth[a.berth_index.get()].push((NodeIndex(i), a.start_time));
            }
        }
        for seq in &mut per_berth {
            seq.sort_by_key(|&(_, t)| t);
        }

        let mut chain_set = ChainSet::new(r_len, b_len);
        for b in 0..b_len {
            let start = chain_set.start_of_chain(ChainIndex(b));
            let end = chain_set.end_of_chain(ChainIndex(b));
            let mut tail = start;
            for (ni, _) in per_berth[b].iter().copied() {
                chain_set.set_next(tail, ni);
                tail = ni;
            }
            chain_set.set_next(tail, end);
        }

        // 5) seed state (engine recomputes costs)
        SolverSearchState::new(model, chain_set, interval_vars, decision_vars, 0, 0)
    }
}

fn find_earliest_fit_in_berth_calendar<T>(
    model: &SolverModel<'_, T>,
    berth: BerthIndex,
    candidate_start: TimePoint<T>,
    dur: TimeDelta<T>,
) -> Option<TimePoint<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let calendar = model.calendar_for_berth(berth)?;

    // Find the first free slot that could possibly contain our job
    for free_slot in calendar.free_intervals() {
        // If the free slot ends before our candidate can even start, skip it.
        if free_slot.end() < candidate_start {
            continue;
        }

        // The actual start time is the later of our candidate time and the beginning of the free slot.
        let start = max_tp(candidate_start, free_slot.start());

        // Check if the duration fits within this slot from our calculated start time.
        if let Some(end) = start.checked_add(dur) {
            if end <= free_slot.end() {
                // It fits! This is the earliest possible placement.
                return Some(start);
            }
        }
    }

    // Scanned all free slots and couldn't find a fit.
    None
}

#[inline]
fn max_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a >= b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::traits::Opening;
    use berth_alloc_core::prelude::TimeInterval;
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
