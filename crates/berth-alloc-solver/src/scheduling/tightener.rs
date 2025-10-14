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
    core::intervalvar::IntervalVar,
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    scheduling::{
        err::{FeasiblyWindowViolationError, NotAllowedOnBerthError, SchedulingError},
        traits::Propagator,
    },
    state::chain_set::{
        index::NodeIndex,
        view::{ChainRef, ChainSetView},
    },
};
use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use std::cmp::{max, min};

/// A constraint propagator that tightens the start-time bounds of `IntervalVar`s along a chain.
///
/// This propagator is a core component of constraint-based scheduling, reducing the search
/// space by eliminating impossible time ranges for each request. It operates in two sequential passes:
///
/// 1.  **Forward Pass**: Iterates from the first to the last request in the chain. For each request,
///     it calculates the earliest possible start time based on its predecessor's earliest finish time,
///     its own time windows, and calendar availability. This pass raises the `start_time_lower_bound` (LB)
///     of each `IntervalVar`.
///
/// 2.  **Backward Pass**: Iterates from the last to the first request. For each request, it calculates
///     the latest possible start time based on its successor's latest start time, its own windows,
///     and calendar availability. This pass lowers the `start_time_upper_bound` (UB) of each `IntervalVar`.
///
/// A key constraint is that each request must fit **entirely within a single free segment** of the berth's calendar.
/// The propagator uses monotonic cursors to scan the calendar efficiently and includes defensive guards against malformed chain cycles.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundsTightener;

impl<T> Propagator<T> for BoundsTightener
where
    T: Copy + Ord + CheckedAdd + CheckedSub + std::fmt::Debug,
{
    fn propagate_slice<'a, C: ChainSetView>(
        &self,
        solver_model: &SolverModel<'a, T>,
        chain: ChainRef<'_, C>,
        start_node: NodeIndex,
        end_node_exclusive: Option<NodeIndex>, // None => chain end
        iv: &mut [IntervalVar<T>],             // aligned by RequestIndex = node.get()
    ) -> Result<(), SchedulingError> {
        let (first_opt, end_excl) = chain.resolve_slice(start_node, end_node_exclusive);
        let Some(first) = first_opt else {
            return Ok(());
        };

        let Some(last_inclusive) = chain.prev_real(end_excl) else {
            return Ok(()); // The slice is empty
        };

        let b = BerthIndex(chain.chain_index().get());
        let cal = solver_model.calendar_for_berth(b).ok_or_else(|| {
            SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                RequestIndex(chain.start().get()),
            ))
        })?;
        let free = cal.free_intervals();

        let mut earliest_finish_of_pred = if let Some(pred) = chain.prev_real(first) {
            let r = RequestIndex(pred.get());
            let pt = solver_model
                .processing_time(r, b)
                .flatten()
                .ok_or_else(|| {
                    SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(r, b))
                })?;
            iv[r.get()].start_time_lower_bound.checked_add(pt)
        } else {
            None
        };

        let mut seg_idx = 0usize;
        {
            let mut guard = iv.len().saturating_add(2);
            let mut n = Some(first);
            while let Some(cur) = n {
                if guard == 0 {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(RequestIndex(cur.get())),
                    ));
                }
                guard -= 1;

                if cur == end_excl {
                    break;
                } // respect exclusive bound

                let r = RequestIndex(cur.get());
                let i = r.get();
                let w = solver_model.feasible_intervals()[i];
                let pt = solver_model
                    .processing_time(r, b)
                    .flatten()
                    .ok_or_else(|| {
                        SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(r, b))
                    })?;

                let lb_from_pred = earliest_finish_of_pred.unwrap_or_else(|| w.start());
                let (eff_lb, eff_ub) = start_bounds_lb_pass(w, &iv[i], lb_from_pred, pt, r)?;
                if eff_lb > eff_ub {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(r),
                    ));
                }

                seg_idx = advance_to_segment(free, seg_idx, eff_lb);
                let (new_lb, new_finish) =
                    earliest_fit_in_calendar_full_fit(free, &mut seg_idx, eff_lb, eff_ub, pt)
                        .ok_or_else(|| {
                            SchedulingError::FeasiblyWindowViolation(
                                FeasiblyWindowViolationError::new(r),
                            )
                        })?;

                if new_lb > iv[i].start_time_lower_bound {
                    iv[i].start_time_lower_bound = new_lb;
                }
                earliest_finish_of_pred = Some(new_finish);

                n = chain.next_real(cur);
            }
        }

        let mut latest_start_of_succ = chain
            .first_real_node(end_excl)
            .map(|s| iv[RequestIndex(s.get()).get()].start_time_upper_bound);

        let mut back_idx: Option<usize> = None;
        {
            let mut guard = iv.len().saturating_add(2);
            let mut n = Some(last_inclusive);
            while let Some(cur) = n {
                if guard == 0 {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(RequestIndex(cur.get())),
                    ));
                }
                guard -= 1;

                let r = RequestIndex(cur.get());
                let i = r.get();
                let w = solver_model.feasible_intervals()[i];
                let pt = solver_model
                    .processing_time(r, b)
                    .flatten()
                    .ok_or_else(|| {
                        SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(r, b))
                    })?;

                let cap = latest_start_of_succ.and_then(|s| s.checked_sub(pt));
                let (eff_lb, eff_ub) = start_bounds_ub_pass(w, &iv[i], cap, pt, r)?;
                if eff_lb > eff_ub {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(r),
                    ));
                }

                let init =
                    *back_idx.get_or_insert_with(|| retreat_to_segment(free, free.len(), eff_ub));
                let (new_ub, _finish, used) = latest_fit_in_calendar_full_fit(
                    free, init, eff_lb, eff_ub, pt,
                )
                .ok_or_else(|| {
                    SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(r))
                })?;
                back_idx = Some(used);

                if new_ub < iv[i].start_time_upper_bound {
                    iv[i].start_time_upper_bound = new_ub;
                }

                latest_start_of_succ = Some(iv[i].start_time_upper_bound);

                // stop when we reach (and process) the first node
                if cur == first {
                    break;
                }
                n = chain.prev_real(cur);
            }
        }

        // Sanity check: ensure all interval variables in the slice have valid bounds
        // This catches any edge cases where bounds crossing wasn't detected earlier
        #[cfg(debug_assertions)]
        {
            let mut n = Some(first);
            let mut guard = iv.len().saturating_add(1); // More defensive guard
            while let Some(cur) = n {
                if guard == 0 {
                    // This should never happen with well-formed chains
                    debug_assert!(
                        false,
                        "Chain traversal exceeded expected length - possible cycle"
                    );
                    break;
                }
                guard -= 1;

                if cur == end_excl {
                    break;
                }

                let i = RequestIndex(cur.get()).get();
                debug_assert!(
                    iv[i].start_time_lower_bound <= iv[i].start_time_upper_bound,
                    "Invalid bounds for request {}: LB={:?} > UB={:?}",
                    i,
                    iv[i].start_time_lower_bound,
                    iv[i].start_time_upper_bound
                );

                n = chain.next_real(cur);
            }
        }

        // In release builds, do a final check only if bounds were actually modified
        // This catches genuine feasibility issues that might have been missed
        {
            let mut n = Some(first);
            while let Some(cur) = n {
                if cur == end_excl {
                    break;
                }
                let i = RequestIndex(cur.get()).get();
                if iv[i].start_time_lower_bound > iv[i].start_time_upper_bound {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(RequestIndex(i)),
                    ));
                }
                n = chain.next_real(cur);
            }
        }

        Ok(())
    }
}

/// Computes the effective start-time bounds for the forward pass (tightening LBs).
/// This version assumes a closed start-time domain [L, U].
#[inline(always)]
fn start_bounds_lb_pass<T: Copy + Ord + CheckedSub>(
    feasible_window: TimeInterval<T>,
    ivar: &IntervalVar<T>,
    earliest_precedence_time: TimePoint<T>,
    processing_time: TimeDelta<T>,
    req: RequestIndex,
) -> Result<(TimePoint<T>, TimePoint<T>), SchedulingError> {
    // Combine precedence, request window, and prior lower bound.
    let lower_bound = max(
        earliest_precedence_time,
        max(ivar.start_time_lower_bound, feasible_window.start()),
    );

    // Latest start allowed by the request’s own feasible window.
    let ub_from_window = feasible_window
        .end()
        .checked_sub(processing_time)
        .ok_or_else(|| {
            SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(req))
        })?;

    // Latest start allowed by current domain (already a start bound → no subtraction).
    let upper_bound = min(ivar.start_time_upper_bound, ub_from_window);

    Ok((lower_bound, upper_bound))
}

/// Computes the effective start-time bounds for the backward pass (tightening UBs).
/// Also assumes a closed start-time domain [L, U].
#[inline(always)]
fn start_bounds_ub_pass<T: Copy + Ord + CheckedSub>(
    feasible_window: TimeInterval<T>,
    ivar: &IntervalVar<T>,
    precedence_cap_on_start: Option<TimePoint<T>>,
    processing_time: TimeDelta<T>,
    req: RequestIndex,
) -> Result<(TimePoint<T>, TimePoint<T>), SchedulingError> {
    // Lower bound respects both prior propagation and feasible window.
    let lower_bound = max(ivar.start_time_lower_bound, feasible_window.start());

    // Latest start allowed by feasible window.
    let ub_from_window = feasible_window
        .end()
        .checked_sub(processing_time)
        .ok_or_else(|| {
            SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(req))
        })?;

    // Combine domain UB, feasible window, and successor constraint.
    let mut upper_bound = min(ivar.start_time_upper_bound, ub_from_window);
    if let Some(cap) = precedence_cap_on_start {
        upper_bound = min(upper_bound, cap);
    }

    Ok((lower_bound, upper_bound))
}

/// Advances a cursor `index` through `free_intervals` to the first segment
/// that could possibly contain `time_point`.
#[inline(always)]
fn advance_to_segment<T: Copy + Ord>(
    free_intervals: &[TimeInterval<T>],
    start_index: usize,
    time_point: TimePoint<T>,
) -> usize {
    if start_index >= free_intervals.len() {
        return free_intervals.len();
    }

    // For small steps, linear search is faster due to cache locality
    if start_index + 8 >= free_intervals.len() {
        let mut index = start_index;
        while index < free_intervals.len() && free_intervals[index].end() <= time_point {
            index += 1;
        }
        return index;
    }

    // For larger jumps, use binary search from the start_index
    start_index
        + free_intervals[start_index..]
            .binary_search_by(|interval| interval.end().cmp(&time_point))
            .unwrap_or_else(|pos| pos)
}

/// Moves a cursor backward to the last segment that starts before `time_point`.
/// This is a conservative starting point for the backward search.
#[inline(always)]
fn retreat_to_segment<T: Copy + Ord>(
    free_intervals: &[TimeInterval<T>],
    mut index_hint: usize,
    time_point: TimePoint<T>,
) -> usize {
    if free_intervals.is_empty() {
        return 0;
    }
    if index_hint > free_intervals.len() {
        index_hint = free_intervals.len();
    }

    // Start from the last valid index
    let mut i = index_hint.saturating_sub(1);

    // Walk left from the hint until we find a segment that could contain the time point
    loop {
        let segment = free_intervals[i];
        // A segment can contain the time point if it starts before and ends after it
        if segment.start() <= time_point && time_point <= segment.end() {
            return i;
        }
        // Or if this segment starts before the time point (conservative approach)
        if segment.start() < time_point {
            return i;
        }
        if i == 0 {
            // If we've reached the beginning and no segment works, start from 0
            return 0;
        }
        i -= 1;
    }
}

/// Finds the EARLIEST start `s` in `[lb, ub]` that fits in a single calendar segment.
#[inline]
fn earliest_fit_in_calendar_full_fit<T: Copy + Ord + CheckedAdd + CheckedSub>(
    free_intervals: &[TimeInterval<T>],
    segment_cursor: &mut usize,
    lower_bound: TimePoint<T>,
    upper_bound: TimePoint<T>,
    processing_time: TimeDelta<T>,
) -> Option<(TimePoint<T>, TimePoint<T>)> {
    let mut i = advance_to_segment(free_intervals, *segment_cursor, lower_bound);
    while i < free_intervals.len() {
        let segment = free_intervals[i];
        let latest_start_in_segment = segment.end().checked_sub(processing_time)?;
        let earliest_possible_start = max(lower_bound, segment.start());
        let latest_possible_start = min(upper_bound, latest_start_in_segment);

        if earliest_possible_start <= latest_possible_start
            && let Some(end_time) = earliest_possible_start.checked_add(processing_time)
        {
            *segment_cursor = i;
            return Some((earliest_possible_start, end_time));
        }
        i += 1;
    }
    None
}

/// Finds the LATEST start `s` in `[lb, ub]` that fits in a single calendar segment.
#[inline]
fn latest_fit_in_calendar_full_fit<T: Copy + Ord + CheckedAdd + CheckedSub>(
    free_intervals: &[TimeInterval<T>],
    mut start_index: usize, // Start search from this segment index.
    lower_bound: TimePoint<T>,
    upper_bound: TimePoint<T>,
    processing_time: TimeDelta<T>,
) -> Option<(TimePoint<T>, TimePoint<T>, usize)> {
    if free_intervals.is_empty() {
        return None;
    }
    if start_index >= free_intervals.len() {
        start_index = free_intervals.len() - 1;
    }

    let mut i = start_index;
    loop {
        let segment = free_intervals[i];
        let latest_start_in_segment = segment.end().checked_sub(processing_time)?;
        let earliest_possible_start = max(lower_bound, segment.start());
        let latest_possible_start = min(upper_bound, latest_start_in_segment);

        if earliest_possible_start <= latest_possible_start {
            // Success: we found a valid window. Choose the LATEST time in it.
            if let Some(end_time) = latest_possible_start.checked_add(processing_time) {
                return Some((latest_possible_start, end_time, i));
            }
        }
        if i == 0 {
            break; // Reached the beginning of the calendar.
        }
        i -= 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain_set::{
        base::ChainSet,
        delta::{ChainNextRewire, ChainSetDelta},
        index::{ChainIndex, NodeIndex},
        view::ChainSetView,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    // ---- utilities ----
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
    #[inline]
    fn ri(n: usize) -> RequestIndex {
        RequestIndex(n)
    }

    fn build_problem(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let mut builder = ProblemBuilder::new();
        for (i, windows) in berths_windows.iter().enumerate() {
            let b = Berth::from_windows(bid(i), windows.iter().map(|&(s, e)| iv(s, e)));
            builder.add_berth(b);
        }
        for (i, &(ws, we)) in request_windows.iter().enumerate() {
            let mut map = BTreeMap::new();
            for (j, p) in processing[i].iter().copied().enumerate() {
                if let Some(dur) = p {
                    map.insert(bid(j), td(dur));
                }
            }
            let req =
                Request::<FlexibleKind, i64>::new(rid(i), iv(ws, we), 1, map).expect("request ok");
            builder.add_flexible(req);
        }
        builder.build().expect("problem ok")
    }

    fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
        m.feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect()
    }

    fn link_chain(cs: &mut ChainSet, c: usize, nodes: &[usize]) {
        let s = cs.start_of_chain(ChainIndex(c));
        let e = cs.end_of_chain(ChainIndex(c));
        if nodes.is_empty() {
            return;
        }
        let mut delta = ChainSetDelta::new();
        delta.push_rewire(ChainNextRewire::new(s, NodeIndex(nodes[0])));
        for w in nodes.windows(2) {
            delta.push_rewire(ChainNextRewire::new(NodeIndex(w[0]), NodeIndex(w[1])));
        }
        delta.push_rewire(ChainNextRewire::new(NodeIndex(*nodes.last().unwrap()), e));
        cs.apply_delta(delta);
    }

    #[test]
    fn test_backward_tightens_ub_to_latest_fit_in_segment() {
        // free: [0,5), [8,20); window [0,30), PT=4; UB large ⇒ latest start is 16
        let p = build_problem(&[vec![(0, 5), (8, 20)]], &[(0, 30)], &[vec![Some(4)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        // Make UB permissive
        ivars[0].start_time_upper_bound = tp(10_000);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        BoundsTightener.propagate(&m, c0, &mut ivars).unwrap();

        assert_eq!(ivars[0].start_time_upper_bound, tp(16));
    }

    #[test]
    fn test_backward_respects_precedence_cap_from_successor() {
        // free: [0,100); R0 pt=5, R1 pt=7.
        // Set successor UB = 12 ⇒ R0 must satisfy s0 + 5 ≤ 12 ⇒ s0 ≤ 7 (plus window).
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(7)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[1].start_time_upper_bound = tp(12); // successor cap

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]);

        let c0 = cs.chain(ChainIndex(0));
        BoundsTightener.propagate(&m, c0, &mut ivars).unwrap();

        assert_eq!(ivars[0].start_time_upper_bound, tp(7)); // 12 - pt0(5)
    }

    #[test]
    fn forward_and_backward_squeeze_to_meet_in_gap() {
        // free: [0,5), [10,15); pt=4; window [0,20)
        // Forward LB from 0 ⇒ s >= 0, first fit [0,4)
        // Backward UB with latest fit ≤ 15-4=11 ⇒ UB becomes 11
        let p = build_problem(&[vec![(0, 5), (10, 15)]], &[(0, 20)], &[vec![Some(4)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_upper_bound = tp(10_000);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        BoundsTightener.propagate(&m, c0, &mut ivars).unwrap();

        assert_eq!(ivars[0].start_time_lower_bound, tp(0));
        assert_eq!(ivars[0].start_time_upper_bound, tp(11));
    }

    #[test]
    fn test_backward_detects_infeasibility_when_ub_drops_below_lb() {
        // free [0,100), pt=10, window [0,100)
        // Force successor UB tight so predecessor UB < LB.
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[vec![Some(10)], vec![Some(1)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        // successor UB = 5 ⇒ pred s ≤ 5 - 10 = -5 (underflows to FWV via bounds check later)
        ivars[1].start_time_upper_bound = tp(5);
        // pred LB high:
        ivars[0].start_time_lower_bound = tp(0);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]);

        let c0 = cs.chain(ChainIndex(0));
        let err = BoundsTightener.propagate(&m, c0, &mut ivars).unwrap_err();
        match err {
            SchedulingError::FeasiblyWindowViolation(_) => {}
            x => panic!("expected FWV, got {:?}", x),
        }
    }

    #[test]
    fn test_backward_not_allowed_on_berth_propagates_error() {
        // two berths; req0 allowed only on berth1; chain is on berth0 -> error
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100)],
            &[vec![None, Some(10)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let err = BoundsTightener.propagate(&m, c0, &mut ivars).unwrap_err();
        match err {
            SchedulingError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request(), ri(0));
                assert_eq!(e.berth(), bi(0));
            }
            x => panic!("expected NotAllowedOnBerth, got {:?}", x),
        }
    }

    #[test]
    fn test_forward_pass_respects_ivar_ub_as_start_time() {
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(10)]]);
        let m = SolverModel::from_problem(&p).unwrap();
        let mut ivars = default_ivars(&m);

        // Manually set the narrow bounds that trigger the bug.
        ivars[0].start_time_lower_bound = tp(45);
        ivars[0].start_time_upper_bound = tp(50);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);
        let c0 = cs.chain(ChainIndex(0));

        // With the fix, this should now succeed.
        let result = BoundsTightener.propagate(&m, c0, &mut ivars);
        assert!(result.is_ok());

        // The forward pass should find the earliest start at 45.
        // The backward pass will tighten the upper bound to 50.
        assert_eq!(ivars[0].start_time_lower_bound, tp(45));
        assert_eq!(ivars[0].start_time_upper_bound, tp(50));
    }
}
