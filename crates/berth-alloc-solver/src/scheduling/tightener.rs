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
    scheduling::{
        err::{FeasiblyWindowViolationError, NotAllowedOnBerthError, SchedulingError},
        traits::Propagator,
    },
    state::{
        chain_set::view::ChainViewDyn,
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
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
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn propagate(
        &self,
        solver_model: &SolverModel<'_, T>,
        chain_view: &dyn ChainViewDyn,
        interval_variables: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError> {
        let berth_index = BerthIndex(chain_view.chain_index().get());
        let berth_calendar = solver_model
            .calendar_for_berth(berth_index)
            .ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                    RequestIndex(chain_view.start().get()),
                ))
            })?;
        let free_intervals = berth_calendar.free_intervals();

        // Forward pass: Propagate constraints forward to tighten Lower Bounds (LBs).
        {
            // Start traversal from the first actual request in the chain.
            let Some(mut current_node_index) = chain_view.real_start() else {
                return Ok(()); // Chain is empty, nothing to propagate.
            };
            let mut loop_guard = interval_variables.len().saturating_add(2);

            // This tracks the earliest time the *previous* request could finish.
            // For the first request, this is None.
            let mut earliest_finish_of_predecessor: Option<TimePoint<T>> = None;
            // A cursor for scanning the calendar's free segments. It only moves forward.
            let mut calendar_segment_cursor: usize = 0;

            loop {
                // Defensive guard against malformed (cyclic) chains.
                if loop_guard == 0 {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(RequestIndex(current_node_index.get())),
                    ));
                }
                loop_guard -= 1;

                let request_index = RequestIndex(current_node_index.get());
                let request_array_index = request_index.get();

                // Fetch constraints for the current request.
                let feasible_window = solver_model.feasible_intervals()[request_array_index];
                let processing_time = solver_model
                    .processing_time(request_index, berth_index)
                    .flatten()
                    .ok_or_else(|| {
                        SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                            request_index,
                            berth_index,
                        ))
                    })?;

                // Determine the earliest this request can start, considering its predecessor.
                let lower_bound_from_precedence =
                    earliest_finish_of_predecessor.unwrap_or_else(|| feasible_window.start());

                // Combine all constraints to find the effective time window for this request.
                let (effective_lower_bound, effective_upper_bound) = start_bounds_lb_pass(
                    feasible_window,
                    &interval_variables[request_array_index],
                    lower_bound_from_precedence,
                    processing_time,
                    request_index,
                )?;

                if effective_lower_bound > effective_upper_bound {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                }

                // Find the EARLIEST possible placement in the calendar within the effective window.
                calendar_segment_cursor = advance_to_segment(
                    free_intervals,
                    calendar_segment_cursor,
                    effective_lower_bound,
                );
                let (new_lower_bound, new_finish_time) = earliest_fit_in_calendar_full_fit(
                    free_intervals,
                    &mut calendar_segment_cursor,
                    effective_lower_bound,
                    effective_upper_bound,
                    processing_time,
                )
                .ok_or_else(|| {
                    SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                        request_index,
                    ))
                })?;

                // TIGHTEN the Lower Bound if we found a stricter one.
                let current_ivar = &mut interval_variables[request_array_index];
                if new_lower_bound > current_ivar.start_time_lower_bound {
                    current_ivar.start_time_lower_bound = new_lower_bound;
                }

                // The finish time of this request becomes the earliest start for the next one.
                earliest_finish_of_predecessor = Some(new_finish_time);

                // Move to the next request in the chain.
                if let Some(next_node) = chain_view.next_real(current_node_index) {
                    current_node_index = next_node;
                } else {
                    break; // End of chain.
                }
            }
        }
        // Backward pass: Propagate constraints backward to tighten Upper Bounds (UBs).
        {
            // Start traversal from the last actual request in the chain.
            let Some(mut current_node_index) = chain_view.real_end() else {
                return Ok(()); // Chain is empty.
            };
            let mut loop_guard = interval_variables.len().saturating_add(2);

            // This tracks the latest start time of the *next* request (the successor).
            let mut latest_start_time_of_successor: Option<TimePoint<T>> = None;
            // A cursor for scanning calendar segments. It starts at the end and moves backward.
            let mut calendar_segment_cursor: Option<usize> = None;

            loop {
                // Defensive guard against cycles.
                if loop_guard == 0 {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(RequestIndex(current_node_index.get())),
                    ));
                }
                loop_guard -= 1;

                let request_index = RequestIndex(current_node_index.get());
                let request_array_index = request_index.get();

                // Fetch constraints for the current request.
                let feasible_window = solver_model.feasible_intervals()[request_array_index];
                let processing_time = solver_model
                    .processing_time(request_index, berth_index)
                    .flatten()
                    .ok_or_else(|| {
                        SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                            request_index,
                            berth_index,
                        ))
                    })?;

                // The finish time of this request is constrained by the start time of its successor.
                // finish_i <= start_{i+1}  ==>  start_i + pt_i <= start_{i+1}  ==>  start_i <= start_{i+1} - pt_i
                let precedence_cap_on_start_time: Option<TimePoint<T>> =
                    if let Some(successor_latest_start) = latest_start_time_of_successor {
                        successor_latest_start.checked_sub(processing_time)
                    } else {
                        None
                    };

                // Combine all constraints to find the effective time window.
                let (effective_lower_bound, effective_upper_bound) = start_bounds_ub_pass(
                    feasible_window,
                    &interval_variables[request_array_index],
                    precedence_cap_on_start_time,
                    processing_time,
                    request_index,
                )?;

                if effective_lower_bound > effective_upper_bound {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                }

                // Find the LATEST possible placement in the calendar within the effective window.
                // Lazily initialize the backward cursor to start searching from the end of the time window.
                let initial_segment_index = calendar_segment_cursor.unwrap_or_else(|| {
                    retreat_to_segment(free_intervals, free_intervals.len(), effective_upper_bound)
                });

                let (new_upper_bound, _finish_time, used_segment_index) =
                    latest_fit_in_calendar_full_fit(
                        free_intervals,
                        initial_segment_index,
                        effective_lower_bound,
                        effective_upper_bound,
                        processing_time,
                    )
                    .ok_or_else(|| {
                        SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                            request_index,
                        ))
                    })?;

                // The cursor is now set for the next (previous) iteration.
                calendar_segment_cursor = Some(used_segment_index);

                // TIGHTEN the Upper Bound if we found a stricter one.
                let current_ivar = &mut interval_variables[request_array_index];
                if new_upper_bound < current_ivar.start_time_upper_bound {
                    current_ivar.start_time_upper_bound = new_upper_bound;
                }

                // The tightened start time of this request now constrains its predecessor.
                latest_start_time_of_successor = Some(current_ivar.start_time_upper_bound);

                // Move to the previous request in the chain.
                if let Some(previous_node) = chain_view.prev_real(current_node_index) {
                    current_node_index = previous_node;
                } else {
                    break; // Start of chain.
                }
            }

            // Final sanity check: ensure no interval has LB > UB.
            {
                let mut cur = chain_view.real_start();
                let mut guard = interval_variables.len();

                while let Some(n) = cur {
                    if guard == 0 {
                        break;
                    } // defensive
                    guard -= 1;

                    let i = RequestIndex(n.get()).get();
                    let ivar = &interval_variables[i];
                    if ivar.start_time_lower_bound > ivar.start_time_upper_bound {
                        return Err(SchedulingError::FeasiblyWindowViolation(
                            FeasiblyWindowViolationError::new(RequestIndex(i)),
                        ));
                    }
                    cur = chain_view.next_real(n);
                }
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
#[inline]
fn advance_to_segment<T: Copy + Ord>(
    free_intervals: &[TimeInterval<T>],
    mut index: usize,
    time_point: TimePoint<T>,
) -> usize {
    while index < free_intervals.len() && free_intervals[index].end() <= time_point {
        index += 1;
    }
    index
}

/// Moves a cursor backward to the last segment that starts before `time_point`.
/// This is a conservative starting point for the backward search.
#[inline]
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
    // Walk left from the hint until we find a segment that could contain the time point.
    let mut i = index_hint.saturating_sub(1);
    loop {
        if free_intervals[i].start() < time_point {
            return i;
        }
        if i == 0 {
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
    use crate::state::{
        chain_set::{
            base::ChainSet,
            delta::{ChainNextRewire, ChainSetDelta},
            index::{ChainIndex, NodeIndex},
            view::{ChainSetView, ChainViewDynAdapter},
        },
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
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

    /* ---------- forward pass sanity stays as before (covered elsewhere) ---------- */

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
        let dyn_c0 = ChainViewDynAdapter(c0);
        BoundsTightener.propagate(&m, &dyn_c0, &mut ivars).unwrap();

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
        let dyn_c0 = ChainViewDynAdapter(c0);
        BoundsTightener.propagate(&m, &dyn_c0, &mut ivars).unwrap();

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
        let dyn_c0 = ChainViewDynAdapter(c0);
        BoundsTightener.propagate(&m, &dyn_c0, &mut ivars).unwrap();

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
        let dyn_c0 = ChainViewDynAdapter(c0);
        let err = BoundsTightener
            .propagate(&m, &dyn_c0, &mut ivars)
            .unwrap_err();
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
        let dyn_c0 = ChainViewDynAdapter(c0);
        let err = BoundsTightener
            .propagate(&m, &dyn_c0, &mut ivars)
            .unwrap_err();
        match err {
            SchedulingError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request(), ri(0));
                assert_eq!(e.berth(), bi(0));
            }
            x => panic!("expected NotAllowedOnBerth, got {:?}", x),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::state::{
            chain_set::{
                base::ChainSet,
                delta::{ChainNextRewire, ChainSetDelta},
                index::{ChainIndex, NodeIndex},
                view::{ChainSetView, ChainViewDynAdapter},
            },
            index::{BerthIndex, RequestIndex},
            model::SolverModel,
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
                let req = Request::<FlexibleKind, i64>::new(rid(i), iv(ws, we), 1, map)
                    .expect("request ok");
                builder.add_flexible(req);
            }
            builder.build().expect("problem ok")
        }

        fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
            m.feasible_intervals()
                .iter()
                .zip(m.processing_times())
                .map(|(w, p_times)| {
                    // Find the first valid processing time for this request to use for initialization.
                    // If none exist, it can't be scheduled anywhere, so we can default to 0 duration.
                    let first_valid_pt = p_times
                        .iter()
                        .find_map(|&opt| Some(opt)) // Find the first Some(value) and return it
                        .unwrap_or_else(|| td(0)); // Default to duration 0 if no valid times

                    let ub = w
                        .end()
                        .checked_sub(first_valid_pt)
                        .unwrap_or_else(TimePoint::zero);
                    IntervalVar::new(w.start(), ub)
                })
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
            let dyn_c0 = ChainViewDynAdapter(c0);

            // With the fix, this should now succeed.
            let result = BoundsTightener.propagate(&m, &dyn_c0, &mut ivars);
            assert!(result.is_ok());

            // The forward pass should find the earliest start at 45.
            // The backward pass will tighten the upper bound to 50.
            assert_eq!(ivars[0].start_time_lower_bound, tp(45));
            assert_eq!(ivars[0].start_time_upper_bound, tp(50));
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
            let dyn_c0 = ChainViewDynAdapter(c0);
            BoundsTightener.propagate(&m, &dyn_c0, &mut ivars).unwrap();

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
            let dyn_c0 = ChainViewDynAdapter(c0);
            BoundsTightener.propagate(&m, &dyn_c0, &mut ivars).unwrap();

            assert_eq!(ivars[0].start_time_upper_bound, tp(7)); // 12 - pt0(5)
        }

        #[test]
        fn forward_and_backward_squeeze_to_meet_in_gap() {
            // free: [0,5), [10,15); pt=4; window [0,20)
            // Forward pass finds earliest start is 0, so LB becomes 0.
            // Backward pass finds latest start is 11 (to finish by 15), so UB becomes 11.
            let p = build_problem(&[vec![(0, 5), (10, 15)]], &[(0, 20)], &[vec![Some(4)]]);
            let m = SolverModel::from_problem(&p).unwrap();

            let mut ivars = default_ivars(&m);
            ivars[0].start_time_upper_bound = tp(10_000); // Make UB permissive

            let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
            link_chain(&mut cs, 0, &[0]);

            let c0 = cs.chain(ChainIndex(0));
            let dyn_c0 = ChainViewDynAdapter(c0);
            BoundsTightener.propagate(&m, &dyn_c0, &mut ivars).unwrap();

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
            // successor UB = 5 ⇒ pred s ≤ 5 - 10 = -5.
            ivars[1].start_time_upper_bound = tp(5);
            // pred LB high:
            ivars[0].start_time_lower_bound = tp(0);

            let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
            link_chain(&mut cs, 0, &[0, 1]);

            let c0 = cs.chain(ChainIndex(0));
            let dyn_c0 = ChainViewDynAdapter(c0);
            let err = BoundsTightener
                .propagate(&m, &dyn_c0, &mut ivars)
                .unwrap_err();
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
            let dyn_c0 = ChainViewDynAdapter(c0);
            let err = BoundsTightener
                .propagate(&m, &dyn_c0, &mut ivars)
                .unwrap_err();
            match err {
                SchedulingError::NotAllowedOnBerth(e) => {
                    assert_eq!(e.request(), ri(0));
                    assert_eq!(e.berth(), bi(0));
                }
                x => panic!("expected NotAllowedOnBerth, got {:?}", x),
            }
        }
    }
}
