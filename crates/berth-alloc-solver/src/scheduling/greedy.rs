// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to do so, subject to the following conditions:
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
    scheduling::{
        err::{FeasiblyWindowViolationError, NotAllowedOnBerthError, SchedulingError},
        traits::CalendarScheduler,
    },
    state::{
        chain_set::{
            index::NodeIndex,
            view::{ChainRef, ChainSetView},
        },
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
    },
};
use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};

/// A greedy, forward-in-time scheduler for a single berth's calendar.
///
/// This scheduler operates on a contiguous slice of a request chain, where the chain's
/// index corresponds directly to the berth's index. It iterates through the requests
/// in the specified slice and assigns the earliest possible start time for each one.
///
/// ## Key Behaviors:
/// - **Forward Scheduling**: Processes requests chronologically based on their sequence in the chain.
/// - **Calendar-Aware**: Respects the berth's calendar, placing each request entirely
///   within a single available time segment.
/// - **Constraint Respecting**: Adheres to each request's feasible time window and
///   any additional bounds set by its `IntervalVar`. The upper bound of the `IntervalVar`
///   is treated as a "finish cap," meaning the operation must *end* by this time.
/// - **Safe Traversal**: Safely handles and skips sentinel nodes (placeholders at the
///   start and end of a chain) and includes a defensive loop guard to prevent infinite
///   loops in case of malformed chain structures (e.g., cycles).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GreedyCalendar;

impl<T> CalendarScheduler<T> for GreedyCalendar
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    /// Schedules a slice of a request chain on its corresponding berth.
    ///
    /// This method calculates and assigns a start time for each request in the slice,
    /// mutating the `decision_variables` slice to store the results.
    fn schedule_chain_slice<'a, C: ChainSetView>(
        &self,
        solver_model: &SolverModel<'a, T>,
        chain_view: ChainRef<'_, C>,
        slice_start_node: NodeIndex,
        slice_end_node_exclusive: Option<NodeIndex>,
        interval_variables: &mut [IntervalVar<T>],
        decision_variables: &mut [DecisionVar<T>],
    ) -> Result<(), SchedulingError> {
        // A chain is intrinsically linked to a single berth. Get its index and calendar.
        let berth_index = BerthIndex(chain_view.chain_index().get());
        let berth_calendar = solver_model
            .calendar_for_berth(berth_index)
            .ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                    RequestIndex(slice_start_node.get()),
                ))
            })?;
        let free_intervals = berth_calendar.free_intervals();

        // Determine the actual start and end nodes of the slice, skipping any sentinels.
        let (first_node_in_slice, resolved_end_node_exclusive) =
            chain_view.resolve_slice(slice_start_node, slice_end_node_exclusive);

        // If the slice is empty (contains no real request nodes), we're done.
        let Some(first_actual_node) = first_node_in_slice else {
            return Ok(());
        };

        // If the exclusive end bound is a sentinel, find the first *real* node after it
        // to serve as a concrete boundary for the loop.
        let exclusive_end_actual_node = if chain_view.is_sentinel_node(resolved_end_node_exclusive)
        {
            chain_view.first_real_node(resolved_end_node_exclusive)
        } else {
            Some(resolved_end_node_exclusive)
        };

        // --- Seed the Earliest Possible Start Time ---
        // The first request in our slice can start no earlier than the completion of its
        // predecessor in the full chain, or the beginning of the first available calendar slot.
        let mut earliest_possible_start_time: TimePoint<T> =
            if let Some(predecessor_node) = chain_view.prev_real(first_actual_node) {
                let predecessor_request_index = RequestIndex(predecessor_node.get());
                match decision_variables.get(predecessor_request_index.get()) {
                    // If the predecessor is already scheduled, its finish time is our earliest start.
                    Some(DecisionVar::Assigned(predecessor_decision)) => {
                        // The predecessor *must* be on the same berth as this chain.
                        if predecessor_decision.berth_index != berth_index {
                            return Err(SchedulingError::NotAllowedOnBerth(
                                NotAllowedOnBerthError::new(predecessor_request_index, berth_index),
                            ));
                        }
                        // Calculate predecessor's finish time.
                        let predecessor_processing_time = solver_model
                            .processing_time(predecessor_request_index, berth_index)
                            .flatten()
                            .ok_or_else(|| {
                                SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                                    predecessor_request_index,
                                    berth_index,
                                ))
                            })?;

                        predecessor_decision
                            .start_time
                            .checked_add(predecessor_processing_time)
                            .ok_or_else(|| {
                                SchedulingError::FeasiblyWindowViolation(
                                    FeasiblyWindowViolationError::new(predecessor_request_index),
                                )
                            })?
                    }
                    // If predecessor is unassigned, fall back to the first calendar opening.
                    _ => free_intervals.first().map(|iv| iv.start()).ok_or_else(|| {
                        SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                            RequestIndex(first_actual_node.get()),
                        ))
                    })?,
                }
            } else {
                // No predecessor; start at the first available calendar time.
                free_intervals.first().map(|iv| iv.start()).ok_or_else(|| {
                    SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                        RequestIndex(first_actual_node.get()),
                    ))
                })?
            };

        // A cursor to keep track of our position in the berth's free time intervals.
        // This advances monotonically, preventing redundant searches.
        let mut calendar_segment_cursor =
            advance_to_segment(free_intervals, 0, earliest_possible_start_time);

        // --- Greedy Scheduling Loop ---
        // Iterate through each request node in the chain slice.
        let mut loop_guard = solver_model.flexible_requests_len(); // Defensive guard against cycles.
        let mut current_node = Some(first_actual_node);

        while let Some(current_node_index) = current_node {
            // Stop if we've reached the exclusive end node of the slice.
            if let Some(end_node) = exclusive_end_actual_node
                && current_node_index == end_node
            {
                break;
            }

            // Defensive check: If we loop more times than there are requests,
            // there's likely a cycle in the chain's linked-list structure.
            if loop_guard == 0 {
                return Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(RequestIndex(current_node_index.get())),
                ));
            }
            loop_guard -= 1;

            let request_index = RequestIndex(current_node_index.get());
            let request_array_index = request_index.get();

            // Fetch this request's constraints: its feasible window and processing time on this berth.
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

            // Calculate the effective start time bounds, considering all constraints.
            let (start_lower_bound, start_upper_bound) = start_bounds(
                feasible_window,
                &interval_variables[request_array_index],
                earliest_possible_start_time,
                processing_time,
                request_index,
            )?;

            // If the time window is invalid (e.g., lower bound > upper bound), scheduling is impossible.
            if start_lower_bound > start_upper_bound {
                return Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(request_index),
                ));
            }

            // Find the earliest time slot in the calendar that can fully accommodate the request.
            let (assigned_start_time, assigned_end_time) = earliest_fit_in_calendar_full_fit(
                free_intervals,
                &mut calendar_segment_cursor,
                start_lower_bound,
                start_upper_bound,
                processing_time,
            )
            .ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                    request_index,
                ))
            })?;

            // Success! Commit the assignment to the decision variables.
            decision_variables[request_array_index] =
                DecisionVar::assigned(berth_index, assigned_start_time);

            // The next request cannot start before the current one finishes.
            earliest_possible_start_time = assigned_end_time;
            // Advance the calendar cursor to the segment containing the new end time for efficiency.
            calendar_segment_cursor = advance_to_segment(
                free_intervals,
                calendar_segment_cursor,
                earliest_possible_start_time,
            );

            // Move to the next request in the chain.
            current_node = chain_view.next_real(current_node_index);
        }

        Ok(())
    }

    /// Validates if a hypothetical greedy schedule for a chain slice is feasible.
    ///
    /// This method performs the same logic as `schedule_chain_slice` but without
    /// writing any assignments to the `decision_variables`. It is a read-only check
    /// to determine if a valid schedule *could* be constructed for the given slice
    /// and constraints.
    fn valid_schedule_slice<'a, C: ChainSetView>(
        &self,
        solver_model: &SolverModel<'a, T>,
        chain_view: ChainRef<'_, C>,
        slice_start_node: NodeIndex,
        slice_end_node_exclusive: Option<NodeIndex>,
        interval_variables: &[IntervalVar<T>],
    ) -> Result<(), SchedulingError> {
        // Setup is identical to the scheduling method.
        let berth_index = BerthIndex(chain_view.chain_index().get());
        let berth_calendar = solver_model
            .calendar_for_berth(berth_index)
            .ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                    RequestIndex(slice_start_node.get()),
                ))
            })?;
        let free_intervals = berth_calendar.free_intervals();

        let (first_node_in_slice, resolved_end_node_exclusive) =
            chain_view.resolve_slice(slice_start_node, slice_end_node_exclusive);

        let Some(first_actual_node) = first_node_in_slice else {
            return Ok(()); // An empty slice is trivially valid.
        };

        let exclusive_end_actual_node = if chain_view.is_sentinel_node(resolved_end_node_exclusive)
        {
            chain_view.first_real_node(resolved_end_node_exclusive)
        } else {
            Some(resolved_end_node_exclusive)
        };

        // Seed the earliest start time. Since this is a validation without external assignments,
        // we always start from the beginning of the berth's calendar.
        let mut earliest_possible_start_time =
            free_intervals.first().map(|iv| iv.start()).ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                    RequestIndex(first_actual_node.get()),
                ))
            })?;

        let mut calendar_segment_cursor =
            advance_to_segment(free_intervals, 0, earliest_possible_start_time);

        // --- Greedy Validation Loop ---
        let mut loop_guard = solver_model.flexible_requests_len();
        let mut current_node = Some(first_actual_node);

        while let Some(current_node_index) = current_node {
            if let Some(end_node) = exclusive_end_actual_node
                && current_node_index == end_node
            {
                break;
            }

            if loop_guard == 0 {
                return Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(RequestIndex(current_node_index.get())),
                ));
            }
            loop_guard -= 1;

            let request_index = crate::state::index::RequestIndex(current_node_index.get());
            let request_array_index = request_index.get();

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

            let (start_lower_bound, start_upper_bound) = start_bounds(
                feasible_window,
                &interval_variables[request_array_index],
                earliest_possible_start_time,
                processing_time,
                request_index,
            )?;

            if start_lower_bound > start_upper_bound {
                return Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(request_index),
                ));
            }

            // Check for a valid fit, but do not store the result.
            let (_, assigned_end_time) = earliest_fit_in_calendar_full_fit(
                free_intervals,
                &mut calendar_segment_cursor,
                start_lower_bound,
                start_upper_bound,
                processing_time,
            )
            .ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                    request_index,
                ))
            })?;

            // Advance time and cursors as if we had scheduled, then continue.
            earliest_possible_start_time = assigned_end_time;
            calendar_segment_cursor = advance_to_segment(
                free_intervals,
                calendar_segment_cursor,
                earliest_possible_start_time,
            );

            current_node = chain_view.next_real(current_node_index);
        }

        Ok(())
    }
}

/// Helper to advance a cursor `index` through a sorted slice of `free_intervals`
/// until it points to the first interval that could possibly contain `time_point`.
#[inline]
fn advance_to_segment<T: Copy + Ord>(
    free_intervals: &[TimeInterval<T>],
    mut index: usize,
    time_point: TimePoint<T>,
) -> usize {
    // Skip all past segments that end at or before the target time point.
    while index < free_intervals.len() && free_intervals[index].end() <= time_point {
        index += 1;
    }
    index
}

/// Computes the effective start-time bounds for a request by combining multiple constraints.
///
/// - **Lower bound**: The latest of three times:
///   1. The finish time of the preceding request (`earliest_precedence_time`).
///   2. The request's own specific start time lower bound from its `IntervalVar`.
///   3. The start of the request's overall feasible window.
/// - **Upper bound**: The earliest of two times, reflecting that the operation must *finish*
///   by the end of its windows. This is calculated by subtracting the processing time from:
///   1. The end of the request's overall feasible window.
///   2. The request's "finish cap" upper bound from its `IntervalVar`.
#[inline]
fn start_bounds<T: Copy + Ord + CheckedSub>(
    feasible_window: TimeInterval<T>,
    interval_variable: &IntervalVar<T>,
    earliest_precedence_time: TimePoint<T>,
    processing_time: TimeDelta<T>,
    request_index: RequestIndex,
) -> Result<(TimePoint<T>, TimePoint<T>), SchedulingError> {
    let lower_bound = max3(
        earliest_precedence_time,
        interval_variable.start_time_lower_bound,
        feasible_window.start(),
    );

    let upper_bound_from_window = feasible_window
        .end()
        .checked_sub(processing_time)
        .ok_or_else(|| {
            SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                request_index,
            ))
        })?;

    let upper_bound_from_finish_cap = interval_variable
        .start_time_upper_bound
        .checked_sub(processing_time)
        .ok_or_else(|| {
            SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                request_index,
            ))
        })?;

    let upper_bound = min2(upper_bound_from_window, upper_bound_from_finish_cap);

    Ok((lower_bound, upper_bound))
}

/// Finds the earliest possible start time `s` within the given bounds `[lb, ub]` such that
/// the entire operation interval `[s, s + pt)` fits within a *single* free calendar segment.
///
/// This function iterates through the `free_intervals` starting from the `segment_cursor`.
/// It returns the start and end times `(s, s + pt)` upon finding the first valid fit.
/// The `segment_cursor` is updated to the index of the successful segment to optimize subsequent searches.
#[inline]
fn earliest_fit_in_calendar_full_fit<T: Copy + Ord + CheckedAdd + CheckedSub>(
    free_intervals: &[TimeInterval<T>],
    segment_cursor: &mut usize,
    start_lower_bound: TimePoint<T>,
    start_upper_bound: TimePoint<T>,
    processing_time: TimeDelta<T>,
) -> Option<(TimePoint<T>, TimePoint<T>)> {
    // Start searching from the first calendar segment that could possibly contain our lower bound.
    let mut current_segment_index =
        advance_to_segment(free_intervals, *segment_cursor, start_lower_bound);

    while current_segment_index < free_intervals.len() {
        let calendar_segment = free_intervals[current_segment_index];

        // To fit, a potential start time `s` must satisfy:
        // s >= max(start_lower_bound, segment.start)
        // s + processing_time <= segment.end  =>  s <= segment.end - processing_time

        // The earliest we can possibly start in *this* segment.
        let effective_start_time = max2(start_lower_bound, calendar_segment.start());

        // The latest we can possibly start in *this* segment while still fitting.
        let latest_possible_start_in_segment =
            calendar_segment.end().checked_sub(processing_time)?;

        // The latest start time allowed by all constraints combined for this segment.
        let latest_valid_start_time = min2(start_upper_bound, latest_possible_start_in_segment);

        // If a valid time window exists in this segment (earliest <= latest), we've found a fit.
        if effective_start_time <= latest_valid_start_time
            && let Some(end_time) = effective_start_time.checked_add(processing_time)
        {
            // Found a valid placement. Update the cursor and return the result.
            *segment_cursor = current_segment_index;
            return Some((effective_start_time, end_time));
        }
        // If adding the processing time overflows, this solution is invalid.
        // Continue to check the next segment as it might start much later.

        // No fit in this segment, try the next one.
        current_segment_index += 1;
    }

    // Scanned all remaining segments and found no fit.
    None
}

// --- Utility Functions ---

#[inline]
fn max2<T: Ord>(a: T, b: T) -> T {
    std::cmp::max(a, b)
}

#[inline]
fn min2<T: Ord>(a: T, b: T) -> T {
    std::cmp::min(a, b)
}

#[inline]
fn max3<T: Ord + Copy>(a: T, b: T, c: T) -> T {
    std::cmp::max(std::cmp::max(a, b), c)
}

// The `tests` module remains unchanged as it correctly tests the logic,
// which has not been altered.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{decisionvar::DecisionVar, intervalvar::IntervalVar};
    use crate::scheduling::traits::CalendarScheduler;
    use crate::state::{
        chain_set::{
            base::ChainSet,
            delta::{ChainNextRewire, ChainSetDelta},
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
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

    // Build a Problem:
    // - berths_windows[b] = vec![(s,e), ...] availability windows for berth b (ids 0..B-1).
    // - request_windows[r] = (s,e) feasible window for request r (ids 0..R-1).
    // - processing[r][b] = Some(dur) if r allowed on berth b with PT=dur; None otherwise.
    fn build_problem(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let b_len = berths_windows.len();
        let r_len = request_windows.len();
        assert_eq!(processing.len(), r_len);
        for row in processing {
            assert_eq!(
                row.len(),
                b_len,
                "processing times per request must match number of berths"
            );
        }

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
                .expect("request should be well-formed");
            builder.add_flexible(req);
        }

        builder.build().expect("problem should build")
    }

    fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
        m.feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect()
    }
    fn default_dvars(m: &SolverModel<'_, i64>) -> Vec<DecisionVar<i64>> {
        vec![DecisionVar::Unassigned; m.flexible_requests_len()]
    }

    // Link nodes onto chain c: start -> nodes[0] -> ... -> nodes[k] -> end
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
    fn test_single_request_at_opening_from_head_sentinel() {
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let sched = GreedyCalendar;

        // Use schedule_chain (starts at sentinel head internally)
        sched
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap();

        let dec = dvars[ri(0).get()].as_assigned().unwrap();
        assert_eq!(dec.berth_index, bi(0));
        assert_eq!(dec.start_time, tp(0));
    }

    #[test]
    fn test_respects_window_and_interval_bounds_with_caps() {
        // window [10,50), PT=7, but start LB=15 -> expect s=15
        let p = build_problem(&[vec![(0, 100)]], &[(10, 50)], &[vec![Some(7)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_lower_bound = tp(15);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap();

        let dec = dvars[0].as_assigned().unwrap();
        assert_eq!(dec.start_time, tp(15));
    }

    #[test]
    fn test_moves_to_next_calendar_segment() {
        // free: [0,5), [8,20); req window [0,30), PT=4; LB=3 ⇒ [3,5) too short -> move to 8
        let p = build_problem(&[vec![(0, 5), (8, 20)]], &[(0, 30)], &[vec![Some(4)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_lower_bound = tp(3);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);
        let c0 = cs.chain(ChainIndex(0));

        GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap();

        let dec = dvars[0].as_assigned().unwrap();
        assert_eq!(dec.start_time, tp(8));
    }

    #[test]
    fn test_chain_two_requests_back_to_back() {
        // free: [0,100); R0 PT=5, R1 PT=7 → s0=0, s1=5
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(7)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap();

        let d0 = dvars[0].as_assigned().unwrap();
        let d1 = dvars[1].as_assigned().unwrap();
        assert_eq!(d0.start_time, tp(0));
        assert_eq!(d1.start_time, tp(5));
    }

    #[test]
    fn test_not_allowed_on_chain_berth_errors() {
        // two berths; req0 allowed only on berth1; chain is on berth0 -> error
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100)],
            &[vec![None, Some(10)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]); // chain 0 ↔ berth 0

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap_err();
        match err {
            SchedulingError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request(), ri(0));
                assert_eq!(e.berth(), bi(0));
            }
            x => panic!("expected NotAllowedOnBerth, got {:?}", x),
        }
    }

    #[test]
    fn test_slice_from_middle_seeded_by_prev_assignment() {
        // free: [0,10), [12,30), all PT=5
        let p = build_problem(
            &[vec![(0, 10), (12, 30)]],
            &[(0, 100), (0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(5)], vec![Some(5)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        // chain 0: [0,1,2]
        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1, 2]);

        // pre-assign R0 @ t=0 on berth0
        dvars[0] = DecisionVar::assigned(bi(0), tp(0));

        // schedule slice [R1..end)
        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .schedule_chain_slice(&m, c0, NodeIndex(1), None, &mut ivars, &mut dvars)
            .unwrap();

        let d1 = dvars[1].as_assigned().unwrap();
        let d2 = dvars[2].as_assigned().unwrap();
        assert_eq!(d1.start_time, tp(5));
        assert_eq!(d2.start_time, tp(12)); // 10 hits segment end -> next free is 12
    }

    #[test]
    fn test_slice_prev_assignment_on_different_berth_is_error() {
        // two berths; prev real assigned on berth1; scheduling chain on berth0 should error
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[vec![Some(5), Some(5)], vec![Some(5), Some(5)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]); // chain 0 ↔ berth 0

        // prev (R0) assigned to berth1 (mismatch)
        dvars[0] = DecisionVar::assigned(bi(1), tp(0));

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .schedule_chain_slice(&m, c0, NodeIndex(1), None, &mut ivars, &mut dvars)
            .unwrap_err();

        match err {
            SchedulingError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request(), ri(0));
                assert_eq!(e.berth(), bi(0));
            }
            x => panic!("expected NotAllowedOnBerth, got {:?}", x),
        }
    }

    #[test]
    fn test_violation_when_no_fit_due_to_upper_bound() {
        // free [0,100), PT=10, but ub=5 ⇒ no start ≤5 can fit 10
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(10)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_upper_bound = tp(5);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap_err();

        match err {
            SchedulingError::FeasiblyWindowViolation(e) => assert_eq!(e.request(), ri(0)),
            x => panic!("expected FWV, got {:?}", x),
        }
    }

    #[test]
    fn test_violation_when_effective_window_shorter_than_processing_time() {
        // window [0,100), PT=10; push LB high so there's <10 left
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(10)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        ivars[0].start_time_lower_bound = tp(95); // lb = 95
        // ub = min(ivar.ub - pt, window.end - pt) = min(100 - 10, 100 - 10) = 90

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap_err();

        match err {
            SchedulingError::FeasiblyWindowViolation(e) => assert_eq!(e.request(), ri(0)),
            x => panic!("expected FWV, got {:?}", x),
        }
    }

    #[test]
    fn test_scans_across_multiple_segments_for_multiple_nodes() {
        // free: [0,5), [10,15), [20,25); PT=3 → s: 0,10,20 (full-fit per segment)
        let p = build_problem(
            &[vec![(0, 5), (10, 15), (20, 25)]],
            &[(0, 30), (0, 30), (0, 30)],
            &[vec![Some(3)], vec![Some(3)], vec![Some(3)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1, 2]);
        let c0 = cs.chain(ChainIndex(0));

        GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap();

        assert_eq!(dvars[0].as_assigned().unwrap().start_time, tp(0));
        assert_eq!(dvars[1].as_assigned().unwrap().start_time, tp(10));
        assert_eq!(dvars[2].as_assigned().unwrap().start_time, tp(20));
    }

    #[test]
    fn test_schedule_slice_respects_end_node_exclusive() {
        // free [0,100), three reqs PT=5; only schedule [0,1)
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(5)], vec![Some(5)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1, 2]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .schedule_chain_slice(
                &m,
                c0,
                NodeIndex(0),
                Some(NodeIndex(1)),
                &mut ivars,
                &mut dvars,
            )
            .unwrap();

        assert!(dvars[0].is_assigned());
        assert!(!dvars[1].is_assigned());
        assert!(!dvars[2].is_assigned());
    }

    #[test]
    fn test_starting_from_head_sentinel_via_slice_works() {
        // validate schedule_chain_slice when start_node is the head sentinel
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(7)]]);
        let m = SolverModel::from_problem(&p).unwrap();
        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let start_sentinel = cs.start_of_chain(ChainIndex(0));
        let c0 = cs.chain(ChainIndex(0));

        GreedyCalendar
            .schedule_chain_slice(&m, c0, start_sentinel, None, &mut ivars, &mut dvars)
            .unwrap();

        assert!(dvars[0].is_assigned());
    }

    #[test]
    fn test_empty_chain_slice_returns_ok_and_assigns_nothing() {
        let p = build_problem(&[vec![(0, 100)]], &[], &[]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars: Vec<IntervalVar<i64>> = vec![];
        let mut dvars: Vec<DecisionVar<i64>> = vec![];

        let cs = ChainSet::new(0, m.berths_len());
        // chain has only sentinels
        let c0 = cs.chain(ChainIndex(0));
        let start = cs.start_of_chain(ChainIndex(0));

        GreedyCalendar
            .schedule_chain_slice(&m, c0, start, None, &mut ivars, &mut dvars)
            .unwrap();
    }

    #[test]
    fn test_malformed_cycle_guard_trips() {
        // Build a minimal problem (one berth, windows ok), two requests. Then create a cycle in the chain.
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(5)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        // link normally: start -> 0 -> 1 -> end
        link_chain(&mut cs, 0, &[0, 1]);

        // Now create a cycle: 1 -> 0 (overwriting 1 -> end)
        let mut delta = ChainSetDelta::new();
        delta.push_rewire(ChainNextRewire::new(NodeIndex(1), NodeIndex(0)));
        cs.apply_delta(delta);

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap_err();

        match err {
            SchedulingError::FeasiblyWindowViolation(_) => {} // guard tripped
            x => panic!("expected FWV due to loop guard, got {:?}", x),
        }
    }

    #[test]
    fn test_valid_slice_single_request_ok() {
        // free [0,100), window [0,100), PT=5 → should be valid
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let ivars = default_ivars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .valid_schedule_slice(&m, c0, cs.start_of_chain(ChainIndex(0)), None, &ivars)
            .unwrap();
    }

    #[test]
    fn test_valid_slice_moves_to_next_calendar_segment_ok() {
        // free: [0,5), [8,20); window [0,30), PT=4; LB=3 ⇒ [3,5) too short → move to 8
        let p = build_problem(&[vec![(0, 5), (8, 20)]], &[(0, 30)], &[vec![Some(4)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_lower_bound = tp(3);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .valid_schedule_slice(&m, c0, cs.start_of_chain(ChainIndex(0)), None, &ivars)
            .unwrap();
    }

    #[test]
    fn test_valid_slice_violation_when_no_fit_due_to_upper_bound() {
        // free [0,100), window [0,100), PT=10, but start_ub=5 ⇒ no start ≤5 can fit 10
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(10)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_upper_bound = tp(5);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .valid_schedule_slice(&m, c0, cs.start_of_chain(ChainIndex(0)), None, &ivars)
            .unwrap_err();

        match err {
            SchedulingError::FeasiblyWindowViolation(e) => assert_eq!(e.request(), ri(0)),
            x => panic!("expected FWV, got {:?}", x),
        }
    }

    #[test]
    fn test_valid_slice_not_allowed_on_berth_errors() {
        // two berths; req0 allowed only on berth1; chain is on berth0 -> error
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100)],
            &[vec![None, Some(10)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let ivars = default_ivars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]); // chain 0 ↔ berth 0

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .valid_schedule_slice(&m, c0, cs.start_of_chain(ChainIndex(0)), None, &ivars)
            .unwrap_err();

        match err {
            SchedulingError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request(), ri(0));
                assert_eq!(e.berth(), bi(0));
            }
            x => panic!("expected NotAllowedOnBerth, got {:?}", x),
        }
    }

    #[test]
    fn test_valid_slice_respects_end_node_exclusive() {
        // free [0,100), three reqs PT=5; validate slice [0,1) only
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(5)], vec![Some(5)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let ivars = default_ivars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1, 2]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .valid_schedule_slice(&m, c0, NodeIndex(0), Some(NodeIndex(1)), &ivars)
            .unwrap();
    }

    #[test]
    fn test_valid_slice_empty_ok() {
        // empty slice: start == end_exclusive → trivially valid
        let p = build_problem(&[vec![(0, 100)]], &[], &[]);
        let m = SolverModel::from_problem(&p).unwrap();

        let ivars: Vec<IntervalVar<i64>> = vec![];

        let cs = ChainSet::new(0, m.berths_len());
        let c0 = cs.chain(ChainIndex(0));

        let start = cs.start_of_chain(ChainIndex(0));
        GreedyCalendar
            .valid_schedule_slice(&m, c0, start, Some(start), &ivars)
            .unwrap();
    }

    #[test]
    fn test_valid_slice_scans_across_multiple_segments_ok() {
        // free: [0,5), [10,15), [20,25); PT=3 → should be valid for three nodes
        let p = build_problem(
            &[vec![(0, 5), (10, 15), (20, 25)]],
            &[(0, 30), (0, 30), (0, 30)],
            &[vec![Some(3)], vec![Some(3)], vec![Some(3)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let ivars = default_ivars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1, 2]);

        let c0 = cs.chain(ChainIndex(0));
        GreedyCalendar
            .valid_schedule_slice(&m, c0, cs.start_of_chain(ChainIndex(0)), None, &ivars)
            .unwrap();
    }

    #[test]
    fn test_valid_slice_starting_from_head_sentinel_ok() {
        // Start at head sentinel; single request fits
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(7)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let ivars = default_ivars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let start_sentinel = cs.start_of_chain(ChainIndex(0));
        let c0 = cs.chain(ChainIndex(0));

        GreedyCalendar
            .valid_schedule_slice(&m, c0, start_sentinel, None, &ivars)
            .unwrap();
    }

    #[test]
    fn test_valid_slice_violation_when_effective_window_too_short() {
        // window [0,100), PT=10; push LB high so there's <10 left → FWV
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(10)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        ivars[0].start_time_lower_bound = tp(95); // lb=95, ub=min(100-10, 100-10)=90 → lb>ub → FWV

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);

        let c0 = cs.chain(ChainIndex(0));
        let err = GreedyCalendar
            .valid_schedule_slice(&m, c0, cs.start_of_chain(ChainIndex(0)), None, &ivars)
            .unwrap_err();

        match err {
            SchedulingError::FeasiblyWindowViolation(e) => assert_eq!(e.request(), ri(0)),
            x => panic!("expected FWV, got {:?}", x),
        }
    }
}
