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

use std::collections::HashMap;

use crate::{
    search::scheduling::{
        err::{FeasiblyWindowViolationError, NotAllowedOnBerthError, SchedulingError},
        schedule::{RequestSchedule, Schedule},
        scheduler::Scheduler,
    },
    state::{
        berth::{
            berthocc::BerthOccupancy,
            traits::{BerthRead, BerthWrite},
        },
        chain_set::{index::ChainIndex, view::ChainSetView},
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::{TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone, Default)]
pub struct GreedyEarliest;

impl<T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug> Scheduler<T>
    for GreedyEarliest
{
    #[inline]
    fn name(&self) -> &str {
        "GreedyEarliest"
    }

    fn schedule<C: ChainSetView>(
        &self,
        solver_state: &SolverSearchState<T>,
        chains: &C,
    ) -> Result<Schedule<T>, SchedulingError> {
        let model: &SolverModel<T> = solver_state.model();
        let mut final_schedule = Schedule::new(model.berths_len());
        let mut modified_berth_occupancies: HashMap<BerthIndex, BerthOccupancy<'_, T>> =
            HashMap::with_capacity(model.berths_len());

        for chain_idx_val in 0..chains.num_chains() {
            let chain_index = ChainIndex(chain_idx_val);
            let berth_index = BerthIndex(chain_index.get());
            let mut previous_request_end_time: Option<TimePoint<T>> = None;

            for request_node in chains.iter_chain(chain_index) {
                let request_index = RequestIndex(request_node.get());
                let feasible_window: TimeInterval<T> =
                    model.feasible_intervals()[request_index.get()];

                let Some(Some(processing_time)) = model.processing_time(request_index, berth_index)
                else {
                    return Err(SchedulingError::NotAllowedOnBerth(
                        NotAllowedOnBerthError::new(request_index, berth_index),
                    ));
                };

                let earliest_arrival = match previous_request_end_time {
                    None => feasible_window.start(),
                    Some(prev_end) => {
                        if prev_end > feasible_window.end() {
                            return Err(SchedulingError::FeasiblyWindowViolation(
                                FeasiblyWindowViolationError::new(request_index),
                            ));
                        }
                        std::cmp::max(prev_end, feasible_window.start())
                    }
                };

                let berth_occupancy = modified_berth_occupancies
                    .entry(berth_index)
                    .or_insert_with(|| {
                        model.baseline_occupancy_for_berth(berth_index).expect(
                            "berth_index is in range; baseline_occupancy_for_berth must succeed",
                        ).clone()
                    });

                let Some(first_available_slot) = berth_occupancy
                    .iter_earliest_fit_intervals_in(feasible_window, processing_time)
                    .next()
                else {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                };

                let actual_start_time =
                    std::cmp::max(earliest_arrival, first_available_slot.start());

                let Some(actual_end_time) = actual_start_time.checked_add(processing_time) else {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                };

                if actual_end_time > first_available_slot.end()
                    || actual_end_time > feasible_window.end()
                {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                }

                let service_interval = TimeInterval::new(actual_start_time, actual_end_time);
                berth_occupancy
                    .occupy(service_interval)
                    .expect("we selected from free space; occupy must succeed");

                final_schedule.add_schedule(RequestSchedule::new(
                    request_index,
                    berth_index,
                    service_interval,
                ));

                previous_request_end_time = Some(actual_end_time);
            }
        }

        Ok(final_schedule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain_set::{
        base::ChainSet,
        delta_builder::ChainSetDeltaBuilder,
        index::{ChainIndex, NodeIndex},
        view::ChainSetView,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::{FixedKind, FlexibleKind};
    use berth_alloc_model::prelude::{Assignment, Berth, BerthIdentifier, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
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

    fn berth(id: usize, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn req_fixed(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn req_flex(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn asg_fixed(
        req: &Request<FixedKind, i64>,
        berth: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new(req.clone(), berth.clone(), tp(start)).unwrap()
    }

    #[inline]
    fn make_state<'p>(model: &'p SolverModel<'p, i64>) -> SolverSearchState<'p, 'p, i64> {
        SolverSearchState::new(model)
    }

    // Link a sequence into a given chain: start -> n0 -> n1 -> ... -> nk -> end
    fn link_sequence(cs: &mut ChainSet, chain: ChainIndex, nodes: &[usize]) {
        // Create a builder that will accumulate all our linking operations.
        let mut builder = ChainSetDeltaBuilder::new(cs);

        // Get the starting node (the "head sentinel") of the target chain.
        let start_node = cs.start_of_chain(chain);

        // We will insert each new node after the previous one, starting from the head.
        let mut current_tail = start_node;

        for &node_id in nodes {
            let node_to_link = NodeIndex(node_id);
            // `insert_after` correctly wires `current_tail -> node_to_link -> old_successor`.
            builder.insert_after(current_tail, node_to_link);
            // The new node becomes the tail for the next insertion.
            current_tail = node_to_link;
        }

        // Build the delta containing all the rewires and apply it to the ChainSet.
        cs.apply_delta(&builder.build());
    }

    #[test]
    fn test_greedy_places_earliest_and_respects_fixed() {
        // b0 availability [0,100), one fixed [10,20).
        // two flex jobs of 5 each on b0, both window [0,100).
        let b0 = berth(1, 0, 100);
        let rf = req_fixed(900, (0, 100), &[(1, 10)]);
        let a = asg_fixed(&rf, &b0, 10);

        let f0 = req_flex(10, (0, 100), &[(1, 5)]);
        let f1 = req_flex(11, (0, 100), &[(1, 5)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_fixed(a);
        pb.add_flexible(f0);
        pb.add_flexible(f1);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let state = make_state(&model);

        // ChainSet with R nodes, 1 berth
        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        // Chain 0 (berth 0) order: [req0, req1]  -> should place [0,5) then [5,10)
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let sched = GreedyEarliest
            .schedule(&state, &cs)
            .expect("should schedule");
        let s_chain0 = sched.schedules_for_berth(bi(0)).unwrap();
        assert_eq!(s_chain0.len(), 2);
        assert_eq!(s_chain0[0].interval(), &iv(0, 5));
        assert_eq!(s_chain0[1].interval(), &iv(5, 10));
    }

    #[test]
    fn test_greedy_skips_small_slot_and_finds_later_fit() {
        // b0 availability [0,100), fixed [8,12) -> free slices [0,8) and [12,100).
        // One flex of len 10, window [0,100). The improved algorithm should now
        // SKIP the first free slice ([0,8)) and use the second one.
        let b0 = berth(1, 0, 100);
        let rf = req_fixed(901, (0, 100), &[(1, 4)]);
        let a = asg_fixed(&rf, &b0, 8); // occupies [8,12)

        let fx = req_flex(10, (0, 100), &[(1, 10)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_fixed(a);
        pb.add_flexible(fx);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0]); // just the single request

        // ASSERT SUCCESS: The schedule should be found successfully.
        let sched = GreedyEarliest
            .schedule(&state, &cs)
            .expect("scheduling should now succeed by skipping the small slot");

        // VERIFY RESULT: Check that the request was placed in the correct interval.
        let s_chain0 = sched.schedules_for_berth(bi(0)).unwrap();
        assert_eq!(s_chain0.len(), 1);
        assert_eq!(s_chain0[0].request(), ri(0));
        // It should be placed in the second slot, which starts at 12.
        assert_eq!(s_chain0[0].interval(), &iv(12, 22));
    }

    #[test]
    fn test_greedy_not_allowed_on_berth_is_reported() {
        // Two berths. Request only allowed on berth 2, but we put it in chain 0 (berth 1).
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        let fx = req_flex(10, (0, 100), &[(2, 5)]); // only b2

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(fx);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        // Place request 0 on chain 0 (maps to berth with id=1)
        link_sequence(&mut cs, ChainIndex(0), &[0]);

        let err = GreedyEarliest.schedule(&state, &cs).unwrap_err();
        match err {
            SchedulingError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request(), ri(0));
                assert_eq!(e.berth(), bi(0));
            }
            other => panic!("expected NotAllowedOnBerth, got {other:?}"),
        }
    }

    #[test]
    fn test_greedy_precedence_past_window_fails() {
        // One berth b0, availability [0,100).
        // First flex len 8, second flex len 5 with window [0,10].
        // Precedence pushes second to start at >=8, end=13>10 -> fail.
        let b0 = berth(1, 0, 100);
        let f0 = req_flex(10, (0, 100), &[(1, 8)]);
        let f1 = req_flex(11, (0, 10), &[(1, 5)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_flexible(f0);
        pb.add_flexible(f1);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let err = GreedyEarliest.schedule(&state, &cs).unwrap_err();
        match err {
            SchedulingError::FeasiblyWindowViolation(e) => {
                assert_eq!(e.request(), ri(1)); // second request violates its window
            }
            other => panic!("expected FeasiblyWindowViolation, got {other:?}"),
        }
    }

    #[test]
    fn test_greedy_places_right_after_predecessor_inside_first_free_slice() {
        // b0: availability [0,100), fixed [20,30) → free [0,20) and [30,100).
        // Two flex len 7 each → both should fit in the first free window due to precedence.
        let b0 = berth(1, 0, 100);
        let rf = req_fixed(50, (0, 100), &[(1, 10)]);
        let a = asg_fixed(&rf, &b0, 20); // [20,30)

        let f0 = req_flex(10, (0, 100), &[(1, 7)]);
        let f1 = req_flex(11, (0, 100), &[(1, 7)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_fixed(a);
        pb.add_flexible(f0);
        pb.add_flexible(f1);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let sched = GreedyEarliest.schedule(&state, &cs).expect("must schedule");
        let s = sched.schedules_for_berth(bi(0)).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].interval(), &iv(0, 7));
        assert_eq!(s[1].interval(), &iv(7, 14));
    }
}
