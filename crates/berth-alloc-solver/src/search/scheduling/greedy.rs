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
    search::scheduling::{
        err::{FeasiblyWindowViolationError, NotAllowedOnBerthError, SchedulingError},
        schedule::Schedule,
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
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct GreedyEarliest;

impl<T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug> Scheduler<T>
    for GreedyEarliest
{
    #[inline]
    fn name(&self) -> &str {
        "GreedyEarliest"
    }

    fn process_schedule<C, F>(
        &self,
        solver_state: &SolverSearchState<T>,
        chains: &C,
        mut on_scheduled_item: F,
    ) -> Result<(), SchedulingError>
    where
        C: ChainSetView,
        F: FnMut(&Schedule<T>),
    {
        let model: &SolverModel<T> = solver_state.model();
        let mut modified_berths: HashMap<BerthIndex, BerthOccupancy<'_, T>> =
            HashMap::with_capacity(model.berths_len());

        for chain_idx_val in 0..chains.num_chains() {
            let chain_index = ChainIndex(chain_idx_val);
            let berth_index = BerthIndex(chain_index.get());
            let mut prev_end_time: Option<TimePoint<T>> = None;

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

                let earliest_arrival = match prev_end_time {
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

                let berth_occupancy = modified_berths.entry(berth_index).or_insert_with(|| {
                    model
                        .baseline_occupancy_for_berth(berth_index)
                        .unwrap()
                        .clone()
                });

                let Some(slot) = berth_occupancy
                    .iter_earliest_fit_intervals_in(feasible_window, processing_time)
                    .next()
                else {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                };

                let start_time = std::cmp::max(earliest_arrival, slot.start());
                let Some(end_time) = start_time.checked_add(processing_time) else {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                };

                if end_time > slot.end() || end_time > feasible_window.end() {
                    return Err(SchedulingError::FeasiblyWindowViolation(
                        FeasiblyWindowViolationError::new(request_index),
                    ));
                }

                let service_interval = TimeInterval::new(start_time, end_time);
                berth_occupancy.occupy(service_interval).unwrap();

                // Call the callback with the scheduled item.
                on_scheduled_item(&Schedule::new(request_index, berth_index, service_interval));

                prev_end_time = Some(end_time);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod solution_tests {
    use super::*;
    use crate::state::{
        chain_set::{
            base::ChainSet,
            delta_builder::ChainSetDeltaBuilder,
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
        },
        cost_policy::WeightedFlowTime,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::{FixedKind, FlexibleKind};
    use berth_alloc_model::prelude::{
        Assignment, Berth, BerthIdentifier, RequestIdentifier, SolutionView,
    };
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    // ---------- small helpers ----------
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

    // Link a sequence into a given chain: head -> n0 -> n1 -> ... -> tail
    fn link_sequence(cs: &mut ChainSet, chain: ChainIndex, nodes: &[usize]) {
        let mut builder = ChainSetDeltaBuilder::new(cs);
        let start_node = cs.start_of_chain(chain);
        let mut current_tail = start_node;
        for &node_id in nodes {
            let node_to_link = NodeIndex(node_id);
            builder.insert_after(current_tail, node_to_link);
            current_tail = node_to_link;
        }
        cs.apply_delta(&builder.build());
    }

    #[test]
    fn test_solution_builds_ref_and_is_valid() {
        // b0 availability [0,100)
        // two flex jobs of 5 each on b0, both [0,100) → placed [0,5) and [5,10)
        let b0 = berth(1, 0, 100);
        let f0 = req_flex(10, (0, 100), &[(1, 5)]);
        let f1 = req_flex(11, (0, 100), &[(1, 5)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_flexible(f0);
        pb.add_flexible(f1);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let sol_ref = GreedyEarliest
            .solution(&state, &cs)
            .expect("solution must build");

        // Basic sanity on sizes via SolutionView
        assert_eq!(sol_ref.fixed_assignments_len(), 0);
        assert_eq!(sol_ref.flexible_assignments_len(), 2);
        assert_eq!(sol_ref.total_assignments_len(), 2);
    }

    #[test]
    fn test_ssolution_window_violation_maps_to_missing_flexible_assignment() {
        // One berth b0, availability [0,100).
        // First flex len 8, second flex len 5 with window [0,10].
        // Precedence pushes second to start at >=8, end=13>10 -> fail.
        let b0 = berth(1, 0, 100);
        let f0 = req_flex(100, (0, 100), &[(1, 8)]);
        let f1 = req_flex(101, (0, 10), &[(1, 5)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_flexible(f0);
        pb.add_flexible(f1);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        assert!(GreedyEarliest.solution(&state, &cs).is_err());
    }

    #[test]
    fn test_ssolution_respects_fixed_and_fills_earliest() {
        // b0 availability [0,100), fixed [10,20).
        // two flex jobs of 5 each on b0, both [0,100). Should place [0,5) then [5,10)
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

        let model = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let sol_ref = GreedyEarliest
            .solution(&state, &cs)
            .expect("solution must build");

        // We only check counts here; detailed interval checks would require reading the
        // inner assignment intervals from the borrowing types. Counts are enough to
        // ensure the solution was produced.
        assert_eq!(sol_ref.fixed_assignments_len(), 1);
        assert_eq!(sol_ref.flexible_assignments_len(), 2);
    }

    #[test]
    fn test_ssolution_skips_small_slot_and_uses_later_fit() {
        // b0 availability [0,100), fixed [8,12) -> free slices [0,8) and [12,100).
        // One flex of len 10, window [0,100). Algorithm should use [12,22).
        let b0 = berth(1, 0, 100);
        let rf = req_fixed(901, (0, 100), &[(1, 4)]);
        let a = asg_fixed(&rf, &b0, 8); // occupies [8,12)

        let fx = req_flex(10, (0, 100), &[(1, 10)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b0);
        pb.add_fixed(a);
        pb.add_flexible(fx);
        let p = pb.build().unwrap();

        let model = SolverModel::from_problem(&p, &WeightedFlowTime::default()).unwrap();
        let state = make_state(&model);

        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0]); // just the single request

        let sol_ref = GreedyEarliest
            .solution(&state, &cs)
            .expect("solution must build");

        // We can at least assert counts
        assert_eq!(sol_ref.flexible_assignments_len(), 1);
    }
}
