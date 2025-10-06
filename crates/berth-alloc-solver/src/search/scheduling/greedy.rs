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
        chain_set::{
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
        },
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
    },
};
use berth_alloc_core::prelude::{TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone, Default)]
pub struct GreedyEarliest;

impl<T> Scheduler<T> for GreedyEarliest
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug,
{
    type ScheduleIter<'a, C: ChainSetView + 'a>
        = GreedyEarliestIter<'a, T, C>
    where
        T: 'a,
        Self: 'a;

    #[inline]
    fn schedules<'a, C: ChainSetView + 'a>(
        &self,
        solver_model: &'a SolverModel<'a, T>,
        chains: &'a C,
    ) -> Self::ScheduleIter<'a, C> {
        GreedyEarliestIter::new(solver_model, chains)
    }
}

#[derive(Debug)]
pub struct GreedyEarliestIter<'a, T, C>
where
    T: Copy + Ord,
    C: ChainSetView + 'a,
{
    solver_model: &'a SolverModel<'a, T>,
    chains: &'a C,
    current_chain_idx: usize,
    num_chains: usize,
    current_node: Option<NodeIndex>,
    end_sentinel: Option<NodeIndex>,
    modified_berths: Vec<Option<BerthOccupancy<'a, T>>>,
    prev_end_time: Option<TimePoint<T>>,
    current_berth_index: BerthIndex,
}

impl<'a, T, C> GreedyEarliestIter<'a, T, C>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug,
    C: ChainSetView + 'a,
{
    #[inline]
    pub fn new(solver_model: &'a SolverModel<'a, T>, chains: &'a C) -> Self {
        Self {
            solver_model,
            chains,
            current_chain_idx: 0,
            num_chains: chains.num_chains(),
            current_node: None,
            end_sentinel: None,
            modified_berths: std::iter::repeat_with(|| None)
                .take(solver_model.berths_len())
                .collect(),
            prev_end_time: None,
            current_berth_index: BerthIndex(0),
        }
    }

    #[inline]
    fn start_next_chain(&mut self) -> bool {
        if self.current_chain_idx >= self.num_chains {
            return false;
        }
        let chain_index = ChainIndex(self.current_chain_idx);
        let start = self.chains.start_of_chain(chain_index);
        let end = self.chains.end_of_chain(chain_index);
        let first = self.chains.next_node(start).unwrap_or(end);

        self.current_node = Some(first);
        self.end_sentinel = Some(end);
        self.prev_end_time = None;
        self.current_berth_index = BerthIndex(self.current_chain_idx);
        self.current_chain_idx += 1;
        true
    }
}

impl<'a, T, C> Iterator for GreedyEarliestIter<'a, T, C>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug,
    C: ChainSetView + 'a,
{
    type Item = Result<Schedule<T>, SchedulingError>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_node.is_none() {
                if !self.start_next_chain() {
                    return None;
                }
            }

            let node = self.current_node.unwrap();
            let end = self.end_sentinel.unwrap();

            if node == end {
                self.current_node = None;
                continue;
            }

            let request_index = RequestIndex(node.get());
            let model = self.solver_model;

            let feasible_window = model.feasible_intervals()[request_index.get()];

            let Some(processing_time) = model
                .processing_time(request_index, self.current_berth_index)
                .flatten()
            else {
                self.current_chain_idx = self.num_chains;
                self.current_node = None;
                return Some(Err(SchedulingError::NotAllowedOnBerth(
                    NotAllowedOnBerthError::new(request_index, self.current_berth_index),
                )));
            };

            let earliest_arrival = match self.prev_end_time {
                None => feasible_window.start(),
                Some(prev_end) => {
                    if prev_end > feasible_window.end() {
                        self.current_chain_idx = self.num_chains;
                        self.current_node = None;
                        return Some(Err(SchedulingError::FeasiblyWindowViolation(
                            FeasiblyWindowViolationError::new(request_index),
                        )));
                    }
                    if prev_end > feasible_window.start() {
                        prev_end
                    } else {
                        feasible_window.start()
                    }
                }
            };

            let b = self.current_berth_index.get();
            let berth_occupancy = self.modified_berths[b].get_or_insert_with(|| {
                model
                    .baseline_occupancy_for_berth(self.current_berth_index)
                    .expect("baseline exists")
                    .clone()
            });

            let Some(slot) = berth_occupancy
                .iter_earliest_fit_intervals_in(feasible_window, processing_time)
                .next()
            else {
                self.current_chain_idx = self.num_chains;
                self.current_node = None;
                return Some(Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(request_index),
                )));
            };

            let start_time = if earliest_arrival > slot.start() {
                earliest_arrival
            } else {
                slot.start()
            };

            let Some(end_time) = start_time.checked_add(processing_time) else {
                self.current_chain_idx = self.num_chains;
                self.current_node = None;
                return Some(Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(request_index),
                )));
            };

            if end_time > slot.end() || end_time > feasible_window.end() {
                self.current_chain_idx = self.num_chains;
                self.current_node = None;
                return Some(Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(request_index),
                )));
            }

            let service_interval = TimeInterval::new(start_time, end_time);
            berth_occupancy
                .occupy(service_interval)
                .expect("occupy must succeed");

            let out = Schedule::new(request_index, self.current_berth_index, service_interval);
            self.prev_end_time = Some(end_time);
            self.current_node = self.chains.next_node(node);
            return Some(Ok(out));
        }
    }
}

#[cfg(test)]
mod solution_tests {
    use super::*;
    use crate::state::chain_set::{
        base::ChainSet,
        delta_builder::ChainSetDeltaBuilder,
        index::{ChainIndex, NodeIndex},
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::{FixedKind, FlexibleKind};
    use berth_alloc_model::prelude::{
        Assignment, Berth, BerthIdentifier, RequestIdentifier, SolutionView,
    };
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
    fn berth(id: usize, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn req_flex(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn req_fixed(id: usize, window: (i64, i64), pts: &[(usize, i64)]) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn asg_fixed(
        req: &Request<FixedKind, i64>,
        berth: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new(req.clone(), berth.clone(), tp(start)).unwrap()
    }

    fn link_sequence(cs: &mut ChainSet, chain: ChainIndex, nodes: &[usize]) {
        let mut builder = ChainSetDeltaBuilder::new(cs);
        let start_node = cs.start_of_chain(chain);
        let mut current_tail = start_node;
        for &node_id in nodes {
            let node_to_link = NodeIndex(node_id);
            builder.insert_after(current_tail, node_to_link);
            current_tail = node_to_link;
        }
        cs.apply_delta(builder.build());
    }

    #[test]
    fn test_solution_builds_ref_and_is_valid() {
        let b0 = berth(1, 0, 100);
        let f0 = req_flex(10, (0, 100), &[(1, 5)]);
        let f1 = req_flex(11, (0, 100), &[(1, 5)]);
        // FIXED: Use plural builder methods
        let p = ProblemBuilder::new()
            .with_berths([b0])
            .with_flexible_requests([f0, f1])
            .build()
            .unwrap();
        let model = SolverModel::from_problem(&p).unwrap();
        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let sol_ref = GreedyEarliest
            .solution(&model, &cs)
            .expect("solution must build");

        assert_eq!(sol_ref.fixed_assignments_len(), 0);
        assert_eq!(sol_ref.flexible_assignments_len(), 2);
        assert_eq!(sol_ref.total_assignments_len(), 2);
    }

    #[test]
    fn test_solution_window_violation_maps_to_missing_flexible_assignment() {
        let b0 = berth(1, 0, 100);
        let f0 = req_flex(100, (0, 100), &[(1, 8)]);
        let f1 = req_flex(101, (0, 10), &[(1, 5)]);
        // FIXED: Use plural builder methods
        let p = ProblemBuilder::new()
            .with_berths([b0])
            .with_flexible_requests([f0, f1])
            .build()
            .unwrap();
        let model = SolverModel::from_problem(&p).unwrap();
        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        assert!(GreedyEarliest.solution(&model, &cs).is_err());
    }

    #[test]
    fn test_solution_respects_fixed_and_fills_earliest() {
        let b0 = berth(1, 0, 100);
        let rf = req_fixed(900, (0, 100), &[(1, 10)]);
        let a = asg_fixed(&rf, &b0, 10);
        let f0 = req_flex(10, (0, 100), &[(1, 5)]);
        let f1 = req_flex(11, (0, 100), &[(1, 5)]);
        // FIXED: Use plural builder methods
        let p = ProblemBuilder::new()
            .with_berths([b0])
            .with_fixed_assignments([a])
            .with_flexible_requests([f0, f1])
            .build()
            .unwrap();
        let model = SolverModel::from_problem(&p).unwrap();
        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0, 1]);

        let sol_ref = GreedyEarliest
            .solution(&model, &cs)
            .expect("solution must build");

        assert_eq!(sol_ref.fixed_assignments_len(), 1);
        assert_eq!(sol_ref.flexible_assignments_len(), 2);
    }

    #[test]
    fn test_solution_skips_small_slot_and_uses_later_fit() {
        let b0 = berth(1, 0, 100);
        let rf = req_fixed(901, (0, 100), &[(1, 4)]);
        let a = asg_fixed(&rf, &b0, 8); // occupies [8,12)
        let fx = req_flex(10, (0, 100), &[(1, 10)]);
        // FIXED: Use plural builder methods
        let p = ProblemBuilder::new()
            .with_berths([b0])
            .with_fixed_assignments([a])
            .with_flexible_requests([fx])
            .build()
            .unwrap();
        let model = SolverModel::from_problem(&p).unwrap();
        let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
        link_sequence(&mut cs, ChainIndex(0), &[0]);

        let sol_ref = GreedyEarliest
            .solution(&model, &cs)
            .expect("solution must build");

        assert_eq!(sol_ref.flexible_assignments_len(), 1);
    }
}
