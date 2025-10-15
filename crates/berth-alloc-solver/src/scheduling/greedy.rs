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
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    scheduling::{
        err::{FeasiblyWindowViolationError, NotAllowedOnBerthError, SchedulingError},
        traits::Scheduler,
    },
    state::chain_set::{
        index::NodeIndex,
        view::{ChainRef, ChainSetView},
    },
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GreedyScheduler;

impl<T> Scheduler<T> for GreedyScheduler
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn schedule_chain_slice<'a, C: ChainSetView>(
        &self,
        solver_model: &SolverModel<'a, T>,
        chain_view: ChainRef<'_, C>,
        slice_start_node: NodeIndex,
        slice_end_node_exclusive: Option<NodeIndex>,
        interval_variables: &mut [IntervalVar<T>],
        decision_variables: &mut [DecisionVar<T>],
    ) -> Result<(), SchedulingError> {
        let berth_index = BerthIndex(chain_view.chain_index().get());

        // Resolve slice bounds (skip sentinels).
        let (first_node_in_slice, resolved_end_node_exclusive) =
            chain_view.resolve_slice(slice_start_node, slice_end_node_exclusive);
        let Some(first_actual_node) = first_node_in_slice else {
            return Ok(());
        };
        let exclusive_end_actual_node = if chain_view.is_sentinel_node(resolved_end_node_exclusive)
        {
            chain_view.first_real_node(resolved_end_node_exclusive)
        } else {
            Some(resolved_end_node_exclusive)
        };

        let mut earliest_time_cursor = if let Some(pred_node) =
            chain_view.prev_real(first_actual_node)
        {
            let pred_req = RequestIndex(pred_node.get());
            match decision_variables.get(pred_req.get()) {
                Some(DecisionVar::Assigned(pred_dec)) if pred_dec.berth_index == berth_index => {
                    let pt = solver_model
                        .processing_time(pred_req, berth_index)
                        .flatten()
                        .ok_or_else(|| {
                            SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                                pred_req,
                                berth_index,
                            ))
                        })?;
                    pred_dec.start_time.checked_add(pt).ok_or_else(|| {
                        SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(
                            pred_req,
                        ))
                    })?
                }
                _ => {
                    // ignore stale/mismatched predecessor DV; seed from current LB
                    let first_idx = RequestIndex(first_actual_node.get());
                    interval_variables[first_idx.get()].start_time_lower_bound
                }
            }
        } else {
            let first_idx = RequestIndex(first_actual_node.get());
            interval_variables[first_idx.get()].start_time_lower_bound
        };

        // Greedy left-justified pass over the slice.
        let mut loop_guard = solver_model.flexible_requests_len();
        let mut current_node = Some(first_actual_node);

        while let Some(node) = current_node {
            if let Some(end_excl) = exclusive_end_actual_node
                && node == end_excl
            {
                break;
            }
            if loop_guard == 0 {
                return Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(RequestIndex(node.get())),
                ));
            }
            loop_guard -= 1;

            let r = RequestIndex(node.get());
            let i = r.get();

            // Trust the interval bounds [L_i, U_i] for start times.
            let ivar = interval_variables[i];
            let lb = core::cmp::max(earliest_time_cursor, ivar.start_time_lower_bound);
            let ub = ivar.start_time_upper_bound;

            if lb > ub {
                return Err(SchedulingError::FeasiblyWindowViolation(
                    FeasiblyWindowViolationError::new(r),
                ));
            }

            // Assign at the left edge.
            let s_i = lb;

            // Update decision variable and advance the cursor by processing time on this berth.
            let pt = solver_model
                .processing_time(r, berth_index)
                .flatten()
                .ok_or_else(|| {
                    SchedulingError::NotAllowedOnBerth(NotAllowedOnBerthError::new(r, berth_index))
                })?;

            let e_i = s_i.checked_add(pt).ok_or_else(|| {
                SchedulingError::FeasiblyWindowViolation(FeasiblyWindowViolationError::new(r))
            })?;

            decision_variables[i] = DecisionVar::assigned(berth_index, s_i);
            earliest_time_cursor = e_i;

            current_node = chain_view.next_real(node);
        }

        Ok(())
    }
}

// The `tests` module remains unchanged as it correctly tests the logic,
// which has not been altered.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{decisionvar::DecisionVar, intervalvar::IntervalVar};
    use crate::scheduling::traits::Scheduler;
    use crate::state::chain_set::overlay::ChainSetOverlay;
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
        let sched = GreedyScheduler;

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
        GreedyScheduler
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap();

        let dec = dvars[0].as_assigned().unwrap();
        assert_eq!(dec.start_time, tp(15));
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
        GreedyScheduler
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
        let err = GreedyScheduler
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
    fn test_slice_prev_assignment_on_different_berth_is_ignored_and_succeeds() {
        // two berths; predecessor (R0) assigned on berth1; we schedule slice on chain0 (berth0)
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

        // predecessor (R0) assigned to berth1 (mismatch) — should be ignored now
        dvars[0] = DecisionVar::assigned(bi(1), tp(0));

        let c0 = cs.chain(ChainIndex(0));
        GreedyScheduler
            .schedule_chain_slice(&m, c0, NodeIndex(1), None, &mut ivars, &mut dvars)
            .unwrap();

        // R1 gets assigned on berth0 at its current LB (0 by default here)
        let d1 = dvars[1].as_assigned().unwrap();
        assert_eq!(d1.berth_index, bi(0));
        assert_eq!(d1.start_time, ivars[1].start_time_lower_bound);
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
        GreedyScheduler
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

        GreedyScheduler
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

        GreedyScheduler
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
        let err = GreedyScheduler
            .schedule_chain(&m, c0, &mut ivars, &mut dvars)
            .unwrap_err();

        match err {
            SchedulingError::FeasiblyWindowViolation(_) => {} // guard tripped
            x => panic!("expected FWV due to loop guard, got {:?}", x),
        }
    }

    #[test]
    fn test_overlay_inter_berth_move_schedules_on_target_chain() {
        // Two berths, all requests allowed on both.
        // c0 (berth0): start -> 0 -> 1 -> end  with PT [3,3]
        // c1 (berth1): start -> 2 -> 3 -> end  with PT [4,5]
        let p = build_problem(
            &[vec![(0, 200)], vec![(0, 200)]],
            &[(0, 200), (0, 200), (0, 200), (0, 200)],
            &[
                vec![Some(3), Some(3)],
                vec![Some(3), Some(3)],
                vec![Some(4), Some(4)],
                vec![Some(5), Some(5)],
            ],
        );
        let m = SolverModel::from_problem(&p).unwrap();
        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        // base chains
        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1]);
        link_chain(&mut cs, 1, &[2, 3]);

        // overlay: move node 2 to the front of chain0:
        //   c0: start->2->0->1->end   ;   c1: start->3->end
        let c0s = cs.start_of_chain(ChainIndex(0));
        let c0e = cs.end_of_chain(ChainIndex(0));
        let c1s = cs.start_of_chain(ChainIndex(1));
        let c1e = cs.end_of_chain(ChainIndex(1));

        let mut delta = ChainSetDelta::new();
        // chain0
        delta.push_rewire(ChainNextRewire::new(c0s, NodeIndex(2)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(2), NodeIndex(0)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(0), NodeIndex(1)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(1), c0e));
        // chain1
        delta.push_rewire(ChainNextRewire::new(c1s, NodeIndex(3)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(3), c1e));

        let overlay = ChainSetOverlay::new(&cs, &delta);
        let oc0 = overlay.chain(ChainIndex(0));
        let oc1 = overlay.chain(ChainIndex(1));

        // Schedule both overlay chains
        GreedyScheduler
            .schedule_chain(&m, oc0, &mut ivars, &mut dvars)
            .unwrap();
        GreedyScheduler
            .schedule_chain(&m, oc1, &mut ivars, &mut dvars)
            .unwrap();

        // On chain0 (berth0): s2=0, s0=4, s1=7
        let d2 = dvars[2].as_assigned().unwrap();
        let d0 = dvars[0].as_assigned().unwrap();
        let d1 = dvars[1].as_assigned().unwrap();
        assert_eq!(d2.berth_index, bi(0));
        assert_eq!(d2.start_time, tp(0));
        assert_eq!(d0.start_time, tp(4)); // after pt2=4
        assert_eq!(d1.start_time, tp(7)); // after pt2+pt0 = 4+3

        // On chain1 (berth1): single job 3 at 0
        let d3 = dvars[3].as_assigned().unwrap();
        assert_eq!(d3.berth_index, bi(1));
        assert_eq!(d3.start_time, tp(0));
    }
}
