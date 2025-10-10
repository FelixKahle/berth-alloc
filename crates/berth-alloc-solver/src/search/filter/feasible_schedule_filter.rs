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
    scheduling::traits::CalendarScheduler,
    search::filter::traits::FeasibilityFilter,
    state::{
        chain_set::{delta::ChainSetDelta, overlay::ChainSetOverlay, view::ChainSetView},
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeasibleScheduleFilter<
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
    S: CalendarScheduler<T> + Send + Sync,
> {
    scheduler: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, S> FeasibleScheduleFilter<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
    S: CalendarScheduler<T> + Send + Sync,
{
    #[inline]
    pub fn new(scheduler: S) -> Self {
        Self {
            scheduler,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'model, 'problem, T, S> FeasibilityFilter<'model, 'problem, T> for FeasibleScheduleFilter<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost> + Send + Sync,
    S: CalendarScheduler<T> + Send + Sync,
{
    #[inline]
    fn complexity(&self) -> usize {
        // Rough O(#affected chains) — kept small by delta scoping.
        4
    }

    #[inline]
    fn is_feasible(
        &self,
        delta: &ChainSetDelta,
        search_state: &SolverSearchState<'model, 'problem, T>,
    ) -> bool {
        // If nothing was marked as affected, there is nothing to re-validate.
        let affected = delta.affected_chains();
        if affected.is_empty() {
            return true;
        }

        let model = search_state.model();
        let chain_set = search_state.chain_set();
        let ivars = search_state.interval_vars();
        let overlay = ChainSetOverlay::new(chain_set, delta);

        for &ci in affected {
            let start_node = match overlay.earliest_impacted_on_chain(ci) {
                Some(n) => n,
                None => continue,
            };

            let chain = overlay.chain(ci);

            if let Err(_e) = self
                .scheduler
                .valid_schedule_slice(model, chain, start_node, None, ivars)
            {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::greedy::GreedyCalendar;
    use crate::state::chain_set::base::ChainSet;
    use crate::state::chain_set::index::{ChainIndex, NodeIndex};
    use crate::state::index::BerthIndex;
    use crate::state::model::SolverModel;
    use crate::state::search_state::SolverSearchState;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    // ---------- helpers ----------
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

    // Build a Problem:
    // berths_windows[b] = vec![(s,e), ...] berth b availability windows (ids 0..B-1).
    // request_windows[r] = (s,e) feasible window for request r (ids 0..R-1).
    // processing[r][b] = Some(dur) allowed on berth b; None if disallowed.
    fn build_problem(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let b_len = berths_windows.len();
        let r_len = request_windows.len();
        assert_eq!(processing.len(), r_len);
        for row in processing {
            assert_eq!(row.len(), b_len, "processing rows must match #berths");
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

    // Convenience to mark a chain as affected and set overlay edges:
    // Build overlay-only chain: start -> nodes[0] -> ... -> nodes[k] -> end
    fn delta_link_chain(base: &ChainSet, chain: ChainIndex, nodes: &[usize]) -> ChainSetDelta {
        let mut delta = ChainSetDelta::new();
        delta.mark_chain(chain);

        if nodes.is_empty() {
            return delta;
        }

        let s = base.start_of_chain(chain);
        let e = base.end_of_chain(chain);

        delta.set_next(s, NodeIndex(nodes[0]));
        for w in nodes.windows(2) {
            delta.set_next(NodeIndex(w[0]), NodeIndex(w[1]));
        }
        delta.set_next(NodeIndex(*nodes.last().unwrap()), e);

        delta
    }

    #[test]
    fn empty_delta_is_trivially_feasible() {
        // No affected chains => should be feasible
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();
        let ss = SolverSearchState::new_unassigned(&m, 0, 0);

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        let delta = ChainSetDelta::new();

        assert!(filter.is_feasible(&delta, &ss));
    }

    #[test]
    fn test_single_chain_feasible_returns_true() {
        // One berth and one feasible request
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();
        let ss = SolverSearchState::new_unassigned(&m, 0, 0);

        let base = ss.chain_set();
        let delta = delta_link_chain(base, ChainIndex(0), &[0]);

        // Earliest impacted must be the inserted node
        let overlay = ChainSetOverlay::new(base, &delta);
        assert_eq!(
            overlay.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(0))
        );

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        assert!(filter.is_feasible(&delta, &ss));
    }

    #[test]
    fn test_single_chain_infeasible_due_to_bounds_returns_false() {
        // Force infeasibility via lb > ub on the only request
        let p = build_problem(&[vec![(0, 100)]], &[(0, 100)], &[vec![Some(10)]]);
        let m = SolverModel::from_problem(&p).unwrap();
        let mut ss = SolverSearchState::new_unassigned(&m, 0, 0);

        // lb > ub makes it impossible regardless of calendar
        let ivars = ss.interval_vars_mut();
        ivars[0].start_time_lower_bound = tp(6);
        ivars[0].start_time_upper_bound = tp(5);

        let base = ss.chain_set();
        let delta = delta_link_chain(base, ChainIndex(0), &[0]);

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        assert!(!filter.is_feasible(&delta, &ss));
    }

    #[test]
    fn test_not_allowed_on_chain_berth_returns_false() {
        // Two berths; request allowed only on berth 1; chain 0 is on berth 0
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100)],
            &[vec![None, Some(7)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();
        let ss = SolverSearchState::new_unassigned(&m, 0, 0);

        let base = ss.chain_set();
        let delta = delta_link_chain(base, ChainIndex(0), &[0]);

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        assert!(!filter.is_feasible(&delta, &ss));
    }

    #[test]
    fn test_moves_to_next_calendar_segment_and_is_feasible() {
        // free: [0,5), [8,20); req window [0,30), PT=4; LB=3 ⇒ [3,5) too short -> next segment at 8
        let p = build_problem(&[vec![(0, 5), (8, 20)]], &[(0, 30)], &[vec![Some(4)]]);
        let m = SolverModel::from_problem(&p).unwrap();
        let mut ss = SolverSearchState::new_unassigned(&m, 0, 0);

        // Sanity check calendar
        let cal = m.calendar_for_berth(bi(0)).unwrap();
        assert_eq!(cal.free_intervals(), &[iv(0, 5), iv(8, 20)]);

        // Add a lower bound at 3 to force skipping the first segment
        ss.interval_vars_mut()[0].start_time_lower_bound = tp(3);

        let base = ss.chain_set();
        let delta = delta_link_chain(base, ChainIndex(0), &[0]);

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        assert!(filter.is_feasible(&delta, &ss));
    }

    #[test]
    fn test_multiple_chains_mixed_feasibility_returns_false() {
        // Two berths and two requests.
        // Chain 0: request 0 feasible.
        // Chain 1: request 1 infeasible due to bounds.
        let p = build_problem(
            &[vec![(0, 100)], vec![(0, 100)]],
            &[(0, 100), (0, 100)],
            &[
                vec![Some(5), Some(5)],   // req 0 allowed on both berths
                vec![Some(10), Some(10)], // req 1 allowed on both; feasibility controlled via bounds below
            ],
        );
        let m = SolverModel::from_problem(&p).unwrap();
        let mut ss = SolverSearchState::new_unassigned(&m, 0, 0);

        // Make request 1 infeasible via lb > ub
        let ivars = ss.interval_vars_mut();
        ivars[1].start_time_lower_bound = tp(6);
        ivars[1].start_time_upper_bound = tp(5);

        let base = ss.chain_set();
        // One delta linking both chains; mark both as affected
        let mut delta = ChainSetDelta::new();
        delta.mark_chain(ChainIndex(0));
        delta.mark_chain(ChainIndex(1));
        // Chain 0: start -> 0 -> end
        let s0 = base.start_of_chain(ChainIndex(0));
        let e0 = base.end_of_chain(ChainIndex(0));
        delta.set_next(s0, NodeIndex(0));
        delta.set_next(NodeIndex(0), e0);
        // Chain 1: start -> 1 -> end
        let s1 = base.start_of_chain(ChainIndex(1));
        let e1 = base.end_of_chain(ChainIndex(1));
        delta.set_next(s1, NodeIndex(1));
        delta.set_next(NodeIndex(1), e1);

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        assert!(!filter.is_feasible(&delta, &ss));
    }

    #[test]
    fn test_chain_marked_but_empty_is_ignored_and_true() {
        // Mark a chain as affected but keep it empty (no overlay edges); should be treated as no-op and true.
        let p = build_problem(&[vec![(0, 100)]], &[], &[]);
        let m = SolverModel::from_problem(&p).unwrap();
        let ss = SolverSearchState::new_unassigned(&m, 0, 0);

        let mut delta = ChainSetDelta::new();
        delta.mark_chain(ChainIndex(0)); // affected, but earliest_impacted_on_chain will return None

        let filter = FeasibleScheduleFilter::<i64, GreedyCalendar>::new(GreedyCalendar);
        assert!(filter.is_feasible(&delta, &ss));
    }
}
