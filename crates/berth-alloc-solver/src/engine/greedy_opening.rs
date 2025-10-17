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
    engine::opening::OpeningStrategy,
    state::{
        registry::ledger::Ledger,
        solver_state::SolverState,
        terminal::{
            err::TerminalUpdateError,
            terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite},
        },
    },
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    prelude::Problem,
    problem::{
        asg::{AssignmentRef, AssignmentView},
        req::RequestView,
    },
};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::{cmp::Reverse, ops::Mul};

#[derive(Debug, Clone)]
pub struct GreedyOpening<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for GreedyOpening<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GreedyOpening<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GreedyOpeningError<T> {
    TerminalUpdate(TerminalUpdateError<T>),
}

impl<T> From<TerminalUpdateError<T>> for GreedyOpeningError<T> {
    fn from(e: TerminalUpdateError<T>) -> Self {
        Self::TerminalUpdate(e)
    }
}

impl<T> std::fmt::Display for GreedyOpeningError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GreedyOpeningError::TerminalUpdate(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for GreedyOpeningError<T> {}

impl<T> OpeningStrategy<T> for GreedyOpening<T>
where
    T: Copy
        + Ord
        + CheckedAdd
        + CheckedSub
        + Zero
        + Into<Cost>
        + Mul<Output = Cost>
        + std::fmt::Debug,
{
    type Error = GreedyOpeningError<T>;

    fn build<'p>(&self, problem: &'p Problem<T>) -> Result<SolverState<'p, T>, Self::Error> {
        let mut ledger = Ledger::new(problem);
        let mut terminal = TerminalOccupancy::new(problem.iter_berths());

        for a in problem.iter_fixed_assignments() {
            terminal.occupy(a.berth_id(), a.interval())?;
        }

        loop {
            let mut reqs: Vec<_> = ledger.iter_unassigned_requests().collect();
            if reqs.is_empty() {
                break;
            }

            reqs.sort_by_key(|r| (r.request_slack(), Reverse(r.weight()), r.id()));
            let mut placed_in_pass = 0usize;

            for req in reqs {
                let candidates: Vec<_> = terminal
                    .iter_free_intervals_for_berths_in(
                        req.iter_allowed_berths_ids(),
                        req.feasible_window(),
                    )
                    .collect();

                if candidates.is_empty() {
                    continue;
                }

                'try_free: for free in candidates {
                    let berth = free.berth();
                    let start = free.interval().start();

                    let asg = match AssignmentRef::new(req, berth, start) {
                        Ok(a) => a,
                        Err(_) => continue, // try next free interval / berth
                    };

                    let iv = asg.interval();
                    if !free.interval().contains_interval(&iv) {
                        continue;
                    }
                    match ledger.commit_assignment(req, berth, start) {
                        Ok(committed) => {
                            if let Err(_e) = terminal.occupy(berth.id(), iv) {
                                let _ = ledger.uncommit_assignment(&committed);
                                continue;
                            }
                            placed_in_pass += 1;
                            break 'try_free;
                        }
                        Err(_e) => {
                            continue;
                        }
                    }
                }
            }

            if placed_in_pass == 0 {
                break;
            }
        }

        Ok(SolverState::new(ledger, terminal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{berth::berthocc::BerthRead, solver_state::SolverStateView};
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use num_traits::Zero;
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
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn fixed_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_one_berth_two_flex() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // r1 pt=10 on b1, r2 pt=5 on b1
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    #[test]
    fn test_build_assigns_all_when_space() {
        let prob = problem_one_berth_two_flex();
        let solver = GreedyOpening::<i64>::new();

        let state = solver.build(&prob).expect("build should succeed");

        // Both requests should be assigned
        assert!(state.is_feasible());
        assert_eq!(state.ledger().iter_assignments().count(), 2);
        assert!(state.cost() > Cost::zero(), "cost should be positive");

        // Intervals should start at earliest feasible times without overlap:
        // r1 (pt=10) at [0,10), r2 (pt=5) at [10,15) or [0,5) then [5,15) depending on order.
        // Verify no overlap via terminal occupancy view.
        let b = state
            .terminal_occupancy()
            .berth(bid(1))
            .expect("berth 1 exists");
        // [0,15) cannot be entirely free now
        assert!(b.is_occupied(iv(0, 15)));
    }

    #[test]
    fn test_build_respects_fixed_assignments_blocking() {
        // One berth [0,100]; fixed assignment occupies [10,20).
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = berth(1, 0, 100);
        berths.insert(b1.clone());

        let rf = fixed_req(100, (0, 100), &[(1, 10)], 1);
        // Fixed assignment: start at 10, pt=10 => [10,20)
        let af = Assignment::<FixedKind, i64>::new_fixed(rf.clone(), b1.clone(), tp(10)).unwrap();
        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        // Two flexible, each pt=10
        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 100), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 100), &[(1, 10)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&prob).expect("build should succeed");

        // Both flexible requests should be assigned around the fixed block: e.g., [0,10) and [20,30).
        let mut times = Vec::new();
        for a in state.ledger().iter_assignments() {
            if a.request_id() == rid(1) || a.request_id() == rid(2) {
                times.push(a.interval());
            }
        }
        assert_eq!(times.len(), 2, "both flexible assigned");
        // Ensure neither overlaps fixed [10,20)
        for ivx in times {
            assert!(
                !ivx.intersects(&iv(10, 20)),
                "flexible assignment must not overlap fixed block"
            );
        }

        // Terminal occupancy should reflect the fixed block too
        let occ = state
            .terminal_occupancy()
            .berth(bid(1))
            .expect("berth 1 exists");
        assert!(occ.is_occupied(iv(10, 20)));
    }

    #[test]
    fn test_build_leaves_unassigned_when_insufficient_capacity() {
        // One berth [0,12]; two requests with pt=10 each. Only one fits.
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 12));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 12), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 12), &[(1, 10)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&prob).expect("build should succeed");

        // Exactly one assigned, one unassigned
        assert_eq!(state.ledger().iter_assignments().count(), 1);
        assert!(!state.is_feasible());
        assert!(state.cost() > 0);
    }

    #[test]
    fn test_weight_tiebreaker_when_slack_equal() {
        // Equal slack windows: both [0,20), both pt=5.
        // Higher weight should be scheduled first (earlier start).
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 20));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // r10 heavier weight=5, r20 weight=1
        flex.insert(flex_req(10, (0, 20), &[(1, 5)], 5));
        flex.insert(flex_req(20, (0, 20), &[(1, 5)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&prob).expect("build should succeed");

        // Both should be assigned [0,5) and [5,10). The heavier (r10) should get [0,5).
        let mut r10_start = None;
        let mut r20_start = None;
        for a in state.ledger().iter_assignments() {
            if a.request_id() == rid(10) {
                r10_start = Some(a.interval().start());
            } else if a.request_id() == rid(20) {
                r20_start = Some(a.interval().start());
            }
        }
        let r10s = r10_start.expect("r10 assigned");
        let r20s = r20_start.expect("r20 assigned");
        assert!(r10s <= r20s, "heavier request should not start later");
        assert_eq!(r10s, tp(0), "heavier should start at earliest feasible");
        assert_eq!(r20s, tp(5));
    }

    #[test]
    fn test_resulting_state_integrity() {
        // Sanity: building returns a consistent state
        let prob = problem_one_berth_two_flex();
        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&prob).expect("build should succeed");

        // Ledger points to the same problem
        assert!(std::ptr::eq(state.problem(), &prob));
        // Terminal occupancy has the right number of berths (1)
        assert_eq!(state.terminal_occupancy().berths().len(), 1);
        // Fitness aligns with ledger: cost > 0 and feasibility matches unassigned count == 0
        assert_eq!(
            state.is_feasible(),
            state.fitness().unassigned_requests == 0
        );
        if state.is_feasible() {
            assert!(state.cost() > 0);
        }
    }
}
