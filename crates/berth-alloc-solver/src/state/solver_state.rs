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

use crate::state::{
    fitness::Fitness,
    plan::Plan,
    registry::ledger::Ledger,
    terminal::terminalocc::{TerminalOccupancy, TerminalWrite},
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    prelude::{Problem, SolutionRef},
    solution::SolutionError,
    validation,
};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

pub trait SolverStateView<'p, T: Copy + Ord> {
    fn ledger(&self) -> &Ledger<'p, T>;
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T>;

    fn fitness(&self) -> &Fitness;

    fn is_feasible(&self) -> bool {
        self.fitness().unassigned_requests == 0
    }

    fn cost(&self) -> Cost {
        self.fitness().cost
    }
}

#[derive(Debug, Clone)]
pub struct SolverState<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
    fitness: Fitness,
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>>
    SolverState<'p, T>
{
    #[inline]
    pub fn new(ledger: Ledger<'p, T>, terminal_occupancy: TerminalOccupancy<'p, T>) -> Self {
        let fitness: Fitness = (&ledger).into();

        Self {
            ledger,
            terminal_occupancy,
            fitness,
        }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    pub fn problem(&self) -> &'p Problem<T> {
        self.ledger.problem()
    }

    #[inline]
    pub fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    #[inline]
    pub fn apply_plan(&mut self, plan: Plan<'p, T>)
    where
        T: std::fmt::Debug,
    {
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                validation::validate_nonoverlap(
                    plan.ledger.fixed_assignments(),
                    plan.ledger.commited_assignments(),
                    plan.ledger.problem(),
                )
                .is_ok(),
            );
            debug_assert!(
                validation::validate_no_extra_flexible_assignments(
                    plan.ledger.commited_assignments()
                )
                .is_ok()
            );
            debug_assert!(
                validation::validate_request_ids_unique(
                    plan.ledger.fixed_assignments(),
                    plan.ledger.commited_assignments(),
                )
                .is_ok()
            );
            debug_assert!(
                validation::validate_no_extra_flexible_requests(
                    plan.ledger.commited_assignments(),
                    plan.ledger.problem(),
                )
                .is_ok()
            );
        }

        #[cfg(debug_assertions)]
        let prev_fit = self.fitness.clone();

        // Swap in proposed ledger then apply terminal delta
        self.ledger = plan.ledger;
        let res = self.terminal_occupancy.apply_delta(plan.terminal_delta);
        debug_assert!(res.is_ok(), "Failed to apply terminal delta: {:?}", res);

        // Recompute fitness from the new ledger
        self.fitness = (&self.ledger).into();

        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.fitness.unassigned_requests as i32,
                prev_fit.unassigned_requests as i32 + plan.delta_unassigned,
                "prev_unassigned + delta_unassigned must equal new unassigned"
            );
            debug_assert_eq!(
                self.fitness.cost,
                prev_fit.cost + plan.delta_cost,
                "prev_cost + delta_cost must equal new cost"
            );
        }
    }
}

impl<'p, T: Copy + Ord> SolverStateView<'p, T> for SolverState<'p, T> {
    #[inline]
    fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    #[inline]
    fn fitness(&self) -> &Fitness {
        &self.fitness
    }
}

impl<'p, T: Copy + Ord + std::fmt::Display> std::fmt::Display for SolverState<'p, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SolverState {{ cost: {}, unassigned_requests: {} }}",
            self.fitness.cost, self.fitness.unassigned_requests
        )
    }
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub> TryInto<SolutionRef<'p, T>>
    for SolverState<'p, T>
{
    type Error = SolutionError;

    fn try_into(self) -> Result<SolutionRef<'p, T>, Self::Error> {
        let problem = self.ledger.problem();
        let fixed_refs = self
            .ledger
            .problem()
            .fixed_assignments()
            .iter()
            .map(|a| a.to_ref())
            .collect();
        let flexible_refs = self.ledger.into_inner();
        SolutionRef::new(fixed_refs, flexible_refs, problem)
    }
}

#[allow(dead_code)]
#[cfg(test)]
mod feasible_state_tests {
    use crate::state::{
        berth::berthocc::{BerthRead, BerthWrite},
        terminal::terminalocc::TerminalRead,
    };

    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::req::RequestView};
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
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_one_berth_two_flex() -> Problem<i64> {
        // berths
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        // fixed (empty)
        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        // flexible: r1 (pt=10 on b1), r2 (pt=5 on b1)
        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn mk_occ<'b>(berths: &'b [Berth<i64>]) -> TerminalOccupancy<'b, i64> {
        TerminalOccupancy::new(berths)
    }

    #[test]
    fn test_new_initializes_fitness_from_ledger() {
        let prob = problem_one_berth_two_flex();
        let ledger = Ledger::new(&prob);

        // Build a terminal occupancy from the same berth shape used in the problem.
        let base = vec![berth(1, 0, 1000)];
        let term = mk_occ(&base);

        let st = SolverState::new(ledger, term);
        assert_eq!(
            st.fitness().unassigned_requests,
            2,
            "two flexible requests initially unassigned"
        );
        assert_eq!(st.fitness().cost, 0, "initial cost is zero");
        assert!(!st.is_feasible(), "not feasible until all assigned");
        assert_eq!(st.cost(), 0, "cost() forwards from fitness");
    }

    #[test]
    fn test_apply_plan_updates_ledger_terminal_and_fitness() {
        use crate::state::berth::berthocc::BerthOccupancy;
        use crate::state::plan::Plan;
        use crate::state::terminal::delta::TerminalDelta;

        let prob = problem_one_berth_two_flex();

        // Base terminal occupancy using the same berth definition as the problem.
        let base = vec![berth(1, 0, 1000)];
        let mut st = {
            let ledger0 = Ledger::new(&prob);
            let term0 = mk_occ(&base);
            SolverState::new(ledger0, term0)
        };

        // Build a new ledger with r1 committed at t=0
        let mut ledger1 = Ledger::new(&prob);
        let req1 = prob
            .flexible_requests()
            .get(RequestIdentifier::new(1))
            .expect("r1 exists");
        let b1 = prob
            .berths()
            .get(BerthIdentifier::new(1))
            .expect("berth 1 exists");
        ledger1
            .commit_assignment(req1, b1, tp(0))
            .expect("commit should succeed");

        // Create a terminal delta that reflects occupying [0,10) on berth 1
        let mut b1_occ = BerthOccupancy::new(&base[0]);
        b1_occ.occupy(iv(0, 10)).expect("occupy should succeed");
        let delta = TerminalDelta::from_updates(vec![(BerthIdentifier::new(1), b1_occ)]);

        // Plan bookkeeping must match SolverState::apply_plan debug assertions
        let delta_cost = ledger1.cost() - st.fitness().cost;
        let delta_unassigned =
            ledger1.unassigned_request_count() as i32 - st.fitness().unassigned_requests as i32;

        // Construct the plan using the updated ledger and delta
        let plan = Plan {
            ledger: ledger1,
            terminal_delta: delta,
            delta_cost,
            delta_unassigned,
        };

        // Apply plan
        st.apply_plan(plan);

        // Ledger should now have r1 assigned, r2 unassigned
        let assigned_ids: Vec<_> = st
            .ledger()
            .iter_assigned_requests()
            .map(|r| r.id())
            .collect();
        assert_eq!(assigned_ids, vec![RequestIdentifier::new(1)]);

        let unassigned_ids: Vec<_> = st
            .ledger()
            .iter_unassigned_requests()
            .map(|r| r.id())
            .collect();
        assert_eq!(unassigned_ids, vec![RequestIdentifier::new(2)]);

        // Fitness should reflect 1 unassigned and positive cost
        assert_eq!(st.fitness().unassigned_requests, 1);
        assert!(st.fitness().cost > 0);
        assert_eq!(st.cost(), st.fitness().cost);

        // Terminal occupancy should mark [0,10) on berth 1 as not free
        let berth_view = st
            .terminal_occupancy()
            .berth(BerthIdentifier::new(1))
            .expect("berth 1 should exist in terminal");
        assert!(
            !berth_view.is_free(iv(0, 10)),
            "interval [0,10) should be occupied"
        );
    }

    #[test]
    fn test_try_into_solution_ref_success() {
        let prob = problem_one_berth_two_flex();

        // Start with empty ledger and terminal occupancy
        let base = vec![berth(1, 0, 1000)];
        let ledger = Ledger::new(&prob);
        let term = mk_occ(&base);

        // Make a state, then commit both assignments so the solution is complete
        let mut st = SolverState::new(ledger, term);
        let r1 = prob.flexible_requests().get(rid(1)).unwrap();
        let r2 = prob.flexible_requests().get(rid(2)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();

        st.ledger
            .commit_assignment(r1, b1, tp(0))
            .expect("commit r1 should succeed");
        st.ledger
            .commit_assignment(r2, b1, tp(20))
            .expect("commit r2 should succeed");

        // Recompute fitness by constructing a fresh state
        let st = SolverState::new(st.ledger, st.terminal_occupancy);

        // Now conversion to SolutionRef should succeed (no unassigned flex requests)
        let sol: Result<SolutionRef<'_, i64>, _> = st.clone().try_into();
        assert!(sol.is_ok(), "conversion to SolutionRef should succeed");
    }

    #[test]
    fn test_is_feasible_when_all_assigned() {
        let prob = problem_one_berth_two_flex();

        // Base term and ledger
        let base = vec![berth(1, 0, 1000)];
        let mut ledger = Ledger::new(&prob);
        let term = mk_occ(&base);

        // Commit both requests
        let r1 = prob.flexible_requests().get(rid(1)).unwrap();
        let r2 = prob.flexible_requests().get(rid(2)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();
        ledger.commit_assignment(r1, b1, tp(0)).unwrap();
        ledger.commit_assignment(r2, b1, tp(20)).unwrap();

        let st = SolverState::new(ledger, term);
        assert!(
            st.is_feasible(),
            "all flexible requests assigned means feasible"
        );
        assert!(
            st.cost() > 0,
            "cost should be positive when assignments exist"
        );
    }
}
