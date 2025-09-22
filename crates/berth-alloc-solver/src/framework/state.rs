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

use berth_alloc_model::{
    prelude::{SolutionRef, StateValidator},
    solution::SolutionError,
};
use num_traits::{CheckedAdd, CheckedSub};

use crate::{
    framework::{
        err::{FeasibilityError, IncompleteSolverStatePlanApplyError, PlanRejectionError},
        planning::Plan,
    },
    registry::ledger::Ledger,
    terminal::terminalocc::{TerminalOccupancy, TerminalWrite},
};

pub trait SolverStateView<'p, T: Copy + Ord> {
    fn ledger(&self) -> &Ledger<'p, T>;
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T>;
}

#[derive(Debug, Clone)]
pub struct IncompleteSolverState<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub> IncompleteSolverState<'p, T> {
    #[inline]
    pub fn new(ledger: Ledger<'p, T>, terminal_occupancy: TerminalOccupancy<'p, T>) -> Self {
        Self {
            ledger,
            terminal_occupancy,
        }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    pub fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    #[inline]
    pub fn apply_plan(
        &mut self,
        plan: Plan<'p, T>,
    ) -> Result<(), IncompleteSolverStatePlanApplyError<T>> {
        StateValidator::validate_nonoverlap(
            plan.ledger().fixed_assignments(),
            plan.ledger().commited_assignments(),
            plan.ledger().problem(),
        )?;
        StateValidator::validate_no_extra_flexible_assignments(
            plan.ledger().commited_assignments(),
        )?;
        StateValidator::validate_request_ids_unique(
            plan.ledger().fixed_assignments(),
            plan.ledger().commited_assignments(),
        )?;
        StateValidator::validate_no_extra_flexible_requests(
            plan.ledger().commited_assignments(),
            plan.ledger().problem(),
        )?;

        let (ledger, delta, _) = plan.into_inner();
        self.ledger = ledger;
        self.terminal_occupancy.apply_delta(delta)?;
        Ok(())
    }
}

impl<'p, T: Copy + Ord> SolverStateView<'p, T> for IncompleteSolverState<'p, T> {
    #[inline]
    fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }
}

#[derive(Debug, Clone)]
pub struct FeasibleSolverState<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub> FeasibleSolverState<'p, T> {
    #[inline]
    pub fn new(ledger: Ledger<'p, T>, terminal_occupancy: TerminalOccupancy<'p, T>) -> Self {
        Self {
            ledger,
            terminal_occupancy,
        }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    pub fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    pub fn validate_plan(&self, plan: &Plan<'p, T>) -> Result<(), PlanRejectionError<T>> {
        let mut scratch = self.terminal_occupancy.clone();
        scratch
            .apply_delta(plan.terminal_delta().clone())
            .map_err(PlanRejectionError::Terminal)?;

        let l = plan.ledger();
        StateValidator::validate_nonoverlap(
            l.fixed_assignments(),
            l.commited_assignments(),
            l.problem(),
        )?;
        StateValidator::validate_no_extra_flexible_assignments(l.commited_assignments())?;
        StateValidator::validate_request_ids_unique(
            l.fixed_assignments(),
            l.commited_assignments(),
        )?;
        StateValidator::validate_no_extra_flexible_requests(l.commited_assignments(), l.problem())?;

        StateValidator::validate_all_flexible_assignments_present(
            l.commited_assignments(),
            l.problem(),
        )?;

        Ok(())
    }

    pub fn apply_plan_validated(&mut self, plan: Plan<'p, T>) -> Result<(), PlanRejectionError<T>> {
        self.validate_plan(&plan)?;
        let (ledger, delta, _cost) = plan.into_inner();
        self.ledger = ledger;
        self.terminal_occupancy
            .apply_delta(delta)
            .map_err(PlanRejectionError::Terminal)?;
        Ok(())
    }
}

impl<'p, T> TryFrom<IncompleteSolverState<'p, T>> for FeasibleSolverState<'p, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    type Error = FeasibilityError;

    fn try_from(value: IncompleteSolverState<'p, T>) -> Result<Self, Self::Error> {
        let ledger = &value.ledger;

        StateValidator::validate_nonoverlap(
            ledger.fixed_assignments(),
            ledger.commited_assignments(),
            ledger.problem(),
        )?;
        StateValidator::validate_no_extra_flexible_assignments(ledger.commited_assignments())?;
        StateValidator::validate_request_ids_unique(
            ledger.fixed_assignments(),
            ledger.commited_assignments(),
        )?;
        StateValidator::validate_no_extra_flexible_requests(
            ledger.commited_assignments(),
            ledger.problem(),
        )?;
        StateValidator::validate_all_flexible_assignments_present(
            ledger.commited_assignments(),
            ledger.problem(),
        )?;

        Ok(Self {
            ledger: value.ledger,
            terminal_occupancy: value.terminal_occupancy,
        })
    }
}

impl<'p, T: Copy + Ord> SolverStateView<'p, T> for FeasibleSolverState<'p, T> {
    #[inline]
    fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }
    #[inline]
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub> TryInto<SolutionRef<'p, T>>
    for FeasibleSolverState<'p, T>
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

#[cfg(test)]
mod feasible_state_tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
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
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn mk_occ<'b>(berths: &'b [Berth<i64>]) -> TerminalOccupancy<'b, i64> {
        TerminalOccupancy::new(berths)
    }

    #[test]
    fn test_feasible_try_from_ok_when_all_flex_assigned_and_no_conflicts() {
        let prob = problem_one_berth_two_flex();

        let mut ledger = Ledger::new(&prob);
        let b = prob.berths().get(bid(1)).unwrap();
        let r1 = prob.flexible_requests().get(rid(1)).unwrap();
        let r2 = prob.flexible_requests().get(rid(2)).unwrap();
        ledger.commit_assignment(r1, b, tp(0)).unwrap(); // [0,10)
        ledger.commit_assignment(r2, b, tp(10)).unwrap(); // [10,15)

        let berths_vec: Vec<Berth<i64>> = prob.berths().iter().cloned().collect();
        let occ = mk_occ(&berths_vec);

        let inc = IncompleteSolverState::new(ledger, occ);
        let feas = FeasibleSolverState::try_from(inc).expect("should be feasible");

        let ids: Vec<_> = feas
            .ledger()
            .iter_assigned_requests()
            .map(|r| r.id())
            .collect();
        assert!(ids.contains(&rid(1)) && ids.contains(&rid(2)));
    }

    #[test]
    fn test_feasible_try_from_err_when_some_flex_unassigned() {
        let prob = problem_one_berth_two_flex();

        let mut ledger = Ledger::new(&prob);
        let b = prob.berths().get(bid(1)).unwrap();
        let r1 = prob.flexible_requests().get(rid(1)).unwrap();
        ledger.commit_assignment(r1, b, tp(0)).unwrap();

        let berths_vec: Vec<Berth<i64>> = prob.berths().iter().cloned().collect();
        let occ = mk_occ(&berths_vec);

        let inc = IncompleteSolverState::new(ledger, occ);
        let err = FeasibleSolverState::try_from(inc).unwrap_err();
        match err {
            FeasibilityError::MissingFlexibleAssignment(e) => {
                assert_eq!(e.request_id(), rid(2));
            }
            other => panic!("expected MissingFlexibleAssignment, got {other:?}"),
        }
    }
}
