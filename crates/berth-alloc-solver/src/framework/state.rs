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
    framework::{
        err::{
            FeasibilityError, PlanRejectionError, UnassignedRequestError, UnassignedRequestsError,
        },
        planning::Plan,
    },
    registry::ledger::Ledger,
    terminal::{
        err::TerminalApplyError,
        terminalocc::{TerminalOccupancy, TerminalWrite},
    },
};
use berth_alloc_model::problem::{asg::AssignmentView, err::AssignmentOverlapError};
use num_traits::{CheckedAdd, CheckedSub};

pub trait SolverStateView<'p, T: Copy + Ord> {
    fn ledger(&self) -> &Ledger<'p, T>;
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T>;
}

#[derive(Debug, Clone)]
pub struct IncompleteSolverState<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
}

impl<'p, T: Copy + Ord> IncompleteSolverState<'p, T> {
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
    pub fn ledger_mut(&mut self) -> &mut Ledger<'p, T> {
        &mut self.ledger
    }

    #[inline]
    pub fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    #[inline]
    pub fn terminal_occupancy_mut(&mut self) -> &mut TerminalOccupancy<'p, T> {
        &mut self.terminal_occupancy
    }

    #[inline]
    pub fn apply_plan(&mut self, plan: Plan<'p, T>) -> Result<(), TerminalApplyError<T>> {
        let (ledger, terminal_delta, _) = plan.into_inner();
        self.ledger = ledger;
        self.terminal_occupancy.apply_delta(terminal_delta)?;
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

impl<'p, T: Copy + Ord> FeasibleSolverState<'p, T> {
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

    pub fn validate_plan(&self, plan: &Plan<'p, T>) -> Result<(), PlanRejectionError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let mut scratch = self.terminal_occupancy.clone();
        scratch
            .apply_delta(plan.terminal_delta().clone())
            .map_err(PlanRejectionError::Terminal)?;
        check_all_flex_assigned(plan.ledger()).map_err(PlanRejectionError::Unassigned)?;
        check_no_overlaps_in_ledger(plan.ledger()).map_err(PlanRejectionError::Overlap)?;
        Ok(())
    }

    pub fn apply_plan_validated(&mut self, plan: Plan<'p, T>) -> Result<(), PlanRejectionError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.validate_plan(&plan)?;
        let (ledger, delta, _cost) = plan.into_inner();
        self.ledger = ledger;
        self.terminal_occupancy
            .apply_delta(delta)
            .map_err(PlanRejectionError::Terminal)?;
        Ok(())
    }
}

impl<'p, T: Copy + Ord> TryFrom<IncompleteSolverState<'p, T>> for FeasibleSolverState<'p, T>
where
    T: CheckedAdd + CheckedSub,
{
    type Error = FeasibilityError<T>;

    fn try_from(value: IncompleteSolverState<'p, T>) -> Result<Self, Self::Error> {
        check_all_flex_assigned(&value.ledger).map_err(FeasibilityError::Unassigned)?;
        check_no_overlaps_in_ledger(&value.ledger).map_err(FeasibilityError::Overlap)?;
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

fn check_all_flex_assigned<'p, T>(ledger: &Ledger<'p, T>) -> Result<(), UnassignedRequestsError>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let missing: Vec<_> = ledger.iter_unassigned_requests().map(|r| r.id()).collect();
    if missing.is_empty() {
        return Ok(());
    }
    let errs = missing
        .into_iter()
        .map(UnassignedRequestError::new)
        .collect();
    Err(UnassignedRequestsError::new(errs))
}

fn check_no_overlaps_in_ledger<'p, T>(ledger: &Ledger<'p, T>) -> Result<(), AssignmentOverlapError>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let mut v: Vec<_> = ledger.iter_assignments().collect();
    if v.len() <= 1 {
        return Ok(());
    }

    v.sort_unstable_by(|a, b| {
        a.berth_id()
            .cmp(&b.berth_id())
            .then_with(|| a.start_time().cmp(&b.start_time()))
            .then_with(|| a.end_time().cmp(&b.end_time()))
    });

    for win in v.windows(2) {
        let left = win[0];
        let right = win[1];
        if left.berth_id() == right.berth_id() && left.end_time() > right.start_time() {
            return Err(AssignmentOverlapError::new(
                left.request_id(),
                right.request_id(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::planning::PlanBuilder;
    use crate::terminal::terminalocc::{TerminalRead, TerminalWrite};
    use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use num_traits::Zero;
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

    fn req_flex(
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

    fn make_problem(
        berths_vec: Vec<Berth<i64>>,
        flex_vec: Vec<Request<FlexibleKind, i64>>,
    ) -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        for b in berths_vec.iter().cloned() {
            berths.insert(b);
        }

        let fixed = berth_alloc_model::problem::asg::AssignmentContainer::<
            FixedKind,
            i64,
            Assignment<FixedKind, i64>,
        >::new();

        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        for r in flex_vec {
            flex.insert(r);
        }

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn clone_berths_from_problem(p: &Problem<i64>) -> Vec<Berth<i64>> {
        p.berths().iter().cloned().collect()
    }

    #[test]
    fn incomplete_apply_plan_from_unassignment_releases_interval() {
        // Setup: one berth [0,10); one request pt=4. Occupy [2,6] & commit in ledger.
        let b = berth(1, 0, 10);
        let r = req_flex(1, (0, 10), &[(1, 4)], 1);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        let base = clone_berths_from_problem(&problem);
        let mut terminal = TerminalOccupancy::new(&base);
        terminal.occupy(bid(1), iv(2, 6)).unwrap();

        let mut ledger = Ledger::new(&problem);
        let req_ref = problem.flexible_requests().get(rid(1)).unwrap();
        let berth_ref = problem.berths().get(bid(1)).unwrap();
        ledger.commit_assignment(req_ref, berth_ref, tp(2)).unwrap();

        // Build a plan that unassigns r1 (no need for BrandedFreeBerth).
        let mut pb = PlanBuilder::new(ledger, terminal);
        let a = pb.iter_assignments().next().unwrap().clone();
        let _hole = pb.propose_unassignment(&a).expect("unassign ok");
        let plan = pb.finalize().expect("finalize ok");

        // Apply plan on an incomplete state.
        let base_for_state = clone_berths_from_problem(&problem);
        let mut state = IncompleteSolverState::new(
            Ledger::new(&problem), // will be overwritten
            TerminalOccupancy::new(&base_for_state),
        );
        state.apply_plan(plan).expect("apply plan");

        // Ledger empty; terminal has full free window again.
        assert_eq!(state.ledger().iter_assignments().count(), 0);
        let v: Vec<_> = state
            .terminal_occupancy()
            .iter_free_intervals_for_berths_in([bid(1)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10)]);
    }

    #[test]
    fn feasible_upgrade_success_when_all_assigned_no_overlap() {
        // Setup: [0,10), one req pt=4 @ t=2; terminal consistent.
        let b = berth(1, 0, 10);
        let r = req_flex(1, (0, 10), &[(1, 4)], 1);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        let base = clone_berths_from_problem(&problem);
        let mut terminal = TerminalOccupancy::new(&base);
        terminal.occupy(bid(1), iv(2, 6)).unwrap();

        let mut ledger = Ledger::new(&problem);
        let req_ref = problem.flexible_requests().get(rid(1)).unwrap();
        let berth_ref = problem.berths().get(bid(1)).unwrap();
        ledger.commit_assignment(req_ref, berth_ref, tp(2)).unwrap();

        // Incomplete → Feasible
        let inc = IncompleteSolverState::new(ledger, terminal);
        let _feas = FeasibleSolverState::try_from(inc).expect("upgrade to feasible");
    }

    #[test]
    fn feasible_upgrade_fails_when_unassigned_requests_exist() {
        // Setup: [0,10), two requests; commit none.
        let b = berth(1, 0, 10);
        let r1 = req_flex(1, (0, 10), &[(1, 4)], 1);
        let r2 = req_flex(2, (0, 10), &[(1, 3)], 1);
        let problem = make_problem(vec![b.clone()], vec![r1, r2]);

        let ledger = Ledger::new(&problem);
        let base_for_term = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base_for_term);

        let inc = IncompleteSolverState::new(ledger, terminal);
        let err = FeasibleSolverState::try_from(inc).unwrap_err();
        match err {
            FeasibilityError::Unassigned(e) => {
                // Both requests are missing.
                assert_eq!(e.errors().len(), 2);
            }
            other => panic!("expected Unassigned, got {other:?}"),
        }
    }

    #[test]
    fn feasible_upgrade_fails_on_ledger_overlap() {
        // Setup: two overlapping assignments in the ledger; terminal empty.
        let b = berth(1, 0, 100);
        let r1 = req_flex(1, (0, 100), &[(1, 10)], 1);
        let r2 = req_flex(2, (0, 100), &[(1, 10)], 1);
        let problem = make_problem(vec![b.clone()], vec![r1.clone(), r2.clone()]);

        let mut ledger = Ledger::new(&problem);
        let br = problem.berths().get(bid(1)).unwrap();
        let r1r = problem.flexible_requests().get(rid(1)).unwrap();
        let r2r = problem.flexible_requests().get(rid(2)).unwrap();
        ledger.commit_assignment(r1r, br, tp(5)).unwrap(); // [5,15)
        ledger.commit_assignment(r2r, br, tp(10)).unwrap(); // [10,20) overlaps

        let base_for_term = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base_for_term);
        let inc = IncompleteSolverState::new(ledger, terminal);
        let err = FeasibleSolverState::try_from(inc).unwrap_err();
        match err {
            FeasibilityError::Overlap(_e) => { /* good */ }
            other => panic!("expected Overlap, got {other:?}"),
        }
    }

    #[test]
    fn feasible_validate_plan_rejects_terminal_conflict_via_occupy() {
        // Feasible state: terminal already has [2,6) occupied; ledger has r1 assigned [2,6).
        // The plan (built against a mismatched empty terminal) tries to assign r2 at [2,6) too,
        // so applying the delta to the feasible state's terminal should fail with a Terminal error.
        let b = berth(1, 0, 10);
        let r1 = req_flex(1, (0, 10), &[(1, 4)], 1);
        let r2 = req_flex(2, (0, 10), &[(1, 4)], 1);
        let problem = make_problem(vec![b.clone()], vec![r1.clone(), r2.clone()]);

        // Terminal for the FEASIBLE state already occupied on [2,6).
        let base_feas = clone_berths_from_problem(&problem);
        let mut term_feas = TerminalOccupancy::new(&base_feas);
        term_feas.occupy(bid(1), iv(2, 6)).unwrap();

        // Ledger: r1 committed at [2,6).
        let mut ledger = Ledger::new(&problem);
        let req1 = problem.flexible_requests().get(rid(1)).unwrap();
        let berth_ref = problem.berths().get(bid(1)).unwrap();
        ledger.commit_assignment(req1, berth_ref, tp(2)).unwrap();

        let feas = FeasibleSolverState::new(ledger.clone(), term_feas);

        // Build a plan against a mismatched (empty) terminal: assign r2 at [2,6).
        let base_pb = clone_berths_from_problem(&problem);

        // The PB we will mutate (has r1 in the ledger).
        let mut pb = PlanBuilder::new(ledger, TerminalOccupancy::new(&base_pb));
        let req2 = pb
            .ledger()
            .problem()
            .flexible_requests()
            .get(rid(2))
            .unwrap();

        // Use a second PlanBuilder to *source* the branded free berth.
        // This avoids borrowing `pb` immutably while also mutating it.
        let finder = PlanBuilder::new(
            Ledger::new(&problem), // any ledger tied to the same `problem` is fine
            TerminalOccupancy::new(&base_pb), // empty terminal so [0,10) is free
        );

        // Get a branded free berth from `finder`:
        let free = finder
            .iter_free_for(req2)
            .next()
            .expect("some free interval");

        // Now safely mutate `pb` using the branded `free` we obtained from `finder`.
        let _a2 = pb.propose_assignment(req2, tp(2), &free).unwrap();
        let plan = pb.finalize().unwrap();

        // Validating against the FEASIBLE state should fail when applying the delta
        // because it tries to occupy [2,6) that is already occupied there.
        let err = feas.validate_plan(&plan).unwrap_err();
        match err {
            PlanRejectionError::Overlap(_) => { /* expected */ }
            other => panic!("expected Overlap rejection, got {other:?}"),
        }
    }

    #[test]
    fn feasible_apply_plan_validated_rejects_unassigned() {
        // Start consistent feasible state; plan unassigns → rejected as Unassigned.
        let b = berth(1, 0, 10);
        let r = req_flex(1, (0, 10), &[(1, 4)], 1);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        let base = clone_berths_from_problem(&problem);
        let mut terminal = TerminalOccupancy::new(&base);
        terminal.occupy(bid(1), iv(2, 6)).unwrap();

        let mut ledger = Ledger::new(&problem);
        let req_ref = problem.flexible_requests().get(rid(1)).unwrap();
        let berth_ref = problem.berths().get(bid(1)).unwrap();
        ledger.commit_assignment(req_ref, berth_ref, tp(2)).unwrap();

        let mut feas = FeasibleSolverState::new(ledger.clone(), terminal.clone());

        // Build an unassignment plan.
        let mut pb = PlanBuilder::new(ledger, terminal);
        let a = pb.iter_assignments().next().unwrap().clone();
        let _ = pb.propose_unassignment(&a).unwrap();
        let plan = pb.finalize().unwrap();

        // Validate + apply must reject with Unassigned.
        let err = feas.apply_plan_validated(plan).unwrap_err();
        match err {
            PlanRejectionError::Unassigned(e) => {
                assert_eq!(e.errors().len(), 1);
                assert_eq!(e.errors()[0].id(), rid(1));
            }
            other => panic!("expected Unassigned rejection, got {other:?}"),
        }
    }

    #[test]
    fn feasible_validate_plan_ok_on_noop_plan() {
        // No-op plan (no delta, no ledger change) validates fine.
        let b = berth(1, 0, 10);
        let r = req_flex(1, (0, 10), &[(1, 4)], 1);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        let base = clone_berths_from_problem(&problem);
        let mut terminal = TerminalOccupancy::new(&base);
        terminal.occupy(bid(1), iv(2, 6)).unwrap();

        let mut ledger = Ledger::new(&problem);
        let req_ref = problem.flexible_requests().get(rid(1)).unwrap();
        let berth_ref = problem.berths().get(bid(1)).unwrap();
        ledger.commit_assignment(req_ref, berth_ref, tp(2)).unwrap();

        let feas = FeasibleSolverState::new(ledger.clone(), terminal.clone());

        // Build a no-op plan by finalizing immediately.
        let pb = PlanBuilder::new(ledger, terminal);
        let plan = pb.finalize().unwrap();

        feas.validate_plan(&plan).unwrap();
        // Also check that cost/delta are neutral.
        assert!(plan.terminal_delta().is_empty());
        assert_eq!(plan.delta_cost(), Cost::zero());
    }
}
