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

use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use berth_alloc_model::{
    common::FlexibleKind,
    prelude::{Berth, Request},
    problem::{
        asg::{AssignmentRef, AssignmentView},
        req::RequestView,
    },
};
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

use crate::{
    search::err::{BerthNotFreeError, ProposeAssignmentError, ProposeUnassignmentError},
    state::{
        plan::Plan,
        registry::ledger::Ledger,
        solver_state::SolverState,
        terminal::{
            err::BerthIdentifierNotFoundError,
            sandbox::TerminalSandbox,
            terminalocc::{TerminalOccupancy, TerminalRead},
        },
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeBerth<'p, T: Copy + Ord> {
    interval: TimeInterval<T>,
    berth: &'p Berth<T>,
}

impl<'p, T: Copy + Ord> FreeBerth<'p, T> {
    #[inline]
    pub fn new(interval: TimeInterval<T>, berth: &'p Berth<T>) -> Self {
        Self { interval, berth }
    }

    #[inline]
    pub fn interval(&self) -> &TimeInterval<T> {
        &self.interval
    }

    #[inline]
    pub fn berth(&self) -> &'p Berth<T> {
        self.berth
    }
}

impl<'p, T: Copy + Ord + std::fmt::Display> std::fmt::Display for FreeBerth<'p, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FreeBerth(berth: {}, interval: {})",
            self.berth.id(),
            self.interval
        )
    }
}

#[derive(Debug, Clone)]
pub struct PlanExplorer<'pb, 'p, T: Copy + Ord> {
    ledger: &'pb Ledger<'p, T>,
    sandbox: &'pb TerminalSandbox<'p, T>,
}

impl<'pb, 'p, T: Copy + Ord> PlanExplorer<'pb, 'p, T> {
    #[inline]
    pub fn new(ledger: &'pb Ledger<'p, T>, sandbox: &'pb TerminalSandbox<'p, T>) -> Self {
        Self { ledger, sandbox }
    }

    #[inline]
    pub fn ledger(&self) -> &'pb Ledger<'p, T> {
        self.ledger
    }

    #[inline]
    pub fn sandbox(&self) -> &'pb TerminalSandbox<'p, T> {
        self.sandbox
    }

    #[inline]
    pub fn iter_free_for<R>(&self, req: &'pb R) -> impl Iterator<Item = FreeBerth<'p, T>> + 'pb
    where
        T: CheckedAdd + CheckedSub,
        R: RequestView<T> + ?Sized,
    {
        let window = req.feasible_window();
        let allowed = req.iter_allowed_berths_ids();
        self.sandbox
            .inner()
            .iter_free_intervals_for_berths_in(allowed, window)
            .map(|fb| FreeBerth::new(fb.interval(), fb.berth()))
    }

    #[inline]
    pub fn iter_unassigned_requests(&self) -> impl Iterator<Item = &'p Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger.iter_unassigned_requests()
    }

    #[inline]
    pub fn iter_assigned_requests(&self) -> impl Iterator<Item = &'p Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger.iter_assigned_requests()
    }

    #[inline]
    pub fn iter_assignments(
        &self,
    ) -> impl Iterator<Item = &AssignmentRef<'p, 'p, FlexibleKind, T>> {
        self.ledger.iter_assignments()
    }

    #[inline]
    pub fn peek_cost(
        &self,
        request: &'p Request<FlexibleKind, T>,
        start_time: TimePoint<T>,
        free_berth: &FreeBerth<'p, T>,
    ) -> Option<Cost>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let berth = free_berth.berth();
        let asg = AssignmentRef::new(request, berth, start_time).ok()?;
        Some(asg.cost())
    }
}

#[derive(Debug, Clone)]
pub struct PlanBuilder<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    sandbox: TerminalSandbox<'p, T>,
    delta_cost: Cost,
    delta_unassigned: i32,
}

impl<'p, T: Copy + Ord> PlanBuilder<'p, T> {
    #[inline]
    pub fn new(ledger: Ledger<'p, T>, terminal: TerminalOccupancy<'p, T>) -> Self {
        Self {
            ledger,
            sandbox: TerminalSandbox::new(terminal),
            delta_cost: Cost::zero(),
            delta_unassigned: 0,
        }
    }

    #[inline]
    pub fn delta_cost(&self) -> Cost {
        self.delta_cost
    }

    #[inline]
    pub fn delta_unassigned(&self) -> i32 {
        self.delta_unassigned
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    pub fn sandbox(&self) -> &TerminalSandbox<'p, T> {
        &self.sandbox
    }

    #[inline]
    pub fn propose_assignment(
        &mut self,
        request: &'p Request<FlexibleKind, T>,
        start_time: TimePoint<T>,
        free_berth: &FreeBerth<'p, T>,
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, ProposeAssignmentError<T>>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let berth = free_berth.berth();
        let free_iv = free_berth.interval();
        let asg = AssignmentRef::new(request, berth, start_time)?;
        let asg_iv = asg.interval();

        if !free_iv.contains_interval(&asg_iv) {
            return Err(ProposeAssignmentError::NotFree(BerthNotFreeError::new(
                berth.id(),
                asg_iv,
                *free_iv,
            )));
        }

        let asg = self
            .ledger
            .commit_assignment(request, berth, start_time)
            .map_err(ProposeAssignmentError::from)?;

        self.sandbox
            .occupy(berth.id(), asg_iv)
            .map_err(ProposeAssignmentError::from)?;

        self.delta_cost += asg.cost();
        self.delta_unassigned -= 1;

        Ok(asg)
    }

    #[inline]
    pub fn propose_unassignment(
        &mut self,
        asg: &AssignmentRef<'p, 'p, FlexibleKind, T>,
    ) -> Result<FreeBerth<'p, T>, ProposeUnassignmentError<T>>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let id = asg.berth().id();
        let iv = asg.interval();

        self.ledger
            .uncommit_assignment(asg)
            .map_err(ProposeUnassignmentError::from)?;
        self.sandbox
            .release(id, iv)
            .map_err(ProposeUnassignmentError::from)?;

        self.delta_cost -= asg.cost();
        self.delta_unassigned += 1;

        Ok(FreeBerth::new(iv, asg.berth()))
    }

    #[inline]
    pub fn with_explorer<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&PlanExplorer<'_, 'p, T>) -> R,
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let explorer: PlanExplorer<'_, 'p, T> = PlanExplorer::new(&self.ledger, &self.sandbox);
        f(&explorer)
    }

    #[inline(always)]
    pub fn discard(self) {
        // No-op for now, as we do not have external resources to clean up.
        // Ledger and sandbox will be dropped automatically.
    }

    #[inline]
    pub fn peek_cost(
        &self,
        request: &'p Request<FlexibleKind, T>,
        start_time: TimePoint<T>,
        free_berth: &FreeBerth<'p, T>,
    ) -> Option<Cost>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let berth = free_berth.berth();
        let asg = AssignmentRef::new(request, berth, start_time).ok()?;
        Some(asg.cost())
    }

    #[inline]
    fn finalize(self) -> Result<Plan<'p, T>, BerthIdentifierNotFoundError> {
        Ok(Plan::new_delta(
            self.ledger,
            self.sandbox.delta()?,
            self.delta_cost,
            self.delta_unassigned,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct PlanningContext<'s, 'p, T: Copy + Ord> {
    state: &'s SolverState<'p, T>,
}

impl<'s, 'p, T: Copy + Ord> PlanningContext<'s, 'p, T> {
    #[inline]
    pub fn new(state: &'s SolverState<'p, T>) -> Self {
        Self { state }
    }

    pub fn state(&self) -> &'s SolverState<'p, T> {
        self.state
    }

    // Build a plan using a closure to configure the builder.
    #[inline]
    pub fn with_builder<F>(&self, f: F) -> Result<Plan<'p, T>, BerthIdentifierNotFoundError>
    where
        F: FnOnce(&mut PlanBuilder<'p, T>),
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        // For now we do not use specialized overlays. Just the cloned ledger and terminal,
        // so proposals can be made on them independently from the master state.
        let mut pb = PlanBuilder::new(
            self.state.ledger().clone(),
            self.state.terminal_occupancy().clone(),
        );
        f(&mut pb);
        pb.finalize()
    }

    #[inline]
    pub fn builder(&self) -> PlanBuilder<'p, T>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        PlanBuilder::new(
            self.state.ledger().clone(),
            self.state.terminal_occupancy().clone(),
        )
    }
}

impl<'s, 'p, T: Copy + Ord + std::fmt::Display> std::fmt::Display for PlanningContext<'s, 'p, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PlanningContext(state: {})", self.state)
    }
}

#[cfg(test)]
mod tests {
    use crate::state::berth::berthocc::BerthRead;

    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
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

    fn mk_occ<'b>(berths: &'b [Berth<i64>]) -> TerminalOccupancy<'b, i64> {
        TerminalOccupancy::new(berths)
    }

    #[test]
    fn test_plan_builder_propose_assignment_and_finalize() {
        let prob = problem_one_berth_two_flex();
        let base = vec![berth(1, 0, 1000)];

        let ledger = Ledger::new(&prob);
        let term = mk_occ(&base);
        let mut pb = PlanBuilder::new(ledger, term);

        let req1 = prob.flexible_requests().get(rid(1)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();

        // Free interval across full availability
        let free = FreeBerth::new(iv(0, 1000), b1);

        // Propose assignment of r1 at t=0
        let asg = pb
            .propose_assignment(req1, tp(0), &free)
            .expect("propose assignment should succeed");

        // Ledger mutated
        assert_eq!(pb.ledger().iter_assignments().count(), 1);
        assert_eq!(
            pb.ledger().iter_assigned_requests().next().unwrap().id(),
            rid(1)
        );

        // Sandbox mutated: [0,10) now occupied on b1
        let occ = pb.sandbox().inner().berth(bid(1)).expect("b1 exists");
        assert!(
            !occ.is_free(asg.interval()),
            "assigned interval must be occupied"
        );

        // Deltas tracked
        assert!(pb.delta_cost() > 0, "delta_cost should increase");
        assert_eq!(
            pb.delta_unassigned(),
            -1,
            "one assignment lowers unassigned by 1"
        );

        // Finalize plan
        let plan = pb.finalize().expect("finalize should succeed");
        assert!(plan.delta_cost > 0);
        assert_eq!(plan.delta_unassigned, -1);
        assert!(
            !plan.terminal_delta.is_empty(),
            "delta must carry touched berth"
        );
    }

    #[test]
    fn test_plan_builder_propose_unassignment_roundtrip() {
        let prob = problem_one_berth_two_flex();
        let base = vec![berth(1, 0, 1000)];

        let ledger = Ledger::new(&prob);
        let term = mk_occ(&base);
        let mut pb = PlanBuilder::new(ledger, term);

        let req1 = prob.flexible_requests().get(rid(1)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();
        let free = FreeBerth::new(iv(0, 1000), b1);

        let asg = pb
            .propose_assignment(req1, tp(0), &free)
            .expect("assign ok");
        assert_eq!(pb.ledger().iter_assignments().count(), 1);

        // Now unassign
        let fb = pb
            .propose_unassignment(&asg)
            .expect("unassignment should succeed");

        // Ledger back to zero assignments
        assert_eq!(pb.ledger().iter_assignments().count(), 0);

        // Returned free berth interval should equal the assignment interval
        assert_eq!(fb.interval(), &asg.interval());
        assert_eq!(fb.berth().id(), asg.berth().id());

        // Delta accounting returns to zero
        assert_eq!(pb.delta_unassigned(), 0);
        assert_eq!(pb.delta_cost(), 0);
    }

    #[test]
    fn test_plan_explorer_iterators_reflect_builder_state() {
        let prob = problem_one_berth_two_flex();
        let base = vec![berth(1, 0, 1000)];

        let ledger = Ledger::new(&prob);
        let term = mk_occ(&base);
        let mut pb = PlanBuilder::new(ledger, term);

        // Initially: 2 unassigned, 0 assigned
        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_unassigned_requests().count(), 2);
            assert_eq!(ex.iter_assigned_requests().count(), 0);

            // For r1, we should see at least one free interval on allowed berth in window
            let r1 = ex
                .iter_unassigned_requests()
                .find(|r| r.id() == rid(1))
                .unwrap();
            assert!(ex.iter_free_for(r1).next().is_some());
        });

        // Assign r1
        let r1 = prob.flexible_requests().get(rid(1)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();
        let free = FreeBerth::new(iv(0, 1000), b1);
        pb.propose_assignment(r1, tp(0), &free).expect("assign ok");

        // Now: 1 unassigned, 1 assigned
        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_unassigned_requests().count(), 1);
            assert_eq!(ex.iter_assigned_requests().count(), 1);
            // iter_assignments surfaces the one assignment
            assert_eq!(ex.iter_assignments().count(), 1);
        });
    }

    #[test]
    fn test_planning_context_with_builder_isolated_from_master() {
        let prob = problem_one_berth_two_flex();
        let base = vec![berth(1, 0, 1000)];

        // Build a solver state
        let ledger = Ledger::new(&prob);
        let term = mk_occ(&base);
        let state = SolverState::new(ledger, term);

        // Context over the state
        let ctx = PlanningContext::new(&state);

        // Build a plan that assigns r1
        let plan = ctx
            .with_builder(|pb| {
                let r1 = prob.flexible_requests().get(rid(1)).unwrap();
                let b1 = prob.berths().get(bid(1)).unwrap();
                let free = FreeBerth::new(iv(0, 1000), b1);
                pb.propose_assignment(r1, tp(0), &free).expect("assign ok");
            })
            .expect("plan finalize ok");

        // Master state remains unchanged (no assignments)
        assert_eq!(state.ledger().iter_assignments().count(), 0);

        // Plan carries a mutated ledger with one assignment
        assert_eq!(plan.ledger.iter_assignments().count(), 1);
        assert_eq!(plan.delta_unassigned, -1);
        assert!(plan.delta_cost > 0);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_propose_assignment_not_free_error() {
        let prob = problem_one_berth_two_flex();
        let base = vec![berth(1, 0, 1000)];

        let ledger = Ledger::new(&prob);
        let term = mk_occ(&base);
        let mut pb = PlanBuilder::new(ledger, term);

        let r1 = prob.flexible_requests().get(rid(1)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();

        // Construct a "free berth" interval that is smaller than the assignment window,
        // guaranteeing NotFree before any sandbox/ledger mutation.
        let narrow = FreeBerth::new(iv(5, 7), b1);

        let err = pb.propose_assignment(r1, tp(0), &narrow).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(s.contains("not free"), "expected NotFree error, got: {s}");

        // No mutations occurred
        assert_eq!(pb.ledger().iter_assignments().count(), 0);
        assert_eq!(pb.delta_unassigned(), 0);
        assert_eq!(pb.delta_cost(), 0);
    }
}
