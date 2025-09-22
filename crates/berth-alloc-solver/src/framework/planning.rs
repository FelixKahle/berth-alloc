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
    framework::err::{BerthNotFreeError, ProposeAssignmentError, ProposeUnassignmentError},
    registry::ledger::Ledger,
    terminal::{
        delta::TerminalDelta,
        err::BerthIdentifierNotFoundError,
        sandbox::TerminalSandbox,
        terminalocc::{TerminalOccupancy, TerminalRead},
    },
};
use berth_alloc_core::{
    prelude::{Cost, TimeInterval, TimePoint},
    utils::marker::Brand,
};
use berth_alloc_model::{
    common::{FlexibleKind, Kind},
    prelude::{Berth, Request},
    problem::asg::{AssignmentRef, AssignmentView},
};
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct BrandedRequest<'pb, 'p, K: Kind, T: Copy + Ord> {
    req: &'p Request<K, T>,
    _brand: Brand<'pb>,
}

impl<'pb, 'p, K: Kind, T: Copy + Ord> BrandedRequest<'pb, 'p, K, T> {
    #[inline]
    fn new(req: &'p Request<K, T>) -> Self {
        Self {
            req,
            _brand: Brand::new(),
        }
    }

    #[inline]
    pub fn req(&self) -> &'p Request<K, T> {
        self.req
    }
}

#[derive(Debug, Clone)]
pub struct BrandedAssignmentRef<'brand, 'p, K: Kind, T: Copy + Ord> {
    asg: AssignmentRef<'p, 'p, K, T>,
    _brand: Brand<'brand>,
}

impl<'brand, 'p, K: Kind, T: Copy + Ord> BrandedAssignmentRef<'brand, 'p, K, T> {
    #[inline]
    fn new(asg: AssignmentRef<'p, 'p, K, T>) -> Self {
        Self {
            asg,
            _brand: Brand::new(),
        }
    }

    #[inline]
    pub fn asg(&self) -> &AssignmentRef<'p, 'p, K, T> {
        &self.asg
    }
}

#[derive(Debug, Clone)]
pub struct BrandedFreeBerth<'brand, 'p, T: Copy + Ord> {
    interval: TimeInterval<T>,
    berth: &'p Berth<T>,
    _brand: Brand<'brand>,
}

impl<'brand, 'p, T: Copy + Ord> BrandedFreeBerth<'brand, 'p, T> {
    #[inline]
    fn new(interval: TimeInterval<T>, berth: &'p Berth<T>) -> Self {
        Self {
            interval,
            berth,
            _brand: Brand::new(),
        }
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

#[derive(Debug, Clone)]
pub struct Plan<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_delta: TerminalDelta<'p, T>,
    delta_cost: Cost,
}

impl<'p, T: Copy + Ord> Plan<'p, T> {
    #[inline]
    fn new(ledger: Ledger<'p, T>, terminal_delta: TerminalDelta<'p, T>, delta_cost: Cost) -> Self {
        Self {
            ledger,
            terminal_delta,
            delta_cost,
        }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    pub fn terminal_delta(&self) -> &TerminalDelta<'p, T> {
        &self.terminal_delta
    }

    #[inline]
    pub fn into_inner(self) -> (Ledger<'p, T>, TerminalDelta<'p, T>, Cost) {
        (self.ledger, self.terminal_delta, self.delta_cost)
    }

    #[inline]
    pub fn delta_cost(&self) -> Cost {
        self.delta_cost
    }
}

#[derive(Debug, Clone)]
pub struct PlanExplorer<'brand, 'pb, 'p, T: Copy + Ord> {
    ledger: &'pb Ledger<'p, T>,
    sandbox: &'pb TerminalSandbox<'p, T>,
    _brand: Brand<'brand>,
}

impl<'brand, 'pb, 'p, T: Copy + Ord> PlanExplorer<'brand, 'pb, 'p, T> {
    #[inline]
    pub fn new(ledger: &'pb Ledger<'p, T>, sandbox: &'pb TerminalSandbox<'p, T>) -> Self {
        Self {
            ledger,
            sandbox,
            _brand: Brand::new(),
        }
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
    pub fn iter_free_for(
        &self,
        req: BrandedRequest<'brand, 'p, FlexibleKind, T>,
    ) -> impl Iterator<Item = BrandedFreeBerth<'brand, 'p, T>> + 'pb
    where
        T: CheckedAdd + CheckedSub,
    {
        let window = req.req().feasible_window();
        let allowed = req.req().iter_allowed_berths_ids();
        self.sandbox
            .inner()
            .iter_free_intervals_for_berths_in(allowed, window)
            .map(|fb| BrandedFreeBerth::new(fb.interval(), fb.berth()))
    }

    #[inline]
    pub fn iter_unassigned_requests(
        &self,
    ) -> impl Iterator<Item = BrandedRequest<'brand, 'p, FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger
            .iter_unassigned_requests()
            .map(BrandedRequest::new)
    }

    #[inline]
    pub fn iter_assigned_requests(
        &self,
    ) -> impl Iterator<Item = BrandedRequest<'brand, 'p, FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger
            .iter_assigned_requests()
            .map(BrandedRequest::new)
    }

    #[inline]
    pub fn iter_assignments(
        &self,
    ) -> impl Iterator<Item = BrandedAssignmentRef<'brand, 'p, FlexibleKind, T>> {
        self.ledger
            .iter_assignments()
            .map(|a| BrandedAssignmentRef::new(*a))
    }
}

#[derive(Debug, Clone)]
pub struct PlanBuilder<'brand, 'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    sandbox: TerminalSandbox<'p, T>,
    delta_cost: Cost,
    _brand: Brand<'brand>,
}

impl<'brand, 'p, T: Copy + Ord> PlanBuilder<'brand, 'p, T> {
    #[inline]
    pub fn new(ledger: Ledger<'p, T>, terminal: TerminalOccupancy<'p, T>) -> Self {
        Self {
            ledger,
            sandbox: TerminalSandbox::new(terminal),
            delta_cost: Cost::zero(),
            _brand: Brand::new(),
        }
    }

    #[inline]
    pub fn delta_cost(&self) -> Cost {
        self.delta_cost
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
        request: BrandedRequest<'_, 'p, FlexibleKind, T>,
        start_time: TimePoint<T>,
        free_berth: &BrandedFreeBerth<'_, 'p, T>,
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, ProposeAssignmentError<T>>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let berth = free_berth.berth();
        let free_iv = free_berth.interval();
        let asg = AssignmentRef::new(request.req(), berth, start_time)?;
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
            .commit_assignment(request.req(), berth, start_time)
            .map_err(ProposeAssignmentError::from)?;

        self.sandbox
            .occupy(berth.id(), asg_iv)
            .map_err(ProposeAssignmentError::from)?;

        self.delta_cost += asg.cost();

        Ok(asg)
    }

    #[inline]
    pub fn propose_unassignment(
        &mut self,
        asg: &BrandedAssignmentRef<'brand, 'p, FlexibleKind, T>,
    ) -> Result<BrandedFreeBerth<'brand, 'p, T>, ProposeUnassignmentError<T>>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let asg = asg.asg();
        let id = asg.berth().id();
        let iv = asg.interval();

        self.ledger
            .uncommit_assignment(asg)
            .map_err(ProposeUnassignmentError::from)?;
        self.sandbox
            .release(id, iv)
            .map_err(ProposeUnassignmentError::from)?;
        self.delta_cost -= asg.cost();

        Ok(BrandedFreeBerth::new(iv, asg.berth()))
    }

    #[inline]
    pub fn with_explorer<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&PlanExplorer<'brand, '_, 'p, T>) -> R,
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let explorer: PlanExplorer<'brand, '_, 'p, T> =
            PlanExplorer::new(&self.ledger, &self.sandbox);
        f(&explorer)
    }

    #[inline]
    pub fn finalize(self) -> Result<Plan<'p, T>, BerthIdentifierNotFoundError> {
        Ok(Plan::new(
            self.ledger,
            self.sandbox.delta()?,
            self.delta_cost,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct PlanningContext<'s, 'p, T: Copy + Ord> {
    ledger: &'s Ledger<'p, T>,
    terminal: &'s TerminalOccupancy<'p, T>,
}

impl<'s, 'p, T: Copy + Ord> PlanningContext<'s, 'p, T> {
    #[inline]
    pub fn new(ledger: &'s Ledger<'p, T>, terminal: &'s TerminalOccupancy<'p, T>) -> Self {
        Self { ledger, terminal }
    }

    #[inline]
    pub fn ledger(&self) -> &'s Ledger<'p, T> {
        self.ledger
    }

    #[inline]
    pub fn terminal(&self) -> &'s TerminalOccupancy<'p, T> {
        self.terminal
    }

    #[inline]
    pub fn with_builder<F>(&self, f: F) -> Result<Plan<'p, T>, BerthIdentifierNotFoundError>
    where
        F: for<'brand> FnOnce(&mut PlanBuilder<'brand, 'p, T>),
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        // For now we do not use specialized overlays. Just the cloned ledger and terminal,
        // so proposals can be made on them independently from the master state.
        let mut pb = PlanBuilder::new(self.ledger.clone(), self.terminal.clone());
        f(&mut pb);
        pb.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::terminalocc::TerminalWrite;
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
    fn test_plan_builder_assign_finalize_and_apply_delta() {
        // Problem: one berth id=1 free on [0,10); one flexible request with pt=4 on b1, weight=3
        let b = berth(1, 0, 10);
        let r = req_flex(100, (0, 10), &[(1, 4)], 3);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        // TerminalOccupancy is built from a Vec<Berth<_>>; keep the Vec alive for 'p.
        let base_berths = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base_berths);

        let ledger = Ledger::new(&problem);
        let mut pb = PlanBuilder::new(ledger, terminal);

        // Pull the request reference from the problem to satisfy 'p.
        let req_ref = pb
            .ledger
            .problem()
            .flexible_requests()
            .get(rid(100))
            .expect("request exists");

        // Find free slot and assign at start=2 -> [2,6)
        let (iv_copy, berth_id) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(req_ref))
                .next()
                .expect("expected some free interval");
            (*tmp.interval(), tmp.berth().id())
        });
        let berth_ref = pb.ledger.problem().berths().get(berth_id).unwrap();
        let free = BrandedFreeBerth::new(iv_copy, berth_ref);

        let asg = pb
            .propose_assignment(BrandedRequest::new(req_ref), tp(2), &free)
            .expect("assignment should succeed");

        // Cost tracking
        assert_eq!(pb.delta_cost(), asg.cost());

        // Finalize and apply the delta to a fresh terminal
        let plan = pb.finalize().expect("finalize ok");
        assert_eq!(plan.delta_cost(), asg.cost());
        assert!(!plan.terminal_delta().is_empty());

        let base_berths2 = clone_berths_from_problem(plan.ledger().problem());
        let mut apply_target = TerminalOccupancy::new(&base_berths2);

        apply_target
            .apply_delta(plan.terminal_delta().clone())
            .expect("apply delta");

        // After occupying [2,6) on berth 1, free windows are [0,2) and [6,10)
        let v: Vec<_> = apply_target
            .iter_free_intervals_for_berths_in([bid(1)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 2), iv(6, 10)]);
    }

    #[test]
    fn test_plan_builder_assign_then_unassign_roundtrip_cost_zero() {
        // Problem: one berth [0,10), one flex req with pt=4 (weight arbitrary)
        let b = berth(1, 0, 10);
        let r = req_flex(200, (0, 10), &[(1, 4)], 5);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        let base_berths = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base_berths);

        let ledger = Ledger::new(&problem);
        let mut pb = PlanBuilder::new(ledger, terminal);

        let req_ref = problem.flexible_requests().get(rid(200)).unwrap();

        // Assign at start=3 -> [3,7)
        let (iv_copy, berth_id) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(req_ref))
                .next()
                .unwrap();
            (*tmp.interval(), tmp.berth().id())
        });
        let berth_ref = pb.ledger.problem().berths().get(berth_id).unwrap();
        let free = BrandedFreeBerth::new(iv_copy, berth_ref);
        let asg = pb
            .propose_assignment(BrandedRequest::new(req_ref), tp(3), &free)
            .unwrap();

        // Then unassign; delta_cost must return to zero
        let branded_asg = BrandedAssignmentRef::new(asg);
        let freed = pb.propose_unassignment(&branded_asg).expect("unassign ok");
        assert_eq!(freed.berth().id(), bid(1));
        assert_eq!(pb.delta_cost(), Cost::zero());

        // Finalize; applying the delta on a fresh terminal should be a no-op
        let plan = pb.finalize().expect("finalize ok");

        let base_berths2 = clone_berths_from_problem(plan.ledger().problem());
        let mut target = TerminalOccupancy::new(&base_berths2);
        target
            .apply_delta(plan.terminal_delta().clone())
            .expect("apply ok");

        let v: Vec<_> = target
            .iter_free_intervals_for_berths_in([bid(1)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10)]);
    }

    #[test]
    fn test_plan_builder_propose_assignment_errors_when_not_free() {
        // Problem: berth 1 [0,10); request window [0,20), pt=8 on b1
        let b = berth(1, 0, 10);
        let r = req_flex(300, (0, 20), &[(1, 8)], 1);
        let problem = make_problem(vec![b.clone()], vec![r.clone()]);

        let base_berths = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base_berths);

        let ledger = Ledger::new(&problem);
        let mut pb = PlanBuilder::new(ledger, terminal);

        let req_ref = pb
            .ledger
            .problem()
            .flexible_requests()
            .get(rid(300))
            .unwrap();

        // Pick the (only) free interval [0,10); starting at 5 -> [5,13) which is *not* fully free
        let (iv_copy, berth_id) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(req_ref))
                .next()
                .unwrap();
            (*tmp.interval(), tmp.berth().id())
        });
        let berth_ref = pb.ledger.problem().berths().get(berth_id).unwrap();
        let free = BrandedFreeBerth::new(iv_copy, berth_ref);
        let err = pb
            .propose_assignment(BrandedRequest::new(req_ref), tp(5), &free)
            .unwrap_err();

        match err {
            ProposeAssignmentError::NotFree(e) => {
                assert_eq!(e.id(), berth_id);
                assert_eq!(e.requested(), iv(5, 13));
                assert_eq!(e.available(), *free.interval());
            }
            other => panic!("expected NotFree, got {other:?}"),
        }
    }

    #[test]
    fn test_iter_free_for_respects_allowed_berth_ids() {
        // Problem: two berths; request only allows berth 2 (pt=6)
        // b1: [0,10), b2: [5,15)
        let b1 = berth(1, 0, 10);
        let b2 = berth(2, 5, 15);
        let r = req_flex(400, (0, 50), &[(2, 6)], 1);
        let problem = make_problem(vec![b1.clone(), b2.clone()], vec![r.clone()]);

        let base_berths = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base_berths);

        let ledger = Ledger::new(&problem);
        let pb = PlanBuilder::new(ledger, terminal);

        let req_ref = pb
            .ledger
            .problem()
            .flexible_requests()
            .get(rid(400))
            .unwrap();

        // Only berth 2 should produce free intervals here.
        let v: Vec<_> = pb.with_explorer(|explorer| {
            explorer
                .iter_free_for(BrandedRequest::new(req_ref))
                .map(|fb| (fb.berth().id(), fb.interval().clone()))
                .collect()
        });

        assert!(
            v.iter().all(|(id, _)| *id == bid(2)),
            "expected only berth 2, got: {:?}",
            v
        );
        // sanity: within the request window and berth2 availability
        assert!(v.iter().any(|(_, ivv)| *ivv == iv(5, 15)));
    }

    #[test]
    fn test_iter_free_for_shows_hole_from_prior_assignment() {
        // One berth [0,30); r1 pt=10, r2 pt=5 (both allowed on b1)
        let b = berth(1, 0, 30);
        let r1 = req_flex(10, (0, 30), &[(1, 10)], 1);
        let r2 = req_flex(20, (0, 30), &[(1, 5)], 1);
        let problem = make_problem(vec![b.clone()], vec![r1.clone(), r2.clone()]);

        let base = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base);

        let ledger = Ledger::new(&problem);
        let mut pb = PlanBuilder::new(ledger, terminal);

        let r1_ref = pb
            .ledger
            .problem()
            .flexible_requests()
            .get(rid(10))
            .unwrap();
        let r2_ref = pb
            .ledger
            .problem()
            .flexible_requests()
            .get(rid(20))
            .unwrap();

        // Assign r1 at start=5 → [5,15)
        let (iv1, bid1) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(r1_ref))
                .next()
                .expect("some free for r1");
            (*tmp.interval(), tmp.berth().id())
        });
        let b1_ref = pb.ledger.problem().berths().get(bid1).unwrap();
        let free1 = BrandedFreeBerth::new(iv1, b1_ref);
        pb.propose_assignment(BrandedRequest::new(r1_ref), tp(5), &free1)
            .expect("assign r1");

        // Now r2 should see a hole: free windows [0,5) and [15,30)
        let v: Vec<_> = pb.with_explorer(|explorer| {
            explorer
                .iter_free_for(BrandedRequest::new(r2_ref))
                .map(|fb| fb.interval().clone())
                .collect()
        });
        assert_eq!(v, vec![iv(0, 5), iv(15, 30)]);
    }

    #[test]
    fn test_plan_roundtrip_apply_then_partial_unapply() {
        // One berth [0,20). r1 pt=6, r2 pt=4, r3 pt=3 (all on b1).
        let b = berth(1, 0, 20);
        let r1 = req_flex(1, (0, 20), &[(1, 6)], 1);
        let r2 = req_flex(2, (0, 20), &[(1, 4)], 1);
        let r3 = req_flex(3, (0, 20), &[(1, 3)], 1);
        let problem = make_problem(vec![b.clone()], vec![r1.clone(), r2.clone(), r3.clone()]);

        // Build terminal + planner
        let base = clone_berths_from_problem(&problem);
        let terminal = TerminalOccupancy::new(&base);
        let ledger = Ledger::new(&problem);
        let mut pb = PlanBuilder::new(ledger, terminal);

        let r1_ref = problem.flexible_requests().get(rid(1)).unwrap();
        let r2_ref = problem.flexible_requests().get(rid(2)).unwrap();
        let r3_ref = problem.flexible_requests().get(rid(3)).unwrap();

        // Assign r1 at t=2 → [2,8)
        let (iv1, bid1) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(r1_ref))
                .next()
                .expect("free for r1");
            (*tmp.interval(), tmp.berth().id())
        });
        let b1_ref = problem.berths().get(bid1).unwrap();
        let free1 = BrandedFreeBerth::new(iv1, b1_ref);
        let _a1 = pb
            .propose_assignment(BrandedRequest::new(r1_ref), tp(2), &free1)
            .expect("assign r1");

        // Assign r2 at t=10 → [10,14)
        let (iv2, bid2) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(r2_ref))
                .last()
                .expect("free for r2");
            (*tmp.interval(), tmp.berth().id())
        });
        assert_eq!(bid1, bid2);
        let b2_ref = problem.berths().get(bid2).unwrap();
        let free2 = BrandedFreeBerth::new(iv2, b2_ref);
        let _a2 = pb
            .propose_assignment(BrandedRequest::new(r2_ref), tp(10), &free2)
            .expect("assign r2");

        // Finalize plan and apply delta to a fresh terminal
        let plan1 = pb.finalize().expect("finalize ok");
        let base2 = clone_berths_from_problem(plan1.ledger().problem());
        let mut term_applied = TerminalOccupancy::new(&base2);
        term_applied
            .apply_delta(plan1.terminal_delta().clone())
            .expect("apply delta");

        // With [2,8) and [10,14) occupied, r3 should see: [0,2), [8,10), [14,20)
        let pb2_ledger = plan1.ledger().clone();
        let mut pb2 = PlanBuilder::new(pb2_ledger, term_applied);
        let v_before_unassign: Vec<_> = pb2.with_explorer(|explorer| {
            explorer
                .iter_free_for(BrandedRequest::new(r3_ref))
                .map(|fb| fb.interval().clone())
                .collect()
        });
        assert_eq!(v_before_unassign, vec![iv(0, 2), iv(8, 10), iv(14, 20)]);

        // === Borrow-safe unassignment of r2 ===
        // 1) Read scalars while immutably borrowing pb2.
        let (rid2, bid2, start2) = pb2.with_explorer(|explorer| {
            let a2_ref = explorer
                .iter_assignments()
                .find(|a| a.asg().request_id() == rid(2))
                .expect("r2 assignment present");
            (
                a2_ref.asg().request_id(),
                a2_ref.asg().berth_id(),
                a2_ref.asg().start_time(),
            )
        }); // immutable borrow ends here

        // 2) Rebuild an independent AssignmentRef from the *problem* (not pb2).
        let req2_ref = problem.flexible_requests().get(rid2).unwrap();
        let berth2_ref = problem.berths().get(bid2).unwrap();
        let a2_again =
            AssignmentRef::<FlexibleKind, i64>::new(req2_ref, berth2_ref, start2).unwrap();

        // 3) Wrap in a branded ref and then mutably borrow pb2 to unassign.
        let branded_a2_again = BrandedAssignmentRef::new(a2_again);
        pb2.propose_unassignment(&branded_a2_again)
            .expect("unassign r2");

        // Finalize the unassignment plan and apply its delta
        let plan2 = pb2.finalize().expect("finalize ok");

        // Bind the berths vec to extend its lifetime (fixes E0716).
        let base_after = clone_berths_from_problem(plan2.ledger().problem());
        let mut term_after = TerminalOccupancy::new(&base_after);
        term_after
            .apply_delta(plan1.terminal_delta().clone())
            .expect("reapply plan1 delta");
        term_after
            .apply_delta(plan2.terminal_delta().clone())
            .expect("apply plan2 delta");

        // Now only r1 remains → occupied [2,8), so free: [0,2) and [8,20)
        let v_after: Vec<_> = term_after
            .iter_free_intervals_for_berths_in([bid(1)], iv(0, 20))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v_after, vec![iv(0, 2), iv(8, 20)]);
    }

    #[test]
    fn test_reassign_into_freed_hole() {
        // Minimal: one berth [0,10], r1 pt=6, r2 pt=4.
        let b = berth(1, 0, 10);
        let r1 = req_flex(1, (0, 10), &[(1, 6)], 1);
        let r2 = req_flex(2, (0, 10), &[(1, 2)], 1);
        let problem = make_problem(vec![b.clone()], vec![r1.clone(), r2.clone()]);

        // Start with r1 at [2,8) applied.
        let base = clone_berths_from_problem(&problem);
        let mut term = TerminalOccupancy::new(&base);
        term.occupy(bid(1), iv(2, 8)).unwrap();

        let ledger = Ledger::new(&problem);
        let mut pb = PlanBuilder::new(ledger, term);

        // Insert r2 into a new problem instance for reference access (or keep r2 locally).
        // Here we simply use `r2` directly as the source of truth:
        let r2_ref = problem.flexible_requests().get(rid(2)).unwrap();

        // Expect free windows [0,2) and [8,10); can fit r2 at start=8.
        let (iv_free, bid_) = pb.with_explorer(|explorer| {
            let tmp = explorer
                .iter_free_for(BrandedRequest::new(r2_ref))
                .last()
                .unwrap();
            (*tmp.interval(), tmp.berth().id())
        });
        assert_eq!(iv_free, iv(8, 10));
        let b_ref = problem.berths().get(bid_).unwrap();
        let free = BrandedFreeBerth::new(iv_free, b_ref);
        let _a = pb
            .propose_assignment(BrandedRequest::new(r2_ref), tp(8), &free)
            .unwrap();

        // finalization must succeed
        let plan = pb.finalize().unwrap();
        assert!(!plan.terminal_delta().is_empty());
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    macro_rules! test_integer_types {
        ($($t:ty),*) => {
            $(
                assert_impl_all!(Plan<'static, $t>: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
