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

use crate::registry::err::{LedgerCommitError, LedgerUncomitError};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use berth_alloc_model::{
    common::{FixedKind, FlexibleKind},
    prelude::{
        Assignment, AssignmentContainer, Berth, BerthIdentifier, Problem, Request,
        RequestContainer, StateValidator,
    },
    problem::asg::{AnyAssignmentRef, AssignmentRef, AssignmentView},
};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct Ledger<'p, T: Copy + Ord> {
    problem: &'p Problem<T>,
    commited: AssignmentContainer<FlexibleKind, T, AssignmentRef<'p, 'p, FlexibleKind, T>>,
}

impl<'p, T: Copy + Ord> Ledger<'p, T> {
    #[inline]
    pub fn new(problem: &'p Problem<T>) -> Self {
        Self {
            problem,
            commited: AssignmentContainer::new(),
        }
    }

    #[inline]
    pub fn problem(&self) -> &'p Problem<T> {
        self.problem
    }

    #[inline]
    pub fn commited_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, AssignmentRef<'p, 'p, FlexibleKind, T>> {
        &self.commited
    }

    #[inline]
    pub fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem.fixed_assignments()
    }

    #[inline]
    pub fn flexible_requests(&self) -> &RequestContainer<FlexibleKind, T>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem.flexible_requests()
    }

    #[inline]
    pub fn commit_assignment(
        &mut self,
        request: &'p Request<FlexibleKind, T>,
        berth: &'p Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, LedgerCommitError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let assignment = AssignmentRef::new(request, berth, start_time)?;
        StateValidator::validate_nooverlap_with(
            self.problem().fixed_assignments(),
            self.commited_assignments(),
            self.problem(),
            &assignment,
        )?;
        StateValidator::validate_no_extra_flexible_assignments_with(
            self.commited_assignments(),
            &assignment,
        )?;
        StateValidator::validate_request_ids_unique_with(
            self.problem().fixed_assignments(),
            self.commited_assignments(),
            &assignment,
        )?;
        StateValidator::validate_no_extra_flexible_requests_with(
            self.commited_assignments(),
            self.problem(),
            &assignment,
        )?;

        self.commited.insert(assignment);
        Ok(assignment)
    }

    #[inline]
    pub fn uncommit_assignment(
        &mut self,
        assignment: &AssignmentRef<'p, 'p, FlexibleKind, T>,
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, LedgerUncomitError>
    where
        T: CheckedAdd + CheckedSub,
    {
        if let Some(a) = self.commited.remove_assignment(assignment) {
            Ok(a)
        } else {
            Err(LedgerUncomitError::new(assignment.request_id()))
        }
    }

    #[inline]
    pub fn iter_unassigned_requests(&self) -> impl Iterator<Item = &'p Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem
            .flexible_requests()
            .iter()
            .filter(move |r| !self.commited.contains_id(r.id()))
    }

    #[inline]
    pub fn iter_assigned_requests(&self) -> impl Iterator<Item = &'p Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem()
            .flexible_requests()
            .iter()
            .filter(move |r| self.commited.contains_id(r.id()))
    }

    #[inline]
    pub fn iter_flexible_assignments(
        &self,
    ) -> impl Iterator<Item = &AssignmentRef<'p, 'p, FlexibleKind, T>> {
        self.commited.iter()
    }

    #[inline]
    pub fn apply(&mut self, other: Self) {
        *self = other;
    }

    #[inline]
    pub fn into_inner(
        self,
    ) -> AssignmentContainer<FlexibleKind, T, AssignmentRef<'p, 'p, FlexibleKind, T>> {
        self.commited
    }

    #[inline]
    pub fn cost(&self) -> Cost
    where
        T: CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    {
        let fixed_cost: Cost = self.fixed_assignments().iter().map(|a| a.cost()).sum();
        let flexible_cost: Cost = self.commited_assignments().iter().map(|a| a.cost()).sum();
        fixed_cost + flexible_cost
    }

    #[inline]
    pub fn committed_in_window(
        &self,
        win: TimeInterval<T>,
    ) -> impl Iterator<Item = &AssignmentRef<'p, 'p, FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.commited
            .iter()
            .filter(move |a| a.interval().intersects(&win))
    }

    #[inline]
    pub fn committed_count_in_window(&self, win: TimeInterval<T>) -> usize
    where
        T: CheckedAdd + CheckedSub,
    {
        self.committed_in_window(win).count()
    }

    #[inline]
    pub fn fixed_in_window(
        &self,
        win: TimeInterval<T>,
    ) -> impl Iterator<Item = &Assignment<FixedKind, T>> + '_
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem
            .fixed_assignments()
            .iter()
            .filter(move |a| a.interval().intersects(&win))
    }

    #[inline]
    pub fn fixed_count_in_window(&self, win: TimeInterval<T>) -> usize
    where
        T: CheckedAdd + CheckedSub,
    {
        self.fixed_in_window(win).count()
    }

    #[inline]
    pub fn total_count_in_window(&self, win: TimeInterval<T>) -> usize
    where
        T: CheckedAdd + CheckedSub,
    {
        self.fixed_count_in_window(win) + self.committed_count_in_window(win)
    }

    #[inline]
    pub fn fixed_assignment_ratio_in_window(&self, win: TimeInterval<T>) -> f64
    where
        T: CheckedAdd + CheckedSub,
    {
        let fixed = self.fixed_count_in_window(win) as f64;
        let total = self.total_count_in_window(win) as f64;
        if total == 0.0 { 0.0 } else { fixed / total }
    }

    #[inline]
    pub fn flexible_assignment_ratio_in_window(&self, win: TimeInterval<T>) -> f64
    where
        T: CheckedAdd + CheckedSub,
    {
        let flex = self.committed_count_in_window(win) as f64;
        let total = self.total_count_in_window(win) as f64;
        if total == 0.0 { 0.0 } else { flex / total }
    }

    #[inline]
    pub fn total_in_window(
        &self,
        win: TimeInterval<T>,
    ) -> impl Iterator<Item = AnyAssignmentRef<'_, '_, T>> + '_
    where
        T: CheckedAdd + CheckedSub,
    {
        let fixed_iter = self
            .fixed_in_window(win)
            .map(|a| AnyAssignmentRef::from(a.to_ref()));
        let flex_iter = self
            .committed_in_window(win)
            .copied()
            .map(AnyAssignmentRef::from);
        fixed_iter.chain(flex_iter)
    }

    pub fn envelope_of_assignments_in_window(&self, win: TimeInterval<T>) -> Option<TimeInterval<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let mut min_s: Option<TimePoint<T>> = None;
        let mut max_e: Option<TimePoint<T>> = None;

        for a in self.total_in_window(win) {
            let (s, e) = match a {
                AnyAssignmentRef::Fixed(fr) => (fr.start_time(), fr.end_time()),
                AnyAssignmentRef::Flexible(fr) => (fr.start_time(), fr.end_time()),
            };

            min_s = Some(match min_s {
                Some(cur) => cur.min(s),
                None => s,
            });
            max_e = Some(match max_e {
                Some(cur) => cur.max(e),
                None => e,
            });
        }

        Some(TimeInterval::new(min_s?, max_e?))
    }

    #[inline]
    pub fn request_pressure_on_berth(&self, bid: BerthIdentifier) -> usize
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem
            .iter_any_requests()
            .filter(|r| r.processing_time_for(bid).is_some())
            .count()
    }

    #[inline]
    pub fn work_pressure_on_berth(&self, bid: BerthIdentifier) -> TimeDelta<T>
    where
        T: CheckedAdd + CheckedSub + num_traits::Zero,
    {
        self.problem
            .iter_any_requests()
            .filter_map(|r| r.processing_time_for(bid))
            .sum()
    }

    #[inline]
    pub fn flexible_pressure_on_berth(&self, bid: BerthIdentifier) -> usize
    where
        T: CheckedAdd + CheckedSub,
    {
        self.problem
            .iter_flexible_requests()
            .filter(|r| r.processing_time_for(bid).is_some())
            .count()
    }

    #[inline]
    pub fn flexible_work_pressure_on_berth(&self, bid: BerthIdentifier) -> TimeDelta<T>
    where
        T: CheckedAdd + CheckedSub + num_traits::Zero,
    {
        self.problem
            .iter_flexible_requests()
            .filter_map(|r| r.processing_time_for(bid))
            .sum()
    }
}

impl<'p, T> std::fmt::Display for Ledger<'p, T>
where
    T: std::fmt::Display + Copy + Ord + CheckedAdd + CheckedSub,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ledger(problem with {} berths, {} fixed assignments, {} flexible requests; {} committed assignments)",
            self.problem().berths().len(),
            self.problem().fixed_assignments().len(),
            self.problem().flexible_requests().len(),
            self.commited.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::RequestIdentifier;
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

    fn fixed_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    // Build a tiny problem with given berths / fixed / flexible sets.
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

    fn problem_with_fixed_and_two_flex() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = berth(1, 0, 1000);
        berths.insert(b1.clone());

        // Fixed: rf(100) on b1, pt=10, start=50 -> [50,60)
        let rf = fixed_req(100, (0, 1000), &[(1, 10)], 1);
        let af = Assignment::<FixedKind, i64>::new_fixed(rf, b1.clone(), tp(50))
            .expect("fixed assignment create");

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        // Flex: r1(1) pt=20, r2(2) pt=15 on b1
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        flex.insert(flex_req(1, (0, 500), &[(1, 20)], 1)); // will commit at 5 -> [5,25)
        flex.insert(flex_req(2, (0, 500), &[(1, 15)], 1)); // will commit at 80 -> [80,95)

        Problem::new(berths, fixed, flex).unwrap()
    }

    #[test]
    fn new_ledger_is_empty_and_points_to_problem() {
        let prob = problem_one_berth_two_flex();
        let ledger = Ledger::new(&prob);

        assert_eq!(ledger.problem().flexible_requests().len(), 2);
        assert!(ledger.commited_assignments().is_empty());
    }

    #[test]
    fn test_commit_and_uncommit_roundtrip() {
        let prob = problem_one_berth_two_flex();
        let mut ledger = Ledger::new(&prob);

        // pull the request & berth from the problem containers
        let req = ledger
            .problem()
            .flexible_requests()
            .get(rid(1))
            .expect("flexible request exists");
        let b = ledger.problem().berths().get(bid(1)).expect("berth exists");

        // commit
        let a = ledger
            .commit_assignment(req, b, tp(0))
            .expect("commit should succeed");

        // visible in committed
        assert!(ledger.commited_assignments().contains_id(rid(1)));
        assert_eq!(ledger.iter_flexible_assignments().count(), 1);

        // now uncommit
        let removed = ledger
            .uncommit_assignment(&a)
            .expect("uncommit should succeed");
        assert_eq!(removed.request_id(), rid(1));
        assert!(ledger.commited_assignments().is_empty());
        assert_eq!(ledger.iter_flexible_assignments().count(), 0);
    }

    #[test]
    fn test_double_commit_returns_duplicate_flexible_error() {
        let prob = problem_one_berth_two_flex();
        let mut ledger = Ledger::new(&prob);

        let req = ledger.problem().flexible_requests().get(rid(1)).unwrap();
        let b = ledger.problem().berths().get(bid(1)).unwrap();

        ledger.commit_assignment(req, b, tp(10)).unwrap();

        let err = ledger.commit_assignment(req, b, tp(20)).unwrap_err();
        match err {
            LedgerCommitError::ExtraFlexibleAssignment(e) => {
                // ensure the validator surfaces the right rid
                assert_eq!(e.request_id(), rid(1));
            }
            other => panic!("expected ExtraFlexible, got {other:?}"),
        }
    }

    #[test]
    fn test_commit_invalid_assignment_bubbles_error() {
        // request window [10, 100), start at 0 should fail via underlying AssignmentRef::new
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        flex.insert(flex_req(9, (10, 100), &[(1, 5)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let mut ledger = Ledger::new(&prob);

        let req = ledger.problem().flexible_requests().get(rid(9)).unwrap();
        let b = ledger.problem().berths().get(bid(1)).unwrap();

        let res = ledger.commit_assignment(req, b, tp(0));
        assert!(res.is_err(), "expected commit to error for invalid timing");
        // Optionally assert variant:
        if let Err(LedgerCommitError::Assignment(_)) = res {
            // ok
        } else {
            panic!("expected InvalidAssignment variant");
        }
    }

    #[test]
    fn test_iterators_assigned_and_unassigned_work() {
        let prob = problem_one_berth_two_flex();
        let mut ledger = Ledger::new(&prob);

        // initially: two unassigned, none assigned
        let unassigned_ids: Vec<_> = ledger.iter_unassigned_requests().map(|r| r.id()).collect();
        assert_eq!(unassigned_ids.len(), 2);
        assert!(ledger.iter_assigned_requests().next().is_none());

        // commit r1
        let req1 = ledger.problem().flexible_requests().get(rid(1)).unwrap();
        let b = ledger.problem().berths().get(bid(1)).unwrap();
        ledger.commit_assignment(req1, b, tp(0)).unwrap();

        // now: r1 assigned, only r2 unassigned
        let unassigned_after: Vec<_> = ledger.iter_unassigned_requests().map(|r| r.id()).collect();
        assert_eq!(unassigned_after, vec![rid(2)]);

        let assigned_ids: Vec<_> = ledger.iter_assigned_requests().map(|r| r.id()).collect();
        assert_eq!(assigned_ids, vec![rid(1)]);
    }

    #[test]
    fn test_fixed_assignments_passthrough_is_correct() {
        // Build a problem with one fixed assignment so we can ensure the passthrough works.
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = berth(1, 0, 1000);
        berths.insert(b1.clone());

        // fixed req on b1, pt=10, start=0
        let rf = fixed_req(100, (0, 100), &[(1, 10)], 1);
        let af = Assignment::<FixedKind, i64>::new_fixed(rf.clone(), b1.clone(), tp(0)).unwrap();

        let mut fixed_cont =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed_cont.insert(af);

        // one flexible (unused in this test)
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        flex.insert(flex_req(200, (0, 100), &[(1, 5)], 1));

        let prob = Problem::new(berths, fixed_cont, flex).unwrap();
        let ledger = Ledger::new(&prob);

        assert_eq!(ledger.fixed_assignments().len(), 1);
        // sanity: the id present is 100
        let ids: Vec<_> = ledger
            .fixed_assignments()
            .iter()
            .map(|a| a.request_id())
            .collect();
        assert_eq!(ids, vec![rid(100)]);
    }

    #[test]
    fn test_apply_replaces_self() {
        let prob = problem_one_berth_two_flex();
        let mut ledger1 = Ledger::new(&prob);
        let mut ledger2 = Ledger::new(&prob);
        let req1 = ledger1.problem().flexible_requests().get(rid(1)).unwrap();
        let req2 = ledger2.problem().flexible_requests().get(rid(2)).unwrap();
        let b = ledger1.problem().berths().get(bid(1)).unwrap();
        ledger1.commit_assignment(req1, b, tp(0)).unwrap();
        ledger2.commit_assignment(req2, b, tp(10)).unwrap();
        assert_eq!(ledger1.iter_flexible_assignments().count(), 1);
        assert_eq!(ledger2.iter_flexible_assignments().count(), 1);
        ledger1.apply(ledger2);
        let assigned_ids: Vec<_> = ledger1.iter_assigned_requests().map(|r| r.id()).collect();
        assert_eq!(assigned_ids, vec![rid(2)]);
    }

    #[test]
    fn test_cost_accumulates_and_updates_on_commit_uncommit() {
        // Problem with one berth and two flexible requests, distinct processing times / weights
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 500));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        // r1: pt=10, weight=2
        flex.insert(flex_req(10, (0, 300), &[(1, 10)], 2));
        // r2: pt=15, weight=3
        flex.insert(flex_req(20, (0, 300), &[(1, 15)], 3));

        let prob = Problem::new(berths, fixed, flex).expect("problem build");
        let mut ledger = Ledger::new(&prob);

        let r1 = prob.flexible_requests().get(rid(10)).unwrap();
        let r2 = prob.flexible_requests().get(rid(20)).unwrap();
        let b1 = prob.berths().get(bid(1)).unwrap();

        assert_eq!(ledger.cost(), Cost::zero(), "initial cost should be zero");

        let a1 = ledger.commit_assignment(r1, b1, tp(0)).expect("commit r1");
        let cost_after_r1 = ledger.cost();

        assert!(
            cost_after_r1 > Cost::zero(),
            "cost should increase after first commit"
        );

        let a2 = ledger
            .commit_assignment(r2, b1, tp(100))
            .expect("commit r2");
        let cost_after_r2 = ledger.cost();

        assert!(
            cost_after_r2 > cost_after_r1,
            "cost should increase after second commit"
        );

        // Derive expected sum from assignments directly to be robust against formula changes
        let expected_sum = a1.cost() + a2.cost();
        assert_eq!(
            cost_after_r2, expected_sum,
            "ledger cost equals sum of assignment costs"
        );

        // Uncommit second assignment: cost should revert to the first assignment's cost
        ledger.uncommit_assignment(&a2).expect("uncommit r2");
        let cost_after_uncommit = ledger.cost();
        assert_eq!(
            cost_after_uncommit,
            a1.cost(),
            "cost after uncommitting second assignment should match r1 cost"
        );

        // Uncommit first assignment: cost returns to zero
        ledger.uncommit_assignment(&a1).expect("uncommit r1");
        assert_eq!(
            ledger.cost(),
            Cost::zero(),
            "cost should be zero after all uncommitted"
        );
    }

    #[test]
    fn test_committed_fixed_total_and_union_over_windows() {
        let prob = problem_with_fixed_and_two_flex();
        let mut ledger = Ledger::new(&prob);

        let r1 = ledger.problem().flexible_requests().get(rid(1)).unwrap();
        let r2 = ledger.problem().flexible_requests().get(rid(2)).unwrap();
        let b1 = ledger.problem().berths().get(bid(1)).unwrap();

        // Commit two flexible assignments: [5,25) and [80,95)
        ledger.commit_assignment(r1, b1, tp(5)).unwrap();
        ledger.commit_assignment(r2, b1, tp(80)).unwrap();

        // committed_in_window([0,30)) -> r1 only
        let ids_committed_0_30: Vec<_> = ledger
            .committed_in_window(iv(0, 30))
            .map(|a| a.request_id())
            .collect();
        assert_eq!(ids_committed_0_30, vec![rid(1)]);

        // fixed_in_window([55,200)) -> fixed [50,60) intersects; [60,80) does not (touching)
        let ids_fixed_55_200: Vec<_> = ledger
            .fixed_in_window(iv(55, 200))
            .map(|a| a.request_id())
            .collect();
        assert_eq!(ids_fixed_55_200, vec![rid(100)]);

        let ids_fixed_60_80: Vec<_> = ledger
            .fixed_in_window(iv(60, 80))
            .map(|a| a.request_id())
            .collect();
        assert!(
            ids_fixed_60_80.is_empty(),
            "touching at endpoint is not intersecting"
        );

        // total_in_window([0,70)) -> r1 + fixed
        use berth_alloc_model::problem::asg::AnyAssignmentRef as ARef;
        let ids_total_0_70: std::collections::BTreeSet<_> = ledger
            .total_in_window(iv(0, 70))
            .map(|a| match a {
                ARef::Fixed(fr) => fr.request_id(),
                ARef::Flexible(fr) => fr.request_id(),
            })
            .collect();
        assert!(ids_total_0_70.contains(&rid(1)));
        assert!(ids_total_0_70.contains(&rid(100)));
        assert_eq!(ids_total_0_70.len(), 2);

        // total_in_window([70,100)) -> r2 only
        let ids_total_70_100: Vec<_> = ledger
            .total_in_window(iv(70, 100))
            .map(|a| match a {
                ARef::Fixed(fr) => fr.request_id(),
                ARef::Flexible(fr) => fr.request_id(),
            })
            .collect();
        assert_eq!(ids_total_70_100, vec![rid(2)]);

        // union_of_assignments_in_window over big window -> [5,95)
        let u_all = ledger.envelope_of_assignments_in_window(iv(0, 1000));
        assert_eq!(u_all, Some(iv(5, 95)));

        // union over [60,80) (touching only) -> None
        let u_touch = ledger.envelope_of_assignments_in_window(iv(60, 80));
        assert!(u_touch.is_none());
    }

    #[test]
    fn test_request_and_work_pressure_on_berth() {
        let prob = problem_with_fixed_and_two_flex();
        let ledger = Ledger::new(&prob);

        // berth 1 has two flexible requests that can be assigned to it (r1, r2) + one fixed (rf)
        assert_eq!(ledger.request_pressure_on_berth(bid(1)), 3);

        let total = ledger.work_pressure_on_berth(bid(1));
        assert_eq!(total.value(), 45);
        assert!(total > TimeDelta::zero());
    }

    #[test]
    fn test_into_inner_returns_committed_container() {
        let prob = problem_one_berth_two_flex();
        let mut ledger = Ledger::new(&prob);

        let r1 = ledger.problem().flexible_requests().get(rid(1)).unwrap();
        let b1 = ledger.problem().berths().get(bid(1)).unwrap();
        ledger.commit_assignment(r1, b1, tp(0)).unwrap();

        let inner = ledger.into_inner();
        assert_eq!(inner.len(), 1);
        assert!(inner.contains_id(rid(1)));
    }

    #[test]
    fn test_display_shows_counts() {
        let prob = problem_one_berth_two_flex();
        let ledger = Ledger::new(&prob);
        let s = format!("{ledger}");
        // Expect mentions of counts; exact string is known from Display impl
        assert!(s.contains("problem with"));
        assert!(s.contains("berths"));
        assert!(s.contains("fixed assignments"));
        assert!(s.contains("flexible requests"));
        assert!(s.contains("committed assignments"));
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    macro_rules! test_integer_types {
        ($($t:ty),*) => {
            $(
                assert_impl_all!(Ledger<'static, $t>: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
