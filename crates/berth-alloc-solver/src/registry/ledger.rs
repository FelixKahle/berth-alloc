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

use crate::registry::err::{LedgerComitError, LedgerUncomitError};
use berth_alloc_core::prelude::TimePoint;
use berth_alloc_model::{
    common::{FixedKind, FlexibleKind},
    prelude::{Assignment, AssignmentContainer, Berth, Problem, Request, RequestContainer},
    problem::asg::{AssignmentRef, AssignmentView},
};
use num_traits::{CheckedAdd, CheckedSub};

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
    pub fn flexible_assignments(&self) -> &RequestContainer<FlexibleKind, T>
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
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, LedgerComitError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let assignment = AssignmentRef::new(request, berth, start_time)?;
        if self.commited.contains_assignment(&assignment) {
            Err(LedgerComitError::AlreadyCommitted(request.id()))
        } else {
            self.commited.insert(assignment);
            Ok(assignment)
        }
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
    pub fn iter_assigned_requests(&self) -> impl Iterator<Item = &Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.commited.iter().map(|a| a.request())
    }

    #[inline]
    pub fn iter_assignments(
        &self,
    ) -> impl Iterator<Item = &AssignmentRef<'p, 'p, FlexibleKind, T>> {
        self.commited.iter()
    }

    #[inline]
    pub fn apply(&mut self, other: Self) {
        *self = other;
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(ledger.iter_assignments().count(), 1);

        // now uncommit
        let removed = ledger
            .uncommit_assignment(&a)
            .expect("uncommit should succeed");
        assert_eq!(removed.request_id(), rid(1));
        assert!(ledger.commited_assignments().is_empty());
        assert_eq!(ledger.iter_assignments().count(), 0);
    }

    #[test]
    fn test_double_commit_returns_already_committed() {
        let prob = problem_one_berth_two_flex();
        let mut ledger = Ledger::new(&prob);

        let req = ledger.problem().flexible_requests().get(rid(1)).unwrap();
        let b = ledger.problem().berths().get(bid(1)).unwrap();

        ledger.commit_assignment(req, b, tp(10)).unwrap();

        let err = ledger.commit_assignment(req, b, tp(20)).unwrap_err();
        match err {
            LedgerComitError::AlreadyCommitted(id) => assert_eq!(id, rid(1)),
            other => panic!("expected AlreadyCommitted, got {other:?}"),
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
        assert_eq!(ledger1.iter_assignments().count(), 1);
        assert_eq!(ledger2.iter_assignments().count(), 1);
        ledger1.apply(ledger2);
        let assigned_ids: Vec<_> = ledger1.iter_assigned_requests().map(|r| r.id()).collect();
        assert_eq!(assigned_ids, vec![rid(2)]);
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
