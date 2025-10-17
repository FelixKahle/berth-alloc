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
    common::{FixedKind, FlexibleKind},
    problem::{
        asg::{Assignment, AssignmentContainer, AssignmentRef, AssignmentView},
        prob::Problem,
        req::RequestIdentifier,
    },
    solution::SolutionError,
    validation,
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

pub trait SolutionView<T>
where
    T: Copy + Ord,
{
    type FixedAssignmentView: AssignmentView<FixedKind, T>;
    type FlexibleAssignmentView: AssignmentView<FlexibleKind, T>;

    fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Self::FixedAssignmentView>;
    fn flexible_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, Self::FlexibleAssignmentView>;

    fn fixed_assignments_len(&self) -> usize {
        self.fixed_assignments().len()
    }
    fn flexible_assignments_len(&self) -> usize {
        self.flexible_assignments().len()
    }
    fn total_assignments_len(&self) -> usize {
        self.fixed_assignments_len() + self.flexible_assignments_len()
    }

    fn is_empty(&self) -> bool {
        self.fixed_assignments().is_empty() && self.flexible_assignments().is_empty()
    }

    fn contains_fixed(&self, rid: RequestIdentifier) -> bool {
        self.fixed_assignments().get(rid).is_some()
    }
    fn contains_flexible(&self, rid: RequestIdentifier) -> bool {
        self.flexible_assignments().get(rid).is_some()
    }

    fn cost(&self) -> Cost
    where
        T: Mul<Output = Cost> + CheckedAdd + CheckedSub + Into<Cost>,
    {
        let fixed_cost: Cost = self.fixed_assignments().iter().map(|a| a.cost()).sum();
        let flex_cost: Cost = self.flexible_assignments().iter().map(|a| a.cost()).sum();
        fixed_cost + flex_cost
    }
}

#[derive(Debug, Clone)]
pub struct Solution<T: Copy + Ord> {
    fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
    flexible_assignments: AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>,
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub> Solution<T> {
    #[inline]
    pub fn new(
        fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
        flexible_assignments: AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>,
        problem: &Problem<T>,
    ) -> Result<Self, SolutionError> {
        validation::validate_no_extra_fixed_assignments(&fixed_assignments)?;
        validation::validate_no_extra_flexible_assignments(&flexible_assignments)?;
        validation::validate_request_ids_unique(&fixed_assignments, &flexible_assignments)?;
        validation::validate_all_fixed_assignments_present(&fixed_assignments, problem)?;
        validation::validate_all_flexible_assignments_present(&flexible_assignments, problem)?;
        validation::validate_no_extra_fixed_requests(&fixed_assignments, problem)?;
        validation::validate_no_extra_flexible_requests(&flexible_assignments, problem)?;

        Ok(Self {
            fixed_assignments,
            flexible_assignments,
        })
    }

    #[inline]
    pub fn as_ref(&self) -> SolutionRef<'_, T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        let fixed = self
            .fixed_assignments
            .iter()
            .map(|a| a.to_ref())
            .collect::<AssignmentContainer<FixedKind, T, AssignmentRef<'_, '_, FixedKind, T>>>();

        let flex = self
                    .flexible_assignments
                    .iter()
                    .map(|a| a.to_ref())
                    .collect::<AssignmentContainer<
                        FlexibleKind,
                        T,
                        AssignmentRef<'_, '_, FlexibleKind, T>,
                    >>();

        // Skip validation here; it was done when creating the original Solution
        SolutionRef {
            fixed_assignments: fixed,
            flexible_assignments: flex,
        }
    }
}

impl<T: Copy + Ord> SolutionView<T> for Solution<T> {
    type FixedAssignmentView = Assignment<FixedKind, T>;
    type FlexibleAssignmentView = Assignment<FlexibleKind, T>;

    fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>> {
        &self.fixed_assignments
    }

    fn flexible_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>> {
        &self.flexible_assignments
    }
}

#[derive(Debug, Clone)]
pub struct SolutionRef<'p, T: Copy + Ord> {
    fixed_assignments: AssignmentContainer<FixedKind, T, AssignmentRef<'p, 'p, FixedKind, T>>,
    flexible_assignments:
        AssignmentContainer<FlexibleKind, T, AssignmentRef<'p, 'p, FlexibleKind, T>>,
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub> SolutionRef<'p, T> {
    #[inline]
    pub fn new(
        fixed_assignments: AssignmentContainer<FixedKind, T, AssignmentRef<'p, 'p, FixedKind, T>>,
        flexible_assignments: AssignmentContainer<
            FlexibleKind,
            T,
            AssignmentRef<'p, 'p, FlexibleKind, T>,
        >,
        problem: &'p Problem<T>,
    ) -> Result<Self, SolutionError> {
        validation::validate_no_extra_fixed_assignments(&fixed_assignments)?;
        validation::validate_no_extra_flexible_assignments(&flexible_assignments)?;
        validation::validate_request_ids_unique(&fixed_assignments, &flexible_assignments)?;
        validation::validate_all_fixed_assignments_present(&fixed_assignments, problem)?;
        validation::validate_all_flexible_assignments_present(&flexible_assignments, problem)?;
        validation::validate_no_extra_fixed_requests(&fixed_assignments, problem)?;
        validation::validate_no_extra_flexible_requests(&flexible_assignments, problem)?;

        Ok(Self {
            fixed_assignments,
            flexible_assignments,
        })
    }

    #[inline]
    pub fn to_owned(&self) -> Solution<T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        let fixed = self
            .fixed_assignments
            .iter()
            .map(|a| a.to_owned())
            .collect::<AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>>();

        let flex = self
            .flexible_assignments
            .iter()
            .map(|a| a.to_owned())
            .collect::<AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>>();

        // Skip validation here; it was done when creating the original SolutionRef
        Solution {
            fixed_assignments: fixed,
            flexible_assignments: flex,
        }
    }

    #[inline]
    pub fn into_owned(self) -> Solution<T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        self.to_owned()
    }
}

impl<'p, T: Copy + Ord> SolutionView<T> for SolutionRef<'p, T> {
    type FixedAssignmentView = AssignmentRef<'p, 'p, FixedKind, T>;
    type FlexibleAssignmentView = AssignmentRef<'p, 'p, FlexibleKind, T>;

    fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Self::FixedAssignmentView> {
        &self.fixed_assignments
    }

    fn flexible_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, Self::FlexibleAssignmentView> {
        &self.flexible_assignments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::{FixedKind, FlexibleKind},
        problem::{
            asg::{Assignment, AssignmentContainer},
            berth::{Berth, BerthContainer, BerthIdentifier},
            prob::Problem,
            req::{Request, RequestContainer, RequestIdentifier},
        },
        validation::err::CrossValidationError,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use std::collections::BTreeMap;

    // ---------- small helpers ----------
    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn mk_berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn req_fixed(id: u32, window: (i64, i64), pts: &[(u32, i64)]) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn req_flex(id: u32, window: (i64, i64), pts: &[(u32, i64)]) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), 1, m).unwrap()
    }

    fn asg_fixed(
        r: &Request<FixedKind, i64>,
        b: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new_fixed(r.clone(), b.clone(), tp(start)).unwrap()
    }

    fn asg_flex(
        r: &Request<FlexibleKind, i64>,
        b: &Berth<i64>,
        start: i64,
    ) -> Assignment<FlexibleKind, i64> {
        Assignment::<FlexibleKind, i64>::new_flexible(r.clone(), b.clone(), tp(start)).unwrap()
    }

    // ---------- Solution::new() constructor validations ----------

    #[test]
    fn solution_new_ok_basic() {
        let b1 = mk_berth(1, 0, 200);
        let b2 = mk_berth(2, 0, 200);

        // fixed: id 11 on b1, len=10, in [0,100]
        let rf = req_fixed(11, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        // flexible: id 21 on b2, len=10, in [0,100]
        let rx = req_flex(21, (0, 100), &[(2, 10)]);
        let ax = asg_flex(&rx, &b2, 0);

        // Problem definition
        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());
        berths.insert(b2.clone());

        let mut prob_fixed =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        prob_fixed.insert(af.clone());

        let mut prob_flex: RequestContainer<i64, Request<FlexibleKind, i64>> =
            RequestContainer::new();
        prob_flex.insert(rx.clone());

        let prob = Problem::new(berths, prob_fixed, prob_flex).unwrap();

        // Candidate solution
        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(af);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(ax);

        // Should construct fine
        let _sol = Solution::new(s_fixed, s_flex, &prob).expect("valid solution");
    }

    #[test]
    fn solution_new_missing_fixed_is_err() {
        let b1 = mk_berth(1, 0, 200);

        let rf = req_fixed(1, (0, 50), &[(1, 5)]);
        let af = asg_fixed(&rf, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed_in_prob =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed_in_prob.insert(af);

        let prob = Problem::new(berths, fixed_in_prob, RequestContainer::new()).unwrap();

        // Solution misses the fixed requirement
        let s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let s_flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        assert!(Solution::new(s_fixed, s_flex, &prob).is_err());
    }

    #[test]
    fn solution_new_missing_flexible_is_err() {
        let b1 = mk_berth(1, 0, 200);

        let rx = req_flex(2, (0, 50), &[(1, 5)]);
        let mut flex_in_prob: RequestContainer<i64, Request<FlexibleKind, i64>> =
            RequestContainer::new();
        flex_in_prob.insert(rx);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob = Problem::new(berths, AssignmentContainer::new(), flex_in_prob).unwrap();

        // Solution misses the flexible requirement
        let s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let s_flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        assert!(Solution::new(s_fixed, s_flex, &prob).is_err());
    }

    #[test]
    fn solution_new_extra_flexible_request_is_err() {
        let b1 = mk_berth(1, 0, 200);

        // The only flexible in the problem:
        let r_req = req_flex(10, (0, 50), &[(1, 5)]);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let mut flex_in_prob: RequestContainer<i64, Request<FlexibleKind, i64>> =
            RequestContainer::new();
        flex_in_prob.insert(r_req.clone());

        let prob = Problem::new(berths, AssignmentContainer::new(), flex_in_prob).unwrap();

        // Solution includes required one AND an extra (id 11) → should fail constructor
        let a_req = asg_flex(&r_req, &b1, 0);
        let r_extra = req_flex(11, (0, 50), &[(1, 5)]);
        let a_extra = asg_flex(&r_extra, &b1, 10);

        let s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(a_req);
        s_flex.insert(a_extra);

        assert!(Solution::new(s_fixed, s_flex, &prob).is_err());
    }

    // ---------- Cross-assignment feasibility (non-overlap & berth existence) ----------

    #[test]
    fn nonoverlap_unknown_berth_is_reported() {
        // Problem knows only berth 1, but solution uses berth 2.
        let b_known = mk_berth(1, 0, 1000);
        let b_unknown = mk_berth(2, 0, 1000);

        let r_fix = req_fixed(5, (0, 100), &[(2, 10)]); // request supports berth 2
        let a_fix = asg_fixed(&r_fix, &b_unknown, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b_known); // berth 2 NOT inserted into problem

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut fixed_solution =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed_solution.insert(a_fix);

        let flex_solution =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let err =
            validation::validate_nonoverlap(&fixed_solution, &flex_solution, &prob).unwrap_err();

        match err {
            CrossValidationError::UnknownBerth(e) => {
                assert_eq!(e.request(), rid(5));
                assert_eq!(e.requested_berth(), bid(2));
            }
            other => panic!("expected UnknownBerth, got {other}"),
        }
    }

    #[test]
    fn nonoverlap_reports_overlap_between_fixed_and_flexible_on_same_berth() {
        let b1 = mk_berth(1, 0, 1000);

        // fixed rid=30 on b1, len=10, start=0 -> [0,10)
        let rf = req_fixed(30, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        // flex rid=31 on b1, len=10, start=5 -> [5,15) overlaps
        let rx = req_flex(31, (0, 100), &[(1, 10)]);
        let ax = asg_flex(&rx, &b1, 5);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(af);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(ax);

        let err = validation::validate_nonoverlap(&s_fixed, &s_flex, &prob).unwrap_err();
        match err {
            CrossValidationError::Overlap(e) => {
                let a = e.first();
                let b = e.second();
                assert!([rid(30), rid(31)].contains(&a));
                assert!([rid(30), rid(31)].contains(&b));
                assert_ne!(a, b);
            }
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn nonoverlap_ok_same_berth_back_to_back() {
        let b1 = mk_berth(1, 0, 1000);

        // Two assignments on the same berth touching at boundary: [0,10) and [10,20)
        let r1 = req_fixed(101, (0, 100), &[(1, 10)]);
        let r2 = req_flex(102, (0, 100), &[(1, 10)]);

        let a1 = asg_fixed(&r1, &b1, 0);
        let a2 = asg_flex(&r2, &b1, 10);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(a1);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(a2);

        validation::validate_nonoverlap(&s_fixed, &s_flex, &prob).unwrap();
    }

    #[test]
    fn nonoverlap_ok_same_time_different_berths() {
        let b1 = mk_berth(1, 0, 1000);
        let b2 = mk_berth(2, 0, 1000);

        // Same time window but different berths: OK.
        let r1 = req_fixed(201, (0, 100), &[(1, 10)]);
        let r2 = req_flex(202, (0, 100), &[(2, 10)]);

        let a1 = asg_fixed(&r1, &b1, 0);
        let a2 = asg_flex(&r2, &b2, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1);
        berths.insert(b2);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(a1);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(a2);

        validation::validate_nonoverlap(&s_fixed, &s_flex, &prob).unwrap();
    }

    #[test]
    fn nonoverlap_ignores_zero_length_intervals() {
        // zero-length [10,10) must be ignored.
        let b1 = mk_berth(1, 0, 1000);

        let r_zero = req_fixed(301, (0, 100), &[(1, 0)]); // pt = 0
        let r_norm = req_flex(302, (0, 100), &[(1, 10)]);

        let a_zero = asg_fixed(&r_zero, &b1, 10); // [10,10)
        let a_norm = asg_flex(&r_norm, &b1, 0); // [0,10)

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(a_zero);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(a_norm);

        validation::validate_nonoverlap(&s_fixed, &s_flex, &prob).unwrap();
    }

    // ---------- Incremental check: validate_nooverlap_with ----------

    #[test]
    fn nooverlap_with_unknown_berth() {
        let b1 = mk_berth(1, 0, 1000);
        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        // Candidate on berth 2 which is not in problem
        let b2 = mk_berth(2, 0, 1000);
        let r = req_fixed(900, (0, 100), &[(2, 10)]);
        let cand = asg_fixed(&r, &b2, 0);

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let err = validation::validate_nooverlap_with::<_, FixedKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap_err();

        match err {
            CrossValidationError::UnknownBerth(e) => {
                assert_eq!(e.request(), rid(900));
                assert_eq!(e.requested_berth(), bid(2));
            }
            other => panic!("expected UnknownBerth, got {other}"),
        }
    }

    #[test]
    fn nooverlap_with_detects_overlap_against_fixed() {
        // Existing fixed: [0,10) on berth 1. Candidate overlaps: [5,15) on berth 1.
        let b1 = mk_berth(1, 0, 1000);
        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let rf = req_fixed(11, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let rx = req_flex(12, (0, 100), &[(1, 10)]);
        let cand = asg_flex(&rx, &b1, 5);

        let err = validation::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap_err();

        match err {
            CrossValidationError::Overlap(e) => {
                assert_eq!(e.first(), rid(11));
                assert_eq!(e.second(), rid(12));
            }
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn nooverlap_with_ok_back_to_back() {
        // Existing fixed: [0,10) on berth 1. Candidate: [10,20) on berth 1 → OK.
        let b1 = mk_berth(1, 0, 1000);
        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let rf = req_fixed(21, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let rx = req_flex(22, (0, 100), &[(1, 10)]);
        let cand = asg_flex(&rx, &b1, 10);

        validation::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap();
    }

    #[test]
    fn nooverlap_with_ok_different_berth_same_time() {
        // Existing fixed on berth 1; candidate on berth 2, same time → OK.
        let b1 = mk_berth(1, 0, 1000);
        let b2 = mk_berth(2, 0, 1000);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());
        berths.insert(b2.clone());

        let rf = req_fixed(31, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let rx = req_flex(32, (0, 100), &[(2, 10)]);
        let cand = asg_flex(&rx, &b2, 0);

        validation::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap();
    }

    #[test]
    fn nooverlap_with_ignores_zero_length_candidate() {
        // Candidate has zero processing time → ignored.
        let b1 = mk_berth(1, 0, 1000);
        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let rf = req_fixed(41, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let r_zero = req_flex(42, (0, 100), &[(1, 0)]); // pt = 0
        let cand = asg_flex(&r_zero, &b1, 5); // [5,5)

        validation::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap();
    }

    #[test]
    fn nooverlap_with_skips_same_request_id() {
        // Existing assignment rid=55 on berth 1; candidate has same rid=55 overlapping.
        // No error here; duplicates are handled by the other validators.
        let b1 = mk_berth(1, 0, 1000);
        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let r_fix = req_fixed(55, (0, 100), &[(1, 10)]);
        let a_existing = asg_fixed(&r_fix, &b1, 0);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_existing);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        // Same request id, overlapping time
        let r_same = req_fixed(55, (0, 100), &[(1, 10)]);
        let cand = asg_fixed(&r_same, &b1, 5);

        validation::validate_nooverlap_with::<_, FixedKind, _, _, _>(&fixed, &flex, &prob, &cand)
            .unwrap();
    }
}
