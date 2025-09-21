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

use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use rangemap::RangeSet;

use crate::{
    common::{FixedKind, FlexibleKind, Kind},
    problem::{
        asg::{Assignment, AssignmentContainer, AssignmentRef, AssignmentView},
        berth::{BerthContainer, BerthIdentifier},
        err::{
            AssignmenStartsBeforeFeasibleWindowError, AssignmentEndsAfterFeasibleWindowError,
            AssignmentOverlapError, BerthNotFoundError, IncomatibleBerthError,
        },
        prob::Problem,
        req::RequestIdentifier,
    },
    solution::err::SolutionValidationError,
};
use std::{
    collections::BTreeMap,
    ops::{Mul, Range},
};

type StartIndex<T> = BTreeMap<TimePoint<T>, (TimePoint<T>, RequestIdentifier)>;

#[derive(Clone, Copy)]
struct Validated<T: Copy + Ord> {
    rid: RequestIdentifier,
    bid: BerthIdentifier,
    start: TimePoint<T>,
    end: TimePoint<T>,
}

impl<T: Copy + Ord> Validated<T> {
    #[inline]
    fn range(&self) -> Range<TimePoint<T>> {
        self.start..self.end
    }
}

#[derive(Debug, Clone)]
struct BerthSchedule<T: Copy + Ord> {
    occupied: RangeSet<TimePoint<T>>,
    starts: StartIndex<T>,
}

impl<T: Copy + Ord> Default for BerthSchedule<T> {
    #[inline]
    fn default() -> Self {
        Self {
            occupied: RangeSet::new(),
            starts: StartIndex::new(),
        }
    }
}

type PerBerth<T> = BTreeMap<BerthIdentifier, BerthSchedule<T>>;
type ValidateOneResult<T> = Result<Validated<T>, SolutionValidationError<T>>;

fn validate_one<K: Kind, T: Copy + Ord + CheckedAdd + CheckedSub>(
    a: &impl AssignmentView<K, T>,
    berths: &BerthContainer<T>,
) -> ValidateOneResult<T> {
    let bid = a.berth_id();
    let rid = a.request_id();

    if !berths.contains_id(bid) {
        return Err(SolutionValidationError::UnknownBerth(
            BerthNotFoundError::new(rid, bid),
        ));
    }

    let req = a.request();
    if !req.is_berth_feasible(bid) {
        return Err(SolutionValidationError::Incompatible(
            IncomatibleBerthError::new(rid, bid),
        ));
    }

    let window = req.feasible_window();
    let start = a.start_time();
    if start < window.start() {
        return Err(
            SolutionValidationError::AssignmentStartsBeforeFeasibleWindow(
                AssignmenStartsBeforeFeasibleWindowError::new(rid, window.start(), start),
            ),
        );
    }

    let end = a.end_time();
    if end > window.end() {
        return Err(SolutionValidationError::AssignmentEndsAfterFeasibleWindow(
            AssignmentEndsAfterFeasibleWindowError::new(rid, end, window),
        ));
    }

    Ok(Validated {
        rid,
        bid,
        start,
        end,
    })
}

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

    #[inline]
    fn validate(&self, prob: &Problem<T>) -> Result<(), SolutionValidationError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.validate_assignment_coverage(prob)?;
        self.validate_assignments_and_overlaps(prob)
    }

    #[inline]
    fn validate_assignment_coverage(
        &self,
        prob: &Problem<T>,
    ) -> Result<(), SolutionValidationError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        for assignment in prob.iter_fixed_assignments() {
            let rid = assignment.request_id();
            if !self.fixed_assignments().contains_id(rid) {
                return Err(SolutionValidationError::MissingFixed(rid));
            }
        }

        for request in prob.iter_flexible_requests() {
            let rid = request.id();
            if !self.flexible_assignments().contains_id(rid) {
                return Err(SolutionValidationError::MissingFlexible(rid));
            }
        }

        for a in self.flexible_assignments().iter() {
            let rid = a.request_id();
            let exists_in_problem = prob.flexible_requests().iter().any(|r| r.id() == rid);
            if !exists_in_problem {
                return Err(SolutionValidationError::ExtraFlexible(rid));
            }
        }

        for a in self.fixed_assignments().iter() {
            let rid = a.request_id();
            let exists_in_problem = prob
                .fixed_assignments()
                .iter()
                .any(|r| r.request_id() == rid);
            if !exists_in_problem {
                return Err(SolutionValidationError::ExtraFixed(rid));
            }
        }

        Ok(())
    }

    fn validate_assignments_and_overlaps(
        &self,
        prob: &Problem<T>,
    ) -> Result<(), SolutionValidationError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let mut per_berth: PerBerth<T> = BTreeMap::new();
        let mut push = |v: Validated<T>| -> Result<(), SolutionValidationError<T>> {
            if v.start >= v.end {
                return Ok(());
            }

            let sched = per_berth.entry(v.bid).or_default();
            let iv = v.range();

            if sched.occupied.overlaps(&iv) {
                if let Some((_s_pred, &(e_pred, rid_pred))) =
                    sched.starts.range(..=v.start).next_back()
                    && e_pred > v.start
                {
                    return Err(SolutionValidationError::Overlap(
                        AssignmentOverlapError::new(rid_pred, v.rid),
                    ));
                }

                if let Some((_s_succ, &(_e_succ, rid_succ))) =
                    sched.starts.range(v.start..v.end).next()
                {
                    return Err(SolutionValidationError::Overlap(
                        AssignmentOverlapError::new(rid_succ, v.rid),
                    ));
                }

                if let Some((_s_any, &(_e_any, rid_any))) = sched.starts.iter().next() {
                    return Err(SolutionValidationError::Overlap(
                        AssignmentOverlapError::new(rid_any, v.rid),
                    ));
                }
            }

            sched.occupied.insert(iv);
            sched.starts.insert(v.start, (v.end, v.rid));
            Ok(())
        };

        for a in self.fixed_assignments().iter() {
            push(validate_one(a, prob.berths())?)?;
        }
        for a in self.flexible_assignments().iter() {
            push(validate_one(a, prob.berths())?)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Solution<T: Copy + Ord> {
    fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
    flexible_assignments: AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>,
}

impl<T: Copy + Ord> Solution<T> {
    #[inline]
    pub fn new(
        fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
        flexible_assignments: AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>,
    ) -> Self {
        Self {
            fixed_assignments,
            flexible_assignments,
        }
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

        SolutionRef::new(fixed, flex)
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

impl<'p, T: Copy + Ord> SolutionRef<'p, T> {
    #[inline]
    pub fn new(
        fixed_assignments: AssignmentContainer<FixedKind, T, AssignmentRef<'p, 'p, FixedKind, T>>,
        flexible_assignments: AssignmentContainer<
            FlexibleKind,
            T,
            AssignmentRef<'p, 'p, FlexibleKind, T>,
        >,
    ) -> Self {
        Self {
            fixed_assignments,
            flexible_assignments,
        }
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

        Solution::new(fixed, flex)
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
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
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

    #[test]
    fn test_validate_ok_basic() {
        let b1 = mk_berth(1, 0, 200);
        let b2 = mk_berth(2, 0, 200);

        // fixed: id 11 on b1, len=10, in [0,100]
        let rf = req_fixed(11, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        // flexible: id 21 on b2, len=10, in [0,100]
        let rx = req_flex(21, (0, 100), &[(2, 10)]);
        let ax = asg_flex(&rx, &b2, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());
        berths.insert(b2.clone());

        let mut prob_fixed =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        prob_fixed.insert(af.clone());

        let mut prob_flex = RequestContainer::<FlexibleKind, i64>::new();
        prob_flex.insert(rx.clone());

        let prob = Problem::new(berths, prob_fixed, prob_flex).unwrap();

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(af);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(ax);

        let sol = Solution::new(s_fixed, s_flex);
        sol.as_ref().validate(&prob).unwrap();
    }

    #[test]
    fn test_missing_fixed_is_reported() {
        let b1 = mk_berth(1, 0, 200);

        let rf = req_fixed(1, (0, 50), &[(1, 5)]);
        let af = asg_fixed(&rf, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed_in_prob =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed_in_prob.insert(af);

        let prob = Problem::new(berths, fixed_in_prob, RequestContainer::new()).unwrap();

        let sol = Solution::<i64>::new(AssignmentContainer::new(), AssignmentContainer::new());
        let err = sol.as_ref().validate(&prob).unwrap_err();
        match err {
            SolutionValidationError::MissingFixed(id) => assert_eq!(id, rid(1)),
            other => panic!("expected MissingFixed, got {other}"),
        }
    }

    #[test]
    fn test_missing_flexible_is_reported() {
        let b1 = mk_berth(1, 0, 200);

        let rx = req_flex(2, (0, 50), &[(1, 5)]);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut flex_in_prob = RequestContainer::<FlexibleKind, i64>::new();
        flex_in_prob.insert(rx);

        let prob = Problem::new(berths, AssignmentContainer::new(), flex_in_prob).unwrap();

        let sol = Solution::<i64>::new(AssignmentContainer::new(), AssignmentContainer::new());
        let err = sol.as_ref().validate(&prob).unwrap_err();
        match err {
            SolutionValidationError::MissingFlexible(id) => assert_eq!(id, rid(2)),
            other => panic!("expected MissingFlexible, got {other}"),
        }
    }

    #[test]
    fn test_extra_flexible_is_reported() {
        let b1 = mk_berth(1, 0, 200);

        // The only required flexible in the problem:
        let r_req = req_flex(10, (0, 50), &[(1, 5)]);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let mut flex_in_prob = RequestContainer::<FlexibleKind, i64>::new();
        flex_in_prob.insert(r_req.clone());

        let prob = Problem::new(berths, AssignmentContainer::new(), flex_in_prob).unwrap();

        // Solution assigns the required one AND an extra one (id 11) → ExtraFlexible
        let a_req = asg_flex(&r_req, &b1, 0);
        let r_extra = req_flex(11, (0, 50), &[(1, 5)]);
        let a_extra = asg_flex(&r_extra, &b1, 10);

        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(a_req);
        s_flex.insert(a_extra);

        let sol = Solution::new(AssignmentContainer::new(), s_flex);
        let err = sol.as_ref().validate(&prob).unwrap_err();
        match err {
            SolutionValidationError::ExtraFlexible(id) => assert_eq!(id, rid(11)),
            other => panic!("expected ExtraFlexible, got {other}"),
        }
    }

    #[test]
    fn test_overlap_across_fixed_and_flexible_is_reported() {
        let b1 = mk_berth(1, 0, 200);

        // fixed req id=30 on b1, len=10, start=0 -> [0,10)
        let rf = req_fixed(30, (0, 100), &[(1, 10)]);
        let af = asg_fixed(&rf, &b1, 0);

        // flex req id=31 on b1, len=10, start=5 -> [5,15) overlaps
        let rx = req_flex(31, (0, 100), &[(1, 10)]);
        let ax = asg_flex(&rx, &b1, 5);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed_in_prob =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed_in_prob.insert(af.clone());

        let mut flex_in_prob = RequestContainer::<FlexibleKind, i64>::new();
        flex_in_prob.insert(rx.clone());

        let prob = Problem::new(berths, fixed_in_prob, flex_in_prob).unwrap();

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(af);
        let mut s_flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        s_flex.insert(ax);

        let sol = Solution::new(s_fixed, s_flex);
        let err = sol.as_ref().validate(&prob).unwrap_err();
        match err {
            SolutionValidationError::Overlap(e) => {
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
    fn test_unknown_berth_on_fixed_is_reported_even_if_request_id_matches() {
        // Problem only knows berth 1
        let b1 = mk_berth(1, 0, 200);

        // Problem's fixed requirement (id=40) scheduled on b1
        let rf_prob = req_fixed(40, (0, 100), &[(1, 5)]);
        let af_prob = asg_fixed(&rf_prob, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let mut fixed_in_prob =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed_in_prob.insert(af_prob);

        // No flexible requirements in the problem here
        let prob = Problem::new(berths, fixed_in_prob, RequestContainer::new()).unwrap();

        // Solution provides a *different* assignment for the *same request id* 40,
        // but on an unknown berth 3.
        let b3 = mk_berth(3, 0, 200);
        let rf_sol = req_fixed(40, (0, 100), &[(3, 5)]); // note: maps to berth 3
        let af_sol = asg_fixed(&rf_sol, &b3, 10);

        let mut s_fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        s_fixed.insert(af_sol);

        let sol = Solution::new(s_fixed, AssignmentContainer::new());
        let err = sol.as_ref().validate(&prob).unwrap_err();
        match err {
            SolutionValidationError::UnknownBerth(e) => {
                assert_eq!(e.request(), rid(40));
                assert_eq!(e.requested_berth(), bid(3));
            }
            other => panic!("expected UnknownBerth, got {other}"),
        }
    }
}
