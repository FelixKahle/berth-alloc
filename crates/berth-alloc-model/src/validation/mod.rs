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

pub mod err;

use crate::{
    common::{FixedKind, FlexibleKind, Kind},
    problem::{
        asg::{AssignmentContainer, AssignmentView},
        err::AssignmentOverlapError,
        prob::Problem,
        req::{RequestIdentifier, RequestView},
    },
    validation::err::{CrossValidationError, RequestIdNotUniqueError},
};
use num_traits::{CheckedAdd, CheckedSub};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct StateValidator;

impl StateValidator {
    #[inline]
    pub fn validate_no_extra_fixed_assignments<T, AV>(
        fixed: &AssignmentContainer<FixedKind, T, AV>,
    ) -> Result<(), err::ExtraFixedAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FixedKind, T>,
    {
        let mut seen: BTreeSet<crate::problem::req::RequestIdentifier> = BTreeSet::new();
        for a in fixed.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(err::ExtraFixedAssignmentError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_fixed_assignments_with<T, AV>(
        fixed: &AssignmentContainer<FixedKind, T, AV>,
        asg: &AV,
    ) -> Result<(), err::ExtraFixedAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FixedKind, T>,
    {
        let mut seen: BTreeSet<RequestIdentifier> = BTreeSet::new();
        for a in fixed.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(err::ExtraFixedAssignmentError::new(rid));
            }
        }

        let rid_new = asg.request_id();
        if seen.contains(&rid_new) {
            return Err(err::ExtraFixedAssignmentError::new(rid_new));
        }

        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_flexible_assignments<T, AV>(
        flexible: &AssignmentContainer<FlexibleKind, T, AV>,
    ) -> Result<(), err::ExtraFlexibleAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FlexibleKind, T>,
    {
        let mut seen: BTreeSet<crate::problem::req::RequestIdentifier> = BTreeSet::new();
        for a in flexible.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(err::ExtraFlexibleAssignmentError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_flexible_assignments_with<T, AV>(
        flexible: &AssignmentContainer<FlexibleKind, T, AV>,
        asg: &AV,
    ) -> Result<(), err::ExtraFlexibleAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FlexibleKind, T>,
    {
        let mut seen: BTreeSet<crate::problem::req::RequestIdentifier> = BTreeSet::new();
        for a in flexible.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(err::ExtraFlexibleAssignmentError::new(rid));
            }
        }

        let rid_new = asg.request_id();
        if seen.contains(&rid_new) {
            return Err(err::ExtraFlexibleAssignmentError::new(rid_new));
        }

        Ok(())
    }

    #[inline]
    pub fn validate_request_ids_unique<T, AVF, AVX>(
        fixed: &AssignmentContainer<FixedKind, T, AVF>,
        flexible: &AssignmentContainer<FlexibleKind, T, AVX>,
    ) -> Result<(), RequestIdNotUniqueError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AVF: AssignmentView<FixedKind, T>,
        AVX: AssignmentView<FlexibleKind, T>,
    {
        let mut seen: BTreeSet<crate::problem::req::RequestIdentifier> = BTreeSet::new();

        for a in fixed.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(RequestIdNotUniqueError::new(rid));
            }
        }
        for a in flexible.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(RequestIdNotUniqueError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_request_ids_unique_with<T, K, V, AVF, AVX>(
        fixed: &AssignmentContainer<FixedKind, T, AVF>,
        flexible: &AssignmentContainer<FlexibleKind, T, AVX>,
        asg: &V,
    ) -> Result<(), RequestIdNotUniqueError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        K: Kind,
        V: AssignmentView<K, T>,
        AVF: AssignmentView<FixedKind, T>,
        AVX: AssignmentView<FlexibleKind, T>,
    {
        let mut seen: BTreeSet<crate::problem::req::RequestIdentifier> = BTreeSet::new();

        for a in fixed.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(RequestIdNotUniqueError::new(rid));
            }
        }
        for a in flexible.iter() {
            let rid = a.request_id();
            if !seen.insert(rid) {
                return Err(RequestIdNotUniqueError::new(rid));
            }
        }

        let rid_new = asg.request_id();
        if !seen.insert(rid_new) {
            return Err(RequestIdNotUniqueError::new(rid_new));
        }

        Ok(())
    }

    #[inline]
    pub fn validate_all_fixed_assignments_present<T, AV>(
        fixed: &AssignmentContainer<FixedKind, T, AV>,
        problem: &Problem<T>,
    ) -> Result<(), err::MissingFixedAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FixedKind, T>,
    {
        for assignment in problem.iter_fixed_assignments() {
            let rid = assignment.request_id();
            if !fixed.contains_id(rid) {
                return Err(err::MissingFixedAssignmentError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_all_fixed_assignments_present_with<T, AV>(
        fixed: &AssignmentContainer<FixedKind, T, AV>,
        problem: &Problem<T>,
        asg: &AV,
    ) -> Result<(), err::MissingFixedAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FixedKind, T>,
    {
        let rid_new = asg.request_id();

        for assignment in problem.iter_fixed_assignments() {
            let rid = assignment.request_id();
            if !fixed.contains_id(rid) && rid != rid_new {
                return Err(err::MissingFixedAssignmentError::new(rid));
            }
        }

        Ok(())
    }

    #[inline]
    pub fn validate_all_flexible_assignments_present<T, AV>(
        flexible: &AssignmentContainer<FlexibleKind, T, AV>,
        problem: &Problem<T>,
    ) -> Result<(), err::MissingFlexibleAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FlexibleKind, T>,
    {
        for request in problem.iter_flexible_requests() {
            let rid = request.id();
            if !flexible.contains_id(rid) {
                return Err(err::MissingFlexibleAssignmentError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_all_flexible_assignments_present_with<T, AV>(
        flexible: &AssignmentContainer<FlexibleKind, T, AV>,
        problem: &Problem<T>,
        asg: &AV,
    ) -> Result<(), err::MissingFlexibleAssignmentError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FlexibleKind, T>,
    {
        let rid_new = asg.request_id();

        for request in problem.iter_flexible_requests() {
            let rid = request.id();
            if !flexible.contains_id(rid) && rid != rid_new {
                return Err(err::MissingFlexibleAssignmentError::new(rid));
            }
        }

        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_fixed_requests<T, AV>(
        fixed: &AssignmentContainer<FixedKind, T, AV>,
        problem: &Problem<T>,
    ) -> Result<(), err::ExtraFixedRequestError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FixedKind, T>,
    {
        for a in fixed.iter() {
            let rid = a.request_id();
            if !problem.fixed_assignments().contains_id(rid) {
                return Err(err::ExtraFixedRequestError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_fixed_requests_with<T, AV>(
        fixed: &AssignmentContainer<FixedKind, T, AV>,
        problem: &Problem<T>,
        asg: &AV,
    ) -> Result<(), err::ExtraFixedRequestError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FixedKind, T>,
    {
        for a in fixed.iter() {
            let rid = a.request_id();
            if !problem.fixed_assignments().contains_id(rid) {
                return Err(err::ExtraFixedRequestError::new(rid));
            }
        }

        let rid_new = asg.request_id();
        if !problem.fixed_assignments().contains_id(rid_new) {
            return Err(err::ExtraFixedRequestError::new(rid_new));
        }

        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_flexible_requests<T, AV>(
        flexible: &AssignmentContainer<FlexibleKind, T, AV>,
        problem: &Problem<T>,
    ) -> Result<(), err::ExtraFlexibleRequestError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FlexibleKind, T>,
    {
        for a in flexible.iter() {
            let rid = a.request_id();
            if !problem.flexible_requests().contains_id(rid) {
                return Err(err::ExtraFlexibleRequestError::new(rid));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn validate_no_extra_flexible_requests_with<T, AV>(
        flexible: &AssignmentContainer<FlexibleKind, T, AV>,
        problem: &Problem<T>,
        asg: &AV,
    ) -> Result<(), err::ExtraFlexibleRequestError>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
        AV: AssignmentView<FlexibleKind, T>,
    {
        for a in flexible.iter() {
            let rid = a.request_id();
            if !problem.flexible_requests().contains_id(rid) {
                return Err(err::ExtraFlexibleRequestError::new(rid));
            }
        }

        let rid_new = asg.request_id();
        if !problem.flexible_requests().contains_id(rid_new) {
            return Err(err::ExtraFlexibleRequestError::new(rid_new));
        }

        Ok(())
    }

    #[inline]
    pub fn validate_nonoverlap<T, AVF, AVX>(
        fixed: &AssignmentContainer<FixedKind, T, AVF>,
        flexible: &AssignmentContainer<FlexibleKind, T, AVX>,
        problem: &Problem<T>,
    ) -> Result<(), CrossValidationError>
    where
        T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
        AVF: AssignmentView<FixedKind, T>,
        AVX: AssignmentView<FlexibleKind, T>,
    {
        let mut per_berth: crossvalidation::PerBerth<T> = BTreeMap::new();
        let berths = problem.berths();

        let mut process_assignment =
            |validated: crossvalidation::Validated<T>| -> Result<(), CrossValidationError> {
                let interval = validated.interval();
                if interval.is_empty() {
                    return Ok(());
                }

                let schedule = per_berth.entry(validated.berth_id()).or_default();
                let range = validated.range();

                if schedule.occupied().overlaps(&range) {
                    if let Some((_s_pred, &(e_pred, rid_pred))) =
                        schedule.starts().range(..=validated.start()).next_back()
                        && e_pred > validated.start()
                    {
                        return Err(
                            AssignmentOverlapError::new(rid_pred, validated.request_id()).into(),
                        );
                    }
                    if let Some((_s_succ, &(_e_succ, rid_succ))) = schedule
                        .starts()
                        .range(validated.start()..validated.end())
                        .next()
                    {
                        return Err(
                            AssignmentOverlapError::new(rid_succ, validated.request_id()).into(),
                        );
                    }
                    if let Some((_, &(_, rid_any))) = schedule.starts().iter().next() {
                        return Err(
                            AssignmentOverlapError::new(rid_any, validated.request_id()).into()
                        );
                    }
                }

                schedule.occupied_mut().insert(range);
                schedule
                    .starts_mut()
                    .insert(validated.start(), (validated.end(), validated.request_id()));
                Ok(())
            };

        let all_assignments = fixed
            .iter()
            .map(|a| crossvalidation::to_validated(a, berths))
            .chain(
                flexible
                    .iter()
                    .map(|a| crossvalidation::to_validated(a, berths)),
            );

        for validated_result in all_assignments {
            let validated = validated_result?;
            process_assignment(validated)?;
        }

        Ok(())
    }

    #[inline]
    pub fn validate_nooverlap_with<T, K, V, AVF, AVX>(
        fixed: &AssignmentContainer<FixedKind, T, AVF>,
        flexible: &AssignmentContainer<FlexibleKind, T, AVX>,
        problem: &Problem<T>,
        asg: &V,
    ) -> Result<(), CrossValidationError>
    where
        T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
        K: Kind,
        V: AssignmentView<K, T>,
        AVF: AssignmentView<FixedKind, T>,
        AVX: AssignmentView<FlexibleKind, T>,
    {
        let berths = problem.berths();
        let v = crossvalidation::to_validated(asg, berths)?;

        let v_interval = v.interval();
        if v_interval.is_empty() {
            return Ok(());
        }

        let conflicting_assignment = fixed
            .iter()
            .map(|a| (a.request_id(), a.berth_id(), a.interval()))
            .chain(
                flexible
                    .iter()
                    .map(|a| (a.request_id(), a.berth_id(), a.interval())),
            )
            .filter(|(rid, bid, interval)| {
                *rid != v.request_id() && *bid == v.berth_id() && !interval.is_empty()
            })
            .find(|(_, _, a_interval)| v_interval.intersects(a_interval));

        if let Some((conflicting_rid, _, _)) = conflicting_assignment {
            return Err(AssignmentOverlapError::new(conflicting_rid, v.request_id()).into());
        }

        Ok(())
    }
}

mod crossvalidation {
    use crate::common::Kind;
    use crate::problem::asg::AssignmentView;
    use crate::problem::berth::BerthContainer;
    use crate::problem::berth::BerthIdentifier;
    use crate::problem::err::BerthNotFoundError;
    use crate::problem::req::RequestIdentifier;
    use crate::validation::err::CrossValidationError;
    use berth_alloc_core::prelude::TimeInterval;
    use berth_alloc_core::prelude::TimePoint;
    use rangemap::RangeSet;
    use std::collections::BTreeMap;
    use std::ops::Range;

    type StartIndex<T> = BTreeMap<TimePoint<T>, (TimePoint<T>, RequestIdentifier)>;

    #[derive(Clone, Copy)]
    pub struct Validated<T: Copy + Ord> {
        rid: RequestIdentifier,
        bid: BerthIdentifier,
        start: TimePoint<T>,
        end: TimePoint<T>,
    }

    impl<T: Copy + Ord> Validated<T> {
        #[inline]
        pub fn request_id(&self) -> RequestIdentifier {
            self.rid
        }

        #[inline]
        pub fn berth_id(&self) -> BerthIdentifier {
            self.bid
        }

        #[inline]
        pub fn start(&self) -> TimePoint<T> {
            self.start
        }

        #[inline]
        pub fn end(&self) -> TimePoint<T> {
            self.end
        }

        #[inline]
        pub fn range(&self) -> Range<TimePoint<T>> {
            self.start..self.end
        }

        #[inline]
        pub fn interval(&self) -> TimeInterval<T> {
            TimeInterval::new(self.start, self.end)
        }
    }

    #[derive(Debug, Clone)]
    pub struct BerthSchedule<T: Copy + Ord> {
        occupied: RangeSet<TimePoint<T>>,
        starts: StartIndex<T>,
    }

    impl<T: Copy + Ord> BerthSchedule<T> {
        #[inline]
        pub fn occupied(&self) -> &RangeSet<TimePoint<T>> {
            &self.occupied
        }

        #[inline]
        pub fn starts(&self) -> &StartIndex<T> {
            &self.starts
        }

        #[inline]
        pub fn starts_mut(&mut self) -> &mut StartIndex<T> {
            &mut self.starts
        }

        #[inline]
        pub fn occupied_mut(&mut self) -> &mut RangeSet<TimePoint<T>> {
            &mut self.occupied
        }
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

    pub type PerBerth<T> = BTreeMap<BerthIdentifier, BerthSchedule<T>>;

    #[inline]
    pub fn to_validated<K: Kind, T>(
        a: &impl AssignmentView<K, T>,
        berths: &BerthContainer<T>,
    ) -> Result<Validated<T>, CrossValidationError>
    where
        T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
    {
        let bid = a.berth_id();
        let rid = a.request_id();

        if !berths.contains_id(bid) {
            return Err(BerthNotFoundError::new(rid, bid).into());
        }

        let start = a.start_time();
        let end = a.end_time();

        Ok(Validated {
            rid,
            bid,
            start,
            end,
        })
    }
}

#[cfg(test)]
mod more_tests {
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
    fn test_validate_no_extra_fixed_assignments_with_rejects_dup_candidate() {
        let b1 = mk_berth(1, 0, 1000);
        let r1 = req_fixed(1, (0, 100), &[(1, 10)]);
        let existing = asg_fixed(&r1, &b1, 0);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(existing);

        // same rid=1 coming in → error
        let r_same = req_fixed(1, (0, 100), &[(1, 10)]);
        let cand = asg_fixed(&r_same, &b1, 20);

        let err =
            StateValidator::validate_no_extra_fixed_assignments_with(&fixed, &cand).unwrap_err();
        assert_eq!(err.request_id(), rid(1));
    }

    #[test]
    fn test_validate_no_extra_flexible_assignments_with_rejects_dup_candidate() {
        let b1 = mk_berth(1, 0, 1000);
        let r1 = req_flex(10, (0, 100), &[(1, 10)]);
        let existing = asg_flex(&r1, &b1, 0);

        let mut flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        flex.insert(existing);

        // same rid=10 coming in → error
        let r_same = req_flex(10, (0, 100), &[(1, 10)]);
        let cand = asg_flex(&r_same, &b1, 20);

        let err =
            StateValidator::validate_no_extra_flexible_assignments_with(&flex, &cand).unwrap_err();
        assert_eq!(err.request_id(), rid(10));
    }

    // --------- request_ids_unique_with ---------------------------------------

    #[test]
    fn test_validate_request_ids_unique_with_ok_when_new_unique() {
        let b1 = mk_berth(1, 0, 1000);
        let r_fix = req_fixed(1, (0, 100), &[(1, 5)]);
        let r_flex = req_flex(2, (0, 100), &[(1, 5)]);

        let a_fix = asg_fixed(&r_fix, &b1, 0);
        let a_flex = asg_flex(&r_flex, &b1, 10);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_fix);
        let mut flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        flex.insert(a_flex);

        // candidate rid=3 → unique
        let r_new = req_flex(3, (0, 100), &[(1, 1)]);
        let cand = asg_flex(&r_new, &b1, 20);

        StateValidator::validate_request_ids_unique_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &cand,
        )
        .unwrap();
    }

    #[test]
    fn test_validate_request_ids_unique_with_rejects_when_collides_other_kind() {
        let b1 = mk_berth(1, 0, 1000);
        let r_fix = req_fixed(7, (0, 100), &[(1, 5)]);
        let a_fix = asg_fixed(&r_fix, &b1, 0);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_fix);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        // candidate rid=7 (collides with fixed)
        let r_new = req_flex(7, (0, 100), &[(1, 1)]);
        let cand = asg_flex(&r_new, &b1, 20);

        let err = StateValidator::validate_request_ids_unique_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &cand,
        )
        .unwrap_err();
        assert_eq!(err.request_id(), rid(7));
    }

    // --------- *_present_with -------------------------------------------------

    #[test]
    fn test_validate_all_fixed_assignments_present_with_ok_when_candidate_fills_missing() {
        let b1 = mk_berth(1, 0, 1000);
        let req_needed = req_fixed(100, (0, 100), &[(1, 5)]);
        let asg_needed = asg_fixed(&req_needed, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let mut prob_fixed =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        prob_fixed.insert(asg_needed.clone());

        let prob = Problem::new(berths, prob_fixed, RequestContainer::new()).unwrap();

        // current solution fixed set is empty; candidate is the needed one.
        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        StateValidator::validate_all_fixed_assignments_present_with(&fixed, &prob, &asg_needed)
            .unwrap();
    }

    #[test]
    fn test_validate_all_fixed_assignments_present_with_errors_when_candidate_is_different() {
        let b1 = mk_berth(1, 0, 1000);
        let req_needed = req_fixed(200, (0, 100), &[(1, 5)]);
        let asg_needed = asg_fixed(&req_needed, &b1, 0);

        let req_other = req_fixed(201, (0, 100), &[(1, 5)]);
        let asg_other = asg_fixed(&req_other, &b1, 10);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut prob_fixed =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        prob_fixed.insert(asg_needed);

        let prob = Problem::new(berths, prob_fixed, RequestContainer::new()).unwrap();

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let err =
            StateValidator::validate_all_fixed_assignments_present_with(&fixed, &prob, &asg_other)
                .unwrap_err();
        assert_eq!(err.request_id(), rid(200));
    }

    #[test]
    fn test_validate_all_flexible_assignments_present_with_ok_when_candidate_fills_missing() {
        let b1 = mk_berth(1, 0, 1000);
        let req_needed = req_flex(300, (0, 100), &[(1, 5)]);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let mut prob_flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        prob_flex.insert(req_needed.clone());

        let prob = Problem::new(berths, AssignmentContainer::new(), prob_flex).unwrap();

        let cand = asg_flex(&req_needed, &b1, 0);
        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        StateValidator::validate_all_flexible_assignments_present_with(&flex, &prob, &cand)
            .unwrap();
    }

    #[test]
    fn test_validate_all_flexible_assignments_present_with_errors_when_candidate_is_different() {
        let b1 = mk_berth(1, 0, 1000);
        let req_needed = req_flex(301, (0, 100), &[(1, 5)]);
        let req_other = req_flex(302, (0, 100), &[(1, 5)]);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let mut prob_flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        prob_flex.insert(req_needed.clone());

        let prob = Problem::new(berths, AssignmentContainer::new(), prob_flex).unwrap();

        let cand = asg_flex(&req_other, &b1, 0);
        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let err =
            StateValidator::validate_all_flexible_assignments_present_with(&flex, &prob, &cand)
                .unwrap_err();
        assert_eq!(err.request_id(), rid(301));
    }

    #[test]
    fn test_validate_no_extra_fixed_requests_with_ok_when_in_problem() {
        let b1 = mk_berth(1, 0, 1000);
        let req_needed = req_fixed(400, (0, 100), &[(1, 5)]);
        let asg_needed = asg_fixed(&req_needed, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut prob_fixed =
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        prob_fixed.insert(asg_needed.clone());

        let prob = Problem::new(berths, prob_fixed, RequestContainer::new()).unwrap();

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        StateValidator::validate_no_extra_fixed_requests_with(&fixed, &prob, &asg_needed).unwrap();
    }

    #[test]
    fn test_validate_no_extra_fixed_requests_with_errors_when_not_in_problem() {
        let b1 = mk_berth(1, 0, 1000);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        // Problem expects no fixed requests at all
        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::new(),
        )
        .unwrap();

        // Candidate with rid=401 is thus "extra"
        let r = req_fixed(401, (0, 100), &[(1, 5)]);
        let cand = asg_fixed(&r, &b1, 0);

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let err = StateValidator::validate_no_extra_fixed_requests_with(&fixed, &prob, &cand)
            .unwrap_err();
        assert_eq!(err.request_id(), rid(401));
    }

    #[test]
    fn test_validate_no_extra_flexible_requests_with_ok_when_in_problem() {
        let b1 = mk_berth(1, 0, 1000);
        let r = req_flex(500, (0, 100), &[(1, 5)]);
        let cand = asg_flex(&r, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut prob_flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        prob_flex.insert(r);

        let prob = Problem::new(berths, AssignmentContainer::new(), prob_flex).unwrap();

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        StateValidator::validate_no_extra_flexible_requests_with(&flex, &prob, &cand).unwrap();
    }

    #[test]
    fn test_validate_no_extra_flexible_requests_with_errors_when_not_in_problem() {
        let b1 = mk_berth(1, 0, 1000);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        // Problem expects no flexible requests
        let prob = Problem::new(
            berths,
            AssignmentContainer::new(),
            RequestContainer::<i64, Request<FlexibleKind, i64>>::new(),
        )
        .unwrap();

        let r = req_flex(501, (0, 100), &[(1, 5)]);
        let cand = asg_flex(&r, &b1, 0);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let err = StateValidator::validate_no_extra_flexible_requests_with(&flex, &prob, &cand)
            .unwrap_err();
        assert_eq!(err.request_id(), rid(501));
    }

    #[test]
    fn test_nonoverlap_reports_overlap_within_fixed_only() {
        let b1 = mk_berth(1, 0, 1000);
        let r1 = req_fixed(600, (0, 100), &[(1, 10)]);
        let r2 = req_fixed(601, (0, 100), &[(1, 10)]);
        let a1 = asg_fixed(&r1, &b1, 0); // [0,10)
        let a2 = asg_fixed(&r2, &b1, 5); // [5,15)

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::new(),
        )
        .unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a1);
        fixed.insert(a2);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        let err = StateValidator::validate_nonoverlap(&fixed, &flex, &prob).unwrap_err();
        match err {
            CrossValidationError::Overlap(e) => {
                assert!([rid(600), rid(601)].contains(&e.first()));
                assert!([rid(600), rid(601)].contains(&e.second()));
                assert_ne!(e.first(), e.second());
            }
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn test_nonoverlap_reports_overlap_within_flexible_only() {
        let b1 = mk_berth(1, 0, 1000);
        let r1 = req_flex(700, (0, 100), &[(1, 12)]);
        let r2 = req_flex(701, (0, 100), &[(1, 12)]);
        let a1 = asg_flex(&r1, &b1, 0); // [0,12)
        let a2 = asg_flex(&r2, &b1, 1); // [1,13)

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob = Problem::new(
            berths,
            AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new(),
            RequestContainer::new(),
        )
        .unwrap();

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let mut flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        flex.insert(a1);
        flex.insert(a2);

        let err = StateValidator::validate_nonoverlap(&fixed, &flex, &prob).unwrap_err();
        match err {
            CrossValidationError::Overlap(e) => {
                assert!([rid(700), rid(701)].contains(&e.first()));
                assert!([rid(700), rid(701)].contains(&e.second()));
                assert_ne!(e.first(), e.second());
            }
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn test_nonoverlap_reports_equal_intervals_as_overlap() {
        let b1 = mk_berth(1, 0, 1000);
        let r1 = req_fixed(800, (0, 100), &[(1, 10)]);
        let r2 = req_flex(801, (0, 100), &[(1, 10)]);
        let a1 = asg_fixed(&r1, &b1, 30); // [30,40)
        let a2 = asg_flex(&r2, &b1, 30); // [30,40)

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let prob =
            Problem::new(berths, AssignmentContainer::new(), RequestContainer::new()).unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a1);
        let mut flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        flex.insert(a2);

        let err = StateValidator::validate_nonoverlap(&fixed, &flex, &prob).unwrap_err();
        match err {
            CrossValidationError::Overlap(_) => {}
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn test_validate_nooverlap_with_candidate_inside_existing_is_overlap() {
        let b1 = mk_berth(1, 0, 1000);
        // existing [10,40)
        let r_exist = req_fixed(900, (0, 100), &[(1, 30)]);
        let a_exist = asg_fixed(&r_exist, &b1, 10);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let prob =
            Problem::new(berths, AssignmentContainer::new(), RequestContainer::new()).unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_exist);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        // candidate [20,25) inside
        let r_cand = req_flex(901, (0, 100), &[(1, 5)]);
        let cand = asg_flex(&r_cand, &b1, 20);

        let err = StateValidator::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap_err();

        match err {
            CrossValidationError::Overlap(e) => {
                assert_eq!(e.second(), rid(901));
                assert_eq!(e.first(), rid(900));
            }
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn test_validate_nooverlap_with_existing_inside_candidate_is_overlap() {
        let b1 = mk_berth(1, 0, 1000);
        // existing [20,25)
        let r_exist = req_flex(910, (0, 100), &[(1, 5)]);
        let a_exist = asg_flex(&r_exist, &b1, 20);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let prob =
            Problem::new(berths, AssignmentContainer::new(), RequestContainer::new()).unwrap();

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let mut flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        flex.insert(a_exist);

        // candidate [10,40)
        let r_cand = req_fixed(911, (0, 100), &[(1, 30)]);
        let cand = asg_fixed(&r_cand, &b1, 10);

        let err = StateValidator::validate_nooverlap_with::<_, FixedKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap_err();

        match err {
            CrossValidationError::Overlap(e) => {
                assert_eq!(e.first(), rid(910));
                assert_eq!(e.second(), rid(911));
            }
            other => panic!("expected Overlap, got {other}"),
        }
    }

    #[test]
    fn test_validate_nooverlap_with_adjacency_is_ok() {
        let b1 = mk_berth(1, 0, 1000);
        // existing [0,10)
        let r_exist = req_fixed(920, (0, 100), &[(1, 10)]);
        let a_exist = asg_fixed(&r_exist, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let prob =
            Problem::new(berths, AssignmentContainer::new(), RequestContainer::new()).unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_exist);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        // candidate starts exactly at 10 → [10,20)
        let r_cand = req_flex(921, (0, 100), &[(1, 10)]);
        let cand = asg_flex(&r_cand, &b1, 10);

        StateValidator::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap();
    }

    #[test]
    fn test_validate_nooverlap_with_same_rid_is_ignored() {
        let b1 = mk_berth(1, 0, 1000);
        // existing rid=930 [0,10)
        let r_exist = req_fixed(930, (0, 100), &[(1, 10)]);
        let a_exist = asg_fixed(&r_exist, &b1, 0);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());

        let prob =
            Problem::new(berths, AssignmentContainer::new(), RequestContainer::new()).unwrap();

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_exist);

        let flex = AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();

        // candidate has same rid=930 and overlaps; should be ignored by _with
        let r_cand = req_fixed(930, (0, 100), &[(1, 10)]);
        let cand = asg_fixed(&r_cand, &b1, 5);

        StateValidator::validate_nooverlap_with::<_, FixedKind, _, _, _>(
            &fixed, &flex, &prob, &cand,
        )
        .unwrap();
    }

    #[test]
    fn test_validate_nooverlap_with_many_ok_mix() {
        let b1 = mk_berth(1, 0, 1000);
        let b2 = mk_berth(2, 0, 1000);

        let mut berths = BerthContainer::new();
        berths.insert(b1.clone());
        berths.insert(b2.clone());

        let prob =
            Problem::new(berths, AssignmentContainer::new(), RequestContainer::new()).unwrap();

        // existing: berth1 [0,10), berth2 [5,15)
        let r_f1 = req_fixed(940, (0, 100), &[(1, 10)]);
        let a_f1 = asg_fixed(&r_f1, &b1, 0);

        let r_x2 = req_flex(941, (0, 100), &[(2, 10)]);
        let a_x2 = asg_flex(&r_x2, &b2, 5);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a_f1);
        let mut flex =
            AssignmentContainer::<FlexibleKind, i64, Assignment<FlexibleKind, i64>>::new();
        flex.insert(a_x2);

        // candidate on berth1 [10,20) → OK
        let r_c1 = req_flex(942, (0, 100), &[(1, 10)]);
        let cand1 = asg_flex(&r_c1, &b1, 10);
        StateValidator::validate_nooverlap_with::<_, FlexibleKind, _, _, _>(
            &fixed, &flex, &prob, &cand1,
        )
        .unwrap();

        // candidate on berth2 [0,5) → OK (before)
        let r_c2 = req_fixed(943, (0, 100), &[(2, 5)]);
        let cand2 = asg_fixed(&r_c2, &b2, 0);
        StateValidator::validate_nooverlap_with::<_, FixedKind, _, _, _>(
            &fixed, &flex, &prob, &cand2,
        )
        .unwrap();
    }
}
