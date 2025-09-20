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

use berth_alloc_core::prelude::TimePoint;
use num_traits::{CheckedAdd, CheckedSub};

use crate::{
    common::{FixedKind, FlexibleKind, Kind},
    problem::{
        AssignmenStartsBeforeFeasibleWindowError, Assignment,
        AssignmentEndsAfterFeasibleWindowError, AssignmentOverlapError, AssignmentRef, Berth,
        BerthIdentifier, BerthNotFoundError, IncomatibleBerthError, Problem, RequestIdentifier,
    },
    solution::err::SolutionValidationError,
};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct Solution<T: Copy + Ord> {
    fixed_assignments: HashMap<RequestIdentifier, Assignment<FixedKind, T>>,
    flexible_assignments: HashMap<RequestIdentifier, Assignment<FlexibleKind, T>>,
}

impl<T: Copy + Ord> Solution<T> {
    #[inline]
    pub fn new(
        fixed_assignments: HashMap<RequestIdentifier, Assignment<FixedKind, T>>,
        flexible_assignments: HashMap<RequestIdentifier, Assignment<FlexibleKind, T>>,
    ) -> Self {
        Self {
            fixed_assignments,
            flexible_assignments,
        }
    }

    #[inline]
    pub fn fixed_assignments(&self) -> &HashMap<RequestIdentifier, Assignment<FixedKind, T>> {
        &self.fixed_assignments
    }

    #[inline]
    pub fn flexible_assignments(&self) -> &HashMap<RequestIdentifier, Assignment<FlexibleKind, T>> {
        &self.flexible_assignments
    }

    #[inline]
    pub fn total_assignments(&self) -> usize {
        self.fixed_assignments.len() + self.flexible_assignments.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fixed_assignments.is_empty() && self.flexible_assignments.is_empty()
    }

    #[inline]
    pub fn as_ref(&self) -> SolutionRef<'_, T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        let fixed = self
            .fixed_assignments
            .iter()
            .map(|(&rid, a)| (rid, a.to_ref()))
            .collect::<HashMap<RequestIdentifier, AssignmentRef<'_, '_, FixedKind, T>>>();
        let flex = self
            .flexible_assignments
            .iter()
            .map(|(&rid, a)| (rid, a.to_ref()))
            .collect::<HashMap<RequestIdentifier, AssignmentRef<'_, '_, FlexibleKind, T>>>();
        SolutionRef::new(fixed, flex)
    }
}

#[derive(Debug, Clone)]
pub struct SolutionRef<'p, T: Copy + Ord> {
    fixed_assignments: HashMap<RequestIdentifier, AssignmentRef<'p, 'p, FixedKind, T>>,
    flexible_assignments: HashMap<RequestIdentifier, AssignmentRef<'p, 'p, FlexibleKind, T>>,
}

impl<'p, T: Copy + Ord> SolutionRef<'p, T> {
    #[inline]
    pub fn new(
        fixed_assignments: HashMap<RequestIdentifier, AssignmentRef<'p, 'p, FixedKind, T>>,
        flexible_assignments: HashMap<RequestIdentifier, AssignmentRef<'p, 'p, FlexibleKind, T>>,
    ) -> Self {
        Self {
            fixed_assignments,
            flexible_assignments,
        }
    }

    #[inline]
    pub fn fixed_assignments(
        &self,
    ) -> &HashMap<RequestIdentifier, AssignmentRef<'p, 'p, FixedKind, T>> {
        &self.fixed_assignments
    }

    #[inline]
    pub fn flexible_assignments(
        &self,
    ) -> &HashMap<RequestIdentifier, AssignmentRef<'p, 'p, FlexibleKind, T>> {
        &self.flexible_assignments
    }

    #[inline]
    pub fn total_assignments(&self) -> usize {
        self.fixed_assignments.len() + self.flexible_assignments.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fixed_assignments.is_empty() && self.flexible_assignments.is_empty()
    }

    #[inline]
    pub fn to_owned(&self) -> Solution<T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        let fixed = self
            .fixed_assignments
            .iter()
            .map(|(&rid, a)| (rid, a.to_owned()))
            .collect::<HashMap<RequestIdentifier, Assignment<FixedKind, T>>>();

        let flex = self
            .flexible_assignments
            .iter()
            .map(|(&rid, a)| (rid, a.to_owned()))
            .collect::<HashMap<RequestIdentifier, Assignment<FlexibleKind, T>>>();

        Solution {
            fixed_assignments: fixed,
            flexible_assignments: flex,
        }
    }
}

#[derive(Debug)]
struct ValidatedAssignment<T> {
    req_id: RequestIdentifier,
    berth_id: BerthIdentifier,
    start: TimePoint<T>,
    end: TimePoint<T>,
}

impl<'p, T: Copy + Ord> SolutionRef<'p, T> {
    #[inline]
    pub fn validate(&self, prob: &Problem<T>) -> Result<(), SolutionValidationError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.validate_assignment_coverage(prob)?;
        self.validate_assignments_and_overlaps(prob)
    }

    fn validate_assignment_coverage(
        &self,
        prob: &Problem<T>,
    ) -> Result<(), SolutionValidationError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        for assignment in prob.fixed_assignments() {
            let rid = assignment.request_id();
            if !self.fixed_assignments.contains_key(&rid) {
                return Err(SolutionValidationError::MissingFixed(rid));
            }
        }

        for request in prob.flexible_requests() {
            let rid = request.id();
            if !self.flexible_assignments.contains_key(&rid) {
                return Err(SolutionValidationError::MissingFlexible(rid));
            }
        }

        // Check for extra flexible assignments
        for &rid in self.flexible_assignments.keys() {
            if !prob.flexible_requests().iter().any(|r| r.id() == rid) {
                return Err(SolutionValidationError::ExtraFlexible(rid));
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
        #[derive(Clone, Copy)]
        struct Assignment<T> {
            req: RequestIdentifier,
            start: TimePoint<T>,
            end: TimePoint<T>,
        }

        let mut per_berth: BTreeMap<BerthIdentifier, Vec<Assignment<T>>> = BTreeMap::new();

        for assignment in self.fixed_assignments.values() {
            let validated = Self::validate_single_assignment(assignment, prob.berths())?;
            per_berth
                .entry(validated.berth_id)
                .or_default()
                .push(Assignment {
                    req: validated.req_id,
                    start: validated.start,
                    end: validated.end,
                });
        }

        for assignment in self.flexible_assignments.values() {
            let validated = Self::validate_single_assignment(assignment, prob.berths())?;
            per_berth
                .entry(validated.berth_id)
                .or_default()
                .push(Assignment {
                    req: validated.req_id,
                    start: validated.start,
                    end: validated.end,
                });
        }

        for assignments in per_berth.values_mut() {
            assignments.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

            for window in assignments.windows(2) {
                if window[0].end > window[1].start {
                    return Err(SolutionValidationError::Overlap(
                        AssignmentOverlapError::new(window[0].req, window[1].req),
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_single_assignment<K: Kind>(
        assignment: &AssignmentRef<'_, '_, K, T>,
        berths_map: &HashMap<BerthIdentifier, Berth<T>>,
    ) -> Result<ValidatedAssignment<T>, SolutionValidationError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let bid = assignment.berth_id();
        let rid = assignment.request_id();

        if !berths_map.contains_key(&bid) {
            return Err(SolutionValidationError::UnknownBerth(
                BerthNotFoundError::new(rid, bid),
            ));
        }

        let req = assignment.request();
        if !req.is_berth_feasible(bid) {
            return Err(SolutionValidationError::Incompatible(
                IncomatibleBerthError::new(rid, bid),
            ));
        };

        let window = req.feasible_window();
        let start = assignment.start_time();

        if start < window.start() {
            return Err(
                SolutionValidationError::AssignmentStartsBeforeFeasibleWindow(
                    AssignmenStartsBeforeFeasibleWindowError::new(rid, window.start(), start),
                ),
            );
        }

        let end = assignment.end_time();
        if end > window.end() {
            return Err(SolutionValidationError::AssignmentEndsAfterFeasibleWindow(
                AssignmentEndsAfterFeasibleWindowError::new(rid, end, window),
            ));
        }

        Ok(ValidatedAssignment {
            req_id: rid,
            berth_id: bid,
            start,
            end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::{FixedKind, FlexibleKind},
        problem::{
            asg::Assignment, berth::Berth, berth::BerthIdentifier, prob::Problem, req::Request,
            req::RequestIdentifier,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use std::collections::{BTreeMap, HashMap, HashSet};

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
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), m).unwrap()
    }

    fn req_flex(id: u32, window: (i64, i64), pts: &[(u32, i64)]) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), m).unwrap()
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

        let mut berths = HashMap::new();
        berths.insert(b1.id(), b1.clone());
        berths.insert(b2.id(), b2.clone());

        let mut fixed_reqset = HashSet::new();
        fixed_reqset.insert(af.clone());

        let mut flex_reqset = HashSet::new();
        flex_reqset.insert(rx.clone());

        let prob = Problem::new(berths, fixed_reqset, flex_reqset).unwrap();

        let mut s_fixed = HashMap::new();
        s_fixed.insert(rid(11), af);
        let mut s_flex = HashMap::new();
        s_flex.insert(rid(21), ax);

        let sol = Solution::new(s_fixed, s_flex);
        sol.as_ref().validate(&prob).unwrap();
    }

    #[test]
    fn test_missing_fixed_is_reported() {
        let b1 = mk_berth(1, 0, 200);

        let rf = req_fixed(1, (0, 50), &[(1, 5)]);
        let af = asg_fixed(&rf, &b1, 0);

        let mut berths = HashMap::new();
        berths.insert(b1.id(), b1);

        let mut fixed_in_prob = HashSet::new();
        fixed_in_prob.insert(af);

        let prob = Problem::new(berths, fixed_in_prob, HashSet::new()).unwrap();

        let sol = Solution::<i64>::new(HashMap::new(), HashMap::new());
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

        let mut berths = HashMap::new();
        berths.insert(b1.id(), b1);

        let mut flex_in_prob = HashSet::new();
        flex_in_prob.insert(rx);

        let prob = Problem::new(berths, HashSet::new(), flex_in_prob).unwrap();

        let sol = Solution::<i64>::new(HashMap::new(), HashMap::new());
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

        let mut berths = HashMap::new();
        berths.insert(b1.id(), b1.clone());

        let mut flex_in_prob = HashSet::new();
        flex_in_prob.insert(r_req.clone());

        let prob = Problem::new(berths, HashSet::new(), flex_in_prob).unwrap();

        // Solution assigns the required one AND an extra one (id 11) → ExtraFlexible
        let a_req = asg_flex(&r_req, &b1, 0);
        let r_extra = req_flex(11, (0, 50), &[(1, 5)]);
        let a_extra = asg_flex(&r_extra, &b1, 10);

        let mut s_flex = HashMap::new();
        s_flex.insert(rid(10), a_req);
        s_flex.insert(rid(11), a_extra);

        let sol = Solution::new(HashMap::new(), s_flex);
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

        let mut berths = HashMap::new();
        berths.insert(b1.id(), b1);

        let mut fixed_in_prob = HashSet::new();
        fixed_in_prob.insert(af.clone());
        let mut flex_in_prob = HashSet::new();
        flex_in_prob.insert(rx.clone());

        let prob = Problem::new(berths, fixed_in_prob, flex_in_prob).unwrap();

        let mut s_fixed = HashMap::new();
        s_fixed.insert(rid(30), af);
        let mut s_flex = HashMap::new();
        s_flex.insert(rid(31), ax);

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

        let mut berths = HashMap::new();
        berths.insert(b1.id(), b1.clone());

        let mut fixed_in_prob = HashSet::new();
        fixed_in_prob.insert(af_prob);

        // No flexible requirements in the problem here
        let prob = Problem::new(berths, fixed_in_prob, HashSet::new()).unwrap();

        // Solution provides a *different* assignment for the *same request id* 40,
        // but on an unknown berth 3. This passes the "MissingFixed" check
        // (request id present) and should fail as UnknownBerth afterwards.
        let b3 = mk_berth(3, 0, 200);
        let rf_sol = req_fixed(40, (0, 100), &[(3, 5)]); // note: maps to berth 3
        let af_sol = asg_fixed(&rf_sol, &b3, 10);

        let mut s_fixed = HashMap::new();
        s_fixed.insert(rid(40), af_sol);

        let sol = Solution::new(s_fixed, HashMap::new());
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
