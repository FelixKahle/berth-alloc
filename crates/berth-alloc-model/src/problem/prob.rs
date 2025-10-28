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

use crate::common::{FixedKind, FlexibleKind};
use crate::prelude::Berth;
use crate::problem::req::RequestView;
use crate::problem::{
    asg::{Assignment, AssignmentContainer, AssignmentView},
    berth::BerthContainer,
    err::{AssignmentOverlapError, BerthNotFoundError, ProblemError},
    req::{Request, RequestContainer},
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem<T: Copy + Ord> {
    berths: BerthContainer<T>,
    fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
    flexible_requests: RequestContainer<T, Request<FlexibleKind, T>>,
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub> Problem<T> {
    #[inline]
    pub fn new(
        berths: BerthContainer<T>,
        fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
        flexible_requests: RequestContainer<T, Request<FlexibleKind, T>>,
    ) -> Result<Self, ProblemError> {
        // Every fixed assignment must reference a known berth
        for a in fixed_assignments.iter() {
            if !berths.contains_id(a.berth_id()) {
                return Err(ProblemError::from(BerthNotFoundError::new(
                    a.request_id(),
                    a.berth_id(),
                )));
            }
        }

        // Every berth referenced by any flexible request must be known
        for r in flexible_requests.iter() {
            for &bid in r.processing_times().keys() {
                if !berths.contains_id(bid) {
                    return Err(ProblemError::from(BerthNotFoundError::new(r.id(), bid)));
                }
            }
        }

        // No fixed overlaps per-berth
        validate_no_fixed_overlaps(&fixed_assignments)?;

        Ok(Self {
            berths,
            fixed_assignments,
            flexible_requests,
        })
    }

    #[inline]
    pub fn berths(&self) -> &BerthContainer<T> {
        &self.berths
    }

    #[inline]
    pub fn iter_berths(&self) -> impl Iterator<Item = &Berth<T>> {
        self.berths.iter()
    }

    #[inline]
    pub fn fixed_assignments(
        &self,
    ) -> &AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>> {
        &self.fixed_assignments
    }

    #[inline]
    pub fn flexible_requests(&self) -> &RequestContainer<T, Request<FlexibleKind, T>> {
        &self.flexible_requests
    }

    #[inline]
    pub fn iter_flexible_requests(&self) -> impl Iterator<Item = &Request<FlexibleKind, T>> {
        self.flexible_requests.iter()
    }

    #[inline]
    pub fn iter_fixed_assignments(&self) -> impl Iterator<Item = &Assignment<FixedKind, T>> {
        self.fixed_assignments.iter()
    }

    #[inline]
    pub fn request_count(&self) -> usize {
        self.flexible_requests.len() + self.fixed_assignments.len()
    }
}

fn validate_no_fixed_overlaps<T: Copy + Ord + CheckedAdd + CheckedSub>(
    fixed: &AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
) -> Result<(), AssignmentOverlapError> {
    // Collect & sort by (berth_id, start, end)
    let mut v: Vec<&Assignment<FixedKind, T>> = fixed.iter().collect();
    if v.len() <= 1 {
        return Ok(());
    }

    v.sort_unstable_by(|a, b| {
        a.berth_id()
            .cmp(&b.berth_id())
            .then_with(|| a.start_time().cmp(&b.start_time()))
            .then_with(|| a.end_time().cmp(&b.end_time()))
    });

    // Adjacent overlaps per berth
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
    use crate::common::{FixedKind, FlexibleKind};
    use crate::problem::berth::{Berth, BerthIdentifier};
    use crate::problem::req::RequestIdentifier;
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

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
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
        req: &Request<FixedKind, i64>,
        berth: &Berth<i64>,
        start: i64,
    ) -> Assignment<FixedKind, i64> {
        Assignment::<FixedKind, i64>::new(req.clone(), berth.clone(), tp(start)).unwrap()
    }

    #[test]
    fn test_empty_everything_is_ok() {
        let berths = BerthContainer::<i64>::new();
        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let p = Problem::new(berths, fixed, flex).unwrap();
        assert_eq!(p.berths().iter().count(), 0);
        assert_eq!(p.fixed_assignments().iter().count(), 0);
        assert_eq!(p.flexible_requests().iter().count(), 0);
    }

    #[test]
    fn test_no_fixed_assignments_ok_even_with_flex() {
        let b1 = berth(1, 0, 1_000);
        let b2 = berth(2, 0, 1_000);

        let mut berths = BerthContainer::new();
        berths.insert(b1);
        berths.insert(b2);

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let f1 = req_flex(1, (0, 100), &[(1, 10)]);
        let f2 = req_flex(2, (50, 200), &[(2, 10)]);
        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(f1);
        flex.insert(f2);

        let p = Problem::new(berths, fixed, flex).unwrap();
        assert_eq!(p.flexible_requests().iter().count(), 2);
    }

    #[test]
    fn test_fixed_assignment_refers_to_unknown_berth_is_rejected() {
        // Create an assignment on berth(1), but do NOT include that berth in the set.
        let missing_berth = berth(1, 0, 100);
        let present_berth = berth(2, 0, 100);

        // req needs processing time for berth 1 (compatible with the assignment)
        let r = req_fixed(10, (0, 100), &[(1, 10)]);

        let a = asg_fixed(&r, &missing_berth, 0);

        let mut berths = BerthContainer::new();
        berths.insert(present_berth); // berth 1 intentionally omitted

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let err = Problem::new(berths, fixed, flex).unwrap_err();
        match err {
            ProblemError::BerthNotFound(e) => {
                assert_eq!(e.request(), rid(10));
                assert_eq!(e.requested_berth(), bid(1));
            }
            other => panic!("expected BerthNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_flexible_request_with_unknown_berth_is_rejected() {
        // Present only berth(2)
        let mut berths = BerthContainer::new();
        let b = berth(2, 0, 100);
        berths.insert(b);

        // Flex request that references berth(1) in its processing-time map.
        let r = req_flex(7, (0, 100), &[(1, 5)]);
        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(r);

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let err = Problem::new(berths, fixed, flex).unwrap_err();
        match err {
            ProblemError::BerthNotFound(e) => {
                assert_eq!(e.request(), rid(7));
                assert_eq!(e.requested_berth(), bid(1));
            }
            other => panic!("expected BerthNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_touching_intervals_on_same_berth_are_ok() {
        // [0,10) and [10,20) should NOT overlap
        let b1 = berth(1, 0, 1_000);

        let ra = req_fixed(20, (0, 1000), &[(1, 10)]);
        let rb = req_fixed(21, (0, 1000), &[(1, 10)]);

        let a = asg_fixed(&ra, &b1, 0);
        let b = asg_fixed(&rb, &b1, 10);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a);
        fixed.insert(b);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let _ = Problem::new(berths, fixed, flex).unwrap();
    }

    #[test]
    fn test_disjoint_intervals_on_same_berth_are_ok() {
        let b1 = berth(1, 0, 1_000);

        let ra = req_fixed(30, (0, 1000), &[(1, 5)]);
        let rb = req_fixed(31, (0, 1000), &[(1, 7)]);

        let a = asg_fixed(&ra, &b1, 0); // [0,5)
        let b = asg_fixed(&rb, &b1, 10); // [10,17)

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a);
        fixed.insert(b);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let _ = Problem::new(berths, fixed, flex).unwrap();
    }

    #[test]
    fn test_overlapping_intervals_on_same_berth_are_rejected() {
        let b1 = berth(1, 0, 1_000);

        let ra = req_fixed(40, (0, 1000), &[(1, 10)]);
        let rb = req_fixed(41, (0, 1000), &[(1, 10)]);

        // [5,15) overlaps [10,20)
        let a = asg_fixed(&ra, &b1, 5);
        let b = asg_fixed(&rb, &b1, 10);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a);
        fixed.insert(b);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let err = Problem::new(berths, fixed, flex).unwrap_err();
        match err {
            ProblemError::FixedAssignmentOverlap(e) => {
                let ids = [e.first(), e.second()];
                assert!(ids.contains(&rid(40)) && ids.contains(&rid(41)));
            }
            other => panic!("expected FixedAssignmentOverlap, got {other:?}"),
        }
    }

    #[test]
    fn test_overlapping_on_different_berths_are_ok() {
        let b1 = berth(1, 0, 1_000);
        let b2 = berth(2, 0, 1_000);

        let ra = req_fixed(50, (0, 1000), &[(1, 20)]);
        let rb = req_fixed(51, (0, 1000), &[(2, 20)]);

        // same time, different berths
        let a = asg_fixed(&ra, &b1, 100); // [100,120)
        let b = asg_fixed(&rb, &b2, 100); // [100,120)

        let mut berths = BerthContainer::new();
        berths.insert(b1);
        berths.insert(b2);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a);
        fixed.insert(b);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let _ = Problem::new(berths, fixed, flex).unwrap();
    }

    #[test]
    fn test_unordered_input_still_detects_overlap() {
        let b1 = berth(1, 0, 1_000);

        let r1 = req_fixed(60, (0, 1000), &[(1, 10)]); // [50,60)
        let r2 = req_fixed(61, (0, 1000), &[(1, 15)]); // [58,73) overlap
        let r3 = req_fixed(62, (0, 1000), &[(1, 5)]); // [ 0, 5) disjoint

        let a1 = asg_fixed(&r1, &b1, 50);
        let a2 = asg_fixed(&r2, &b1, 58);
        let a3 = asg_fixed(&r3, &b1, 0);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        // Intentionally insert in a jumbled order:
        fixed.insert(a2);
        fixed.insert(a3);
        fixed.insert(a1);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let err = Problem::new(berths, fixed, flex).unwrap_err();
        match err {
            ProblemError::FixedAssignmentOverlap(e) => {
                let ids = [e.first(), e.second()];
                assert!(ids.contains(&rid(60)) && ids.contains(&rid(61)));
            }
            other => panic!("expected FixedAssignmentOverlap, got {other:?}"),
        }
    }

    #[test]
    fn test_many_non_overlapping_chain_ok() {
        let b1 = berth(1, 0, 10_000);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        // 10 sequential non-overlapping fixed assignments, each len=10, spaced by 10
        for i in 0..10 {
            let rid_u = 100 + i;
            let start = i * 10;
            let r = req_fixed(rid_u, (0, 10_000), &[(1, 10)]);
            fixed.insert(asg_fixed(&r, &b1, start as i64));
        }

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let _ = Problem::new(berths, fixed, flex).unwrap();
    }

    #[test]
    fn test_zero_length_processing_does_not_overlap() {
        // Two zero-length jobs at the same instant: [t,t) and [t,t) — OK
        let b1 = berth(1, 0, 1_000);

        let r1 = req_fixed(200, (0, 1000), &[(1, 0)]);
        let r2 = req_fixed(201, (0, 1000), &[(1, 0)]);

        let a1 = asg_fixed(&r1, &b1, 100); // [100,100)
        let a2 = asg_fixed(&r2, &b1, 100); // [100,100)

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a1);
        fixed.insert(a2);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let _ = Problem::new(berths, fixed, flex).unwrap();
    }

    #[test]
    fn test_zero_length_and_nonzero_at_same_start_are_ok() {
        let b1 = berth(1, 0, 1_000);

        let r0 = req_fixed(210, (0, 1000), &[(1, 0)]); // [100,100)
        let r1 = req_fixed(211, (0, 1000), &[(1, 10)]); // [100,110)

        let a0 = asg_fixed(&r0, &b1, 100);
        let a1 = asg_fixed(&r1, &b1, 100);

        let mut berths = BerthContainer::new();
        berths.insert(b1);

        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(a0);
        fixed.insert(a1);

        let flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();

        let _ = Problem::new(berths, fixed, flex).unwrap();
    }
}
