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

use std::ops::Mul;

use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};

use crate::state::registry::ledger::Ledger;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fitness {
    pub cost: Cost,
    pub unassigned_requests: usize,
}

impl Fitness {
    #[inline]
    pub fn new(cost: Cost, unassigned_requests: usize) -> Self {
        Self {
            cost,
            unassigned_requests,
        }
    }
}

impl PartialOrd for Fitness {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // First compare by unassigned requests, then by cost
        match self.unassigned_requests.cmp(&other.unassigned_requests) {
            std::cmp::Ordering::Equal => self.cost.partial_cmp(&other.cost),
            ord => Some(ord),
        }
    }
}

impl Ord for Fitness {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by unassigned requests, then by cost
        match self.unassigned_requests.cmp(&other.unassigned_requests) {
            std::cmp::Ordering::Equal => self.cost.cmp(&other.cost),
            ord => ord,
        }
    }
}

impl std::fmt::Display for Fitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fitness(cost: {}, unassigned_requests: {})",
            self.cost, self.unassigned_requests
        )
    }
}

impl<'p, T> From<&Ledger<'p, T>> for Fitness
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
{
    #[inline]
    fn from(ledger: &Ledger<'p, T>) -> Self {
        Fitness::new(ledger.cost(), ledger.unassigned_request_count())
    }
}

#[cfg(test)]
mod tests {
    use berth_alloc_model::problem::asg::AssignmentView;

    use super::*;

    #[test]
    fn test_unassigned_primary_ordering() {
        let a = Fitness::new(100, 0);
        let b = Fitness::new(10, 1);
        // a has fewer unassigned (0 < 1) so a < b
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_cost_secondary_ordering_when_unassigned_equal() {
        let a = Fitness::new(10, 1);
        let b = Fitness::new(20, 1);
        // Same unassigned; lower cost is less
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_equality_when_both_equal() {
        let a = Fitness::new(10, 1);
        let b = Fitness::new(10, 1);
        assert_eq!(a, b);
        assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn test_sorting_mixed_values() {
        let mut v = vec![
            Fitness::new(50, 2),
            Fitness::new(10, 1),
            Fitness::new(5, 3),
            Fitness::new(20, 1),
            Fitness::new(0, 0),
            Fitness::new(0, 2),
        ];
        v.sort();
        let by_tuple: Vec<_> = v.iter().map(|f| (f.unassigned_requests, f.cost)).collect();
        assert_eq!(
            by_tuple,
            vec![(0, 0), (1, 10), (1, 20), (2, 0), (2, 50), (3, 5)]
        );
    }

    #[test]
    fn test_partial_cmp_matches_cmp() {
        use std::cmp::Ordering::*;
        let cases = vec![
            (Fitness::new(1, 0), Fitness::new(2, 0), Less),
            (Fitness::new(1, 1), Fitness::new(1, 0), Greater),
            (Fitness::new(10, 2), Fitness::new(5, 2), Greater),
            (Fitness::new(7, 3), Fitness::new(7, 3), Equal),
        ];
        for (a, b, expected) in cases {
            assert_eq!(a.cmp(&b), expected);
            assert_eq!(a.partial_cmp(&b), Some(expected));
        }
    }

    #[test]
    fn test_from_ledger_empty_uses_cost_and_unassigned() {
        use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
        use berth_alloc_model::prelude::*;
        use std::collections::BTreeMap;

        fn tp(v: i64) -> TimePoint<i64> {
            TimePoint::new(v)
        }
        fn td(v: i64) -> TimeDelta<i64> {
            TimeDelta::new(v)
        }
        fn iv(a: i64, b: i64) -> TimeInterval<i64> {
            TimeInterval::new(tp(a), tp(b))
        }

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = Berth::from_windows(BerthIdentifier::new(1), [iv(0, 1000)]);
        berths.insert(b1);

        // no fixed
        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        // two flexible requests on berth 1
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        let mut pt = BTreeMap::new();
        pt.insert(BerthIdentifier::new(1), td(10));
        let r1 =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1), iv(0, 200), 1, pt.clone())
                .unwrap();
        let r2 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(2), iv(0, 200), 1, pt)
            .unwrap();
        flex.insert(r1);
        flex.insert(r2);

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let ledger = crate::state::registry::ledger::Ledger::new(&prob);

        let f = Fitness::from(&ledger);
        assert_eq!(f.cost, 0, "empty ledger has zero cost");
        assert_eq!(
            f.unassigned_requests, 2,
            "all flexible requests are unassigned"
        );
    }

    #[test]
    fn test_from_ledger_after_one_commit_updates_values() {
        use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
        use berth_alloc_model::prelude::*;
        use std::collections::BTreeMap;

        fn tp(v: i64) -> TimePoint<i64> {
            TimePoint::new(v)
        }
        fn td(v: i64) -> TimeDelta<i64> {
            TimeDelta::new(v)
        }
        fn iv(a: i64, b: i64) -> TimeInterval<i64> {
            TimeInterval::new(tp(a), tp(b))
        }

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = Berth::from_windows(BerthIdentifier::new(1), [iv(0, 1000)]);
        let b1_id = b1.id();
        berths.insert(b1.clone());

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        let mut pt = BTreeMap::new();
        pt.insert(b1_id, td(10));
        let r1 =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1), iv(0, 200), 1, pt.clone())
                .unwrap();
        let r2 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(2), iv(0, 200), 1, pt)
            .unwrap();
        flex.insert(r1.clone());
        flex.insert(r2.clone());

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let mut ledger = crate::state::registry::ledger::Ledger::new(&prob);

        // Commit one of the flexible requests
        let req1 = prob.flexible_requests().get(r1.id()).unwrap();
        let berth = prob.berths().get(b1_id).unwrap();
        ledger
            .commit_assignment(req1, berth, tp(0))
            .expect("commit should succeed");

        let f = Fitness::from(&ledger);
        assert_eq!(
            f.unassigned_requests, 1,
            "one flexible request should remain unassigned"
        );
        assert_eq!(f.cost, ledger.cost(), "fitness cost mirrors ledger cost");
        assert!(f.cost > 0, "cost should be positive after a commit");
    }

    #[test]
    fn test_from_ledger_includes_fixed_cost() {
        use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
        use berth_alloc_model::prelude::*;
        use std::collections::BTreeMap;

        fn tp(v: i64) -> TimePoint<i64> {
            TimePoint::new(v)
        }
        fn td(v: i64) -> TimeDelta<i64> {
            TimeDelta::new(v)
        }
        fn iv(a: i64, b: i64) -> TimeInterval<i64> {
            TimeInterval::new(tp(a), tp(b))
        }

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = Berth::from_windows(BerthIdentifier::new(1), [iv(0, 1000)]);
        let b1_id = b1.id();
        berths.insert(b1.clone());

        // Build one fixed assignment on b1
        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        let mut pt_fixed = BTreeMap::new();
        pt_fixed.insert(b1_id, td(10));
        let rf =
            Request::<FixedKind, i64>::new(RequestIdentifier::new(100), iv(0, 200), 2, pt_fixed)
                .unwrap();
        let af = Assignment::<FixedKind, i64>::new_fixed(rf.clone(), b1.clone(), tp(0)).unwrap();
        let expected_fixed_cost = af.cost();
        fixed.insert(af);

        // One flexible request, unassigned
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<FlexibleKind, i64>::new();
        let mut pt_flex = BTreeMap::new();
        pt_flex.insert(b1_id, td(5));
        let rflex =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(200), iv(0, 200), 1, pt_flex)
                .unwrap();
        flex.insert(rflex);

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let ledger = crate::state::registry::ledger::Ledger::new(&prob);

        let f = Fitness::from(&ledger);
        assert_eq!(
            f.cost, expected_fixed_cost,
            "fitness cost includes fixed assignment cost"
        );
        assert_eq!(
            f.unassigned_requests, 1,
            "the single flexible request is unassigned"
        );
    }
}
