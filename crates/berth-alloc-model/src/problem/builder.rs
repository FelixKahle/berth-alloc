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
        BerthIdentifier, ProblemError, asg::Assignment, berth::Berth, prob::Problem, req::Request,
    },
};
use num_traits::{CheckedAdd, CheckedSub};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

#[derive(Debug, Clone)]
pub struct ProblemBuilder<T: Copy + Ord> {
    berths: HashMap<BerthIdentifier, Berth<T>>,
    fixed_assignments: HashSet<Assignment<FixedKind, T>>,
    flexible_requests: HashSet<Request<FlexibleKind, T>>,
}

impl<T: Copy + Ord> Default for ProblemBuilder<T> {
    fn default() -> Self {
        Self {
            berths: HashMap::new(),
            fixed_assignments: HashSet::new(),
            flexible_requests: HashSet::new(),
        }
    }
}

impl<T: Copy + Ord + Hash> ProblemBuilder<T> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_capacities(berths: usize, fixed: usize, flex: usize) -> Self {
        Self {
            berths: HashMap::with_capacity(berths),
            fixed_assignments: HashSet::with_capacity(fixed),
            flexible_requests: HashSet::with_capacity(flex),
        }
    }

    #[inline]
    pub fn with_berths<I>(mut self, berths: I) -> Self
    where
        I: IntoIterator<Item = Berth<T>>,
    {
        self.berths.clear();
        self.berths.extend(berths.into_iter().map(|b| (b.id(), b)));
        self
    }

    #[inline]
    pub fn with_fixed_assignments<I>(mut self, assignments: I) -> Self
    where
        I: IntoIterator<Item = Assignment<FixedKind, T>>,
    {
        self.fixed_assignments.clear();
        self.fixed_assignments.extend(assignments);
        self
    }

    #[inline]
    pub fn with_flexible_requests<I>(mut self, requests: I) -> Self
    where
        I: IntoIterator<Item = Request<FlexibleKind, T>>,
    {
        self.flexible_requests.clear();
        self.flexible_requests.extend(requests);
        self
    }

    #[inline]
    pub fn add_berth(&mut self, berth: Berth<T>) -> &mut Self {
        self.berths.insert(berth.id(), berth);
        self
    }

    #[inline]
    pub fn extend_berths<I>(&mut self, berths: I) -> &mut Self
    where
        I: IntoIterator<Item = Berth<T>>,
    {
        self.berths.extend(berths.into_iter().map(|b| (b.id(), b)));
        self
    }

    #[inline]
    pub fn add_fixed(&mut self, a: Assignment<FixedKind, T>) -> &mut Self {
        self.fixed_assignments.insert(a);
        self
    }

    #[inline]
    pub fn extend_fixed<I>(&mut self, it: I) -> &mut Self
    where
        I: IntoIterator<Item = Assignment<FixedKind, T>>,
    {
        self.fixed_assignments.extend(it);
        self
    }

    #[inline]
    pub fn add_flexible(&mut self, r: Request<FlexibleKind, T>) -> &mut Self {
        self.flexible_requests.insert(r);
        self
    }

    #[inline]
    pub fn extend_flexible<I>(&mut self, it: I) -> &mut Self
    where
        I: IntoIterator<Item = Request<FlexibleKind, T>>,
    {
        self.flexible_requests.extend(it);
        self
    }

    #[inline]
    pub fn build(self) -> Result<Problem<T>, ProblemError>
    where
        T: CheckedAdd + CheckedSub,
    {
        Problem::new(self.berths, self.fixed_assignments, self.flexible_requests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::FlexibleKind,
        problem::{berth::BerthIdentifier, req::RequestIdentifier},
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

    fn make_berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn make_request_flex(
        id: u32,
        win: (i64, i64),
        pts: &[(u32, i64)],
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, t) in pts {
            m.insert(bid(*b), td(*t));
        }
        Request::<FlexibleKind, i64>::new_flexible(rid(id), iv(win.0, win.1), m).unwrap()
    }

    #[test]
    fn build_empty() {
        let p = ProblemBuilder::<i64>::new().build().unwrap();
        assert!(p.berths().is_empty());
        assert!(p.fixed_assignments().is_empty());
        assert!(p.flexible_requests().is_empty());
    }

    #[test]
    fn with_capacities_and_add() {
        let mut b = ProblemBuilder::<i64>::with_capacities(4, 2, 8);
        b.add_berth(make_berth(1, 0, 10))
            .add_berth(make_berth(2, 5, 15));
        let r = make_request_flex(1, (0, 100), &[(1, 5), (2, 6)]);
        b.add_flexible(r);

        let p = b.build().unwrap();
        assert_eq!(p.berths().len(), 2);
        assert_eq!(p.flexible_requests().len(), 1);
        assert_eq!(p.fixed_assignments().len(), 0);
    }

    #[test]
    fn with_bulk_replacers() {
        let berths = vec![make_berth(1, 0, 10), make_berth(2, 0, 20)];
        let r1 = make_request_flex(1, (0, 30), &[(1, 5)]);
        let r2 = make_request_flex(2, (10, 40), &[(2, 7)]);

        let p = ProblemBuilder::<i64>::new()
            .with_berths(berths.clone())
            .with_flexible_requests(vec![r1.clone(), r2.clone()])
            .build()
            .unwrap();

        assert_eq!(p.berths().len(), 2);
        assert_eq!(p.flexible_requests().len(), 2);

        // Verify `with_*` replaced (not appended) by comparing against a fresh set.
        let mut hb = HashMap::new();
        hb.extend(berths.into_iter().map(|b| (b.id(), b)));
        assert_eq!(p.berths(), &hb);
    }

    #[test]
    fn extend_and_dedup() {
        let b1 = make_berth(1, 0, 10);
        let b2 = make_berth(2, 0, 20);
        let mut builder = ProblemBuilder::<i64>::new();
        builder.extend_berths(vec![b1.clone(), b2.clone()]);
        builder.extend_berths(vec![b1.clone()]); // duplicate; should be deduped by HashSet

        let p = builder.build().unwrap();
        assert_eq!(p.berths().len(), 2);
        assert!(p.berths().contains_key(&bid(1)));
        assert!(p.berths().contains_key(&bid(2)));
    }
}
