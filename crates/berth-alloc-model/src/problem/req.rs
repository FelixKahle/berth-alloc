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
    common::{FixedKind, FlexibleKind, Identifier, IdentifierMarkerName, Kind},
    problem::{
        berth::BerthIdentifier,
        err::{EmptyBerthMapError, NoFeasibleAssignmentError, RequestError},
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval};
use num_traits::CheckedSub;
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    hash::Hasher,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestIdentifierMarker;

impl IdentifierMarkerName for RequestIdentifierMarker {
    const NAME: &'static str = "RequestId";
}

pub type RequestIdentifier = Identifier<u32, RequestIdentifierMarker>;

#[derive(Debug, Clone)]
pub struct Request<K: Kind, T: Ord + Copy> {
    id: RequestIdentifier,
    feasible_window: TimeInterval<T>,
    processing_times: BTreeMap<BerthIdentifier, TimeDelta<T>>,
    weight: Cost,
    _phantom: std::marker::PhantomData<K>,
}

impl<K: Kind, T: Ord + Copy> PartialEq for Request<K, T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<K: Kind, T: Ord + Copy> Eq for Request<K, T> {}

impl<K: Kind, T: Ord + Copy> std::hash::Hash for Request<K, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Ord + Copy + CheckedSub> Request<FixedKind, T> {
    #[inline]
    pub fn new_fixed(
        id: RequestIdentifier,
        feasible_window: TimeInterval<T>,
        weight: Cost,
        processing_times: BTreeMap<BerthIdentifier, TimeDelta<T>>,
    ) -> Result<Self, RequestError> {
        Request::<FixedKind, T>::new(id, feasible_window, weight, processing_times)
    }
}

impl<T: Ord + Copy + CheckedSub> Request<FlexibleKind, T> {
    #[inline]
    pub fn new_flexible(
        id: RequestIdentifier,
        feasible_window: TimeInterval<T>,
        weight: Cost,
        processing_times: BTreeMap<BerthIdentifier, TimeDelta<T>>,
    ) -> Result<Self, RequestError> {
        Request::<FlexibleKind, T>::new(id, feasible_window, weight, processing_times)
    }
}

impl<K: Kind, T: Ord + Copy + CheckedSub> Request<K, T> {
    #[inline]
    pub fn new(
        id: RequestIdentifier,
        feasible_window: TimeInterval<T>,
        weight: Cost,
        mut processing_times: BTreeMap<BerthIdentifier, TimeDelta<T>>,
    ) -> Result<Self, RequestError> {
        if processing_times.is_empty() {
            return Err(EmptyBerthMapError)?;
        }

        let cap = feasible_window.length();
        processing_times.retain(|_, dt| *dt <= cap);

        if processing_times.is_empty() {
            return Err(NoFeasibleAssignmentError::new(id))?;
        }

        Ok(Self {
            id,
            feasible_window,
            weight,
            processing_times,
            _phantom: std::marker::PhantomData,
        })
    }

    #[inline]
    pub fn id(&self) -> RequestIdentifier {
        self.id
    }

    #[inline]
    pub fn feasible_window(&self) -> TimeInterval<T> {
        self.feasible_window
    }

    #[inline]
    pub fn weight(&self) -> Cost {
        self.weight
    }

    #[inline]
    pub fn processing_times(&self) -> &BTreeMap<BerthIdentifier, TimeDelta<T>> {
        &self.processing_times
    }

    #[inline]
    pub fn iter_allowed_berths_ids(&self) -> impl Iterator<Item = BerthIdentifier> + '_ {
        self.processing_times.keys().copied()
    }

    #[inline]
    pub fn processing_time_for(&self, berth_id: BerthIdentifier) -> Option<TimeDelta<T>> {
        self.processing_times.get(&berth_id).copied()
    }

    #[inline]
    pub fn is_berth_feasible(&self, berth_id: BerthIdentifier) -> bool {
        self.processing_times.contains_key(&berth_id)
    }
}

impl<K: Kind, T: Ord + Copy + Display> std::fmt::Display for Request<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let processing_times_str: Vec<String> = self
            .processing_times
            .iter()
            .map(|(berth_id, time)| format!("{}: {}", berth_id, time))
            .collect();

        write!(
            f,
            "{}-Request: Id: {}, Feasible Window {}, Processing Times {}",
            K::NAME,
            self.id,
            self.feasible_window,
            processing_times_str.join(", ")
        )
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct RequestContainer<K: Kind, T: Ord + Copy>(HashMap<RequestIdentifier, Request<K, T>>);

impl<K: Kind, T: Copy + Ord> Default for RequestContainer<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Kind, T: Copy + Ord> RequestContainer<K, T> {
    #[inline]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(HashMap::with_capacity(cap))
    }

    #[inline]
    pub fn insert(&mut self, request: Request<K, T>) -> Option<Request<K, T>>
    where
        T: CheckedSub,
    {
        self.0.insert(request.id(), request)
    }

    #[inline]
    pub fn remove(&mut self, id: RequestIdentifier) -> Option<Request<K, T>> {
        self.0.remove(&id)
    }

    #[inline]
    pub fn contains_id(&self, id: RequestIdentifier) -> bool {
        self.0.contains_key(&id)
    }

    #[inline]
    pub fn contains_request(&self, request: &Request<K, T>) -> bool
    where
        T: CheckedSub,
    {
        let id = request.id();
        self.0.contains_key(&id)
    }

    #[inline]
    pub fn get(&self, id: RequestIdentifier) -> Option<&Request<K, T>> {
        self.0.get(&id)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Request<K, T>> {
        self.0.values()
    }
}

impl<K: Kind, T: Copy + Ord + CheckedSub> FromIterator<Request<K, T>> for RequestContainer<K, T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Request<K, T>>>(iter: I) -> Self {
        let mut c = Self::new();
        for r in iter {
            c.insert(r);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_new_fixed_ok_and_accessors() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        pt.insert(bid(3), td(12));

        let w = iv(10, 50); // capacity = 40, both entries stay
        let r = Request::<FixedKind, i64>::new_fixed(RequestIdentifier::new(7), w, 1, pt.clone())
            .expect("non-empty map must be ok");

        assert_eq!(r.id(), RequestIdentifier::new(7));
        assert_eq!(r.feasible_window(), w);
        assert_eq!(r.processing_times().len(), 2);
        assert_eq!(r.processing_time_for(bid(1)), Some(td(5)));
        assert_eq!(r.processing_time_for(bid(2)), None);
        assert_eq!(r.processing_time_for(bid(3)), Some(td(12)));
    }

    #[test]
    fn test_new_flexible_ok_and_accessors() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(2), td(9));

        let w = iv(-5, 5); // capacity = 10
        let r =
            Request::<FlexibleKind, i64>::new_flexible(RequestIdentifier::new(9), w, 1, pt.clone())
                .expect("non-empty map must be ok");
        assert_eq!(r.id(), RequestIdentifier::new(9));
        assert_eq!(r.feasible_window(), w);
        assert_eq!(r.processing_times(), &pt);
        assert_eq!(r.processing_time_for(bid(2)), Some(td(9)));
    }

    #[test]
    fn test_empty_map_rejected_fixed() {
        let empty: BTreeMap<BerthIdentifier, TimeDelta<i64>> = BTreeMap::new();
        let err =
            Request::<FixedKind, i64>::new_fixed(RequestIdentifier::new(1), iv(0, 10), 1, empty)
                .expect_err("empty map must be rejected");
        assert_eq!(err, RequestError::EmptyBerthMap(EmptyBerthMapError));
    }

    #[test]
    fn test_empty_map_rejected_flexible() {
        let empty: BTreeMap<BerthIdentifier, TimeDelta<i64>> = BTreeMap::new();
        let err = Request::<FlexibleKind, i64>::new_flexible(
            RequestIdentifier::new(2),
            iv(0, 10),
            1,
            empty,
        )
        .expect_err("empty map must be rejected");
        assert_eq!(err, RequestError::EmptyBerthMap(EmptyBerthMapError));
    }

    #[test]
    fn test_pruning_keeps_all_when_within_capacity() {
        // capacity = 40; both <= cap
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(12));
        pt.insert(bid(2), td(40));
        let w = iv(10, 50);
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(10), w, 1, pt.clone())
            .expect("both should remain");
        assert_eq!(r.processing_times().len(), 2);
        assert_eq!(r.processing_time_for(bid(1)), Some(td(12)));
        assert_eq!(r.processing_time_for(bid(2)), Some(td(40)));
    }

    #[test]
    fn test_pruning_drops_only_excessive_entries() {
        // capacity = 20; 25 gets pruned, 15 remains
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(25)); // > 20
        pt.insert(bid(2), td(15)); // <= 20
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(11), iv(0, 20), 1, pt)
            .expect("one feasible berth remains");
        let keys: Vec<_> = r.processing_times().keys().copied().collect();
        assert_eq!(keys, vec![bid(2)]);
        assert_eq!(r.processing_time_for(bid(1)), None);
        assert_eq!(r.processing_time_for(bid(2)), Some(td(15)));
    }

    #[test]
    fn test_pruning_exact_fit_is_allowed() {
        // exact fit (== cap) stays
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(12), iv(0, 10), 1, pt)
            .expect("exact fit should be accepted");
        assert_eq!(r.processing_time_for(bid(1)), Some(td(10)));
    }

    #[test]
    fn test_pruning_all_removed_yields_no_feasible_assignment() {
        // capacity = 20; all entries > cap -> error
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(30));
        pt.insert(bid(2), td(40));
        let err = Request::<FixedKind, i64>::new(RequestIdentifier::new(13), iv(0, 20), 1, pt)
            .expect_err("no berth fits");
        assert_eq!(
            err,
            RequestError::NoFeasibleAssignment(NoFeasibleAssignmentError::new(
                RequestIdentifier::new(13)
            ))
        );
    }

    #[test]
    fn test_zero_length_window_with_positive_processing_yields_no_feasible_assignment() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(1)); // > 0
        let err = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(14), iv(50, 50), 1, pt)
            .expect_err("no room for positive processing time in zero-length window");
        assert_eq!(
            err,
            RequestError::NoFeasibleAssignment(NoFeasibleAssignmentError::new(
                RequestIdentifier::new(14)
            ))
        );
    }

    #[test]
    fn test_zero_length_window_with_zero_processing_is_allowed() {
        // If your TimeDelta permits zero, we should keep it because 0 <= 0
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(0));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(15), iv(100, 100), 1, pt)
            .expect("zero processing fits zero-length window");
        assert_eq!(r.processing_times().len(), 1);
        assert_eq!(r.processing_time_for(bid(1)), Some(td(0)));
    }

    #[test]
    fn test_processing_times_are_sorted_by_berth_id() {
        // Out-of-order insertions -> BTreeMap ensures ordering
        let mut pt = BTreeMap::new();
        pt.insert(bid(10), td(1));
        pt.insert(bid(3), td(2));
        pt.insert(bid(7), td(3));
        let r =
            Request::<FixedKind, i64>::new(RequestIdentifier::new(16), iv(0, 100), 1, pt).unwrap();
        let keys: Vec<_> = r.processing_times().keys().copied().collect();
        assert_eq!(keys, vec![bid(3), bid(7), bid(10)]);
    }

    #[test]
    fn test_generic_constructor_matches_specialized_constructors() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(3));
        let w = iv(0, 100);

        // Flexible
        let a = Request::<FlexibleKind, i64>::new_flexible(
            RequestIdentifier::new(20),
            w,
            1,
            pt.clone(),
        )
        .unwrap();
        let b = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(20), w, 1, pt.clone())
            .unwrap();
        assert_eq!(a, b);

        // Fixed
        let c = Request::<FixedKind, i64>::new_fixed(RequestIdentifier::new(21), w, 1, pt.clone())
            .unwrap();
        let d =
            Request::<FixedKind, i64>::new(RequestIdentifier::new(21), w, 1, pt.clone()).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn test_clone_and_eq_work() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(7));
        let r1 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(30), iv(-1, 9), 1, pt)
            .unwrap();
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    #[test]
    fn processing_time_for_returns_none_for_forbidden_berth() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(5), td(5)); // <= cap (10) so it survives
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(31), iv(0, 10), 1, pt)
            .expect("should construct; at least one feasible berth remains");
        assert_eq!(r.processing_time_for(bid(5)), Some(td(5)));
        assert_eq!(r.processing_time_for(bid(6)), None); // forbidden/absent
    }

    #[test]
    fn processing_time_over_capacity_yields_no_feasible_assignment() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(5), td(20)); // > cap (10) → pruned → empty
        let err = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(31), iv(0, 10), 1, pt)
            .expect_err("all entries pruned -> no feasible assignment");
        assert_eq!(
            err,
            RequestError::NoFeasibleAssignment(NoFeasibleAssignmentError::new(
                RequestIdentifier::new(31)
            ))
        );
    }

    #[test]
    fn test_display_contains_kind_id_window_and_times() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        pt.insert(bid(2), td(8));

        let w = iv(100, 200);
        let r = Request::<FixedKind, i64>::new_fixed(RequestIdentifier::new(42), w, 1, pt).unwrap();
        let s = format!("{r}");

        // Don’t assert exact formatting; check critical parts are present.
        assert!(s.contains("Fixed"));
        assert!(s.contains("RequestId(42)"));
        assert!(s.contains(&format!("{}", w)));
        assert!(s.contains("Processing Times"));
        assert!(s.contains(&format!("{}", bid(1))));
        assert!(s.contains(&format!("{}", bid(2))));
    }
}
