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
    common::{FixedKind, FlexibleKind, Kind},
    problem::{
        berth::{Berth, BerthIdentifier},
        err::{
            AssignmentStartsBeforeFeasibleWindowError, AssignmentEndsAfterFeasibleWindowError,
            AssignmentError, IncompatibleBerthError,
        },
        req::{Request, RequestIdentifier, RequestView},
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    ops::Mul,
};

pub trait AssignmentView<K, T>
where
    K: Kind,
    T: Ord + Copy,
{
    fn request(&self) -> &Request<K, T>;

    fn request_id(&self) -> RequestIdentifier
    where
        T: CheckedAdd + CheckedSub,
    {
        self.request().id()
    }

    fn berth(&self) -> &Berth<T>;

    fn berth_id(&self) -> BerthIdentifier {
        self.berth().id()
    }

    fn start_time(&self) -> TimePoint<T>;

    fn processing_time(&self) -> TimeDelta<T>;

    fn end_time(&self) -> TimePoint<T>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.start_time() + self.request().processing_time_for(self.berth_id()).expect(
            "The processing time for the assigned berth must be defined, as the assignment is only valid if so.",
        )
    }

    fn interval(&self) -> TimeInterval<T>
    where
        T: CheckedAdd + CheckedSub,
    {
        TimeInterval::new(self.start_time(), self.end_time())
    }

    #[inline]
    fn waiting_time(&self) -> TimeDelta<T>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.start_time() - self.request().feasible_window().start()
    }

    #[inline]
    fn turnaround_time(&self) -> TimeDelta<T>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.processing_time() + self.waiting_time()
    }

    #[inline]
    fn cost(&self) -> Cost
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let weight = self.request().weight();
        self.turnaround_time().value().into() * weight
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assignment<K: Kind, T: Ord + Copy> {
    request: Request<K, T>,
    berth: Berth<T>,
    processing_time: TimeDelta<T>,
    start_time: TimePoint<T>,
    _phantom: std::marker::PhantomData<K>,
}

impl<T: Ord + Copy + CheckedSub + CheckedAdd> Assignment<FixedKind, T> {
    #[inline]
    pub fn new_fixed(
        request: Request<FixedKind, T>,
        berth: Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<Self, AssignmentError<T>> {
        Assignment::<FixedKind, T>::new(request, berth, start_time)
    }
}

impl<T: Ord + Copy + CheckedSub + CheckedAdd> Assignment<FlexibleKind, T> {
    #[inline]
    pub fn new_flexible(
        request: Request<FlexibleKind, T>,
        berth: Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<Self, AssignmentError<T>> {
        Assignment::<FlexibleKind, T>::new(request, berth, start_time)
    }
}

impl<K: Kind, T: Ord + CheckedSub + Copy + CheckedAdd> Assignment<K, T> {
    #[inline]
    pub fn new(
        request: Request<K, T>,
        berth: Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<Self, AssignmentError<T>> {
        let Some(processing_time) = request.processing_time_for(berth.id()) else {
            return Err(AssignmentError::Incompatible(IncompatibleBerthError::new(
                request.id(),
                berth.id(),
            )));
        };

        let window = request.feasible_window();
        if start_time < window.start() {
            return Err(AssignmentError::AssignmentStartsBeforeFeasibleWindow(
                AssignmentStartsBeforeFeasibleWindowError::new(
                    request.id(),
                    window.start(),
                    start_time,
                ),
            ));
        }

        let end_time = start_time + processing_time;
        if end_time > window.end() {
            return Err(AssignmentError::AssignmentEndsAfterFeasibleWindow(
                AssignmentEndsAfterFeasibleWindowError::new(request.id(), end_time, window),
            ));
        }

        Ok(Self {
            request,
            berth,
            start_time,
            processing_time,
            _phantom: std::marker::PhantomData,
        })
    }

    #[inline]
    pub fn to_ref<'a>(&'a self) -> AssignmentRef<'a, 'a, K, T> {
        AssignmentRef {
            request: &self.request,
            berth: &self.berth,
            start_time: self.start_time,
            processing_time: self.processing_time,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<K: Kind, T: Ord + Copy> AssignmentView<K, T> for Assignment<K, T> {
    fn request(&self) -> &Request<K, T> {
        &self.request
    }

    fn berth(&self) -> &Berth<T> {
        &self.berth
    }

    fn start_time(&self) -> TimePoint<T> {
        self.start_time
    }

    fn processing_time(&self) -> TimeDelta<T> {
        self.processing_time
    }
}

impl<K: Kind, T: Ord + Copy + CheckedSub + Display> std::fmt::Display for Assignment<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assignment<{}>(Request: {}, Berth: {}, StartTime: {})",
            K::NAME,
            self.request.id(),
            self.berth.id(),
            self.start_time
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssignmentRef<'r, 'b, K: Kind, T: Ord + Copy> {
    request: &'r Request<K, T>,
    berth: &'b Berth<T>,
    start_time: TimePoint<T>,
    processing_time: TimeDelta<T>,
    _phantom: std::marker::PhantomData<K>,
}

impl<'r, 'b, T: Ord + Copy + CheckedSub + CheckedAdd> AssignmentRef<'r, 'b, FixedKind, T> {
    #[inline]
    pub fn new_fixed(
        request: &'r Request<FixedKind, T>,
        berth: &'b Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<Self, AssignmentError<T>> {
        AssignmentRef::<FixedKind, T>::new(request, berth, start_time)
    }
}

impl<'r, 'b, T: Ord + Copy + CheckedSub + CheckedAdd> AssignmentRef<'r, 'b, FlexibleKind, T> {
    #[inline]
    pub fn new_flexible(
        request: &'r Request<FlexibleKind, T>,
        berth: &'b Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<Self, AssignmentError<T>> {
        AssignmentRef::<FlexibleKind, T>::new(request, berth, start_time)
    }
}

impl<'r, 'b, K: Kind, T: Ord + Copy + CheckedSub + CheckedAdd> AssignmentRef<'r, 'b, K, T> {
    #[inline]
    pub fn new(
        request: &'r Request<K, T>,
        berth: &'b Berth<T>,
        start_time: TimePoint<T>,
    ) -> Result<Self, AssignmentError<T>> {
        let Some(processing_time) = request.processing_time_for(berth.id()) else {
            return Err(AssignmentError::Incompatible(IncompatibleBerthError::new(
                request.id(),
                berth.id(),
            )));
        };

        let window = request.feasible_window();
        if start_time < window.start() {
            return Err(AssignmentError::AssignmentStartsBeforeFeasibleWindow(
                AssignmentStartsBeforeFeasibleWindowError::new(
                    request.id(),
                    window.start(),
                    start_time,
                ),
            ));
        }

        let end_time = start_time + processing_time;
        if end_time > window.end() {
            return Err(AssignmentError::AssignmentEndsAfterFeasibleWindow(
                AssignmentEndsAfterFeasibleWindowError::new(request.id(), end_time, window),
            ));
        }

        Ok(Self {
            request,
            berth,
            start_time,
            processing_time,
            _phantom: std::marker::PhantomData,
        })
    }

    #[inline]
    pub fn berth(&self) -> &'b Berth<T> {
        self.berth
    }

    #[inline]
    pub fn to_owned(&self) -> Assignment<K, T> {
        Assignment {
            request: self.request.clone(),
            berth: self.berth.clone(),
            start_time: self.start_time,
            processing_time: self.processing_time,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn into_owned(self) -> Assignment<K, T> {
        Assignment {
            request: self.request.clone(),
            berth: self.berth.clone(),
            start_time: self.start_time,
            processing_time: self.processing_time,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'r, 'b, K: Kind, T: Ord + Copy> AssignmentView<K, T> for AssignmentRef<'r, 'b, K, T> {
    fn request(&self) -> &Request<K, T> {
        self.request
    }

    fn berth(&self) -> &'b Berth<T> {
        self.berth
    }

    fn start_time(&self) -> TimePoint<T> {
        self.start_time
    }

    fn processing_time(&self) -> TimeDelta<T> {
        self.processing_time
    }
}

impl<'r, 'b, K: Kind, T: Ord + Copy + CheckedSub + Display> std::fmt::Display
    for AssignmentRef<'r, 'b, K, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AssignmentRef<{}>(Request: {}, Berth: {}, StartTime: {})",
            K::NAME,
            self.request.id(),
            self.berth.id(),
            self.start_time
        )
    }
}

#[derive(Debug, Clone)]
pub struct AssignmentContainer<K, T, V>
where
    K: Kind,
    T: Copy + Ord,
    V: AssignmentView<K, T>,
{
    inner: HashMap<RequestIdentifier, V>,
    _phantom: std::marker::PhantomData<(K, T)>,
}

impl<K, T, V> Default for AssignmentContainer<K, T, V>
where
    K: Kind,
    T: Copy + Ord,
    V: AssignmentView<K, T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, T, V> AssignmentContainer<K, T, V>
where
    K: Kind,
    T: Copy + Ord,
    V: AssignmentView<K, T>,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn insert(&mut self, value: V) -> Option<V>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.inner.insert(value.request_id(), value)
    }

    #[inline]
    pub fn remove(&mut self, rid: RequestIdentifier) -> Option<V> {
        self.inner.remove(&rid)
    }

    pub fn remove_assignment(&mut self, assignment: &V) -> Option<V>
    where
        T: CheckedAdd + CheckedSub,
    {
        let rid = assignment.request_id();
        self.remove(rid)
    }

    #[inline]
    pub fn get(&self, rid: RequestIdentifier) -> Option<&V> {
        self.inner.get(&rid)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &V> {
        self.inner.values()
    }

    #[inline]
    pub fn contains_id(&self, rid: RequestIdentifier) -> bool {
        self.inner.contains_key(&rid)
    }

    #[inline]
    pub fn contains_assignment(&self, assignment: &V) -> bool
    where
        T: CheckedAdd + CheckedSub,
    {
        let rid = assignment.request_id();
        self.contains_id(rid)
    }
}

impl<K, T, V> FromIterator<V> for AssignmentContainer<K, T, V>
where
    K: Kind,
    T: Copy + Ord + CheckedAdd + CheckedSub,
    V: AssignmentView<K, T>,
{
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        let mut container = AssignmentContainer::new();
        for v in iter {
            container.insert(v);
        }
        container
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnyAssignment<T: Ord + Copy> {
    Fixed(Assignment<FixedKind, T>),
    Flexible(Assignment<FlexibleKind, T>),
}

impl<T: Copy + Ord + CheckedSub + std::fmt::Display> std::fmt::Display for AnyAssignment<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyAssignment::Fixed(a) => write!(f, "{}", a),
            AnyAssignment::Flexible(a) => write!(f, "{}", a),
        }
    }
}

impl<T: Copy + Ord> From<Assignment<FixedKind, T>> for AnyAssignment<T> {
    fn from(a: Assignment<FixedKind, T>) -> Self {
        AnyAssignment::Fixed(a)
    }
}

impl<T: Copy + Ord> From<Assignment<FlexibleKind, T>> for AnyAssignment<T> {
    fn from(a: Assignment<FlexibleKind, T>) -> Self {
        AnyAssignment::Flexible(a)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyAssignmentRef<'r, 'b, T: Ord + Copy> {
    Fixed(AssignmentRef<'r, 'b, FixedKind, T>),
    Flexible(AssignmentRef<'r, 'b, FlexibleKind, T>),
}

impl<T: Copy + Ord + CheckedSub + std::fmt::Display> std::fmt::Display
    for AnyAssignmentRef<'_, '_, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyAssignmentRef::Fixed(a) => write!(f, "{}", a),
            AnyAssignmentRef::Flexible(a) => write!(f, "{}", a),
        }
    }
}

impl<'r, 'b, T: Copy + Ord> From<AssignmentRef<'r, 'b, FixedKind, T>>
    for AnyAssignmentRef<'r, 'b, T>
{
    fn from(a: AssignmentRef<'r, 'b, FixedKind, T>) -> Self {
        AnyAssignmentRef::Fixed(a)
    }
}

impl<'r, 'b, T: Copy + Ord> From<AssignmentRef<'r, 'b, FlexibleKind, T>>
    for AnyAssignmentRef<'r, 'b, T>
{
    fn from(a: AssignmentRef<'r, 'b, FlexibleKind, T>) -> Self {
        AnyAssignmentRef::Flexible(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use std::collections::BTreeMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

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
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }

    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn mk_berth_single(id: usize, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn mk_berth_multi() -> Berth<i64> {
        // Availability: [0, 20) ∪ [25, 50)
        Berth::from_windows(
            bid(2),
            vec![
                TimeInterval::new(tp(0), tp(20)),
                TimeInterval::new(tp(25), tp(50)),
            ],
        )
    }

    fn mk_req_fixed_with(
        id: usize,
        win: (i64, i64),
        pt: &[(usize, i64)],
    ) -> Request<FixedKind, i64> {
        let mut map = BTreeMap::new();
        for (b, d) in pt {
            map.insert(bid(*b), td(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(win.0, win.1), 1, map).unwrap()
    }

    fn mk_req_flexible_with(
        id: usize,
        win: (i64, i64),
        pt: &[(usize, i64)],
    ) -> Request<FlexibleKind, i64> {
        let mut map = BTreeMap::new();
        for (b, d) in pt {
            map.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(win.0, win.1), 1, map).unwrap()
    }

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn test_new_fixed_ok_and_accessors() {
        let req = mk_req_fixed_with(100, (0, 100), &[(2, 7)]);
        let berth = mk_berth_multi();
        let start = tp(12);

        let a = Assignment::<FixedKind, i64>::new_fixed(req.clone(), berth.clone(), start).unwrap();

        assert_eq!(a.request().id(), req.id());
        assert_eq!(a.request_id(), req.id());
        assert_eq!(a.berth().id(), berth.id());
        assert_eq!(a.berth_id(), berth.id());
        assert_eq!(a.start_time(), start);
    }

    #[test]
    fn test_new_flexible_ok_and_accessors() {
        let req = mk_req_flexible_with(200, (0, 60), &[(2, 9)]);
        let berth = mk_berth_multi();
        let start = tp(25);

        let a = Assignment::<FlexibleKind, i64>::new_flexible(req.clone(), berth.clone(), start)
            .unwrap();

        assert_eq!(a.request().id(), req.id());
        assert_eq!(a.berth().id(), berth.id());
        assert_eq!(a.start_time(), start);
    }

    #[test]
    fn test_generic_new_matches_specific_constructors() {
        // Fixed
        let req_f = mk_req_fixed_with(1, (0, 100), &[(2, 5)]);
        let berth_f = mk_berth_multi();
        let af = Assignment::<FixedKind, i64>::new_fixed(req_f.clone(), berth_f.clone(), tp(15))
            .unwrap();
        let ag = Assignment::<FixedKind, i64>::new(req_f, berth_f, tp(15)).unwrap();
        assert_eq!(af, ag);

        // Flexible
        let req_x = mk_req_flexible_with(2, (0, 100), &[(2, 6)]);
        let berth_x = mk_berth_multi();
        let bf =
            Assignment::<FlexibleKind, i64>::new_flexible(req_x.clone(), berth_x.clone(), tp(30))
                .unwrap();
        let bg = Assignment::<FlexibleKind, i64>::new(req_x, berth_x, tp(30)).unwrap();
        assert_eq!(bf, bg);
    }

    #[test]
    fn test_end_time_is_start_plus_processing_time() {
        // processing time on berth 1 is 10
        let req = mk_req_fixed_with(3, (0, 100), &[(1, 10)]);
        let berth = mk_berth_single(1, 0, 1000);
        let start = tp(40);
        let a = Assignment::<FixedKind, i64>::new_fixed(req, berth, start).unwrap();
        assert_eq!(a.end_time(), tp(50));
    }

    #[test]
    fn test_assignment_ref_new_ok_and_to_owned_roundtrip() {
        let req = mk_req_fixed_with(4, (0, 100), &[(2, 7)]);
        let berth = mk_berth_multi();
        let start = tp(30);

        let r = AssignmentRef::<FixedKind, i64>::new(&req, &berth, start).unwrap();
        assert_eq!(r.request().id(), req.id());
        assert_eq!(r.berth().id(), berth.id());
        assert_eq!(r.start_time(), start);

        let owned = r.to_owned();
        assert_eq!(owned.request_id(), req.id());
        assert_eq!(owned.berth_id(), berth.id());
        assert_eq!(owned.start_time(), start);
    }

    #[test]
    fn test_assignment_ref_specific_constructors_ok() {
        let req_f = mk_req_fixed_with(5, (0, 100), &[(1, 3)]);
        let berth_f = mk_berth_single(1, 0, 100);
        let rr = AssignmentRef::<FixedKind, i64>::new_fixed(&req_f, &berth_f, tp(10)).unwrap();
        assert_eq!(rr.request_id(), req_f.id());
        assert_eq!(rr.berth_id(), berth_f.id());

        let req_x = mk_req_flexible_with(6, (0, 100), &[(2, 4)]);
        let berth_x = mk_berth_multi();
        let rx =
            AssignmentRef::<FlexibleKind, i64>::new_flexible(&req_x, &berth_x, tp(12)).unwrap();
        assert_eq!(rx.request_id(), req_x.id());
        assert_eq!(rx.berth_id(), berth_x.id());
    }

    #[test]
    fn test_new_returns_error_when_request_has_no_processing_time_for_berth() {
        // Request allows only berth 1; we try to assign to berth 2 → error
        let req = mk_req_fixed_with(7, (0, 100), &[(1, 8)]);
        let berth_wrong = mk_berth_single(2, 0, 100);
        let res = Assignment::<FixedKind, i64>::new(req, berth_wrong, tp(10));
        assert!(res.is_err());
    }

    #[test]
    fn test_ref_new_returns_error_when_request_has_no_processing_time_for_berth() {
        let req = mk_req_flexible_with(8, (0, 100), &[(1, 2)]);
        let berth_wrong = mk_berth_single(2, 0, 100);
        let res = AssignmentRef::<FlexibleKind, i64>::new(&req, &berth_wrong, tp(0));
        assert!(res.is_err());
    }

    #[test]
    fn test_display_contains_kind_ids_and_start() {
        let req_f = mk_req_fixed_with(9, (0, 100), &[(2, 5)]);
        let berth = mk_berth_multi();
        let af =
            Assignment::<FixedKind, i64>::new_fixed(req_f.clone(), berth.clone(), tp(18)).unwrap();
        let sf = format!("{af}");
        assert!(sf.contains("Assignment<Fixed>"));
        assert!(sf.contains(&format!("{}", req_f.id())));
        assert!(sf.contains(&format!("{}", berth.id())));
        assert!(sf.contains(&format!("{}", tp(18))));

        let req_x = mk_req_flexible_with(10, (0, 100), &[(2, 8)]);
        let ax =
            Assignment::<FlexibleKind, i64>::new_flexible(req_x.clone(), berth.clone(), tp(22))
                .unwrap();
        let sx = format!("{ax}");
        assert!(sx.contains("Assignment<Flexible>"));
        assert!(sx.contains(&format!("{}", req_x.id())));
        assert!(sx.contains(&format!("{}", berth.id())));
        assert!(sx.contains(&format!("{}", tp(22))));

        // Ref display
        let rx = ax.to_ref();
        let srx = format!("{rx}");
        assert!(srx.contains("AssignmentRef<Flexible>"));
        assert!(srx.contains(&format!("{}", rx.request_id())));
        assert!(srx.contains(&format!("{}", rx.berth_id())));
        assert!(srx.contains(&format!("{}", rx.start_time())));
    }

    #[test]
    fn test_debug_format_exists() {
        let req = mk_req_flexible_with(11, (0, 100), &[(2, 6)]);
        let berth = mk_berth_multi();
        let a = Assignment::<FlexibleKind, i64>::new_flexible(req, berth, tp(33)).unwrap();
        let r = a.to_ref();

        let da = format!("{a:?}");
        let dr = format!("{r:?}");

        assert!(da.contains("Assignment"));
        assert!(dr.contains("AssignmentRef"));
    }

    #[test]
    fn test_eq_hash_clone_behavior() {
        let req = mk_req_fixed_with(12, (0, 100), &[(2, 7)]);
        let berth = mk_berth_multi();

        let a1 =
            Assignment::<FixedKind, i64>::new_fixed(req.clone(), berth.clone(), tp(11)).unwrap();
        let a2 = a1.clone();

        assert_eq!(a1, a2);
        assert_eq!(hash_of(&a1), hash_of(&a2));

        // Different start time -> different eq/hash
        let a3 =
            Assignment::<FixedKind, i64>::new_fixed(req.clone(), berth.clone(), tp(12)).unwrap();
        assert_ne!(a1, a3);
        assert_ne!(hash_of(&a1), hash_of(&a3));
    }

    #[test]
    fn test_assignment_new_errors_when_start_before_window() {
        // Window [10, 100), pt=5, start=9 -> before window
        let req = mk_req_flexible_with(20, (10, 100), &[(2, 5)]);
        let berth = mk_berth_single(2, 0, 1000);

        let err = Assignment::<FlexibleKind, i64>::new_flexible(req.clone(), berth.clone(), tp(9))
            .unwrap_err();

        match err {
            AssignmentError::AssignmentStartsBeforeFeasibleWindow(e) => {
                assert_eq!(e.id(), rid(20));
                assert_eq!(e.assigned_time(), tp(9));
                assert_eq!(e.start_time(), tp(10));
            }
            other => panic!("expected AssignmentStartsBeforeFeasibleWindow, got {other}"),
        }
    }

    #[test]
    fn test_assignment_new_errors_when_end_after_window() {
        // Window [0, 50), pt=10, start=45 -> end=55 > 50
        let req = mk_req_fixed_with(21, (0, 50), &[(1, 10)]);
        let berth = mk_berth_single(1, 0, 1000);

        let err = Assignment::<FixedKind, i64>::new_fixed(req.clone(), berth.clone(), tp(45))
            .unwrap_err();

        match err {
            AssignmentError::AssignmentEndsAfterFeasibleWindow(e) => {
                assert_eq!(e.id(), rid(21));
                assert_eq!(e.end_time(), tp(55));
                assert_eq!(e.window(), iv(0, 50));
            }
            other => panic!("expected AssignmentEndsAfterFeasibleWindow, got {other}"),
        }
    }

    #[test]
    fn test_assignment_ref_new_errors_when_start_before_window() {
        // Same scenario as above, but via AssignmentRef::new_*.
        let req = mk_req_flexible_with(30, (10, 100), &[(2, 5)]);
        let berth = mk_berth_single(2, 0, 1000);

        let err = AssignmentRef::<FlexibleKind, i64>::new(&req, &berth, tp(9)).unwrap_err();

        match err {
            AssignmentError::AssignmentStartsBeforeFeasibleWindow(e) => {
                assert_eq!(e.id(), rid(30));
                assert_eq!(e.assigned_time(), tp(9));
                assert_eq!(e.start_time(), tp(10));
            }
            other => panic!("expected AssignmentStartsBeforeFeasibleWindow, got {other}"),
        }
    }

    #[test]
    fn test_assignment_ref_new_errors_when_end_after_window() {
        // Window [0, 50), pt=10, start=45 -> end=55 > 50
        let req = mk_req_fixed_with(31, (0, 50), &[(1, 10)]);
        let berth = mk_berth_single(1, 0, 1000);

        let err = AssignmentRef::<FixedKind, i64>::new(&req, &berth, tp(45)).unwrap_err();

        match err {
            AssignmentError::AssignmentEndsAfterFeasibleWindow(e) => {
                assert_eq!(e.id(), rid(31));
                assert_eq!(e.end_time(), tp(55));
                assert_eq!(e.window(), iv(0, 50));
            }
            other => panic!("expected AssignmentEndsAfterFeasibleWindow, got {other}"),
        }
    }

    #[test]
    fn test_assignment_new_allows_start_at_window_start_and_end_at_window_end() {
        // Window [0, 50), pt=10
        let req = mk_req_fixed_with(41, (0, 50), &[(1, 10)]);
        let berth = mk_berth_single(1, 0, 1000);

        // start exactly at window start -> OK
        let a1 =
            Assignment::<FixedKind, i64>::new_fixed(req.clone(), berth.clone(), tp(0)).unwrap();
        assert_eq!(a1.start_time(), tp(0));
        assert_eq!(a1.end_time(), tp(10));

        // start so that end == window end -> OK
        let a2 =
            Assignment::<FixedKind, i64>::new_fixed(req.clone(), berth.clone(), tp(40)).unwrap();
        assert_eq!(a2.end_time(), tp(50));
    }

    #[test]
    fn test_assignment_ref_new_allows_boundaries() {
        // Window [5, 25), pt=5
        let req = mk_req_flexible_with(42, (5, 25), &[(2, 5)]);
        let berth = mk_berth_single(2, 0, 1000);

        // start == window.start -> OK
        let r1 = AssignmentRef::<FlexibleKind, i64>::new(&req, &berth, tp(5)).unwrap();
        assert_eq!(r1.start_time(), tp(5));
        assert_eq!(r1.end_time(), tp(10));

        // end == window.end -> OK
        let r2 = AssignmentRef::<FlexibleKind, i64>::new(&req, &berth, tp(20)).unwrap();
        assert_eq!(r2.end_time(), tp(25));
    }

    #[test]
    fn test_waiting_turnaround_cost_nonzero_wait() {
        // Window [10,100), start=15 -> waiting=5; pt=7 -> turnaround=12; weight=3 -> cost=36
        let berth = mk_berth_multi(); // id=2 available in [0,20) ∪ [25,50)
        let mut m = BTreeMap::new();
        m.insert(bid(2), td(7));

        let req = Request::<FixedKind, i64>::new(rid(77), iv(10, 100), 3, m).unwrap();
        let a = Assignment::<FixedKind, i64>::new_fixed(req, berth, tp(15)).unwrap();

        assert_eq!(a.waiting_time(), td(5));
        assert_eq!(a.turnaround_time(), td(12));

        // turnaround(12) * weight(3) = 36
        assert_eq!(a.cost(), 36);
        // (if you prefer type-driven calc)
        // assert_eq!(a.cost(), a.turnaround_time().value().into() * 3.into());
    }

    #[test]
    fn test_waiting_turnaround_cost_zero_wait() {
        // Window [0,50), start=0 -> waiting=0; pt=10 -> turnaround=10; weight=5 -> cost=50
        let berth = mk_berth_single(1, 0, 1000);
        let mut m = BTreeMap::new();
        m.insert(bid(1), td(10));

        let req = Request::<FixedKind, i64>::new(rid(78), iv(0, 50), 5, m).unwrap();
        let a = Assignment::<FixedKind, i64>::new_fixed(req, berth, tp(0)).unwrap();

        assert_eq!(a.waiting_time(), td(0));
        assert_eq!(a.turnaround_time(), td(10));

        // turnaround(10) * weight(5) = 50
        assert_eq!(a.cost(), 50);
    }
}
