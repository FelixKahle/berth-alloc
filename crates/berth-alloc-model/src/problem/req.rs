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
use average::{Estimate, Mean, Variance};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval};
use num_traits::{CheckedAdd, CheckedSub};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    hash::Hasher,
};

pub trait RequestView<T: Ord + Copy> {
    fn id(&self) -> RequestIdentifier;
    fn feasible_window(&self) -> TimeInterval<T>;
    fn weight(&self) -> Cost;
    fn processing_times(&self) -> &BTreeMap<BerthIdentifier, TimeDelta<T>>;
    fn iter_allowed_berths_ids<'a>(&'a self) -> impl Iterator<Item = BerthIdentifier> + 'a
    where
        T: 'a,
    {
        self.processing_times().keys().copied()
    }

    #[inline]
    fn processing_time_for(&self, berth_id: BerthIdentifier) -> Option<TimeDelta<T>> {
        self.processing_times().get(&berth_id).copied()
    }

    #[inline]
    fn is_berth_feasible(&self, berth_id: BerthIdentifier) -> bool {
        self.processing_times().contains_key(&berth_id)
    }

    #[inline]
    fn request_allowed_degree(&self) -> usize {
        self.processing_times().len()
    }

    #[inline]
    fn request_slack(&self) -> TimeDelta<T>
    where
        T: CheckedAdd + CheckedSub + Copy + Ord + num_traits::Zero,
    {
        let best_pt = self
            .processing_times()
            .values()
            .copied()
            .min()
            .unwrap_or_else(TimeDelta::zero);

        self.feasible_window().length() - best_pt
    }

    #[inline]
    fn best_processing_time(&self) -> Option<TimeDelta<T>> {
        self.processing_times().values().copied().min()
    }

    #[inline]
    fn worst_processing_time(&self) -> Option<TimeDelta<T>> {
        self.processing_times().values().copied().max()
    }

    #[inline]
    fn range_processing_time(&self) -> Option<TimeDelta<T>>
    where
        T: Ord + Copy + CheckedSub,
    {
        Some(self.worst_processing_time()? - self.best_processing_time()?)
    }

    fn mad_processing_time(&self) -> Option<TimeDelta<T>>
    where
        T: Ord + Copy + num_traits::Zero + std::ops::Sub<Output = T>,
    {
        if self.processing_times().is_empty() {
            return None;
        }
        let med = self.median_processing_time().value();
        let mut devs: Vec<T> = self
            .processing_times()
            .values()
            .map(|d| {
                let v = d.value();
                if v >= med { v - med } else { med - v }
            })
            .collect();
        let n = devs.len();
        let mid = (n - 1) / 2;
        let (_, m, _) = devs.select_nth_unstable(mid);
        Some(TimeDelta::new(*m))
    }

    #[inline]
    fn median_processing_time(&self) -> TimeDelta<T> {
        assert!(!self.processing_times().is_empty());

        let mut vals: Vec<TimeDelta<T>> = self.processing_times().values().copied().collect();
        let n = vals.len();
        let mid = (n - 1) / 2;
        let (_, m, _) = vals.select_nth_unstable(mid);
        *m
    }

    #[inline]
    fn mean_processing_time(&self) -> Option<f64>
    where
        T: CheckedSub + num_traits::ToPrimitive,
    {
        if self.processing_times().is_empty() {
            return None;
        }

        let mut m = average::Mean::new();
        for x in self
            .processing_times()
            .values()
            .filter_map(|d| d.value().to_f64())
        {
            m.add(x);
        }
        Some(m.mean())
    }

    #[inline]
    fn stddev_processing_time(&self) -> Option<f64>
    where
        T: CheckedSub + num_traits::ToPrimitive,
    {
        if self.processing_times().is_empty() {
            return None;
        }

        let mut v = Variance::new();
        for x in self
            .processing_times()
            .values()
            .filter_map(|d| d.value().to_f64())
        {
            v.add(x);
        }
        Some(v.estimate().sqrt())
    }

    #[inline]
    fn coefficient_of_variation_processing_time(&self) -> Option<f64>
    where
        T: CheckedSub + num_traits::ToPrimitive,
    {
        if self.processing_times().is_empty() {
            return None;
        }

        let mut mean = Mean::new();
        let mut var = Variance::new();
        for x in self
            .processing_times()
            .values()
            .filter_map(|d| d.value().to_f64())
        {
            mean.add(x);
            var.add(x);
        }
        let m = mean.mean();
        if m == 0.0 {
            return Some(0.0);
        }
        Some(var.estimate().sqrt() / m)
    }

    #[inline]
    fn slack_ratio(&self) -> f64
    where
        T: CheckedAdd + CheckedSub + num_traits::Zero + num_traits::ToPrimitive,
    {
        let len = self
            .feasible_window()
            .length()
            .value()
            .to_f64()
            .unwrap_or(0.0);
        if len == 0.0 {
            return 0.0;
        }
        let slack = self.request_slack().value().to_f64().unwrap_or(0.0);
        (slack / len).clamp(0.0, 1.0)
    }

    fn iqr_processing_time(&self) -> Option<TimeDelta<T>>
    where
        T: Ord + Copy + CheckedSub,
    {
        if self.processing_times().is_empty() {
            return None;
        }

        let mut vals: Vec<TimeDelta<T>> = self.processing_times().values().copied().collect();
        let n = vals.len();

        let q1_idx = (n - 1) / 4;
        let q3_idx = (3 * (n - 1)) / 4;

        let q1 = {
            let (_, m, _) = vals.select_nth_unstable(q1_idx);
            *m
        };
        let q3 = {
            let (_, m, _) = vals.select_nth_unstable(q3_idx);
            *m
        };

        Some(q3 - q1)
    }

    #[inline]
    fn tightness_min(&self) -> f64
    where
        T: CheckedSub + num_traits::ToPrimitive + num_traits::Zero,
    {
        let len = self
            .feasible_window()
            .length()
            .value()
            .to_f64()
            .unwrap_or(0.0);
        if len == 0.0 {
            return 1.0;
        }
        let pt = self
            .best_processing_time()
            .unwrap_or_else(TimeDelta::zero)
            .value()
            .to_f64()
            .unwrap_or(0.0);
        (pt / len).clamp(0.0, 1.0)
    }

    #[inline]
    fn tightness_max(&self) -> f64
    where
        T: CheckedSub + num_traits::ToPrimitive + num_traits::Zero,
    {
        let len = self
            .feasible_window()
            .length()
            .value()
            .to_f64()
            .unwrap_or(0.0);
        if len == 0.0 {
            return 1.0;
        }
        let pt = self
            .worst_processing_time()
            .unwrap_or_else(TimeDelta::zero)
            .value()
            .to_f64()
            .unwrap_or(0.0);
        (pt / len).clamp(0.0, 1.0)
    }

    fn tightness_median(&self) -> f64
    where
        T: num_traits::ToPrimitive + CheckedSub,
    {
        let len = self
            .feasible_window()
            .length()
            .value()
            .to_f64()
            .unwrap_or(0.0);
        if len == 0.0 {
            return 1.0;
        }
        let med = self
            .median_processing_time()
            .value()
            .to_f64()
            .unwrap_or(0.0);
        (med / len).clamp(0.0, 1.0)
    }

    #[inline]
    fn argmin_processing_time(&self) -> (BerthIdentifier, TimeDelta<T>) {
        self.processing_times()
            .iter()
            .fold(None, |best, (&bid, &pt)| match best {
                None => Some((bid, pt)),
                Some((bbid, bpt)) => {
                    if pt < bpt || (pt == bpt && bid < bbid) {
                        Some((bid, pt))
                    } else {
                        Some((bbid, bpt))
                    }
                }
            })
            .unwrap()
    }

    #[inline]
    fn berth_preference(&self) -> Vec<(BerthIdentifier, TimeDelta<T>)> {
        let mut v: Vec<_> = self
            .processing_times()
            .iter()
            .map(|(b, d)| (*b, *d))
            .collect();
        v.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        v
    }

    #[inline]
    fn time_fraction_on(&self, bid: BerthIdentifier) -> Option<f64>
    where
        T: CheckedSub + num_traits::ToPrimitive,
    {
        let pt = self.processing_time_for(bid)?;
        let len = self
            .feasible_window()
            .length()
            .value()
            .to_f64()
            .unwrap_or(0.0);
        if len == 0.0 {
            return Some(0.0);
        }
        let x = pt.value().to_f64().unwrap_or(0.0);
        Some((x / len).clamp(0.0, 1.0))
    }

    #[inline]
    fn allowed_signature(&self) -> std::collections::BTreeSet<BerthIdentifier> {
        self.processing_times().keys().copied().collect()
    }

    #[inline]
    fn latest_start_offset_for(&self, bid: BerthIdentifier) -> Option<TimeDelta<T>>
    where
        T: CheckedSub,
    {
        self.processing_time_for(bid)
            .map(|pt| self.feasible_window().length() - pt)
    }
}

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
}

impl<K: Kind, T: Ord + Copy> RequestView<T> for Request<K, T> {
    #[inline]
    fn id(&self) -> RequestIdentifier {
        self.id
    }

    #[inline]
    fn feasible_window(&self) -> TimeInterval<T> {
        self.feasible_window
    }

    #[inline]
    fn weight(&self) -> Cost {
        self.weight
    }

    #[inline]
    fn processing_times(&self) -> &BTreeMap<BerthIdentifier, TimeDelta<T>> {
        &self.processing_times
    }

    #[inline]
    fn iter_allowed_berths_ids<'a>(&'a self) -> impl Iterator<Item = BerthIdentifier> + 'a
    where
        T: 'a,
    {
        self.processing_times.keys().copied()
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnyRequest<T: Ord + Copy> {
    Fixed(Request<FixedKind, T>),
    Flexible(Request<FlexibleKind, T>),
}

impl<T: Ord + Copy> RequestView<T> for AnyRequest<T> {
    fn id(&self) -> RequestIdentifier {
        match self {
            AnyRequest::Fixed(r) => r.id(),
            AnyRequest::Flexible(r) => r.id(),
        }
    }

    fn feasible_window(&self) -> TimeInterval<T> {
        match self {
            AnyRequest::Fixed(r) => r.feasible_window(),
            AnyRequest::Flexible(r) => r.feasible_window(),
        }
    }

    fn weight(&self) -> Cost {
        match self {
            AnyRequest::Fixed(r) => r.weight(),
            AnyRequest::Flexible(r) => r.weight(),
        }
    }

    fn processing_times(&self) -> &BTreeMap<BerthIdentifier, TimeDelta<T>> {
        match self {
            AnyRequest::Fixed(r) => r.processing_times(),
            AnyRequest::Flexible(r) => r.processing_times(),
        }
    }
}

impl<T: Copy + Ord + std::fmt::Display> std::fmt::Display for AnyRequest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyRequest::Fixed(r) => write!(f, "{}", r),
            AnyRequest::Flexible(r) => write!(f, "{}", r),
        }
    }
}

impl<T: Copy + Ord> From<Request<FixedKind, T>> for AnyRequest<T> {
    #[inline]
    fn from(r: Request<FixedKind, T>) -> Self {
        AnyRequest::Fixed(r)
    }
}

impl<T: Copy + Ord> From<Request<FlexibleKind, T>> for AnyRequest<T> {
    #[inline]
    fn from(r: Request<FlexibleKind, T>) -> Self {
        AnyRequest::Flexible(r)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnyRequestRef<'a, T: Ord + Copy> {
    Fixed(&'a Request<FixedKind, T>),
    Flexible(&'a Request<FlexibleKind, T>),
}

impl<'a, T: Ord + Copy> RequestView<T> for AnyRequestRef<'a, T> {
    fn id(&self) -> RequestIdentifier {
        match self {
            AnyRequestRef::Fixed(r) => r.id(),
            AnyRequestRef::Flexible(r) => r.id(),
        }
    }

    fn feasible_window(&self) -> TimeInterval<T> {
        match self {
            AnyRequestRef::Fixed(r) => r.feasible_window(),
            AnyRequestRef::Flexible(r) => r.feasible_window(),
        }
    }

    fn weight(&self) -> Cost {
        match self {
            AnyRequestRef::Fixed(r) => r.weight(),
            AnyRequestRef::Flexible(r) => r.weight(),
        }
    }

    fn processing_times(&self) -> &BTreeMap<BerthIdentifier, TimeDelta<T>> {
        match self {
            AnyRequestRef::Fixed(r) => r.processing_times(),
            AnyRequestRef::Flexible(r) => r.processing_times(),
        }
    }
}

impl<'a, T: Copy + Ord + std::fmt::Display> std::fmt::Display for AnyRequestRef<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyRequestRef::Fixed(r) => write!(f, "{}", r),
            AnyRequestRef::Flexible(r) => write!(f, "{}", r),
        }
    }
}

impl<'a, T: Copy + Ord> From<&'a Request<FixedKind, T>> for AnyRequestRef<'a, T> {
    #[inline]
    fn from(r: &'a Request<FixedKind, T>) -> Self {
        AnyRequestRef::Fixed(r)
    }
}

impl<'a, T: Copy + Ord> From<&'a Request<FlexibleKind, T>> for AnyRequestRef<'a, T> {
    #[inline]
    fn from(r: &'a Request<FlexibleKind, T>) -> Self {
        AnyRequestRef::Flexible(r)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct RequestContainer<T: Ord + Copy, V: RequestView<T>> {
    inner: HashMap<RequestIdentifier, V>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + Ord, V: RequestView<T>> Default for RequestContainer<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord, V: RequestView<T>> RequestContainer<T, V> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(cap),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn insert(&mut self, request: V) -> Option<V>
    where
        T: CheckedSub,
    {
        self.inner.insert(request.id(), request)
    }

    #[inline]
    pub fn remove(&mut self, id: RequestIdentifier) -> Option<V> {
        self.inner.remove(&id)
    }

    #[inline]
    pub fn contains_id(&self, id: RequestIdentifier) -> bool {
        self.inner.contains_key(&id)
    }

    #[inline]
    pub fn contains_request(&self, request: &V) -> bool
    where
        T: CheckedSub,
    {
        let id = request.id();
        self.inner.contains_key(&id)
    }

    #[inline]
    pub fn get(&self, id: RequestIdentifier) -> Option<&V> {
        self.inner.get(&id)
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
}

impl<T: Copy + Ord + CheckedSub, V: RequestView<T>> FromIterator<V> for RequestContainer<T, V> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
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

    #[test]
    fn test_iter_allowed_berths_ids_and_is_berth_feasible_and_degree() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        pt.insert(bid(3), td(7));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(100), iv(0, 100), 1, pt)
            .unwrap();

        let ids: Vec<_> = r.iter_allowed_berths_ids().collect();
        assert_eq!(ids, vec![bid(1), bid(3)]); // BTreeMap is ordered by key

        assert!(r.is_berth_feasible(bid(1)));
        assert!(!r.is_berth_feasible(bid(2)));
        assert!(r.is_berth_feasible(bid(3)));
        assert_eq!(r.request_allowed_degree(), 2);
    }

    #[test]
    fn test_request_slack_and_slack_ratio() {
        // Window [0,30), PTs {1: 10, 2: 15} -> best=10 => slack=20
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        pt.insert(bid(2), td(15));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(101), iv(0, 30), 1, pt)
            .unwrap();

        assert_eq!(r.request_slack(), td(20));

        let ratio = r.slack_ratio();
        // ratio = slack/len = 20/30 = 0.666..
        assert!((ratio - (20.0 / 30.0)).abs() < 1e-9);
    }

    #[test]
    fn test_best_and_worst_processing_time_some() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(2), td(8));
        pt.insert(bid(1), td(5));
        pt.insert(bid(5), td(12));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(102), iv(0, 50), 1, pt)
            .unwrap();

        assert_eq!(r.best_processing_time(), Some(td(5)));
        assert_eq!(r.worst_processing_time(), Some(td(12)));
    }

    #[test]
    fn test_median_processing_time_odd_and_even_lower_median() {
        // Odd count: {3,7,9} -> median = 7
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(9));
        pt1.insert(bid(2), td(3));
        pt1.insert(bid(3), td(7));
        let r1 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(103), iv(0, 50), 1, pt1)
            .unwrap();
        assert_eq!(r1.median_processing_time(), td(7));

        // Even count (lower median): {2,5,5,9} -> lower median = element index 1 => 5
        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(5));
        pt2.insert(bid(2), td(9));
        pt2.insert(bid(3), td(2));
        pt2.insert(bid(4), td(5));
        let r2 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(104), iv(0, 50), 1, pt2)
            .unwrap();
        assert_eq!(r2.median_processing_time(), td(5));
    }

    #[test]
    fn test_mean_stddev_and_cv_processing_time_with_average_crate() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 (classic example: mean = 5, stddev sample ~ 2.138...)
        let vals = [2, 4, 4, 4, 5, 5, 7, 9];

        let mut pt = BTreeMap::new();
        for (i, v) in vals.iter().enumerate() {
            pt.insert(BerthIdentifier::new(i as u32 + 1), TimeDelta::new(*v));
        }
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(105), iv(0, 100), 1, pt)
            .unwrap();

        // mean_processing_time
        let mut m = Mean::new();
        for &x in &vals {
            m.add(x as f64);
        }
        let expected_mean = m.mean();
        assert_eq!(r.mean_processing_time().unwrap(), expected_mean);

        // stddev_processing_time uses Variance.estimate().sqrt() in the impl.
        let mut v = Variance::new();
        for &x in &vals {
            v.add(x as f64);
        }
        let expected_std = v.estimate().sqrt();
        assert!((r.stddev_processing_time().unwrap() - expected_std).abs() < 1e-12);

        // coefficient_of_variation_processing_time = stddev / mean (with mean != 0)
        let expected_cv = expected_std / expected_mean;
        assert!(
            (r.coefficient_of_variation_processing_time().unwrap() - expected_cv).abs() < 1e-12
        );
    }

    #[test]
    fn test_tightness_min_max_and_zero_len_window_behavior() {
        // |window| = 20, best=5, worst=15 -> min = 5/20=0.25, max = 15/20=0.75
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(15));
        pt.insert(bid(2), td(5));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(106), iv(0, 20), 1, pt)
            .unwrap();

        assert!((r.tightness_min() - 0.25).abs() < 1e-12);
        assert!((r.tightness_max() - 0.75).abs() < 1e-12);

        // Zero-length window -> both return 1.0
        let mut pt0 = BTreeMap::new();
        pt0.insert(bid(1), td(0));
        let r0 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(107), iv(10, 10), 1, pt0)
            .unwrap();
        assert_eq!(r0.tightness_min(), 1.0);
        assert_eq!(r0.tightness_max(), 1.0);
    }

    #[test]
    fn test_argmin_processing_time_tie_breaks_by_berth_id() {
        // { (2,5), (1,5), (3,7) } -> argmin = (1,5) since tie on PT, lower berth id wins
        let mut pt = BTreeMap::new();
        pt.insert(bid(2), td(5));
        pt.insert(bid(1), td(5));
        pt.insert(bid(3), td(7));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(108), iv(0, 50), 1, pt)
            .unwrap();

        let (b, d) = r.argmin_processing_time();
        assert_eq!(b, bid(1));
        assert_eq!(d, td(5));
    }

    #[test]
    fn test_berth_preference_sorted_by_pt_then_id() {
        // Expect sorting by PT ascending, then by berth id
        // Entries: (id, pt) = (2,5), (1,5), (3,7) -> order: (1,5), (2,5), (3,7)
        let mut pt = BTreeMap::new();
        pt.insert(bid(2), td(5));
        pt.insert(bid(1), td(5));
        pt.insert(bid(3), td(7));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(109), iv(0, 50), 1, pt)
            .unwrap();

        let pref = r.berth_preference();
        assert_eq!(
            pref,
            vec![(bid(1), td(5)), (bid(2), td(5)), (bid(3), td(7))]
        );
    }

    #[test]
    fn test_time_fraction_on_and_forbidden_berth_and_zero_window() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        pt.insert(bid(2), td(15));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(110), iv(0, 20), 1, pt)
            .unwrap();

        // 5/20 = 0.25
        assert!((r.time_fraction_on(bid(1)).unwrap() - 0.25).abs() < 1e-12);
        // forbidden berth -> None
        assert!(r.time_fraction_on(bid(3)).is_none());

        // zero-length window -> Some(0.0)
        let mut pt0 = BTreeMap::new();
        pt0.insert(bid(1), td(0));
        let r0 =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(111), iv(100, 100), 1, pt0)
                .unwrap();
        assert_eq!(r0.time_fraction_on(bid(1)).unwrap(), 0.0);
    }

    #[test]
    fn test_allowed_signature_contains_all_allowed_berths() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(2), td(5));
        pt.insert(bid(5), td(7));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(112), iv(0, 50), 1, pt)
            .unwrap();

        let sig = r.allowed_signature();
        let expected: std::collections::BTreeSet<_> = [bid(2), bid(5)].into_iter().collect();
        assert_eq!(sig, expected);
    }

    #[test]
    fn test_latest_start_offset_for_some_and_none() {
        // |window| = 20, pt(1)=5, pt(2)=20
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        pt.insert(bid(2), td(20));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(113), iv(0, 20), 1, pt)
            .unwrap();

        assert_eq!(r.latest_start_offset_for(bid(1)), Some(td(15)));
        assert_eq!(r.latest_start_offset_for(bid(2)), Some(td(0)));
        assert_eq!(r.latest_start_offset_for(bid(3)), None);
    }

    #[test]
    fn test_request_container_insert_remove_get_iter_len_contains() {
        let mut c = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);

        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(5));
        let r1 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(201), iv(0, 10), 1, pt1)
            .unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(2), td(3));
        let r2 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(202), iv(0, 10), 1, pt2)
            .unwrap();

        // insert
        c.insert(r1.clone());
        c.insert(r2.clone());

        assert!(!c.is_empty());
        assert_eq!(c.len(), 2);
        assert!(c.contains_id(r1.id()));
        assert!(c.contains_request(&r2));
        assert!(c.get(r1.id()).is_some());

        // iter
        let ids: std::collections::BTreeSet<_> = c.iter().map(|r| r.id()).collect();
        let expected: std::collections::BTreeSet<_> = [r1.id(), r2.id()].into_iter().collect();
        assert_eq!(ids, expected);

        // remove
        let removed = c.remove(r1.id()).unwrap();
        assert_eq!(removed.id(), r1.id());
        assert_eq!(c.len(), 1);
        assert!(!c.contains_id(r1.id()));
        assert!(c.contains_id(r2.id()));
    }

    #[test]
    fn test_any_request_and_any_request_ref_conversions_and_display() {
        // Fixed and Flexible requests
        let mut pt_f = BTreeMap::new();
        pt_f.insert(bid(1), td(5));
        let rf =
            Request::<FixedKind, i64>::new_fixed(RequestIdentifier::new(301), iv(0, 10), 1, pt_f)
                .unwrap();

        let mut pt_x = BTreeMap::new();
        pt_x.insert(bid(2), td(7));
        let rx = Request::<FlexibleKind, i64>::new_flexible(
            RequestIdentifier::new(302),
            iv(0, 20),
            1,
            pt_x,
        )
        .unwrap();

        // AnyRequest From
        let ar_f = AnyRequest::from(rf.clone());
        let ar_x = AnyRequest::from(rx.clone());
        assert_eq!(format!("{ar_f}").contains("Fixed-Request"), true);
        assert_eq!(format!("{ar_x}").contains("Flexible-Request"), true);

        // AnyRequestRef From and methods
        let arf = AnyRequestRef::from(&rf);
        let arx = AnyRequestRef::from(&rx);
        assert_eq!(arf.id(), rf.id());
        assert_eq!(arx.id(), rx.id());
        assert_eq!(arf.feasible_window(), rf.feasible_window());
        assert_eq!(arx.feasible_window(), rx.feasible_window());
        assert_eq!(
            arf.processing_time_for(bid(1)),
            rf.processing_time_for(bid(1))
        );
        assert_eq!(
            arx.processing_time_for(bid(2)),
            rx.processing_time_for(bid(2))
        );

        // Display for AnyRequestRef
        assert!(format!("{arf}").contains("Fixed-Request"));
        assert!(format!("{arx}").contains("Flexible-Request"));
    }

    #[test]
    fn test_range_processing_time_some_and_singleton_zero() {
        // Multiple entries: best=5, worst=12 -> range=7
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        pt.insert(bid(2), td(12));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1201), iv(0, 100), 1, pt)
            .unwrap();
        assert_eq!(r.range_processing_time(), Some(td(12 - 5)));

        // Singleton: best=worst=7 -> range=0
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(3), td(7));
        let r1 =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1202), iv(0, 100), 1, pt1)
                .unwrap();
        assert_eq!(r1.range_processing_time(), Some(td(0)));
    }

    #[test]
    fn test_mad_processing_time_zero_and_nonzero() {
        // Data: {1,2,3,4,100} => median=3, deviations={2,1,0,1,97}, median(deviations)=1
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(1));
        pt.insert(bid(2), td(2));
        pt.insert(bid(3), td(3));
        pt.insert(bid(4), td(4));
        pt.insert(bid(5), td(100));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1210), iv(0, 200), 1, pt)
            .unwrap();
        assert_eq!(r.mad_processing_time(), Some(td(1)));

        // All equal -> deviations all zero -> MAD = 0
        let mut pt_eq = BTreeMap::new();
        pt_eq.insert(bid(1), td(5));
        pt_eq.insert(bid(2), td(5));
        pt_eq.insert(bid(3), td(5));
        let r_eq =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1211), iv(0, 10), 1, pt_eq)
                .unwrap();
        assert_eq!(r_eq.mad_processing_time(), Some(td(0)));
    }

    #[test]
    fn test_tightness_median_and_zero_window_behavior() {
        // |window|=20, PTs {6,10,14} -> median=10 => tightness_median=10/20=0.5
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(6));
        pt.insert(bid(2), td(10));
        pt.insert(bid(3), td(14));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1220), iv(0, 20), 1, pt)
            .unwrap();
        assert!((r.tightness_median() - 0.5).abs() < 1e-12);

        // Zero-length window -> returns 1.0 by definition
        let mut pt0 = BTreeMap::new();
        pt0.insert(bid(1), td(0));
        let r0 =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1221), iv(100, 100), 1, pt0)
                .unwrap();
        assert_eq!(r0.tightness_median(), 1.0);
    }

    #[test]
    fn test_weight_accessor() {
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(5));
        let w = 7;
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1230), iv(0, 10), w, pt)
            .unwrap();
        assert_eq!(r.weight(), w);
    }

    #[test]
    fn test_request_container_with_capacity_and_basic_ops() {
        let mut c = RequestContainer::<i64, Request<FlexibleKind, i64>>::with_capacity(16);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);

        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(3));
        let r1 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1240), iv(0, 10), 1, pt1)
            .unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(2), td(4));
        let r2 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1241), iv(0, 10), 1, pt2)
            .unwrap();

        c.insert(r1.clone());
        c.insert(r2.clone());

        assert_eq!(c.len(), 2);
        assert!(c.contains_id(r1.id()));
        assert!(c.get(r2.id()).is_some());

        let removed = c.remove(r1.id()).unwrap();
        assert_eq!(removed.id(), r1.id());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_iqr_processing_time_basic_and_even_counts() {
        // Odd count: {1,2,3,4,100} => q1 idx=(5-1)/4=1 -> 2, q3 idx=(3*4)/4=3 -> 4 => IQR=2
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(1));
        pt.insert(bid(2), td(2));
        pt.insert(bid(3), td(3));
        pt.insert(bid(4), td(4));
        pt.insert(bid(5), td(100));
        let r = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1301), iv(0, 200), 1, pt)
            .unwrap();
        assert_eq!(r.iqr_processing_time(), Some(td(2)));

        // Even count: {1,2,3,4} => q1 idx=(4-1)/4=0 -> 1, q3 idx=(3*3)/4=2 -> 3 => IQR=2
        let mut pt_even = BTreeMap::new();
        pt_even.insert(bid(1), td(1));
        pt_even.insert(bid(2), td(2));
        pt_even.insert(bid(3), td(3));
        pt_even.insert(bid(4), td(4));
        let r_even =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1302), iv(0, 100), 1, pt_even)
                .unwrap();
        assert_eq!(r_even.iqr_processing_time(), Some(td(2)));
    }

    #[test]
    fn test_iqr_processing_time_all_equal_and_singleton() {
        // All equal -> q1==q3 -> IQR=0
        let mut pt_eq = BTreeMap::new();
        pt_eq.insert(bid(1), td(5));
        pt_eq.insert(bid(2), td(5));
        pt_eq.insert(bid(3), td(5));
        pt_eq.insert(bid(4), td(5));
        let r_eq =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1303), iv(0, 10), 1, pt_eq)
                .unwrap();
        assert_eq!(r_eq.iqr_processing_time(), Some(td(0)));

        // Singleton -> q1=q3=that value -> IQR=0
        let mut pt_one = BTreeMap::new();
        pt_one.insert(bid(1), td(7));
        let r_one =
            Request::<FlexibleKind, i64>::new(RequestIdentifier::new(1304), iv(0, 10), 1, pt_one)
                .unwrap();
        assert_eq!(r_one.iqr_processing_time(), Some(td(0)));
    }
}
