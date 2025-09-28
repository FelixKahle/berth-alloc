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

use crate::common::{Identifier, IdentifierMarkerName};
use berth_alloc_core::prelude::{TimeInterval, TimePoint};
use num_traits::Zero;
use rangemap::RangeSet;
use std::{collections::HashMap, fmt::Debug, hash::Hash};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BerthIdentifierMarker;

impl IdentifierMarkerName for BerthIdentifierMarker {
    const NAME: &'static str = "BerthId";
}

pub type BerthIdentifier = Identifier<usize, BerthIdentifierMarker>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Berth<T: Ord + Copy> {
    id: BerthIdentifier,
    availability: RangeSet<TimePoint<T>>,
}

impl<T: Ord + Copy + Hash> std::hash::Hash for Berth<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        for r in self.availability.iter() {
            r.start.hash(state);
            r.end.hash(state);
        }
    }
}

impl<T: Ord + Copy> Berth<T> {
    #[inline]
    pub fn from_windows<I>(id: BerthIdentifier, windows: I) -> Self
    where
        I: IntoIterator<Item = TimeInterval<T>>,
    {
        Self {
            id,
            availability: RangeSet::from_iter(windows.into_iter().map(|iv| iv.into_range())),
        }
    }

    #[inline]
    pub fn id(&self) -> BerthIdentifier {
        self.id
    }

    #[inline]
    pub fn is_open_at(&self, t: TimePoint<T>) -> bool {
        self.availability.contains(&t)
    }

    #[inline]
    pub fn is_closed_at(&self, t: TimePoint<T>) -> bool {
        !self.is_open_at(t)
    }

    #[inline]
    pub fn iter_availability_windows(&self) -> impl Iterator<Item = TimeInterval<T>> + '_ {
        self.availability
            .iter()
            .map(|r| TimeInterval::new(r.start, r.end))
    }

    #[inline]
    pub fn overlaps(&self, iv: TimeInterval<T>) -> bool {
        let (s, e) = iv.into_inner();
        if s >= e {
            return false;
        }
        self.availability.overlaps(&(s..e))
    }

    #[inline]
    pub fn covers(&self, iv: TimeInterval<T>) -> bool {
        if iv.is_empty() {
            return true;
        }
        let (s, e) = iv.into_inner();
        self.availability.gaps(&(s..e)).next().is_none()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.availability.is_empty()
    }

    #[inline]
    pub fn windows(&self) -> Vec<TimeInterval<T>> {
        self.iter_availability_windows().collect()
    }

    #[inline]
    pub fn availability(&self) -> &RangeSet<TimePoint<T>> {
        &self.availability
    }

    #[inline]
    pub fn horizon_opt(&self) -> Option<TimePoint<T>> {
        self.availability.last().map(|r| r.end)
    }

    #[inline]
    pub fn horizon(&self) -> TimePoint<T>
    where
        T: Zero,
    {
        self.horizon_opt()
            .unwrap_or_else(|| TimePoint::new(T::zero()))
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct BerthContainer<T: Copy + Ord>(HashMap<BerthIdentifier, Berth<T>>);

impl<T: Copy + Ord> Default for BerthContainer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord> BerthContainer<T> {
    #[inline]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(HashMap::with_capacity(cap))
    }

    #[inline]
    pub fn insert(&mut self, berth: Berth<T>) -> Option<Berth<T>> {
        self.0.insert(berth.id(), berth)
    }

    #[inline]
    pub fn remove(&mut self, id: BerthIdentifier) -> Option<Berth<T>> {
        self.0.remove(&id)
    }

    #[inline]
    pub fn contains_id(&self, id: BerthIdentifier) -> bool {
        self.0.contains_key(&id)
    }

    #[inline]
    pub fn contains_berth(&self, berth: &Berth<T>) -> bool {
        let id = berth.id();
        self.contains_id(id)
    }

    #[inline]
    pub fn get(&self, id: BerthIdentifier) -> Option<&Berth<T>> {
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
    pub fn iter(&self) -> impl Iterator<Item = &Berth<T>> {
        self.0.values()
    }
}

impl<T: Copy + Ord> FromIterator<Berth<T>> for BerthContainer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Berth<T>>>(iter: I) -> Self {
        let mut c = Self::new();
        for b in iter {
            c.insert(b);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }

    #[test]
    fn test_from_windows_coalesces_and_is_half_open() {
        // Adjacent [0,10) and [10,20) must coalesce into [0,20).
        let b = Berth::from_windows(bid(1), vec![iv(0, 10), iv(10, 20), iv(25, 30)]);
        let windows = b.windows();
        assert_eq!(windows, vec![iv(0, 20), iv(25, 30)]);

        // Half-open behavior at the edges.
        assert!(b.is_open_at(tp(0)));
        assert!(b.is_open_at(tp(19)));
        assert!(!b.is_open_at(tp(20))); // exclusive end
        assert!(b.is_closed_at(tp(24)));
    }

    #[test]
    fn test_id_exposed() {
        let b = Berth::from_windows(bid(42), vec![iv(1, 2)]);
        assert_eq!(b.id(), bid(42));
    }

    #[test]
    fn test_is_open_at_and_is_closed_at() {
        let b = Berth::from_windows(bid(2), vec![iv(5, 10), iv(15, 18)]);
        assert!(b.is_open_at(tp(5)));
        assert!(b.is_open_at(tp(17)));
        assert!(b.is_closed_at(tp(10))); // exclusive end
        assert!(b.is_closed_at(tp(14))); // gap
    }

    #[test]
    fn test_overlaps_semantics() {
        let b = Berth::from_windows(bid(3), vec![iv(10, 20), iv(30, 35)]);
        // True overlaps.
        assert!(b.overlaps(iv(15, 25)));
        assert!(b.overlaps(iv(29, 31)));
        // Touching only (half-open) — not overlapping.
        assert!(!b.overlaps(iv(20, 30)));
        assert!(!b.overlaps(iv(0, 10)));
        // Disjoint / empty
        assert!(!b.overlaps(iv(40, 50)));
        assert!(!b.overlaps(iv(12, 12))); // empty query
    }

    #[test]
    fn test_covers_semantics() {
        let b = Berth::from_windows(bid(4), vec![iv(1, 5), iv(7, 10)]);
        // Fully covered inside a single window.
        assert!(b.covers(iv(2, 4)));
        assert!(b.covers(iv(7, 9)));
        // Spanning a gap is not covered.
        assert!(!b.covers(iv(4, 8)));
        // Empty interval is always considered covered.
        assert!(b.covers(iv(7, 7)));
        // Edges not covered.
        assert!(!b.covers(iv(0, 2)));
        assert!(!b.covers(iv(9, 11)));
    }

    #[test]
    fn test_iter_availability_windows_sorted_and_coalesced() {
        // Out-of-order and adjacent windows should come out sorted & coalesced.
        let b = Berth::from_windows(bid(5), vec![iv(20, 30), iv(0, 10), iv(10, 20)]);
        let it: Vec<_> = b.iter_availability_windows().collect();
        assert_eq!(it, vec![iv(0, 30)]);
        // windows() materialization matches the iterator.
        assert_eq!(b.windows(), vec![iv(0, 30)]);
    }

    #[test]
    fn test_is_empty_and_availability_reference() {
        let b = Berth::from_windows(bid(6), Vec::<TimeInterval<i64>>::new());
        assert!(b.is_empty());
        // availability() returns an empty set as well.
        assert!(b.availability().is_empty());
    }

    #[test]
    fn test_negative_coordinates_and_mixed() {
        let b = Berth::from_windows(bid(7), vec![iv(-5, -1), iv(1, 3)]);
        let w: Vec<_> = b.iter_availability_windows().collect();
        assert_eq!(w, vec![iv(-5, -1), iv(1, 3)]);
        assert!(b.is_open_at(tp(-2)));
        assert!(!b.is_open_at(tp(0)));
    }

    #[test]
    fn test_horizon_non_empty_picks_end_of_last() {
        // Coalesces to [0,20), then [25,30) ⇒ horizon is 30.
        let b = Berth::from_windows(bid(1), vec![iv(0, 10), iv(10, 20), iv(25, 30)]);
        assert_eq!(b.horizon_opt(), Some(tp(30)));
        // Using the total API (Zero fallback):
        assert_eq!(b.horizon(), tp(30));
    }

    #[test]
    fn test_horizon_with_unsorted_input() {
        // Out of order + adjacency coalesce to [0,30) ⇒ horizon 30.
        let b = Berth::from_windows(bid(2), vec![iv(20, 30), iv(0, 10), iv(10, 20)]);
        assert_eq!(b.horizon_opt(), Some(tp(30)));
    }

    #[test]
    fn test_horizon_empty() {
        let b = Berth::from_windows(bid(3), Vec::<TimeInterval<i64>>::new());
        assert_eq!(b.horizon_opt(), None);
        // Zero fallback for i64:
        assert_eq!(b.horizon(), tp(0));
    }
}
