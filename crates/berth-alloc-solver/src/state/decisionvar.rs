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

use crate::model::index::{BerthIndex, RequestIndex};
use berth_alloc_core::prelude::TimePoint;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Decision<T> {
    pub berth_index: BerthIndex,
    pub start_time: TimePoint<T>,
}

impl<T> Decision<T> {
    #[inline]
    pub fn new(berth_index: BerthIndex, start_time: TimePoint<T>) -> Self {
        Self {
            berth_index,
            start_time,
        }
    }
}

impl Ord for Decision<i64> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.berth_index.cmp(&other.berth_index) {
            std::cmp::Ordering::Equal => self.start_time.cmp(&other.start_time),
            ord => ord,
        }
    }
}

impl PartialOrd for Decision<i64> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Decision<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth: {}, Start Time: {}",
            self.berth_index, self.start_time
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DecisionVar<T> {
    Unassigned,
    Assigned(Decision<T>),
}

impl<T> DecisionVar<T> {
    #[inline]
    pub fn unassigned() -> Self {
        Self::Unassigned
    }

    #[inline]
    pub fn assigned(berth_index: BerthIndex, start_time: TimePoint<T>) -> Self {
        Self::Assigned(Decision::new(berth_index, start_time))
    }

    #[inline]
    pub fn is_assigned(&self) -> bool {
        matches!(self, Self::Assigned(_))
    }

    #[inline]
    pub fn as_assigned(&self) -> Option<&Decision<T>> {
        match self {
            Self::Assigned(decision) => Some(decision),
            Self::Unassigned => None,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for DecisionVar<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unassigned => write!(f, "Unassigned"),
            Self::Assigned(decision) => write!(f, "{}", decision),
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionVarVec<T>(Vec<DecisionVar<T>>);

impl<T> DecisionVarVec<T> {
    #[inline]
    pub fn new(vec: Vec<DecisionVar<T>>) -> Self {
        Self(vec)
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    #[inline]
    pub fn from_slice(slice: &[DecisionVar<T>]) -> Self
    where
        T: Clone,
    {
        Self(slice.to_vec())
    }

    #[inline]
    pub fn as_slice(&self) -> &[DecisionVar<T>] {
        &self.0
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [DecisionVar<T>] {
        &mut self.0
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
    pub fn push(&mut self, value: DecisionVar<T>) {
        self.0.push(value);
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&DecisionVar<T>> {
        self.0.get(index)
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DecisionVar<T>> {
        self.0.get_mut(index)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, DecisionVar<T>> {
        self.0.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, DecisionVar<T>> {
        self.0.iter_mut()
    }

    #[inline]
    pub fn enumerate(&self) -> impl Iterator<Item = (RequestIndex, &DecisionVar<T>)> {
        self.0
            .iter()
            .enumerate()
            .map(|(i, dv)| (RequestIndex(i), dv))
    }
}

impl<T> std::ops::Index<RequestIndex> for DecisionVarVec<T> {
    type Output = DecisionVar<T>;

    #[inline]
    fn index(&self, index: RequestIndex) -> &Self::Output {
        &self.0[index.get()]
    }
}

impl<T> std::ops::IndexMut<RequestIndex> for DecisionVarVec<T> {
    #[inline]
    fn index_mut(&mut self, index: RequestIndex) -> &mut Self::Output {
        &mut self.0[index.get()]
    }
}

impl<T> IntoIterator for DecisionVarVec<T> {
    type Item = DecisionVar<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> FromIterator<DecisionVar<T>> for DecisionVarVec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = DecisionVar<T>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> From<Vec<DecisionVar<T>>> for DecisionVarVec<T> {
    #[inline]
    fn from(vec: Vec<DecisionVar<T>>) -> Self {
        Self::new(vec)
    }
}

impl<T> From<&[DecisionVar<T>]> for DecisionVarVec<T>
where
    T: Clone,
{
    #[inline]
    fn from(slice: &[DecisionVar<T>]) -> Self {
        Self::from_slice(slice)
    }
}

impl<T> std::ops::Deref for DecisionVarVec<T> {
    type Target = [DecisionVar<T>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::index::BerthIndex;
    use berth_alloc_core::prelude::TimePoint;
    use std::cmp::Ordering;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }

    #[test]
    fn test_decision_new_sets_fields() {
        let berth_idx = BerthIndex::new(2);
        let start = tp(10);
        let d = Decision::new(berth_idx, start);

        assert_eq!(d.berth_index, berth_idx);
        assert_eq!(d.start_time, start);
    }

    #[test]
    fn test_decision_display_includes_berth_and_time() {
        let berth_idx = BerthIndex::new(5);
        let start = tp(42);
        let d = Decision::new(berth_idx, start);

        let s = d.to_string();
        // Ensure both components are present via their own Display impls
        assert!(
            s.contains(&format!("{}", berth_idx)),
            "display should include berth index; got: {s}"
        );
        assert!(
            s.contains(&format!("{}", start)),
            "display should include start time; got: {s}"
        );
    }

    #[test]
    fn test_decisionvar_unassigned_constructors_and_accessors() {
        let u1: DecisionVar<i64> = DecisionVar::unassigned();
        let u2: DecisionVar<i64> = DecisionVar::Unassigned;

        assert_eq!(u1, u2);
        assert!(!u1.is_assigned());
        assert!(u1.as_assigned().is_none());

        let disp = u1.to_string();
        assert_eq!(disp, "Unassigned");
    }

    #[test]
    fn test_decisionvar_assigned_constructors_and_accessors() {
        let berth_idx = BerthIndex::new(7);
        let start = tp(123);

        let v1 = DecisionVar::assigned(berth_idx, start);
        let v2 = DecisionVar::Assigned(Decision::new(berth_idx, start));

        assert_eq!(v1, v2);
        assert!(v1.is_assigned());

        let dref = v1.as_assigned().expect("expected Some decision");
        assert_eq!(dref.berth_index, berth_idx);
        assert_eq!(dref.start_time, start);

        // Display should include both parts
        let s = v1.to_string();
        assert!(
            s.contains(&format!("{}", berth_idx)),
            "assigned display should include berth index; got: {s}"
        );
        assert!(
            s.contains(&format!("{}", start)),
            "assigned display should include start time; got: {s}"
        );
    }

    #[test]
    fn test_new_and_with_capacity_and_basic_ops() {
        let mut dvv = DecisionVarVec::with_capacity(4);
        assert!(dvv.is_empty());
        assert_eq!(dvv.len(), 0);

        dvv.push(DecisionVar::unassigned());
        dvv.push(DecisionVar::assigned(BerthIndex::new(1), tp(10)));

        assert!(!dvv.is_empty());
        assert_eq!(dvv.len(), 2);

        // as_slice reflects pushed elements
        let s = dvv.as_slice();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], DecisionVar::Unassigned);
        assert!(s[1].is_assigned());

        // as_mut_slice mutation
        let m = dvv.as_mut_slice();
        m[0] = DecisionVar::assigned(BerthIndex::new(3), tp(7));
        assert!(dvv.as_slice()[0].is_assigned());
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut dvv = DecisionVarVec::new(vec![
            DecisionVar::unassigned(),
            DecisionVar::assigned(BerthIndex::new(2), tp(20)),
        ]);

        assert!(matches!(dvv.get(0), Some(DecisionVar::Unassigned)));
        assert!(matches!(dvv.get(1), Some(DecisionVar::Assigned(_))));
        assert!(dvv.get(99).is_none());

        if let Some(slot) = dvv.get_mut(0) {
            *slot = DecisionVar::assigned(BerthIndex::new(3), tp(30));
        }
        assert!(matches!(dvv.get(0), Some(DecisionVar::Assigned(_))));
    }

    #[test]
    fn test_index_and_index_mut_with_request_index() {
        let mut dvv = DecisionVarVec::from(vec![
            DecisionVar::assigned(BerthIndex::new(1), tp(1)),
            DecisionVar::assigned(BerthIndex::new(2), tp(2)),
            DecisionVar::unassigned(),
        ]);

        // Index
        let i1 = RequestIndex::new(1);
        assert!(dvv[i1].is_assigned());

        // IndexMut
        let i2 = RequestIndex::new(2);
        dvv[i2] = DecisionVar::assigned(BerthIndex::new(9), tp(99));
        assert!(dvv[i2].is_assigned());
        let d = dvv[i2].as_assigned().unwrap();
        assert_eq!(d.berth_index, BerthIndex::new(9));
        assert_eq!(d.start_time, tp(99));
    }

    #[test]
    fn test_iter_and_iter_mut() {
        let mut dvv = DecisionVarVec::from(vec![
            DecisionVar::unassigned(),
            DecisionVar::assigned(BerthIndex::new(1), tp(10)),
            DecisionVar::unassigned(),
        ]);

        // iter
        let assigned_count = dvv.iter().filter(|dv| dv.is_assigned()).count();
        assert_eq!(assigned_count, 1);

        // iter_mut: flip all to unassigned
        for dv in dvv.iter_mut() {
            *dv = DecisionVar::unassigned();
        }
        assert!(dvv.iter().all(|dv| !dv.is_assigned()));
    }

    #[test]
    fn test_enumerate_yields_request_index() {
        let dvv = DecisionVarVec::from(vec![
            DecisionVar::unassigned(),
            DecisionVar::assigned(BerthIndex::new(2), tp(20)),
            DecisionVar::assigned(BerthIndex::new(3), tp(30)),
        ]);

        let items: Vec<_> = dvv.enumerate().collect();
        assert_eq!(items.len(), 3);

        // Check indices and values align
        assert_eq!(items[0].0, RequestIndex::new(0));
        assert!(matches!(items[0].1, DecisionVar::Unassigned));

        assert_eq!(items[1].0, RequestIndex::new(1));
        assert!(items[1].1.is_assigned());

        assert_eq!(items[2].0, RequestIndex::new(2));
        assert!(items[2].1.is_assigned());
    }

    #[test]
    fn test_into_iter_and_from_iter() {
        let base = vec![
            DecisionVar::assigned(BerthIndex::new(1), tp(10)),
            DecisionVar::unassigned(),
        ];
        let dvv = DecisionVarVec::from(base.clone());

        // into_iter consumes and yields items in order
        let collected: Vec<_> = dvv.clone().into_iter().collect();
        assert_eq!(collected, base);

        // from_iter builds DecisionVarVec
        let rebuilt: DecisionVarVec<_> = base.clone().into_iter().collect();
        assert_eq!(rebuilt.as_slice(), &base[..]);
    }

    #[test]
    fn test_clear_and_reserve() {
        let mut dvv: DecisionVarVec<i64> =
            DecisionVarVec::from(vec![DecisionVar::unassigned(), DecisionVar::unassigned()]);
        assert_eq!(dvv.len(), 2);

        dvv.reserve(10); // should not panic
        dvv.clear();
        assert!(dvv.is_empty());
    }

    #[test]
    fn test_deref_to_slice() {
        let dvv = DecisionVarVec::from(vec![
            DecisionVar::unassigned(),
            DecisionVar::assigned(BerthIndex::new(4), tp(40)),
        ]);

        // Deref target is [DecisionVar<T>]
        let s: &[DecisionVar<i64>] = &dvv;
        assert_eq!(s.len(), 2);
        assert!(s[1].is_assigned());
    }

    #[test]
    fn test_decision_ord_lex_by_berth_then_start() {
        let d11 = Decision::new(BerthIndex::new(1), tp(10));
        let d12 = Decision::new(BerthIndex::new(1), tp(20));
        let d01 = Decision::new(BerthIndex::new(0), tp(100));
        let d20 = Decision::new(BerthIndex::new(2), tp(0));

        // Same berth, compare by start_time
        assert!(d11 < d12);
        assert_eq!(d11.cmp(&d12), Ordering::Less);
        assert_eq!(d12.cmp(&d11), Ordering::Greater);

        // Different berths: berth decides regardless of start_time
        assert!(d01 < d12);
        assert_eq!(d01.cmp(&d12), Ordering::Less);
        assert!(d20 > d12);
        assert_eq!(d20.cmp(&d12), Ordering::Greater);

        // Equality
        let d11_dup = Decision::new(BerthIndex::new(1), tp(10));
        assert_eq!(d11, d11_dup);
        assert_eq!(d11.cmp(&d11_dup), Ordering::Equal);

        // Sorting is stable and lexicographic
        let mut v = vec![d12, d01, d20, d11];
        v.sort(); // uses Ord
        assert_eq!(v, vec![d01, d11, d12, d20]);
    }

    #[test]
    fn test_decision_partial_ord_matches_total_ord() {
        let cases = [
            (
                Decision::new(BerthIndex::new(1), tp(10)),
                Decision::new(BerthIndex::new(1), tp(20)),
            ),
            (
                Decision::new(BerthIndex::new(0), tp(100)),
                Decision::new(BerthIndex::new(1), tp(0)),
            ),
            (
                Decision::new(BerthIndex::new(2), tp(0)),
                Decision::new(BerthIndex::new(2), tp(0)),
            ),
        ];

        for (a, b) in cases {
            assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
            assert_eq!(b.partial_cmp(&a), Some(b.cmp(&a)));
        }
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    // Decision and DecisionVar should be lightweight and thread-safe for common Ts.
    assert_impl_all!(Decision<i64>: Copy, Clone, Send, Sync, std::fmt::Debug, PartialEq, Eq);
    assert_impl_all!(DecisionVar<i64>: Copy, Clone, Send, Sync, std::fmt::Debug, PartialEq, Eq);
}
