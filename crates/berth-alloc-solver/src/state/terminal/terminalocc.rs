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
    model::index::BerthIndex,
    state::{
        berth::{
            berthocc::{BerthOccupancy, BerthRead, BerthWrite},
            err::{BerthApplyError, BerthUpdateError},
        },
        terminal::delta::TerminalDelta,
    },
};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]

pub struct FreeBerth<T: Copy + Ord> {
    interval: TimeInterval<T>,
    berth_index: BerthIndex,
}

impl<T: Copy + Ord> FreeBerth<T> {
    #[inline]
    pub fn new(interval: TimeInterval<T>, berth_index: BerthIndex) -> Self {
        Self {
            interval,
            berth_index,
        }
    }
    #[inline]
    pub fn interval(&self) -> TimeInterval<T> {
        self.interval
    }
    #[inline]
    pub fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }
}

pub trait TerminalRead<'b, T: Copy + Ord> {
    fn berths(&self) -> &[BerthOccupancy<'b, T>];
    fn berths_len(&self) -> usize;
    fn berth(&self, idx: BerthIndex) -> Option<&BerthOccupancy<'b, T>>;

    fn iter_free_intervals_for_berths_in_slice<'a>(
        &'a self,
        berths: &'a [BerthIndex],
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<T>> + 'a;

    fn iter_free_intervals_for_berths_in<'a, I>(
        &'a self,
        berths: I,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<T>> + 'a
    where
        I: IntoIterator<Item = BerthIndex> + 'a;
}

pub trait TerminalWrite<'b, T: Copy + Ord>: TerminalRead<'b, T> {
    fn occupy(
        &mut self,
        idx: BerthIndex,
        interval: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>>;

    fn release(
        &mut self,
        idx: BerthIndex,
        interval: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>>;

    fn berth_mut(&mut self, index: BerthIndex) -> Option<&mut BerthOccupancy<'b, T>>;

    fn apply_delta(&mut self, delta: TerminalDelta<'b, T>) -> Result<(), BerthApplyError<T>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalOccupancy<'b, T: Copy + Ord> {
    berths: Vec<BerthOccupancy<'b, T>>,
}

impl<'b, T: Copy + Ord> TerminalOccupancy<'b, T> {
    #[inline]
    pub fn from_slice(berths: &'b [Berth<T>]) -> Self {
        Self::new(berths)
    }

    #[inline]
    pub fn new<I>(berths: I) -> Self
    where
        I: IntoIterator<Item = &'b Berth<T>>,
    {
        let berths_occ: Vec<_> = berths.into_iter().map(BerthOccupancy::new).collect();
        Self { berths: berths_occ }
    }
}

impl<'b, T> TerminalRead<'b, T> for TerminalOccupancy<'b, T>
where
    T: Copy + Ord + 'b,
{
    #[inline]
    fn berths(&self) -> &[BerthOccupancy<'b, T>] {
        &self.berths
    }

    #[inline]
    fn berth(&self, index: BerthIndex) -> Option<&BerthOccupancy<'b, T>> {
        self.berths.get(index.get())
    }

    fn iter_free_intervals_for_berths_in<'a, I>(
        &'a self,
        berths: I,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<T>> + 'a
    where
        I: IntoIterator<Item = BerthIndex> + 'a,
    {
        berths
            .into_iter()
            .filter_map(|ix| self.berth(ix).map(move |occ| (ix, occ)))
            .flat_map(move |(ix, occ)| {
                occ.iter_free_intervals_in(window)
                    .map(move |iv| FreeBerth::new(iv, ix))
            })
    }

    fn iter_free_intervals_for_berths_in_slice<'a>(
        &'a self,
        berths: &'a [BerthIndex],
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<T>> + 'a {
        berths
            .iter()
            .filter_map(|ix| self.berth(*ix).map(|occ| (ix, occ)))
            .flat_map(move |(ix, occ)| {
                occ.iter_free_intervals_in(window)
                    .map(move |iv| FreeBerth::new(iv, *ix))
            })
    }

    #[inline]
    fn berths_len(&self) -> usize {
        self.berths.len()
    }
}

impl<'b, T> TerminalWrite<'b, T> for TerminalOccupancy<'b, T>
where
    T: Copy + Ord + 'b,
{
    #[inline]
    fn occupy(
        &mut self,
        berth_index: BerthIndex,
        interval: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>> {
        let index = berth_index.get();
        debug_assert!(index < self.berths.len());

        self.berths[index].occupy(interval)
    }

    #[inline]
    fn release(
        &mut self,
        idx: BerthIndex,
        interval: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>> {
        let index = idx.get();
        debug_assert!(index < self.berths.len());

        self.berths[index].release(interval)
    }

    #[inline]
    fn apply_delta(&mut self, delta: TerminalDelta<'b, T>) -> Result<(), BerthApplyError<T>> {
        for (berth_index, free) in delta {
            let index = berth_index.get();
            debug_assert!(index < self.berths.len());

            self.berths[index].apply(free)?;
        }
        Ok(())
    }

    #[inline]
    fn berth_mut(&mut self, index: BerthIndex) -> Option<&mut BerthOccupancy<'b, T>> {
        self.berths.get_mut(index.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeInterval, TimePoint};

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex::new(n)
    }

    fn mk_berths() -> Vec<Berth<i64>> {
        vec![
            // index 0 (id:1): windows: [0,10), [20,30)
            Berth::from_windows(bid(1), vec![iv(0, 10), iv(20, 30)]),
            // index 1 (id:2): windows: [5,15)
            Berth::from_windows(bid(2), vec![iv(5, 15)]),
            // index 2 (id:3): windows: [-10,-5), [40,50)
            Berth::from_windows(bid(3), vec![iv(-10, -5), iv(40, 50)]),
        ]
    }

    #[test]
    fn test_new_builds_index_and_exposes_berths() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        assert_eq!(term.berths().len(), base.len());
        for i in 0..base.len() {
            let got = term.berth(bi(i)).expect("berth index must exist");
            // The underlying berth is the one we inserted at that index
            assert_eq!(got.berth().id(), base[i].id());
        }
    }

    #[test]
    fn test_iter_free_across_selected_indices() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Select indices 0 then 2, expect concatenation of their free windows in that order
        let idxs = [bi(0), bi(2)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(idxs, iv(-20, 60))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10), iv(20, 30), iv(-10, -5), iv(40, 50)]);
    }

    #[test]
    fn test_iter_free_clamps_to_window() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Window [8,25) clamps index 0’s [0,10) to [8,10) and [20,30) to [20,25)
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(8, 25))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(8, 10), iv(20, 25)]);
    }

    #[test]
    fn test_iter_free_empty_window_yields_none() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        assert!(
            term.iter_free_intervals_for_berths_in([bi(0), bi(1), bi(2)], iv(10, 10))
                .next()
                .is_none()
        );
    }

    #[test]
    fn test_occupy_then_query_then_release() {
        let base = vec![Berth::from_windows(bid(10), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        // Occupy [3,7) on index 0.
        term.occupy(bi(0), iv(3, 7)).unwrap();

        // Now free must be [0,3) and [7,10)
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 3), iv(7, 10)]);

        // Release [5,8) — merges with the [7,10) tail, yielding [5,10)
        term.release(bi(0), iv(5, 8)).unwrap();
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v2, vec![iv(0, 3), iv(5, 10)]);
    }

    #[test]
    fn test_occupy_outside_availability_is_error() {
        let base = vec![Berth::from_windows(bid(20), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        let err = term.occupy(bi(0), iv(9, 15)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(
            s.contains("outside"),
            "expected outside-availability error, got: {s}"
        );
    }

    #[test]
    fn test_release_outside_availability_is_error() {
        let base = vec![Berth::from_windows(bid(21), vec![iv(10, 20)])];
        let mut term = TerminalOccupancy::new(&base);

        let err = term.release(bi(0), iv(0, 25)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(
            s.contains("outside"),
            "expected outside-availability error, got: {s}"
        );
    }

    #[test]
    fn test_occupy_when_not_free_is_error() {
        let base = vec![Berth::from_windows(bid(30), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        term.occupy(bi(0), iv(2, 6)).unwrap();
        let err = term.occupy(bi(0), iv(4, 8)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(s.contains("not free"), "expected not-free error, got: {s}");
    }

    #[test]
    #[should_panic]
    fn test_unknown_index_panics_on_mutations() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // Index 999 doesn’t exist — current implementation will panic on OOB.
        let _ = term.occupy(bi(999), iv(0, 1)).unwrap_err();
    }

    #[test]
    fn test_mutate_one_berth_does_not_affect_others() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // Occupy index 1 fully ([5,15) for that berth).
        term.occupy(bi(1), iv(5, 15)).unwrap();

        // Index 0 remains intact.
        let v0: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v0, vec![iv(0, 10), iv(20, 30)]);

        // Index 1 becomes empty within its window.
        let v1: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(1)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert!(v1.is_empty());

        // Index 2 remains intact.
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(2)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v2, vec![iv(-10, -5), iv(40, 50)]);
    }

    #[test]
    fn test_zero_length_mutations_are_noops() {
        let base = vec![Berth::from_windows(bid(77), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        // Occupy and release empty interval should both be Ok and change nothing.
        term.occupy(bi(0), iv(5, 5)).unwrap();
        term.release(bi(0), iv(7, 7)).unwrap();

        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10)]);
    }

    #[test]
    fn test_iter_free_mixed_order_indices_yields_concatenated() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Note the mixed order: 2, 0, 1 – by index
        let idxs = [bi(2), bi(0), bi(1)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(idxs, iv(-20, 60))
            .map(|fb| fb.interval())
            .collect();

        // Expect windows from idx2, then idx0, then idx1.
        assert_eq!(
            v,
            vec![iv(-10, -5), iv(40, 50), iv(0, 10), iv(20, 30), iv(5, 15)]
        );
    }

    #[test]
    fn test_iter_free_indices_respects_order_and_window() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // order: 2, 0, (skip unknown), 1
        let idxs = [bi(2), bi(0), bi(1)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(idxs, iv(-20, 60))
            .map(|fb| (fb.berth_index(), fb.interval()))
            .collect();

        assert_eq!(
            v,
            vec![
                (bi(2), iv(-10, -5)),
                (bi(2), iv(40, 50)),
                (bi(0), iv(0, 10)),
                (bi(0), iv(20, 30)),
                (bi(1), iv(5, 15)),
            ]
        );

        // clamp check for a single index
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(8, 25))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v2, vec![iv(8, 10), iv(20, 25)]);
    }

    #[test]
    fn test_apply_delta_updates_multiple_berths() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // Prepare updates:
        // - For idx 0 (id:1), occupy [3,7) -> free becomes [0,3) and [7,10) for that berth
        // - For idx 2 (id:3), occupy [40,50) -> free becomes [-10,-5)
        let mut occ0 = term.berth(bi(0)).cloned().expect("exists");
        occ0.occupy(iv(3, 7)).unwrap();

        let mut occ2 = term.berth(bi(2)).cloned().expect("exists");
        occ2.occupy(iv(40, 50)).unwrap();

        let delta = crate::state::terminal::delta::TerminalDelta::from_updates(vec![
            (bi(0), occ0),
            (bi(2), occ2),
        ]);
        term.apply_delta(delta).expect("apply_delta must succeed");

        // Verify idx 0 changed as expected
        let v0: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v0, vec![iv(0, 3), iv(7, 10), iv(20, 30)]);

        // Verify idx 1 unchanged
        let v1: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(1)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v1, vec![iv(5, 15)]);

        // Verify idx 2 changed as expected
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(2)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v2, vec![iv(-10, -5)]);
    }

    #[test]
    fn test_apply_delta_mismatched_ids_errors_and_does_not_mutate() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // Create an occupancy with a different berth id than index 0 (which is id:1)
        let wrong_berth = Berth::from_windows(bid(999), vec![iv(0, 10), iv(20, 30)]);
        let occ_wrong = BerthOccupancy::new(&wrong_berth);

        let delta =
            crate::state::terminal::delta::TerminalDelta::from_updates(vec![(bi(0), occ_wrong)]);
        let err = term.apply_delta(delta).unwrap_err();
        match err {
            BerthApplyError::MismatchedBerthIds(_) => {}
            _ => panic!("expected MismatchedBerthIds error, got: {err:?}"),
        }

        // Verify berth index 0 remains unchanged
        let v0: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v0, vec![iv(0, 10), iv(20, 30)]);
    }

    #[test]
    fn test_berth_mut_allows_direct_mutation() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // Mutate berth at index 1 (id:2) directly
        let b1 = term.berth_mut(bi(1)).expect("exists");
        b1.occupy(iv(5, 10)).unwrap();

        // Now free on index 1 should be [10,15)
        let v1: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(1)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v1, vec![iv(10, 15)]);
    }

    #[test]
    fn test_iter_free_for_slice_skips_unknown_indices_and_keeps_order() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        let idxs = [bi(0), BerthIndex::new(999), bi(2)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in_slice(&idxs, iv(-100, 100))
            .map(|fb| (fb.berth_index(), fb.interval()))
            .collect();

        // Unknown index is skipped; order is preserved for known indices.
        assert_eq!(
            v,
            vec![
                (bi(0), iv(0, 10)),
                (bi(0), iv(20, 30)),
                (bi(2), iv(-10, -5)),
                (bi(2), iv(40, 50)),
            ]
        );
    }

    #[test]
    fn test_berths_len_and_berth_none_for_out_of_range() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        assert_eq!(term.berths_len(), base.len());
        assert!(term.berth(BerthIndex::new(999)).is_none());
    }
}
