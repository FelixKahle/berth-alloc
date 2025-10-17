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

use crate::state::{
    berth::berthocc::{BerthOccupancy, BerthRead, BerthWrite},
    terminal::{
        delta::TerminalDelta,
        err::{BerthIdentifierNotFoundError, TerminalApplyError, TerminalUpdateError},
    },
};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeBerth<'b, T: Copy + Ord> {
    interval: TimeInterval<T>,
    berth: &'b Berth<T>,
}

impl<'b, T: Copy + Ord> FreeBerth<'b, T> {
    fn new(interval: TimeInterval<T>, berth: &'b Berth<T>) -> Self {
        Self { interval, berth }
    }

    pub fn interval(&self) -> TimeInterval<T> {
        self.interval
    }

    pub fn berth(&self) -> &'b Berth<T> {
        self.berth
    }
}

pub trait TerminalRead<'b, T: Copy + Ord> {
    fn berths(&self) -> &[BerthOccupancy<'b, T>];
    fn berth(&self, id: BerthIdentifier) -> Option<&BerthOccupancy<'b, T>>;

    fn iter_free_intervals_for_berths_in<'a, I>(
        &'a self,
        berths: I,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<'b, T>> + 'a
    where
        T: 'b,
        I: IntoIterator<Item = BerthIdentifier>,
        'b: 'a,
        <I as IntoIterator>::IntoIter: 'a;
}

pub trait TerminalWrite<'b, T: Copy + Ord>: TerminalRead<'b, T> {
    fn occupy(
        &mut self,
        berth_id: BerthIdentifier,
        interval: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>>;

    fn release(
        &mut self,
        berth_id: BerthIdentifier,
        interval: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>>;

    fn apply_delta(&mut self, delta: TerminalDelta<'b, T>) -> Result<(), TerminalApplyError<T>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalOccupancy<'b, T: Copy + Ord> {
    berths: Vec<BerthOccupancy<'b, T>>,
    index_map: HashMap<BerthIdentifier, usize>,
}

impl<'b, T: Copy + Ord> TerminalOccupancy<'b, T> {
    pub fn new<I>(berths: I) -> Self
    where
        I: IntoIterator<Item = &'b Berth<T>>,
    {
        let berths_occ: Vec<_> = berths.into_iter().map(BerthOccupancy::new).collect();
        let index_map = berths_occ
            .iter()
            .enumerate()
            .map(|(i, occ)| (occ.berth().id(), i))
            .collect();

        Self {
            berths: berths_occ,
            index_map,
        }
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
    fn berth(&self, id: BerthIdentifier) -> Option<&BerthOccupancy<'b, T>> {
        self.index_map.get(&id).map(|&i| &self.berths[i])
    }

    fn iter_free_intervals_for_berths_in<'a, I>(
        &'a self,
        berths: I,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<'b, T>> + 'a
    where
        I: IntoIterator<Item = BerthIdentifier>,
        'b: 'a,
        <I as IntoIterator>::IntoIter: 'a,
    {
        let occs = &self.berths;
        let index_map = &self.index_map;

        berths
            .into_iter()
            .filter_map(move |id| index_map.get(&id).copied())
            .flat_map(move |ix| {
                let berth_ref = occs[ix].berth();
                occs[ix]
                    .iter_free_intervals_in(window)
                    .map(move |iv| FreeBerth::new(iv, berth_ref))
            })
    }
}

impl<'b, T> TerminalWrite<'b, T> for TerminalOccupancy<'b, T>
where
    T: Copy + Ord + 'b,
{
    #[inline]
    fn occupy(
        &mut self,
        berth_id: BerthIdentifier,
        interval: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>> {
        let ix = self
            .index_map
            .get(&berth_id)
            .copied()
            .ok_or_else(|| BerthIdentifierNotFoundError::new(berth_id))?;

        let occ = self
            .berths
            .get_mut(ix)
            .ok_or_else(|| BerthIdentifierNotFoundError::new(berth_id))?;

        occ.occupy(interval).map_err(Into::into)
    }

    #[inline]
    fn release(
        &mut self,
        berth_id: BerthIdentifier,
        interval: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>> {
        let ix = self
            .index_map
            .get(&berth_id)
            .copied()
            .ok_or_else(|| BerthIdentifierNotFoundError::new(berth_id))?;

        let occ = self
            .berths
            .get_mut(ix)
            .ok_or_else(|| BerthIdentifierNotFoundError::new(berth_id))?;

        occ.release(interval).map_err(Into::into)
    }

    #[inline]
    fn apply_delta(&mut self, delta: TerminalDelta<'b, T>) -> Result<(), TerminalApplyError<T>> {
        for (id, free) in delta.into_iter() {
            let ix = self
                .index_map
                .get(&id)
                .copied()
                .ok_or_else(|| BerthIdentifierNotFoundError::new(id))?;

            let bocc = self
                .berths
                .get_mut(ix)
                .ok_or_else(|| BerthIdentifierNotFoundError::new(id))?;

            bocc.apply(free)?;
        }
        Ok(())
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

    fn mk_berths() -> Vec<Berth<i64>> {
        vec![
            // id:1 windows: [0,10), [20,30)
            Berth::from_windows(bid(1), vec![iv(0, 10), iv(20, 30)]),
            // id:2 windows: [5,15)
            Berth::from_windows(bid(2), vec![iv(5, 15)]),
            // id:3 windows: [-10,-5), [40,50)
            Berth::from_windows(bid(3), vec![iv(-10, -5), iv(40, 50)]),
        ]
    }

    #[test]
    fn new_builds_index_and_exposes_berths() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        assert_eq!(term.berths().len(), base.len());
        for b in &base {
            let got = term.berth(b.id()).expect("berth id must exist");
            // same id exposed back out
            assert_eq!(got.berth().id(), b.id());
        }
    }

    #[test]
    fn iter_free_across_selected_ids() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Select berth 1 then 3 (by id), expect concatenation of their free windows
        let ids = [bid(1), bid(3)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(ids, iv(-20, 60))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10), iv(20, 30), iv(-10, -5), iv(40, 50)]);
    }

    #[test]
    fn iter_free_clamps_to_window() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Window [8,25) clamps berth 1’s [0,10) to [8,10) and [20,30) to [20,25)
        let ids = [bid(1)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(ids, iv(8, 25))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(8, 10), iv(20, 25)]);
    }

    #[test]
    fn iter_free_empty_window_yields_none() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        let ids = [bid(1), bid(2), bid(3)];
        assert!(
            term.iter_free_intervals_for_berths_in(ids, iv(10, 10))
                .next()
                .is_none()
        );
    }

    #[test]
    fn occupy_then_query_then_release() {
        let base = vec![Berth::from_windows(bid(10), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        // Occupy [3,7) on berth 10.
        term.occupy(bid(10), iv(3, 7)).unwrap();

        // Now free must be [0,3) and [7,10)
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(10)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 3), iv(7, 10)]);

        // Release [5,8) — merges with the [7,10) tail, yielding [5,10)
        term.release(bid(10), iv(5, 8)).unwrap();
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(10)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v2, vec![iv(0, 3), iv(5, 10)]);
    }

    #[test]
    fn occupy_outside_availability_is_error() {
        let base = vec![Berth::from_windows(bid(20), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        let err = term.occupy(bid(20), iv(9, 15)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(
            s.contains("outside"),
            "expected outside-availability error, got: {s}"
        );
    }

    #[test]
    fn release_outside_availability_is_error() {
        let base = vec![Berth::from_windows(bid(21), vec![iv(10, 20)])];
        let mut term = TerminalOccupancy::new(&base);

        let err = term.release(bid(21), iv(0, 25)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(
            s.contains("outside"),
            "expected outside-availability error, got: {s}"
        );
    }

    #[test]
    fn occupy_when_not_free_is_error() {
        let base = vec![Berth::from_windows(bid(30), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        term.occupy(bid(30), iv(2, 6)).unwrap();
        let err = term.occupy(bid(30), iv(4, 8)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(s.contains("not free"), "expected not-free error, got: {s}");
    }

    #[test]
    fn unknown_berth_id_is_error_for_mutations() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // ID 999 doesn’t exist.
        let err1 = term.occupy(bid(999), iv(0, 1)).unwrap_err();
        let err2 = term.release(bid(999), iv(0, 1)).unwrap_err();

        let s1 = err1.to_string().to_lowercase();
        let s2 = err2.to_string().to_lowercase();
        assert!(s1.contains("not found"), "expected not-found; got: {s1}");
        assert!(s2.contains("not found"), "expected not-found; got: {s2}");
    }

    #[test]
    fn mutate_one_berth_does_not_affect_others() {
        let base = mk_berths();
        let mut term = TerminalOccupancy::new(&base);

        // Occupy berth 2 fully.
        term.occupy(bid(2), iv(5, 15)).unwrap();

        // Berth 1 remains intact.
        let v1: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(1)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v1, vec![iv(0, 10), iv(20, 30)]);

        // Berth 2 becomes empty within its window.
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(2)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert!(v2.is_empty());

        // Berth 3 remains intact.
        let v3: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(3)], iv(-100, 100))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v3, vec![iv(-10, -5), iv(40, 50)]);
    }

    #[test]
    fn zero_length_mutations_are_noops() {
        let base = vec![Berth::from_windows(bid(77), vec![iv(0, 10)])];
        let mut term = TerminalOccupancy::new(&base);

        // Occupy and release empty interval should both be Ok and change nothing.
        term.occupy(bid(77), iv(5, 5)).unwrap();
        term.release(bid(77), iv(7, 7)).unwrap();

        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(77)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10)]);
    }

    #[test]
    fn iter_free_mixed_order_ids_yields_concatenated_per_id() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Note the mixed order: 3, 1, 2 – by id
        let ids = [bid(3), bid(1), bid(2)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(ids, iv(-20, 60))
            .map(|fb| fb.interval())
            .collect();

        // Expect windows from 3, then 1, then 2.
        assert_eq!(
            v,
            vec![iv(-10, -5), iv(40, 50), iv(0, 10), iv(20, 30), iv(5, 15)]
        );
    }

    #[test]
    fn iter_free_ids_respects_order_and_window() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // order: 3, 1, 999 (unknown), 2
        let ids = [bid(3), bid(1), bid(999), bid(2)];
        let v: Vec<_> = term
            .iter_free_intervals_for_berths_in(ids, iv(-20, 60))
            .map(|fb| (fb.berth().id(), fb.interval()))
            .collect();

        assert_eq!(
            v,
            vec![
                (bid(3), iv(-10, -5)),
                (bid(3), iv(40, 50)),
                (bid(1), iv(0, 10)),
                (bid(1), iv(20, 30)),
                (bid(2), iv(5, 15)),
            ]
        );

        // clamp check
        let v2: Vec<_> = term
            .iter_free_intervals_for_berths_in([bid(1)], iv(8, 25))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v2, vec![iv(8, 10), iv(20, 25)]);
    }
}
