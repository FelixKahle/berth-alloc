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

use crate::model::berth::{
    err::{BerthUpdateError, NotFreeError, OutsideAvailabilityError},
    traits::{BerthRead, BerthWrite},
};
use berth_alloc_core::prelude::{TimeInterval, TimePoint};
use berth_alloc_model::prelude::Berth;
use rangemap::RangeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BerthOccupancy<'b, T: Copy + Ord> {
    berth: &'b Berth<T>,
    free: RangeSet<TimePoint<T>>, // This is always a subset of berth.availability()
}

impl<'b, T: Copy + Ord> BerthOccupancy<'b, T> {
    #[inline]
    pub fn new(berth: &'b Berth<T>) -> Self {
        Self {
            berth,
            free: berth.availability().clone(),
        }
    }
}

impl<'b, T: Copy + Ord> BerthRead<'b, T> for BerthOccupancy<'b, T> {
    #[inline]
    fn is_free(&self, interval: TimeInterval<T>) -> bool {
        if interval.is_empty() {
            return true;
        }
        let (s, e) = interval.into_inner();
        self.free.gaps(&(s..e)).next().is_none()
    }

    #[inline]
    fn is_occupied(&self, interval: TimeInterval<T>) -> bool {
        !self.is_free(interval)
    }

    #[inline]
    fn berth(&self) -> &'b Berth<T> {
        self.berth
    }

    #[inline]
    fn iter_free_intervals_in(
        &self,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = TimeInterval<T>> {
        std::iter::once(window)
            .filter(|w| !w.is_empty())
            .flat_map(move |w| {
                let (b0, b1) = w.into_inner();
                self.free.overlapping(b0..b1).map(move |r| {
                    let s = if r.start < b0 { b0 } else { r.start };
                    let e = if r.end > b1 { b1 } else { r.end };
                    TimeInterval::new(s, e)
                })
            })
    }
}

impl<'b, T: Copy + Ord> BerthWrite<'b, T> for BerthOccupancy<'b, T> {
    #[inline]
    fn occupy(&mut self, interval: TimeInterval<T>) -> Result<(), BerthUpdateError<T>> {
        if interval.is_empty() {
            return Ok(());
        }
        if !self.berth.covers(interval) {
            return Err(OutsideAvailabilityError::new(interval).into());
        }
        if !self.is_free(interval) {
            return Err(BerthUpdateError::NotFree(NotFreeError::new(interval)));
        }

        let (s, e) = interval.into_inner();
        self.free.remove(s..e);
        Ok(())
    }

    #[inline]
    fn release(&mut self, interval: TimeInterval<T>) -> Result<(), BerthUpdateError<T>> {
        if interval.is_empty() {
            return Ok(());
        }
        if !self.berth.covers(interval) {
            return Err(OutsideAvailabilityError::new(interval).into());
        }
        let (s, e) = interval.into_inner();
        self.free.insert(s..e);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeInterval, TimePoint};
    use berth_alloc_model::prelude::BerthIdentifier;

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
    fn test_free_starts_as_availability() {
        let b = Berth::from_windows(bid(1), vec![iv(0, 10), iv(20, 30)]);
        let occ = BerthOccupancy::new(&b);
        let v: Vec<_> = occ.iter_free_intervals_in(iv(0, 50)).collect();
        assert_eq!(v, vec![iv(0, 10), iv(20, 30)]);
        assert!(occ.is_free(iv(0, 10)));
        assert!(occ.is_free(iv(22, 25)));
        assert!(!occ.is_free(iv(10, 20)));
    }

    #[test]
    fn test_occupy_then_query() {
        let b = Berth::from_windows(bid(2), vec![iv(0, 10)]);
        let mut occ = BerthOccupancy::new(&b);
        occ.occupy(iv(3, 7)).unwrap();
        let v: Vec<_> = occ.iter_free_intervals_in(iv(0, 10)).collect();
        assert_eq!(v, vec![iv(0, 3), iv(7, 10)]);
        assert!(occ.is_free(iv(0, 3)));
        assert!(occ.is_free(iv(7, 10)));
        assert!(!occ.is_free(iv(2, 8)));
    }

    #[test]
    fn test_release_outside_availability_errors_then_in_bounds_succeeds() {
        let b = Berth::from_windows(bid(3), vec![iv(10, 20)]);
        let mut occ = BerthOccupancy::new(&b);

        // Occupy everything first.
        occ.occupy(iv(10, 20)).unwrap();
        assert!(occ.iter_free_intervals_in(iv(0, 100)).next().is_none());

        // Releasing beyond availability now returns an error (no clamping).
        let err = occ.release(iv(5, 25)).unwrap_err();
        match err {
            BerthUpdateError::OutsideAvailability(e) => {
                assert_eq!(e.requested(), iv(5, 25));
            }
            _ => panic!("expected OutsideAvailability"),
        }

        // Releasing inside availability succeeds and restores the window.
        occ.release(iv(10, 20)).unwrap();
        let v: Vec<_> = occ.iter_free_intervals_in(iv(0, 100)).collect();
        assert_eq!(v, vec![iv(10, 20)]);
    }

    #[test]
    fn test_overlapping_windows_queries_only_touching_parts() {
        let b = Berth::from_windows(bid(4), vec![iv(0, 10), iv(20, 30)]);
        let occ = BerthOccupancy::new(&b);
        // Query in the middle of the gap:
        assert!(occ.iter_free_intervals_in(iv(12, 18)).next().is_none());
        // Query that spans end of first and gap:
        let v: Vec<_> = occ.iter_free_intervals_in(iv(8, 15)).collect();
        assert_eq!(v, vec![iv(8, 10)]);
    }

    #[test]
    fn test_occupy_not_free_errors() {
        let b = Berth::from_windows(bid(1), vec![iv(0, 10)]);
        let mut occ = BerthOccupancy::new(&b);

        // Occupy middle chunk.
        occ.occupy(iv(3, 7)).unwrap();

        // Trying to occupy overlapping again must error with NotFree.
        let err = occ.occupy(iv(5, 8)).unwrap_err();
        match err {
            BerthUpdateError::NotFree(e) => {
                assert_eq!(e.requested(), iv(5, 8));
            }
            _ => panic!("expected NotFree"),
        }
    }

    #[test]
    fn test_occupy_outside_availability_errors() {
        let b = Berth::from_windows(bid(2), vec![iv(0, 10)]);
        let mut occ = BerthOccupancy::new(&b);

        let err = occ.occupy(iv(9, 12)).unwrap_err();
        match err {
            BerthUpdateError::OutsideAvailability(e) => {
                assert_eq!(e.requested(), iv(9, 12));
            }
            _ => panic!("expected OutsideAvailability"),
        }
    }

    #[test]
    fn test_release_outside_availability_errors() {
        let b = Berth::from_windows(bid(3), vec![iv(10, 20)]);
        let mut occ = BerthOccupancy::new(&b);

        let err = occ.release(iv(5, 25)).unwrap_err();
        match err {
            BerthUpdateError::OutsideAvailability(e) => {
                assert_eq!(e.requested(), iv(5, 25));
            }
            _ => panic!("expected OutsideAvailability"),
        }
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    macro_rules! test_integer_types {
        ($($t:ty),*) => {
            $(
                assert_impl_all!(BerthOccupancy<'static, $t>: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
