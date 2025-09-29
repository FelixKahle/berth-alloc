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

use crate::state::berth::berthocc::{BerthOccupancy, BerthRead, BerthWrite};
use crate::state::terminal::delta::TerminalDelta;
use crate::state::terminal::err::{
    BerthIdentifierNotFoundError, TerminalApplyError, TerminalUpdateError,
};
use crate::state::terminal::terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::prelude::BerthIdentifier;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

#[derive(Debug, Clone)]
pub struct TerminalSandbox<'t, 'p, T: Copy + Ord> {
    base: &'t TerminalOccupancy<'p, T>,
    touched: BTreeMap<BerthIdentifier, BerthOccupancy<'p, T>>,
}

impl<'t, 'p, T: Copy + Ord> TerminalSandbox<'t, 'p, T> {
    #[inline]
    pub fn new(base: &'t TerminalOccupancy<'p, T>) -> Self {
        Self {
            base,
            touched: BTreeMap::new(),
        }
    }

    /// Return overlay entry if present; otherwise base.
    #[inline]
    fn merged(&self, id: BerthIdentifier) -> Option<&BerthOccupancy<'p, T>> {
        self.touched.get(&id).or_else(|| self.base.berth(id))
    }

    /// Ensure berth `id` exists in overlay (clone-on-first-touch) and return &mut to it.
    #[inline]
    fn ensure_owned(
        &mut self,
        id: BerthIdentifier,
    ) -> Result<&mut BerthOccupancy<'p, T>, BerthIdentifierNotFoundError> {
        match self.touched.entry(id) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(v) => {
                let base_occ = self
                    .base
                    .berth(id)
                    .ok_or_else(|| BerthIdentifierNotFoundError::new(id))?;
                Ok(v.insert(base_occ.clone()))
            }
        }
    }

    #[inline]
    pub fn delta(self) -> Result<TerminalDelta<'p, T>, BerthIdentifierNotFoundError> {
        let updates: Vec<_> = self.touched.into_iter().collect();
        Ok(TerminalDelta::from_updates(updates))
    }
}

impl<'t, 'p, T> TerminalRead<'p, T> for TerminalSandbox<'t, 'p, T>
where
    T: Copy + Ord + 'p,
{
    /// Iterate berths in the **base order**, shadowed by overlay if present.
    #[inline]
    fn berths<'a>(&'a self) -> impl Iterator<Item = &'a BerthOccupancy<'p, T>> + 'a
    where
        'p: 'a,
        T: 'p,
    {
        self.base.berths().map(move |b| {
            let id = b.berth().id();
            self.touched.get(&id).unwrap_or(b)
        })
    }

    /// Single-berth lookup with overlay shadowing.
    #[inline]
    fn berth(&self, id: BerthIdentifier) -> Option<&BerthOccupancy<'p, T>> {
        self.merged(id)
    }

    /// Free-interval iterator using the merged view.
    #[inline]
    fn iter_free_intervals_for_berths_in<'a, I>(
        &'a self,
        berths: I,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = super::terminalocc::FreeBerth<'p, T>> + 'a
    where
        T: 'p,
        I: IntoIterator<Item = BerthIdentifier>,
        'p: 'a,
        <I as IntoIterator>::IntoIter: 'a,
    {
        use super::terminalocc::FreeBerth;

        berths
            .into_iter()
            .filter_map(move |id| self.merged(id))
            .flat_map(move |occ| {
                let berth_ref = occ.berth();
                occ.iter_free_intervals_in(window)
                    .map(move |iv| FreeBerth::new(iv, berth_ref))
            })
    }
}

impl<'t, 'p, T> TerminalWrite<'p, T> for TerminalSandbox<'t, 'p, T>
where
    T: Copy + Ord + 'p,
{
    #[inline]
    fn occupy(
        &mut self,
        berth_id: BerthIdentifier,
        interval: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>> {
        self.ensure_owned(berth_id)?.occupy(interval)?;
        Ok(())
    }

    #[inline]
    fn release(
        &mut self,
        berth_id: BerthIdentifier,
        interval: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>> {
        self.ensure_owned(berth_id)?.release(interval)?;
        Ok(())
    }

    #[inline]
    fn apply_delta(&mut self, delta: TerminalDelta<'p, T>) -> Result<(), TerminalApplyError<T>> {
        for (id, new_occ) in delta.into_iter() {
            self.touched.insert(id, new_occ);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::state::terminal::terminalocc::TerminalOccupancy;

    use super::*;
    use berth_alloc_core::prelude::{TimeInterval, TimePoint};
    use berth_alloc_model::prelude::{Berth, BerthIdentifier};

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

    fn mk_berths() -> Vec<Berth<i64>> {
        vec![
            // id:1 windows: [0,10), [20,30)
            Berth::from_windows(bid(1), vec![iv(0, 10), iv(20, 30)]),
            // id:2 windows: [5,15)
            Berth::from_windows(bid(2), vec![iv(5, 15)]),
            // id:3 windows: [0, 100)
            Berth::from_windows(bid(3), vec![iv(0, 100)]),
        ]
    }

    #[test]
    fn new_builds_overlay_lazily_and_iteration_shadows_base() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);
        let sb = TerminalSandbox::new(&base);

        // No touches yet; berths() yields the base order and objects
        let base_ids: Vec<_> = base.berths().map(|b| b.berth().id()).collect();
        let sb_ids: Vec<_> = sb.berths().map(|b| b.berth().id()).collect();
        assert_eq!(base_ids, sb_ids);

        // Lookup mirrors base before touch
        assert!(sb.berth(bid(1)).unwrap().is_free(iv(0, 10)));
        assert!(sb.berth(bid(2)).unwrap().is_free(iv(5, 15)));
    }

    #[test]
    fn occupy_release_are_clone_on_first_touch_and_local() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);
        let mut sb = TerminalSandbox::new(&base);

        // Occupy only in sandbox
        assert!(sb.occupy(bid(1), iv(2, 4)).is_ok());
        // Base unaffected
        assert!(base.berth(bid(1)).unwrap().is_free(iv(2, 4)));
        // Sandbox sees the change
        assert!(sb.berth(bid(1)).unwrap().is_occupied(iv(2, 4)));

        // Release in sandbox (merge back free)
        assert!(sb.release(bid(1), iv(2, 4)).is_ok());
        assert!(sb.berth(bid(1)).unwrap().is_free(iv(2, 4)));

        // Touch another berth and keep both tracked
        assert!(sb.occupy(bid(2), iv(6, 8)).is_ok());
        assert!(sb.berth(bid(2)).unwrap().is_occupied(iv(6, 8)));
        // Base still unchanged on berth 2
        assert!(base.berth(bid(2)).unwrap().is_free(iv(6, 8)));
    }

    #[test]
    fn iter_free_intervals_uses_merged_overlay() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);
        let mut sb = TerminalSandbox::new(&base);

        // Occupy [6,8) on berth 1 only in sandbox
        assert!(sb.occupy(bid(1), iv(6, 8)).is_ok());

        // Query in window [0,10) should reflect sandbox occupancy
        let window = iv(0, 10);
        let free: Vec<_> = sb
            .iter_free_intervals_for_berths_in([bid(1)], window)
            .collect();

        // There should be free intervals and they correspond to id=1
        assert!(!free.is_empty());
        assert!(free.iter().all(|fb| fb.berth().id() == bid(1)));
    }

    #[test]
    fn apply_delta_merges_overlays() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);
        let mut sb = TerminalSandbox::new(&base);

        // Prepare a delta that marks berth 2 as fully free [5,15) (i.e., base clone)
        let cloned = base.berth(bid(2)).unwrap().clone();
        let delta = TerminalDelta::from_updates(vec![(bid(2), cloned)]);
        assert!(sb.apply_delta(delta).is_ok());

        // Now overlay shadows base for berth 2
        assert!(sb.berth(bid(2)).unwrap().is_free(iv(5, 15)));
    }

    #[test]
    fn delta_produces_only_touched_updates() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);

        // Build a small sandbox, touch berths 1 and 3
        let mut sb = TerminalSandbox::new(&base);
        assert!(sb.occupy(bid(1), iv(2, 4)).is_ok());
        assert!(sb.occupy(bid(3), iv(10, 20)).is_ok());

        let delta = sb.delta().expect("delta");
        let got: Vec<_> = delta.iter().map(|(id, _)| *id).collect();
        let mut expected = vec![bid(1), bid(3)];
        expected.sort();
        let mut got_sorted = got.clone();
        got_sorted.sort();
        assert_eq!(got_sorted, expected);
    }

    #[test]
    fn berth_lookup_shadowing_and_missing_id_errors() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);
        let mut sb = TerminalSandbox::new(&base);

        // Touch berth 1, then lookup should return overlay (clone-on-first-touch mutated state)
        assert!(sb.occupy(bid(1), iv(1, 2)).is_ok());
        let o1 = sb.berth(bid(1)).unwrap();
        let b1 = base.berth(bid(1)).unwrap();

        // Overlay reflects the local mutation
        assert!(o1.is_occupied(iv(1, 2)));
        assert!(o1.is_free(iv(0, 1)));
        assert!(o1.is_free(iv(2, 10)));

        // Base remains unchanged
        assert!(b1.is_free(iv(1, 2)));

        // A missing id should propagate an error through occupy/release
        let bad = BerthIdentifier::new(99999);
        let e1 = sb.occupy(bad, iv(0, 1)).unwrap_err();
        match e1 {
            TerminalUpdateError::BerthIdentifierNotFound(e) => {
                assert_eq!(e.identifier(), bad);
            }
            other => panic!("expected not found error, got {other:?}"),
        }

        let e2 = sb.release(bad, iv(0, 1)).unwrap_err();
        match e2 {
            TerminalUpdateError::BerthIdentifierNotFound(e) => {
                assert_eq!(e.identifier(), bad);
            }
            other => panic!("expected not found error, got {other:?}"),
        }
    }

    #[test]
    fn berths_iterator_yields_base_order_with_overrides() {
        let bs = mk_berths();
        let base = TerminalOccupancy::new(&bs);
        let mut sb = TerminalSandbox::new(&base);

        // Touch berth 3 only
        assert!(sb.occupy(bid(3), iv(10, 20)).is_ok());

        // Iterate and confirm order equals base order, but entry 3 is overlay
        let base_ids: Vec<_> = base.berths().map(|b| b.berth().id()).collect();
        let sb_ids: Vec<_> = sb.berths().map(|b| b.berth().id()).collect();
        assert_eq!(base_ids, sb_ids);

        // And the occupancy for id=3 matches the overlay state
        let from_iter = sb
            .berths()
            .find(|b| b.berth().id() == bid(3))
            .unwrap()
            .clone();
        assert!(from_iter.is_occupied(iv(10, 20)));
    }
}
