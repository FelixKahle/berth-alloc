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
            err::BerthUpdateError,
        },
        terminal::{
            delta::TerminalDelta,
            terminalocc::{FreeBerth, TerminalOccupancy, TerminalRead, TerminalWrite},
        },
    },
};
use berth_alloc_core::prelude::TimeInterval;
use fx_hash::FxHashMap;

#[derive(Debug, Clone)]
pub struct TerminalSandbox<'t, 'p, T: Copy + Ord> {
    base: &'t TerminalOccupancy<'p, T>,
    overrides: FxHashMap<usize, BerthOccupancy<'p, T>>,
}

impl<'t, 'p, T: Copy + Ord> TerminalSandbox<'t, 'p, T> {
    #[inline]
    pub fn new(base: &'t TerminalOccupancy<'p, T>) -> Self {
        Self {
            base,
            overrides: FxHashMap::default(),
        }
    }

    /// Compatibility shim: callers that used to do `sandbox.inner().method()`
    /// can keep doing so; `TerminalSandbox` implements `TerminalRead` directly.
    #[inline]
    pub fn inner(&self) -> &Self {
        self
    }

    /// Build a delta consisting only of touched/overridden berths.
    #[inline]
    pub fn delta(&self) -> TerminalDelta<'p, T> {
        let mut updates = Vec::with_capacity(self.overrides.len());
        for (&idx, occ) in &self.overrides {
            updates.push((BerthIndex::new(idx), occ.clone()));
        }
        TerminalDelta::from_updates(updates)
    }

    #[inline]
    fn key(idx: BerthIndex) -> usize {
        idx.get()
    }

    #[inline]
    fn eff_berth(&self, idx: BerthIndex) -> Option<&BerthOccupancy<'p, T>> {
        let k = Self::key(idx);
        if let Some(ov) = self.overrides.get(&k) {
            Some(ov)
        } else {
            self.base.berth(idx)
        }
    }

    #[inline]
    fn eff_berth_mut(&mut self, idx: BerthIndex) -> Option<&mut BerthOccupancy<'p, T>> {
        let k = Self::key(idx);
        if !self.overrides.contains_key(&k) {
            // Seed override by cloning from the base
            let base = self.base.berth(idx)?.clone();
            self.overrides.insert(k, base);
        }
        // Now guaranteed to exist
        self.overrides.get_mut(&k)
    }
}

impl<'t, 'p, T: Copy + Ord> TerminalRead<'p, T> for TerminalSandbox<'t, 'p, T> {
    // NOTE: This returns the base slice. If a caller needs override-aware
    // access they should call `berth()`/iterators which do honor overrides.
    #[inline]
    fn berths(&self) -> &[BerthOccupancy<'p, T>] {
        self.base.berths()
    }

    #[inline]
    fn berths_len(&self) -> usize {
        self.base.berths_len()
    }

    #[inline]
    fn berth(&self, idx: BerthIndex) -> Option<&BerthOccupancy<'p, T>> {
        self.eff_berth(idx)
    }

    #[inline]
    fn iter_free_intervals_for_berths_in_slice<'a>(
        &'a self,
        berths: &'a [BerthIndex],
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = FreeBerth<T>> + 'a {
        berths
            .iter()
            .copied()
            .filter_map(move |bi| self.eff_berth(bi).map(|occ| (bi, occ)))
            .flat_map(move |(bi, occ)| {
                occ.iter_free_intervals_in(window)
                    .map(move |iv| FreeBerth::new(iv, bi))
            })
    }

    #[inline]
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
            .filter_map(move |bi| self.eff_berth(bi).map(|occ| (bi, occ)))
            .flat_map(move |(bi, occ)| {
                occ.iter_free_intervals_in(window)
                    .map(move |iv| FreeBerth::new(iv, bi))
            })
    }
}

impl<'t, 'p, T: Copy + Ord> TerminalWrite<'p, T> for TerminalSandbox<'t, 'p, T> {
    #[inline]
    fn occupy(
        &mut self,
        idx: BerthIndex,
        interval: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>> {
        let occ = self
            .eff_berth_mut(idx)
            .expect("berth index out of range in sandbox occupy");
        occ.occupy(interval)
    }

    #[inline]
    fn release(
        &mut self,
        idx: BerthIndex,
        interval: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>> {
        let occ = self
            .eff_berth_mut(idx)
            .expect("berth index out of range in sandbox release");
        occ.release(interval)
    }

    #[inline]
    fn berth_mut(&mut self, index: BerthIndex) -> Option<&mut BerthOccupancy<'p, T>> {
        self.eff_berth_mut(index)
    }

    #[inline]
    fn apply_delta(
        &mut self,
        delta: TerminalDelta<'p, T>,
    ) -> Result<(), crate::state::berth::err::BerthApplyError<T>> {
        for (bi, occ) in delta.updates() {
            self.overrides.insert(bi.get(), occ.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::index::BerthIndex;
    use crate::state::terminal::delta::TerminalDelta;
    use crate::state::terminal::terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite};
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
    fn test_new_and_inner_passthrough() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let sandbox = TerminalSandbox::new(&term);

        // inner() returns &Self which implements TerminalRead
        let inner = sandbox.inner();
        assert_eq!(inner.berths_len(), term.berths_len());

        // berths() is passthrough to base slice
        assert_eq!(sandbox.berths().len(), base.len());
        for i in 0..base.len() {
            assert_eq!(sandbox.berths()[i].berth().id(), base[i].id());
        }
    }

    #[test]
    fn test_iterators_without_overrides_match_base_with_clamping() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let sandbox = TerminalSandbox::new(&term);

        // Clamp window ensures we only see parts within [8,25)
        let window = iv(8, 25);

        // On index 0 base has [0,10), [20,30) => clamped to [8,10), [20,25)
        let v_sandbox_slice: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in_slice(&[bi(0)], window)
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v_sandbox_slice, vec![iv(8, 10), iv(20, 25)]);

        // Using IntoIterator version should yield the same
        let v_sandbox_iter: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(0)], window)
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v_sandbox_iter, vec![iv(8, 10), iv(20, 25)]);
    }

    #[test]
    fn test_occupy_within_window_creates_override_and_updates_iterators() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        // Occupy [2,6) on berth index 0 which is inside [0,10).
        sandbox
            .occupy(bi(0), iv(2, 6))
            .expect("occupy should succeed");

        // The sandbox iterators must reflect the change
        let v: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        // Free intervals should be [0,2) and [6,10)
        assert_eq!(v, vec![iv(0, 2), iv(6, 10)]);

        // The base remains unchanged (berths() is passthrough to base)
        let v_base: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v_base, vec![iv(0, 10)]);
    }

    #[test]
    fn test_release_after_occupy_merges_and_respects_availability() {
        let base = vec![Berth::from_windows(bid(10), vec![iv(0, 10)])];
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        // Occupy [3,7)
        sandbox.occupy(bi(0), iv(3, 7)).unwrap();
        // Release [5,8) -> merges with tail producing free [5,10)
        sandbox.release(bi(0), iv(5, 8)).unwrap();

        let v: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 3), iv(5, 10)]);
    }

    #[test]
    fn test_occupy_outside_availability_is_error() {
        let base = vec![Berth::from_windows(bid(20), vec![iv(0, 10)])];
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        let err = sandbox.occupy(bi(0), iv(9, 15)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(
            s.contains("outside"),
            "expected outside-availability error, got: {s}"
        );
    }

    #[test]
    fn test_occupy_when_not_free_is_error() {
        let base = vec![Berth::from_windows(bid(30), vec![iv(0, 10)])];
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        sandbox.occupy(bi(0), iv(2, 6)).unwrap();
        let err = sandbox.occupy(bi(0), iv(4, 8)).unwrap_err();
        let s = err.to_string().to_lowercase();
        assert!(
            s.contains("not free"),
            "expected not-free error after overlapping occupy, got: {s}"
        );
    }

    #[test]
    fn test_delta_contains_only_overridden_berths() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        // Touch indices 0 and 2
        sandbox.occupy(bi(0), iv(0, 2)).unwrap();
        sandbox.occupy(bi(2), iv(40, 45)).unwrap();

        let d = sandbox.delta();
        let mut idxs: Vec<_> = d.updates().iter().map(|(ix, _)| ix.get()).collect();
        idxs.sort_unstable();
        assert_eq!(idxs, vec![0, 2], "delta must include only touched berths");
    }

    #[test]
    fn test_apply_delta_overrides_berths() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);

        // Build a delta manually: override berth 1 (index=1) to occupy [6,10) within its [5,15).
        let mut occ1 = term.berth(bi(1)).cloned().expect("exists");
        occ1.occupy(iv(6, 10))
            .expect("occupy in delta must succeed");
        let delta = TerminalDelta::from_updates(vec![(bi(1), occ1.clone())]);

        let mut sandbox = TerminalSandbox::new(&term);
        sandbox
            .apply_delta(delta)
            .expect("apply_delta must succeed");

        // Now sandbox should reflect the overridden occ for index 1
        let free: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(1)], iv(5, 15))
            .map(|fb| fb.interval())
            .collect();
        // Original [5,15) minus [6,10) -> [5,6), [10,15)
        assert_eq!(free, vec![iv(5, 6), iv(10, 15)]);
    }

    #[test]
    fn test_berth_mut_creates_override_and_mutates_independently_of_base() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        // Mutate via berth_mut
        {
            let occ = sandbox.berth_mut(bi(0)).expect("exists");
            // Occupy [1,3) inside [0,10)
            occ.occupy(iv(1, 3)).unwrap();
        }

        // Sandbox sees [0,1), [3,10) on index 0
        let free_sandbox: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(free_sandbox, vec![iv(0, 1), iv(3, 10)]);

        // Base remains unchanged: still [0,10)
        let free_base: Vec<_> = term
            .iter_free_intervals_for_berths_in([bi(0)], iv(0, 10))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(free_base, vec![iv(0, 10)]);
    }

    #[test]
    #[should_panic]
    fn test_unknown_index_panics_on_mutations() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let mut sandbox = TerminalSandbox::new(&term);

        // Index 999 does not exist — TerminalSandbox will panic on OOB similarly to TerminalOccupancy.
        let _ = sandbox.occupy(BerthIndex::new(999), iv(0, 1)).unwrap_err();
    }

    #[test]
    fn test_iterators_across_multiple_indices_concatenate_in_input_order() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let sandbox = TerminalSandbox::new(&term);

        // Select [0,2] in this order, window [-20,60) — should concatenate their free windows in input order.
        let v: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(0), bi(2)], iv(-20, 60))
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(v, vec![iv(0, 10), iv(20, 30), iv(-10, -5), iv(40, 50)]);
    }

    #[test]
    fn test_iterators_slice_vs_into_iterator_equivalence() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let sandbox = TerminalSandbox::new(&term);

        let indices = [bi(0), bi(1), bi(2)];
        let window = iv(-100, 100);

        let a: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in_slice(&indices, window)
            .map(|fb| fb.interval())
            .collect();
        let b: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in(indices, window)
            .map(|fb| fb.interval())
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_iterators_clamp_to_window() {
        let base = mk_berths();
        let term = TerminalOccupancy::new(&base);
        let sandbox = TerminalSandbox::new(&term);

        // Window [10,10) is empty → no intervals
        assert!(
            sandbox
                .iter_free_intervals_for_berths_in([bi(0), bi(1), bi(2)], iv(10, 10))
                .next()
                .is_none()
        );

        // Window [9, 22) — verify clamping behavior on index 0 and 1
        let v: Vec<_> = sandbox
            .iter_free_intervals_for_berths_in([bi(0), bi(1)], iv(9, 22))
            .map(|fb| fb.interval())
            .collect();
        // index 0: [0,10) -> [9,10), [20,30) -> [20,22)
        // index 1: [5,15) -> [9,15)
        assert_eq!(v, vec![iv(9, 10), iv(20, 22), iv(9, 15)]);
    }
}
