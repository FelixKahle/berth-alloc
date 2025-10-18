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
        berth::err::BerthUpdateError,
        terminal::{
            delta::TerminalDelta,
            terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite},
        },
    },
};
use berth_alloc_core::prelude::TimeInterval;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct TerminalSandbox<'p, T: Copy + Ord> {
    inner: TerminalOccupancy<'p, T>,
    touched: BTreeSet<BerthIndex>,
}

impl<'p, T: Copy + Ord> TerminalSandbox<'p, T> {
    #[inline]
    pub fn new(inner: TerminalOccupancy<'p, T>) -> Self {
        Self {
            inner,
            touched: BTreeSet::new(),
        }
    }

    #[inline]
    pub fn inner(&self) -> &TerminalOccupancy<'p, T> {
        &self.inner
    }

    #[inline]
    pub fn occupy(
        &mut self,
        index: BerthIndex,
        iv: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>> {
        self.inner.occupy(index, iv)?;
        self.touched.insert(index);
        Ok(())
    }

    #[inline]
    pub fn release(
        &mut self,
        index: BerthIndex,
        iv: TimeInterval<T>,
    ) -> Result<(), BerthUpdateError<T>> {
        self.inner.release(index, iv)?;
        self.touched.insert(index);
        Ok(())
    }

    #[inline]
    pub fn delta(&self) -> TerminalDelta<'p, T> {
        let mut updates = Vec::with_capacity(self.touched.len());

        for berth_index in self.touched.iter().copied() {
            let index = berth_index.get();
            debug_assert!(index < self.inner.berths_len());

            let occ = self.inner.berth(berth_index).cloned().unwrap();
            updates.push((berth_index, occ));
        }
        TerminalDelta::from_updates(updates)
    }
}
