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

use crate::state::terminal::{
    delta::TerminalDelta,
    err::{BerthIdentifierNotFoundError, TerminalUpdateError},
    terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite},
};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::prelude::BerthIdentifier;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct TerminalSandbox<'p, T: Copy + Ord> {
    inner: TerminalOccupancy<'p, T>,
    touched: BTreeSet<BerthIdentifier>,
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
        id: BerthIdentifier,
        iv: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>> {
        self.inner.occupy(id, iv)?;
        self.touched.insert(id);
        Ok(())
    }

    #[inline]
    pub fn release(
        &mut self,
        id: BerthIdentifier,
        iv: TimeInterval<T>,
    ) -> Result<(), TerminalUpdateError<T>> {
        self.inner.release(id, iv)?;
        self.touched.insert(id);
        Ok(())
    }

    #[inline]
    pub fn delta(&self) -> Result<TerminalDelta<'p, T>, BerthIdentifierNotFoundError> {
        let mut updates = Vec::with_capacity(self.touched.len());
        for &id in &self.touched {
            let Some(occ) = self.inner.berth(id).cloned() else {
                return Err(BerthIdentifierNotFoundError::new(id));
            };
            updates.push((id, occ));
        }
        Ok(TerminalDelta::from_updates(updates))
    }
}
