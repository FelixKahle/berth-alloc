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

use crate::state::{registry::ledger::Ledger, terminal::delta::TerminalDelta};
use berth_alloc_core::prelude::Cost;

#[derive(Debug, Clone)]
pub struct Plan<'p, T: Copy + Ord> {
    pub ledger: Ledger<'p, T>,
    pub terminal_delta: TerminalDelta<'p, T>,
    pub delta_cost: Cost,
    pub unassigned: usize,
}

impl<'p, T: Copy + Ord> Plan<'p, T> {
    #[inline]
    pub fn new(
        ledger: Ledger<'p, T>,
        terminal_delta: TerminalDelta<'p, T>,
        delta_cost: Cost,
        unassigned: usize,
    ) -> Self {
        Self {
            ledger,
            terminal_delta,
            delta_cost,
            unassigned,
        }
    }
}
