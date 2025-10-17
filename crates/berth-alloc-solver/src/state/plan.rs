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
    /// Full snapshot of the proposed ledger (source of truth).
    pub ledger: Ledger<'p, T>,
    /// Sparse terminal updates to apply to the current state.
    pub terminal_delta: TerminalDelta<'p, T>,
    /// Cost change relative to the *base* state this plan was built against.
    pub delta_cost: Cost,
    /// Unassigned change (can be negative). Relative to the *base* state.
    pub delta_unassigned: i32,
}

impl<'p, T: Copy + Ord> Plan<'p, T> {
    #[inline]
    pub fn new_delta(
        ledger: Ledger<'p, T>,
        terminal_delta: TerminalDelta<'p, T>,
        delta_cost: Cost,
        delta_unassigned: i32,
    ) -> Self {
        Self {
            ledger,
            terminal_delta,
            delta_cost,
            delta_unassigned,
        }
    }

    /// Convenience: build deltas from absolute “new” values + a base snapshot.
    #[inline]
    pub fn from_absolute(
        ledger: Ledger<'p, T>,
        terminal_delta: TerminalDelta<'p, T>,
        new_cost: Cost,
        new_unassigned: usize,
        base_cost: Cost,
        base_unassigned: usize,
    ) -> Self {
        let delta_cost = new_cost - base_cost;
        let delta_unassigned = new_unassigned as i32 - base_unassigned as i32;
        Self {
            ledger,
            terminal_delta,
            delta_cost,
            delta_unassigned,
        }
    }
}
