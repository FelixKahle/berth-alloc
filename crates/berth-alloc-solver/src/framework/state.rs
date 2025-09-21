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
    registry::ledger::Ledger,
    terminal::terminalocc::TerminalOccupancy,
};

#[derive(Debug, Clone)]
pub struct SolverState<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
}

impl<'p, T: Copy + Ord> SolverState<'p, T> {
    pub fn new(ledger: Ledger<'p, T>, terminal_occupancy: TerminalOccupancy<'p, T>) -> Self {
        Self {
            ledger,
            terminal_occupancy,
        }
    }

    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    pub fn ledger_mut(&mut self) -> &mut Ledger<'p, T> {
        &mut self.ledger
    }

    pub fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    pub fn terminal_occupancy_mut(&mut self) -> &mut TerminalOccupancy<'p, T> {
        &mut self.terminal_occupancy
    }

    pub fn apply(&mut self, other: Self) {
        self.ledger.apply(other.ledger);
        //self.terminal_occupancy.apply(other.terminal_occupancy);
    }
}
