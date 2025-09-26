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

use crate::{registry::ledger::Ledger, terminal::terminalocc::TerminalOccupancy};

#[derive(Debug, Clone)]
pub struct TerminalLedger<'problem, T: Copy + Ord> {
    ledger: Ledger<'problem, T>,
    terminal: TerminalOccupancy<'problem, T>,
}

impl<'problem, T: Copy + Ord> TerminalLedger<'problem, T> {
    #[inline]
    pub fn new(ledger: Ledger<'problem, T>, terminal: TerminalOccupancy<'problem, T>) -> Self {
        Self { ledger, terminal }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'problem, T> {
        &self.ledger
    }

    #[inline]
    pub fn terminal(&self) -> &TerminalOccupancy<'problem, T> {
        &self.terminal
    }
}
