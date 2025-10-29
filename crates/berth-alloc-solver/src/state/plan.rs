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
    model::index::RequestIndex,
    state::{decisionvar::DecisionVar, fitness::FitnessDelta, terminal::delta::TerminalDelta},
};

#[derive(Debug, Clone)]
pub struct DecisionVarPatch<T> {
    pub index: RequestIndex,
    pub patch: DecisionVar<T>,
}

impl<T: Copy + Ord> DecisionVarPatch<T> {
    #[inline]
    pub const fn new(index: RequestIndex, patch: DecisionVar<T>) -> Self {
        Self { index, patch }
    }
}

impl<T: Copy + Ord + std::fmt::Display> std::fmt::Display for DecisionVarPatch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DecisionVarPatch(index: {}, patch: {})",
            self.index, self.patch
        )
    }
}

#[derive(Debug, Clone)]
pub struct Plan<'p, T: Copy + Ord> {
    pub decision_var_patches: Vec<DecisionVarPatch<T>>,
    pub terminal_delta: TerminalDelta<'p, T>,
    pub fitness_delta: FitnessDelta,
}

impl<'p, T: Copy + Ord> Plan<'p, T> {
    #[inline]
    pub fn new_delta(
        decision_var_patches: Vec<DecisionVarPatch<T>>,
        terminal_delta: TerminalDelta<'p, T>,
        fitness_delta: FitnessDelta,
    ) -> Self {
        Self {
            decision_var_patches,
            terminal_delta,
            fitness_delta,
        }
    }
}
