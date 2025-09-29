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

#![allow(dead_code)]

use berth_alloc_core::prelude::{TimeDelta, TimePoint};
use berth_alloc_model::prelude::Problem;
use num_traits::{CheckedAdd, CheckedSub};

use crate::state::{
    chain::double_chain::DoubleChain,
    index::{BerthIndex, RequestIndex},
    solver::{err::SolverModelBuildError, solver_model::SolverModel},
    terminal::terminalocc::TerminalOccupancy,
};

pub struct SolverState<'b, T: Copy + Ord> {
    model: SolverModel<T>,
    chain: DoubleChain,
    terminal: TerminalOccupancy<'b, T>, // seeded with berth availability (+ fixed)
    start: Vec<Option<TimePoint<T>>>,   // per request/node
    end: Vec<Option<TimePoint<T>>>,
}

impl<'b, T: Copy + Ord + CheckedAdd + CheckedSub> SolverState<'b, T> {
    pub fn new(problem: &'b Problem<T>) -> Result<Self, SolverModelBuildError> {
        let model: SolverModel<T> = problem.try_into()?;

        let r = model.requests_len();
        let b = model.berths_len();

        let chain = DoubleChain::new(r, b);
        let terminal = TerminalOccupancy::new(problem.berths().iter());

        let num_nodes = model.requests_len();
        let start = vec![None; num_nodes];
        let end = vec![None; num_nodes];

        Ok(Self {
            model,
            chain,
            terminal,
            start,
            end,
        })
    }

    #[inline]
    pub fn model(&self) -> &SolverModel<T> {
        &self.model
    }

    #[inline]
    pub fn chain(&self) -> &DoubleChain {
        &self.chain
    }

    #[inline]
    pub fn chain_mut(&mut self) -> &mut DoubleChain {
        &mut self.chain
    }

    #[inline]
    pub fn terminal(&self) -> &TerminalOccupancy<'b, T> {
        &self.terminal
    }

    #[inline]
    pub fn terminal_mut(&mut self) -> &mut TerminalOccupancy<'b, T> {
        &mut self.terminal
    }

    #[inline]
    fn processing_time(&self, n: RequestIndex, b: BerthIndex) -> Option<Option<TimeDelta<T>>> {
        self.model
            .processing_time(RequestIndex(n.0), BerthIndex(b.0))
    }
}
