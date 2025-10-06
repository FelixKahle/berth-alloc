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

use crate::state::{chain_set::base::ChainSet, cost_policy::CostPolicy, model::SolverModel};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone)]
pub struct SolverSearchState<'model, 'problem, T: Copy + Ord, C: CostPolicy<T>> {
    model: &'model SolverModel<'problem, T>,
    cost_policy: C,
    chain_set: ChainSet,
}

impl<'problem, 'model, T: Copy + Ord + CheckedAdd + CheckedSub, C: CostPolicy<T>>
    SolverSearchState<'model, 'problem, T, C>
{
    #[inline]
    pub fn new(model: &'model SolverModel<'problem, T>, cost_policy: C) -> Self {
        let num_chains = model.berths_len();
        let num_nodes = model.flexible_requests_len();

        Self {
            model,
            cost_policy,
            chain_set: ChainSet::new(num_chains, num_nodes),
        }
    }

    #[inline]
    pub fn model(&self) -> &'model SolverModel<'problem, T> {
        self.model
    }

    #[inline]
    pub fn chain_set(&self) -> &ChainSet {
        &self.chain_set
    }

    #[inline]
    pub fn cost_policy(&self) -> &C {
        &self.cost_policy
    }
}
