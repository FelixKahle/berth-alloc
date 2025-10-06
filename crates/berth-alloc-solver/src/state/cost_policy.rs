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

use crate::state::{
    index::{BerthIndex, RequestIndex},
    model::SolverModel,
};
use berth_alloc_core::prelude::{Cost, TimeDelta};
use num_traits::{CheckedAdd, CheckedSub};

pub trait CostPolicy<T: Copy + Ord> {
    fn unperformed_penalty(&self, request_index: RequestIndex) -> Cost;
    fn scheduled_cost(&self, request_index: RequestIndex, berth_index: BerthIndex) -> Option<Cost>;
    fn wait_cost(&self, req: RequestIndex, wait: TimeDelta<T>) -> Cost;
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedFlowTime<'model, 'problem, T: Copy + Ord> {
    model: &'model SolverModel<'problem, T>,
}

impl<'problem, 'model, T: Copy + Ord> WeightedFlowTime<'model, 'problem, T> {
    #[inline]
    pub fn new(model: &'model SolverModel<'problem, T>) -> Self {
        Self { model }
    }
}

impl<'problem, 'model, T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>> CostPolicy<T>
    for WeightedFlowTime<'model, 'problem, T>
{
    #[inline]
    fn unperformed_penalty(&self, request_index: RequestIndex) -> Cost {
        let w: Cost = self.model.weights()[request_index.get()];
        let len: Cost = self.model.feasible_intervals()[request_index.get()]
            .length()
            .value()
            .into();
        w.saturating_mul(len)
    }

    #[inline]
    fn scheduled_cost(&self, request_index: RequestIndex, berth_index: BerthIndex) -> Option<Cost> {
        let Some(Some(dt)) = self.model.processing_time(request_index, berth_index) else {
            return None;
        };
        let w: Cost = self.model.weights()[request_index.get()];
        let dur: Cost = dt.value().into();
        Some(w.saturating_mul(dur))
    }

    fn wait_cost(&self, r: RequestIndex, wait: TimeDelta<T>) -> Cost {
        let w = self.model.weights()[r.get()];
        let wcost: Cost = wait.value().into();
        w.saturating_mul(wcost)
    }
}
