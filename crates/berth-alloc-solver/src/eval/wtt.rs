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
    eval::objective::Objective,
    state::{
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightedTurnaroundTimeObjective;

impl<T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>> Objective<T>
    for WeightedTurnaroundTimeObjective
{
    fn assignment_cost(
        &self,
        model: &SolverModel<'_, T>,
        request_index: RequestIndex,
        berth_index: BerthIndex,
        _start_time: TimePoint<T>,
    ) -> Option<Cost> {
        let processing_time = model
            .processing_time(request_index, berth_index)
            .flatten()?;
        let weight = model.weights()[request_index.get()];
        Some(weight.saturating_mul(processing_time.value().into()))
    }

    fn unassignment_cost(
        &self,
        model: &SolverModel<'_, T>,
        request: RequestIndex,
    ) -> berth_alloc_core::prelude::Cost {
        let weight = model.weights()[request.get()];
        let feasible_window_length = model.feasible_intervals()[request.get()].length();
        weight.saturating_mul(feasible_window_length.value().into())
    }
}
