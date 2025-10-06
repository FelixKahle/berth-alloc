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

use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{prelude::Berth, problem::req::RequestView};
use num_traits::CheckedSub;

pub trait CostPolicy {
    fn unperformed_penalty<T: Copy + Ord + CheckedSub + Into<Cost>, V: RequestView<T>>(
        &self,
        req: &V,
    ) -> Cost;
    fn scheduled_cost<T: Copy + Ord + CheckedSub + Into<Cost>, V: RequestView<T>>(
        &self,
        req: &V,
        berth: &Berth<T>,
    ) -> Option<Cost>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WeightedFlowTime;

impl CostPolicy for WeightedFlowTime {
    fn unperformed_penalty<'p, T: Copy + Ord + CheckedSub + Into<Cost>, V: RequestView<T>>(
        &self,
        req: &V,
    ) -> Cost {
        let weight = req.weight();
        weight.saturating_mul(req.feasible_window().length().value().into())
    }

    fn scheduled_cost<T: Copy + Ord + CheckedSub + Into<Cost>, V: RequestView<T>>(
        &self,
        req: &V,
        berth: &Berth<T>,
    ) -> Option<Cost> {
        let weight = req.weight();
        let processing_time_on = req.processing_time_for(berth.id())?;
        Some(weight.saturating_mul(processing_time_on.value().into()))
    }
}
