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
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};

#[derive(Debug, Clone, PartialEq)]
pub struct SearchObjective<R> {
    base: R,
    lambda: f64, // λ ≥ 0.0
}

impl<R> SearchObjective<R> {
    #[inline]
    pub fn new(base: R, lambda: f64) -> Self {
        Self {
            base,
            lambda: lambda.max(0.0),
        }
    }

    #[inline]
    pub fn set_base(&mut self, base: R) {
        self.base = base;
    }

    #[inline]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda.max(0.0);
    }

    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    #[inline]
    pub fn base(&self) -> &R {
        &self.base
    }

    /// Change λ on the fly (e.g. per annealing band).
    #[inline]
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda.max(0.0);
        self
    }
}

#[inline(always)]
fn scale_cost(base: Cost, lambda: f64) -> Cost {
    if lambda <= 0.0 {
        return base;
    }
    let mult = 1.0 + lambda.max(0.0);
    let x = (base as f64) * mult;
    if x.is_finite() {
        let r = x.round();
        if r <= 0.0 {
            0
        } else if r >= (Cost::MAX as f64) {
            Cost::MAX
        } else {
            r as Cost
        }
    } else {
        Cost::MAX
    }
}

impl<T, R> Objective<T> for SearchObjective<R>
where
    T: Copy + Ord,
    R: Objective<T>,
{
    #[inline]
    fn assignment_cost(
        &self,
        model: &SolverModel<'_, T>,
        request_index: RequestIndex,
        berth_index: BerthIndex,
        start_time: TimePoint<T>,
    ) -> Option<Cost> {
        self.base
            .assignment_cost(model, request_index, berth_index, start_time)
    }

    #[inline]
    fn unassignment_cost(&self, model: &SolverModel<'_, T>, request_index: RequestIndex) -> Cost {
        let base = self.base.unassignment_cost(model, request_index);
        scale_cost(base, self.lambda)
    }
}
