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

use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

use crate::{
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        fitness::Fitness,
    },
};

pub trait CostEvaluator<T: Copy + Ord>: Send {
    fn name(&self) -> &str;
    fn eval_request<'m>(
        &self,
        model: &SolverModel<'m, T>,
        request: RequestIndex,
        start_time: TimePoint<T>,
        berth_index: BerthIndex,
    ) -> Option<Cost>;

    fn eval_decision<'m>(
        &self,
        model: &SolverModel<'m, T>,
        request: RequestIndex,
        decision: &Decision<T>,
    ) -> Option<Cost> {
        self.eval_request(model, request, decision.start_time, decision.berth_index)
    }

    #[inline]
    fn eval_fitness<'m>(&self, model: &SolverModel<'m, T>, vars: &[DecisionVar<T>]) -> Fitness {
        vars.iter()
            .enumerate()
            .fold(Fitness::new(Cost::zero(), 0), |mut fitness, (i, d)| {
                match d {
                    DecisionVar::Unassigned => fitness.unassigned_requests += 1,
                    DecisionVar::Assigned(v) => {
                        fitness.cost += self.eval_decision(model, RequestIndex::new(i), v).unwrap()
                    }
                }
                fitness
            })
    }
}

pub struct DefaultCostEvaluator;
impl<T> CostEvaluator<T> for DefaultCostEvaluator
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
{
    #[inline]
    fn name(&self) -> &str {
        "DefaultCostEvaluator"
    }

    #[inline]
    fn eval_request<'m>(
        &self,
        model: &SolverModel<'m, T>,
        request: RequestIndex,
        start_time: TimePoint<T>,
        berth_index: BerthIndex,
    ) -> Option<Cost> {
        model.cost_of_assignment(request, berth_index, start_time)
    }
}
