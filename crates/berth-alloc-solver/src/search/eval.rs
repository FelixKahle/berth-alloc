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

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    // Build a small instance for evaluator behavior:
    //
    // Berths:
    //   B1: [0, 100)
    //   B2: [0, 100)
    //
    // Requests (flexible):
    //   R10: window [0, 100), weight 3, PT: {B2: 4}    // only berth 2 feasible
    //   R20: window [0, 100), weight 7, PT: {B1: 5, B2: 9}
    fn make_problem_with_requests() -> Problem<i64> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);

        let mut pt_r10 = BTreeMap::new();
        pt_r10.insert(bid(2), td(4));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 3, pt_r10)
            .expect("r10 should be feasible");

        let mut pt_r20 = BTreeMap::new();
        pt_r20.insert(bid(1), td(5));
        pt_r20.insert(bid(2), td(9));
        let r20 = Request::<FlexibleKind, i64>::new(rid(20), iv(0, 100), 7, pt_r20)
            .expect("r20 should be feasible");

        let mut builder = ProblemBuilder::new();
        // Insert in reverse to ensure we don't depend on insertion order
        builder.add_berth(b2);
        builder.add_berth(b1);
        builder.add_flexible(r20);
        builder.add_flexible(r10);

        builder.build().expect("valid problem")
    }

    #[test]
    fn test_default_cost_evaluator_name() {
        let ev = DefaultCostEvaluator;
        assert_eq!(
            <DefaultCostEvaluator as CostEvaluator<i64>>::name(&ev),
            "DefaultCostEvaluator"
        );
    }

    #[test]
    fn test_eval_request_and_decision_some_and_none() {
        let problem = make_problem_with_requests();
        let model = SolverModel::try_from(&problem).expect("solver model creation should succeed");
        let ev = DefaultCostEvaluator;

        let im = model.index_manager();

        // R10 on B2 at t=10:
        // - arrival = 0
        // - waiting = 10
        // - PT = 4
        // - turnaround = 14
        // - weight = 3
        // - cost = 3 * 14 = 42
        let r10 = im.request_index(rid(10)).expect("r10 index");
        let b2 = im.berth_index(bid(2)).expect("b2 index");

        let c1 = ev
            .eval_request(&model, r10, tp(10), b2)
            .expect("should be feasible");
        assert_eq!(c1, 42);

        let d = Decision::new(b2, tp(10));
        let c2 = ev
            .eval_decision(&model, r10, &d)
            .expect("decision should yield same cost");
        assert_eq!(c2, 42);

        // Start before arrival => waiting_time = None => cost None
        let none_cost = ev.eval_request(&model, r10, tp(-1), b2);
        assert!(none_cost.is_none());
    }

    #[test]
    fn test_eval_fitness_counts_and_costs() {
        let problem = make_problem_with_requests();
        let model = SolverModel::try_from(&problem).expect("solver model creation should succeed");
        let ev = DefaultCostEvaluator;
        let im = model.index_manager();

        let r10 = im.request_index(rid(10)).expect("r10 index");
        let r20 = im.request_index(rid(20)).expect("r20 index");
        let b1 = im.berth_index(bid(1)).expect("b1 index");
        let b2 = im.berth_index(bid(2)).expect("b2 index");

        // Prepare decision vars aligned with model's request index order.
        let mut vars = vec![DecisionVar::<i64>::Unassigned; model.flexible_requests_len()];

        // Assign R10 on B2 at t=6:
        // waiting = 6, PT=4, turnaround=10, weight=3 => cost=30
        vars[r10.get()] = DecisionVar::assigned(b2, tp(6));
        // R20 remains unassigned for now
        let fit1 = ev.eval_fitness(&model, &vars);
        assert_eq!(fit1.cost, 30);
        assert_eq!(fit1.unassigned_requests, 1);

        // Now assign R20 on B1 at t=2:
        // waiting = 2, PT=5, turnaround=7, weight=7 => cost=49
        // Total = 30 + 49 = 79, unassigned = 0
        vars[r20.get()] = DecisionVar::assigned(b1, tp(2));
        let fit2 = ev.eval_fitness(&model, &vars);
        assert_eq!(fit2.cost, 79);
        assert_eq!(fit2.unassigned_requests, 0);
    }
}
