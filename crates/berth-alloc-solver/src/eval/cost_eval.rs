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
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};

pub trait CostEvaluator<T> {
    fn unperformed_penalty(&self, request_index: RequestIndex) -> Cost;
    fn scheduled_cost(&self, request_index: RequestIndex, berth_index: BerthIndex) -> Option<Cost>;
    fn wait_cost(&self, req: RequestIndex, wait: TimeDelta<T>) -> Cost;
}

#[derive(Debug, Clone)]
pub struct WeightedTotalTurnaroundCostEvaluator {
    weights: Vec<Cost>,       // len = R
    costs: Vec<Option<Cost>>, // len = R * B
    penalties: Vec<Cost>,     // len = R
    request_count: usize,     // R
    berth_count: usize,       // B
}

impl WeightedTotalTurnaroundCostEvaluator {
    pub fn from_model<T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>>(
        model: &SolverModel<T>,
    ) -> Self {
        let request_count = model.flexible_requests_len();
        let berth_count = model.berths_len();

        let mut costs = vec![None; request_count * berth_count];
        let mut penalties = vec![Cost::zero(); request_count];

        for r in 0..request_count {
            let w: Cost = model.weights()[r];
            let len = model.feasible_intervals()[r].length();
            penalties[r] = w.saturating_mul(len.value().into());

            for b in 0..berth_count {
                if let Some(Some(dt)) =
                    model.processing_time(RequestIndex::new(r), BerthIndex::new(b))
                {
                    let dur: Cost = dt.value().into();
                    costs[r * berth_count + b] = Some(w.saturating_mul(dur));
                }
            }
        }

        Self {
            weights: model.weights().to_vec(),
            costs,
            penalties,
            request_count,
            berth_count,
        }
    }

    #[inline(always)]
    fn flat_index(&self, req: RequestIndex, berth: BerthIndex) -> usize {
        debug_assert!(req.0 < self.request_count);
        debug_assert!(berth.0 < self.berth_count);

        req.0 * self.berth_count + berth.0
    }
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>> CostEvaluator<T>
    for WeightedTotalTurnaroundCostEvaluator
{
    #[inline]
    fn unperformed_penalty(&self, request_index: RequestIndex) -> Cost {
        self.penalties[request_index.get()]
    }

    #[inline]
    fn scheduled_cost(&self, request_index: RequestIndex, berth_index: BerthIndex) -> Option<Cost> {
        self.costs[self.flat_index(request_index, berth_index)]
    }

    #[inline]
    fn wait_cost(&self, r: RequestIndex, wait: TimeDelta<T>) -> Cost {
        self.weights[r.get()].saturating_mul(wait.value().into())
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
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn req_flex(
        id: usize,
        window: (i64, i64),
        weight: i64,
        pts: &[(usize, i64)],
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    #[test]
    fn test_from_model_builds_costs_and_penalties() {
        // Two berths; requests with different weights and PTs:
        // r10: w=3, PTs: b1->5, b2->9
        // r20: w=7, PTs: b2->4 only
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);

        let r10 = req_flex(10, (0, 100), 3, &[(1, 5), (2, 9)]);
        let r20 = req_flex(20, (0, 100), 7, &[(2, 4)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r10);
        pb.add_flexible(r20);
        let p: Problem<i64> = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let eval = WeightedTotalTurnaroundCostEvaluator::from_model(&model);

        // Pin T = i64 so trait method calls are unambiguous
        let ce: &dyn CostEvaluator<i64> = &eval;

        // Index mapping is by ascending IDs:
        // r10->ri(0), r20->ri(1); b1->bi(0), b2->bi(1)
        let ri10 = RequestIndex(0);
        let ri20 = RequestIndex(1);
        let bi1 = BerthIndex(0);
        let bi2 = BerthIndex(1);

        // Penalties = weight * feasible_length = w * 100
        assert_eq!(ce.unperformed_penalty(ri10), 3 * 100);
        assert_eq!(ce.unperformed_penalty(ri20), 7 * 100);

        // Scheduled costs = weight * duration (when allowed), else None
        assert_eq!(ce.scheduled_cost(ri10, bi1), Some(3 * 5));
        assert_eq!(ce.scheduled_cost(ri10, bi2), Some(3 * 9));
        assert_eq!(ce.scheduled_cost(ri20, bi1), None);
        assert_eq!(ce.scheduled_cost(ri20, bi2), Some(7 * 4));

        // Wait costs = weight * wait
        assert_eq!(ce.wait_cost(ri10, td(6)), 3 * 6);
        assert_eq!(ce.wait_cost(ri20, td(1)), 7 * 1);
    }

    #[test]
    fn test_wait_cost_and_penalty_saturating_behavior() {
        // One berth; one request with weight = i64::MAX to test saturating mul.
        let b1 = Berth::from_windows(bid(1), [iv(0, 10)]);
        let r = req_flex(99, (0, 10), i64::MAX, &[(1, 2)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let eval = WeightedTotalTurnaroundCostEvaluator::from_model(&model);
        let ce: &dyn CostEvaluator<i64> = &eval;

        let ri = RequestIndex(0);
        let bi = BerthIndex(0);

        // scheduled_cost = MAX * 2 -> saturates to MAX
        assert_eq!(ce.scheduled_cost(ri, bi), Some(i64::MAX));

        // wait_cost = MAX * 2 -> saturates to MAX
        assert_eq!(ce.wait_cost(ri, td(2)), i64::MAX);

        // penalty = MAX * feasible_length(10) -> saturates to MAX
        assert_eq!(ce.unperformed_penalty(ri), i64::MAX);
    }

    #[test]
    fn test_scheduled_cost_none_for_disallowed_berth() {
        // Two berths; request feasible only on berth 2
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);

        let r = req_flex(7, (0, 100), 5, &[(2, 3)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let eval = WeightedTotalTurnaroundCostEvaluator::from_model(&model);
        let ce: &dyn CostEvaluator<i64> = &eval;

        let ri = RequestIndex(0);
        let bi1 = BerthIndex(0);
        let bi2 = BerthIndex(1);

        assert_eq!(ce.scheduled_cost(ri, bi1), None);
        assert_eq!(ce.scheduled_cost(ri, bi2), Some(5 * 3));
    }
}
