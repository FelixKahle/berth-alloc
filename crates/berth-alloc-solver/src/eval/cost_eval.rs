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

pub trait CostEvaluator {
    fn unperformed_penalty(&self, request_index: RequestIndex) -> Cost;
    fn scheduled_cost(&self, request_index: RequestIndex, berth_index: BerthIndex) -> Option<Cost>;
    fn wait_cost<T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>>(
        &self,
        req: RequestIndex,
        wait: TimeDelta<T>,
    ) -> Cost;
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

    #[inline]
    pub fn request_count(&self) -> usize {
        self.request_count
    }

    #[inline]
    pub fn berth_count(&self) -> usize {
        self.berth_count
    }

    #[inline]
    pub fn weights(&self) -> &[Cost] {
        &self.weights
    }

    #[inline]
    pub fn costs(&self) -> &[Option<Cost>] {
        &self.costs
    }

    #[inline]
    pub fn penalties(&self) -> &[Cost] {
        &self.penalties
    }
}

impl CostEvaluator for WeightedTotalTurnaroundCostEvaluator {
    #[inline]
    fn unperformed_penalty(&self, request_index: RequestIndex) -> Cost {
        self.penalties[request_index.get()]
    }

    #[inline]
    fn scheduled_cost(&self, request_index: RequestIndex, berth_index: BerthIndex) -> Option<Cost> {
        self.costs[self.flat_index(request_index, berth_index)]
    }

    #[inline]
    fn wait_cost<T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>>(
        &self,
        r: RequestIndex,
        wait: TimeDelta<T>,
    ) -> Cost {
        self.weights[r.get()].saturating_mul(wait.value().into())
    }
}

pub struct DynamicCostEvaluator<'eval> {
    base: &'eval WeightedTotalTurnaroundCostEvaluator,
    lambda_unperf_q16: u32,
    lambda_wait_q16: u32,
    req_boost_q16: Box<[u32]>, // len = R
}

impl<'eval> DynamicCostEvaluator<'eval> {
    #[inline]
    pub fn new(base: &'eval WeightedTotalTurnaroundCostEvaluator) -> Self {
        let request_count = base.request_count();
        Self {
            base,
            lambda_unperf_q16: 1 << 16,
            lambda_wait_q16: 1 << 16,
            req_boost_q16: vec![1 << 16; request_count].into_boxed_slice(),
        }
    }

    #[inline]
    pub fn set_global_unperf_scale_q16(&mut self, q16: u32) {
        self.lambda_unperf_q16 = q16;
    }
    #[inline]
    pub fn set_global_wait_scale_q16(&mut self, q16: u32) {
        self.lambda_wait_q16 = q16;
    }

    #[inline]
    pub fn bump_request_boost_q16(&mut self, r: RequestIndex, factor_q16: u32) {
        let (a, b) = (self.req_boost_q16[r.get()] as u64, factor_q16 as u64);
        let prod = (a * b) >> 16;
        self.req_boost_q16[r.get()] = prod.min(u32::MAX as u64) as u32;
    }
}

#[inline(always)]
fn scale_q16_sat(x: Cost, q16: u32) -> Cost {
    if q16 == 0 || x == 0 {
        return 0;
    }
    let q = q16 as i64;
    let x_div = x >> 16;
    let x_rem = x - (x_div << 16);
    let high = x_div.saturating_mul(q);
    let low = (x_rem * q) >> 16;
    high.saturating_add(low)
}

impl<'eval> CostEvaluator for DynamicCostEvaluator<'eval> {
    #[inline(always)]
    fn unperformed_penalty(&self, r: RequestIndex) -> Cost {
        let base = self.base.unperformed_penalty(r);
        let s1 = scale_q16_sat(base, self.lambda_unperf_q16);
        scale_q16_sat(s1, self.req_boost_q16[r.get()])
    }

    #[inline(always)]
    fn scheduled_cost(&self, r: RequestIndex, b: BerthIndex) -> Option<Cost> {
        self.base.scheduled_cost(r, b)
    }

    #[inline(always)]
    fn wait_cost<T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>>(
        &self,
        r: RequestIndex,
        wait: TimeDelta<T>,
    ) -> Cost {
        let base = self.base.wait_cost(r, wait);
        scale_q16_sat(base, self.lambda_wait_q16)
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

    // -------------------- Base evaluator tests --------------------

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

        // Index mapping is by ascending IDs:
        // r10->ri(0), r20->ri(1); b1->bi(0), b2->bi(1)
        let ri10 = RequestIndex(0);
        let ri20 = RequestIndex(1);
        let bi1 = BerthIndex(0);
        let bi2 = BerthIndex(1);

        // Penalties = weight * feasible_length = w * 100
        assert_eq!(eval.unperformed_penalty(ri10), 3 * 100);
        assert_eq!(eval.unperformed_penalty(ri20), 7 * 100);

        // Scheduled costs = weight * duration (when allowed), else None
        assert_eq!(eval.scheduled_cost(ri10, bi1), Some(3 * 5));
        assert_eq!(eval.scheduled_cost(ri10, bi2), Some(3 * 9));
        assert_eq!(eval.scheduled_cost(ri20, bi1), None);
        assert_eq!(eval.scheduled_cost(ri20, bi2), Some(7 * 4));

        // Wait costs = weight * wait
        assert_eq!(eval.wait_cost(ri10, td(6)), 3 * 6);
        assert_eq!(eval.wait_cost(ri20, td(1)), 7 * 1);
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

        let ri = RequestIndex(0);
        let bi = BerthIndex(0);

        // scheduled_cost = MAX * 2 -> saturates to MAX
        assert_eq!(eval.scheduled_cost(ri, bi), Some(i64::MAX));

        // wait_cost = MAX * 2 -> saturates to MAX
        assert_eq!(eval.wait_cost(ri, td(2)), i64::MAX);

        // penalty = MAX * feasible_length(10) -> saturates to MAX
        assert_eq!(eval.unperformed_penalty(ri), i64::MAX);
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

        let ri = RequestIndex(0);
        let bi1 = BerthIndex(0);
        let bi2 = BerthIndex(1);

        assert_eq!(eval.scheduled_cost(ri, bi1), None);
        assert_eq!(eval.scheduled_cost(ri, bi2), Some(5 * 3));
    }

    // -------------------- Dynamic evaluator tests --------------------

    #[test]
    fn test_dynamic_unperformed_and_wait_scaling() {
        // One berth; one request
        // base penalty = w*len = 10*100=1000
        // base wait cost for wait=4: 10*4 = 40
        // apply unperf scale 0.5 and wait scale 2.0
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r = req_flex(10, (0, 100), 10, &[(1, 3)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let base = WeightedTotalTurnaroundCostEvaluator::from_model(&model);
        let mut dyn_eval = DynamicCostEvaluator::new(&base);

        let ri = RequestIndex(0);

        // Q16 factors
        let half = 1u32 << 15; // 0.5
        let two = 1u32 << 17; // 2.0

        dyn_eval.set_global_unperf_scale_q16(half);
        dyn_eval.set_global_wait_scale_q16(two);

        // unperf: 1000 * 0.5 = 500
        assert_eq!(dyn_eval.unperformed_penalty(ri), 500);

        // wait: 40 * 2 = 80
        assert_eq!(dyn_eval.wait_cost(ri, td(4)), 80);
    }

    #[test]
    fn test_dynamic_request_boost_accumulates() {
        // base penalty = 8*100 = 800
        // boost 1.5 then 1.5 => 2.25 total => 800 * 2.25 = 1800
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r = req_flex(1, (0, 100), 8, &[(1, 2)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let base = WeightedTotalTurnaroundCostEvaluator::from_model(&model);
        let mut dyn_eval = DynamicCostEvaluator::new(&base);

        let ri = RequestIndex(0);

        // 1.5 in Q16
        let one_point_five = (1u32 << 16) + (1u32 << 15); // 98304
        dyn_eval.bump_request_boost_q16(ri, one_point_five);
        dyn_eval.bump_request_boost_q16(ri, one_point_five);

        // expected 800 * 2.25 = 1800
        assert_eq!(dyn_eval.unperformed_penalty(ri), 1800);
    }

    #[test]
    fn test_dynamic_scheduled_cost_passthrough() {
        // scheduled_cost should be the same as base
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let r = req_flex(7, (0, 100), 5, &[(1, 3)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let base = WeightedTotalTurnaroundCostEvaluator::from_model(&model);
        let dyn_eval = DynamicCostEvaluator::new(&base);

        let ri = RequestIndex(0);
        let bi = BerthIndex(0);

        assert_eq!(dyn_eval.scheduled_cost(ri, bi), base.scheduled_cost(ri, bi));
    }

    #[test]
    fn test_dynamic_saturation_limits() {
        // Large values saturate properly
        // base penalty = MAX * 10 -> saturates to MAX; scaling keeps MAX
        // wait cost = MAX * 2 -> MAX; scaling 2.0 -> still MAX with saturation
        let b1 = Berth::from_windows(bid(1), [iv(0, 10)]);
        let r = req_flex(42, (0, 10), i64::MAX, &[(1, 2)]);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r);
        let p = pb.build().unwrap();

        let model = SolverModel::try_from(&p).unwrap();
        let base = WeightedTotalTurnaroundCostEvaluator::from_model(&model);
        let mut dyn_eval = DynamicCostEvaluator::new(&base);

        let ri = RequestIndex(0);

        // set scales > 1.0 to stress saturation
        let two = 1u32 << 17;
        dyn_eval.set_global_unperf_scale_q16(two);
        dyn_eval.set_global_wait_scale_q16(two);

        // penalty stays saturated
        assert_eq!(dyn_eval.unperformed_penalty(ri), i64::MAX);

        // wait stays saturated
        assert_eq!(dyn_eval.wait_cost(ri, td(2)), i64::MAX);
    }
}
