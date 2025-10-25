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

use super::features::FeatureExtractor;
use super::penalty::PenaltyStore;
use crate::{
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    search::planner::{CostEvaluator, DefaultCostEvaluator},
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use smallvec::SmallVec;
use std::sync::Arc;

/// Cost evaluator wrapper: base + λ * sum(penalties(features)).
#[derive(Clone)]
pub struct AugmentedCostEvaluator<B, T, FX: ?Sized>
where
    FX: FeatureExtractor<T>,
{
    base: B,
    penalties: PenaltyStore,
    lambda_cost: Cost,
    feats: Arc<FX>,
    _phantom: std::marker::PhantomData<T>,
}

impl<B, T, FX: ?Sized> AugmentedCostEvaluator<B, T, FX>
where
    FX: FeatureExtractor<T>,
{
    pub fn new(base: B, penalties: PenaltyStore, lambda_cost: Cost, feats: Arc<FX>) -> Self {
        Self {
            base,
            penalties,
            lambda_cost,
            feats,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Tnum, B, FX: ?Sized> CostEvaluator<Tnum> for AugmentedCostEvaluator<B, Tnum, FX>
where
    Tnum: Copy + Ord,
    B: CostEvaluator<Tnum>,
    FX: FeatureExtractor<Tnum>,
{
    fn eval_request<'m>(
        &self,
        model: &SolverModel<'m, Tnum>,
        request: RequestIndex,
        start_time: TimePoint<Tnum>,
        berth_index: BerthIndex,
    ) -> Option<Cost> {
        let base = self
            .base
            .eval_request(model, request, start_time, berth_index)?;
        let mut buf: SmallVec<[super::features::Feature; 6]> = SmallVec::new();
        self.feats
            .features_for(request, berth_index, start_time, &mut buf);
        let p = self.penalties.sum(buf.iter()) as Cost;
        Some(base.saturating_add(self.lambda_cost.saturating_mul(p)))
    }
}

/// Compute fitness(state) + λ * sum penalties over assigned features.
#[inline]
pub fn augmented_cost_of_state<T, FX>(
    state: &SolverState<'_, T>,
    feats: &FX,
    store: &PenaltyStore,
    lambda_cost: Cost,
) -> Cost
where
    T: Copy + Ord,
    FX: FeatureExtractor<T> + ?Sized,
{
    use super::features::Feature;
    let mut buf: SmallVec<[Feature; 6]> = SmallVec::new();
    let mut acc: i64 = 0;

    for (i, dv) in state.decision_variables().iter().enumerate() {
        if let DecisionVar::Assigned(Decision {
            berth_index,
            start_time,
        }) = *dv
        {
            buf.clear();
            feats.features_for(RequestIndex::new(i), berth_index, start_time, &mut buf);
            acc = acc.saturating_add(store.sum(buf.iter()));
        }
    }
    state
        .fitness()
        .cost
        .saturating_add(lambda_cost.saturating_mul(acc as Cost))
}

/// Convenience builder commonly used by local planners.
pub fn make_augmented_default<T>(
    penalties: PenaltyStore,
    lambda_cost: Cost,
    feats: Arc<impl FeatureExtractor<T> + Send + Sync + 'static>,
) -> AugmentedCostEvaluator<DefaultCostEvaluator, T, dyn FeatureExtractor<T>> {
    AugmentedCostEvaluator::new(DefaultCostEvaluator, penalties, lambda_cost, feats)
}

#[cfg(test)]
mod tests {
    use super::super::features::{DefaultFeatureExtractor as DefFX, Feature, FeatureExtractor};
    use super::super::penalty::PenaltyStore;
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
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
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_one_berth_one_flex(weight: i64, pt: i64) -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 200), &[(1, pt)], weight));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn problem_one_berth_two_flex(p1: i64, w1: i64, p2: i64, w2: i64) -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));
        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 200), &[(1, p1)], w1));
        flex.insert(flex_req(2, (0, 200), &[(1, p2)], w2));

        Problem::new(berths, fixed, flex).unwrap()
    }

    // ---------- tiny custom feature extractors for deterministic tests ----------

    // Always returns a fixed set of features, regardless of inputs.
    #[derive(Clone)]
    struct FixedFeaturesExtractor {
        fixed: Vec<Feature>,
    }
    impl FixedFeaturesExtractor {
        fn new(fixed: Vec<Feature>) -> Self {
            Self { fixed }
        }
    }
    impl FeatureExtractor<i64> for FixedFeaturesExtractor {
        fn features_for(
            &self,
            _request: crate::model::index::RequestIndex,
            _berth: crate::model::index::BerthIndex,
            _start_time: TimePoint<i64>,
            out: &mut smallvec::SmallVec<[Feature; 6]>,
        ) {
            out.extend(self.fixed.iter().cloned());
        }
    }

    // Returns only Feature::Request { req } per assignment; ignores berth/time.
    #[derive(Clone, Default)]
    struct PerRequestExtractor;
    impl FeatureExtractor<i64> for PerRequestExtractor {
        fn features_for(
            &self,
            request: crate::model::index::RequestIndex,
            _berth: crate::model::index::BerthIndex,
            _start_time: TimePoint<i64>,
            out: &mut smallvec::SmallVec<[Feature; 6]>,
        ) {
            out.push(Feature::Request { req: request.get() });
        }
    }

    #[test]
    fn test_eval_adds_lambda_times_penalties_on_fixed_features() {
        // Model: 1 berth, 1 request; weight=1, pt=5 -> base cost at t=0 is 5.
        let prob = problem_one_berth_one_flex(1, 5);
        let model = SolverModel::try_from(&prob).expect("model ok");

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let t0 = tp(0);

        // Feature set: RB(0,0) with p=3, TB(1) with p=2 => sum=5
        let feats = FixedFeaturesExtractor::new(vec![
            Feature::RequestBerth {
                req: r_ix.get(),
                berth: b_ix.get(),
            },
            Feature::TimeBucket { tb: 1 },
        ]);

        let mut store = PenaltyStore::new();
        store.map.insert(
            Feature::RequestBerth {
                req: r_ix.get(),
                berth: b_ix.get(),
            },
            3,
        );
        store.map.insert(Feature::TimeBucket { tb: 1 }, 2);

        let lambda = 10;
        let eval = AugmentedCostEvaluator::new(
            DefaultCostEvaluator,
            store.clone(),
            lambda,
            Arc::new(feats),
        );

        let base = model
            .cost_of_assignment(r_ix, b_ix, t0)
            .expect("base cost defined");
        assert_eq!(base, 5, "sanity: base cost");

        let aug = eval
            .eval_request(&model, r_ix, t0, b_ix)
            .expect("augmented eval must be Some");
        // base(5) + lambda(10) * sum(5) = 55
        assert_eq!(aug, 55);
    }

    #[test]
    fn test_augmented_cost_of_state_sums_all_assigned_features() {
        // Model: 1 berth, 2 requests; arbitrary weights and pts (not used by augmented-of-state).
        let prob = problem_one_berth_two_flex(5, 1, 3, 1);
        let model = SolverModel::try_from(&prob).expect("model ok");

        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        // Build a state with 2 assigned requests on the same berth (times arbitrary).
        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(b_ix, tp(0)),
            DecisionVar::assigned(b_ix, tp(10)),
        ]);
        let term = TerminalOccupancy::new(prob.berths().iter());
        let base_cost = 100; // arbitrary base fitness cost
        let st = SolverState::new(dv, term, Fitness::new(base_cost, 0));

        // Feature extractor: only Feature::Request { req } per assignment.
        let feats = PerRequestExtractor::default();

        // Penalties: Request{0} = 4, Request{1} = 7; sum over two assigned = 11
        let mut store = PenaltyStore::new();
        store.map.insert(Feature::Request { req: 0 }, 4);
        store.map.insert(Feature::Request { req: 1 }, 7);

        let lambda = 2;
        let aug_state_cost = super::augmented_cost_of_state(&st, &feats, &store, lambda);
        // base_cost(100) + lambda(2) * sum(11) = 122
        assert_eq!(aug_state_cost, 122);
    }

    #[test]
    fn make_augmented_default_type_erased_extractor_works() {
        // Model: 1 berth, 1 request; weight=2, pt=5 -> base cost at t=0 is 10.
        let prob = problem_one_berth_one_flex(2, 5);
        let model = SolverModel::try_from(&prob).expect("model ok");

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let t0 = tp(0);

        // Default extractor with only TimeBucket enabled, bucketizer returns 42 → Feature::TimeBucket{42}
        fn bucketizer(_: TimePoint<i64>) -> i64 {
            42
        }
        let feats: DefFX<i64> = DefFX::new(bucketizer)
            .set_include_req_berth(false)
            .set_include_berth_time(false)
            .set_include_request(false)
            .set_include_berth(false)
            .set_include_time(true)
            .set_include_req_time(false);

        // Penalty only on that time bucket
        let mut store = PenaltyStore::new();
        store.map.insert(Feature::TimeBucket { tb: 42 }, 5);

        let lambda = 3;
        let eval = super::make_augmented_default::<i64>(store, lambda, Arc::new(feats));

        let base = model
            .cost_of_assignment(r_ix, b_ix, t0)
            .expect("base cost defined");
        assert_eq!(base, 10, "sanity: base cost");

        let aug = eval
            .eval_request(&model, r_ix, t0, b_ix)
            .expect("augmented eval must be Some");
        // base(10) + lambda(3)*penalty(5) = 25
        assert_eq!(aug, 25);
    }
}
