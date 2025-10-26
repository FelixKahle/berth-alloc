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
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor},
        neighbors,
        operators::{LocalPool, SoftmaxSelector},
        search::{SearchContext, SearchStrategy},
        strategy_support::{
            MedianHistoryEpsilon, StaleTracker, materially_better, patience_from_pulse_threshold,
        },
    },
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    search::{
        operator::LocalMoveOperator,
        operator_library::local::{
            CascadeInsertPolicy, CascadeRelocateK, CrossExchangeAcrossBerths,
            CrossExchangeBestAcrossBerths, HillClimbBestSwapSameBerth, HillClimbRelocateBest,
            OrOptBlockRelocate, RandomRelocateAnywhere, RandomizedGreedyRelocateRcl,
            RelocateSingleBest, RelocateSingleBestAllowWorsening, ShiftEarlierOnSameBerth,
            SwapPairSameBerth,
        },
        planner::{CostEvaluator, DefaultCostEvaluator, PlanningContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::ToPrimitive;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::{Arc, atomic::Ordering as AtomicOrdering};

// ===============================
// Features & penalties
// ===============================

/// Penalizable feature keys for GLS.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Feature {
    /// (request, berth)
    RequestBerth { req: usize, berth: usize },
    /// request only (vessel-scoped)
    Request { req: usize },
    /// berth only
    Berth { berth: usize },
    /// global time bucket
    TimeBucket { tb: i64 },
    /// (berth, time bucket)
    BerthTime { berth: usize, tb: i64 },
    /// (request, time bucket)
    RequestTime { req: usize, tb: i64 },
}

pub trait FeatureExtractor<T> {
    fn features_for(
        &self,
        request: RequestIndex,
        berth: BerthIndex,
        start_time: TimePoint<T>,
        out: &mut SmallVec<[Feature; 6]>,
    );
}

/// Configurable extractor using a time bucketizer.
#[derive(Clone)]
pub struct DefaultFeatureExtractor<T, FBucket>
where
    FBucket: Fn(TimePoint<T>) -> i64 + Send + Sync + Clone,
{
    bucketizer: FBucket,
    include_req_berth: bool,
    include_request: bool,
    include_berth: bool,
    include_time: bool,
    include_berth_time: bool,
    include_req_time: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, FBucket> DefaultFeatureExtractor<T, FBucket>
where
    FBucket: Fn(TimePoint<T>) -> i64 + Send + Sync + Clone,
{
    pub fn new(bucketizer: FBucket) -> Self {
        Self {
            bucketizer,
            include_req_berth: true,
            include_request: false,
            include_berth: false,
            include_time: true,
            include_berth_time: true,
            include_req_time: false,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn set_include_req_berth(mut self, yes: bool) -> Self {
        self.include_req_berth = yes;
        self
    }
    pub fn set_include_request(mut self, yes: bool) -> Self {
        self.include_request = yes;
        self
    }
    pub fn set_include_berth(mut self, yes: bool) -> Self {
        self.include_berth = yes;
        self
    }
    pub fn set_include_time(mut self, yes: bool) -> Self {
        self.include_time = yes;
        self
    }
    pub fn set_include_berth_time(mut self, yes: bool) -> Self {
        self.include_berth_time = yes;
        self
    }
    pub fn set_include_req_time(mut self, yes: bool) -> Self {
        self.include_req_time = yes;
        self
    }
}

impl<T, FBucket> FeatureExtractor<T> for DefaultFeatureExtractor<T, FBucket>
where
    FBucket: Fn(TimePoint<T>) -> i64 + Send + Sync + Clone,
{
    #[inline]
    fn features_for(
        &self,
        req: RequestIndex,
        berth: BerthIndex,
        start_time: TimePoint<T>,
        out: &mut SmallVec<[Feature; 6]>,
    ) {
        let r = req.get();
        let b = berth.get();
        let tb = (self.bucketizer)(start_time);

        if self.include_req_berth {
            out.push(Feature::RequestBerth { req: r, berth: b });
        }
        if self.include_request {
            out.push(Feature::Request { req: r });
        }
        if self.include_berth {
            out.push(Feature::Berth { berth: b });
        }
        if self.include_time {
            out.push(Feature::TimeBucket { tb });
        }
        if self.include_berth_time {
            out.push(Feature::BerthTime { berth: b, tb });
        }
        if self.include_req_time {
            out.push(Feature::RequestTime { req: r, tb });
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DecayMode {
    Multiplicative { num: u32, den: u32 },
    Subtractive { step: i64 },
}

#[derive(Clone)]
pub struct PenaltyStore {
    map: HashMap<Feature, i64>,
    decay: Option<DecayMode>,
    max_penalty: i64,
}

impl Default for PenaltyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PenaltyStore {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            decay: None,
            max_penalty: i64::MAX / 4,
        }
    }
    pub fn with_decay(mut self, d: DecayMode) -> Self {
        self.decay = Some(d);
        self
    }
    pub fn with_max_penalty(mut self, cap: i64) -> Self {
        self.max_penalty = cap.max(1);
        self
    }

    #[inline]
    pub fn add_one(&mut self, f: Feature, step: i64) {
        let e = self.map.entry(f).or_insert(0);
        *e = (*e + step).min(self.max_penalty);
    }
    #[inline]
    pub fn get(&self, f: &Feature) -> i64 {
        *self.map.get(f).unwrap_or(&0)
    }
    #[inline]
    pub fn sum<'a>(&self, feats: impl IntoIterator<Item = &'a Feature>) -> i64 {
        let mut s = 0i64;
        for f in feats {
            s = s.saturating_add(self.get(f));
        }
        s
    }

    pub fn decay_once(&mut self) {
        if let Some(d) = self.decay {
            match d {
                DecayMode::Multiplicative { num, den } => {
                    if den == 0 || num >= den {
                        return;
                    }
                    for v in self.map.values_mut() {
                        if *v > 0 {
                            *v = ((*v as i128 * num as i128) / den as i128) as i64;
                        }
                    }
                }
                DecayMode::Subtractive { step } => {
                    for v in self.map.values_mut() {
                        *v = (*v - step).max(0);
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct AugmentedCostEvaluator<B, T, FX>
where
    FX: FeatureExtractor<T> + ?Sized,
{
    base: B,
    penalties: PenaltyStore,
    lambda_cost: Cost,
    feats: Arc<FX>,
    _phantom: std::marker::PhantomData<T>,
}

impl<B, T, FX> AugmentedCostEvaluator<B, T, FX>
where
    FX: FeatureExtractor<T> + ?Sized,
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

impl<Tnum, B, FX> CostEvaluator<Tnum> for AugmentedCostEvaluator<B, Tnum, FX>
where
    Tnum: Copy + Ord,
    B: CostEvaluator<Tnum>,
    FX: FeatureExtractor<Tnum> + ?Sized,
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
        let mut buf: SmallVec<[Feature; 6]> = SmallVec::new();
        self.feats
            .features_for(request, berth_index, start_time, &mut buf);
        let pen_sum = self.penalties.sum(buf.iter()) as Cost;
        Some(base.saturating_add(self.lambda_cost.saturating_mul(pen_sum)))
    }
}

// ===============================
// Pure helpers
// ===============================

#[inline]
fn gls_lex_accept(
    cur_unassigned: usize,
    cur_aug_cost: Cost,
    cand_unassigned: usize,
    cand_aug: Cost,
) -> bool {
    (cand_unassigned < cur_unassigned)
        || (cand_unassigned == cur_unassigned && cand_aug < cur_aug_cost)
}

fn recompute_true_delta<T>(
    model: &SolverModel<'_, T>,
    before: &SolverState<'_, T>,
    patches: &[(crate::model::index::RequestIndex, DecisionVar<T>)],
) -> Cost
where
    T: SolveNumeric,
{
    let mut base_delta: Cost = 0.into();
    let mut last: HashMap<usize, DecisionVar<T>> = HashMap::with_capacity(patches.len());
    for (ri, dv) in patches {
        last.insert(ri.get(), *dv);
    }

    for (ri_u, patch) in last {
        let ri = RequestIndex::new(ri_u);
        let old_dv = before.decision_variables()[ri.get()];
        if let DecisionVar::Assigned(old) = old_dv
            && let Some(c) = model.cost_of_assignment(ri, old.berth_index, old.start_time)
        {
            base_delta = base_delta.saturating_sub(c);
        }
        if let DecisionVar::Assigned(new_dec) = patch
            && let Some(c) = model.cost_of_assignment(ri, new_dec.berth_index, new_dec.start_time)
        {
            base_delta = base_delta.saturating_add(c);
        }
    }
    base_delta
}

fn augmented_cost_of_state<T, FX>(
    state: &SolverState<'_, T>,
    feats: &FX,
    store: &PenaltyStore,
    lambda_cost: Cost,
) -> Cost
where
    T: Copy + Ord,
    FX: FeatureExtractor<T> + ?Sized,
{
    let mut buf: SmallVec<[Feature; 6]> = SmallVec::new();
    let mut p_sum: i64 = 0;

    for (i, dv) in state.decision_variables().iter().enumerate() {
        if let DecisionVar::Assigned(Decision {
            berth_index,
            start_time,
        }) = *dv
        {
            buf.clear();
            feats.features_for(RequestIndex::new(i), berth_index, start_time, &mut buf);
            p_sum = p_sum.saturating_add(store.sum(buf.iter()));
        }
    }
    state
        .fitness()
        .cost
        .saturating_add(lambda_cost.saturating_mul(p_sum as Cost))
}

fn penalty_share_of_state<T, FX>(
    state: &SolverState<'_, T>,
    feats: &FX,
    store: &PenaltyStore,
    lambda_cost: i64,
) -> f64
where
    T: Copy + Ord,
    FX: FeatureExtractor<T> + ?Sized,
{
    let mut buf: SmallVec<[Feature; 6]> = SmallVec::new();
    let mut p_sum: i64 = 0;

    for (i, dv) in state.decision_variables().iter().enumerate() {
        if let DecisionVar::Assigned(Decision {
            berth_index,
            start_time,
        }) = *dv
        {
            buf.clear();
            feats.features_for(RequestIndex::new(i), berth_index, start_time, &mut buf);
            p_sum = p_sum.saturating_add(store.sum(buf.iter()));
        }
    }
    let true_cost = state.fitness().cost as f64;
    let pen_cost = (lambda_cost as f64) * (p_sum as f64);
    if true_cost <= 0.0 && pen_cost <= 0.0 {
        0.0
    } else {
        pen_cost / (true_cost + pen_cost)
    }
}

fn pulse_penalties<T, FX>(
    model: &SolverModel<'_, T>,
    current: &SolverState<'_, T>,
    feats: &FX,
    store: &mut PenaltyStore,
    top_k: usize,
    step: i64,
    dv_buf: &mut [DecisionVar<T>],
) where
    T: SolveNumeric,
    FX: FeatureExtractor<T> + ?Sized,
{
    let base_eval = DefaultCostEvaluator;
    let mut pc = PlanningContext::new(model, current, &base_eval, dv_buf);

    let mut util_by_feature: HashMap<Feature, i128> = HashMap::new();

    pc.builder().with_explorer(|ex| {
        let mut feats_buf: SmallVec<[Feature; 6]> = SmallVec::new();
        for (i, dv) in ex.decision_vars().iter().enumerate() {
            if let DecisionVar::Assigned(Decision {
                berth_index,
                start_time,
            }) = *dv
                && let Some(base) = ex.peek_cost(RequestIndex::new(i), start_time, berth_index)
            {
                feats_buf.clear();
                feats.features_for(
                    RequestIndex::new(i),
                    berth_index,
                    start_time,
                    &mut feats_buf,
                );
                for f in feats_buf.iter().cloned() {
                    let p = *store.map.get(&f).unwrap_or(&0) as i128;
                    let u = (base as i128) / (1 + p); // higher base, lower penalty → higher utility
                    *util_by_feature.entry(f).or_insert(0) += u;
                }
            }
        }
    });

    let mut items: Vec<(Feature, i128)> = util_by_feature.into_iter().collect();
    items.sort_by_key(|&(_, u)| -u);
    for (rank, (f, _)) in items.into_iter().enumerate() {
        if rank >= top_k {
            break;
        }
        store.add_one(f, step);
    }
}

// ===============================
// Strategy
// ===============================

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

pub struct GuidedLocalSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + ?Sized,
{
    // Operator pool (augmented objective)
    pool: LocalPool<T, AugmentedCostEvaluator<DefaultCostEvaluator, T, FX>, R>,

    // GLS parameters
    lambda: i64,
    penalty_step: i64,
    stagnation_rounds_before_pulse: usize,
    pulse_top_k: usize,
    max_local_steps: usize,

    // Penalties & extractor
    penalty_store: PenaltyStore,
    feature_extractor: Arc<FX>,

    // Acceptance/publish on TRUE objective
    true_acceptor: LexStrictAcceptor,

    // Refetch & restarts
    refetch_after_stale: usize,
    hard_refetch_every: usize,
    hard_refetch_mode: HardRefetchMode,
    restart_on_publish: bool,
    reset_on_refetch: bool,
    kick_steps_on_reset: usize,

    // -------- Online λ control --------
    adaptive_lambda: bool,
    tgt_pen_share_low: f64,
    tgt_pen_share_high: f64,
    lambda_step_frac: f64,
    lambda_min: i64,
    lambda_max: i64,
}

impl<T, R, FX> GuidedLocalSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + ?Sized,
{
    pub fn new(feature_extractor: Arc<FX>) -> Self {
        Self {
            pool: LocalPool::new().with_selector(
                SoftmaxSelector::default()
                    .with_base_temp(1.0)
                    .with_min_p(1e-6)
                    .with_power(1.0),
            ),
            lambda: 4,
            penalty_step: 1,
            stagnation_rounds_before_pulse: 16,
            pulse_top_k: 8,
            max_local_steps: 512,
            penalty_store: PenaltyStore::new()
                .with_decay(DecayMode::Multiplicative { num: 9, den: 10 }),
            feature_extractor,
            true_acceptor: LexStrictAcceptor,
            refetch_after_stale: 0,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
            restart_on_publish: true,
            reset_on_refetch: true,
            kick_steps_on_reset: 3,
            // online λ defaults
            adaptive_lambda: false,
            tgt_pen_share_low: 0.18,
            tgt_pen_share_high: 0.38,
            lambda_step_frac: 0.08,
            lambda_min: 1,
            lambda_max: i64::MAX / 4,
        }
    }

    // ---------- builders ----------
    pub fn with_local_op(
        mut self,
        op: Box<
            dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator, T, FX>, R>
                + Send
                + Sync,
        >,
    ) -> Self {
        self.pool.push(op);
        self
    }
    pub fn with_lambda(mut self, lambda: i64) -> Self {
        self.lambda = lambda.max(1);
        self
    }
    pub fn with_penalty_step(mut self, step: i64) -> Self {
        self.penalty_step = step.max(1);
        self
    }
    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps.max(1);
        self
    }
    pub fn with_refetch_after_stale(mut self, rounds: usize) -> Self {
        self.refetch_after_stale = rounds;
        self
    }
    pub fn with_hard_refetch_every(mut self, period: usize) -> Self {
        self.hard_refetch_every = period;
        self
    }
    pub fn with_hard_refetch_mode(mut self, mode: HardRefetchMode) -> Self {
        self.hard_refetch_mode = mode;
        self
    }
    pub fn with_restart_on_publish(mut self, yes: bool) -> Self {
        self.restart_on_publish = yes;
        self
    }
    pub fn with_reset_on_refetch(mut self, yes: bool) -> Self {
        self.reset_on_refetch = yes;
        self
    }
    pub fn with_kick_steps_on_reset(mut self, k: usize) -> Self {
        self.kick_steps_on_reset = k;
        self
    }
    pub fn with_decay(mut self, d: DecayMode) -> Self {
        self.penalty_store = self.penalty_store.clone().with_decay(d);
        self
    }
    pub fn with_max_penalty(mut self, cap: i64) -> Self {
        self.penalty_store = self.penalty_store.clone().with_max_penalty(cap);
        self
    }
    pub fn with_pulse_params(mut self, stagnation_rounds: usize, top_k: usize) -> Self {
        self.stagnation_rounds_before_pulse = stagnation_rounds.max(1);
        self.pulse_top_k = top_k.max(1);
        self
    }
    // online λ
    pub fn with_adaptive_lambda(mut self, yes: bool) -> Self {
        self.adaptive_lambda = yes;
        self
    }
    pub fn with_target_penalty_share(mut self, low: f64, high: f64) -> Self {
        self.tgt_pen_share_low = low.clamp(0.0, 0.95);
        self.tgt_pen_share_high = high.max(self.tgt_pen_share_low + 0.02).min(0.98);
        self
    }
    pub fn with_lambda_step_frac(mut self, step: f64) -> Self {
        self.lambda_step_frac = step.clamp(0.01, 0.25);
        self
    }
    pub fn with_lambda_bounds(mut self, lo: i64, hi: i64) -> Self {
        self.lambda_min = lo.max(1);
        self.lambda_max = hi.max(self.lambda_min + 1);
        self
    }

    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    #[inline]
    fn periodic_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        best_true: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        outer_rounds: usize,
        eps: i64,
    ) -> bool {
        if !self.should_hard_refetch(outer_rounds) {
            return false;
        }
        let inc = context.shared_incumbent().peek();
        let do_fetch = match self.hard_refetch_mode {
            HardRefetchMode::Always => true,
            HardRefetchMode::IfBetter => materially_better(current.fitness(), &inc, eps),
        };
        if do_fetch {
            tracing::debug!(
                "GLS: periodic refetch at round {} (curr {}, inc {}, eps={})",
                outer_rounds,
                current.fitness(),
                inc,
                eps
            );
            let snap = context.shared_incumbent().snapshot();
            *current = snap.clone();
            if self
                .true_acceptor
                .accept(best_true.fitness(), snap.fitness())
            {
                *best_true = snap;
            }
            return true;
        }
        false
    }

    #[inline]
    fn stale_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        best_true: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        eps: i64,
    ) -> bool {
        let inc = context.shared_incumbent().peek();
        let allowed = match self.hard_refetch_mode {
            HardRefetchMode::Always => true,
            HardRefetchMode::IfBetter => materially_better(current.fitness(), &inc, eps),
        };
        if allowed {
            tracing::debug!(
                "GLS: stale refetch (curr {}, inc {}, eps={})",
                current.fitness(),
                inc,
                eps
            );
            let snap = context.shared_incumbent().snapshot();
            *current = snap.clone();
            if self
                .true_acceptor
                .accept(best_true.fitness(), snap.fitness())
            {
                *best_true = snap;
            }
            return true;
        }
        false
    }

    fn reset_state<'p>(
        &mut self,
        model: &SolverModel<'p, T>,
        rng: &mut R,
        dv_buf: &mut [DecisionVar<T>],
        current: &mut SolverState<'p, T>,
        label: &str,
    ) {
        if self.kick_steps_on_reset == 0 || self.pool.is_empty() {
            tracing::debug!("GLS: reset ({}) — no kick", label);
            return;
        }

        let aug_eval = AugmentedCostEvaluator::new(
            DefaultCostEvaluator,
            self.penalty_store.clone(),
            self.lambda as Cost,
            self.feature_extractor.clone(),
        );

        for _ in 0..self.kick_steps_on_reset {
            let mut pc = PlanningContext::new(model, current, &aug_eval, dv_buf);
            let mut prop = self.pool.apply(&mut pc, rng, None);

            if let Some(mut plan) = prop.take_plan() {
                let patches: Vec<(RequestIndex, DecisionVar<T>)> = plan
                    .decision_var_patches
                    .iter()
                    .map(|p| (p.index, p.patch))
                    .collect();

                let delta = recompute_true_delta(model, current, &patches);
                plan.delta_cost = delta;

                current.apply_plan(plan);
                prop.accept(0);
            } else {
                prop.reject();
            }
        }

        tracing::debug!(
            "GLS: reset ({}) — kick_steps={}",
            label,
            self.kick_steps_on_reset
        );
    }
}

impl<T, R, FX> SearchStrategy<T, R> for GuidedLocalSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng + Send + Sync,
    FX: FeatureExtractor<T> + Send + Sync + 'static + ?Sized,
{
    fn name(&self) -> &str {
        "Guided Local Search"
    }

    #[tracing::instrument(level = "debug", name = "GLS Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();
        if self.pool.is_empty() {
            tracing::warn!("GLS: no local operators configured");
            return;
        }

        // Working states
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best_true: SolverState<'p, T> = current.clone();

        // Buffers & trackers
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut eps_src = MedianHistoryEpsilon::new(32, 1);
        let mut stale_best = StaleTracker::new(*current.fitness(), 32);
        let mut pulse_stale_rounds = 0usize;

        let patience_epochs = if self.refetch_after_stale == 0 {
            patience_from_pulse_threshold(self.stagnation_rounds_before_pulse)
        } else {
            self.refetch_after_stale
        };

        let mut outer_rounds = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            outer_rounds = outer_rounds.saturating_add(1);

            self.penalty_store.decay_once();
            let eps = eps_src.epsilon();

            if self.periodic_refetch(&mut current, &mut best_true, context, outer_rounds, eps)
                && self.reset_on_refetch
            {
                self.reset_state(
                    model,
                    context.rng(),
                    dv_buf.as_mut_slice(),
                    &mut current,
                    "periodic-refetch",
                );
            }

            let mut accepted_any = false;

            let aug_eval = AugmentedCostEvaluator::new(
                DefaultCostEvaluator,
                self.penalty_store.clone(),
                self.lambda as Cost,
                self.feature_extractor.clone(),
            );

            // Local augmented improvement
            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut pc =
                    PlanningContext::new(model, &current, &aug_eval, dv_buf.as_mut_slice());
                let mut prop = self.pool.apply(&mut pc, context.rng(), None);

                let Some(mut plan) = prop.take_plan() else {
                    prop.reject();
                    break;
                };

                let patches: Vec<(RequestIndex, DecisionVar<T>)> = plan
                    .decision_var_patches
                    .iter()
                    .map(|p| (p.index, p.patch))
                    .collect();
                let base_delta = recompute_true_delta(model, &current, &patches);
                plan.delta_cost = base_delta;

                let mut cand = current.clone();
                cand.apply_plan(plan);

                let cur_aug = augmented_cost_of_state(
                    &current,
                    self.feature_extractor.as_ref(),
                    &self.penalty_store,
                    self.lambda as Cost,
                );
                let cand_aug = augmented_cost_of_state(
                    &cand,
                    self.feature_extractor.as_ref(),
                    &self.penalty_store,
                    self.lambda as Cost,
                );

                let cur_unassigned = current.fitness().unassigned_requests;
                let cand_unassigned = cand.fitness().unassigned_requests;

                let true_improvement = current
                    .fitness()
                    .cost
                    .saturating_sub(cand.fitness().cost)
                    .max(0);

                if gls_lex_accept(cur_unassigned, cur_aug, cand_unassigned, cand_aug) {
                    current = cand;
                    prop.accept(true_improvement);

                    if self
                        .true_acceptor
                        .accept(best_true.fitness(), current.fitness())
                    {
                        let drop = best_true
                            .fitness()
                            .cost
                            .saturating_sub(current.fitness().cost)
                            .max(0);
                        eps_src.record(drop);
                        best_true = current.clone();
                        let _ = context.shared_incumbent().try_update(&best_true, model);

                        if self.restart_on_publish {
                            current = best_true.clone();
                            self.reset_state(
                                model,
                                context.rng(),
                                dv_buf.as_mut_slice(),
                                &mut current,
                                "publish",
                            );
                        }
                    }

                    accepted_any = true;
                } else {
                    prop.reject();
                    break;
                }
            }

            // Pulse/Refetch
            if !accepted_any {
                pulse_stale_rounds = pulse_stale_rounds.saturating_add(1);
            } else {
                pulse_stale_rounds = 0;
            }

            if let Some(delta) = stale_best.on_round_end(*best_true.fitness()) {
                eps_src.record(delta);
            }

            if pulse_stale_rounds >= self.stagnation_rounds_before_pulse {
                pulse_penalties(
                    model,
                    &current,
                    self.feature_extractor.as_ref(),
                    &mut self.penalty_store,
                    self.pulse_top_k,
                    self.penalty_step,
                    dv_buf.as_mut_slice(),
                );
                pulse_stale_rounds = 0;
                tracing::trace!(
                    "GLS: penalty pulse (top_k={}, step={})",
                    self.pulse_top_k,
                    self.penalty_step
                );
            }

            if stale_best.is_stale(patience_epochs) {
                let eps_now = eps_src.epsilon();
                if self.stale_refetch(&mut current, &mut best_true, context, eps_now)
                    && self.reset_on_refetch
                {
                    self.reset_state(
                        model,
                        context.rng(),
                        dv_buf.as_mut_slice(),
                        &mut current,
                        "stale-refetch",
                    );
                    stale_best.arm_cooldown_until_next_improvement();
                }
            }

            // -------- Online λ control --------
            if self.adaptive_lambda {
                let share = penalty_share_of_state(
                    &current,
                    self.feature_extractor.as_ref(),
                    &self.penalty_store,
                    self.lambda,
                );
                let step = self.lambda_step_frac;
                if share < self.tgt_pen_share_low {
                    let nl = ((self.lambda as f64) * (1.0 + step)).round() as i64;
                    self.lambda = nl.clamp(self.lambda_min, self.lambda_max);
                    tracing::trace!("GLS: λ↑ → {} (penalty share {:.3})", self.lambda, share);
                } else if share > self.tgt_pen_share_high {
                    let nl = ((self.lambda as f64) * (1.0 - step)).round().max(1.0) as i64;
                    self.lambda = nl.clamp(self.lambda_min, self.lambda_max);
                    tracing::trace!("GLS: λ↓ → {} (penalty share {:.3})", self.lambda, share);
                }
            }
        }

        let _ = context.shared_incumbent().try_update(&best_true, model);
    }
}

// -------------------------------
// Factory
// -------------------------------

#[allow(clippy::type_complexity)]
pub fn gls_strategy<T, R>(
    model: &crate::model::solver_model::SolverModel<T>,
) -> GuidedLocalSearchStrategy<T, R, DefaultFeatureExtractor<T, fn(TimePoint<T>) -> i64>>
where
    T: SolveNumeric + ToPrimitive + Copy + From<i32>,
    R: rand::Rng + Send + Sync,
{
    let bucketizer: fn(TimePoint<T>) -> i64 = |t: TimePoint<T>| -> i64 {
        let v_i64 = t
            .value()
            .to_i64()
            .expect("TimePoint<T> must be i64-convertible");
        v_i64 / 75
    };

    let feats = DefaultFeatureExtractor::new(bucketizer)
        .set_include_req_berth(true)
        .set_include_time(true)
        .set_include_berth_time(true)
        .set_include_req_time(true)
        .set_include_berth(true)
        .set_include_request(true);

    let feats_arc = Arc::new(feats);

    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    GuidedLocalSearchStrategy::new(feats_arc)
        .with_lambda(9)
        .with_penalty_step(2)
        .with_decay(DecayMode::Multiplicative { num: 95, den: 100 })
        .with_max_penalty(1_000_000_000)
        .with_pulse_params(8, 20)
        .with_max_local_steps(2100)
        .with_refetch_after_stale(0)
        .with_hard_refetch_every(24)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_restart_on_publish(true)
        .with_reset_on_refetch(true)
        .with_kick_steps_on_reset(6)
        // online λ defaults (engine may override)
        .with_adaptive_lambda(true)
        .with_target_penalty_share(0.18, 0.38)
        .with_lambda_step_frac(0.08)
        .with_lambda_bounds(1, 2_000_000)
        // ------------------------- Local operators -------------------------
        .with_local_op(Box::new(
            ShiftEarlierOnSameBerth::new(18..=52).with_neighbors(neighbors_same_berth.clone()),
        ))
        .with_local_op(Box::new(
            RelocateSingleBest::new(20..=64).with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            SwapPairSameBerth::new(36..=96).with_neighbors(neighbors_same_berth.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeAcrossBerths::new(48..=128)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            OrOptBlockRelocate::new(5..=9, 1.4..=1.9).with_neighbors(neighbors_same_berth.clone()),
        ))
        .with_local_op(Box::new(
            RelocateSingleBestAllowWorsening::new(12..=24)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            RandomRelocateAnywhere::new(12..=24).with_neighbors(neighbors_any.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbRelocateBest::new(24..=72)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbBestSwapSameBerth::new(48..=120).with_neighbors(neighbors_same_berth.clone()),
        ))
        .with_local_op(Box::new(
            RandomizedGreedyRelocateRcl::new(18..=48, 1.5..=2.2)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeBestAcrossBerths::new(32..=96).with_neighbors(neighbors_any.clone()),
        ))
        .with_local_op(Box::new(
            CascadeRelocateK::new(3..=4, 8..=12, 12..=24)
                .with_neighbors(neighbors_direct_competitors)
                .with_insert_policy(CascadeInsertPolicy::BestEarliest),
        ))
}
