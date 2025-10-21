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
        search::{SearchContext, SearchStrategy},
    },
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    search::{
        operator::LocalMoveOperator,
        operator_library::local::{
            CrossExchangeAcrossBerths, HillClimbBestSwapSameBerth, HillClimbRelocateBest,
            OrOptBlockRelocate, RandomRelocateAnywhere, RelocateSingleBest,
            RelocateSingleBestAllowWorsening, ShiftEarlierOnSameBerth, SwapPairSameBerth,
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
use rand::seq::SliceRandom;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::{Arc, atomic::Ordering as AtomicOrdering};

/// Generalized penalizable feature.
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

/// Extracts all features “touched” by placing `request` at `(berth, start_time)`.
pub trait FeatureExtractor<T> {
    fn features_for(
        &self,
        request: RequestIndex,
        berth: BerthIndex,
        start_time: TimePoint<T>,
        out: &mut SmallVec<[Feature; 6]>,
    );
}

/// Configurable extractor using a caller-provided time bucketizer.
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
    /// Multiply by num/den each outer round (integer arithmetic).
    Multiplicative { num: u32, den: u32 },
    /// Subtract constant (floored at 0).
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

    /// Apply decay once per outer round.
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
    fn eval<'m>(
        &self,
        model: &SolverModel<'m, Tnum>,
        request: RequestIndex,
        start_time: TimePoint<Tnum>,
        berth_index: BerthIndex,
    ) -> Option<Cost> {
        let base = self.base.eval(model, request, start_time, berth_index)?;
        let mut buf: SmallVec<[Feature; 6]> = SmallVec::new();
        self.feats
            .features_for(request, berth_index, start_time, &mut buf);
        let pen_sum = self.penalties.sum(buf.iter()) as Cost;
        Some(base.saturating_add(self.lambda_cost.saturating_mul(pen_sum)))
    }
}

trait AugmentedAcceptor {
    #[allow(dead_code)]
    fn name(&self) -> &str;
    fn accept_aug(
        &self,
        cur_unassigned: usize,
        cur_aug_cost: Cost,
        cand_unassigned: usize,
        cand_aug_cost: Cost,
    ) -> bool;
}

#[derive(Debug, Default, Clone)]
struct GlsLexStrictAcceptor;
impl AugmentedAcceptor for GlsLexStrictAcceptor {
    fn name(&self) -> &str {
        "GlsLexStrictAcceptor"
    }
    #[inline]
    fn accept_aug(
        &self,
        cur_unassigned: usize,
        cur_aug_cost: Cost,
        cand_unassigned: usize,
        cand_aug_cost: Cost,
    ) -> bool {
        (cand_unassigned < cur_unassigned)
            || (cand_unassigned == cur_unassigned && cand_aug_cost < cur_aug_cost)
    }
}

#[inline]
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

// =============== Strategy ===============

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
    // Operators (evaluated on augmented costs)
    #[allow(clippy::type_complexity)]
    local_ops: Vec<
        Box<
            dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator, T, FX>, R>
                + Send
                + Sync,
        >,
    >,

    // GLS parameters
    lambda: i64,
    penalty_step: i64,
    stagnation_rounds_before_pulse: usize,
    pulse_top_k: usize,
    max_local_steps: usize,

    // Penalty store & extractor
    penalty_store: PenaltyStore,
    feature_extractor: Arc<FX>,

    // Acceptance
    gls_acceptor: GlsLexStrictAcceptor, // augmented objective
    true_acceptor: LexStrictAcceptor,   // true objective

    // ILS-like refetch knobs
    refetch_after_stale: usize, // 0 => disabled
    hard_refetch_every: usize,  // 0 => disabled
    hard_refetch_mode: HardRefetchMode,

    // Reset / restart behavior (penalties are preserved across resets)
    restart_on_publish: bool,
    reset_on_refetch: bool,
    kick_steps_on_reset: usize,
}

impl<T, R, FX> GuidedLocalSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + ?Sized,
{
    pub fn new(feature_extractor: Arc<FX>) -> Self {
        Self {
            local_ops: Vec::new(),
            lambda: 4,
            penalty_step: 1,
            stagnation_rounds_before_pulse: 16,
            pulse_top_k: 8,
            max_local_steps: 512,
            penalty_store: PenaltyStore::new()
                .with_decay(DecayMode::Multiplicative { num: 9, den: 10 }),
            feature_extractor,
            gls_acceptor: GlsLexStrictAcceptor,
            true_acceptor: LexStrictAcceptor,
            refetch_after_stale: 128,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
            restart_on_publish: true,
            reset_on_refetch: true,
            kick_steps_on_reset: 3,
        }
    }

    pub fn with_local_op(
        mut self,
        op: Box<
            dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator, T, FX>, R>
                + Send
                + Sync,
        >,
    ) -> Self {
        self.local_ops.push(op);
        self
    }
    pub fn with_lambda(mut self, lambda: i64) -> Self {
        self.lambda = lambda;
        self
    }
    pub fn with_penalty_step(mut self, step: i64) -> Self {
        self.penalty_step = step;
        self
    }
    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps;
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

    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    /// Periodic refetch; returns true if a refetch happened.
    #[inline]
    fn periodic_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        best_true: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        outer_rounds: usize,
    ) -> bool {
        if !self.should_hard_refetch(outer_rounds) {
            return false;
        }
        let inc = context.shared_incumbent().peek();
        let do_fetch = match self.hard_refetch_mode {
            HardRefetchMode::IfBetter => self.true_acceptor.accept(current.fitness(), &inc),
            HardRefetchMode::Always => true,
        };
        if do_fetch {
            tracing::debug!(
                "GLS: periodic refetch at round {} (curr {}, inc {})",
                outer_rounds,
                current.fitness(),
                inc
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

    /// Staleness-triggered refetch; returns true if a refetch happened.
    #[inline]
    fn stale_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        best_true: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        stale_rounds: usize,
    ) -> bool {
        if self.refetch_after_stale == 0 || stale_rounds < self.refetch_after_stale {
            return false;
        }
        let inc = context.shared_incumbent().peek();
        if self.true_acceptor.accept(current.fitness(), &inc) {
            tracing::debug!(
                "GLS: staleness refetch after {} rounds ({} -> {})",
                stale_rounds,
                current.fitness(),
                inc
            );
            let snap = context.shared_incumbent().snapshot();
            *current = snap.clone();
            if self
                .true_acceptor
                .accept(best_true.fitness(), snap.fitness())
            {
                *best_true = snap;
            }
            true
        } else {
            false
        }
    }

    /// Reset local climb state around `current` and optionally apply a few random kick moves
    /// using the *augmented* evaluator (penalties preserved).
    fn reset_state<'p>(
        &self,
        model: &SolverModel<'p, T>,
        rng: &mut R,
        dv_buf: &mut [DecisionVar<T>],
        current: &mut SolverState<'p, T>,
        stale_rounds: &mut usize,
        label: &str,
    ) {
        *stale_rounds = 0;

        if self.kick_steps_on_reset > 0 {
            let aug_eval = AugmentedCostEvaluator::new(
                DefaultCostEvaluator,
                self.penalty_store.clone(),
                self.lambda as Cost,
                self.feature_extractor.clone(), // cheap Arc clone
            );
            for _ in 0..self.kick_steps_on_reset {
                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(rng);
                let mut kicked = false;
                for &oi in &order {
                    let op = &self.local_ops[oi];
                    let mut pc = PlanningContext::new(model, current, &aug_eval, dv_buf);
                    if let Some(plan) = op.propose(&mut pc, rng) {
                        current.apply_plan(plan);
                        kicked = true;
                        break;
                    }
                }
                if !kicked {
                    break;
                }
            }
        }

        tracing::debug!(
            "GLS: reset ({}) — staleness=0, kick_steps={}",
            label,
            self.kick_steps_on_reset
        );
    }
}

impl<T, R, FX> SearchStrategy<T, R> for GuidedLocalSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + Send + Sync + 'static + ?Sized,
{
    fn name(&self) -> &str {
        "Guided Local Search"
    }

    #[tracing::instrument(level = "debug", name = "GLS Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();
        if self.local_ops.is_empty() {
            tracing::warn!("GLS: no local operators configured");
            return;
        }

        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best_true: SolverState<'p, T> = current.clone();

        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut stale = 0usize;
        let mut outer_rounds = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            outer_rounds = outer_rounds.saturating_add(1);

            // time decay
            self.penalty_store.decay_once();

            // periodic refetch
            if self.periodic_refetch(&mut current, &mut best_true, context, outer_rounds)
                && self.reset_on_refetch
            {
                self.reset_state(
                    model,
                    context.rng(),
                    dv_buf.as_mut_slice(),
                    &mut current,
                    &mut stale,
                    "periodic-refetch",
                );
            }

            let mut accepted_any = false;

            // snapshot augmented evaluator for this round
            let aug_eval = AugmentedCostEvaluator::new(
                DefaultCostEvaluator,
                self.penalty_store.clone(),
                self.lambda as Cost,
                self.feature_extractor.clone(), // cheap Arc clone
            );

            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(context.rng());

                let mut step_taken = false;

                for &i in &order {
                    let op = &self.local_ops[i];
                    let mut pc =
                        PlanningContext::new(model, &current, &aug_eval, dv_buf.as_mut_slice());

                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        let mut cand = current.clone();
                        cand.apply_plan(plan);

                        // GLS acceptor on augmented objective
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

                        if self.gls_acceptor.accept_aug(
                            cur_unassigned,
                            cur_aug,
                            cand_unassigned,
                            cand_aug,
                        ) {
                            current = cand;

                            // If we also improved the TRUE objective, capture & publish.
                            if self
                                .true_acceptor
                                .accept(best_true.fitness(), current.fitness())
                            {
                                best_true = current.clone();
                                let _ = context.shared_incumbent().try_update(&best_true);

                                if self.restart_on_publish {
                                    current = best_true.clone();
                                    self.reset_state(
                                        model,
                                        context.rng(),
                                        dv_buf.as_mut_slice(),
                                        &mut current,
                                        &mut stale,
                                        "publish",
                                    );
                                }
                            }

                            step_taken = true;
                            accepted_any = true;
                            break; // restart local climb from new state
                        }
                    }
                }

                if !step_taken {
                    break;
                }
            }

            if !accepted_any {
                stale = stale.saturating_add(1);

                if stale >= self.stagnation_rounds_before_pulse {
                    // Build utilities per FEATURE on base evaluator.
                    let base_eval = DefaultCostEvaluator;
                    let mut pc =
                        PlanningContext::new(model, &current, &base_eval, dv_buf.as_mut_slice());

                    let mut util_by_feature: HashMap<Feature, i128> = HashMap::new();

                    pc.builder().with_explorer(|ex| {
                        let mut feats_buf: SmallVec<[Feature; 6]> = SmallVec::new();
                        for (i, dv) in ex.decision_vars().iter().enumerate() {
                            if let DecisionVar::Assigned(Decision {
                                berth_index,
                                start_time,
                            }) = *dv
                                && let Some(base) =
                                    ex.peek_cost(RequestIndex::new(i), start_time, berth_index)
                            {
                                feats_buf.clear();
                                self.feature_extractor.features_for(
                                    RequestIndex::new(i),
                                    berth_index,
                                    start_time,
                                    &mut feats_buf,
                                );
                                for f in feats_buf.iter().cloned() {
                                    let p = *self.penalty_store.map.get(&f).unwrap_or(&0) as i128;
                                    // GLS proxy utility: higher base & lower current penalty -> higher utility
                                    let u = (base as i128) / (1 + p);
                                    *util_by_feature.entry(f).or_insert(0) += u;
                                }
                            }
                        }
                    });

                    let mut items: Vec<(Feature, i128)> = util_by_feature.into_iter().collect();
                    items.sort_by_key(|&(_, u)| -u);
                    for (rank, (f, _)) in items.into_iter().enumerate() {
                        if rank >= self.pulse_top_k {
                            break;
                        }
                        self.penalty_store.add_one(f, self.penalty_step);
                    }

                    stale = 0;
                    tracing::trace!(
                        "GLS: penalty pulse (top_k={}, step={})",
                        self.pulse_top_k,
                        self.penalty_step
                    );
                } else if self.stale_refetch(&mut current, &mut best_true, context, stale)
                    && self.reset_on_refetch
                {
                    self.reset_state(
                        model,
                        context.rng(),
                        dv_buf.as_mut_slice(),
                        &mut current,
                        &mut stale,
                        "stale-refetch",
                    );
                }
            } else {
                stale = 0;
            }
        }

        // Final publish (no-op if not better).
        let _ = context.shared_incumbent().try_update(&best_true);
    }
}

#[allow(clippy::type_complexity)]
pub fn gls_strategy<T, R>(
    model: &crate::model::solver_model::SolverModel<T>,
) -> GuidedLocalSearchStrategy<T, R, DefaultFeatureExtractor<T, fn(TimePoint<T>) -> i64>>
where
    T: SolveNumeric + ToPrimitive + Copy + From<i32>,
    R: rand::Rng,
{
    let bucketizer: fn(TimePoint<T>) -> i64 = |t: TimePoint<T>| -> i64 {
        let v_i64 = t
            .value()
            .to_i64()
            .expect("TimePoint<T>::value() must be convertible to i64 for bucketing");
        v_i64 / 90
    };

    let feats = DefaultFeatureExtractor::new(bucketizer)
        .set_include_req_berth(true)
        .set_include_time(true)
        .set_include_berth_time(true)
        .set_include_req_time(true)
        .set_include_berth(true)
        .set_include_request(true);

    let feats_arc = std::sync::Arc::new(feats);

    // --- neighbor scopes (preconfigured) ---
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    // Aggressive GLS for ~20 berths / ~250 ships / PT 20–120
    GuidedLocalSearchStrategy::new(feats_arc)
        .with_lambda(7)
        .with_penalty_step(2)
        .with_decay(DecayMode::Multiplicative { num: 95, den: 100 }) // 95%
        .with_max_penalty(1_000_000_000)
        .with_pulse_params(10, 18) // stagnation, top-k
        .with_max_local_steps(2000)
        .with_refetch_after_stale(120)
        .with_hard_refetch_every(24) // periodic gentle shake-up
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_restart_on_publish(true)
        .with_reset_on_refetch(true)
        .with_kick_steps_on_reset(5)
        // ------------------------- Local operators (aggressive) -------------------------
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
            CrossExchangeAcrossBerths::new(36..=96)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            OrOptBlockRelocate::new(3..=6, 1.5..=2.5).with_neighbors(neighbors_same_berth.clone()),
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
}
