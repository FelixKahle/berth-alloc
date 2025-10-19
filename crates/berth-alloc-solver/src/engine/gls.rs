// crates/berth-alloc-solver/src/engine/gls.rs

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
        acceptor::{Acceptor, LexStrictAcceptor}, // true-objective acceptor
        search::{SearchContext, SearchStrategy},
    },
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    search::{
        operator::LocalMoveOperator,
        operator_library::local::{
            CrossExchangeAcrossBerths, OrOptBlockRelocate, RelocateSingleBest,
            ShiftEarlierOnSameBerth, SwapPairSameBerth,
        },
        planner::{CostEvaluator, DefaultCostEvaluator, PlanningContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::sync::atomic::Ordering as AtomicOrdering;

// (request_raw, berth_raw)
type FeatureKey = (usize, usize);

/// Wrap a base evaluator and add λ * penalty(feature) to the cost seen by operators.
#[derive(Clone)]
pub struct AugmentedCostEvaluator<B> {
    base: B,
    penalties: HashMap<FeatureKey, i64>,
    lambda_cost: Cost,
}

impl<B> AugmentedCostEvaluator<B> {
    pub fn new(base: B, penalties: HashMap<FeatureKey, i64>, lambda_cost: Cost) -> Self {
        Self {
            base,
            penalties,
            lambda_cost,
        }
    }

    #[inline]
    fn feature_key(req: RequestIndex, berth: BerthIndex) -> FeatureKey {
        (req.get(), berth.get())
    }
}

impl<T, B> CostEvaluator<T> for AugmentedCostEvaluator<B>
where
    T: Copy + Ord,
    B: CostEvaluator<T>,
{
    fn eval<'m>(
        &self,
        model: &SolverModel<'m, T>,
        request: RequestIndex,
        start_time: TimePoint<T>,
        berth_index: BerthIndex,
    ) -> Option<Cost> {
        let base = self.base.eval(model, request, start_time, berth_index)?;
        let pen = *self
            .penalties
            .get(&Self::feature_key(request, berth_index))
            .unwrap_or(&0) as Cost;
        Some(base.saturating_add(self.lambda_cost.saturating_mul(pen)))
    }
}

/// GLS acceptance on the augmented objective:
/// lexicographic on (unassigned_requests, augmented_cost).
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

/// Sum penalties for all assigned (request, berth) pairs in a state.
#[inline]
fn penalty_sum_for_state<T>(state: &SolverState<'_, T>, penalties: &HashMap<FeatureKey, i64>) -> i64
where
    T: Copy + Ord,
{
    let mut sum: i64 = 0;
    for (i, dv) in state.decision_variables().iter().enumerate() {
        if let DecisionVar::Assigned(Decision { berth_index, .. }) = *dv {
            let key = (i, berth_index.get());
            sum = sum.saturating_add(*penalties.get(&key).unwrap_or(&0));
        }
    }
    sum
}

/// Compute base_cost + λ * penalty_sum(state).
#[inline]
fn augmented_cost_of_state<T>(
    state: &SolverState<'_, T>,
    penalties: &HashMap<FeatureKey, i64>,
    lambda_cost: Cost,
) -> Cost
where
    T: Copy + Ord,
{
    let p = penalty_sum_for_state(state, penalties) as Cost;
    state
        .fitness()
        .cost
        .saturating_add(lambda_cost.saturating_mul(p))
}

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

pub struct GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Operators run against the augmented evaluator
    local_ops: Vec<Box<dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator>, R>>>,

    // GLS parameters
    lambda: i64,
    penalty_step: i64,
    stagnation_rounds_before_pulse: usize,
    pulse_top_k: usize,
    max_local_steps: usize,

    // Penalty store
    penalties: HashMap<FeatureKey, i64>,

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

impl<T, R> Default for GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    pub fn new() -> Self {
        Self {
            local_ops: Vec::new(),
            lambda: 4,
            penalty_step: 1,
            stagnation_rounds_before_pulse: 16,
            pulse_top_k: 8,
            max_local_steps: 512,
            penalties: HashMap::new(),
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
        op: Box<dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator>, R>>,
    ) -> Self {
        self.local_ops.push(op);
        self
    }
    pub fn with_lambda(mut self, lambda: i64) -> Self {
        self.lambda = lambda;
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
    fn reset_state<'m, 'p>(
        &self,
        model: &'m SolverModel<'p, T>,
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
                self.penalties.clone(),
                self.lambda as Cost,
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

impl<T, R> SearchStrategy<T, R> for GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
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

        // Two states:
        // - `current`: hill-climbs on augmented objective (can worsen true objective)
        // - `best_true`: best by true objective; only this is published.
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best_true: SolverState<'p, T> = current.clone();

        // Scratch DV buffer for PlanningContext.
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut stale = 0usize;
        let mut outer_rounds = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            outer_rounds = outer_rounds.saturating_add(1);

            // Periodic refetch (ILS-style). If we do refetch, optionally reset local climb.
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

            // Fresh augmented evaluator view for this round.
            let aug_eval = AugmentedCostEvaluator::new(
                DefaultCostEvaluator,
                self.penalties.clone(),
                self.lambda as Cost,
            );

            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // Shuffle local operator order each step.
                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(context.rng());

                let mut step_taken = false;

                for &i in &order {
                    let op = &self.local_ops[i];

                    // Operators "see" augmented costs via aug_eval.
                    let mut pc =
                        PlanningContext::new(model, &current, &aug_eval, dv_buf.as_mut_slice());

                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        let mut cand = current.clone();
                        cand.apply_plan(plan);

                        // Decide with GLS acceptor on augmented objective.
                        let cur_aug =
                            augmented_cost_of_state(&current, &self.penalties, self.lambda as Cost);
                        let cand_aug =
                            augmented_cost_of_state(&cand, &self.penalties, self.lambda as Cost);

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

                                // Restart around the just-published best, but KEEP penalties.
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
                            break; // restart climb from new state
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
                    // Pulse penalties on highest-utility features (computed on base cost).
                    let base_eval = DefaultCostEvaluator;
                    let mut pc =
                        PlanningContext::new(model, &current, &base_eval, dv_buf.as_mut_slice());

                    // Collect utilities and bump top-k penalties.
                    let utils = pc.builder().with_explorer(|ex| {
                        let mut tmp: Vec<(FeatureKey, i64)> = Vec::new();
                        for (i, dv) in ex.decision_vars().iter().enumerate() {
                            if let DecisionVar::Assigned(Decision {
                                berth_index,
                                start_time,
                            }) = *dv
                                && let Some(base) =
                                    ex.peek_cost(RequestIndex::new(i), start_time, berth_index)
                            {
                                let key = (i, berth_index.get());
                                let p = *self.penalties.get(&key).unwrap_or(&0);
                                // GLS proxy: utility = base_cost / (1 + penalty)
                                let util = base / (1 + p) as Cost;
                                tmp.push((key, util));
                            }
                        }
                        tmp.sort_by_key(|&(_, util)| -util); // descending
                        tmp
                    });

                    for (idx, (key, _)) in utils.into_iter().enumerate() {
                        if idx >= self.pulse_top_k {
                            break;
                        }
                        *self.penalties.entry(key).or_insert(0) += self.penalty_step;
                    }

                    stale = 0;
                    tracing::trace!("GLS: penalty pulse (top_k={})", self.pulse_top_k);
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

pub fn gls_strategy<T, R>(
    _: &crate::model::solver_model::SolverModel<T>,
) -> GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    GuidedLocalSearchStrategy::new()
        .with_lambda(4)
        .with_max_local_steps(1024)
        .with_refetch_after_stale(128)
        .with_hard_refetch_every(0)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_restart_on_publish(true) // restart around new best (penalties kept)
        .with_reset_on_refetch(true) // reset after refetch (penalties kept)
        .with_kick_steps_on_reset(3) // small diversification after reset
        // Local improvement operators (evaluated on augmented costs)
        .with_local_op(Box::new(ShiftEarlierOnSameBerth {
            number_of_candidates_to_try_range: 8..=24,
        }))
        .with_local_op(Box::new(RelocateSingleBest {
            number_of_candidates_to_try_range: 8..=24,
        }))
        .with_local_op(Box::new(SwapPairSameBerth {
            number_of_pair_attempts_to_try_range: 10..=40,
        }))
        .with_local_op(Box::new(CrossExchangeAcrossBerths {
            number_of_pair_attempts_to_try_range: 12..=48,
        }))
        .with_local_op(Box::new(OrOptBlockRelocate::new(2..=4, 1.4..=2.0)))
}
