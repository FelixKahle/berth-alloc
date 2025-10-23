// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to do so, subject to the following conditions:
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

#![allow(clippy::needless_return)]

use crate::engine::adaptive::tuning::{LocalCountTargetTuner, OrOptBlockKTuner};
use crate::search::operator_library::local::{
    CrossExchangeBestAcrossBerths, HillClimbBestSwapSameBerth, HillClimbRelocateBest,
    RandomRelocateAnywhere, RandomizedGreedyRelocateRcl, RelocateSingleBestAllowWorsening,
};
use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor},
        adaptive::{
            ops_book::OperatorBook, selection::SoftmaxSelector, tuning::DefaultOperatorTuner,
        },
        neighbors,
        search::{SearchContext, SearchStrategy},
    },
    model::index::RequestIndex,
    search::{
        operator::{LocalMoveOperator, OperatorKind},
        operator_library::local::{
            CrossExchangeAcrossBerths, OrOptBlockRelocate, RelocateSingleBest,
            ShiftEarlierOnSameBerth, SwapPairSameBerth,
        },
        planner::{DefaultCostEvaluator, PlanningContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        fitness::Fitness,
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::ToPrimitive;
use rand::seq::SliceRandom;
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    ops::RangeInclusive,
    sync::{Arc, atomic::Ordering as AtomicOrdering},
};

// ---- Feature-signal toolkit (shared with GLS)
use crate::engine::feature_signal::prelude::{
    AugmentedCostEvaluator, DecayMode, DefaultFeatureExtractor, Feature, FeatureExtractor,
    PenaltyStore, augmented_cost_of_state,
};

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

/// Lexicographic acceptor on (unassigned, augmented_cost).
#[derive(Debug, Default, Clone)]
struct LexAugAcceptor;
impl LexAugAcceptor {
    #[inline]
    fn accept(
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

/// Tabu Search parameterized by a feature extractor `FX`.
/// Operators run against the augmented evaluator; publishing uses the true objective.
pub struct TabuSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + ?Sized,
{
    // Local operators evaluated on the augmented objective.
    #[allow(clippy::type_complexity)]
    local_ops: Vec<
        Box<
            dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator, T, FX>, R>
                + Send
                + Sync,
        >,
    >,

    // Adaptive operator book for local ops (augmented evaluator)
    local_book: OperatorBook<T, R>,

    // Tabu parameters
    tabu_tenure_rounds: RangeInclusive<usize>,
    max_local_steps: usize,
    samples_per_step: usize,

    // GLS-like enhancements
    lambda: i64,
    penalty_step: i64,
    stagnation_rounds_before_pulse: usize,
    pulse_top_k: usize,

    // Penalty memory and feature extractor
    penalty_store: PenaltyStore,
    feature_extractor: Arc<FX>,

    // Acceptors:
    // - true_acceptor: lexicographic on TRUE objective (publish/best_true)
    // - lex_aug: lexicographic on augmented objective (walker ranking)
    true_acceptor: LexStrictAcceptor,
    lex_aug: LexAugAcceptor,

    // ILS-like sync/refetch knobs
    refetch_after_stale: usize, // 0 => disabled
    hard_refetch_every: usize,  // 0 => disabled
    hard_refetch_mode: HardRefetchMode,

    // Restart / Reset controls
    restart_on_publish: bool,
    reset_on_refetch: bool,
    kick_steps_on_reset: usize,
}

impl<T, R, FX> TabuSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + ?Sized,
{
    pub fn new(feature_extractor: Arc<FX>) -> Self {
        Self {
            local_ops: Vec::new(),
            local_book: OperatorBook::new(
                OperatorKind::Local,
                Box::new(SoftmaxSelector::default()),
            ),
            tabu_tenure_rounds: 16..=32,
            max_local_steps: 512,
            samples_per_step: 96,
            lambda: 4,
            penalty_step: 1,
            stagnation_rounds_before_pulse: 16,
            pulse_top_k: 8,
            penalty_store: PenaltyStore::new()
                .with_decay(DecayMode::Multiplicative { num: 9, den: 10 }),
            feature_extractor,
            true_acceptor: LexStrictAcceptor,
            lex_aug: LexAugAcceptor,
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
        // Register a default tuner for this operator slot
        let _ = self
            .local_book
            .register_operator(Box::new(DefaultOperatorTuner::default()));
        self
    }

    pub fn with_local_op_tuned(
        mut self,
        op: Box<
            dyn LocalMoveOperator<T, AugmentedCostEvaluator<DefaultCostEvaluator, T, FX>, R>
                + Send
                + Sync,
        >,
        tuner: Box<dyn crate::engine::adaptive::tuning::OperatorTuner<T>>,
    ) -> Self {
        self.local_ops.push(op);
        let _ = self.local_book.register_operator(tuner);
        self
    }

    pub fn with_tabu_tenure(mut self, rounds: RangeInclusive<usize>) -> Self {
        self.tabu_tenure_rounds = rounds;
        self
    }
    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps.max(1);
        self
    }
    pub fn with_samples_per_step(mut self, k: usize) -> Self {
        self.samples_per_step = k.max(8);
        self
    }
    pub fn with_lambda(mut self, lambda: i64) -> Self {
        self.lambda = lambda.max(0);
        self
    }
    pub fn with_penalty_step(mut self, step: i64) -> Self {
        self.penalty_step = step.max(1);
        self
    }
    pub fn with_pulse_params(mut self, stagnation_rounds: usize, top_k: usize) -> Self {
        self.stagnation_rounds_before_pulse = stagnation_rounds.max(1);
        self.pulse_top_k = top_k.max(1);
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
    fn sample_tenure(&self, rng: &mut R) -> usize {
        let lo = *self.tabu_tenure_rounds.start();
        let hi = *self.tabu_tenure_rounds.end();
        if lo == hi {
            lo
        } else {
            rng.random_range(lo..=hi)
        }
    }

    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    /// Reset the walker's memory and optionally perform a few random kicks around `current`.
    #[allow(clippy::too_many_arguments)]
    fn reset_state<'p>(
        &self,
        model: &crate::model::solver_model::SolverModel<'p, T>,
        rng: &mut R,
        dv_buf: &mut [DecisionVar<T>],
        current: &mut SolverState<'p, T>,
        last_best_current: &mut Fitness,
        stale_rounds: &mut usize,
        tabu_until: &mut HashMap<usize, usize>,
        label: &str,
    ) {
        tabu_until.clear();
        *last_best_current = current.fitness().clone();
        *stale_rounds = 0;

        if self.kick_steps_on_reset > 0 {
            let aug_eval = AugmentedCostEvaluator::new(
                DefaultCostEvaluator,
                self.penalty_store.clone(),
                self.lambda as Cost,
                self.feature_extractor.clone(),
            );
            for _ in 0..self.kick_steps_on_reset {
                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(rng);
                let mut kicked = false;
                for &oi in &order {
                    let op = &self.local_ops[oi];
                    let mut pc = PlanningContext::new(model, current, &aug_eval, dv_buf);
                    if let Some(mut plan) = op.propose(&mut pc, rng) {
                        // Recompute base delta for the kick plan
                        use crate::model::index::RequestIndex;
                        use std::collections::HashMap;

                        let mut base_delta: Cost = Cost::from(0);
                        let mut last_patch: HashMap<usize, DecisionVar<T>> = HashMap::new();
                        for p in &plan.decision_var_patches {
                            last_patch.insert(p.index.get(), p.patch);
                        }
                        for (ri_u, patch) in last_patch {
                            let ri = RequestIndex::new(ri_u);
                            let old_dv = current.decision_variables()[ri.get()];
                            if let DecisionVar::Assigned(old) = old_dv
                                && let Some(c) =
                                    model.cost_of_assignment(ri, old.berth_index, old.start_time)
                            {
                                base_delta = base_delta.saturating_sub(c);
                            }
                            if let DecisionVar::Assigned(new_dec) = patch
                                && let Some(c) = model.cost_of_assignment(
                                    ri,
                                    new_dec.berth_index,
                                    new_dec.start_time,
                                )
                            {
                                base_delta = base_delta.saturating_add(c);
                            }
                        }
                        plan.delta_cost = base_delta;

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
            "Tabu: reset ({}) — cleared tabu, staleness reset, kick_steps={}",
            label,
            self.kick_steps_on_reset
        );
    }
}

impl<T, R, FX> SearchStrategy<T, R> for TabuSearchStrategy<T, R, FX>
where
    T: SolveNumeric,
    R: rand::Rng,
    FX: FeatureExtractor<T> + Send + Sync + 'static + ?Sized,
{
    fn name(&self) -> &str {
        "Tabu Search"
    }

    #[tracing::instrument(level = "debug", name = "Tabu Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();

        if self.local_ops.is_empty() {
            tracing::warn!("Tabu: no local operators configured");
            return;
        }

        // States:
        // - current: walker state (can worsen true objective)
        // - best_true: best by TRUE objective; only this is published
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best_true: SolverState<'p, T> = current.clone();

        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Tabu list keyed by request raw id → expire at round
        let mut tabu_until: HashMap<usize, usize> = HashMap::new();

        // Loop counters and staleness
        let mut round: usize = 0;
        let mut stale_rounds: usize = 0;
        let mut last_best_current = current.fitness().clone();

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            round = round.saturating_add(1);

            // Decay the penalty memory each outer round
            self.penalty_store.decay_once();

            // Periodic refetch (ILS-style)
            if self.should_hard_refetch(round) {
                let inc = context.shared_incumbent().peek();
                let do_fetch = match self.hard_refetch_mode {
                    HardRefetchMode::IfBetter => LexStrictAcceptor.accept(current.fitness(), &inc),
                    HardRefetchMode::Always => true,
                };
                if do_fetch {
                    tracing::debug!(
                        "Tabu: periodic refetch at round {} (curr {}, inc {})",
                        round,
                        current.fitness(),
                        inc
                    );
                    let snap = context.shared_incumbent().snapshot();
                    current = snap.clone();
                    if LexStrictAcceptor.accept(best_true.fitness(), snap.fitness()) {
                        best_true = snap;
                    }
                    last_best_current = current.fitness().clone();

                    if self.reset_on_refetch {
                        self.reset_state(
                            model,
                            context.rng(),
                            dv_buf.as_mut_slice(),
                            &mut current,
                            &mut last_best_current,
                            &mut stale_rounds,
                            &mut tabu_until,
                            "refetch",
                        );
                    }
                }
            }

            let mut improved_this_round = false;

            // Snapshot augmented evaluator for this round
            let aug_eval = AugmentedCostEvaluator::new(
                DefaultCostEvaluator,
                self.penalty_store.clone(),
                self.lambda as Cost,
                self.feature_extractor.clone(),
            );

            // Multiple tabu steps per outer round
            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // Retune locals once per step using global stats from current
                let global_stats = current.stats(model);

                let stuck_factor =
                    (stale_rounds as f64 / self.stagnation_rounds_before_pulse as f64).min(1.0);
                let stagnation = crate::engine::adaptive::tuning::Stagnation {
                    stale_rounds,
                    stuck_factor,
                };

                self.local_book.retune_all(&global_stats, &stagnation);

                // Candidate buffers (augmented lex metric + true fitness for aspiration)
                struct Cand<'p, T: SolveNumeric> {
                    state: SolverState<'p, T>, // candidate state after applying plan
                    moved: Vec<usize>,
                    aug_unassigned: usize,
                    aug_cost: Cost,
                    true_fit: Fitness,
                    op_index: usize,
                    base_delta: Cost,
                    seq_id: usize,
                }

                let mut best_admissible: Option<Cand<'p, T>> = None;
                let mut best_overall: Option<Cand<'p, T>> = None;
                let mut proposals: Vec<(
                    usize, /*seq*/
                    usize, /*op*/
                    f64,   /*base_delta*/
                )> = Vec::new();
                let mut seq: usize = 0;

                // Sample a subset of neighborhood
                for _ in 0..self.samples_per_step {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    // Retune locals once per step using global stats from current
                    let global_stats = current.stats(model);

                    let stuck_factor =
                        (stale_rounds as f64 / self.stagnation_rounds_before_pulse as f64).min(1.0);
                    let stagnation = crate::engine::adaptive::tuning::Stagnation {
                        stale_rounds,
                        stuck_factor,
                    };

                    self.local_book.retune_all(&global_stats, &stagnation);

                    // Select operator index via adaptive selector
                    let oi = self
                        .local_book
                        .select(&global_stats, &stagnation, context.rng());

                    let op = &mut self.local_ops[oi];

                    // Push tuning to operator
                    let tuning = *self.local_book.tuning_for(oi);
                    op.tune(&tuning, &global_stats);

                    // Build planner on augmented evaluator
                    let mut pc =
                        PlanningContext::new(model, &current, &aug_eval, dv_buf.as_mut_slice());

                    // Time proposal
                    let t0 = self.local_book.propose_started();
                    if let Some(mut plan) = op.propose(&mut pc, context.rng()) {
                        self.local_book.record_propose(oi, t0, true);

                        // Collect moved request IDs
                        let mut moved: HashSet<usize> = HashSet::new();
                        for p in &plan.decision_var_patches {
                            moved.insert(p.index.get());
                        }
                        if moved.is_empty() {
                            continue;
                        }

                        // Recompute base delta for the plan vs `current` before applying
                        use crate::model::index::RequestIndex;

                        let mut base_delta: Cost = Cost::from(0);
                        // Only the last patch per request matters
                        let mut last_patch: HashMap<usize, DecisionVar<T>> = HashMap::new();
                        for p in &plan.decision_var_patches {
                            last_patch.insert(p.index.get(), p.patch);
                        }
                        for (ri_u, patch) in last_patch.iter() {
                            let ri = RequestIndex::new(*ri_u);
                            let old_dv = current.decision_variables()[ri.get()];
                            if let DecisionVar::Assigned(old) = old_dv
                                && let Some(c) =
                                    model.cost_of_assignment(ri, old.berth_index, old.start_time)
                            {
                                base_delta = base_delta.saturating_sub(c);
                            }
                            if let DecisionVar::Assigned(new_dec) = patch
                                && let Some(c) = model.cost_of_assignment(
                                    ri,
                                    new_dec.berth_index,
                                    new_dec.start_time,
                                )
                            {
                                base_delta = base_delta.saturating_add(c);
                            }
                        }
                        plan.delta_cost = base_delta;

                        // Build candidate by applying the corrected plan
                        let mut cand = current.clone();
                        cand.apply_plan(plan);

                        // Compute augmented objective on the candidate state
                        let aug_cost = augmented_cost_of_state(
                            &cand,
                            self.feature_extractor.as_ref(),
                            &self.penalty_store,
                            self.lambda as Cost,
                        );
                        let aug_unassigned = cand.fitness().unassigned_requests;

                        // True fitness (for aspiration and publishing)
                        let true_fit = cand.fitness().clone();

                        let is_tabu = moved
                            .iter()
                            .any(|rid| tabu_until.get(rid).is_some_and(|&e| e > round));

                        // Aspiration: beats local/global best on TRUE objective
                        let beats_local_best =
                            self.true_acceptor.accept(best_true.fitness(), &true_fit);
                        let beats_shared = self
                            .true_acceptor
                            .accept(&context.shared_incumbent().peek(), &true_fit);

                        let seq_id = seq;
                        seq = seq.saturating_add(1);
                        proposals.push((seq_id, oi, base_delta as f64));

                        let cand_rec = Cand {
                            state: cand,
                            moved: moved.iter().copied().collect(),
                            aug_unassigned,
                            aug_cost,
                            true_fit,
                            op_index: oi,
                            base_delta,
                            seq_id,
                        };

                        // Update best_overall (ignoring tabu) on augmented lex metric
                        let better_overall = match &best_overall {
                            None => true,
                            Some(b) => self.lex_aug.accept(
                                b.aug_unassigned,
                                b.aug_cost,
                                cand_rec.aug_unassigned,
                                cand_rec.aug_cost,
                            ),
                        };
                        if better_overall {
                            best_overall = Some(Cand {
                                state: cand_rec.state.clone(),
                                moved: cand_rec.moved.clone(),
                                aug_unassigned: cand_rec.aug_unassigned,
                                aug_cost: cand_rec.aug_cost,
                                true_fit: cand_rec.true_fit.clone(),
                                op_index: cand_rec.op_index,
                                base_delta: cand_rec.base_delta,
                                seq_id: cand_rec.seq_id,
                            });
                        }

                        // Update best_admissible (non-tabu or aspiration)
                        if !is_tabu || beats_local_best || beats_shared {
                            let better_adm = match &best_admissible {
                                None => true,
                                Some(b) => self.lex_aug.accept(
                                    b.aug_unassigned,
                                    b.aug_cost,
                                    cand_rec.aug_unassigned,
                                    cand_rec.aug_cost,
                                ),
                            };
                            if better_adm {
                                best_admissible = Some(cand_rec);
                            }
                        }
                    } else {
                        self.local_book.record_propose(oi, t0, false);
                    }
                }

                // Choose: admissible first, else overall (classic tabu fallback)
                let chosen = match (best_admissible, best_overall) {
                    (Some(a), _) => Some(a),
                    (None, Some(o)) => Some(o),
                    (None, None) => None,
                };

                let Some(ch) = chosen else {
                    // Record outcomes for proposals (all rejected)
                    for (seq_id, oi, base_delta) in proposals.drain(..) {
                        let _ = seq_id; // unused here
                        self.local_book.record_outcome(oi, false, base_delta);
                    }
                    break; // no candidates this step
                };

                // Tenure lock for moved requests
                let tenure = self.sample_tenure(context.rng());
                for rid in &ch.moved {
                    tabu_until.insert(*rid, round.saturating_add(tenure));
                }

                // Commit the chosen candidate as current
                current = ch.state;

                // Record outcomes for all proposals: chosen = accepted; others rejected
                let mut accepted_recorded = false;
                for (seq_id, oi, base_delta) in proposals.drain(..) {
                    if !accepted_recorded && seq_id == ch.seq_id {
                        self.local_book.record_outcome(oi, true, base_delta);
                        accepted_recorded = true;
                    } else {
                        self.local_book.record_outcome(oi, false, base_delta);
                    }
                }
                if !accepted_recorded {
                    // Fallback in unlikely case chosen wasn't in proposals buffer
                    self.local_book
                        .record_outcome(ch.op_index, true, ch.base_delta as f64);
                }

                // If TRUE objective improved, publish and (optionally) restart around best basin
                if self
                    .true_acceptor
                    .accept(best_true.fitness(), current.fitness())
                {
                    best_true = current.clone();
                    let _ = context.shared_incumbent().try_update(&best_true, model);

                    if self.restart_on_publish {
                        self.reset_state(
                            model,
                            context.rng(),
                            dv_buf.as_mut_slice(),
                            &mut current,
                            &mut last_best_current,
                            &mut stale_rounds,
                            &mut tabu_until,
                            "publish",
                        );
                    }
                }

                // Track local TRUE improvement for staleness
                if self
                    .true_acceptor
                    .accept(&last_best_current, current.fitness())
                {
                    last_best_current = current.fitness().clone();
                    improved_this_round = true;
                    stale_rounds = 0;
                }
            }

            // Stagnation logic: GLS-style penalty pulse or refetch
            if !improved_this_round {
                stale_rounds = stale_rounds.saturating_add(1);

                if stale_rounds >= self.stagnation_rounds_before_pulse {
                    // Compute GLS utilities per *feature* on base evaluator
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

                    stale_rounds = 0;
                    tracing::trace!(
                        "Tabu: penalty pulse (top_k={}, step={})",
                        self.pulse_top_k,
                        self.penalty_step
                    );
                } else if self.refetch_after_stale > 0 && stale_rounds >= self.refetch_after_stale {
                    let inc = context.shared_incumbent().peek();
                    if self.true_acceptor.accept(current.fitness(), &inc) {
                        tracing::debug!(
                            "Tabu: staleness refetch after {} rounds ({} -> {})",
                            stale_rounds,
                            current.fitness(),
                            inc
                        );
                        let snap = context.shared_incumbent().snapshot();
                        current = snap.clone();
                        if self
                            .true_acceptor
                            .accept(best_true.fitness(), snap.fitness())
                        {
                            best_true = snap;
                        }
                        last_best_current = current.fitness().clone();

                        if self.reset_on_refetch {
                            self.reset_state(
                                model,
                                context.rng(),
                                dv_buf.as_mut_slice(),
                                &mut current,
                                &mut last_best_current,
                                &mut stale_rounds,
                                &mut tabu_until,
                                "stale-refetch",
                            );
                        }
                    }
                    stale_rounds = 0;
                }
            }
        }

        // Final publish (no-op if not better).
        let _ = context.shared_incumbent().try_update(&best_true, model);
    }
}

// ================= Tabu (shorter tenure, quicker samples) ===================
pub fn tabu_strategy<T, R>(
    model: &crate::model::solver_model::SolverModel<T>,
) -> TabuSearchStrategy<T, R, DefaultFeatureExtractor<T>>
where
    T: SolveNumeric + num_traits::ToPrimitive + Copy + From<i32>,
    R: rand::Rng,
{
    use crate::engine::adaptive::tuning::{
        LocalCountTargetTuner, OrOptBlockKTuner, WorkBudgetTuner,
    };

    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    let bucketizer: fn(TimePoint<T>) -> i64 = |t| t.value().to_i64().unwrap() / 90;

    let feats = DefaultFeatureExtractor::new(bucketizer)
        .set_include_req_berth(true)
        .set_include_time(true)
        .set_include_berth_time(true)
        .set_include_req_time(true)
        .set_include_berth(true)
        .set_include_request(true);

    let feats_arc = std::sync::Arc::new(feats);

    let ultra = || {
        WorkBudgetTuner::default()
            .with_soft_time_budget_ms(0.55)
            .with_intensity_bounds(0.04, 0.30)
            .with_max_greediness(0.62)
            .with_max_locality(0.70)
    };

    TabuSearchStrategy::new(feats_arc)
        .with_lambda(7)
        .with_penalty_step(2)
        .with_decay(DecayMode::Multiplicative { num: 94, den: 100 })
        .with_max_penalty(1_000_000_000)
        .with_max_local_steps(1800)
        .with_tabu_tenure(34..=54)
        .with_samples_per_step(120)
        .with_refetch_after_stale(52)
        .with_hard_refetch_every(24)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_restart_on_publish(true)
        .with_reset_on_refetch(true)
        .with_kick_steps_on_reset(5)
        .with_pulse_params(6, 16)
        // locals
        .with_local_op_tuned(
            Box::new(
                ShiftEarlierOnSameBerth::new(1..=1).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(LocalCountTargetTuner::new(18.0, 52.0)),
        )
        .with_local_op_tuned(
            Box::new(
                RelocateSingleBest::new(1..=1).with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(LocalCountTargetTuner::new(20.0, 64.0)),
        )
        .with_local_op_tuned(
            Box::new(SwapPairSameBerth::new(1..=1).with_neighbors(neighbors_same_berth.clone())),
            Box::new(LocalCountTargetTuner::new(36.0, 96.0)),
        )
        .with_local_op_tuned(
            Box::new(
                CrossExchangeAcrossBerths::new(1..=1)
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(LocalCountTargetTuner::new(48.0, 128.0)),
        )
        .with_local_op_tuned(
            Box::new(
                OrOptBlockRelocate::new(2..=3, 1.48).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(OrOptBlockKTuner::default()),
        )
        .with_local_op_tuned(
            Box::new(
                OrOptBlockRelocate::new(5..=9, 1.60).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(OrOptBlockKTuner::default()),
        )
        .with_local_op_tuned(
            Box::new(
                RelocateSingleBestAllowWorsening::new(1..=1)
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(LocalCountTargetTuner::new(12.0, 24.0)),
        )
        .with_local_op_tuned(
            Box::new(RandomRelocateAnywhere::new(1..=1).with_neighbors(neighbors_any.clone())),
            Box::new(LocalCountTargetTuner::new(12.0, 24.0)),
        )
        .with_local_op_tuned(
            Box::new(
                HillClimbRelocateBest::new(1..=1)
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(ultra()),
        )
        .with_local_op_tuned(
            Box::new(
                HillClimbBestSwapSameBerth::new(1..=1).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(ultra()),
        )
        .with_local_op_tuned(
            Box::new(
                RandomizedGreedyRelocateRcl::new(1..=1, 1.80)
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(ultra()),
        )
        .with_local_op_tuned(
            Box::new(
                CrossExchangeBestAcrossBerths::new(1..=1).with_neighbors(neighbors_any.clone()),
            ),
            Box::new(ultra()),
        )
}
