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

use rand::seq::SliceRandom;
use std::{ops::RangeInclusive, sync::atomic::Ordering as AtomicOrdering};

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor, RepairAcceptor},
        neighbors,
        search::{SearchContext, SearchStrategy},
        strategy_support::{
            ApplyOnceByIndex, MedianHistoryEpsilon, StaleTracker, deterministic_kick,
            materially_better, patience_from_exploration_budget,
        },
    },
    model::solver_model::SolverModel,
    search::{
        operator::{DestroyOperator, LocalMoveOperator, RepairOperator},
        operator_library::{
            destroy::{
                RandomKRatioDestroy, ShawRelatedDestroy, TimeClusterDestroy, WorstCostDestroy,
            },
            local::{
                CrossExchangeAcrossBerths, HillClimbRelocateBest, OrOptBlockRelocate,
                RelocateSingleBest, ShiftEarlierOnSameBerth, SwapPairSameBerth,
            },
            repair::{GreedyInsertion, KRegretInsertion},
        },
        planner::{CostEvaluator, DefaultCostEvaluator, PlanningContext},
    },
    state::{
        decisionvar::DecisionVar,
        fitness::Fitness,
        solver_state::{SolverState, SolverStateView},
    },
};

/// Optional periodic refetch policy. Even when `Always`, we still refetch
/// **only when stale** (see `run`); this enum only controls the “periodic due” gate.
#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

/// Standard ILS with:
/// - Local improvement (A)
/// - Ruin (B), Repair (C)
/// - ε-guarded refetch of incumbent when stale (optional periodic cadence)
/// - Deterministic kick after refetch to avoid resynchronization
pub struct IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Operators
    destroy_ops: Vec<Box<dyn DestroyOperator<T, DefaultCostEvaluator, R>>>,
    repair_ops: Vec<Box<dyn RepairOperator<T, DefaultCostEvaluator, R>>>,
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,

    // Acceptors
    local_acceptor: LexStrictAcceptor, // Phase A
    repair_acceptor: RepairAcceptor,   // Phase C

    // Local improvement budget (Phase A)
    max_local_steps: usize,
    local_steps_range: Option<RangeInclusive<usize>>, // if Some, sample per round

    // Local acceptance tweaks
    allow_sideways_in_local: bool, // accept equal fitness
    accept_worsening_local_with_prob: Option<f64>, // (unused in standard ILS; keep for completeness)

    // Ruin-and-repair caps per round
    max_destroy_attempts_per_round: Option<usize>,
    max_repair_attempts_per_round: Option<usize>,

    // Shuffle policy: reshuffle local op order each inner step (or only once per round)
    shuffle_local_each_step: bool,

    // Staleness patience override for refetch (if None, derived from exploration budget)
    stale_min_rounds_override: Option<usize>,

    // Optional periodic cadence; still gated by staleness+materiality
    hard_refetch_every: usize,          // 0 => disabled
    hard_refetch_mode: HardRefetchMode, // IfBetter / Always

    // Post-refetch deterministic kick: apply first k local ops once
    kick_ops_after_refetch: usize, // 0 => no kick
}

impl<T, R> Default for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    pub fn new() -> Self {
        Self {
            destroy_ops: Vec::new(),
            repair_ops: Vec::new(),
            local_ops: Vec::new(),
            local_acceptor: LexStrictAcceptor,
            repair_acceptor: RepairAcceptor,
            max_local_steps: 64,
            local_steps_range: None,
            allow_sideways_in_local: false,
            accept_worsening_local_with_prob: None,
            max_destroy_attempts_per_round: None,
            max_repair_attempts_per_round: None,
            shuffle_local_each_step: true,
            stale_min_rounds_override: None,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
            kick_ops_after_refetch: 4,
        }
    }

    // --------------------- Builder / Tuners ---------------------
    pub fn with_destroy_op(
        mut self,
        op: Box<dyn DestroyOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.destroy_ops.push(op);
        self
    }
    pub fn with_repair_op(
        mut self,
        op: Box<dyn RepairOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.repair_ops.push(op);
        self
    }
    pub fn with_local_op(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.local_ops.push(op);
        self
    }

    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps.max(1);
        self
    }
    pub fn with_local_steps_range(mut self, range: RangeInclusive<usize>) -> Self {
        assert!(!range.is_empty());
        self.local_steps_range = Some(range);
        self
    }

    pub fn with_local_sideways(mut self, yes: bool) -> Self {
        self.allow_sideways_in_local = yes;
        self
    }
    pub fn with_local_worsening_prob(mut self, p: f64) -> Self {
        self.accept_worsening_local_with_prob = Some(p.clamp(0.0, 1.0));
        self
    }

    pub fn with_destroy_attempts(mut self, attempts: usize) -> Self {
        self.max_destroy_attempts_per_round = Some(attempts.max(1));
        self
    }
    pub fn with_repair_attempts(mut self, attempts: usize) -> Self {
        self.max_repair_attempts_per_round = Some(attempts.max(1));
        self
    }

    pub fn with_shuffle_local_each_step(mut self, yes: bool) -> Self {
        self.shuffle_local_each_step = yes;
        self
    }

    /// Provide an explicit staleness threshold (rounds) before refetch is considered.
    /// If not set, patience is derived from exploration budget.
    pub fn with_refetch_after_stale(mut self, rounds: usize) -> Self {
        self.stale_min_rounds_override = Some(rounds.max(1));
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
    pub fn with_kick_ops_after_refetch(mut self, k: usize) -> Self {
        self.kick_ops_after_refetch = k;
        self
    }

    // --------------------- Internal helpers ---------------------

    #[inline]
    fn periodic_refetch_due(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    /// Compute plan.delta_cost via full fitness evals (`CostEvaluator::eval_fitness`).
    /// Keeps correctness without re-implementing base-delta logic.
    fn set_plan_delta_via_eval<'m>(
        &self,
        model: &SolverModel<'m, T>,
        eval: &DefaultCostEvaluator,
        current_vars: &[DecisionVar<T>],
        plan: &mut crate::state::plan::Plan<'m, T>,
    ) {
        // Build new DV vector with “last patch wins”.
        let mut new_vars = current_vars.to_vec();
        for p in &plan.decision_var_patches {
            let i = p.index.get();
            if i < new_vars.len() {
                new_vars[i] = p.patch;
            }
        }

        let old_fit: Fitness = eval.eval_fitness(model, current_vars);
        let new_fit: Fitness = eval.eval_fitness(model, &new_vars);

        plan.delta_cost = new_fit.cost.saturating_sub(old_fit.cost);
    }
}

/// Adapter to run a deterministic “kick” by applying the first k local ops once.
struct KickAdapter<'a, 'm, 'p, Tnum, Rng>
where
    Tnum: SolveNumeric,
    Rng: rand::Rng,
{
    model: &'m SolverModel<'p, Tnum>,
    eval: &'a DefaultCostEvaluator,
    locals: &'a [Box<dyn LocalMoveOperator<Tnum, DefaultCostEvaluator, Rng>>],
    state: &'a mut SolverState<'p, Tnum>,
    dv_buf: &'a mut [DecisionVar<Tnum>],
    rng: &'a mut Rng,
}

impl<'a, 'm, 'p, Tnum, Rng> ApplyOnceByIndex for KickAdapter<'a, 'm, 'p, Tnum, Rng>
where
    Tnum: SolveNumeric,
    Rng: rand::Rng,
{
    fn apply_once(&mut self, index: usize) -> bool {
        if index >= self.locals.len() {
            return false;
        }
        let op = &self.locals[index];

        let mut pc = PlanningContext::new(self.model, &*self.state, self.eval, self.dv_buf);

        if let Some(mut plan) = op.propose(&mut pc, self.rng) {
            // Compute delta using evaluator-based fitness before/after for correctness.
            let cur_vars = self.state.decision_variables();
            let mut new_vars = cur_vars.to_vec();
            for p in &plan.decision_var_patches {
                let i = p.index.get();
                if i < new_vars.len() {
                    new_vars[i] = p.patch;
                }
            }
            let old_fit = self.eval.eval_fitness(self.model, cur_vars);
            let new_fit = self.eval.eval_fitness(self.model, &new_vars);
            plan.delta_cost = new_fit.cost.saturating_sub(old_fit.cost);

            let prev = *self.state.fitness();
            self.state.apply_plan(plan);
            return *self.state.fitness() != prev;
        }
        false
    }
}

impl<T, R> SearchStrategy<T, R> for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "Iterated Local Search"
    }

    #[tracing::instrument(level = "debug", name = "ILS Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let model = context.model();
        let stop = context.stop();

        if self.local_ops.is_empty() && (self.destroy_ops.is_empty() || self.repair_ops.is_empty())
        {
            tracing::warn!(
                "ILS: no operators configured (local={}, destroy={}, repair={})",
                self.local_ops.len(),
                self.destroy_ops.len(),
                self.repair_ops.len()
            );
            return;
        }

        // Seed working state from the global incumbent once at start.
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();

        debug_assert_eq!(
            current.decision_variables().len(),
            model.flexible_requests_len(),
            "incumbent DV vector length must match model"
        );

        // Scratch buffer for PlanningContext.
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Stale tracking & epsilon (data-driven).
        let mut stale = StaleTracker::new(*current.fitness(), /*history_cap*/ 32);
        let mut eps_src = MedianHistoryEpsilon::new(/*history_cap*/ 32, /*min_eps*/ 1);

        // Derive default patience from exploration budget (neutral); allow external override.
        let destroy_cap = self
            .max_destroy_attempts_per_round
            .unwrap_or_else(|| self.destroy_ops.len().max(1));
        let repair_cap = self
            .max_repair_attempts_per_round
            .unwrap_or_else(|| self.repair_ops.len().max(1));
        let batches_per_round = 1 + destroy_cap + repair_cap;

        let inner_steps_mean = match &self.local_steps_range {
            Some(r) => ((*r.start() + *r.end()) / 2).max(1),
            None => self.max_local_steps.max(1),
        };

        let derived_patience = patience_from_exploration_budget(
            batches_per_round,
            inner_steps_mean,
            self.shuffle_local_each_step,
        );
        let patience_s = self.stale_min_rounds_override.unwrap_or(derived_patience);

        let eval = DefaultCostEvaluator;

        let mut outer_rounds = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            outer_rounds = outer_rounds.saturating_add(1);

            // --------------- Phase A: Local improvement ---------------
            let mut improved_in_round = false;
            let steps_budget = {
                match &self.local_steps_range {
                    Some(r) => {
                        let lo = *r.start();
                        let hi = *r.end();
                        if lo == hi {
                            lo
                        } else {
                            context.rng().random_range(lo..=hi)
                        }
                    }
                    None => self.max_local_steps,
                }
            };

            // Persistent per-round order (if not reshuffling each step).
            let mut round_order: Vec<usize> = (0..self.local_ops.len()).collect();
            if !self.shuffle_local_each_step {
                round_order.shuffle(context.rng());
            }

            for _ in 0..steps_budget {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut accepted_this_step = false;

                // Decide operator visiting order for this step
                let order: Vec<usize> = if self.shuffle_local_each_step {
                    let mut v = (0..self.local_ops.len()).collect::<Vec<_>>();
                    v.shuffle(context.rng());
                    v
                } else {
                    round_order.clone()
                };

                for &i in &order {
                    let op = &self.local_ops[i];

                    let mut pc =
                        PlanningContext::new(model, &current, &eval, dv_buf.as_mut_slice());

                    if let Some(mut plan) = op.propose(&mut pc, context.rng()) {
                        // Compute delta via evaluator-based full fitness (correctness-first).
                        self.set_plan_delta_via_eval(
                            model,
                            &eval,
                            current.decision_variables(),
                            &mut plan,
                        );

                        let mut tmp = current.clone();
                        tmp.apply_plan(plan);

                        let cur_fit = current.fitness();
                        let tmp_fit = tmp.fitness();

                        let better = self.local_acceptor.accept(cur_fit, tmp_fit);
                        let sideways = self.allow_sideways_in_local && (tmp_fit == cur_fit);
                        let worse_random = if !better && !sideways {
                            if let Some(p) = self.accept_worsening_local_with_prob {
                                context.rng().random::<f64>() < p
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if better || sideways || worse_random {
                            if better {
                                // Feed strict-improvement size into epsilon history.
                                let drop = cur_fit.cost.saturating_sub(tmp_fit.cost).max(0);
                                eps_src.record(drop);
                                improved_in_round = true;
                            }
                            current = tmp;
                            accepted_this_step = true;

                            // Publish if we beat the shared incumbent (no-op otherwise).
                            let _ = context.shared_incumbent().try_update(&current, model);

                            tracing::trace!(
                                "ILS: accepted local op {} (better={}, sideways={}, worse_rand={})",
                                op.name(),
                                better,
                                sideways,
                                worse_random
                            );
                            break; // restart local climb from updated state
                        }
                    }
                }

                if !accepted_this_step {
                    break; // no acceptable local move found right now
                }
            }

            if !improved_in_round {
                // ---------------------- Phase B: Destroy + Phase C: Repair ----------------------
                if self.destroy_ops.is_empty() || self.repair_ops.is_empty() {
                    tracing::debug!("ILS: no perturbation operators configured; stopping.");
                    break 'outer;
                }

                let baseline = current.clone();

                // Destroy attempts
                let destroy_attempts = destroy_cap;
                let mut destroyed = false;
                for _ in 0..destroy_attempts {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }
                    let idx = if self.destroy_ops.len() == 1 {
                        0
                    } else {
                        context.rng().random_range(0..self.destroy_ops.len())
                    };
                    let d = &self.destroy_ops[idx];

                    let mut pc =
                        PlanningContext::new(model, &current, &eval, dv_buf.as_mut_slice());
                    if let Some(mut plan) = d.propose(&mut pc, context.rng()) {
                        self.set_plan_delta_via_eval(
                            model,
                            &eval,
                            current.decision_variables(),
                            &mut plan,
                        );
                        current.apply_plan(plan);
                        destroyed = true;
                        tracing::trace!("ILS: applied destroy op {}", d.name());
                        break;
                    }
                }

                if !destroyed {
                    tracing::debug!("ILS: no destroy operator produced a plan; stopping.");
                    break 'outer;
                }

                // Repair attempts
                let repair_attempts = repair_cap;
                let mut repaired_and_accepted = false;

                let mut repair_indices: Vec<usize> = (0..self.repair_ops.len()).collect();
                repair_indices.shuffle(context.rng());

                for &ri in repair_indices.iter().take(repair_attempts) {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let r = &self.repair_ops[ri];
                    let mut temp = current.clone();

                    let mut pc = PlanningContext::new(model, &temp, &eval, dv_buf.as_mut_slice());

                    if let Some(mut plan) = r.repair(&mut pc, context.rng()) {
                        self.set_plan_delta_via_eval(
                            model,
                            &eval,
                            temp.decision_variables(),
                            &mut plan,
                        );
                        temp.apply_plan(plan);

                        if self
                            .repair_acceptor
                            .accept(baseline.fitness(), temp.fitness())
                        {
                            current = temp;
                            repaired_and_accepted = true;

                            let _ = context.shared_incumbent().try_update(&current, model);
                            tracing::trace!("ILS: accepted repair op {}", r.name());
                            break;
                        }
                    }
                }

                if !repaired_and_accepted {
                    // Revert to baseline if repair couldn't beat it.
                    current = baseline;
                    tracing::debug!("ILS: repair failed to beat baseline.");
                }
            }

            // --- End of outer round: update stale tracking and ε history on strict improvement ---
            if let Some(delta) = stale.on_round_end(*current.fitness()) {
                eps_src.record(delta);
            }

            // --- Refetch: ONLY when stale, and incumbent is materially better (ε-guarded). ---
            if stale.is_stale(patience_s) {
                let inc_fit = context.shared_incumbent().peek();
                let materially = materially_better(current.fitness(), &inc_fit, eps_src.epsilon());

                // Even with periodic cadence configured, we *still* require staleness + materiality.
                let periodic_due = self.periodic_refetch_due(outer_rounds);
                let allowed_by_mode = match self.hard_refetch_mode {
                    HardRefetchMode::Always => true,
                    HardRefetchMode::IfBetter => true, // materiality already checked
                };

                if materially && allowed_by_mode && (periodic_due || self.hard_refetch_every == 0) {
                    tracing::debug!(
                        "ILS: stale refetch (round={}, patience={}, eps={}) current={} incumbent={}",
                        outer_rounds,
                        patience_s,
                        eps_src.epsilon(),
                        current.fitness(),
                        inc_fit
                    );

                    // Refetch to incumbent snapshot
                    let mut snap = context.shared_incumbent().snapshot();

                    // Deterministic kick to avoid herd/sync lock-in (apply first K local ops once).
                    if self.kick_ops_after_refetch > 0 && !self.local_ops.is_empty() {
                        let k = self.kick_ops_after_refetch.min(self.local_ops.len());
                        let mut adapter = KickAdapter {
                            model,
                            eval: &eval,
                            locals: &self.local_ops,
                            state: &mut snap,
                            dv_buf: dv_buf.as_mut_slice(),
                            rng: context.rng(),
                        };
                        let _changed = deterministic_kick(&mut adapter, k);
                    }

                    current = snap;

                    // Cooldown until the next strict improvement to avoid immediate refetch loops.
                    stale.arm_cooldown_until_next_improvement();
                }
            }
        }

        // Final publish (no-op if we didn't beat the incumbent).
        let _ = context.shared_incumbent().try_update(&current, model);
    }
}

pub fn ils_strategy<T, R>(model: &SolverModel<T>) -> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    // ILS — “standard but bolder” (no worsening in local; bigger ruins; stronger repair).
    IteratedLocalSearchStrategy::new()
        // -------- Local budget & acceptance --------
        .with_local_steps_range(1200..=2200)
        .with_local_sideways(true)
        .with_local_worsening_prob(0.0) // standard ILS: no worsening in local phase
        // -------- Ruin/Repair attempts per outer round --------
        .with_destroy_attempts(12)
        .with_repair_attempts(28)
        .with_shuffle_local_each_step(true)
        // -------- Refetch cadence (ε-guarded) --------
        .with_refetch_after_stale(40)
        .with_hard_refetch_every(14)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_kick_ops_after_refetch(8)
        // ------------------------- Local improvement (compact core) -------------------------
        .with_local_op(Box::new(
            RelocateSingleBest::new(24..=64).with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            SwapPairSameBerth::new(40..=100).with_neighbors(neighbors_same_berth.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeAcrossBerths::new(48..=120)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbRelocateBest::new(24..=72)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            OrOptBlockRelocate::new(4..=8, 1.25..=1.65)
                .with_neighbors(neighbors_same_berth.clone()),
        ))
        .with_local_op(Box::new(
            ShiftEarlierOnSameBerth::new(16..=48).with_neighbors(neighbors_same_berth.clone()),
        ))
        // ---------------------- Destroy (moderate → bold) ----------------------
        .with_destroy_op(Box::new(
            RandomKRatioDestroy::new(0.32..=0.58).with_neighbors(neighbors_any.clone()),
        ))
        .with_destroy_op(Box::new(
            WorstCostDestroy::new(0.30..=0.48).with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_destroy_op(Box::new(
            ShawRelatedDestroy::new(
                0.28..=0.40,
                1.6..=2.2,
                1.into(), // weight_abs_start_gap
                1.into(), // weight_abs_end_gap
                5.into(), // penalty_berth_mismatch
            )
            .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_destroy_op(Box::new(
            TimeClusterDestroy::<T>::new(
                0.32..=0.50,
                berth_alloc_core::prelude::TimeDelta::new(24.into()),
            )
            .with_alpha(1.55..=1.90)
            .with_neighbors(neighbors_any.clone()),
        ))
        // ---------------------- Repair (slightly stronger K) ----------------------
        .with_repair_op(Box::new(KRegretInsertion::new(8..=11)))
        .with_repair_op(Box::new(GreedyInsertion))
}
