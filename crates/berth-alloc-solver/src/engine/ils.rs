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

use berth_alloc_core::prelude::{Cost, TimeDelta};

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor, RepairAcceptor},
        adaptive::{
            ops_book::OperatorBook,
            selection::SoftmaxSelector,
            tuning::{
                DefaultOperatorTuner, DestroyRatioTuner, KRegretKTuner, LocalCountTargetTuner,
                OrOptBlockKTuner,
            },
        },
        neighbors,
        search::{SearchContext, SearchStrategy},
    },
    model::solver_model::SolverModel,
    search::{
        operator::{DestroyOperator, LocalMoveOperator, OperatorKind, RepairOperator},
        operator_library::{
            destroy::{
                BerthBandDestroy, BerthNeighborsDestroy, ProcessingTimeClusterDestroy,
                RandomKRatioDestroy, ShawRelatedDestroy, StringBlockDestroy, TimeClusterDestroy,
                TimeWindowBandDestroy, WorstCostDestroy,
            },
            local::{
                CrossExchangeAcrossBerths, CrossExchangeBestAcrossBerths,
                HillClimbBestSwapSameBerth, HillClimbRelocateBest, OrOptBlockRelocate,
                RandomRelocateAnywhere, RandomizedGreedyRelocateRcl, RelocateSingleBest,
                RelocateSingleBestAllowWorsening, ShiftEarlierOnSameBerth, SwapPairSameBerth,
            },
            repair::{GreedyInsertion, KRegretInsertion, RandomizedGreedyInsertion},
        },
        planner::{DefaultCostEvaluator, PlanningContext},
    },
    state::solver_state::{SolverState, SolverStateView},
};
use std::{ops::RangeInclusive, sync::atomic::Ordering as AtomicOrdering};

/// Periodic hard-refetch policy.
#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    /// Replace working state with the shared incumbent only if the incumbent is strictly better.
    IfBetter,
    /// Replace unconditionally on the period (for stronger convergence/sync).
    Always,
}

pub struct IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Operators
    destroy_ops: Vec<Box<dyn DestroyOperator<T, DefaultCostEvaluator, R>>>,
    repair_ops: Vec<Box<dyn RepairOperator<T, DefaultCostEvaluator, R>>>,
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,

    // Adaptive operator books (selection + stats + tuners)
    local_book: OperatorBook<T, R>,
    destroy_book: OperatorBook<T, R>,
    repair_book: OperatorBook<T, R>,

    // Acceptors
    local_acceptor: LexStrictAcceptor, // Phase A
    repair_acceptor: RepairAcceptor,   // Phase C

    // Local improvement budget (Phase A)
    max_local_steps: usize,
    local_steps_range: Option<RangeInclusive<usize>>, // if Some, sample per round

    // Local acceptance tweaks
    allow_sideways_in_local: bool, // accept equal fitness
    accept_worsening_local_with_prob: Option<f64>, // probabilistic accept for worsening

    // How many destroy/repair attempts per outer round (caps)
    max_destroy_attempts_per_round: Option<usize>,
    max_repair_attempts_per_round: Option<usize>,

    // Shuffle policy: reshuffle local op order each inner step (or only once per round)
    shuffle_local_each_step: bool,

    /// After this many outer iterations without improvement, refetch if incumbent is better.
    /// 0 => disabled.
    refetch_after_stale: usize,

    /// Perform a hard refetch every N outer rounds. 0 => disabled.
    hard_refetch_every: usize,

    /// Policy for periodic hard refetch.
    hard_refetch_mode: HardRefetchMode,
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

            // initialize adaptive books with Softmax selector
            local_book: OperatorBook::new(
                OperatorKind::Local,
                Box::new(SoftmaxSelector::default()),
            ),
            destroy_book: OperatorBook::new(
                OperatorKind::Destroy,
                Box::new(SoftmaxSelector::default()),
            ),
            repair_book: OperatorBook::new(
                OperatorKind::Repair,
                Box::new(SoftmaxSelector::default()),
            ),

            local_acceptor: LexStrictAcceptor,
            repair_acceptor: RepairAcceptor,
            max_local_steps: 64,
            local_steps_range: None,
            allow_sideways_in_local: false,
            accept_worsening_local_with_prob: None,
            max_destroy_attempts_per_round: None,
            max_repair_attempts_per_round: None,
            shuffle_local_each_step: true,
            refetch_after_stale: 8,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
        }
    }

    // --------------------- Builder ---------------------

    /// Register a destroy operator with a default tuner.
    pub fn with_destroy_op(
        mut self,
        op: Box<dyn DestroyOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.destroy_ops.push(op);
        let _ = self
            .destroy_book
            .register_operator(Box::new(DefaultOperatorTuner::default()));
        self
    }

    /// Register a destroy operator with a custom tuner.
    pub fn with_destroy_op_tuned(
        mut self,
        op: Box<dyn DestroyOperator<T, DefaultCostEvaluator, R>>,
        tuner: Box<dyn crate::engine::adaptive::tuning::OperatorTuner<T>>,
    ) -> Self {
        self.destroy_ops.push(op);
        let _ = self.destroy_book.register_operator(tuner);
        self
    }

    /// Register a repair operator with a default tuner.
    pub fn with_repair_op(
        mut self,
        op: Box<dyn RepairOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.repair_ops.push(op);
        let _ = self
            .repair_book
            .register_operator(Box::new(DefaultOperatorTuner::default()));
        self
    }

    /// Register a repair operator with a custom tuner.
    pub fn with_repair_op_tuned(
        mut self,
        op: Box<dyn RepairOperator<T, DefaultCostEvaluator, R>>,
        tuner: Box<dyn crate::engine::adaptive::tuning::OperatorTuner<T>>,
    ) -> Self {
        self.repair_ops.push(op);
        let _ = self.repair_book.register_operator(tuner);
        self
    }

    /// Register a local operator with a default tuner.
    pub fn with_local_op(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.local_ops.push(op);
        let _ = self
            .local_book
            .register_operator(Box::new(DefaultOperatorTuner::default()));
        self
    }

    /// Register a local operator with a custom tuner.
    pub fn with_local_op_tuned(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
        tuner: Box<dyn crate::engine::adaptive::tuning::OperatorTuner<T>>,
    ) -> Self {
        self.local_ops.push(op);
        let _ = self.local_book.register_operator(tuner);
        self
    }

    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps.max(1);
        self
    }
    /// Sample a per-round local step budget; overrides the fixed `max_local_steps` if set.
    pub fn with_local_steps_range(mut self, range: RangeInclusive<usize>) -> Self {
        assert!(!range.is_empty());
        self.local_steps_range = Some(range);
        self
    }

    /// Accept equal-fitness local moves (sideways).
    pub fn with_local_sideways(mut self, yes: bool) -> Self {
        self.allow_sideways_in_local = yes;
        self
    }
    /// Randomly accept worsening local moves with probability `p` (0..1].
    pub fn with_local_worsening_prob(mut self, p: f64) -> Self {
        self.accept_worsening_local_with_prob = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Cap the number of destroy attempts per round (default: len(destroy_ops)).
    pub fn with_destroy_attempts(mut self, attempts: usize) -> Self {
        self.max_destroy_attempts_per_round = Some(attempts.max(1));
        self
    }
    /// Cap the number of repair attempts per round (default: len(repair_ops)).
    pub fn with_repair_attempts(mut self, attempts: usize) -> Self {
        self.max_repair_attempts_per_round = Some(attempts.max(1));
        self
    }

    /// If true, reshuffle local operator order at each step; else only once per round.
    pub fn with_shuffle_local_each_step(mut self, yes: bool) -> Self {
        self.shuffle_local_each_step = yes;
        self
    }

    /// Set stale refetch threshold. 0 disables staleness-based refetch.
    pub fn with_refetch_after_stale(mut self, rounds: usize) -> Self {
        self.refetch_after_stale = rounds;
        self
    }
    /// Set periodic hard-refetch cadence. 0 disables periodic hard refetch.
    pub fn with_hard_refetch_every(mut self, period: usize) -> Self {
        self.hard_refetch_every = period;
        self
    }
    pub fn with_hard_refetch_mode(mut self, mode: HardRefetchMode) -> Self {
        self.hard_refetch_mode = mode;
        self
    }

    // --------------------- Internal helpers ---------------------
    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    #[inline]
    fn maybe_apply_periodic_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        outer_rounds: usize,
    ) {
        if !self.should_hard_refetch(outer_rounds) {
            return;
        }
        let best_now = context.shared_incumbent().peek();
        let do_fetch = match self.hard_refetch_mode {
            HardRefetchMode::IfBetter => best_now < *current.fitness(),
            HardRefetchMode::Always => true,
        };
        if do_fetch {
            tracing::debug!(
                "ILS: periodic refetch at round {} (current {}, incumbent {})",
                outer_rounds,
                current.fitness(),
                best_now
            );
            *current = context.shared_incumbent().snapshot();
        }
    }

    #[inline]
    fn maybe_apply_stale_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        stale_rounds: usize,
    ) {
        if self.refetch_after_stale == 0 || stale_rounds < self.refetch_after_stale {
            return;
        }
        let best_now = context.shared_incumbent().peek();
        if best_now < *current.fitness() {
            tracing::debug!(
                "ILS: staleness refetch after {} stale rounds ({} -> {})",
                stale_rounds,
                current.fitness(),
                best_now
            );
            *current = context.shared_incumbent().snapshot();
        } else {
            tracing::trace!(
                "ILS: staleness refetch skipped; incumbent ({}) not better than current ({})",
                best_now,
                current.fitness()
            );
        }
    }

    #[inline]
    fn draw_local_budget<Rng: rand::Rng>(&self, rng: &mut Rng) -> usize {
        match self.local_steps_range.as_ref() {
            Some(r) => {
                let lo = *r.start();
                let hi = *r.end();
                if lo == hi {
                    lo
                } else {
                    rng.random_range(lo..=hi)
                }
            }
            None => self.max_local_steps,
        }
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

        // Pre-allocate scratch buffer for PlanningContext.
        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Bookkeeping.
        let mut stale_rounds = 0usize;
        let mut outer_rounds = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }

            // Periodic hard refetch policy.
            self.maybe_apply_periodic_refetch(&mut current, context, outer_rounds);

            // --------------- Phase A: Local improvement ---------------
            let mut improved_in_round = false;
            let steps_budget = self.draw_local_budget(context.rng());

            for _ in 0..steps_budget {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut accepted_this_step = false;

                // Global stats for tuning/selection
                let global_stats = current.stats(model);

                // Stagnation from outer stale rounds (scale heuristic).
                let stuck_factor = (stale_rounds as f64 / 50.0).min(1.0);
                let stagnation = crate::engine::adaptive::tuning::Stagnation {
                    stale_rounds,
                    stuck_factor,
                };

                // Retune all local operators once per step
                self.local_book.retune_all(&global_stats, &stagnation);

                // Try up to N attempts this step (N = number of local operators)
                let attempts = self.local_ops.len().max(1);
                for _try in 0..attempts {
                    // Select operator index via adaptive selector (stagnation-aware)
                    let i = self
                        .local_book
                        .select(&global_stats, &stagnation, context.rng());
                    let op = &mut self.local_ops[i];

                    // Push tuning to operator
                    let tuning = *self.local_book.tuning_for(i);
                    op.tune(&tuning, &global_stats);

                    // Time and propose
                    let t0 = self.local_book.propose_started();
                    let mut pc = PlanningContext::new(
                        model,
                        &current,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );
                    if let Some(mut plan) = op.propose(&mut pc, context.rng()) {
                        self.local_book.record_propose(i, t0, true);

                        // Recompute base delta for this plan before applying
                        use crate::model::index::RequestIndex;
                        use crate::state::decisionvar::DecisionVar;
                        use std::collections::HashMap;

                        let mut base_delta: Cost = Cost::from(0);

                        // Keep last patch per request (in case operators emit multiple)
                        let mut last: HashMap<usize, DecisionVar<T>> = HashMap::new();
                        for p in &plan.decision_var_patches {
                            last.insert(p.index.get(), p.patch);
                        }

                        for (ri_u, patch) in last {
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

                        let accepted = better || sideways || worse_random;

                        // Record outcome with true/base delta (new - old)
                        self.local_book
                            .record_outcome(i, accepted, base_delta as f64);

                        if accepted {
                            current = tmp;
                            accepted_this_step = true;
                            improved_in_round |= better; // count only strict better as “improved”
                            let _ = context.shared_incumbent().try_update(&current, model);
                            tracing::trace!(
                                "ILS: accepted local op {} (better={}, sideways={}, worse_random={})",
                                op.name(),
                                better,
                                sideways,
                                worse_random
                            );
                            break; // restart local climb from updated state
                        }
                    } else {
                        self.local_book.record_propose(i, t0, false);
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
                let destroy_attempts = self
                    .max_destroy_attempts_per_round
                    .unwrap_or_else(|| self.destroy_ops.len().max(1));

                let mut destroyed = false;
                for _ in 0..destroy_attempts {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let global_stats = current.stats(model);

                    let stuck_factor = (stale_rounds as f64 / 50.0).min(1.0);
                    let stagnation = crate::engine::adaptive::tuning::Stagnation {
                        stale_rounds,
                        stuck_factor,
                    };

                    self.destroy_book.retune_all(&global_stats, &stagnation);

                    let idx = self
                        .destroy_book
                        .select(&global_stats, &stagnation, context.rng());
                    let d = &mut self.destroy_ops[idx];

                    // Tune operator
                    let tuning = *self.destroy_book.tuning_for(idx);
                    d.tune(&tuning, &global_stats);

                    let t0 = self.destroy_book.propose_started();
                    let mut pc = PlanningContext::new(
                        model,
                        &current,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );
                    if let Some(mut plan) = d.propose(&mut pc, context.rng()) {
                        self.destroy_book.record_propose(idx, t0, true);

                        // Recompute base delta vs current
                        use crate::model::index::RequestIndex;
                        use crate::state::decisionvar::DecisionVar;
                        use std::collections::HashMap;

                        let mut base_delta: Cost = Cost::from(0);
                        let mut last: HashMap<usize, DecisionVar<T>> = HashMap::new();
                        for p in &plan.decision_var_patches {
                            last.insert(p.index.get(), p.patch);
                        }
                        for (ri_u, patch) in last {
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
                        destroyed = true;
                        self.destroy_book
                            .record_outcome(idx, true, base_delta as f64);

                        tracing::trace!("ILS: applied destroy op {}", d.name());
                        break;
                    } else {
                        self.destroy_book.record_propose(idx, t0, false);
                    }
                }

                if !destroyed {
                    tracing::debug!("ILS: no destroy operator produced a plan; stopping.");
                    break 'outer;
                }

                // Repair attempts
                let repair_attempts = self
                    .max_repair_attempts_per_round
                    .unwrap_or_else(|| self.repair_ops.len().max(1));

                let mut repaired_and_accepted = false;

                for _ in 0..repair_attempts {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let global_stats = current.stats(model);

                    let stuck_factor = (stale_rounds as f64 / 50.0).min(1.0);
                    let stagnation = crate::engine::adaptive::tuning::Stagnation {
                        stale_rounds,
                        stuck_factor,
                    };

                    self.repair_book.retune_all(&global_stats, &stagnation);

                    let ri = self
                        .repair_book
                        .select(&global_stats, &stagnation, context.rng());
                    let r = &mut self.repair_ops[ri];

                    // Tune operator
                    let tuning = *self.repair_book.tuning_for(ri);
                    r.tune(&tuning, &global_stats);

                    let mut temp = current.clone();
                    let mut pc = PlanningContext::new(
                        model,
                        &temp,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );
                    let t0 = self.repair_book.propose_started();
                    if let Some(mut plan) = r.repair(&mut pc, context.rng()) {
                        self.repair_book.record_propose(ri, t0, true);

                        // Recompute base delta vs temp (pre-application)
                        use crate::model::index::RequestIndex;
                        use crate::state::decisionvar::DecisionVar;
                        use std::collections::HashMap;

                        let mut base_delta: Cost = Cost::from(0);
                        let mut last: HashMap<usize, DecisionVar<T>> = HashMap::new();
                        for p in &plan.decision_var_patches {
                            last.insert(p.index.get(), p.patch);
                        }
                        for (ri_u, patch) in last {
                            let req_ix = RequestIndex::new(ri_u);
                            let old_dv = temp.decision_variables()[req_ix.get()];
                            if let DecisionVar::Assigned(old) = old_dv
                                && let Some(c) = model.cost_of_assignment(
                                    req_ix,
                                    old.berth_index,
                                    old.start_time,
                                )
                            {
                                base_delta = base_delta.saturating_sub(c);
                            }
                            if let DecisionVar::Assigned(new_dec) = patch
                                && let Some(c) = model.cost_of_assignment(
                                    req_ix,
                                    new_dec.berth_index,
                                    new_dec.start_time,
                                )
                            {
                                base_delta = base_delta.saturating_add(c);
                            }
                        }
                        plan.delta_cost = base_delta;

                        temp.apply_plan(plan);

                        if self
                            .repair_acceptor
                            .accept(baseline.fitness(), temp.fitness())
                        {
                            current = temp;
                            repaired_and_accepted = true;

                            self.repair_book.record_outcome(ri, true, base_delta as f64);

                            // Publish improvement vs baseline (may or may not beat global best).
                            let _ = context.shared_incumbent().try_update(&current, model);
                            tracing::trace!("ILS: accepted repair op {}", r.name());
                            break;
                        } else {
                            // produced but rejected
                            self.repair_book.record_outcome(ri, false, 0.0);
                        }
                    } else {
                        self.repair_book.record_propose(ri, t0, false);
                    }
                }

                if repaired_and_accepted {
                    stale_rounds = 0;
                } else {
                    // Revert to baseline if repair couldn't beat it.
                    current = baseline;
                    stale_rounds = stale_rounds.saturating_add(1);
                    tracing::debug!(
                        "ILS: repair failed to beat baseline (stale_rounds={})",
                        stale_rounds
                    );
                }
            } else {
                stale_rounds = 0;
            }

            // Staleness-triggered refetch (only if incumbent strictly better).
            self.maybe_apply_stale_refetch(&mut current, context, stale_rounds);

            outer_rounds = outer_rounds.saturating_add(1);
        }

        // Final publish (harmless if we didn't beat the incumbent).
        let _ = context.shared_incumbent().try_update(&current, model);
    }
}

// ============ ILS (aggressive, fast converge, two-tier K-regret) ============
pub fn ils_strategy<T, R>(model: &SolverModel<T>) -> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    use crate::engine::adaptive::tuning::{
        LocalCountTargetTuner, OrOptBlockKTuner, WorkBudgetTuner,
    };

    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    // clamp heavy locals/shakers so they never hog the loop
    let ultra = || {
        WorkBudgetTuner::default()
            .with_soft_time_budget_ms(0.50)
            .with_intensity_bounds(0.04, 0.30)
            .with_max_greediness(0.60)
            .with_max_locality(0.70)
    };

    IteratedLocalSearchStrategy::new()
        // cadence & acceptance
        .with_local_steps_range(900..=1600)
        .with_local_sideways(true)
        .with_local_worsening_prob(0.020)
        // attempts per round
        .with_destroy_attempts(12)
        .with_repair_attempts(28)
        .with_shuffle_local_each_step(true)
        // sync
        .with_refetch_after_stale(45)
        .with_hard_refetch_every(24)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        // -------- Local improvement (cheap first) --------
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
        // micro Or-Opt to seal tiny gaps early
        .with_local_op_tuned(
            Box::new(
                OrOptBlockRelocate::new(2..=3, 1.48).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(OrOptBlockKTuner::default()),
        )
        // main Or-Opt for medium blocks
        .with_local_op_tuned(
            Box::new(
                OrOptBlockRelocate::new(6..=10, 1.62).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(OrOptBlockKTuner::default()),
        )
        // diversification (light)
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
        // hill climbers (clamped)
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
        // ---------------- Destroy operators (tighter windows) ----------------
        .with_destroy_op_tuned(
            Box::new(RandomKRatioDestroy::new(0.0).with_neighbors(neighbors_any.clone())),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.24,
                ratio_max: 0.40,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(
                WorstCostDestroy::new(0.0).with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.30,
                ratio_max: 0.42,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(
                ShawRelatedDestroy::new(0.0, 1.80, 1.into(), 1.into(), 5.into())
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.26,
                ratio_max: 0.38,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(
                TimeClusterDestroy::<T>::new(0.0, TimeDelta::new(24.into()))
                    .with_alpha(1.60)
                    .with_neighbors(neighbors_any.clone()),
            ),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.30,
                ratio_max: 0.44,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(
                TimeWindowBandDestroy::<T>::new(0.0, 1.60, TimeDelta::new(16.into()))
                    .with_neighbors(neighbors_any.clone()),
            ),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.42,
                ratio_max: 0.54,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(BerthBandDestroy::new(0.0, 1.60, 1)),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.24,
                ratio_max: 0.36,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(StringBlockDestroy::new(0.0).with_alpha(1.80)),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.30,
                ratio_max: 0.42,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(
                BerthNeighborsDestroy::new(0.0, 1.60).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.26,
                ratio_max: 0.40,
                ..Default::default()
            }),
        )
        .with_destroy_op_tuned(
            Box::new(
                ProcessingTimeClusterDestroy::new(0.0, 1.85)
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(DestroyRatioTuner {
                ratio_min: 0.24,
                ratio_max: 0.34,
                ..Default::default()
            }),
        )
        // ---------------- Repair (two-tier K-regret) ----------------
        .with_repair_op_tuned(
            Box::new(KRegretInsertion::new(1..=1)),
            Box::new(KRegretKTuner {
                k_min: 8.0,
                k_max: 9.0,
                soft_time_ms: 0.55,
                min_intensity: 0.25,
                max_intensity: 0.70,
            }),
        )
        .with_repair_op_tuned(
            Box::new(KRegretInsertion::new(1..=1)),
            Box::new(KRegretKTuner {
                k_min: 10.0,
                k_max: 12.0,
                soft_time_ms: 0.75,
                min_intensity: 0.45,
                max_intensity: 0.95,
            }),
        )
        .with_repair_op_tuned(
            Box::new(RandomizedGreedyInsertion::new(1.65)),
            Box::new(DefaultOperatorTuner::default()),
        )
        .with_repair_op_tuned(
            Box::new(GreedyInsertion),
            Box::new(DefaultOperatorTuner::default()),
        )
        // -------- Post-repair shakers (clamped) --------
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
