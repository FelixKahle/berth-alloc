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

use berth_alloc_core::prelude::{Cost, TimeDelta};
use rand::seq::SliceRandom;

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor, RepairAcceptor},
        neighbors,
        search::{SearchContext, SearchStrategy},
    },
    model::solver_model::SolverModel,
    search::{
        operator::{DestroyOperator, LocalMoveOperator, RepairOperator},
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

            // Prepare a (maybe) persistent per-round order if we don't shuffle every step.
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

                    let mut pc = PlanningContext::new(
                        model,
                        &current,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );
                    if let Some(mut plan) = op.propose(&mut pc, context.rng()) {
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
                                ) {
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

                        if better || sideways || worse_random {
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
                    let idx = if self.destroy_ops.len() == 1 {
                        0
                    } else {
                        context.rng().random_range(0..self.destroy_ops.len())
                    };
                    let d = &self.destroy_ops[idx];

                    let mut pc = PlanningContext::new(
                        model,
                        &current,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );
                    if let Some(mut plan) = d.propose(&mut pc, context.rng()) {
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
                                ) {
                                    base_delta = base_delta.saturating_add(c);
                                }
                        }
                        plan.delta_cost = base_delta;

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
                let repair_attempts = self
                    .max_repair_attempts_per_round
                    .unwrap_or_else(|| self.repair_ops.len().max(1));

                let mut repaired_and_accepted = false;
                // We'll iterate repairs in a shuffled order each round and cap by attempts.
                let mut repair_indices: Vec<usize> = (0..self.repair_ops.len()).collect();
                repair_indices.shuffle(context.rng());

                for &ri in repair_indices.iter().take(repair_attempts) {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let r = &self.repair_ops[ri];
                    let mut temp = current.clone();
                    let mut pc = PlanningContext::new(
                        model,
                        &temp,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );
                    if let Some(mut plan) = r.repair(&mut pc, context.rng()) {
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
                                ) {
                                    base_delta = base_delta.saturating_sub(c);
                                }
                            if let DecisionVar::Assigned(new_dec) = patch
                                && let Some(c) = model.cost_of_assignment(
                                    req_ix,
                                    new_dec.berth_index,
                                    new_dec.start_time,
                                ) {
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

                            // Publish improvement vs baseline (may or may not beat global best).
                            let _ = context.shared_incumbent().try_update(&current, model);
                            tracing::trace!("ILS: accepted repair op {}", r.name());
                            break;
                        }
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

pub fn ils_strategy<T, R>(model: &SolverModel<T>) -> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    // Intensify packing with tighter, structured ruins and stronger repair
    IteratedLocalSearchStrategy::new()
        // Local budget and acceptance (tighter noise)
        .with_local_steps_range(900..=1600)
        .with_local_sideways(true)
        .with_local_worsening_prob(0.015)
        // Attempts per outer round (favor repair)
        .with_destroy_attempts(12)
        .with_repair_attempts(28)
        .with_shuffle_local_each_step(true)
        // Refetch: stale-based with light periodic sync
        .with_refetch_after_stale(45)
        .with_hard_refetch_every(24)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        // ------------------------- Local improvement -------------------------
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
            OrOptBlockRelocate::new(6..=10, 1.3..=1.8).with_neighbors(neighbors_same_berth.clone()),
        ))
        // Diversification
        .with_local_op(Box::new(
            RelocateSingleBestAllowWorsening::new(12..=24)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            RandomRelocateAnywhere::new(12..=24).with_neighbors(neighbors_any.clone()),
        ))
        // Hill climbers
        .with_local_op(Box::new(
            HillClimbRelocateBest::new(24..=72)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbBestSwapSameBerth::new(48..=120).with_neighbors(neighbors_same_berth.clone()),
        ))
        // ---------------------- Destroy (tight, structured) ----------------------
        .with_destroy_op(Box::new(
            RandomKRatioDestroy::new(0.26..=0.42).with_neighbors(neighbors_any.clone()),
        ))
        .with_destroy_op(Box::new(
            WorstCostDestroy::new(0.28..=0.42).with_neighbors(neighbors_direct_competitors.clone()),
        ))
        // NEW: relatedness-driven seed expansion (time and berth proximity)
        .with_destroy_op(Box::new(
            ShawRelatedDestroy::new(
                0.24..=0.36, // modest, focused ruin size
                1.6..=2.2,   // greedy bias when picking neighbors
                1.into(),    // weight_abs_start_gap
                1.into(),    // weight_abs_end_gap
                5.into(),    // penalty_berth_mismatch
            )
            .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_destroy_op(Box::new(
            TimeClusterDestroy::<T>::new(0.28..=0.42, TimeDelta::new(24.into()))
                .with_alpha(1.5..=1.75)
                .with_neighbors(neighbors_any.clone()),
        ))
        .with_destroy_op(Box::new(
            TimeWindowBandDestroy::<T>::new(0.44..=0.56, 1.4..=1.9, TimeDelta::new(16.into()))
                .with_neighbors(neighbors_any.clone()),
        ))
        // NEW: berth-centric band around a seed (kept localized)
        .with_destroy_op(Box::new(
            BerthBandDestroy::new(
                0.26..=0.40, // small-to-moderate band size
                1.4..=1.9,   // seed greediness (longer rectangles)
                1,           // half_berth_span (seed berth ±1)
            ), // omit neighbors to prefer same-berth micro-bands via operator's default
        ))
        .with_destroy_op(Box::new(
            StringBlockDestroy::new(0.32..=0.46).with_alpha(1.5..=2.0),
        ))
        // Same-berth micro-ruins to tighten dense lanes
        .with_destroy_op(Box::new(
            BerthNeighborsDestroy::new(0.28..=0.44, 1.4..=1.8)
                .with_neighbors(neighbors_same_berth.clone()),
        ))
        // NEW: cluster by processing time (help pack similar-length vessels)
        .with_destroy_op(Box::new(
            ProcessingTimeClusterDestroy::new(0.22..=0.34, 1.7..=2.0)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        // ---------------------- Repair (denser packing) ----------------------
        .with_repair_op(Box::new(KRegretInsertion::new(9..=11)))
        .with_repair_op(Box::new(RandomizedGreedyInsertion::new(1.4..=2.0)))
        .with_repair_op(Box::new(GreedyInsertion))
        // Post-repair shakers
        .with_local_op(Box::new(
            RandomizedGreedyRelocateRcl::new(18..=48, 1.5..=2.1)
                .with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeBestAcrossBerths::new(32..=96).with_neighbors(neighbors_any.clone()),
        ))
}
