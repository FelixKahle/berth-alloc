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

use std::{ops::RangeInclusive, sync::atomic::Ordering as AtomicOrdering};

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor, RepairAcceptor},
        neighbors,
        operators::{DestroyPool, LocalPool, RepairPool, SoftmaxSelector},
        search::{SearchContext, SearchStrategy},
        strategy_support::{
            MedianHistoryEpsilon, StaleTracker, materially_better, patience_from_exploration_budget,
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
                CascadeInsertPolicy, CascadeRelocateK, CrossExchangeAcrossBerths,
                HillClimbRelocateBest, OrOptBlockRelocate, RelocateSingleBest,
                ShiftEarlierOnSameBerth, SwapPairSameBerth,
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

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

pub struct IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Pools
    local_pool: LocalPool<T, DefaultCostEvaluator, R>,
    destroy_pool: DestroyPool<T, DefaultCostEvaluator, R>,
    repair_pool: RepairPool<T, DefaultCostEvaluator, R>,

    // Acceptors
    local_acceptor: LexStrictAcceptor,
    repair_acceptor: RepairAcceptor,

    // Local improvement budget
    max_local_steps: usize,
    local_steps_range: Option<RangeInclusive<usize>>,

    // Local acceptance tweaks
    allow_sideways_in_local: bool,
    accept_worsening_local_with_prob: Option<f64>,

    // Ruin/repair caps per round (static caps if set)
    max_destroy_attempts_per_round: Option<usize>,
    max_repair_attempts_per_round: Option<usize>,

    // Legacy toggle
    shuffle_local_each_step: bool,

    // Staleness → refetch
    stale_min_rounds_override: Option<usize>,
    hard_refetch_every: usize,
    hard_refetch_mode: HardRefetchMode,
    kick_ops_after_refetch: usize,

    // ---------- Online perturbation sizing ----------
    online_perturbation: bool,
    destroy_bounds: Option<(usize, usize)>,
    repair_bounds: Option<(usize, usize)>,
    cur_destroy_cap: Option<usize>,
    cur_repair_cap: Option<usize>,

    online_ewma_beta: f64,           // smoothing for RR success
    online_sr_low: f64,              // target success-band (low)
    online_sr_high: f64,             // target success-band (high)
    online_cap_update_period: usize, // update caps every N rounds
    online_cap_step_max: usize,      // clamp Δcaps per update
    online_bias_explore_on_stagnation: bool,
}

/* ----------------------------- public API ----------------------------- */

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
        let selector = SoftmaxSelector::default()
            .with_base_temp(1.0)
            .with_min_p(1e-6)
            .with_power(1.0);

        Self {
            local_pool: LocalPool::new().with_selector(selector.clone()),
            destroy_pool: DestroyPool::new().with_selector(selector.clone()),
            repair_pool: RepairPool::new().with_selector(selector),
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

            // online perturbation
            online_perturbation: false,
            destroy_bounds: None,
            repair_bounds: None,
            cur_destroy_cap: None,
            cur_repair_cap: None,

            // online tuning defaults
            online_ewma_beta: 0.30,
            online_sr_low: 0.18,
            online_sr_high: 0.45,
            online_cap_update_period: 5,
            online_cap_step_max: 2,
            online_bias_explore_on_stagnation: true,
        }
    }

    // operator registration
    pub fn with_destroy_op(
        mut self,
        op: Box<dyn DestroyOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.destroy_pool.push(op);
        self
    }
    pub fn with_repair_op(
        mut self,
        op: Box<dyn RepairOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.repair_pool.push(op);
        self
    }
    pub fn with_local_op(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.local_pool.push(op);
        self
    }

    // knobs
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

    // online perturbation
    pub fn with_online_perturbation(mut self, yes: bool) -> Self {
        self.online_perturbation = yes;
        self
    }
    pub fn with_destroy_cap_bounds(mut self, min_cap: usize, max_cap: usize) -> Self {
        self.destroy_bounds = Some((min_cap.max(1), max_cap.max(min_cap.max(1))));
        self
    }
    pub fn with_repair_cap_bounds(mut self, min_cap: usize, max_cap: usize) -> Self {
        self.repair_bounds = Some((min_cap.max(1), max_cap.max(min_cap.max(1))));
        self
    }

    pub fn with_online_ewma_beta(mut self, beta: f64) -> Self {
        self.online_ewma_beta = beta.clamp(0.05, 0.95);
        self
    }
    pub fn with_online_success_band(mut self, low: f64, high: f64) -> Self {
        let lo = low.clamp(0.0, 1.0);
        let hi = high.clamp(lo, 1.0);
        self.online_sr_low = lo;
        self.online_sr_high = hi;
        self
    }
    pub fn with_online_cap_update_period(mut self, period: usize) -> Self {
        self.online_cap_update_period = period.max(1);
        self
    }
    pub fn with_online_cap_step_max(mut self, step: usize) -> Self {
        self.online_cap_step_max = step.max(1);
        self
    }
    pub fn with_online_bias_explore_on_stagnation(mut self, yes: bool) -> Self {
        self.online_bias_explore_on_stagnation = yes;
        self
    }
}

/* ----------------------------- helpers ----------------------------- */

#[inline]
fn periodic_refetch_due(period: usize, outer_rounds: usize) -> bool {
    period > 0 && outer_rounds > 0 && outer_rounds.is_multiple_of(period)
}

fn set_plan_delta_via_eval<'m, T: SolveNumeric>(
    model: &SolverModel<'m, T>,
    eval: &DefaultCostEvaluator,
    current_vars: &[DecisionVar<T>],
    plan: &mut crate::state::plan::Plan<'m, T>,
) {
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

#[inline]
fn steps_budget(
    range: &Option<RangeInclusive<usize>>,
    def_max: usize,
    rng: &mut impl rand::Rng,
) -> usize {
    if let Some(r) = range {
        let lo = *r.start();
        let hi = *r.end();
        if lo == hi {
            lo
        } else {
            rng.random_range(lo..=hi)
        }
    } else {
        def_max
    }
}

#[allow(clippy::too_many_arguments)]
fn try_one_local_step<'p, T, R>(
    local_pool: &mut LocalPool<T, DefaultCostEvaluator, R>,
    local_acceptor: &LexStrictAcceptor,
    allow_sideways_in_local: bool,
    accept_worsening_local_with_prob: Option<f64>,
    model: &SolverModel<'p, T>,
    eval: &DefaultCostEvaluator,
    dv_buf: &mut [DecisionVar<T>],
    current: &mut SolverState<'p, T>,
    rng: &mut R,
) -> (bool, i64)
where
    T: SolveNumeric,
    R: rand::Rng,
{
    let mut pc = PlanningContext::new(model, &*current, eval, dv_buf);
    let mut prop = local_pool.apply(&mut pc, rng, None);

    let Some(mut plan) = prop.take_plan() else {
        prop.reject();
        return (false, 0);
    };

    set_plan_delta_via_eval(model, eval, current.decision_variables(), &mut plan);

    let mut cand = current.clone();
    cand.apply_plan(plan);

    let cur_fit = current.fitness();
    let new_fit = cand.fitness();
    let better = local_acceptor.accept(cur_fit, new_fit);
    let sideways = allow_sideways_in_local && (new_fit == cur_fit);
    let worse_random = if !better && !sideways {
        if let Some(p) = accept_worsening_local_with_prob {
            rng.random::<f64>() < p
        } else {
            false
        }
    } else {
        false
    };

    if better || sideways || worse_random {
        let strict_drop = cur_fit.cost.saturating_sub(new_fit.cost).max(0);
        *current = cand;
        prop.accept(strict_drop);
        (true, strict_drop)
    } else {
        prop.reject();
        (false, 0)
    }
}

fn run_destroy_phase<'p, T, R>(
    pool: &mut DestroyPool<T, DefaultCostEvaluator, R>,
    attempts: usize,
    model: &SolverModel<'p, T>,
    eval: &DefaultCostEvaluator,
    dv_buf: &mut [DecisionVar<T>],
    current: &mut SolverState<'p, T>,
    rng: &mut R,
) -> bool
where
    T: SolveNumeric,
    R: rand::Rng,
{
    for _ in 0..attempts {
        let mut pc = PlanningContext::new(model, &*current, eval, dv_buf);
        let mut prop = pool.apply(&mut pc, rng, None);
        if let Some(mut plan) = prop.take_plan() {
            set_plan_delta_via_eval(model, eval, current.decision_variables(), &mut plan);
            current.apply_plan(plan);
            prop.accept(0);
            return true;
        } else {
            prop.reject();
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn run_repair_phase<'p, T, R>(
    pool: &mut RepairPool<T, DefaultCostEvaluator, R>,
    acceptor: &RepairAcceptor,
    attempts: usize,
    model: &SolverModel<'p, T>,
    eval: &DefaultCostEvaluator,
    dv_buf: &mut [DecisionVar<T>],
    baseline: &SolverState<'p, T>,
    rng: &mut R,
) -> (bool, SolverState<'p, T>, i64)
where
    T: SolveNumeric,
    R: rand::Rng,
{
    for _ in 0..attempts {
        let mut tmp = baseline.clone();
        let mut pc = PlanningContext::new(model, &tmp, eval, dv_buf);
        let mut prop = pool.apply(&mut pc, rng, None);

        if let Some(mut plan) = prop.take_plan() {
            set_plan_delta_via_eval(model, eval, tmp.decision_variables(), &mut plan);
            tmp.apply_plan(plan);

            if acceptor.accept(baseline.fitness(), tmp.fitness()) {
                let drop = baseline
                    .fitness()
                    .cost
                    .saturating_sub(tmp.fitness().cost)
                    .max(0);
                prop.accept(drop);
                return (true, tmp, drop);
            } else {
                prop.reject();
            }
        } else {
            prop.reject();
        }
    }
    (false, baseline.clone(), 0)
}

/* ----------------------------- main strategy ----------------------------- */

impl<T, R> SearchStrategy<T, R> for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng + Send + Sync,
{
    fn name(&self) -> &str {
        "Iterated Local Search"
    }

    #[tracing::instrument(level = "debug", name = "ILS Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let model = context.model();
        let stop = context.stop();

        if self.local_pool.is_empty()
            && (self.destroy_pool.is_empty() || self.repair_pool.is_empty())
        {
            tracing::warn!(
                "ILS: no operators configured (local_pool={}, destroy_pool={}, repair_pool={})",
                self.local_pool.len(),
                self.destroy_pool.len(),
                self.repair_pool.len()
            );
            return;
        }

        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        debug_assert_eq!(
            current.decision_variables().len(),
            model.flexible_requests_len()
        );

        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut stale = StaleTracker::new(*current.fitness(), 32);
        let mut eps_src = MedianHistoryEpsilon::new(32, 1);

        // seed caps
        let pool_destroy_default = self.destroy_pool.len().max(1);
        let pool_repair_default = self.repair_pool.len().max(1);

        let mut destroy_cap = self
            .cur_destroy_cap
            .or(self.max_destroy_attempts_per_round)
            .unwrap_or(pool_destroy_default);
        let mut repair_cap = self
            .cur_repair_cap
            .or(self.max_repair_attempts_per_round)
            .unwrap_or(pool_repair_default);

        if let Some((mn, mx)) = self.destroy_bounds {
            destroy_cap = destroy_cap.clamp(mn, mx);
        }
        if let Some((mn, mx)) = self.repair_bounds {
            repair_cap = repair_cap.clamp(mn, mx);
        }

        // patience
        let batches_per_round = 1 + destroy_cap + repair_cap;
        let inner_steps_mean = self
            .local_steps_range
            .as_ref()
            .map(|r| ((*r.start() + *r.end()) / 2).max(1))
            .unwrap_or(self.max_local_steps.max(1));

        let derived_patience = patience_from_exploration_budget(
            batches_per_round,
            inner_steps_mean,
            self.shuffle_local_each_step,
        );
        let patience_s = self.stale_min_rounds_override.unwrap_or(derived_patience);

        let eval = DefaultCostEvaluator;
        let mut outer_rounds = 0usize;

        // online smoothing state
        let ew_beta = self.online_ewma_beta;
        let mut rr_succ_ewma: f64 = 0.0;
        let mut rr_try_ewma: f64 = 0.0;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            outer_rounds = outer_rounds.saturating_add(1);

            // ---- Phase A: Local
            let budget = steps_budget(&self.local_steps_range, self.max_local_steps, context.rng());
            let mut improved_in_round = false;

            for _ in 0..budget {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let (accepted, strict_drop) = try_one_local_step(
                    &mut self.local_pool,
                    &self.local_acceptor,
                    self.allow_sideways_in_local,
                    self.accept_worsening_local_with_prob,
                    model,
                    &eval,
                    dv_buf.as_mut_slice(),
                    &mut current,
                    context.rng(),
                );

                if accepted {
                    if strict_drop > 0 {
                        eps_src.record(strict_drop);
                        improved_in_round = true;
                    }
                    let _ = context.shared_incumbent().try_update(&current, model);
                } else {
                    break;
                }
            }

            // ---- Phase B/C: RR if local didn’t move
            let mut had_repair = false;
            let mut repair_drop = 0i64;

            if !improved_in_round {
                if self.destroy_pool.is_empty() || self.repair_pool.is_empty() {
                    tracing::debug!("ILS: no perturbation operators configured; stopping.");
                    break 'outer;
                }

                let baseline = current.clone();
                let destroyed = run_destroy_phase(
                    &mut self.destroy_pool,
                    destroy_cap,
                    model,
                    &eval,
                    dv_buf.as_mut_slice(),
                    &mut current,
                    context.rng(),
                );
                if !destroyed {
                    tracing::debug!("ILS: no destroy operator produced a plan; stopping.");
                    break 'outer;
                }

                let (repaired, repaired_state, drop_vs_base) = run_repair_phase(
                    &mut self.repair_pool,
                    &self.repair_acceptor,
                    repair_cap,
                    model,
                    &eval,
                    dv_buf.as_mut_slice(),
                    &current,
                    context.rng(),
                );
                rr_try_ewma = ew_beta * rr_try_ewma + (1.0 - ew_beta) * 1.0;

                if repaired {
                    current = repaired_state;
                    let _ = context.shared_incumbent().try_update(&current, model);
                    had_repair = true;
                    repair_drop = drop_vs_base;
                    let succ = if drop_vs_base > 0 { 1.0 } else { 0.0 };
                    rr_succ_ewma = ew_beta * rr_succ_ewma + (1.0 - ew_beta) * succ;
                    if drop_vs_base > 0 {
                        eps_src.record(drop_vs_base);
                    }
                } else {
                    current = baseline;
                    rr_succ_ewma = ew_beta * rr_succ_ewma + (1.0 - ew_beta) * 0.0;
                    tracing::debug!("ILS: repair failed to beat baseline.");
                }
            }

            // end-of-round stale book-keeping
            if let Some(delta) = stale.on_round_end(*current.fitness()) {
                eps_src.record(delta);
            }

            // stale refetch (ε-guarded) plus cadence
            if stale.is_stale(patience_s) {
                let inc = context.shared_incumbent().peek();
                let mat = materially_better(current.fitness(), &inc, eps_src.epsilon());
                let periodic_ok = periodic_refetch_due(self.hard_refetch_every, outer_rounds);

                if mat && (periodic_ok || self.hard_refetch_every == 0) {
                    tracing::debug!(
                        "ILS: stale refetch (round={}, patience={}, eps={}) current={} incumbent={}",
                        outer_rounds,
                        patience_s,
                        eps_src.epsilon(),
                        current.fitness(),
                        inc
                    );
                    let mut snap = context.shared_incumbent().snapshot();

                    if self.kick_ops_after_refetch > 0 && !self.local_pool.is_empty() {
                        let k = self.kick_ops_after_refetch.min(self.local_pool.len());
                        kick_with_local_pool_internal(
                            self,
                            k,
                            model,
                            &eval,
                            dv_buf.as_mut_slice(),
                            &mut snap,
                            context.rng(),
                        );
                    }
                    current = snap;
                    stale.arm_cooldown_until_next_improvement();
                }
            }

            // ---- Online caps update (adjustable pacing) ----
            if self.online_perturbation
                && outer_rounds.is_multiple_of(self.online_cap_update_period)
            {
                let (dmin, dmax) = self.destroy_bounds.unwrap_or((1, usize::MAX));
                let (rmin, rmax) = self.repair_bounds.unwrap_or((1, usize::MAX));

                let sr = if rr_try_ewma > 1e-9 {
                    (rr_succ_ewma / rr_try_ewma).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let mut d = destroy_cap as isize;
                let mut r = repair_cap as isize;

                // primary controller by success band
                if sr < self.online_sr_low {
                    d += 1;
                    r += 1;
                } else if sr > self.online_sr_high {
                    d -= 1;
                    r -= 1;
                }

                // secondary nudges from last outcome
                if had_repair && repair_drop == 0 {
                    r += 1;
                } else if improved_in_round {
                    d -= 1;
                } else if !had_repair {
                    r += 1;
                }

                // optional exploration bias on stagnation
                if self.online_bias_explore_on_stagnation && eps_src.epsilon() <= 0 {
                    d += 1;
                }

                // clamp step & bounds
                let clamp_step = |cur: isize, prev: isize| -> isize {
                    (cur - prev).clamp(
                        -(self.online_cap_step_max as isize),
                        self.online_cap_step_max as isize,
                    ) + prev
                };
                d = clamp_step(d, destroy_cap as isize).clamp(dmin as isize, dmax as isize);
                r = clamp_step(r, repair_cap as isize).clamp(rmin as isize, rmax as isize);

                destroy_cap = d as usize;
                repair_cap = r as usize;
                self.cur_destroy_cap = Some(destroy_cap);
                self.cur_repair_cap = Some(repair_cap);

                tracing::trace!(
                    "ILS: caps update @round {} → destroy={}, repair={}, sr={:.3}",
                    outer_rounds,
                    destroy_cap,
                    repair_cap,
                    sr
                );
            }
        }

        let _ = context.shared_incumbent().try_update(&current, model);
    }
}

/* ----------------------------- internal kick ----------------------------- */

fn kick_with_local_pool_internal<'p, T, R>(
    strat: &mut IteratedLocalSearchStrategy<T, R>,
    k: usize,
    model: &SolverModel<'p, T>,
    eval: &DefaultCostEvaluator,
    dv_buf: &mut [DecisionVar<T>],
    state: &mut SolverState<'p, T>,
    rng: &mut R,
) where
    T: SolveNumeric,
    R: rand::Rng,
{
    if k == 0 || strat.local_pool.is_empty() {
        return;
    }
    for _ in 0..k {
        let mut pc = PlanningContext::new(model, &*state, eval, dv_buf);
        let mut prop = strat.local_pool.apply(&mut pc, rng, None);
        let Some(mut plan) = prop.take_plan() else {
            prop.reject();
            continue;
        };

        set_plan_delta_via_eval(model, eval, state.decision_variables(), &mut plan);

        let mut tmp = state.clone();
        tmp.apply_plan(plan);

        let cur_fit = state.fitness();
        let new_fit = tmp.fitness();
        let better = strat.local_acceptor.accept(cur_fit, new_fit);
        let sideways = strat.allow_sideways_in_local && (new_fit == cur_fit);
        let worse_random = if !better && !sideways {
            if let Some(p) = strat.accept_worsening_local_with_prob {
                rng.random::<f64>() < p
            } else {
                false
            }
        } else {
            false
        };

        if better || sideways || worse_random {
            let drop = cur_fit.cost.saturating_sub(new_fit.cost).max(0);
            *state = tmp;
            prop.accept(drop);
        } else {
            prop.reject();
        }
    }
}

/* ----------------------------- preset ----------------------------- */

pub fn ils_strategy<T, R>(model: &SolverModel<T>) -> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    IteratedLocalSearchStrategy::new()
        // Local
        .with_local_steps_range(1200..=2200)
        .with_local_sideways(true)
        .with_local_worsening_prob(0.0)
        // Ruin/Repair
        .with_destroy_attempts(12)
        .with_repair_attempts(28)
        .with_shuffle_local_each_step(true)
        // Refetch
        .with_refetch_after_stale(40)
        .with_hard_refetch_every(14)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_kick_ops_after_refetch(8)
        // Online perturbation
        .with_online_perturbation(true)
        .with_destroy_cap_bounds(4, 32)
        .with_repair_cap_bounds(8, 40)
        // ------------------------- Local ops -------------------------
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
        .with_local_op(Box::new(
            CascadeRelocateK::new(3..=4, 8..=12, 12..=24)
                .with_neighbors(neighbors_direct_competitors.clone())
                .with_insert_policy(CascadeInsertPolicy::BestEarliest),
        ))
        // ---------------------- Destroy ----------------------
        .with_destroy_op(Box::new(
            RandomKRatioDestroy::new(0.32..=0.58).with_neighbors(neighbors_any.clone()),
        ))
        .with_destroy_op(Box::new(
            WorstCostDestroy::new(0.30..=0.48).with_neighbors(neighbors_direct_competitors.clone()),
        ))
        .with_destroy_op(Box::new(
            ShawRelatedDestroy::new(0.28..=0.40, 1.6..=2.2, 1.into(), 1.into(), 5.into())
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
        // ---------------------- Repair ----------------------
        .with_repair_op(Box::new(KRegretInsertion::new(8..=11)))
        .with_repair_op(Box::new(GreedyInsertion))
}
