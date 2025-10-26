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
            MedianHistoryEpsilon, StaleTracker, materially_better, patience_from_cooling_halving,
        },
    },
    model::solver_model::SolverModel,
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
        decisionvar::DecisionVar,
        fitness::Fitness,
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::Cost;
use std::sync::atomic::Ordering as AtomicOrdering;

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

#[derive(Debug, Clone)]
pub struct EnergyAcceptor {
    big_m: i64,
}
impl Default for EnergyAcceptor {
    fn default() -> Self {
        Self {
            big_m: 1_000_000_000,
        }
    }
}
impl EnergyAcceptor {
    pub fn with_big_m(mut self, big_m: i64) -> Self {
        self.big_m = big_m.max(1);
        self
    }
    #[inline]
    pub fn energy(&self, f: &Fitness) -> i64 {
        (f.unassigned_requests as i64)
            .saturating_mul(self.big_m)
            .saturating_add(f.cost)
    }
}
impl Acceptor for EnergyAcceptor {
    fn name(&self) -> &str {
        "EnergyAcceptor"
    }
    #[inline]
    fn accept(&self, cur: &Fitness, new: &Fitness) -> bool {
        self.energy(new) < self.energy(cur)
    }
}

/// Simulated Annealing with operator pool and online temperature control.
pub struct SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Pool (TRUE objective)
    local_pool: LocalPool<T, DefaultCostEvaluator, R>,

    // Schedule
    temperature: f64,
    init_temperature: f64,
    cooling: f64,
    min_temperature: f64,
    steps_per_epoch: usize,

    // Acceptors
    true_acceptor: LexStrictAcceptor,
    energy_acceptor: EnergyAcceptor,

    // Refetch
    refetch_after_stale_override: Option<usize>,
    hard_refetch_every: usize,
    hard_refetch_mode: HardRefetchMode,
    reheat_factor: f64,
    kick_ops_after_refetch: usize,

    // Acceptance ratio targets
    ar_target_low: f64,
    ar_target_high: f64,

    // ---------- Online temperatures ----------
    auto_temperatures: bool,
    ewma_pos_delta_e: f64,
    ewma_beta: f64,

    // ---------- Online temperature tuning (adjustable) ----------
    ar_nudge_up: f64,                // multiplier when AR < low (was 1.05)
    ar_extra_cooling: f64,           // extra cooling when AR > high (multiplier)
    online_temp_blend_geo_w: f64,    // weight for geometric schedule
    online_temp_blend_target_w: f64, // weight for target-temp estimate
}

impl<T, R> Default for SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<T, R> SimulatedAnnealingStrategy<T, R>
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
            local_pool: LocalPool::new().with_selector(selector),
            temperature: 1.6,
            init_temperature: 1.6,
            cooling: 0.9993,
            min_temperature: 1e-4,
            steps_per_epoch: 600,
            true_acceptor: LexStrictAcceptor,
            energy_acceptor: EnergyAcceptor::default(),
            refetch_after_stale_override: None,
            hard_refetch_every: 24,
            hard_refetch_mode: HardRefetchMode::IfBetter,
            reheat_factor: 0.6,
            kick_ops_after_refetch: 8,
            ar_target_low: 0.10,
            ar_target_high: 0.50,

            auto_temperatures: true,
            ewma_pos_delta_e: 0.0,
            ewma_beta: 0.90,

            // online temp defaults
            ar_nudge_up: 1.05,
            ar_extra_cooling: 1.0, // 1.0 = no extra cooling beyond .with_cooling()
            online_temp_blend_geo_w: 0.70,
            online_temp_blend_target_w: 0.30,
        }
    }

    pub fn with_local_op(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.local_pool.push(op);
        self
    }
    pub fn with_init_temp(mut self, t0: f64) -> Self {
        let t = t0.max(1e-8);
        self.temperature = t;
        self.init_temperature = t;
        self
    }
    pub fn with_cooling(mut self, factor: f64) -> Self {
        self.cooling = factor.clamp(0.9, 0.99999);
        self
    }
    pub fn with_min_temp(mut self, tmin: f64) -> Self {
        self.min_temperature = tmin.max(1e-9);
        self
    }
    pub fn with_steps_per_epoch(mut self, k: usize) -> Self {
        self.steps_per_epoch = k.max(1);
        self
    }
    pub fn with_refetch_after_stale(mut self, epochs: usize) -> Self {
        self.refetch_after_stale_override = Some(epochs.max(1));
        self
    }
    pub fn with_hard_refetch_every(mut self, epochs: usize) -> Self {
        self.hard_refetch_every = epochs;
        self
    }
    pub fn with_hard_refetch_mode(mut self, mode: HardRefetchMode) -> Self {
        self.hard_refetch_mode = mode;
        self
    }
    pub fn with_reheat_factor(mut self, f: f64) -> Self {
        self.reheat_factor = f.clamp(0.0, 1.0);
        self
    }
    pub fn with_kick_ops_after_refetch(mut self, k: usize) -> Self {
        self.kick_ops_after_refetch = k;
        self
    }
    pub fn with_big_m_for_energy(mut self, big_m: i64) -> Self {
        self.energy_acceptor = self.energy_acceptor.clone().with_big_m(big_m);
        self
    }
    pub fn with_acceptance_targets(mut self, low: f64, high: f64) -> Self {
        self.ar_target_low = low.clamp(0.0, 1.0);
        self.ar_target_high = high.clamp(self.ar_target_low, 1.0);
        self
    }
    pub fn with_auto_temperatures(mut self, yes: bool) -> Self {
        self.auto_temperatures = yes;
        self
    }
    pub fn with_ewma_beta(mut self, beta: f64) -> Self {
        self.ewma_beta = beta.clamp(0.5, 0.99);
        self
    }

    pub fn with_ar_nudge_up(mut self, factor: f64) -> Self {
        self.ar_nudge_up = factor.clamp(1.0, 1.25);
        self
    }
    pub fn with_ar_extra_cooling(mut self, factor: f64) -> Self {
        // multiply geometric cooling by this when AR > high; 1.0 = none
        self.ar_extra_cooling = factor.clamp(0.90, 1.0);
        self
    }
    pub fn with_online_temp_blend(mut self, geo_w: f64, target_w: f64) -> Self {
        let s = (geo_w + target_w).max(1e-9);
        self.online_temp_blend_geo_w = (geo_w / s).clamp(0.0, 1.0);
        self.online_temp_blend_target_w = (target_w / s).clamp(0.0, 1.0);
        self
    }

    #[inline]
    fn metropolis<RNG: rand::Rng>(rng: &mut RNG, delta_energy: i64, temp: f64) -> bool {
        if delta_energy <= 0 {
            return true;
        }
        let p = (-(delta_energy as f64) / temp).exp();
        rng.random::<f64>() < p
    }
    #[inline]
    fn epoch_due(&self, epoch: usize) -> bool {
        self.hard_refetch_every > 0 && epoch > 0 && epoch.is_multiple_of(self.hard_refetch_every)
    }
    #[inline]
    fn reheat(&self, temp: &mut f64) {
        if self.reheat_factor > 0.0 {
            let target = (self.init_temperature * self.reheat_factor).max(self.min_temperature);
            *temp = temp.max(target);
        }
    }
    #[inline]
    fn acceptance_ratio(acc: usize, tried: usize) -> f64 {
        if tried == 0 {
            0.0
        } else {
            (acc as f64) / (tried as f64)
        }
    }

    fn kick_with_local_pool<'p>(
        &mut self,
        k: usize,
        model: &SolverModel<'p, T>,
        eval: &DefaultCostEvaluator,
        dv_buf: &mut [DecisionVar<T>],
        state: &mut SolverState<'p, T>,
        rng: &mut R,
    ) {
        if k == 0 || self.local_pool.is_empty() {
            return;
        }
        for _ in 0..k {
            let mut pc = PlanningContext::new(model, &*state, eval, dv_buf);
            let mut prop = self.local_pool.apply(&mut pc, rng, Some(1.0));
            let Some(mut plan) = prop.take_plan() else {
                prop.reject();
                continue;
            };

            set_plan_delta_via_eval(model, eval, state.decision_variables(), &mut plan);

            let mut tmp = state.clone();
            tmp.apply_plan(plan);

            let cur_fit = state.fitness();
            let tmp_fit = tmp.fitness();
            let better = self.true_acceptor.accept(cur_fit, tmp_fit);
            let sideways = tmp_fit == cur_fit;

            if better || sideways {
                let drop = cur_fit.cost.saturating_sub(tmp_fit.cost).max(0);
                *state = tmp;
                prop.accept(drop);
            } else {
                prop.reject();
            }
        }
    }
}

/* ----------------------------- free helper ----------------------------- */

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

/* ----------------------------- strategy impl ----------------------------- */

impl<T, R> SearchStrategy<T, R> for SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng + Send + Sync,
{
    fn name(&self) -> &str {
        "Simulated Annealing"
    }

    #[tracing::instrument(level = "debug", name = "SA Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();
        if self.local_pool.is_empty() {
            tracing::warn!("SA: no local operators configured");
            return;
        }

        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best: SolverState<'p, T> = current.clone();

        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut eps_src = MedianHistoryEpsilon::new(32, 1);
        let mut stale = StaleTracker::new(*current.fitness(), 32);

        let patience_epochs = self
            .refetch_after_stale_override
            .unwrap_or_else(|| patience_from_cooling_halving(self.cooling));

        let eval = DefaultCostEvaluator;
        let mut temp = self.temperature;
        let mut epoch = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            epoch = epoch.saturating_add(1);

            // Periodic refetch gate
            if self.epoch_due(epoch) {
                let inc = context.shared_incumbent().peek();
                let do_fetch = match self.hard_refetch_mode {
                    HardRefetchMode::IfBetter => self.true_acceptor.accept(current.fitness(), &inc),
                    HardRefetchMode::Always => true,
                };
                if do_fetch {
                    tracing::debug!(
                        "SA: periodic refetch @epoch {} (curr {}, inc {})",
                        epoch,
                        current.fitness(),
                        inc
                    );
                    let mut snap = context.shared_incumbent().snapshot();

                    if self.kick_ops_after_refetch > 0 && !self.local_pool.is_empty() {
                        self.kick_with_local_pool(
                            self.kick_ops_after_refetch.min(self.local_pool.len()),
                            model,
                            &eval,
                            dv_buf.as_mut_slice(),
                            &mut snap,
                            context.rng(),
                        );
                    }

                    if self.true_acceptor.accept(best.fitness(), snap.fitness()) {
                        best = snap.clone();
                    }
                    current = snap;

                    self.reheat(&mut temp);
                    stale.arm_cooldown_until_next_improvement();
                    tracing::debug!("SA: periodic refetch → reheat to T={}", temp);
                }
            }

            // One epoch
            let mut accepted = 0usize;
            let mut tried = 0usize;

            for _ in 0..self.steps_per_epoch {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut pc = PlanningContext::new(model, &current, &eval, dv_buf.as_mut_slice());
                let mut prop = self.local_pool.apply(
                    &mut pc,
                    context.rng(),
                    Some(temp.max(self.min_temperature)),
                );

                tried = tried.saturating_add(1);

                if let Some(mut plan) = prop.take_plan() {
                    // quick TRUE delta via partial recompute
                    use crate::model::index::RequestIndex;
                    use crate::state::decisionvar::DecisionVar;
                    use std::collections::HashMap;

                    let mut base_delta: Cost = 0.into();
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

                    let mut next = current.clone();
                    next.apply_plan(plan);

                    let e0 = self.energy_acceptor.energy(current.fitness());
                    let e1 = self.energy_acceptor.energy(next.fitness());
                    let delta_e = e1 - e0;

                    if delta_e > 0 {
                        self.ewma_pos_delta_e = self.ewma_beta * self.ewma_pos_delta_e
                            + (1.0 - self.ewma_beta) * (delta_e as f64);
                    }

                    let t_eff = temp.max(self.min_temperature);
                    let accept = Self::metropolis(context.rng(), delta_e, t_eff);

                    if accept {
                        current = next;
                        accepted = accepted.saturating_add(1);

                        let drop = best
                            .fitness()
                            .cost
                            .saturating_sub(current.fitness().cost)
                            .max(0);
                        prop.accept(drop);

                        if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                            let drop_best = best
                                .fitness()
                                .cost
                                .saturating_sub(current.fitness().cost)
                                .max(0);
                            eps_src.record(drop_best);
                            best = current.clone();
                            let _ = context.shared_incumbent().try_update(&best, model);
                        }
                    } else {
                        prop.reject();
                    }
                } else {
                    prop.reject();
                }
            }

            // Stale bookkeeping
            if let Some(delta) = stale.on_round_end(*best.fitness()) {
                eps_src.record(delta);
            }

            // Stale refetch (ε-guarded)
            if stale.is_stale(patience_epochs) {
                let inc = context.shared_incumbent().peek();
                let eps = eps_src.epsilon();
                let material = materially_better(current.fitness(), &inc, eps);

                let allowed = match self.hard_refetch_mode {
                    HardRefetchMode::Always => true,
                    HardRefetchMode::IfBetter => material,
                };

                if allowed && material {
                    tracing::debug!(
                        "SA: stale refetch @epoch {} (pat={}, eps={}) current={} incumbent={}",
                        epoch,
                        patience_epochs,
                        eps,
                        current.fitness(),
                        inc
                    );
                    let mut snap = context.shared_incumbent().snapshot();

                    if self.kick_ops_after_refetch > 0 && !self.local_pool.is_empty() {
                        self.kick_with_local_pool(
                            self.kick_ops_after_refetch.min(self.local_pool.len()),
                            model,
                            &eval,
                            dv_buf.as_mut_slice(),
                            &mut snap,
                            context.rng(),
                        );
                    }

                    if self.true_acceptor.accept(best.fitness(), snap.fitness()) {
                        best = snap.clone();
                    }
                    current = snap;

                    self.reheat(&mut temp);
                    stale.arm_cooldown_until_next_improvement();
                    tracing::debug!("SA: stale refetch → reheat to T={}", temp);
                }
            }

            // Temperature update: geometric + tunable AR nudges + tunable online blend
            let ar = Self::acceptance_ratio(accepted, tried);
            let mut next_temp = temp * self.cooling;

            if ar < self.ar_target_low {
                next_temp = (temp * self.ar_nudge_up).max(next_temp);
            }
            if ar > self.ar_target_high {
                next_temp *= self.ar_extra_cooling; // extra cooling multiplier (≤1.0)
            }

            if self.auto_temperatures && self.ewma_pos_delta_e > 0.0 {
                let p_tgt =
                    ((self.ar_target_low + self.ar_target_high) * 0.5).clamp(1e-6, 0.999999);
                let t_from_target = self.ewma_pos_delta_e / (-p_tgt.ln());
                let t_target = t_from_target.max(self.min_temperature);

                next_temp = self.online_temp_blend_geo_w * next_temp
                    + self.online_temp_blend_target_w * t_target;
            }

            temp = next_temp.max(self.min_temperature);
        }

        let _ = context.shared_incumbent().try_update(&best, model);
    }
}

/* ----------------------------- preset ----------------------------- */

pub fn sa_strategy<T, R>(model: &SolverModel<T>) -> SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    SimulatedAnnealingStrategy::new()
        .with_init_temp(35.0)
        .with_cooling(0.9997)
        .with_min_temp(1e-4)
        .with_steps_per_epoch(1200)
        .with_hard_refetch_every(80)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_refetch_after_stale(60)
        .with_reheat_factor(0.85)
        .with_kick_ops_after_refetch(18)
        .with_big_m_for_energy(900_000_000)
        .with_acceptance_targets(0.22, 0.70)
        .with_auto_temperatures(true)
        .with_ewma_beta(0.90)
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
            CascadeRelocateK::new(3..=5, 6..=10, 10..=20)
                .with_neighbors(neighbors_any)
                .with_insert_policy(CascadeInsertPolicy::Rcl {
                    alpha_range: 1.4..=2.0,
                })
                .allow_step_worsening(30),
        ))
}
