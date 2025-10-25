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

#![allow(clippy::needless_return)]

use std::sync::atomic::Ordering as AtomicOrdering;

use berth_alloc_core::prelude::Cost;

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor},
        neighbors,
        search::{SearchContext, SearchStrategy},
        strategy_support::{
            ApplyOnceByIndex, MedianHistoryEpsilon, StaleTracker, deterministic_kick,
            materially_better, patience_from_cooling_halving,
        },
    },
    model::solver_model::SolverModel,
    search::{
        operator::LocalMoveOperator,
        operator_library::local::{
            CrossExchangeAcrossBerths, CrossExchangeBestAcrossBerths, HillClimbBestSwapSameBerth,
            HillClimbRelocateBest, OrOptBlockRelocate, RandomRelocateAnywhere,
            RandomizedGreedyRelocateRcl, RelocateSingleBest, RelocateSingleBestAllowWorsening,
            ShiftEarlierOnSameBerth, SwapPairSameBerth,
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

/// Energy-ordered acceptor (deterministic compare) used for SA "energy".
/// Energy = BIG_M * unassigned + cost (i64), so lex-like prioritization.
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

/// Simulated Annealing with:
/// - Standard Metropolis acceptance on an “energy” (BIG-M + cost)
/// - Geometric cooling + AR-guided nudges (classic practice)
/// - Data-driven staleness/epsilon via helpers, incumbent refetch + reheat
/// - Bandit EMA for operator choice (keeps all operators live)
pub struct SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Operators evaluated on the TRUE/Base objective
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,

    // Temperature schedule (geometric cooling)
    temperature: f64,
    init_temperature: f64,
    cooling: f64,
    min_temperature: f64,
    steps_per_epoch: usize,

    // Acceptors
    true_acceptor: LexStrictAcceptor,
    energy_acceptor: EnergyAcceptor,

    // Refetch policy
    refetch_after_stale_override: Option<usize>, // epochs; if None, derived from cooling
    hard_refetch_every: usize,
    hard_refetch_mode: HardRefetchMode,
    reheat_factor: f64, // 0..1; fraction of init T to reheat to on refetch
    kick_ops_after_refetch: usize, // deterministic kick count after refetch

    // Bandit scheduling (EMA)
    op_weights: Vec<f64>,
    op_ema_alpha: f64,
    op_min_weight: f64,

    // Acceptance ratio guidance (nudges, not replacement of geometric cooling)
    ar_target_low: f64,
    ar_target_high: f64,
}

// -------------- Builder --------------
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
        Self {
            local_ops: Vec::new(),
            // schedule
            temperature: 1.6,
            init_temperature: 1.6,
            cooling: 0.9993,
            min_temperature: 1e-4,
            steps_per_epoch: 600,
            // acceptors
            true_acceptor: LexStrictAcceptor,
            energy_acceptor: EnergyAcceptor::default(),
            // refetch
            refetch_after_stale_override: None,
            hard_refetch_every: 24,
            hard_refetch_mode: HardRefetchMode::IfBetter,
            reheat_factor: 0.6,
            kick_ops_after_refetch: 8,
            // bandit
            op_weights: Vec::new(),
            op_ema_alpha: 0.25,
            op_min_weight: 0.05,
            // AR guidance
            ar_target_low: 0.10,
            ar_target_high: 0.50,
        }
    }

    pub fn with_local_op(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
    ) -> Self {
        self.local_ops.push(op);
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
    pub fn with_op_ema_alpha(mut self, a: f64) -> Self {
        self.op_ema_alpha = a.clamp(0.01, 0.9);
        self
    }
    pub fn with_op_min_weight(mut self, w: f64) -> Self {
        self.op_min_weight = w.clamp(0.0, 1.0);
        self
    }
    pub fn with_acceptance_targets(mut self, low: f64, high: f64) -> Self {
        self.ar_target_low = low.clamp(0.0, 1.0);
        self.ar_target_high = high.clamp(self.ar_target_low, 1.0);
        self
    }

    // --------- internals ---------
    #[inline]
    fn metropolis<RNG: rand::Rng>(&self, rng: &mut RNG, delta_energy: i64, temp: f64) -> bool {
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

    #[inline]
    fn pick_op_index(&self, rng: &mut R, weights: &[f64]) -> usize {
        let s: f64 = weights.iter().copied().sum();
        if !s.is_finite() || s <= 0.0 {
            return rng.random_range(0..weights.len());
        }
        let mut r = rng.random::<f64>() * s;
        for (i, w) in weights.iter().enumerate() {
            r -= *w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    #[inline]
    fn update_weight(w: &mut f64, alpha: f64, reward: f64, min_w: f64) {
        let base = if w.is_finite() && *w > 0.0 { *w } else { 1.0 };
        let nw = (1.0 - alpha) * base + alpha * reward.max(0.0);
        *w = nw.max(min_w);
    }
}

// Adapter to run a deterministic kick by applying the first k local ops once.
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
            // Compute delta with evaluator fitness before/after
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

impl<T, R> SearchStrategy<T, R> for SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "Simulated Annealing"
    }

    #[tracing::instrument(level = "debug", name = "SA Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();
        if self.local_ops.is_empty() {
            tracing::warn!("SA: no local operators configured");
            return;
        }

        // Working states
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best: SolverState<'p, T> = current.clone();

        // Scratch buffer for PlanningContext
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Data-driven epsilon & stale tracking via helpers
        let mut eps_src = MedianHistoryEpsilon::new(/*history_cap*/ 32, /*min_eps*/ 1);
        let mut stale = StaleTracker::new(*current.fitness(), /*history_cap*/ 32);

        // Derive patience if not overridden: epochs to halve temperature
        let patience_epochs = self
            .refetch_after_stale_override
            .unwrap_or_else(|| patience_from_cooling_halving(self.cooling));

        // Bandit weights: lazy init to 1.0 each
        if self.op_weights.len() != self.local_ops.len() {
            self.op_weights.clear();
            self.op_weights.resize(self.local_ops.len(), 1.0);
        }

        let eval = DefaultCostEvaluator;
        let mut temp = self.temperature;
        let mut epoch = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            epoch = epoch.saturating_add(1);

            // -------- Periodic refetch (sync) gate; optional reheat --------
            if self.hard_refetch_every > 0 && self.epoch_due(epoch) {
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

                    // Deterministic kick to avoid synchronization lock
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
                        let _applied = deterministic_kick(&mut adapter, k);
                    }

                    if self.true_acceptor.accept(best.fitness(), snap.fitness()) {
                        best = snap.clone();
                    }
                    current = snap;

                    // Reheat and cooldown for staleness
                    self.reheat(&mut temp);
                    stale.arm_cooldown_until_next_improvement();
                    tracing::debug!("SA: periodic refetch → reheat to T={}", temp);
                }
            }

            // -------- One SA epoch: Metropolis steps on TRUE objective --------
            let mut accepted = 0usize;
            let mut tried = 0usize;

            for _ in 0..self.steps_per_epoch {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let oi = self.pick_op_index(context.rng(), &self.op_weights);
                let op = &self.local_ops[oi];

                let mut pc = PlanningContext::new(model, &current, &eval, dv_buf.as_mut_slice());
                tried = tried.saturating_add(1);

                if let Some(mut plan) = op.propose(&mut pc, context.rng()) {
                    // Recompute TRUE/base delta so Fitness stays consistent
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

                    // Candidate
                    let mut next = current.clone();
                    next.apply_plan(plan);

                    // Metropolis on ENERGY
                    let e0 = self.energy_acceptor.energy(current.fitness());
                    let e1 = self.energy_acceptor.energy(next.fitness());
                    let delta_e = e1 - e0;

                    // Temperature floor behavior: still allow exp(-Δ/Tmin)
                    let t_eff = temp.max(self.min_temperature);
                    let accept = self.metropolis(context.rng(), delta_e, t_eff);

                    if accept {
                        current = next;
                        accepted = accepted.saturating_add(1);

                        // Credit operator (accepted)
                        Self::update_weight(
                            &mut self.op_weights[oi],
                            self.op_ema_alpha,
                            if delta_e <= 0 { 0.9 } else { 0.25 },
                            self.op_min_weight,
                        );

                        // If TRUE/base objective improved best → publish & record epsilon-stat
                        if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                            let drop = best
                                .fitness()
                                .cost
                                .saturating_sub(current.fitness().cost)
                                .max(0);
                            eps_src.record(drop);

                            best = current.clone();
                            let _ = context.shared_incumbent().try_update(&best, model);
                        }
                    } else {
                        // Small decay for unsuccessful proposal
                        Self::update_weight(
                            &mut self.op_weights[oi],
                            self.op_ema_alpha,
                            0.0,
                            self.op_min_weight,
                        );
                    }
                }
            }

            // ----- Staleness bookkeeping (per-epoch, on TRUE fitness) -----
            if let Some(delta) = stale.on_round_end(*best.fitness()) {
                eps_src.record(delta);
            }

            // ----- Refetch when STALE & incumbent materially better (ε-guarded) -----
            if stale.is_stale(patience_epochs) {
                let inc = context.shared_incumbent().peek();
                let eps = eps_src.epsilon();
                let material = materially_better(current.fitness(), &inc, eps);

                let allowed = match self.hard_refetch_mode {
                    HardRefetchMode::Always => true,
                    HardRefetchMode::IfBetter => material, // require material better if-guard
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

                    // Deterministic kick after refetch
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
                        let _applied = deterministic_kick(&mut adapter, k);
                    }

                    if self.true_acceptor.accept(best.fitness(), snap.fitness()) {
                        best = snap.clone();
                    }
                    current = snap;

                    // Reheat and cool-down staleness until next strict improvement
                    self.reheat(&mut temp);
                    stale.arm_cooldown_until_next_improvement();
                    tracing::debug!("SA: stale refetch → reheat to T={}", temp);
                }
            }

            // ----- Temperature update (geometric + AR nudges) -----
            let ar = Self::acceptance_ratio(accepted, tried);
            let mut next_temp = temp * self.cooling; // geometric base
            if ar < self.ar_target_low {
                next_temp = (temp * 1.05).max(next_temp);
            } // heat a little if frozen
            if ar > self.ar_target_high {
                next_temp *= self.cooling;
            } // cool extra if too random
            temp = next_temp.max(self.min_temperature);
        }

        // Final publish (no-op if not better than incumbent).
        let _ = context.shared_incumbent().try_update(&best, model);
    }
}

pub fn sa_strategy<T, R>(model: &SolverModel<T>) -> SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    // SA — “hotter & longer” preset for deeper exploration.
    SimulatedAnnealingStrategy::new()
        // ---- schedule ----
        .with_init_temp(35.0)
        .with_cooling(0.9997)
        .with_min_temp(1e-4)
        .with_steps_per_epoch(1200)
        // ---- refetch (ε-guarded in run) ----
        .with_hard_refetch_every(80)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_refetch_after_stale(60)
        .with_reheat_factor(0.85)
        .with_kick_ops_after_refetch(18)
        // ---- energy/bandit/targets ----
        .with_big_m_for_energy(900_000_000)
        .with_op_ema_alpha(0.30)
        .with_op_min_weight(0.12)
        .with_acceptance_targets(0.22, 0.70)
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
}
