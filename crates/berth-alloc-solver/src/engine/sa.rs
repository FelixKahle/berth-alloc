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

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor},
        search::{SearchContext, SearchStrategy},
    },
    search::{
        operator::LocalMoveOperator,
        operator_library::local::{
            CrossExchangeAcrossBerths, HillClimbBestSwapSameBerth, HillClimbRelocateBest,
            OrOptBlockRelocate, RandomRelocateAnywhere, RelocateSingleBest,
            RelocateSingleBestAllowWorsening, ShiftEarlierOnSameBerth, SwapPairSameBerth,
        },
        planner::{DefaultCostEvaluator, PlanningContext},
    },
    state::{fitness::Fitness, solver_state::SolverState},
};
use std::sync::atomic::Ordering as AtomicOrdering;

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

/// Energy-ordered acceptor used by SA for *deterministic* improvements.
/// Uses a BIG-M scalarization to strongly prioritize fewer unassigned requests.
/// Assumes Cost = i64.
#[derive(Debug, Clone)]
pub struct EnergyAcceptor {
    big_m: i64,
}
impl Default for EnergyAcceptor {
    fn default() -> Self {
        Self {
            big_m: 1_000_000_000,
        } // strong priority on unassigned
    }
}
impl EnergyAcceptor {
    pub fn with_big_m(mut self, big_m: i64) -> Self {
        self.big_m = big_m.max(1);
        self
    }
    #[inline]
    pub fn energy(&self, f: &Fitness) -> i64 {
        let ua = f.unassigned_requests as i64;
        let c = f.cost;
        ua.saturating_mul(self.big_m).saturating_add(c)
    }
}
impl Acceptor for EnergyAcceptor {
    fn name(&self) -> &str {
        "EnergyAcceptor"
    }
    #[inline]
    fn accept(&self, current: &Fitness, new: &Fitness) -> bool {
        self.energy(new) < self.energy(current)
    }
}

/// Simulated Annealing with:
/// - bandit-style adaptive operator selection (EMA-weighted roulette)
/// - acceptance-ratio guided temperature adaptation
/// - lex-true upgrades for publishing/shared-incumbent
pub struct SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Local search operators (true objective via DefaultCostEvaluator).
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,

    // Temperature schedule
    temperature: f64,
    init_temperature: f64, // remember initial temp for reheating
    cooling: f64,
    min_temperature: f64,
    steps_per_temp: usize,

    // Acceptors
    true_acceptor: LexStrictAcceptor,
    sa_acceptor: EnergyAcceptor,

    // ILS-like sync/refetch knobs
    refetch_after_stale: usize, // epochs without global improvement before refetch; 0 => off
    hard_refetch_every: usize,  // every N epochs; 0 => off
    hard_refetch_mode: HardRefetchMode,

    // Reheat control: 0.0 => disabled; 1.0 => reset to init_temperature; (0,1) => partial
    reheat_factor: f64,

    // ---- Adaptive operator scheduling (bandit-style) ----
    // Per-operator positive weights used in weighted roulette selection.
    op_weights: Vec<f64>,
    // EMA step for weight updates (0,1]; higher -> faster adaptation.
    op_ema_alpha: f64,
    // Keep all operators selectable.
    op_min_weight: f64,

    // ---- Adaptive acceptance targeting ----
    ar_target_low: f64,  // if acceptance ratio < low -> heat up a bit
    ar_target_high: f64, // if acceptance ratio > high -> cool extra
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
        Self {
            local_ops: Vec::new(),
            temperature: 1.0,
            init_temperature: 1.0,
            cooling: 0.995,
            min_temperature: 1e-4,
            steps_per_temp: 256,
            true_acceptor: LexStrictAcceptor,
            sa_acceptor: EnergyAcceptor::default(),
            refetch_after_stale: 0,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
            reheat_factor: 1.0, // default: full reheat on refetch
            // bandit defaults
            op_weights: Vec::new(),
            op_ema_alpha: 0.15,
            op_min_weight: 0.05,
            // acceptance ratio guidance
            ar_target_low: 0.10,
            ar_target_high: 0.45,
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
        let t = t0.max(1e-9);
        self.temperature = t;
        self.init_temperature = t;
        self
    }

    pub fn with_cooling(mut self, factor: f64) -> Self {
        self.cooling = factor;
        self
    }

    pub fn with_steps_per_temp(mut self, k: usize) -> Self {
        self.steps_per_temp = k.max(1);
        self
    }

    pub fn with_refetch_after_stale(mut self, epochs: usize) -> Self {
        self.refetch_after_stale = epochs;
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

    pub fn with_big_m_for_energy(mut self, big_m: i64) -> Self {
        self.sa_acceptor = self.sa_acceptor.clone().with_big_m(big_m);
        self
    }

    /// 0.0 => no reheat; 1.0 => reset to initial temperature on refetch; (0,1) => partial reheat.
    pub fn with_reheat_factor(mut self, f: f64) -> Self {
        self.reheat_factor = f.clamp(0.0, 1.0);
        self
    }

    // (Optional) public tuners for bandit and acceptance targets:
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

    /// Accept a worse move with probability exp(-Δ/T), where Δ is the SA energy increase.
    #[inline]
    fn accept_worse<RNG: rand::Rng>(&self, rng: &mut RNG, delta_energy: f64, temp: f64) -> bool {
        let p = (-delta_energy / temp).exp();
        rng.random::<f64>() < p
    }

    #[inline]
    fn should_hard_refetch(&self, epoch: usize) -> bool {
        self.hard_refetch_every > 0 && epoch > 0 && epoch.is_multiple_of(self.hard_refetch_every)
    }

    #[inline]
    fn reheat(&self, temp: &mut f64) {
        if self.reheat_factor > 0.0 {
            let target = (self.init_temperature * self.reheat_factor).max(self.min_temperature);
            // raise to at least target, but do not reduce temperature if already higher
            *temp = temp.max(target);
        }
    }

    #[inline]
    fn pick_operator_index(&self, rng: &mut R, weights: &[f64]) -> usize {
        debug_assert!(!weights.is_empty());
        let sum: f64 = weights.iter().copied().sum();
        // fall back to uniform if degenerate
        if !sum.is_finite() || sum <= 0.0 {
            return rng.random_range(0..weights.len());
        }
        let mut r = rng.random::<f64>() * sum;
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
        // Reward in [0,1]; EMA update then clamp to [min_w, +inf).
        let base = if w.is_finite() && *w > 0.0 { *w } else { 1.0 };
        let new_w = (1.0 - alpha) * base + alpha * (reward.max(0.0));
        *w = new_w.max(min_w);
    }

    #[inline]
    fn acceptance_ratio(accepted: usize, tried: usize) -> f64 {
        if tried == 0 {
            0.0
        } else {
            (accepted as f64) / (tried as f64)
        }
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

        // Two states: `current` (walker) and `best` (true objective).
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best: SolverState<'p, T> = current.clone();

        // local scratch buffer
        let mut dv_buf: Vec<crate::state::decisionvar::DecisionVar<T>> = vec![
                crate::state::decisionvar::DecisionVar::unassigned();
                model.flexible_requests_len()
            ];

        let mut temp = self.temperature;
        let mut epoch: usize = 0;
        let mut stale_epochs_without_global_improve: usize = 0;

        // Lazily initialize operator weights (all ones).
        if self.op_weights.len() != self.local_ops.len() {
            // NOTE: we do not mutate &self here; it is &mut self already inside run()
            self.op_weights.clear();
            self.op_weights.resize(self.local_ops.len(), 1.0);
        }

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            epoch = epoch.saturating_add(1);

            // Effective temperature: once below floor, behave like hill-climbing unless reheated.
            let t_eff = if temp < self.min_temperature {
                self.min_temperature
            } else {
                temp
            };

            // Periodic hard refetch (sync threads). Optionally reheat on refetch.
            if self.should_hard_refetch(epoch) {
                let inc = context.shared_incumbent().peek();
                let do_fetch = match self.hard_refetch_mode {
                    HardRefetchMode::IfBetter => self.true_acceptor.accept(current.fitness(), &inc),
                    HardRefetchMode::Always => true,
                };
                if do_fetch {
                    tracing::debug!(
                        "SA: periodic refetch at epoch {} (curr {}, inc {})",
                        epoch,
                        current.fitness(),
                        inc
                    );
                    let snap = context.shared_incumbent().snapshot();
                    current = snap.clone();
                    if self.true_acceptor.accept(best.fitness(), snap.fitness()) {
                        best = snap;
                    }
                    // Reheat and clear staleness, since we jumped basins.
                    self.reheat(&mut temp);
                    stale_epochs_without_global_improve = 0;
                    tracing::debug!("SA: refetch → reheat to T={}", temp);
                }
            }

            let mut improved_global_in_epoch = false;

            // Acceptance accounting for adaptive temperature
            let mut accepted_this_epoch: usize = 0;
            let mut tried_this_epoch: usize = 0;

            for _ in 0..self.steps_per_temp {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // Pick a single operator by weighted roulette.
                let oi = self.pick_operator_index(context.rng(), &self.op_weights);
                let op = &self.local_ops[oi];

                let mut pc = PlanningContext::new(
                    model,
                    &current,
                    &DefaultCostEvaluator,
                    dv_buf.as_mut_slice(),
                );
                tried_this_epoch = tried_this_epoch.saturating_add(1);

                if let Some(plan) = op.propose(&mut pc, context.rng()) {
                    let mut tmp = current.clone();
                    tmp.apply_plan(plan);

                    // Deterministic improvement under SA energy? Accept.
                    if self.sa_acceptor.accept(current.fitness(), tmp.fitness()) {
                        current = tmp.clone();
                        accepted_this_epoch = accepted_this_epoch.saturating_add(1);

                        // Bandit reward: strong credit for true improvement; moderate otherwise.
                        let mut reward = 0.6;
                        if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                            best = current.clone();
                            let _ = context.shared_incumbent().try_update(&best);
                            improved_global_in_epoch = true;
                            reward = 1.0;
                        }
                        Self::update_weight(
                            &mut self.op_weights[oi],
                            self.op_ema_alpha,
                            reward,
                            self.op_min_weight,
                        );
                    } else {
                        // Maybe accept worse probabilistically using SA energy.
                        let e0 = self.sa_acceptor.energy(current.fitness());
                        let e1 = self.sa_acceptor.energy(tmp.fitness());
                        let delta = (e1 - e0) as f64;
                        if delta <= 0.0 || self.accept_worse(context.rng(), delta, t_eff) {
                            current = tmp;
                            accepted_this_epoch = accepted_this_epoch.saturating_add(1);

                            // Small credit for accepted-worse (diversification).
                            Self::update_weight(
                                &mut self.op_weights[oi],
                                self.op_ema_alpha,
                                0.25,
                                self.op_min_weight,
                            );

                            // Upgrade `best` only if true objective improves.
                            if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                                best = current.clone();
                                let _ = context.shared_incumbent().try_update(&best);
                                improved_global_in_epoch = true;

                                // Bonus: a bit more credit if a worse-by-energy
                                // move still improved the true objective.
                                Self::update_weight(
                                    &mut self.op_weights[oi],
                                    self.op_ema_alpha,
                                    0.5,
                                    self.op_min_weight,
                                );
                            }
                        } else {
                            // Tiny penalty (do not drive to zero).
                            Self::update_weight(
                                &mut self.op_weights[oi],
                                self.op_ema_alpha,
                                0.0,
                                self.op_min_weight,
                            );
                        }
                    }
                }
            }

            if improved_global_in_epoch {
                stale_epochs_without_global_improve = 0;
            } else {
                stale_epochs_without_global_improve =
                    stale_epochs_without_global_improve.saturating_add(1);

                // Staleness refetch (only if incumbent strictly better than our current).
                if self.refetch_after_stale > 0
                    && stale_epochs_without_global_improve >= self.refetch_after_stale
                {
                    let inc = context.shared_incumbent().peek();
                    if self.true_acceptor.accept(current.fitness(), &inc) {
                        tracing::debug!(
                            "SA: staleness refetch at epoch {} ({} -> {})",
                            epoch,
                            current.fitness(),
                            inc
                        );
                        let snap = context.shared_incumbent().snapshot();
                        current = snap.clone();
                        if self.true_acceptor.accept(best.fitness(), snap.fitness()) {
                            best = snap;
                        }
                        // Reheat and clear staleness when refetching to a better basin.
                        self.reheat(&mut temp);
                        tracing::debug!("SA: stale-refetch → reheat to T={}", temp);
                    }
                    // Reset either way to avoid immediate refetch loop.
                    stale_epochs_without_global_improve = 0;
                }
            }

            // ---- Adaptive temperature update by acceptance ratio ----
            let ar = Self::acceptance_ratio(accepted_this_epoch, tried_this_epoch);
            // Base cooling
            let mut next_temp = temp * self.cooling;
            // Heat up slightly if we're “frozen”
            if ar < self.ar_target_low {
                next_temp = (temp * 1.06).max(next_temp);
            }
            // Cool extra if we accept too much (too random)
            if ar > self.ar_target_high {
                next_temp *= self.cooling;
            }
            temp = next_temp.max(self.min_temperature);
        }

        // Final publish (no-op if `best` doesn't beat the shared incumbent).
        let _ = context.shared_incumbent().try_update(&best);
    }
}

pub fn sa_strategy<T, R>(
    _: &crate::model::solver_model::SolverModel<T>,
) -> SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    SimulatedAnnealingStrategy::new()
        .with_init_temp(1.0)
        .with_cooling(0.997)
        .with_steps_per_temp(256)
        .with_refetch_after_stale(64)
        .with_hard_refetch_every(0)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_reheat_factor(8.0)
        .with_op_ema_alpha(0.2)
        .with_acceptance_targets(0.12, 0.5)
        .with_big_m_for_energy(1_500_000_000)
        // Local improvement operators (true objective via DefaultCostEvaluator)
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
        // Diversification / worsening-capable moves
        .with_local_op(Box::new(RelocateSingleBestAllowWorsening::new(2..=4)))
        .with_local_op(Box::new(RandomRelocateAnywhere::new(2..=4)))
        // Hill climbers (strictly improving)
        .with_local_op(Box::new(HillClimbRelocateBest::new(12..=36)))
        .with_local_op(Box::new(HillClimbBestSwapSameBerth::new(24..=72)))
}
