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

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor},
        adaptive::{
            ops_book::OperatorBook,
            selection::SoftmaxSelector,
            tuning::{DefaultOperatorTuner, LocalCountTargetTuner, OrOptBlockKTuner},
        },
        neighbors,
        search::{SearchContext, SearchStrategy},
    },
    model::solver_model::SolverModel,
    search::{
        operator::{LocalMoveOperator, OperatorKind},
        operator_library::local::{
            CrossExchangeAcrossBerths, CrossExchangeBestAcrossBerths, HillClimbBestSwapSameBerth,
            HillClimbRelocateBest, OrOptBlockRelocate, RandomRelocateAnywhere,
            RandomizedGreedyRelocateRcl, RelocateSingleBest, RelocateSingleBestAllowWorsening,
            ShiftEarlierOnSameBerth, SwapPairSameBerth,
        },
        planner::{DefaultCostEvaluator, PlanningContext},
    },
    state::{
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

/// Energy-ordered acceptor used by SA for deterministic improvements.
/// Uses a BIG-M scalarization to strongly prioritize fewer unassigned requests.
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
/// - adaptive operator selection (Softmax by EWMA improvement/ms)
/// - per-operator tuning using OperatorTuning
/// - acceptance-ratio guided temperature adaptation
/// - lex-true upgrades for publishing/shared-incumbent
pub struct SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Local search operators (true objective via DefaultCostEvaluator).
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,
    // Adaptive operator book for locals
    local_book: OperatorBook<T, R>,

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

    // Reheat control: 0.0 => disabled; 1.0 => reset to initial temperature on refetch; (0,1) => partial
    reheat_factor: f64,

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
            local_book: OperatorBook::new(
                OperatorKind::Local,
                Box::new(SoftmaxSelector::default()),
            ),
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
        // register default tuner for this operator slot
        let _ = self
            .local_book
            .register_operator(Box::new(DefaultOperatorTuner::default()));
        self
    }

    pub fn with_local_op_tuned(
        mut self,
        op: Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>,
        tuner: Box<dyn crate::engine::adaptive::tuning::OperatorTuner<T>>,
    ) -> Self {
        self.local_ops.push(op);
        let _ = self.local_book.register_operator(tuner);
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

    pub fn with_min_temperature(mut self, t_min: f64) -> Self {
        self.min_temperature = t_min.max(0.0);
        self
    }

    /// 0.0 => no reheat; 1.0 => reset to initial temperature on refetch; (0,1) => partial reheat.
    pub fn with_reheat_factor(mut self, f: f64) -> Self {
        self.reheat_factor = f.clamp(0.0, 1.0);
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

                // Global stats for tuning/selection
                let global_stats = current.stats(model);

                // SA stagnation: epochs without global improvement relative to refetch threshold.
                let denom = self.refetch_after_stale.max(1);
                let stuck_factor =
                    (stale_epochs_without_global_improve as f64 / denom as f64).min(1.0);
                let stagnation = crate::engine::adaptive::tuning::Stagnation {
                    stale_rounds: stale_epochs_without_global_improve,
                    stuck_factor,
                };

                // Retune all local operators once per step
                self.local_book.retune_all(&global_stats, &stagnation);
                // Select operator index via adaptive selector (stagnation-aware)
                let oi = self
                    .local_book
                    .select(&global_stats, &stagnation, context.rng());

                let op = &mut self.local_ops[oi];

                // Push tuning to operator
                let tuning = *self.local_book.tuning_for(oi);
                op.tune(&tuning, &global_stats);

                tried_this_epoch = tried_this_epoch.saturating_add(1);

                let t0 = self.local_book.propose_started();
                let mut pc = PlanningContext::new(
                    model,
                    &current,
                    &DefaultCostEvaluator,
                    dv_buf.as_mut_slice(),
                );

                if let Some(mut plan) = op.propose(&mut pc, context.rng()) {
                    self.local_book.record_propose(oi, t0, true);

                    // Recompute base delta for the plan before applying to keep Fitness true/base.
                    use crate::model::index::RequestIndex;
                    use crate::state::decisionvar::DecisionVar;
                    use std::collections::HashMap;

                    let mut base_delta: Cost = Cost::from(0);

                    // Keep last patch per request if multiple appear
                    let mut last: HashMap<usize, DecisionVar<T>> = HashMap::new();
                    for p in &plan.decision_var_patches {
                        last.insert(p.index.get(), p.patch);
                    }

                    for (ri_u, patch) in last {
                        let ri = RequestIndex::new(ri_u);
                        let old_dv = current.decision_variables()[ri.get()];

                        // subtract old base cost
                        if let DecisionVar::Assigned(old) = old_dv
                            && let Some(c) =
                                model.cost_of_assignment(ri, old.berth_index, old.start_time)
                        {
                            base_delta = base_delta.saturating_sub(c);
                        }
                        // add new base cost
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

                    // Deterministic improvement under SA energy? Accept.
                    if self.sa_acceptor.accept(current.fitness(), tmp.fitness()) {
                        current = tmp.clone();
                        accepted_this_epoch = accepted_this_epoch.saturating_add(1);

                        // Record accepted outcome (true/base delta)
                        self.local_book.record_outcome(oi, true, base_delta as f64);

                        // Upgrade global if true objective improves.
                        if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                            best = current.clone();
                            let _ = context.shared_incumbent().try_update(&best, model);
                            improved_global_in_epoch = true;
                        }
                    } else {
                        // Maybe accept worse probabilistically using SA energy.
                        let e0 = self.sa_acceptor.energy(current.fitness());
                        let e1 = self.sa_acceptor.energy(tmp.fitness());
                        let delta = (e1 - e0) as f64;
                        if delta <= 0.0 || self.accept_worse(context.rng(), delta, t_eff) {
                            current = tmp;
                            accepted_this_epoch = accepted_this_epoch.saturating_add(1);

                            // Record accepted-but-worse outcome too (for stats)
                            self.local_book.record_outcome(oi, true, base_delta as f64);

                            // Upgrade `best` only if true objective improves.
                            if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                                best = current.clone();
                                let _ = context.shared_incumbent().try_update(&best, model);
                                improved_global_in_epoch = true;
                            }
                        } else {
                            // produced but rejected
                            self.local_book.record_outcome(oi, false, 0.0);
                        }
                    }
                } else {
                    // No plan produced
                    self.local_book.record_propose(oi, t0, false);
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
        let _ = context.shared_incumbent().try_update(&best, model);
    }
}

// ====================== SA (hot start, adaptive cool) =======================
pub fn sa_strategy<T, R>(model: &SolverModel<T>) -> SimulatedAnnealingStrategy<T, R>
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

    let heavy = || {
        WorkBudgetTuner::default()
            .with_soft_time_budget_ms(0.60)
            .with_intensity_bounds(0.06, 0.36)
            .with_max_greediness(0.65)
            .with_max_locality(0.75)
    };

    SimulatedAnnealingStrategy::new()
        .with_init_temp(1.55)
        .with_cooling(0.9992)
        .with_min_temperature(1e-4)
        .with_steps_per_temp(520)
        .with_refetch_after_stale(48)
        .with_hard_refetch_every(24)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_reheat_factor(0.70)
        .with_acceptance_targets(0.12, 0.50)
        .with_big_m_for_energy(1_250_000_000)
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
                OrOptBlockRelocate::new(2..=3, 1.50).with_neighbors(neighbors_same_berth.clone()),
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
            Box::new(heavy()),
        )
        .with_local_op_tuned(
            Box::new(
                HillClimbBestSwapSameBerth::new(1..=1).with_neighbors(neighbors_same_berth.clone()),
            ),
            Box::new(heavy()),
        )
        .with_local_op_tuned(
            Box::new(
                RandomizedGreedyRelocateRcl::new(1..=1, 1.80)
                    .with_neighbors(neighbors_direct_competitors.clone()),
            ),
            Box::new(heavy()),
        )
        .with_local_op_tuned(
            Box::new(
                CrossExchangeBestAcrossBerths::new(1..=1).with_neighbors(neighbors_any.clone()),
            ),
            Box::new(heavy()),
        )
}
