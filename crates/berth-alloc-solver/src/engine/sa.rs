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
        search::{SearchContext, SearchStrategy},
    },
    search::{
        operator::LocalMoveOperator,
        operator_library::local::{
            CrossExchangeAcrossBerths, OrOptBlockRelocate, RelocateSingleBest,
            ShiftEarlierOnSameBerth, SwapPairSameBerth,
        },
        planner::{DefaultCostEvaluator, PlanningContext},
    },
    state::{fitness::Fitness, solver_state::SolverState},
};
use rand::seq::SliceRandom;
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
    big_m: i128,
}
impl Default for EnergyAcceptor {
    fn default() -> Self {
        Self {
            big_m: 1_000_000_000,
        } // strong priority on unassigned
    }
}
impl EnergyAcceptor {
    pub fn with_big_m(mut self, big_m: i128) -> Self {
        self.big_m = big_m.max(1);
        self
    }
    #[inline]
    pub fn energy(&self, f: &Fitness) -> i128 {
        let ua = f.unassigned_requests as i128;
        let c = f.cost as i128;
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

/// Simulated Annealing with separate acceptors:
/// - `sa_acceptor` (energy) guides the random walk.
/// - `true_acceptor` (lexicographic) controls upgrades to `best` and publishing.
pub struct SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,
    temperature: f64,
    cooling: f64, // e.g., 0.995
    min_temperature: f64,
    steps_per_temp: usize,

    // Acceptors
    true_acceptor: LexStrictAcceptor,
    sa_acceptor: EnergyAcceptor,

    // ILS-like sync/refetch knobs
    refetch_after_stale: usize, // epochs without global improvement before refetch; 0 => off
    hard_refetch_every: usize,  // every N epochs; 0 => off
    hard_refetch_mode: HardRefetchMode,
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
            cooling: 0.995,
            min_temperature: 1e-4,
            steps_per_temp: 256,
            true_acceptor: LexStrictAcceptor,
            sa_acceptor: EnergyAcceptor::default(),
            refetch_after_stale: 0,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
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
        self.temperature = t0.max(1e-9);
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
    pub fn with_big_m_for_energy(mut self, big_m: i128) -> Self {
        self.sa_acceptor = self.sa_acceptor.clone().with_big_m(big_m);
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

        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut temp = self.temperature;
        let mut epoch: usize = 0;
        let mut stale_epochs_without_global_improve: usize = 0;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            epoch = epoch.saturating_add(1);

            // Clamp effective temperature so tail behaves like hill-climbing
            let t_eff = if temp < self.min_temperature {
                self.min_temperature
            } else {
                temp
            };

            // Periodic hard refetch (sync threads).
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
                }
            }

            let mut improved_global_in_epoch = false;

            for _ in 0..self.steps_per_temp {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // Randomize operator order each inner step.
                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(context.rng());

                let mut moved = false;

                for &i in &order {
                    let op = &self.local_ops[i];
                    let mut pc = PlanningContext::new(
                        model,
                        &current,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );

                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        let mut tmp = current.clone();
                        tmp.apply_plan(plan);

                        // Deterministic improvement under SA energy? Accept.
                        if self.sa_acceptor.accept(current.fitness(), tmp.fitness()) {
                            current = tmp.clone();

                            // If it also improves true objective, upgrade and publish.
                            if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                                best = current.clone();
                                let _ = context.shared_incumbent().try_update(&best);
                                improved_global_in_epoch = true;
                            }

                            moved = true;
                            break;
                        } else {
                            // Maybe accept worse probabilistically using SA energy.
                            let e0 = self.sa_acceptor.energy(current.fitness());
                            let e1 = self.sa_acceptor.energy(tmp.fitness());
                            let delta = (e1 - e0) as f64;
                            if delta <= 0.0 || self.accept_worse(context.rng(), delta, t_eff) {
                                current = tmp;
                                // Do NOT update `best` unless true objective improves.
                                if self.true_acceptor.accept(best.fitness(), current.fitness()) {
                                    best = current.clone();
                                    let _ = context.shared_incumbent().try_update(&best);
                                    improved_global_in_epoch = true;
                                }
                                moved = true;
                                break;
                            }
                        }
                    }
                }

                if !moved {
                    // No accepted move at this inner step; try next step at same T.
                    continue;
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
                    }
                    // Reset either way to avoid immediate refetch loop.
                    stale_epochs_without_global_improve = 0;
                }
            }

            // Cool down but clamp at the floor; keep looping until stop signal.
            temp = (temp * self.cooling).max(self.min_temperature);
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
}
