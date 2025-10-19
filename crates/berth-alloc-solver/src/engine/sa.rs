// src/search/strategy_sa.rs

use crate::{
    core::numeric::SolveNumeric,
    engine::search::{SearchContext, SearchStrategy},
    search::{
        operator::LocalMoveOperator,
        operator_library::{
            local::{
                CrossExchangeAcrossBerths, OrOptBlockRelocate, RelocateSingleBest,
                ShiftEarlierOnSameBerth, SwapPairSameBerth,
            },
            math::MipBlockReoptimize,
        },
        planner::PlanningContext,
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

pub struct SimulatedAnnealingStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    local_ops: Vec<Box<dyn LocalMoveOperator<T, R>>>,
    temperature: f64,
    cooling: f64, // e.g., 0.995
    min_temperature: f64,
    steps_per_temp: usize,

    // ILS-like sync/refetch knobs
    refetch_after_stale: usize, // temps without global improvement before refetch; 0 => off
    hard_refetch_every: usize,  // every N temp epochs; 0 => off
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
            refetch_after_stale: 0,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
        }
    }
    pub fn with_local_op(mut self, op: Box<dyn LocalMoveOperator<T, R>>) -> Self {
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
        self.steps_per_temp = k;
        self
    }
    pub fn with_refetch_after_stale(mut self, temps: usize) -> Self {
        self.refetch_after_stale = temps;
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

    #[inline]
    fn accept_worse<RNG: rand::Rng>(&self, rng: &mut RNG, delta_score: f64, temp: f64) -> bool {
        // delta_score > 0 means worse; standard SA probability
        let p = (-delta_score / temp).exp();
        rng.random::<f64>() < p
    }

    #[inline]
    fn scalar_score(fit: &Fitness) -> (i64, i64) {
        (fit.unassigned_requests as i64, fit.cost)
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

    #[tracing::instrument(name = "SA Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();
        if self.local_ops.is_empty() {
            tracing::warn!("SA: no local operators configured");
            return;
        }

        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();

        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let mut temp = self.temperature;
        let mut epoch: usize = 0;
        let mut stale_epochs_without_global_improve: usize = 0;
        let mut best_global = context.shared_incumbent().peek();

        'outer: while temp > self.min_temperature {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            epoch = epoch.saturating_add(1);

            // Periodic hard refetch (sync threads)
            if self.should_hard_refetch(epoch) {
                let inc = context.shared_incumbent().peek();
                let do_fetch = match self.hard_refetch_mode {
                    HardRefetchMode::IfBetter => inc < *current.fitness(),
                    HardRefetchMode::Always => true,
                };
                if do_fetch {
                    tracing::debug!(
                        "SA: periodic refetch at epoch {} (curr {}, inc {})",
                        epoch,
                        current.fitness(),
                        inc
                    );
                    current = context.shared_incumbent().snapshot();
                }
            }

            let mut improved_global_in_epoch = false;

            for _ in 0..self.steps_per_temp {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(context.rng());

                let mut moved = false;

                for &i in &order {
                    let op = &self.local_ops[i];
                    let mut pc = PlanningContext::new(model, &current, dv_buf.as_mut_slice());
                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        let mut tmp = current.clone();
                        tmp.apply_plan(plan);

                        if tmp.fitness() < current.fitness() {
                            current = tmp;
                            let _ = context.shared_incumbent().try_update(&current);
                            moved = true;
                            break; // continue at same temperature
                        } else {
                            // maybe accept worse
                            let (ua0, c0) = Self::scalar_score(current.fitness());
                            let (ua1, c1) = Self::scalar_score(tmp.fitness());
                            let delta_score = if ua1 != ua0 {
                                ((ua1 - ua0) as f64) * 1e9 // prioritize unassigned
                            } else {
                                (c1 - c0) as f64
                            };
                            if delta_score > 0.0
                                && self.accept_worse(context.rng(), delta_score, temp)
                            {
                                current = tmp;
                                moved = true;
                                break;
                            }
                        }
                    }
                }

                // Track global improvement
                let inc_after = context.shared_incumbent().peek();
                if inc_after < best_global {
                    best_global = inc_after;
                    improved_global_in_epoch = true;
                }

                if !moved {
                    // could not move at this inner step; just try another temperature step
                    continue;
                }
            }

            if improved_global_in_epoch {
                stale_epochs_without_global_improve = 0;
            } else {
                stale_epochs_without_global_improve =
                    stale_epochs_without_global_improve.saturating_add(1);

                // Staleness refetch (only if incumbent strictly better)
                if self.refetch_after_stale > 0
                    && stale_epochs_without_global_improve >= self.refetch_after_stale
                {
                    let inc = context.shared_incumbent().peek();
                    if inc < *current.fitness() {
                        tracing::debug!(
                            "SA: staleness refetch at epoch {} ({} -> {})",
                            epoch,
                            current.fitness(),
                            inc
                        );
                        current = context.shared_incumbent().snapshot();
                        stale_epochs_without_global_improve = 0;
                    } else {
                        // keep going
                        stale_epochs_without_global_improve = 0;
                    }
                }
            }

            temp *= self.cooling;
        }

        let _ = context.shared_incumbent().try_update(&current);
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
        .with_local_op(Box::new(MipBlockReoptimize::same_berth(
            2..=5,  // block length k
            3..=6,  // candidate starts per free interval
            6..=10, // max candidates per request
        )))
        // (B) across-all-berths block reopt — heavier, keep caps tighter
        .with_local_op(Box::new(MipBlockReoptimize::across_all_berths(
            2..=4, // k a bit smaller across berths
            2..=4, // fewer starts per interval
            4..=8, // tighter cap per request
        )))
}
