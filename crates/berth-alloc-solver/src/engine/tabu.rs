// src/search/strategy_tabu.rs

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
    state::{fitness::Fitness, plan::Plan, solver_state::SolverState},
};
use rand::seq::SliceRandom;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::{
    collections::{HashMap, HashSet},
    ops::RangeInclusive,
};

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

pub struct TabuSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    local_ops: Vec<Box<dyn LocalMoveOperator<T, R>>>,
    tabu_tenure_rounds: RangeInclusive<usize>,
    max_local_steps: usize,

    // Neighborhood sampling per tabu step (choose best admissible)
    samples_per_step: usize,

    // ILS-like sync/refetch knobs
    refetch_after_stale: usize, // 0 => disabled
    hard_refetch_every: usize,  // 0 => disabled
    hard_refetch_mode: HardRefetchMode,
}

impl<T, R> Default for TabuSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> TabuSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    pub fn new() -> Self {
        Self {
            local_ops: Vec::new(),
            tabu_tenure_rounds: 12..=24,
            max_local_steps: 512,
            samples_per_step: 64,
            refetch_after_stale: 128,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
        }
    }

    pub fn with_local_op(mut self, op: Box<dyn LocalMoveOperator<T, R>>) -> Self {
        self.local_ops.push(op);
        self
    }
    pub fn with_tabu_tenure(mut self, rounds: RangeInclusive<usize>) -> Self {
        self.tabu_tenure_rounds = rounds;
        self
    }
    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps.max(1);
        self
    }
    pub fn with_samples_per_step(mut self, k: usize) -> Self {
        self.samples_per_step = k.max(8);
        self
    }
    pub fn with_refetch_after_stale(mut self, rounds: usize) -> Self {
        self.refetch_after_stale = rounds;
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

    fn sample_tenure(&self, rng: &mut R) -> usize {
        let lo = *self.tabu_tenure_rounds.start();
        let hi = *self.tabu_tenure_rounds.end();
        if lo == hi {
            lo
        } else {
            rng.random_range(lo..=hi)
        }
    }

    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }
}

impl<T, R> SearchStrategy<T, R> for TabuSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "Tabu Search"
    }

    #[tracing::instrument(name = "Tabu Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();

        if self.local_ops.is_empty() {
            tracing::warn!("Tabu: no local operators configured");
            return;
        }

        // Working state & scratch
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();

        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Tabu list keyed by request raw id → expire round
        let mut tabu_until: HashMap<usize, usize> = HashMap::new();

        // Loop control
        let mut round: usize = 0;
        let mut stale_rounds: usize = 0;
        let mut last_best_current = current.fitness().clone();

        // Helper: compute candidate fitness from current + plan deltas (no need to apply)
        #[inline]
        fn fitness_after<'p, T: SolveNumeric>(base: &Fitness, plan: &Plan<'p, T>) -> Fitness {
            // plan.delta_unassigned is i32; base.unassigned_requests is usize.
            let ua =
                (base.unassigned_requests as i64 + plan.delta_unassigned as i64).max(0) as usize;
            let cost = base.cost.saturating_add(plan.delta_cost);
            Fitness::new(cost, ua)
        }

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            round = round.saturating_add(1);

            // Periodic hard refetch
            if self.should_hard_refetch(round) {
                let best_now = context.shared_incumbent().peek();
                let do_fetch = match self.hard_refetch_mode {
                    HardRefetchMode::IfBetter => best_now < *current.fitness(),
                    HardRefetchMode::Always => true,
                };
                if do_fetch {
                    tracing::debug!(
                        "Tabu: periodic refetch at round {} (curr {}, inc {})",
                        round,
                        current.fitness(),
                        best_now
                    );
                    current = context.shared_incumbent().snapshot();
                    last_best_current = current.fitness().clone();
                }
            }

            let mut improved_this_round = false;

            // Multiple tabu "steps" per outer round
            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // Candidate buffers (store plan + evaluated fitness)
                struct Cand<'p, T: SolveNumeric> {
                    plan: Plan<'p, T>,
                    moved: Vec<usize>,
                    fitness: Fitness,
                }

                let mut best_admissible: Option<Cand<'p, T>> = None;
                let mut best_overall: Option<Cand<'p, T>> = None;

                // Prepare random order of operators; sample neighborhood
                let mut op_order: Vec<usize> = (0..self.local_ops.len()).collect();
                op_order.shuffle(context.rng());
                let mut samples_left = self.samples_per_step;

                while samples_left > 0 {
                    samples_left -= 1;
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let oi = op_order[samples_left % op_order.len()];
                    let op = &self.local_ops[oi];

                    let mut pc = PlanningContext::new(model, &current, dv_buf.as_mut_slice());
                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        // Which requests are touched?
                        let mut moved: HashSet<usize> = HashSet::new();
                        for p in &plan.decision_var_patches {
                            moved.insert(p.index.get());
                        }
                        if moved.is_empty() {
                            continue;
                        }

                        // Evaluate candidate fitness cheaply (no apply)
                        let cand_fit = fitness_after(current.fitness(), &plan);

                        // Tabu / aspiration
                        let is_tabu = moved
                            .iter()
                            .any(|rid| tabu_until.get(rid).is_some_and(|&e| e > round));
                        let beats_global = cand_fit < context.shared_incumbent().peek();

                        // ---- Update "best overall" (no tabu restriction) ----
                        let better_overall = match &best_overall {
                            None => true,
                            Some(b) => cand_fit < b.fitness,
                        };
                        if better_overall {
                            best_overall = Some(Cand {
                                plan: plan.clone(),
                                moved: moved.iter().copied().collect(),
                                fitness: cand_fit.clone(),
                            });
                        }

                        // ---- Update "best admissible" (non-tabu or aspiration) ----
                        if !is_tabu || beats_global {
                            // Recompute or reuse cand_fit (reusing is fine since not moved above yet)
                            let better_adm = match &best_admissible {
                                None => true,
                                Some(b) => cand_fit < b.fitness,
                            };
                            if better_adm {
                                best_admissible = Some(Cand {
                                    plan, // move original plan here
                                    moved: moved.into_iter().collect(),
                                    fitness: cand_fit,
                                });
                            }
                        }
                    }
                }

                // Choose: best admissible, else best overall (classic TS fallback)
                let chosen = match (best_admissible, best_overall) {
                    (Some(a), _) => Some(a),
                    (None, Some(o)) => Some(o),
                    (None, None) => None,
                };

                let Some(ch) = chosen else {
                    // No candidates → stop inner; continue outer (don’t exit whole strategy)
                    break;
                };

                // Tenure lock for moved requests
                let tenure = self.sample_tenure(context.rng());
                for rid in &ch.moved {
                    tabu_until.insert(*rid, round + tenure);
                }

                // Apply selected plan to the real state
                current.apply_plan(ch.plan);
                let _ = context.shared_incumbent().try_update(&current);

                // Track local improvement
                if current.fitness() < &last_best_current {
                    last_best_current = current.fitness().clone();
                    improved_this_round = true;
                    stale_rounds = 0;
                }
            }

            if !improved_this_round {
                stale_rounds = stale_rounds.saturating_add(1);

                // Staleness refetch
                if self.refetch_after_stale > 0 && stale_rounds >= self.refetch_after_stale {
                    let best_now = context.shared_incumbent().peek();
                    if best_now < *current.fitness() {
                        tracing::debug!(
                            "Tabu: staleness refetch after {} rounds ({} -> {})",
                            stale_rounds,
                            current.fitness(),
                            best_now
                        );
                        current = context.shared_incumbent().snapshot();
                        last_best_current = current.fitness().clone();
                        stale_rounds = 0;
                    } else {
                        // Keep going anyway — avoid early exit
                        stale_rounds = 0;
                    }
                }
            }
        }

        let _ = context.shared_incumbent().try_update(&current);
    }
}

// Recommended default config
pub fn tabu_strategy<T, R>(
    _model: &crate::model::solver_model::SolverModel<T>,
) -> TabuSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    TabuSearchStrategy::new()
        .with_max_local_steps(1024)
        .with_tabu_tenure(16..=32)
        .with_samples_per_step(96)
        .with_refetch_after_stale(128)
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
