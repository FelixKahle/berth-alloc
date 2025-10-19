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
    state::{fitness::Fitness, plan::Plan, solver_state::SolverState},
};
use rand::seq::SliceRandom;
use std::{
    collections::{HashMap, HashSet},
    ops::RangeInclusive,
    sync::atomic::Ordering as AtomicOrdering,
};

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

/// Energy-ordered acceptor for ranking candidates in the Tabu neighborhood.
/// BIG-M scalarization strongly prioritizes fewer unassigned requests.
/// Assumes Cost = i64.
#[derive(Debug, Clone)]
pub struct EnergyAcceptor {
    big_m: i128,
}
impl Default for EnergyAcceptor {
    fn default() -> Self {
        Self {
            big_m: 1_000_000_000,
        }
    }
}
impl EnergyAcceptor {
    pub fn with_big_m(mut self, big_m: i128) -> Self {
        self.big_m = big_m.max(1);
        self
    }
    #[inline]
    fn energy(&self, f: &Fitness) -> i128 {
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
    fn accept(&self, cur: &Fitness, cand: &Fitness) -> bool {
        self.energy(cand) < self.energy(cur)
    }
}

pub struct TabuSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    // Local operators evaluated with the true (default) objective.
    local_ops: Vec<Box<dyn LocalMoveOperator<T, DefaultCostEvaluator, R>>>,

    // Tenure range sampled per move.
    tabu_tenure_rounds: RangeInclusive<usize>,

    // Inner steps per outer round.
    max_local_steps: usize,

    // Neighborhood sampling per step (pick best admissible).
    samples_per_step: usize,

    // Acceptors:
    // - true_acceptor: lexicographic on (unassigned, cost) for publishing / "best_true".
    // - walk_acceptor: energy-ordered comparator to rank neighborhood candidates.
    true_acceptor: LexStrictAcceptor,
    walk_acceptor: EnergyAcceptor,

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
            true_acceptor: LexStrictAcceptor,
            walk_acceptor: EnergyAcceptor::default(),
            refetch_after_stale: 128,
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

    /// Tune BIG-M used to rank neighborhood moves (walker metric).
    pub fn with_big_m_for_walker(mut self, big_m: i128) -> Self {
        self.walk_acceptor = self.walk_acceptor.clone().with_big_m(big_m);
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

    #[inline]
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

    #[tracing::instrument(level = "debug", name = "Tabu Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();

        if self.local_ops.is_empty() {
            tracing::warn!("Tabu: no local operators configured");
            return;
        }

        // Two states:
        // - current: the walker (can worsen true objective)
        // - best_true: best by true objective; only this is published
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut best_true: SolverState<'p, T> = current.clone();

        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Tabu list keyed by request raw id → expire at round
        let mut tabu_until: HashMap<usize, usize> = HashMap::new();

        // Loop control
        let mut round: usize = 0;
        let mut stale_rounds: usize = 0;
        let mut last_best_current = current.fitness().clone();

        // Helper: estimate fitness after applying a plan (base + deltas)
        #[inline]
        fn fitness_after<'p, T: SolveNumeric>(base: &Fitness, plan: &Plan<'p, T>) -> Fitness {
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

            // Periodic hard refetch: sync current; upgrade best_true if the snapshot is better
            if self.should_hard_refetch(round) {
                let inc = context.shared_incumbent().peek();
                let do_fetch = match self.hard_refetch_mode {
                    HardRefetchMode::IfBetter => self.true_acceptor.accept(current.fitness(), &inc),
                    HardRefetchMode::Always => true,
                };
                if do_fetch {
                    tracing::debug!(
                        "Tabu: periodic refetch at round {} (curr {}, inc {})",
                        round,
                        current.fitness(),
                        inc
                    );
                    let snap = context.shared_incumbent().snapshot();
                    current = snap.clone();
                    if self
                        .true_acceptor
                        .accept(best_true.fitness(), snap.fitness())
                    {
                        best_true = snap;
                    }
                    last_best_current = current.fitness().clone();
                }
            }

            let mut improved_this_round = false;

            // Multiple tabu "steps" per outer round
            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // Candidate buffers (store plan + moved set + computed fitness)
                struct Cand<'p, T: SolveNumeric> {
                    plan: Plan<'p, T>,
                    moved: Vec<usize>,
                    fitness: Fitness,
                }

                let mut best_admissible: Option<Cand<'p, T>> = None;
                let mut best_overall: Option<Cand<'p, T>> = None;

                // Randomized operator visiting each sample
                let mut op_order: Vec<usize> = (0..self.local_ops.len()).collect();
                op_order.shuffle(context.rng());

                for s in 0..self.samples_per_step {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let oi = op_order[s % op_order.len()];
                    let op = &self.local_ops[oi];

                    // Operators see the true objective via DefaultCostEvaluator
                    let mut pc = PlanningContext::new(
                        model,
                        &current,
                        &DefaultCostEvaluator,
                        dv_buf.as_mut_slice(),
                    );

                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        // Which requests are touched?
                        let mut moved: HashSet<usize> = HashSet::new();
                        for p in &plan.decision_var_patches {
                            moved.insert(p.index.get());
                        }
                        if moved.is_empty() {
                            continue;
                        }

                        // Evaluate candidate fitness (no need to apply)
                        let cand_fit = fitness_after(current.fitness(), &plan);

                        // Tabu / aspiration
                        let is_tabu = moved
                            .iter()
                            .any(|rid| tabu_until.get(rid).is_some_and(|&e| e > round));

                        // Aspiration if candidate beats (a) local best_true or (b) shared incumbent
                        let beats_local_best =
                            self.true_acceptor.accept(best_true.fitness(), &cand_fit);
                        let beats_shared = self
                            .true_acceptor
                            .accept(&context.shared_incumbent().peek(), &cand_fit);

                        // ---- Update "best overall" (ignores tabu) using WALK metric ----
                        let better_overall = match &best_overall {
                            None => true,
                            Some(b) => self.walk_acceptor.accept(&b.fitness, &cand_fit),
                        };
                        if better_overall {
                            best_overall = Some(Cand {
                                plan: plan.clone(),
                                moved: moved.iter().copied().collect(),
                                fitness: cand_fit.clone(),
                            });
                        }

                        // ---- Update "best admissible" (non-tabu or aspiration) using WALK metric ----
                        if !is_tabu || beats_local_best || beats_shared {
                            let better_adm = match &best_admissible {
                                None => true,
                                Some(b) => self.walk_acceptor.accept(&b.fitness, &cand_fit),
                            };
                            if better_adm {
                                best_admissible = Some(Cand {
                                    plan, // move original
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
                    // No candidates → stop inner; continue outer
                    break;
                };

                // Tenure lock for moved requests
                let tenure = self.sample_tenure(context.rng());
                for rid in &ch.moved {
                    tabu_until.insert(*rid, round.saturating_add(tenure));
                }

                // Apply selected plan to the real state (walker may worsen)
                current.apply_plan(ch.plan);

                // Upgrade local best_true and publish only when improved by true objective
                if self
                    .true_acceptor
                    .accept(best_true.fitness(), current.fitness())
                {
                    best_true = current.clone();
                    let _ = context.shared_incumbent().try_update(&best_true);
                }

                // Track local improvement of the walker (for staleness) under TRUE objective.
                if self
                    .true_acceptor
                    .accept(&last_best_current, current.fitness())
                {
                    last_best_current = current.fitness().clone();
                    improved_this_round = true;
                    stale_rounds = 0;
                }
            }

            if !improved_this_round {
                stale_rounds = stale_rounds.saturating_add(1);

                // Staleness-triggered refetch: sync current; upgrade best_true if snapshot is better
                if self.refetch_after_stale > 0 && stale_rounds >= self.refetch_after_stale {
                    let inc = context.shared_incumbent().peek();
                    if self.true_acceptor.accept(current.fitness(), &inc) {
                        tracing::debug!(
                            "Tabu: staleness refetch after {} rounds ({} -> {})",
                            stale_rounds,
                            current.fitness(),
                            inc
                        );
                        let snap = context.shared_incumbent().snapshot();
                        current = snap.clone();
                        if self
                            .true_acceptor
                            .accept(best_true.fitness(), snap.fitness())
                        {
                            best_true = snap;
                        }
                        last_best_current = current.fitness().clone();
                    }
                    stale_rounds = 0; // reset either way; keep exploring
                }
            }
        }

        // Final publish (no-op if not an improvement)
        let _ = context.shared_incumbent().try_update(&best_true);
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
        // Optional: tune BIG-M for walker ranking if desired
        // .with_big_m_for_walker(2_000_000_000)
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
