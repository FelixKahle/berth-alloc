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
        gls,
        greedy_opening::GreedyOpening,
        ils,
        opening::OpeningStrategy,
        popmusic, sa,
        search::{SearchContext, SearchStrategy},
        shared_incumbent::SharedIncumbent,
        tabu,
    },
    model::{err::SolverModelBuildError, solver_model::SolverModel},
    state::solver_state::SolverStateView,
};
use berth_alloc_model::{
    prelude::{Problem, SolutionRef},
    solution::SolutionError,
};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolverEngineConfig {
    pub num_workers: usize,
    /// Legacy single time limit (kept for backward compatibility with builder method).
    /// Not used directly by the engine phases anymore.
    pub time_limit: std::time::Duration,
    /// Heuristics phase time budget
    pub heuristics_time: std::time::Duration,
    /// Finishing POPMUSIC phase time budget
    pub finishing_time: std::time::Duration,
}

impl Default for SolverEngineConfig {
    #[inline]
    fn default() -> Self {
        Self {
            num_workers: 5,
            // Keep as 60s for legacy callers, but phases use fields below
            time_limit: std::time::Duration::from_secs(60),
            // Defaults per request: 30s heuristics + 30s popmusic
            heuristics_time: std::time::Duration::from_secs(30),
            finishing_time: std::time::Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
pub enum EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    SolverModel(SolverModelBuildError),
    Solution(SolutionError),
    NoStrategies,
    OpeningFailed(S::Error),
}

impl<T, S> From<SolutionError> for EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    #[inline]
    fn from(err: SolutionError) -> Self {
        EngineError::Solution(err)
    }
}

impl<T, S> std::fmt::Display for EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Solution(err) => write!(f, "Solution error: {}", err),
            EngineError::NoStrategies => write!(f, "No search strategies provided"),
            EngineError::OpeningFailed(err) => write!(f, "Opening strategy failed: {:?}", err),
            EngineError::SolverModel(err) => write!(f, "Solver model build error: {}", err),
        }
    }
}

impl<T, S> std::error::Error for EngineError<T, S>
where
    T: Copy + Ord + std::fmt::Debug + std::fmt::Display,
    S: OpeningStrategy<T> + std::fmt::Debug,
{
}

impl<T, S> From<SolverModelBuildError> for EngineError<T, S>
where
    T: Copy + Ord,
    S: OpeningStrategy<T>,
{
    #[inline]
    fn from(err: SolverModelBuildError) -> Self {
        EngineError::SolverModel(err)
    }
}

pub struct SolverEngine<T>
where
    T: SolveNumeric,
{
    config: SolverEngineConfig,
    strategies: Vec<Box<dyn SearchStrategy<T, ChaCha8Rng>>>,
    opening: GreedyOpening<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> SolverEngine<T>
where
    T: SolveNumeric,
{
    pub fn new(
        config: SolverEngineConfig,
        strategies: Vec<Box<dyn SearchStrategy<T, ChaCha8Rng>>>,
    ) -> Self {
        Self {
            config,
            strategies,
            opening: GreedyOpening::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    #[tracing::instrument(level = "info", skip(self, problem))]
    pub fn solve<'p>(
        &mut self,
        problem: &'p Problem<T>,
    ) -> Result<Option<SolutionRef<'p, T>>, EngineError<T, GreedyOpening<T>>> {
        let model: SolverModel<'p, T> = SolverModel::from_problem(problem)?;

        // Opening on the main thread
        let initial_state = self
            .opening
            .build(&model)
            .map_err(EngineError::OpeningFailed)?;
        let shared_incumbent = SharedIncumbent::new(initial_state);

        // --- Diversified, randomized strategy portfolio (heuristics only) ---
        if self.strategies.is_empty() {
            let n = self.config.num_workers.max(1);

            #[derive(Clone, Copy, Debug)]
            enum Kind {
                Ils,
                Tabu,
                Sa,
                Gls,
            }

            // Make one of a given kind with randomized knobs.
            fn make_one_variant<Tnum>(
                k: Kind,
                model: &SolverModel<Tnum>,
                rng: &mut ChaCha8Rng,
            ) -> Box<dyn SearchStrategy<Tnum, ChaCha8Rng>>
            where
                Tnum: SolveNumeric + From<i32>,
            {
                match k {
                    Kind::Ils => {
                        let mut s = ils::ils_strategy::<Tnum, ChaCha8Rng>(model);

                        if rng.random_bool(0.35) {
                            s = s.with_max_local_steps(rng.random_range(1200..=2200));
                        } else {
                            let lo = rng.random_range(900..=1500);
                            let hi = rng.random_range((lo + 200)..=(lo + 900));
                            s = s.with_local_steps_range(lo..=hi);
                        }

                        s = s.with_local_sideways(rng.random_bool(0.6));
                        if rng.random_bool(0.7) {
                            s = s.with_local_worsening_prob(rng.random_range(0.06..=0.12));
                        }

                        s = s.with_shuffle_local_each_step(true);

                        if rng.random_bool(0.75) {
                            s = s.with_destroy_attempts(rng.random_range(24..=48));
                        }
                        if rng.random_bool(0.75) {
                            s = s.with_repair_attempts(rng.random_range(24..=48));
                        }

                        if rng.random_bool(0.65) {
                            s = s.with_refetch_after_stale(rng.random_range(64..=160));
                        }
                        if rng.random_bool(0.50) {
                            s = s.with_hard_refetch_every(rng.random_range(16..=48));
                        }

                        Box::new(s)
                    }

                    Kind::Tabu => {
                        let lo = rng.random_range(24..=32);
                        let hi = rng.random_range(40..=64).max(lo + 2);
                        let steps = rng.random_range(900..=2200);
                        let samples = rng.random_range(96..=160);
                        let restart_on_publish = rng.random_bool(0.8);
                        let reset_on_refetch = rng.random_bool(0.9);
                        let kick = rng.random_range(3..=6);

                        let mut s = tabu::tabu_strategy::<Tnum, ChaCha8Rng>(model)
                            .with_tabu_tenure(lo..=hi)
                            .with_max_local_steps(steps)
                            .with_samples_per_step(samples)
                            .with_restart_on_publish(restart_on_publish)
                            .with_reset_on_refetch(reset_on_refetch)
                            .with_kick_steps_on_reset(kick);

                        if rng.random_bool(0.55) {
                            s = s.with_hard_refetch_every(rng.random_range(18..=40));
                        }
                        if rng.random_bool(0.60) {
                            s = s.with_refetch_after_stale(rng.random_range(80..=160));
                        }

                        Box::new(s)
                    }

                    Kind::Sa => {
                        let init_t = rng.random_range(0.9_f64..=1.6);
                        let cooling = rng.random_range(0.9985_f64..=0.9993);
                        let steps = rng.random_range(300..=600);
                        let low = rng.random_range(0.12_f64..=0.18);
                        let high = rng.random_range(0.55_f64..=0.65).max(low + 0.06);
                        let ema = rng.random_range(0.20_f64..=0.35);
                        let reheat = [0.0, 1.0, 8.0].choose(rng).copied().unwrap_or(1.0);
                        let big_m = [1_000_000_000_i64, 1_250_000_000, 1_500_000_000]
                            .choose(rng)
                            .copied()
                            .unwrap_or(1_250_000_000);

                        let mut s = sa::sa_strategy::<Tnum, ChaCha8Rng>(model)
                            .with_init_temp(init_t)
                            .with_cooling(cooling)
                            .with_steps_per_temp(steps)
                            .with_acceptance_targets(low, high)
                            .with_op_ema_alpha(ema)
                            .with_reheat_factor(reheat)
                            .with_big_m_for_energy(big_m);

                        if rng.random_bool(0.45) {
                            s = s.with_hard_refetch_every(rng.random_range(14..=36));
                        }
                        if rng.random_bool(0.70) {
                            s = s.with_refetch_after_stale(rng.random_range(64..=160));
                        }

                        Box::new(s)
                    }

                    Kind::Gls => {
                        let lambda = [5_i64, 6, 7, 8].choose(rng).copied().unwrap_or(6);
                        let step = [2_i64, 3].choose(rng).copied().unwrap_or(2);
                        let max_steps = rng.random_range(1400..=2200);
                        let top_k = rng.random_range(12..=24);
                        let stagnation = rng.random_range(8..=16);
                        let decay = gls::DecayMode::Multiplicative {
                            num: rng.random_range(92u32..=97),
                            den: 100,
                        };
                        let restart_on_publish = rng.random_bool(0.9);
                        let reset_on_refetch = rng.random_bool(0.9);
                        let kick = rng.random_range(3..=6);

                        let mut s = gls::gls_strategy::<Tnum, ChaCha8Rng>(model)
                            .with_lambda(lambda)
                            .with_penalty_step(step)
                            .with_max_local_steps(max_steps)
                            .with_decay(decay)
                            .with_restart_on_publish(restart_on_publish)
                            .with_reset_on_refetch(reset_on_refetch)
                            .with_kick_steps_on_reset(kick)
                            .with_max_penalty(1_000_000_000)
                            .with_pulse_params(stagnation, top_k);

                        if rng.random_bool(0.50) {
                            s = s.with_hard_refetch_every(rng.random_range(12..=36));
                        }
                        if rng.random_bool(0.70) {
                            s = s.with_refetch_after_stale(rng.random_range(64..=160));
                        }

                        Box::new(s)
                    }
                }
            }

            // Start with guaranteed coverage (one of each, up to n).
            let mut kinds: Vec<Kind> = Vec::with_capacity(n);
            let all = [Kind::Ils, Kind::Tabu, Kind::Sa, Kind::Gls];
            for k in all.into_iter().take(n) {
                kinds.push(k);
            }

            // Fill remaining slots from a weighted bag, then shuffle.
            let weights = &[
                (Kind::Ils, 2usize),
                (Kind::Tabu, 1usize),
                (Kind::Sa, 1usize),
                (Kind::Gls, 3usize),
            ];
            let mut bag: Vec<Kind> = Vec::new();
            for (k, w) in weights {
                for _ in 0..*w {
                    bag.push(*k);
                }
            }

            // Use a deterministic per-run RNG for portfolio construction
            let mut pf_rng = rand::rng();
            while kinds.len() < n {
                if let Some(&k) = bag.choose(&mut pf_rng) {
                    kinds.push(k);
                } else {
                    kinds.push(Kind::Ils);
                }
            }
            kinds.shuffle(&mut pf_rng);

            // Materialize diversified strategies with randomized settings
            let mut variant_seed_rng = rand::rng();
            for i in 0..kinds.len() {
                // Derive a seed to randomize the strategy's internal knobs
                let seed = variant_seed_rng.next_u64();
                let mut knob_rng = ChaCha8Rng::seed_from_u64(seed);
                let k = kinds[i];
                self.strategies
                    .push(make_one_variant(k, &model, &mut knob_rng));
            }
        }

        // Prepare shared references for worker phases
        let inc_ref = &shared_incumbent;
        let problem_ref = problem;
        let model_ref = &model;

        // Phase 1: Heuristics only, under heuristics_time budget
        let heur_deadline = Instant::now() + self.config.heuristics_time;
        let heur_stop = Arc::new(AtomicBool::new(false));

        {
            std::thread::scope(|scope| {
                // Watchdog to stop the heuristic workers at deadline
                scope.spawn({
                    let stop = heur_stop.clone();
                    move || {
                        let now = Instant::now();
                        if heur_deadline > now {
                            std::thread::sleep(heur_deadline - now);
                        }
                        stop.store(true, Ordering::SeqCst);
                    }
                });

                // Run heuristic strategies in parallel
                let mut strategies = std::mem::take(&mut self.strategies);
                let max_workers = self.config.num_workers.max(1);
                let take_n = max_workers.min(strategies.len());

                let mut seeder = rand::rng();
                for mut strategy in strategies.drain(..take_n) {
                    let worker_seed: u64 = seeder.next_u64();
                    let stop = heur_stop.clone();

                    scope.spawn(move || {
                        let rng = ChaCha8Rng::seed_from_u64(worker_seed);
                        let mut context =
                            SearchContext::new(problem_ref, model_ref, inc_ref, &*stop, rng);
                        strategy.run(&mut context);
                    });
                }
            });
        }

        // Phase 2: Finishing POPMUSIC, single-threaded, under finishing_time budget
        if self.config.finishing_time > std::time::Duration::from_millis(0) {
            let fini_deadline = Instant::now() + self.config.finishing_time;
            let fini_stop = Arc::new(AtomicBool::new(false));

            std::thread::scope(|scope| {
                // Watchdog for finishing phase
                scope.spawn({
                    let stop = fini_stop.clone();
                    move || {
                        let now = Instant::now();
                        if fini_deadline > now {
                            std::thread::sleep(fini_deadline - now);
                        }
                        stop.store(true, Ordering::SeqCst);
                    }
                });

                // Run exactly one POPMUSIC strategy
                let seed = rand::rng().next_u64();
                let rng = ChaCha8Rng::seed_from_u64(seed);
                let mut context =
                    SearchContext::new(problem_ref, model_ref, inc_ref, &*fini_stop, rng);
                let mut pop = popmusic::PopmusicStrategy::<T, ChaCha8Rng>::new(
                    popmusic::PopmusicParams::default(),
                );
                pop.run(&mut context);
            });
        }

        let best = shared_incumbent.snapshot();
        if best.is_feasible() {
            let sol: SolutionRef<'p, T> = best.into_solution(&model)?;
            Ok(Some(sol))
        } else {
            Ok(None)
        }
    }
}

pub struct SolverEngineBuilder<T>
where
    T: Copy + Ord,
{
    config: SolverEngineConfig,
    strategies: Vec<Box<dyn SearchStrategy<T, ChaCha8Rng>>>,
}

impl<T> Default for SolverEngineBuilder<T>
where
    T: SolveNumeric,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SolverEngineBuilder<T>
where
    T: SolveNumeric,
{
    pub fn new() -> Self {
        Self {
            config: SolverEngineConfig::default(),
            strategies: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: SolverEngineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_worker_count(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }

    /// Explicitly set heuristics phase time.
    pub fn with_heuristics_time(mut self, time: std::time::Duration) -> Self {
        self.config.heuristics_time = time;
        self
    }

    /// Explicitly set finishing POPMUSIC phase time.
    pub fn with_finishing_time(mut self, time: std::time::Duration) -> Self {
        self.config.finishing_time = time;
        self
    }

    pub fn with_strategy(mut self, strategy: Box<dyn SearchStrategy<T, ChaCha8Rng>>) -> Self {
        self.strategies.push(strategy);
        self
    }

    pub fn build(self) -> SolverEngine<T> {
        SolverEngine::new(self.config, self.strategies)
    }
}
