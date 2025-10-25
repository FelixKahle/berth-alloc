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
        gls::{self, DecayMode},
        greedy_opening::GreedyOpening,
        ils::{self, IteratedLocalSearchStrategy},
        opening::OpeningStrategy,
        sa,
        search::{SearchContext, SearchStrategy},
        shared_incumbent::SharedIncumbent,
    },
    model::{err::SolverModelBuildError, solver_model::SolverModel},
    state::solver_state::SolverStateView,
};
use berth_alloc_model::{
    prelude::{Problem, SolutionRef},
    solution::SolutionError,
};
use num_traits::ToPrimitive;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolverEngineConfig {
    pub num_workers: usize,
    pub time_limit: std::time::Duration,
}

impl Default for SolverEngineConfig {
    #[inline]
    fn default() -> Self {
        Self {
            num_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            time_limit: std::time::Duration::from_secs(30),
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

    // ---- helpers: tiny jitter utilities around strong defaults ----

    #[inline]
    fn jitter_i64<R: Rng>(rng: &mut R, base: i64, spread: i64) -> i64 {
        if spread <= 0 {
            base
        } else {
            let off = rng.random_range(-spread..=spread);
            (base + off).max(1)
        }
    }

    #[inline]
    fn jitter_usize<R: rand::Rng>(rng: &mut R, base: usize, pct: f64, min: usize) -> usize {
        let pct = pct.clamp(0.0, 1.0);
        let span_f = (base as f64) * pct;
        let span_i64 = span_f.round() as i64;

        // If span is zero, nothing to jitter—still respect the floor.
        if span_i64 <= 0 {
            return base.max(min);
        }

        // Use i64 for the range since isize doesn't implement SampleUniform on all targets.
        let off: i64 = rng.random_range(-span_i64..=span_i64);
        let jittered = if off >= 0 {
            base.saturating_add(off as usize)
        } else {
            base.saturating_sub((-off) as usize)
        };

        jittered.max(min)
    }

    #[inline]
    fn jitter_f64<R: Rng>(rng: &mut R, base: f64, pct: f64, floor: f64, ceil: f64) -> f64 {
        let pct = pct.clamp(0.0, 1.0);
        let span = base.abs() * pct;
        let off = rng.random_range(-1_000_000i64..=1_000_000i64) as f64 / 1_000_000.0;
        (base + off * span).clamp(floor, ceil)
    }

    #[inline]
    fn jitter_pair_usize<R: Rng>(
        rng: &mut R,
        lo: usize,
        hi: usize,
        pct: f64,
        minw: usize,
    ) -> (usize, usize) {
        let nlo = Self::jitter_usize(rng, lo, pct, 1);
        let nhi = Self::jitter_usize(rng, hi, pct, nlo + minw);
        (nlo, nhi.max(nlo + minw))
    }

    // ---- per-family samplers (wrap the good factories and nudge params) ----

    fn sample_gls<'p, R>(
        model: &SolverModel<'p, T>,
        rng: &mut R,
        slot_idx: usize,
    ) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
    where
        T: SolveNumeric + ToPrimitive + Copy + From<i32>,
        R: rand::Rng,
    {
        let mut s = gls::gls_strategy::<T, ChaCha8Rng>(model);

        // diversify: lambda, penalty_step, pulse, max steps, refetch cadence, kicks
        let lambda = Self::jitter_i64(rng, 500, 10);
        let pstep = Self::jitter_i64(rng, 10, 1);
        let (pulse_stale, pulse_k) = {
            let ps = Self::jitter_usize(rng, 8, 0.35, 2);
            let pk = Self::jitter_usize(rng, 20, 0.35, 5);
            (ps, pk)
        };
        let max_steps = Self::jitter_usize(rng, 2100, 0.25, 600);
        let refetch_every = [0, 16, 20, 24, 28, 32][slot_idx % 6];
        let refetch_after_stale = [0, 40, 60][slot_idx % 3];
        let kicks = Self::jitter_usize(rng, 6, 0.5, 2);

        s = s
            .with_lambda(lambda)
            .with_penalty_step(pstep)
            .with_pulse_params(pulse_stale, pulse_k)
            .with_max_local_steps(max_steps)
            .with_hard_refetch_every(refetch_every)
            .with_decay(DecayMode::Multiplicative { num: 99, den: 100 })
            .with_refetch_after_stale(refetch_after_stale)
            .with_kick_steps_on_reset(kicks);

        Box::new(s)
    }

    fn sample_ils<'p, R>(
        model: &SolverModel<'p, T>,
        rng: &mut R,
        slot_idx: usize,
    ) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
    where
        T: SolveNumeric + From<i32>,
        R: Rng,
    {
        let mut s: IteratedLocalSearchStrategy<T, ChaCha8Rng> = ils::ils_strategy(model);

        // diversify around your “standard but bolder” preset
        let (lo, hi) = Self::jitter_pair_usize(rng, 1200, 2200, 0.25, 100);
        let destroy_attempts = Self::jitter_usize(rng, 12, 0.34, 3);
        let repair_attempts = Self::jitter_usize(rng, 28, 0.34, 6);
        let refetch_after_stale = [32, 40, 48, 56][slot_idx % 4];
        let refetch_every = [10, 12, 14, 16][slot_idx % 4];
        let kicks = Self::jitter_usize(rng, 8, 0.5, 2);

        s = s
            .with_local_steps_range(lo..=hi)
            .with_destroy_attempts(destroy_attempts)
            .with_repair_attempts(repair_attempts)
            .with_refetch_after_stale(refetch_after_stale)
            .with_hard_refetch_every(refetch_every)
            .with_kick_ops_after_refetch(kicks);

        Box::new(s)
    }

    fn sample_sa<'p, R>(
        model: &SolverModel<'p, T>,
        rng: &mut R,
        slot_idx: usize,
    ) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
    where
        T: SolveNumeric + From<i32>,
        R: rand::Rng,
    {
        let mut s = sa::sa_strategy::<T, ChaCha8Rng>(model);

        // diversify schedule + bandit + refetch cadence
        let t0 = Self::jitter_f64(rng, 35.0, 0.35, 3.0, 80.0);
        let cool = Self::jitter_f64(rng, 0.9997, 0.0006, 0.9985, 0.99995);
        let steps = Self::jitter_usize(rng, 1200, 0.35, 300);
        let refetch_every = [60, 70, 80, 100, 120][slot_idx % 5];
        let stale_epochs = [40, 50, 60, 80][slot_idx % 4];
        let reheat = Self::jitter_f64(rng, 0.85, 0.25, 0.4, 0.95);
        let big_m = Self::jitter_i64(rng, 900_000, 150_000_000);
        let alpha = Self::jitter_f64(rng, 0.30, 0.3, 0.05, 0.6);
        let minw = Self::jitter_f64(rng, 0.12, 0.5, 0.01, 0.3);
        let low = Self::jitter_f64(rng, 0.22, 0.35, 0.05, 0.5);
        let high = (low + 0.25).clamp(0.2, 0.85);
        let kicks = Self::jitter_usize(rng, 18, 0.5, 4);

        s = s
            .with_init_temp(t0)
            .with_cooling(cool)
            .with_steps_per_epoch(steps)
            .with_hard_refetch_every(refetch_every)
            .with_refetch_after_stale(stale_epochs)
            .with_reheat_factor(reheat)
            .with_big_m_for_energy(big_m)
            .with_op_ema_alpha(alpha)
            .with_op_min_weight(minw)
            .with_acceptance_targets(low, high)
            .with_kick_ops_after_refetch(kicks);

        Box::new(s)
    }

    fn diversified_strategies<'p>(
        model: &SolverModel<'p, T>,
        n: usize,
        seeder: &mut impl RngCore,
    ) -> Vec<Box<dyn SearchStrategy<T, ChaCha8Rng>>>
    where
        T: SolveNumeric + From<i32> + ToPrimitive + Copy,
    {
        let n = n.max(1);
        let mut out: Vec<Box<dyn SearchStrategy<T, ChaCha8Rng>>> = Vec::with_capacity(n);

        // one RNG per sampler for stability
        for i in 0..n {
            let mut rng = ChaCha8Rng::seed_from_u64(seeder.next_u64());
            // round-robin families, skew a bit towards exploration on later threads
            match i % 3 {
                0 => out.push(Self::sample_gls(model, &mut rng, i)),
                1 => out.push(Self::sample_ils(model, &mut rng, i)),
                _ => out.push(Self::sample_sa(model, &mut rng, i)),
            }
        }
        out
    }

    #[tracing::instrument(level = "info", skip(self, problem))]
    pub fn solve<'p>(
        &mut self,
        problem: &'p Problem<T>,
    ) -> Result<Option<SolutionRef<'p, T>>, EngineError<T, GreedyOpening<T>>>
    where
        // needed for the GLS factory and our diversification
        T: SolveNumeric + From<i32> + ToPrimitive + Copy,
    {
        let model: SolverModel<'p, T> = SolverModel::from_problem(problem)?;

        // Opening on the main thread
        let initial_state = self
            .opening
            .build(&model)
            .map_err(EngineError::OpeningFailed)?;
        let shared_incumbent = SharedIncumbent::new(initial_state);

        let deadline = Instant::now() + self.config.time_limit;
        let stop = AtomicBool::new(false);

        // If the caller didn't provide strategies, synthesize a diversified portfolio sized to workers.
        let mut strategies = std::mem::take(&mut self.strategies);
        if strategies.is_empty() {
            let mut seed_picker = rand::rng();
            strategies = Self::diversified_strategies(
                &model,
                self.config.num_workers.max(1),
                &mut seed_picker,
            );
        }

        let inc_ref = &shared_incumbent;
        let stop_ref = &stop;
        let problem_ref = problem;
        let model_ref = &model;

        std::thread::scope(|scope| {
            let mut seeder = rand::rng();

            scope.spawn(move || {
                let now = Instant::now();
                if deadline > now {
                    std::thread::sleep(deadline - now);
                }
                stop_ref.store(true, Ordering::SeqCst);
            });

            let max_workers = self.config.num_workers.max(1);
            let take_n = max_workers.min(strategies.len());

            for mut strategy in strategies.drain(..take_n) {
                let worker_seed: u64 = seeder.next_u64();

                scope.spawn(move || {
                    let rng = ChaCha8Rng::seed_from_u64(worker_seed);
                    let mut context =
                        SearchContext::new(problem_ref, model_ref, inc_ref, stop_ref, rng);
                    strategy.run(&mut context);
                });
            }
        });

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

    pub fn with_time_limit(mut self, time_limit: std::time::Duration) -> Self {
        self.config.time_limit = time_limit;
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
