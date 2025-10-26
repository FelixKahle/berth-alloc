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

#![allow(dead_code)]

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        gls::{self},
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
use rand::{RngCore, SeedableRng};
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

    // ---- jitter helpers ----

    #[inline]
    fn jitter_i64<R: rand::Rng>(rng: &mut R, base: i64, spread: i64) -> i64 {
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
        if span_i64 <= 0 {
            return base.max(min);
        }
        let off: i64 = rng.random_range(-span_i64..=span_i64);
        let jittered = if off >= 0 {
            base.saturating_add(off as usize)
        } else {
            base.saturating_sub((-off) as usize)
        };
        jittered.max(min)
    }

    #[inline]
    fn jitter_f64<R: rand::Rng>(rng: &mut R, base: f64, pct: f64, floor: f64, ceil: f64) -> f64 {
        let pct = pct.clamp(0.0, 1.0);
        let span = base.abs() * pct;
        let off = rng.random_range(-1_000_000i64..=1_000_000i64) as f64 / 1_000_000.0;
        (base + off * span).clamp(floor, ceil)
    }

    #[inline]
    fn jitter_pair_usize<R: rand::Rng>(
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

    fn sample_gls<'p, R>(
        model: &SolverModel<'p, T>,
        _rng: &mut R,
        slot_idx: usize,
    ) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
    where
        T: SolveNumeric + ToPrimitive + Copy + From<i32>,
        R: rand::Rng,
    {
        let _ = slot_idx;
        let mut s = gls::gls_strategy::<T, ChaCha8Rng>(model);
        s = s
            .with_lambda(10) // Start low
            .with_penalty_step(5) // Smaller penalty step
            .with_decay(gls::DecayMode::Multiplicative {
                num: 990,
                den: 1000,
            }) // Faster decay
            .with_max_penalty(1_000_000_000)
            .with_pulse_params(12, 20) // **More patience before pulsing** (was 8)
            .with_max_local_steps(3000) // **Much more exploitation time** (was 2400)
            .with_refetch_after_stale(40)
            .with_hard_refetch_every(120)
            .with_hard_refetch_mode(gls::HardRefetchMode::IfBetter)
            .with_restart_on_publish(false)
            .with_reset_on_refetch(true)
            .with_kick_steps_on_reset(6) // Smaller kick
            // === THE KEY CHANGE: BACK TO GREEDY ===
            // Force the online control to keep penalties weak.
            // This makes the *true cost* the main driver.
            .with_adaptive_lambda(true)
            .with_target_penalty_share(0.10, 0.22) // **Drastically lower target**
            .with_lambda_step_frac(0.10)
            .with_lambda_bounds(1, 2_000_000); // Lower max bound

        Box::new(s)
    }

    fn sample_ils<'p, R>(
        model: &SolverModel<'p, T>,
        _rng: &mut R,
        slot_idx: usize,
    ) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
    where
        T: SolveNumeric + From<i32>,
        R: rand::Rng,
    {
        let _ = slot_idx;
        let mut s: IteratedLocalSearchStrategy<T, ChaCha8Rng> = ils::ils_strategy(model);
        s = s
            // Local
            .with_local_steps_range(1400..=2000)
            .with_local_sideways(true)
            .with_local_worsening_prob(0.01)
            // Ruin/Repair (smaller static caps; online will widen)
            .with_destroy_attempts(10)
            .with_repair_attempts(24)
            .with_shuffle_local_each_step(true)
            // Refetch cadence
            .with_refetch_after_stale(36)
            .with_hard_refetch_every(12)
            .with_hard_refetch_mode(ils::HardRefetchMode::IfBetter)
            .with_kick_ops_after_refetch(8)
            // Online perturbation controller
            .with_online_perturbation(true)
            .with_destroy_cap_bounds(4, 32)
            .with_repair_cap_bounds(8, 40)
            .with_online_ewma_beta(0.30)
            .with_online_success_band(0.20, 0.46)
            .with_online_cap_update_period(4)
            .with_online_cap_step_max(3)
            .with_online_bias_explore_on_stagnation(true);

        Box::new(s)
    }

    fn sample_sa<'p, R>(
        model: &SolverModel<'p, T>,
        _rng: &mut R,
        slot_idx: usize,
    ) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
    where
        T: SolveNumeric + From<i32>,
        R: rand::Rng,
    {
        let _ = slot_idx;
        let mut s = sa::sa_strategy::<T, ChaCha8Rng>(model);

        // Fixed “slightly hotter, steadier AR control”
        s = s
            .with_init_temp(42.0)
            .with_cooling(0.9996)
            .with_min_temp(1e-4)
            .with_steps_per_epoch(1300)
            .with_hard_refetch_every(90)
            .with_hard_refetch_mode(sa::HardRefetchMode::IfBetter)
            .with_refetch_after_stale(50)
            .with_reheat_factor(0.88)
            .with_kick_ops_after_refetch(18)
            .with_big_m_for_energy(900_000_000)
            // Acceptance ratio targets + online temp knobs
            .with_acceptance_targets(0.26, 0.62)
            .with_ar_nudge_up(1.06)
            .with_ar_extra_cooling(0.996)
            .with_online_temp_blend(0.65, 0.35)
            .with_auto_temperatures(true)
            .with_ewma_beta(0.92);

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

        for i in 0..n {
            let mut rng = ChaCha8Rng::seed_from_u64(seeder.next_u64());
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
        T: SolveNumeric + From<i32> + ToPrimitive + Copy,
    {
        let model: SolverModel<'p, T> = SolverModel::from_problem(problem)?;

        // Opening on main thread
        let initial_state = self
            .opening
            .build(&model)
            .map_err(EngineError::OpeningFailed)?;
        let shared_incumbent = SharedIncumbent::new(initial_state);

        let deadline = Instant::now() + self.config.time_limit;
        let stop = AtomicBool::new(false);

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
