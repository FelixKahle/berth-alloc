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
        sa,
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

        // --- Diversified, randomized strategy portfolio (types + settings) ---
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
                        // Neighbor-wired ILS baseline; we just randomize/tune knobs.
                        let mut s = ils::ils_strategy::<Tnum, ChaCha8Rng>(model);

                        // Local budget: wider, slightly heavier tail for these instance sizes.
                        if rng.random_bool(0.25) {
                            let max_steps = rng.random_range(900..=1700);
                            s = s.with_max_local_steps(max_steps);
                        } else {
                            let lo = rng.random_range(600..=1100);
                            let hi = rng.random_range((lo + 100)..=(lo + 700));
                            s = s.with_local_steps_range(lo..=hi);
                        }

                        // Acceptance tweaks
                        let sideways = rng.random_bool(0.5); // allow plateau moves about half the time
                        s = s.with_local_sideways(sideways);

                        // Small chance to allow random worsening moves in Phase A (diversification)
                        if rng.random_bool(0.40) {
                            let p = rng.random_range(0.02_f64..=0.12);
                            s = s.with_local_worsening_prob(p);
                        }

                        // Shuffle policy (per step vs per round)
                        s = s.with_shuffle_local_each_step(rng.random_bool(0.75));

                        // Cap attempts for perturbation phases (destroy/repair)
                        if rng.random_bool(0.65) {
                            s = s.with_destroy_attempts(rng.random_range(18..=36));
                        }
                        if rng.random_bool(0.65) {
                            s = s.with_repair_attempts(rng.random_range(18..=36));
                        }

                        // Refetch policies (to escape local stagnation)
                        if rng.random_bool(0.55) {
                            s = s.with_refetch_after_stale(rng.random_range(96..=224));
                        }
                        if rng.random_bool(0.30) {
                            s = s.with_hard_refetch_every(rng.random_range(16..=56));
                        }

                        Box::new(s)
                    }

                    Kind::Tabu => {
                        // Neighbor-wired TABU baseline; tune tenure/steps/samples and restart behavior.
                        let lo = rng.random_range(16..=22);
                        let hi = rng.random_range(28..=40).max(lo + 2);
                        let steps = rng.random_range(700..=1600);
                        let samples = rng.random_range(72..=128);
                        let restart_on_publish = rng.random_bool(0.75);
                        let reset_on_refetch = rng.random_bool(0.9);
                        let kick = rng.random_range(2..=6);

                        let mut s = tabu::tabu_strategy::<Tnum, ChaCha8Rng>(model)
                            .with_tabu_tenure(lo..=hi)
                            .with_max_local_steps(steps)
                            .with_samples_per_step(samples)
                            .with_restart_on_publish(restart_on_publish)
                            .with_reset_on_refetch(reset_on_refetch)
                            .with_kick_steps_on_reset(kick);

                        // Periodic refetch can help when tenure is long.
                        if rng.random_bool(0.40) {
                            s = s.with_hard_refetch_every(rng.random_range(18..=54));
                        }
                        // Staleness-based refetch
                        if rng.random_bool(0.50) {
                            s = s.with_refetch_after_stale(rng.random_range(112..=256));
                        }

                        Box::new(s)
                    }

                    Kind::Sa => {
                        // Neighbor-wired SA baseline; tune cooling/temperature/steps and acceptance targets.
                        let init_t = rng.random_range(0.9_f64..=1.6);
                        let cooling = rng.random_range(0.9965_f64..=0.9992);
                        let steps = rng.random_range(220..=440);
                        let low = rng.random_range(0.10_f64..=0.18);
                        let high = rng.random_range(0.45_f64..=0.62).max(low + 0.07);
                        let ema = rng.random_range(0.15_f64..=0.30);
                        let reheat = [0.0, 0.6, 1.0, 8.0].choose(rng).copied().unwrap_or(1.0);
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

                        // Periodic refetch & staleness refetch
                        if rng.random_bool(0.35) {
                            s = s.with_hard_refetch_every(rng.random_range(14..=40));
                        }
                        if rng.random_bool(0.65) {
                            s = s.with_refetch_after_stale(rng.random_range(64..=192));
                        }

                        Box::new(s)
                    }

                    // inside make_one_variant(), in Kind::Gls arm (replace that arm)
                    Kind::Gls => {
                        // Neighbor-wired GLS baseline; randomize lambda/decay/pulse and refetch/reset knobs.
                        let lambda = [4_i64, 5, 6].choose(rng).copied().unwrap_or(5);
                        let step = [1_i64, 2].choose(rng).copied().unwrap_or(1);
                        let max_steps = rng.random_range(1000..=2200);
                        let top_k = rng.random_range(8..=16);
                        let stagnation = rng.random_range(12..=24);
                        let decay = if rng.random_bool(0.85) {
                            // multiplicative decay 88–95%
                            let den = 100u32;
                            let keep = rng.random_range(88u32..=95);
                            gls::DecayMode::Multiplicative { num: keep, den }
                        } else {
                            gls::DecayMode::Subtractive {
                                step: rng.random_range(1_i64..=2),
                            }
                        };
                        let restart_on_publish = rng.random_bool(0.9);
                        let reset_on_refetch = rng.random_bool(0.9);
                        let kick = rng.random_range(2..=6);

                        let mut s = gls::gls_strategy::<Tnum, ChaCha8Rng>(model)
                            .with_lambda(lambda)
                            .with_penalty_step(step)
                            .with_max_local_steps(max_steps)
                            .with_decay(decay)
                            .with_restart_on_publish(restart_on_publish)
                            .with_reset_on_refetch(reset_on_refetch)
                            .with_kick_steps_on_reset(kick)
                            .with_max_penalty(1_000_000_000);

                        s = s.with_pulse_params(stagnation, top_k);

                        if rng.random_bool(0.40) {
                            s = s.with_hard_refetch_every(rng.random_range(12..=40));
                        }
                        if rng.random_bool(0.65) {
                            s = s.with_refetch_after_stale(rng.random_range(96..=224));
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

            // Materialize diversified strategies with randomized settings.
            // Each worker gets its own seed; we derive a sub-seed for the “variant maker”
            // so that creation-time randomization is also independent per worker.
            let mut variant_seed_rng = rand::rng();
            for _ in 0..kinds.len() {
                // Derive a seed to randomize the strategy's internal knobs
                let seed = variant_seed_rng.next_u64();
                let mut knob_rng = ChaCha8Rng::seed_from_u64(seed);
                let k = kinds[self.strategies.len()];
                self.strategies
                    .push(make_one_variant(k, &model, &mut knob_rng));
            }
        }

        let deadline = Instant::now() + self.config.time_limit;
        let stop = AtomicBool::new(false);

        let mut strategies = std::mem::take(&mut self.strategies);

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

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_feasible_two_flex() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // Two small requests that both fit
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn problem_infeasible_two_large() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 12));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // Two requests of length 10 in a 12-length window → only one can fit
        flex.insert(flex_req(1, (0, 12), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 12), &[(1, 10)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    struct DummyStrategy {
        sleep_time: std::time::Duration,
    }

    impl Default for DummyStrategy {
        fn default() -> Self {
            Self {
                sleep_time: std::time::Duration::from_millis(5),
            }
        }
    }

    impl SearchStrategy<i64, ChaCha8Rng> for DummyStrategy {
        fn name(&self) -> &str {
            "DummyStrategy"
        }

        fn run<'e, 'm, 'p>(&mut self, _context: &mut SearchContext<'e, 'm, 'p, i64, ChaCha8Rng>) {
            std::thread::sleep(self.sleep_time);
        }
    }

    fn engine_with_strategies(
        num_workers: usize,
        strategies_count: usize,
        time_ms: u64,
    ) -> SolverEngine<i64> {
        let mut builder = SolverEngineBuilder::<i64>::new()
            .with_worker_count(num_workers)
            .with_time_limit(std::time::Duration::from_millis(time_ms));

        for _ in 0..strategies_count {
            builder = builder.with_strategy(Box::new(DummyStrategy::default()));
        }

        builder.build()
    }

    #[test]
    fn test_solve_returns_solution_when_opening_feasible() {
        let prob = problem_feasible_two_flex();
        let mut engine = engine_with_strategies(2, 2, 50);

        let res = engine.solve(&prob);
        match res {
            Ok(Some(sol)) => {
                // Flexible assignments should be non-empty
                assert!(sol.flexible_assignments().len() == 2);
            }
            other => panic!("expected Some(solution), got: {:?}", other),
        }
    }

    #[test]
    fn test_solve_returns_none_when_opening_infeasible() {
        let prob = problem_infeasible_two_large();
        let mut engine = engine_with_strategies(4, 1, 20);

        let res = engine.solve(&prob);
        match res {
            Ok(None) => { /* infeasible incumbent stays infeasible */ }
            other => panic!("expected Ok(None), got: {:?}", other),
        }
    }
}
