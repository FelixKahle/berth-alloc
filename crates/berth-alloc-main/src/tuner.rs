// Copyright (c) 2025 Felix Kahle.
// MIT License

//! Random-search parameter tuner for ILS / SA / GLS.
//!
//! - Only the *first* instance under `instances/` is used.
//! - One engine per trial, one worker per engine, exactly one strategy.
//! - ILS/SA/GLS each run on their own thread group (coordinator + workers).
//! - Each solve has ~80s time limit (override with env TUNER_SOLVE_SECS).
//! - Control parallelism with env TUNER_THREADS_PER_STRATEGY (default 3).
//! - Control run length with env TUNER_TRIALS_PER_WORKER (default 100000).
//! - Every trial is appended to JSONL per strategy.
//! - A live summary (best/leaderboard aggregates) is continuously updated.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_return)]

use berth_alloc_model::prelude::Problem;
use berth_alloc_model::problem::loader::ProblemLoader;
use berth_alloc_model::solution::SolutionView;
use berth_alloc_solver::{
    core::numeric::SolveNumeric,
    engine::{
        search::SearchStrategy,
        solver_engine::{SolverEngine, SolverEngineBuilder},
    },
    model::solver_model::SolverModel,
};
use chrono::{DateTime, Utc};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{
    env,
    ffi::OsStr,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

/* ------------------------- configuration ------------------------- */

fn cfg_usize(env_key: &str, default: usize) -> usize {
    env::var(env_key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}
fn cfg_u64(env_key: &str, default: u64) -> u64 {
    env::var(env_key)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}
fn cfg_secs(env_key: &str, default: u64) -> u64 {
    env::var(env_key)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

fn solve_time_limit() -> Duration {
    Duration::from_secs(cfg_secs("TUNER_SOLVE_SECS", 80))
}
fn threads_per_strategy() -> usize {
    cfg_usize("TUNER_THREADS_PER_STRATEGY", 3)
}
fn trials_per_worker() -> usize {
    cfg_usize("TUNER_TRIALS_PER_WORKER", 100_000)
}

/* ------------------------- logging ------------------------- */

fn enable_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(FmtSpan::CLOSE)
        .try_init();
}

/* ------------------------- instance loading ------------------------- */

fn find_instances_dir() -> Option<PathBuf> {
    // Start from this crate and walk up to find `instances/`
    let mut cur: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(p) = cur {
        let cand = p.join("instances");
        if cand.is_dir() {
            return Some(cand);
        }
        cur = p.parent();
    }
    None
}

fn load_first_instance() -> (Problem<i64>, String) {
    let inst_dir = find_instances_dir()
        .expect("Could not find an `instances/` directory in any ancestor of CARGO_MANIFEST_DIR");

    let mut files: Vec<PathBuf> = fs::read_dir(&inst_dir)
        .expect("read_dir(instances) failed")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension() == Some(OsStr::new("txt")))
        .collect();

    files.sort();

    let first = files
        .into_iter()
        .next()
        .expect("instances/ has no *.txt instance files");

    let loader = ProblemLoader::default();
    let problem = loader
        .from_path(&first)
        .unwrap_or_else(|e| panic!("Failed to load {:?}: {}", first, e));
    let name = first
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown.txt")
        .to_string();

    (problem, name)
}

/* ------------------------- strategy kinds ------------------------- */

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StrategyKind {
    Ils,
    Sa,
    Gls,
}

impl StrategyKind {
    fn jsonl_basename(self) -> &'static str {
        match self {
            StrategyKind::Ils => "ils.jsonl",
            StrategyKind::Sa => "sa.jsonl",
            StrategyKind::Gls => "gls.jsonl",
        }
    }
    fn summary_basename(self) -> &'static str {
        match self {
            StrategyKind::Ils => "ils_summary.json",
            StrategyKind::Sa => "sa_summary.json",
            StrategyKind::Gls => "gls_summary.json",
        }
    }
}

/* ------------------------- params ------------------------- */

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IlsParams {
    local_lo: usize,
    local_hi: usize,
    allow_sideways: bool,
    destroy_attempts: usize,
    repair_attempts: usize,
    refetch_after_stale: usize,
    hard_refetch_every: usize,
    kick_after_refetch: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SaParams {
    init_temp: f64,
    cooling: f64,
    min_temp: f64,
    steps_per_epoch: usize,
    refetch_after_stale: usize,
    hard_refetch_every: usize,
    reheat_factor: f64,
    kick_after_refetch: usize,
    big_m: i64,
    ar_low: f64,
    ar_high: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GlsParams {
    lambda: i64,
    penalty_step: i64,
    max_local_steps: usize,
    stagnation_before_pulse: usize,
    pulse_top_k: usize,
    hard_refetch_every: usize,
    kick_steps_on_reset: usize,
}

/* ------------------------- random samplers ------------------------- */

fn sample_ils_params<R: rand::Rng>(rng: &mut R) -> IlsParams {
    IlsParams {
        local_lo: rng.random_range(900..=1500),
        local_hi: rng.random_range(1600..=2600),
        allow_sideways: rng.random::<bool>(),
        destroy_attempts: rng.random_range(8..=18),
        repair_attempts: rng.random_range(18..=36),
        refetch_after_stale: rng.random_range(24..=80),
        hard_refetch_every: rng.random_range(8..=32),
        kick_after_refetch: rng.random_range(4..=16),
    }
}

fn sample_sa_params<R: rand::Rng>(rng: &mut R) -> SaParams {
    SaParams {
        init_temp: rng.random_range(10.0..=60.0),
        cooling: rng.random_range(0.9992..=0.99985),
        min_temp: 1e-4,
        steps_per_epoch: rng.random_range(800..=1800),
        refetch_after_stale: rng.random_range(30..=100),
        hard_refetch_every: rng.random_range(50..=120),
        reheat_factor: rng.random_range(0.5..=0.9),
        kick_after_refetch: rng.random_range(6..=22),
        big_m: rng.random_range(600_000_000i64..=1_200_000_000i64),
        ar_low: rng.random_range(0.10..=0.30),
        ar_high: rng.random_range(0.55..=0.80),
    }
}

fn sample_gls_params<R: rand::Rng>(rng: &mut R) -> GlsParams {
    GlsParams {
        lambda: rng.random_range(5..=16),
        penalty_step: rng.random_range(1..=4),
        max_local_steps: rng.random_range(800..=2400),
        stagnation_before_pulse: rng.random_range(6..=18),
        pulse_top_k: rng.random_range(12..=28),
        hard_refetch_every: rng.random_range(10..=36),
        kick_steps_on_reset: rng.random_range(3..=12),
    }
}

/* ------------------------- result records ------------------------- */

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrialResult {
    strategy: StrategyKind,
    params: serde_json::Value,
    seed: u64,
    filename: String,
    start_ts: DateTime<Utc>,
    end_ts: DateTime<Utc>,
    runtime_ms: u128,
    cost: Option<i64>,
}

/* ------------------------- IO helpers ------------------------- */

const OUT_DIR: &str = "tuning_results";

fn ensure_out_dir() {
    let _ = fs::create_dir_all(OUT_DIR);
}

fn jsonl_path(kind: StrategyKind) -> PathBuf {
    PathBuf::from(OUT_DIR).join(kind.jsonl_basename())
}

fn summary_path(kind: StrategyKind) -> PathBuf {
    PathBuf::from(OUT_DIR).join(kind.summary_basename())
}

fn append_jsonl(kind: StrategyKind, res: &TrialResult) {
    ensure_out_dir();
    let line = serde_json::to_string(res).expect("serialize TrialResult");
    let path = jsonl_path(kind);
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("open {} failed: {}", path.display(), e));
    let _ = writeln!(f, "{}", line);
}

/* ------------------------- strategy builders (boxed with ChaCha8Rng) ------------------------- */

fn ils_strategy_with_params<T>(
    model: &SolverModel<T>,
    p: &IlsParams,
) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
where
    T: SolveNumeric + From<i32>,
{
    use berth_alloc_solver::engine::ils::{HardRefetchMode, IteratedLocalSearchStrategy};
    use berth_alloc_solver::engine::neighbors;
    use berth_alloc_solver::search::operator_library::{
        destroy::{RandomKRatioDestroy, ShawRelatedDestroy, TimeClusterDestroy, WorstCostDestroy},
        local::{
            CascadeInsertPolicy, CascadeRelocateK, CrossExchangeAcrossBerths,
            HillClimbRelocateBest, OrOptBlockRelocate, RelocateSingleBest, ShiftEarlierOnSameBerth,
            SwapPairSameBerth,
        },
        repair::{GreedyInsertion, KRegretInsertion},
    };

    let pm = model.proximity_map();
    let n_any = neighbors::any(pm);
    let n_dir = neighbors::direct_competitors(pm);
    let n_same = neighbors::same_berth(pm);

    let strat = IteratedLocalSearchStrategy::new()
        .with_local_steps_range(p.local_lo..=p.local_hi)
        .with_local_sideways(p.allow_sideways)
        .with_local_worsening_prob(0.0)
        .with_destroy_attempts(p.destroy_attempts)
        .with_repair_attempts(p.repair_attempts)
        .with_refetch_after_stale(p.refetch_after_stale)
        .with_hard_refetch_every(p.hard_refetch_every)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_kick_ops_after_refetch(p.kick_after_refetch)
        // locals
        .with_local_op(Box::new(
            RelocateSingleBest::new(24..=64).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            SwapPairSameBerth::new(40..=100).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeAcrossBerths::new(48..=120).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbRelocateBest::new(24..=72).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            OrOptBlockRelocate::new(4..=8, 1.25..=1.65).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            ShiftEarlierOnSameBerth::new(16..=48).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            CascadeRelocateK::new(3..=4, 8..=12, 12..=24)
                .with_neighbors(n_dir.clone())
                .with_insert_policy(CascadeInsertPolicy::BestEarliest),
        ))
        // destroy
        .with_destroy_op(Box::new(
            RandomKRatioDestroy::new(0.32..=0.58).with_neighbors(n_any.clone()),
        ))
        .with_destroy_op(Box::new(
            WorstCostDestroy::new(0.30..=0.48).with_neighbors(n_dir.clone()),
        ))
        .with_destroy_op(Box::new(
            ShawRelatedDestroy::new(0.28..=0.40, 1.6..=2.2, 1.into(), 1.into(), 5.into())
                .with_neighbors(n_dir.clone()),
        ))
        .with_destroy_op(Box::new(
            TimeClusterDestroy::<T>::new(
                0.32..=0.50,
                berth_alloc_core::prelude::TimeDelta::new(24.into()),
            )
            .with_alpha(1.55..=1.90)
            .with_neighbors(n_any),
        ))
        // repair
        .with_repair_op(Box::new(KRegretInsertion::new(8..=11)))
        .with_repair_op(Box::new(GreedyInsertion));

    Box::new(strat)
}

fn sa_strategy_with_params<T>(
    model: &SolverModel<T>,
    p: &SaParams,
) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
where
    T: SolveNumeric + From<i32>,
{
    use berth_alloc_solver::engine::neighbors;
    use berth_alloc_solver::engine::sa::{HardRefetchMode, SimulatedAnnealingStrategy};
    use berth_alloc_solver::search::operator_library::local::*;
    let pm = model.proximity_map();
    let n_any = neighbors::any(pm);
    let n_dir = neighbors::direct_competitors(pm);
    let n_same = neighbors::same_berth(pm);

    let strat = SimulatedAnnealingStrategy::new()
        .with_init_temp(p.init_temp)
        .with_cooling(p.cooling)
        .with_min_temp(p.min_temp)
        .with_steps_per_epoch(p.steps_per_epoch)
        .with_hard_refetch_every(p.hard_refetch_every)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_refetch_after_stale(p.refetch_after_stale)
        .with_reheat_factor(p.reheat_factor)
        .with_kick_ops_after_refetch(p.kick_after_refetch)
        .with_big_m_for_energy(p.big_m)
        .with_acceptance_targets(p.ar_low, p.ar_high)
        // locals (your SA preset set)
        .with_local_op(Box::new(
            ShiftEarlierOnSameBerth::new(18..=52).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            RelocateSingleBest::new(20..=64).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            SwapPairSameBerth::new(36..=96).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeAcrossBerths::new(48..=128).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            OrOptBlockRelocate::new(5..=9, 1.4..=1.9).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            RelocateSingleBestAllowWorsening::new(12..=24).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            RandomRelocateAnywhere::new(12..=24).with_neighbors(n_any.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbRelocateBest::new(24..=72).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            HillClimbBestSwapSameBerth::new(48..=120).with_neighbors(n_same.clone()),
        ))
        .with_local_op(Box::new(
            RandomizedGreedyRelocateRcl::new(18..=48, 1.5..=2.2).with_neighbors(n_dir.clone()),
        ))
        .with_local_op(Box::new(
            CrossExchangeBestAcrossBerths::new(32..=96).with_neighbors(n_any.clone()),
        ))
        .with_local_op(Box::new(
            CascadeRelocateK::new(3..=5, 6..=10, 10..=20)
                .with_neighbors(n_any)
                .with_insert_policy(CascadeInsertPolicy::Rcl {
                    alpha_range: 1.4..=2.0,
                })
                .allow_step_worsening(30),
        ));

    Box::new(strat)
}

fn gls_strategy_with_params<T>(
    model: &SolverModel<T>,
    p: &GlsParams,
) -> Box<dyn SearchStrategy<T, ChaCha8Rng>>
where
    T: SolveNumeric + From<i32>,
{
    use berth_alloc_solver::engine::gls::gls_strategy;
    let s = gls_strategy::<T, ChaCha8Rng>(model)
        .with_lambda(p.lambda)
        .with_penalty_step(p.penalty_step)
        .with_max_local_steps(p.max_local_steps)
        .with_pulse_params(p.stagnation_before_pulse, p.pulse_top_k)
        .with_hard_refetch_every(p.hard_refetch_every)
        .with_kick_steps_on_reset(p.kick_steps_on_reset);
    Box::new(s)
}

/* ------------------------- engine builder ------------------------- */

fn build_engine_with_single_strategy<T>(
    strategy: Box<dyn SearchStrategy<T, ChaCha8Rng>>,
    time_limit: Duration,
) -> SolverEngine<T>
where
    T: SolveNumeric + Send + Sync + 'static,
{
    SolverEngineBuilder::<T>::default()
        .with_worker_count(1)
        .with_time_limit(time_limit)
        .with_strategy(strategy)
        .build()
}

/* ------------------------- trial execution ------------------------- */

fn run_one_trial<T>(
    kind: StrategyKind,
    params_json: &serde_json::Value,
    problem: &Problem<T>,
    filename: &str,
    seed: u64,
) -> TrialResult
where
    T: SolveNumeric + From<i32> + Send + Sync + 'static,
{
    // Build a temporary model to pass to strategy builders (for neighbors, etc.).
    // Build a temporary model to pass to strategy builders (for neighbors, etc.).
    let model = SolverModel::from_problem(problem)
        .expect("SolverModel::from_problem() failed for the instance");

    let strategy: Box<dyn SearchStrategy<T, ChaCha8Rng>> = match kind {
        StrategyKind::Ils => {
            let p: IlsParams = serde_json::from_value(params_json.clone()).unwrap();
            ils_strategy_with_params::<T>(&model, &p)
        }
        StrategyKind::Sa => {
            let p: SaParams = serde_json::from_value(params_json.clone()).unwrap();
            sa_strategy_with_params::<T>(&model, &p)
        }
        StrategyKind::Gls => {
            let p: GlsParams = serde_json::from_value(params_json.clone()).unwrap();
            gls_strategy_with_params::<T>(&model, &p)
        }
    };

    let mut engine = build_engine_with_single_strategy::<T>(strategy, solve_time_limit());

    // If your engine exposes `set_seed`, you can uncomment:
    // engine.set_seed(seed);

    let start_ts = Utc::now();
    let t0 = Instant::now();
    let outcome = engine.solve(problem);
    let runtime = t0.elapsed();
    let end_ts = Utc::now();

    let cost = match outcome {
        Ok(Some(sol)) => Some(sol.cost()),
        _ => None,
    };

    TrialResult {
        strategy: kind,
        params: params_json.clone(),
        seed,
        filename: filename.to_string(),
        start_ts,
        end_ts,
        runtime_ms: runtime.as_millis(),
        cost,
    }
}

/* ------------------------- lightweight live summary ------------------------- */

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LiveSummary {
    strategy: StrategyKind,
    generated_at: DateTime<Utc>,
    trials: usize,
    feasible: usize,
    best_cost: Option<i64>,
    best_params: Option<serde_json::Value>,
    avg_cost: Option<f64>,
    avg_runtime_ms: f64,
}

fn update_and_write_summary(
    kind: StrategyKind,
    acc_trials: &mut usize,
    acc_feasible: &mut usize,
    acc_cost_sum: &mut i128,
    acc_runtime_sum: &mut u128,
    best: &mut Option<(i64, serde_json::Value)>,
    res: &TrialResult,
) {
    *acc_trials += 1;
    *acc_runtime_sum += res.runtime_ms;

    if let Some(c) = res.cost {
        *acc_feasible += 1;
        *acc_cost_sum += c as i128;
        if best.as_ref().map(|(b, _)| c < *b).unwrap_or(true) {
            *best = Some((c, res.params.clone()));
        }
    }

    // write summary every trial (cheap JSON)
    let avg_cost = if *acc_feasible > 0 {
        Some((*acc_cost_sum as f64) / (*acc_feasible as f64))
    } else {
        None
    };
    let avg_rt = (*acc_runtime_sum as f64) / (*acc_trials as f64);
    let (best_cost, best_params) = match best {
        Some((c, p)) => (Some(*c), Some(p.clone())),
        None => (None, None),
    };
    let summary = LiveSummary {
        strategy: kind,
        generated_at: Utc::now(),
        trials: *acc_trials,
        feasible: *acc_feasible,
        best_cost,
        best_params,
        avg_cost,
        avg_runtime_ms: avg_rt,
    };

    ensure_out_dir();
    let path = summary_path(kind);
    let s = serde_json::to_string_pretty(&summary).expect("serialize summary");
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("open {} failed: {}", path.display(), e));
    let _ = f.write_all(s.as_bytes());
}

/* ------------------------- worker loops ------------------------- */

fn worker_loop(
    kind: StrategyKind,
    problem: &Problem<i64>,
    filename: &str,
    trials: usize,
    seed_base: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed_base);
    let mut acc_trials = 0usize;
    let mut acc_feasible = 0usize;
    let mut acc_cost_sum: i128 = 0;
    let mut acc_runtime_sum: u128 = 0;
    let mut best: Option<(i64, serde_json::Value)> = None;

    for i in 0..trials {
        let params_json = match kind {
            StrategyKind::Ils => serde_json::to_value(sample_ils_params(&mut rng)).unwrap(),
            StrategyKind::Sa => serde_json::to_value(sample_sa_params(&mut rng)).unwrap(),
            StrategyKind::Gls => serde_json::to_value(sample_gls_params(&mut rng)).unwrap(),
        };
        let seed = seed_base.wrapping_add((i as u64).wrapping_mul(1_000_003));

        let res = run_one_trial::<i64>(kind, &params_json, problem, filename, seed);

        // Log one-line & JSONL
        if let Some(c) = res.cost {
            tracing::info!(
                "[{:?}] cost={} rt={}ms seed={} params={}",
                kind,
                c,
                res.runtime_ms,
                res.seed,
                res.params
            );
        } else {
            tracing::warn!(
                "[{:?}] infeasible rt={}ms seed={} params={}",
                kind,
                res.runtime_ms,
                res.seed,
                res.params
            );
        }
        append_jsonl(kind, &res);

        // Update live summary
        update_and_write_summary(
            kind,
            &mut acc_trials,
            &mut acc_feasible,
            &mut acc_cost_sum,
            &mut acc_runtime_sum,
            &mut best,
            &res,
        );
    }
}

/* ------------------------- main ------------------------- */

fn main() {
    enable_tracing();

    let (problem, filename) = load_first_instance();
    tracing::info!(
        "Random tuner on first instance: {} (|berths|={}, |flex|={})",
        filename,
        problem.berths().len(),
        problem.flexible_requests().len()
    );

    let threads = threads_per_strategy();
    let trials = trials_per_worker();
    let base_seed = cfg_u64("TUNER_BASE_SEED", 0x00C0_FFEE_1234_ABCD);

    // Run ILS / SA / GLS groups concurrently
    thread::scope(|scope| {
        // ILS
        for w in 0..threads {
            let prob_ref = &problem;
            let fname = filename.clone();
            let seed = base_seed.wrapping_add(0x11_0000).wrapping_add(w as u64);
            scope.spawn(move || worker_loop(StrategyKind::Ils, prob_ref, &fname, trials, seed));
        }
        // SA
        for w in 0..threads {
            let prob_ref = &problem;
            let fname = filename.clone();
            let seed = base_seed.wrapping_add(0x22_0000).wrapping_add(w as u64);
            scope.spawn(move || worker_loop(StrategyKind::Sa, prob_ref, &fname, trials, seed));
        }
        // GLS
        for w in 0..threads {
            let prob_ref = &problem;
            let fname = filename.clone();
            let seed = base_seed.wrapping_add(0x33_0000).wrapping_add(w as u64);
            scope.spawn(move || worker_loop(StrategyKind::Gls, prob_ref, &fname, trials, seed));
        }
    });

    tracing::info!(
        "Tuner finished batch (workers_per_strategy={}, trials_per_worker={}).",
        threads,
        trials
    );
}
