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

use berth_alloc_model::prelude::{Problem, SolutionView};
use berth_alloc_model::problem::loader::ProblemLoader;
use berth_alloc_solver::engine::solver_engine::{SolverEngine, SolverEngineBuilder};
use berth_alloc_solver::engine::{gls, ils, sa, tabu};
use chrono::Utc;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufReader, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum StrategyKind {
    Gls,
    Ils,
    Sa,
    Tabu,
}
impl StrategyKind {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gls" => Some(Self::Gls),
            "ils" => Some(Self::Ils),
            "sa" => Some(Self::Sa),
            "tabu" => Some(Self::Tabu),
            _ => None,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            StrategyKind::Gls => "gls",
            StrategyKind::Ils => "ils",
            StrategyKind::Sa => "sa",
            StrategyKind::Tabu => "tabu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateKind {
    Avg,
    Sum,
}
impl AggregateKind {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "avg" | "average" => Some(Self::Avg),
            "sum" => Some(Self::Sum),
            _ => None,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            AggregateKind::Avg => "avg",
            AggregateKind::Sum => "sum",
        }
    }
}

#[derive(Debug, Clone)]
struct Cli {
    // Instance selection: either a single file (--instance) OR a directory (--instances-dir).
    instance_path: Option<PathBuf>,
    instances_dir: Option<PathBuf>,
    strategy: Option<StrategyKind>,
    trials: usize,
    time_secs_per_instance: u64,
    budget_secs: u64,
    out_dir: PathBuf,
    aggregate: AggregateKind, // avg (default) or sum
}

fn parse_cli() -> Cli {
    let mut args = std::env::args().skip(1);

    let mut instance_path: Option<PathBuf> = None;
    let mut instances_dir: Option<PathBuf> = None;
    let mut strategy: Option<StrategyKind> = None;
    let mut trials: usize = 200;
    let mut time_secs_per_instance: u64 = 30;
    let mut budget_secs: u64 = 24 * 3600; // default 24h
    let mut out_dir: PathBuf = PathBuf::from("tune_out");
    let mut aggregate = AggregateKind::Avg;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--instance" => instance_path = args.next().map(PathBuf::from),
            "--instances-dir" => instances_dir = args.next().map(PathBuf::from),
            "--strategy" => strategy = args.next().and_then(|s| StrategyKind::from_str(&s)),
            "--trials" => trials = args.next().and_then(|x| x.parse().ok()).unwrap_or(trials),
            "--time" | "--time-per-instance" => {
                time_secs_per_instance = args
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(time_secs_per_instance)
            }
            "--budget-secs" => {
                budget_secs = args
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(budget_secs)
            }
            "--aggregate" => {
                aggregate = args
                    .next()
                    .and_then(|s| AggregateKind::from_str(&s))
                    .unwrap_or(aggregate)
            }
            "--out" => out_dir = args.next().map(PathBuf::from).unwrap_or(out_dir),
            _ => {}
        }
    }

    if instance_path.is_none() && instances_dir.is_none() {
        // Back-compat: if nothing provided, look for default folder "instances/"
        let default_dir = PathBuf::from("instances");
        if default_dir.is_dir() {
            instances_dir = Some(default_dir);
        } else {
            // fallback: default single file example (legacy)
            instance_path = Some(PathBuf::from("instances/f200x15-01.txt"));
        }
    }

    Cli {
        instance_path,
        instances_dir,
        strategy,
        trials,
        time_secs_per_instance,
        budget_secs,
        out_dir,
        aggregate,
    }
}

// ------------------------ Records (best only) ------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BestRecord {
    updated_at: String,
    strategy: String,
    suite: String, // folder name or "single"
    cfg_id: String,
    params: TunableParams,
    time_secs_per_instance: u64,
    aggregate: String, // "avg" or "sum"
    instances: usize,  // number of instances evaluated
    score: f64,        // aggregated cost (avg or sum)
}

// ------------------------ Param bag ------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TunableParams {
    // GLS / TABU (penalized-aug variants)
    lambda: Option<i64>,
    penalty_step: Option<i64>,
    decay_num: Option<u32>,
    decay_den: Option<u32>,
    pulse_top_k: Option<usize>,
    stagnation_rounds: Option<usize>,
    max_local_steps: Option<usize>,
    // ILS
    local_lo: Option<usize>,
    local_hi: Option<usize>,
    allow_sideways: Option<bool>,
    worsen_prob: Option<f64>,
    destroy_attempts: Option<usize>,
    repair_attempts: Option<usize>,
    refetch_after_stale: Option<usize>,
    hard_refetch_every: Option<usize>,
    // SA
    sa_init_temp: Option<f64>,
    sa_cooling: Option<f64>,
    sa_min_temp: Option<f64>,
    sa_steps_per_temp: Option<usize>,
    sa_reheat_factor: Option<f64>,
    sa_op_ema_alpha: Option<f64>,
    sa_op_min_weight: Option<f64>,
    sa_ar_low: Option<f64>,
    sa_ar_high: Option<f64>,
    sa_big_m: Option<i64>,
}

// ------------------------ Sampling ------------------------

fn sample_params(kind: StrategyKind, rng: &mut ChaCha8Rng) -> TunableParams {
    match kind {
        StrategyKind::Gls => TunableParams {
            lambda: Some([7, 8, 9, 10].choose(rng).copied().unwrap_or(9)),
            penalty_step: Some([2, 3].choose(rng).copied().unwrap_or(2)),
            decay_num: Some(rng.random_range(92..=96)),
            decay_den: Some(100),
            pulse_top_k: Some(rng.random_range(16..=28)),
            stagnation_rounds: Some(rng.random_range(6..=12)),
            max_local_steps: Some(rng.random_range(1600..=2400)),
            hard_refetch_every: Some(rng.random_range(18..=36)),
            refetch_after_stale: Some(rng.random_range(48..=120)),
            ..Default::default()
        },
        StrategyKind::Tabu => TunableParams {
            lambda: Some([6, 7, 8].choose(rng).copied().unwrap_or(7)),
            penalty_step: Some(2),
            decay_num: Some(rng.random_range(92..=96)),
            decay_den: Some(100),
            pulse_top_k: Some(rng.random_range(16..=24)),
            stagnation_rounds: Some(rng.random_range(6..=12)),
            max_local_steps: Some(rng.random_range(1700..=2200)),
            hard_refetch_every: Some(rng.random_range(18..=36)),
            refetch_after_stale: Some(rng.random_range(48..=120)),
            ..Default::default()
        },
        StrategyKind::Ils => TunableParams {
            local_lo: Some(rng.random_range(800..=1400)),
            local_hi: Some(rng.random_range(1400..=2000)),
            allow_sideways: Some(true),
            worsen_prob: Some(rng.random_range(8..=16) as f64 / 1000.0),
            destroy_attempts: Some(rng.random_range(12..=20)),
            repair_attempts: Some(rng.random_range(18..=36)),
            refetch_after_stale: Some(rng.random_range(40..=80)),
            hard_refetch_every: Some(rng.random_range(18..=36)),
            ..Default::default()
        },
        StrategyKind::Sa => TunableParams {
            sa_init_temp: Some(rng.random_range(14..=22) as f64 / 10.0),
            sa_cooling: Some(1.0 - (rng.random_range(60..=120) as f64 / 100000.0)),
            sa_min_temp: Some(1e-4),
            sa_steps_per_temp: Some(rng.random_range(450..=700)),
            sa_reheat_factor: Some([0.0, 0.4, 0.6, 1.0].choose(rng).copied().unwrap_or(0.6)),
            sa_op_ema_alpha: Some(rng.random_range(20..=40) as f64 / 100.0),
            sa_op_min_weight: Some(rng.random_range(8..=15) as f64 / 100.0),
            sa_ar_low: Some(0.10),
            sa_ar_high: Some(rng.random_range(50..=60) as f64 / 100.0),
            sa_big_m: Some(
                [1_000_000_000, 1_250_000_000, 1_500_000_000]
                    .choose(rng)
                    .copied()
                    .unwrap_or(1_250_000_000),
            ),
            hard_refetch_every: Some(rng.random_range(18..=36)),
            refetch_after_stale: Some(rng.random_range(48..=120)),
            ..Default::default()
        },
    }
}

// ------------------------ Strategy builders ------------------------

fn build_boxed_strategy<T: berth_alloc_solver::core::numeric::SolveNumeric + From<i32>>(
    kind: StrategyKind,
    model: &berth_alloc_solver::model::solver_model::SolverModel<T>,
    params: &TunableParams,
) -> Box<dyn berth_alloc_solver::engine::search::SearchStrategy<T, ChaCha8Rng>> {
    match kind {
        StrategyKind::Gls => {
            let mut s = gls::gls_strategy::<T, ChaCha8Rng>(model);
            if let Some(v) = params.lambda {
                s = s.with_lambda(v);
            }
            if let Some(v) = params.penalty_step {
                s = s.with_penalty_step(v);
            }
            if let (Some(num), Some(den)) = (params.decay_num, params.decay_den) {
                s = s.with_decay(gls::DecayMode::Multiplicative { num, den });
            }
            if let Some(v) = params.max_local_steps {
                s = s.with_max_local_steps(v);
            }
            if let (Some(stag), Some(topk)) = (params.stagnation_rounds, params.pulse_top_k) {
                s = s.with_pulse_params(stag, topk);
            }
            if let Some(v) = params.refetch_after_stale {
                s = s.with_refetch_after_stale(v);
            }
            if let Some(v) = params.hard_refetch_every {
                s = s.with_hard_refetch_every(v);
            }
            Box::new(s)
        }
        StrategyKind::Tabu => {
            let mut s = tabu::tabu_strategy::<T, ChaCha8Rng>(model);
            if let Some(v) = params.lambda {
                s = s.with_lambda(v);
            }
            if let Some(v) = params.penalty_step {
                s = s.with_penalty_step(v);
            }
            if let (Some(num), Some(den)) = (params.decay_num, params.decay_den) {
                use berth_alloc_solver::engine::feature_signal::prelude::DecayMode;
                s = s.with_decay(DecayMode::Multiplicative { num, den });
            }
            if let Some(v) = params.max_local_steps {
                s = s.with_max_local_steps(v);
            }
            if let (Some(stag), Some(topk)) = (params.stagnation_rounds, params.pulse_top_k) {
                s = s.with_pulse_params(stag, topk);
            }
            if let Some(v) = params.refetch_after_stale {
                s = s.with_refetch_after_stale(v);
            }
            if let Some(v) = params.hard_refetch_every {
                s = s.with_hard_refetch_every(v);
            }
            Box::new(s)
        }
        StrategyKind::Ils => {
            let mut s = ils::ils_strategy::<T, ChaCha8Rng>(model);
            if let (Some(lo), Some(hi)) = (params.local_lo, params.local_hi) {
                s = s.with_local_steps_range(lo..=hi);
            }
            if let Some(b) = params.allow_sideways {
                s = s.with_local_sideways(b);
            }
            if let Some(p) = params.worsen_prob {
                s = s.with_local_worsening_prob(p);
            }
            if let Some(v) = params.destroy_attempts {
                s = s.with_destroy_attempts(v);
            }
            if let Some(v) = params.repair_attempts {
                s = s.with_repair_attempts(v);
            }
            if let Some(v) = params.refetch_after_stale {
                s = s.with_refetch_after_stale(v);
            }
            if let Some(v) = params.hard_refetch_every {
                s = s.with_hard_refetch_every(v);
            }
            Box::new(s)
        }
        StrategyKind::Sa => {
            let mut s = sa::sa_strategy::<T, ChaCha8Rng>(model);
            if let Some(v) = params.sa_init_temp {
                s = s.with_init_temp(v);
            }
            if let Some(v) = params.sa_cooling {
                s = s.with_cooling(v);
            }
            if let Some(v) = params.sa_steps_per_temp {
                s = s.with_steps_per_temp(v);
            }
            if let Some(v) = params.sa_reheat_factor {
                s = s.with_reheat_factor(v);
            }
            if let Some(v) = params.sa_op_ema_alpha {
                s = s.with_op_ema_alpha(v);
            }
            if let Some(v) = params.sa_op_min_weight {
                s = s.with_op_min_weight(v);
            }
            if let (Some(lo), Some(hi)) = (params.sa_ar_low, params.sa_ar_high) {
                s = s.with_acceptance_targets(lo, hi);
            }
            if let Some(v) = params.sa_big_m {
                s = s.with_big_m_for_energy(v);
            }
            if let Some(v) = params.refetch_after_stale {
                s = s.with_refetch_after_stale(v);
            }
            if let Some(v) = params.hard_refetch_every {
                s = s.with_hard_refetch_every(v);
            }
            Box::new(s)
        }
    }
}

// ------------------------ I/O helpers ------------------------

fn ensure_dir(p: &Path) {
    if !p.exists() {
        let _ = std::fs::create_dir_all(p);
    }
}

fn read_best(path: &Path) -> Option<BestRecord> {
    if !path.exists() {
        return None;
    }
    let f = File::open(path).ok()?;
    serde_json::from_reader(BufReader::new(f)).ok()
}

fn write_best(path: &Path, rec: &BestRecord) -> std::io::Result<()> {
    let tmp_path = path.with_extension("json.tmp");
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)?;
    let s = serde_json::to_string_pretty(rec).unwrap();
    f.write_all(s.as_bytes())?;
    f.sync_all()?;
    std::fs::rename(tmp_path, path)?;
    Ok(())
}

// ------------------------ Model loader ------------------------

fn load_problem(path: &Path) -> Result<Problem<i64>, Box<dyn Error>> {
    let loader = ProblemLoader::default();
    let problem = loader.from_path(path)?;
    Ok(problem)
}

#[allow(clippy::type_complexity)]
fn load_suite(cli: &Cli) -> Result<(String, Vec<(String, Problem<i64>)>), Box<dyn Error>> {
    // returns (suite_name, [(instance_basename, problem)...])
    let mut items: Vec<(String, Problem<i64>)> = Vec::new();

    if let Some(dir) = &cli.instances_dir {
        let suite_name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("instances")
            .to_string();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() {
                // naive filter: accept .txt or .dat; feel free to widen as needed
                let ok_ext = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| matches!(e, "txt" | "dat" | "json" | "problem"))
                    .unwrap_or(true);
                if !ok_ext {
                    continue;
                }
                let name = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("instance")
                    .to_string();
                let prob = load_problem(&p)?;
                items.push((name, prob));
            }
        }
        items.sort_by(|a, b| a.0.cmp(&b.0));
        Ok((suite_name, items))
    } else if let Some(file) = &cli.instance_path {
        let suite_name = "single".to_string();
        let name = file
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("instance")
            .to_string();
        let prob = load_problem(file)?;
        items.push((name, prob));
        Ok((suite_name, items))
    } else {
        Err("no instance or instances-dir provided".into())
    }
}

// ------------------------ Evaluate params on the entire suite -------------------

fn evaluate_params_on_suite(
    kind: StrategyKind,
    params: &TunableParams,
    suite: &[(String, Problem<i64>)],
    time_secs_per_instance: u64,
    aggregate: AggregateKind,
) -> Option<(f64, usize)> {
    // Require feasibility on ALL instances; otherwise reject (return None).
    // Aggregate cost across the suite (avg or sum).
    let mut total: f64 = 0.0;
    let mut count: usize = 0;

    for (_name, prob) in suite {
        let model = berth_alloc_solver::model::solver_model::SolverModel::from_problem(prob)
            .expect("model");
        let strat = build_boxed_strategy::<i64>(kind, &model, params);

        let mut engine: SolverEngine<i64> = SolverEngineBuilder::default()
            .with_worker_count(2) // <-- 2 workers per engine
            .with_time_limit(Duration::from_secs(time_secs_per_instance))
            .with_strategy(strat)
            .build();

        match engine.solve(prob) {
            Ok(Some(sol)) => {
                total += sol.cost() as f64;
                count += 1;
            }
            _ => {
                // infeasible on one instance disqualifies this attempt
                return None;
            }
        }
    }

    if count == 0 {
        return None;
    }
    let score = match aggregate {
        AggregateKind::Sum => total,
        AggregateKind::Avg => total / (count as f64),
    };
    Some((score, count))
}

// ------------------------ Attempt over suite (returns score/record) -------------

fn run_attempt_over_suite(
    suite_name: &str,
    kind: StrategyKind,
    params: &TunableParams,
    cfg_seed: u64,
    suite: &[(String, Problem<i64>)],
    time_secs_per_instance: u64,
    aggregate: AggregateKind,
) -> Option<(f64, BestRecord)> {
    let (score, count) =
        evaluate_params_on_suite(kind, params, suite, time_secs_per_instance, aggregate)?;

    let cfg_id = format!("{}-{:08x}", kind.as_str(), (cfg_seed & 0xffff_ffff));
    let rec = BestRecord {
        updated_at: Utc::now().to_rfc3339(),
        strategy: kind.as_str().to_string(),
        suite: suite_name.to_string(),
        cfg_id,
        params: params.clone(),
        time_secs_per_instance,
        aggregate: aggregate.as_str().to_string(),
        instances: count,
        score,
    };
    Some((score, rec))
}

// ------------------------ Per-strategy sequential runner ------------------------

#[allow(clippy::too_many_arguments)]
fn run_strategy_for_kind(
    kind: StrategyKind,
    suite_name: &str,
    suite: &[(String, Problem<i64>)],
    out_dir: &Path,
    trials: usize,
    time_secs_per_instance: u64,
    master_seed: u64,
    aggregate: AggregateKind,
    deadline: Instant, // wall-clock cutoff for the whole tuning run
) {
    let best_path = out_dir.join(format!("best_{}_{}.json", kind.as_str(), suite_name));

    // Initialize per-strategy incumbent from disk if present
    let mut best_score: Option<f64> = read_best(&best_path).map(|b| b.score);

    // RNG for this strategy thread; derive per-config seeds from it
    let mut rng = ChaCha8Rng::seed_from_u64(master_seed ^ (kind as u8 as u64));

    for i in 0..trials {
        if Instant::now() >= deadline {
            eprintln!(
                "tune[{}]: budget reached, stopping at trial {}",
                kind.as_str(),
                i
            );
            break;
        }

        let cfg_seed: u64 = rng.random::<u64>() ^ ((i as u64) << 32);
        let mut sampler = ChaCha8Rng::seed_from_u64(cfg_seed);
        let params = sample_params(kind, &mut sampler);

        if let Some((score, rec)) = run_attempt_over_suite(
            suite_name,
            kind,
            &params,
            cfg_seed,
            suite,
            time_secs_per_instance,
            aggregate,
        ) {
            let better = match best_score {
                None => true,
                Some(s) => score < s,
            };
            if better && write_best(&best_path, &rec).is_ok() {
                best_score = Some(score);
                eprintln!(
                    "tune[{}]: improved best: score={:.3} (agg={})",
                    kind.as_str(),
                    score,
                    aggregate.as_str()
                );
            }
        }
    }
}

// ------------------------ Main ------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let cli = parse_cli();
    ensure_dir(&cli.out_dir);

    // Load suite
    let (suite_name, suite) = load_suite(&cli)?;
    if suite.is_empty() {
        return Err("no instances found in the suite".into());
    }

    // Quick banner
    let total_instances = suite.len();
    eprintln!(
        "tune: suite={} (instances={}), trials={}, per-instance time={}s, budget={}s, agg={}, filter={:?}",
        suite_name,
        total_instances,
        cli.trials,
        cli.time_secs_per_instance,
        cli.budget_secs,
        cli.aggregate.as_str(),
        cli.strategy
    );
    eprintln!(
        "tune: writing only per-strategy best JSONs to {}.",
        cli.out_dir.display()
    );

    // Strategy plan
    let all = [
        StrategyKind::Gls,
        StrategyKind::Ils,
        StrategyKind::Sa,
        StrategyKind::Tabu,
    ];
    let plan: Vec<StrategyKind> = match cli.strategy {
        Some(k) => vec![k],
        None => all.to_vec(),
    };

    // Deterministic master seed unless RUST_SEED set
    let master_seed = std::env::var("RUST_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xBAD5EED_u64);

    // Global wall-clock deadline
    let deadline = Instant::now() + Duration::from_secs(cli.budget_secs);

    // One OS thread per strategy; each executes trials sequentially over the entire suite
    let mut handles = Vec::new();
    for &kind in &plan {
        let out_dir_cl = cli.out_dir.clone();
        let suite_name_cl = suite_name.clone();
        let suite_cl = suite.clone(); // Problem<i64> is cloneable
        let trials = cli.trials;
        let time_secs = cli.time_secs_per_instance;
        let seed = master_seed;
        let agg = cli.aggregate;
        let dl = deadline;

        handles.push(thread::spawn(move || {
            run_strategy_for_kind(
                kind,
                &suite_name_cl,
                &suite_cl,
                &out_dir_cl,
                trials,
                time_secs,
                seed,
                agg,
                dl,
            );
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    eprintln!(
        "tune: finished. See {} for per-strategy best_* JSON files.",
        cli.out_dir.display()
    );
    Ok(())
}
