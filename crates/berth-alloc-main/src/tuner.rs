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

//! Concurrent parameter tuner for ILS / SA / GLS.
//!
//! - Uses ONLY the first instance under `instances/` (long per-trial runs).
//! - Each trial: one engine, one worker, exactly one strategy.
//! - SA/ILS/GLS are tuned on different threads concurrently.
//! - Each solve gets ~80 seconds (configurable below).
//! - Writes one JSON per strategy: best params + leaderboard of all trials.

use berth_alloc_model::prelude::{Problem, SolutionView};
use berth_alloc_model::problem::loader::ProblemLoader;
use berth_alloc_solver::{
    core::numeric::SolveNumeric,
    engine::{
        search::SearchStrategy,
        solver_engine::{SolverEngine, SolverEngineBuilder},
    },
    model::solver_model::SolverModel,
};
use chrono::{DateTime, Utc};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

const INSTANCE_LIMIT: usize = 1; // <-- only the first instance
const SOLVE_TIME_LIMIT: Duration = Duration::from_secs(80); // ~80 sec per trial

/* ------------------------- logging ------------------------- */

fn enable_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE)
        .init();
}

/* ------------------------- instances ------------------------- */

fn find_instances_dir() -> Option<PathBuf> {
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

    let mut files: Vec<PathBuf> = std::fs::read_dir(&inst_dir)
        .expect("read_dir(instances) failed")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && e.path().extension().map(|x| x == "txt").unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();
    files.sort();

    let f = files
        .into_iter()
        .take(INSTANCE_LIMIT)
        .next()
        .expect("instances/: no .txt instances found");

    let loader = ProblemLoader::default();
    let problem = loader
        .from_path(&f)
        .unwrap_or_else(|e| panic!("Failed to load {}: {e}", f.display()));
    let name = f
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| f.to_string_lossy().into_owned());

    (problem, name)
}

/* ------------------------- kinds & params ------------------------- */

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StrategyKind {
    ILS,
    SA,
    GLS,
}
impl StrategyKind {
    pub fn json_basename(self) -> &'static str {
        match self {
            StrategyKind::ILS => "tuner_results_ils.json",
            StrategyKind::SA => "tuner_results_sa.json",
            StrategyKind::GLS => "tuner_results_gls.json",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IlsParams {
    pub local_steps_min: usize,
    pub local_steps_max: usize,
    pub destroy_attempts: usize,
    pub repair_attempts: usize,
    pub refetch_after_stale: usize,
    pub hard_refetch_every: usize,
    pub kick_ops_after_refetch: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaParams {
    pub init_temp: f64,
    pub cooling: f64,
    pub min_temp: f64,
    pub steps_per_epoch: usize,
    pub hard_refetch_every: usize,
    pub refetch_after_stale: usize,
    pub reheat_factor: f64,
    pub kick_ops_after_refetch: usize,
    pub big_m: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlsParams {
    pub lambda: i64,
    pub penalty_step: i64,
    pub max_local_steps: usize,
    pub stagnation_before_pulse: usize,
    pub pulse_top_k: usize,
    pub hard_refetch_every: usize,
    pub kick_steps_on_reset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params")]
pub enum ParamSet {
    ILS(IlsParams),
    SA(SaParams),
    GLS(GlsParams),
}

/* ------------------------- results & aggregation ------------------------- */

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub kind: StrategyKind,
    pub params: ParamSet,
    pub instance_name: String,
    pub start_ts: DateTime<Utc>,
    pub end_ts: DateTime<Utc>,
    pub runtime_ms: u128,
    pub cost: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateResult {
    pub kind: StrategyKind,
    pub params: ParamSet,
    pub trials: Vec<TrialResult>,
    pub total_runtime_ms: u128,
    pub feasible_count: usize,
    pub mean_cost: Option<f64>,
}

impl AggregateResult {
    fn from_trials(kind: StrategyKind, params: ParamSet, trials: Vec<TrialResult>) -> Self {
        let total_runtime_ms = trials.iter().map(|t| t.runtime_ms).sum::<u128>();
        let mut feasible = 0usize;
        let mut sum_cost: i128 = 0;
        for t in &trials {
            if let Some(c) = t.cost {
                feasible += 1;
                sum_cost += c as i128;
            }
        }
        let mean_cost = if feasible > 0 {
            Some(sum_cost as f64 / feasible as f64)
        } else {
            None
        };
        Self {
            kind,
            params,
            trials,
            total_runtime_ms,
            feasible_count: feasible,
            mean_cost,
        }
    }
}

fn compare_aggregate(a: &AggregateResult, b: &AggregateResult) -> Ordering {
    // 1) Feasible count (desc)
    match a.feasible_count.cmp(&b.feasible_count).reverse() {
        Ordering::Equal => {}
        ord => return ord,
    }
    // 2) Mean cost (asc, None considered worst)
    match (&a.mean_cost, &b.mean_cost) {
        (Some(ca), Some(cb)) => match ca.partial_cmp(cb).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {}
            ord => return ord,
        },
        (Some(_), None) => return Ordering::Less,
        (None, Some(_)) => return Ordering::Greater,
        (None, None) => {}
    }
    // 3) Total runtime (asc)
    a.total_runtime_ms.cmp(&b.total_runtime_ms)
}

/* ------------------------- param grids ------------------------- */

fn ils_grid() -> Vec<ParamSet> {
    let steps = [(900, 1500), (1200, 2200)];
    let destroy = [8usize, 12];
    let repair = [20usize, 28];
    let refetch = [32usize, 40];
    let cadence = [0usize, 14];
    let kick = [4usize, 8];

    let mut v = Vec::new();
    for (lo, hi) in steps {
        for d in destroy {
            for r in repair {
                for rf in refetch {
                    for he in cadence {
                        for k in kick {
                            v.push(ParamSet::ILS(IlsParams {
                                local_steps_min: lo,
                                local_steps_max: hi,
                                destroy_attempts: d,
                                repair_attempts: r,
                                refetch_after_stale: rf,
                                hard_refetch_every: he,
                                kick_ops_after_refetch: k,
                            }));
                        }
                    }
                }
            }
        }
    }
    v
}

fn sa_grid() -> Vec<ParamSet> {
    let t0s = [25.0, 35.0];
    let cools = [0.9993, 0.9997];
    let tmin = [1e-4];
    let steps = [900usize, 1200];
    let cadence = [0usize, 80];
    let refetch = [40usize, 60];
    let reheat = [0.6, 0.85];
    let kick = [8usize, 18];
    let big_m = [800_000_000i64, 900_000_000];

    let mut v = Vec::new();
    for t0 in t0s {
        for c in cools {
            for tm in tmin {
                for s in steps {
                    for he in cadence {
                        for rf in refetch {
                            for rh in reheat {
                                for k in kick {
                                    for m in big_m {
                                        v.push(ParamSet::SA(SaParams {
                                            init_temp: t0,
                                            cooling: c,
                                            min_temp: tm,
                                            steps_per_epoch: s,
                                            hard_refetch_every: he,
                                            refetch_after_stale: rf,
                                            reheat_factor: rh,
                                            kick_ops_after_refetch: k,
                                            big_m: m,
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

fn gls_grid() -> Vec<ParamSet> {
    let lambdas = [7i64, 9];
    let step = [1i64, 2];
    let max_local = [1500usize, 2100];
    let stagnation = [8usize, 12];
    let topk = [16usize, 20];
    let cadence = [0usize, 24];
    let kick = [4usize, 6];

    let mut v = Vec::new();
    for l in lambdas {
        for ps in step {
            for ml in max_local {
                for st in stagnation {
                    for tk in topk {
                        for he in cadence {
                            for k in kick {
                                v.push(ParamSet::GLS(GlsParams {
                                    lambda: l,
                                    penalty_step: ps,
                                    max_local_steps: ml,
                                    stagnation_before_pulse: st,
                                    pulse_top_k: tk,
                                    hard_refetch_every: he,
                                    kick_steps_on_reset: k,
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

/* ------------------------- strategy builders ------------------------- */

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

    let proximity_map = model.proximity_map();
    let n_any = neighbors::any(proximity_map);
    let n_dir = neighbors::direct_competitors(proximity_map);
    let n_same = neighbors::same_berth(proximity_map);

    let strat = IteratedLocalSearchStrategy::new()
        .with_local_steps_range(p.local_steps_min..=p.local_steps_max)
        .with_local_sideways(true)
        .with_local_worsening_prob(0.0)
        .with_destroy_attempts(p.destroy_attempts)
        .with_repair_attempts(p.repair_attempts)
        .with_refetch_after_stale(p.refetch_after_stale)
        .with_hard_refetch_every(p.hard_refetch_every)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_kick_ops_after_refetch(p.kick_ops_after_refetch)
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
    use berth_alloc_solver::search::operator_library::local::{
        CascadeInsertPolicy, CascadeRelocateK, CrossExchangeAcrossBerths,
        CrossExchangeBestAcrossBerths, HillClimbBestSwapSameBerth, HillClimbRelocateBest,
        OrOptBlockRelocate, RandomRelocateAnywhere, RandomizedGreedyRelocateRcl,
        RelocateSingleBest, RelocateSingleBestAllowWorsening, ShiftEarlierOnSameBerth,
        SwapPairSameBerth,
    };

    let proximity_map = model.proximity_map();
    let n_any = neighbors::any(proximity_map);
    let n_dir = neighbors::direct_competitors(proximity_map);
    let n_same = neighbors::same_berth(proximity_map);

    let strat = SimulatedAnnealingStrategy::new()
        .with_init_temp(p.init_temp)
        .with_cooling(p.cooling)
        .with_min_temp(p.min_temp)
        .with_steps_per_epoch(p.steps_per_epoch)
        .with_hard_refetch_every(p.hard_refetch_every)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_refetch_after_stale(p.refetch_after_stale)
        .with_reheat_factor(p.reheat_factor)
        .with_kick_ops_after_refetch(p.kick_ops_after_refetch)
        .with_big_m_for_energy(p.big_m)
        .with_acceptance_targets(0.22, 0.70)
        // locals
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
    let mut s = gls_strategy::<T, ChaCha8Rng>(model);
    s = s
        .with_lambda(p.lambda)
        .with_penalty_step(p.penalty_step)
        .with_max_local_steps(p.max_local_steps)
        .with_pulse_params(p.stagnation_before_pulse, p.pulse_top_k)
        .with_hard_refetch_every(p.hard_refetch_every)
        .with_kick_steps_on_reset(p.kick_steps_on_reset);
    Box::new(s)
}

/* ------------------------- engine builder (single strategy) ------------------------- */

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

fn run_trial(
    kind: StrategyKind,
    params: &ParamSet,
    problem: &Problem<i64>,
    instance_name: &str,
) -> TrialResult {
    let start_ts = Utc::now();
    let t0 = Instant::now();

    // Build a temporary model for wiring neighbors in strategy constructors.
    let temp_model =
        SolverModel::from_problem(problem).expect("temporary SolverModel should build");

    let boxed_strategy: Box<dyn SearchStrategy<i64, ChaCha8Rng>> = match (kind, params) {
        (StrategyKind::ILS, ParamSet::ILS(p)) => ils_strategy_with_params::<i64>(&temp_model, p),
        (StrategyKind::SA, ParamSet::SA(p)) => sa_strategy_with_params::<i64>(&temp_model, p),
        (StrategyKind::GLS, ParamSet::GLS(p)) => gls_strategy_with_params::<i64>(&temp_model, p),
        _ => unreachable!("param kind mismatch"),
    };

    let mut engine = build_engine_with_single_strategy::<i64>(boxed_strategy, SOLVE_TIME_LIMIT);

    let outcome = engine.solve(problem);
    let runtime = t0.elapsed();
    let end_ts = Utc::now();

    let cost = match outcome {
        Ok(Some(sol)) => Some(sol.cost()),
        _ => None,
    };

    TrialResult {
        kind,
        params: params.clone(),
        instance_name: instance_name.to_string(),
        start_ts,
        end_ts,
        runtime_ms: runtime.as_millis(),
        cost,
    }
}

/* ------------------------- aggregation + report ------------------------- */

fn aggregate(kind: StrategyKind, params: ParamSet, trials: Vec<TrialResult>) -> AggregateResult {
    AggregateResult::from_trials(kind, params, trials)
}

fn write_strategy_report(kind: StrategyKind, mut results: Vec<AggregateResult>) {
    // sort by (feasible desc, mean cost asc, runtime asc)
    results.sort_by(compare_aggregate);

    // best is first
    let out = serde_json::json!({
        "kind": format!("{:?}", kind),
        "best": results.first(),
        "leaderboard": results,
        "generated_at": Utc::now(),
    });

    let out_path = PathBuf::from(kind.json_basename());
    match File::create(&out_path).and_then(|mut f| {
        let s = serde_json::to_string_pretty(&out).expect("serialize report");
        f.write_all(s.as_bytes())
    }) {
        Ok(()) => tracing::info!("Wrote {}", out_path.display()),
        Err(e) => tracing::error!("Failed to write {}: {}", out_path.display(), e),
    }
}

/* ------------------------- runners per strategy (on separate threads) ------------------------- */

fn run_grid_for_kind(
    kind: StrategyKind,
    params: Vec<ParamSet>,
    problem: &Problem<i64>,
    instance_name: &str,
) {
    tracing::info!("Starting tuner for {:?}", kind);

    // Evaluate each param set on the *single* instance.
    // We keep it sequential inside the thread to avoid oversubscribing cores
    // while the three strategy families run concurrently.
    let mut aggregates: Vec<AggregateResult> = Vec::with_capacity(params.len());

    for p in params {
        let trial = run_trial(kind, &p, problem, instance_name);
        let agg = aggregate(kind, p, vec![trial]);
        aggregates.push(agg);
    }

    write_strategy_report(kind, aggregates);
    tracing::info!("Finished tuner for {:?}", kind);
}

/* ------------------------- main ------------------------- */

fn main() {
    enable_tracing();

    let (problem, instance_name) = load_first_instance();
    tracing::info!(
        "Tuning on single instance: {} (|berths|={}, |vessels|={})",
        instance_name,
        problem.berths().len(),
        problem.flexible_requests().len()
    );

    // Build grids
    let ils_params = ils_grid();
    let sa_params = sa_grid();
    let gls_params = gls_grid();

    // Spawn 3 threads — one per strategy family
    let p1 = problem.clone();
    let i1 = instance_name.clone();
    let h_ils = thread::spawn(move || run_grid_for_kind(StrategyKind::ILS, ils_params, &p1, &i1));

    let p2 = problem.clone();
    let i2 = instance_name.clone();
    let h_sa = thread::spawn(move || run_grid_for_kind(StrategyKind::SA, sa_params, &p2, &i2));

    let p3 = problem;
    let i3 = instance_name;
    let h_gls = thread::spawn(move || run_grid_for_kind(StrategyKind::GLS, gls_params, &p3, &i3));

    // Wait
    let _ = (h_ils.join(), h_sa.join(), h_gls.join());
    tracing::info!("All tuners finished.");
}
