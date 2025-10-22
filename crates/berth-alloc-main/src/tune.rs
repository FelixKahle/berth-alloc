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
    ops::RangeInclusive,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};
use tracing::{debug, info, instrument, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

// Additional imports needed to construct strategies/operators directly
use berth_alloc_core::prelude::{TimeDelta, TimePoint};
use num_traits::ToPrimitive;

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
    // GLS / TABU shared (penalized-aug variants)
    lambda: Option<i64>,
    penalty_step: Option<i64>,
    decay_num: Option<u32>,
    decay_den: Option<u32>,
    pulse_top_k: Option<usize>,
    stagnation_rounds: Option<usize>,
    max_local_steps: Option<usize>,
    max_penalty: Option<i64>, // used by GLS/TABU

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

    // TABU-specific
    tabu_tenure_lo: Option<usize>,
    tabu_tenure_hi: Option<usize>,
    tabu_samples_per_step: Option<usize>,

    // ---------------- Operator ranges (common local improvement ops) ----------------
    // ShiftEarlierOnSameBerth::new(usize..=usize)
    op_shift_earlier_lo: Option<usize>,
    op_shift_earlier_hi: Option<usize>,
    // RelocateSingleBest::new(usize..=usize)
    op_relocate_best_lo: Option<usize>,
    op_relocate_best_hi: Option<usize>,
    // SwapPairSameBerth
    op_swap_pair_lo: Option<usize>,
    op_swap_pair_hi: Option<usize>,
    // CrossExchangeAcrossBerths
    op_cross_exchange_lo: Option<usize>,
    op_cross_exchange_hi: Option<usize>,
    // OrOptBlockRelocate(k, alpha)
    op_oropt_k_lo: Option<usize>,
    op_oropt_k_hi: Option<usize>,
    op_oropt_alpha_lo: Option<f64>,
    op_oropt_alpha_hi: Option<f64>,
    // RelocateSingleBestAllowWorsening
    op_relocate_worsen_lo: Option<usize>,
    op_relocate_worsen_hi: Option<usize>,
    // RandomRelocateAnywhere
    op_random_relocate_lo: Option<usize>,
    op_random_relocate_hi: Option<usize>,
    // HillClimbRelocateBest
    op_hc_relocate_lo: Option<usize>,
    op_hc_relocate_hi: Option<usize>,
    // HillClimbBestSwapSameBerth
    op_hc_swap_lo: Option<usize>,
    op_hc_swap_hi: Option<usize>,
    // RandomizedGreedyRelocateRcl(k, alpha)
    op_rgreedy_k_lo: Option<usize>,
    op_rgreedy_k_hi: Option<usize>,
    op_rgreedy_alpha_lo: Option<f64>,
    op_rgreedy_alpha_hi: Option<f64>,
    // CrossExchangeBestAcrossBerths
    op_cross_exchange_best_lo: Option<usize>,
    op_cross_exchange_best_hi: Option<usize>,

    // ---------------- ILS destroy/repair ----------------
    // Destroy ratios/params are f64 ranges in 0..1-ish domain (unless otherwise noted).
    d_random_k_ratio_lo: Option<f64>,
    d_random_k_ratio_hi: Option<f64>,
    d_worst_cost_ratio_lo: Option<f64>,
    d_worst_cost_ratio_hi: Option<f64>,
    d_shaw_ratio_lo: Option<f64>,
    d_shaw_ratio_hi: Option<f64>,
    d_shaw_greedy_lo: Option<f64>,
    d_shaw_greedy_hi: Option<f64>,
    d_time_cluster_ratio_lo: Option<f64>,
    d_time_cluster_ratio_hi: Option<f64>,
    d_time_cluster_alpha_lo: Option<f64>,
    d_time_cluster_alpha_hi: Option<f64>,
    d_time_band_ratio_lo: Option<f64>,
    d_time_band_ratio_hi: Option<f64>,
    d_time_band_alpha_lo: Option<f64>,
    d_time_band_alpha_hi: Option<f64>,
    d_berth_band_ratio_lo: Option<f64>,
    d_berth_band_ratio_hi: Option<f64>,
    d_berth_band_greedy_lo: Option<f64>,
    d_berth_band_greedy_hi: Option<f64>,
    d_berth_band_span: Option<usize>,
    d_string_block_ratio_lo: Option<f64>,
    d_string_block_ratio_hi: Option<f64>,
    d_string_block_alpha_lo: Option<f64>,
    d_string_block_alpha_hi: Option<f64>,
    d_berth_neighbors_ratio_lo: Option<f64>,
    d_berth_neighbors_ratio_hi: Option<f64>,
    d_berth_neighbors_greedy_lo: Option<f64>,
    d_berth_neighbors_greedy_hi: Option<f64>,
    d_ptime_cluster_ratio_lo: Option<f64>,
    d_ptime_cluster_ratio_hi: Option<f64>,
    d_ptime_cluster_greedy_lo: Option<f64>,
    d_ptime_cluster_greedy_hi: Option<f64>,

    // ILS repair
    r_kregret_k_lo: Option<usize>,
    r_kregret_k_hi: Option<usize>,
    r_rand_greedy_alpha_lo: Option<f64>,
    r_rand_greedy_alpha_hi: Option<f64>,
}

// ------------------------ Sampling helpers ------------------------

fn jitter_usize_range(
    rng: &mut ChaCha8Rng,
    base_lo: usize,
    base_hi: usize,
    jit_lo: i32,
    jit_hi: i32,
    min_lo: usize,
    max_hi: usize,
) -> (usize, usize) {
    // Sample i32 deltas and do arithmetic in i64 to avoid isize sampling
    let dlo = rng.random_range(jit_lo..=jit_hi) as i64;
    let dhi = rng.random_range(jit_lo..=jit_hi) as i64;
    let mut lo = base_lo as i64 + dlo;
    let mut hi = base_hi as i64 + dhi;
    if lo < min_lo as i64 {
        lo = min_lo as i64;
    }
    if hi < lo {
        hi = lo;
    }
    if hi > max_hi as i64 {
        hi = max_hi as i64;
    }
    (lo as usize, hi as usize)
}

fn jitter_f64_range(
    rng: &mut ChaCha8Rng,
    base_lo: f64,
    base_hi: f64,
    jit_abs: f64,
    min_lo: f64,
    max_hi: f64,
) -> (f64, f64) {
    let jl = rng.random_range(-jit_abs..=jit_abs);
    let jh = rng.random_range(-jit_abs..=jit_abs);
    let mut lo = base_lo + jl;
    let mut hi = base_hi + jh;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    (lo.clamp(min_lo, max_hi), hi.clamp(lo, max_hi))
}

// ------------------------ Sampling ------------------------

fn sample_params(kind: StrategyKind, rng: &mut ChaCha8Rng) -> TunableParams {
    match kind {
        StrategyKind::Gls => {
            // Defaults from gls::gls_strategy with light jitter
            let (se_lo, se_hi) = jitter_usize_range(rng, 18, 52, -4, 6, 6, 96);
            let (rb_lo, rb_hi) = jitter_usize_range(rng, 20, 64, -6, 8, 6, 96);
            let (sp_lo, sp_hi) = jitter_usize_range(rng, 36, 96, -8, 8, 8, 140);
            let (ce_lo, ce_hi) = jitter_usize_range(rng, 48, 128, -8, 10, 8, 160);
            let (oo_k_lo, oo_k_hi) = jitter_usize_range(rng, 5, 9, -1, 2, 2, 16);
            let (oo_a_lo, oo_a_hi) = jitter_f64_range(rng, 1.4, 1.9, 0.15, 1.0, 3.0);
            let (rw_lo, rw_hi) = jitter_usize_range(rng, 12, 24, -2, 2, 4, 40);
            let (rr_lo, rr_hi) = jitter_usize_range(rng, 12, 24, -2, 4, 4, 48);
            let (hc_lo, hc_hi) = jitter_usize_range(rng, 24, 72, -4, 6, 8, 120);
            let (hs_lo, hs_hi) = jitter_usize_range(rng, 48, 120, -8, 10, 12, 160);
            let (rg_k_lo, rg_k_hi) = jitter_usize_range(rng, 18, 48, -4, 4, 6, 64);
            let (rg_a_lo, rg_a_hi) = jitter_f64_range(rng, 1.5, 2.2, 0.15, 1.0, 3.0);
            let (ceb_lo, ceb_hi) = jitter_usize_range(rng, 32, 96, -6, 8, 8, 140);

            TunableParams {
                // GLS core
                lambda: Some([7, 8, 9, 10].choose(rng).copied().unwrap_or(9)),
                penalty_step: Some([2, 3].choose(rng).copied().unwrap_or(2)),
                decay_num: Some(rng.random_range(92..=96)),
                decay_den: Some(100),
                pulse_top_k: Some(rng.random_range(16..=28)),
                stagnation_rounds: Some(rng.random_range(6..=12)),
                max_local_steps: Some(rng.random_range(1600..=2400)),
                max_penalty: Some(1_000_000_000),
                hard_refetch_every: Some(rng.random_range(18..=36)),
                refetch_after_stale: Some(rng.random_range(48..=120)),

                // Operator params
                op_shift_earlier_lo: Some(se_lo),
                op_shift_earlier_hi: Some(se_hi),
                op_relocate_best_lo: Some(rb_lo),
                op_relocate_best_hi: Some(rb_hi),
                op_swap_pair_lo: Some(sp_lo),
                op_swap_pair_hi: Some(sp_hi),
                op_cross_exchange_lo: Some(ce_lo),
                op_cross_exchange_hi: Some(ce_hi),
                op_oropt_k_lo: Some(oo_k_lo),
                op_oropt_k_hi: Some(oo_k_hi),
                op_oropt_alpha_lo: Some(oo_a_lo),
                op_oropt_alpha_hi: Some(oo_a_hi),
                op_relocate_worsen_lo: Some(rw_lo),
                op_relocate_worsen_hi: Some(rw_hi),
                op_random_relocate_lo: Some(rr_lo),
                op_random_relocate_hi: Some(rr_hi),
                op_hc_relocate_lo: Some(hc_lo),
                op_hc_relocate_hi: Some(hc_hi),
                op_hc_swap_lo: Some(hs_lo),
                op_hc_swap_hi: Some(hs_hi),
                op_rgreedy_k_lo: Some(rg_k_lo),
                op_rgreedy_k_hi: Some(rg_k_hi),
                op_rgreedy_alpha_lo: Some(rg_a_lo),
                op_rgreedy_alpha_hi: Some(rg_a_hi),
                op_cross_exchange_best_lo: Some(ceb_lo),
                op_cross_exchange_best_hi: Some(ceb_hi),

                ..Default::default()
            }
        }
        StrategyKind::Tabu => {
            let (se_lo, se_hi) = jitter_usize_range(rng, 18, 52, -4, 6, 6, 96);
            let (rb_lo, rb_hi) = jitter_usize_range(rng, 20, 64, -6, 8, 6, 96);
            let (sp_lo, sp_hi) = jitter_usize_range(rng, 36, 96, -8, 8, 8, 140);
            let (ce_lo, ce_hi) = jitter_usize_range(rng, 48, 128, -8, 10, 8, 160);
            let (oo_k_lo, oo_k_hi) = jitter_usize_range(rng, 5, 9, -1, 2, 2, 16);
            let (oo_a_lo, oo_a_hi) = jitter_f64_range(rng, 1.4, 1.9, 0.15, 1.0, 3.0);
            let (rw_lo, rw_hi) = jitter_usize_range(rng, 12, 24, -2, 2, 4, 40);
            let (rr_lo, rr_hi) = jitter_usize_range(rng, 12, 24, -2, 4, 4, 48);
            let (hc_lo, hc_hi) = jitter_usize_range(rng, 24, 72, -4, 6, 8, 120);
            let (hs_lo, hs_hi) = jitter_usize_range(rng, 48, 120, -8, 10, 12, 160);
            let (rg_k_lo, rg_k_hi) = jitter_usize_range(rng, 18, 48, -4, 4, 6, 64);
            let (rg_a_lo, rg_a_hi) = jitter_f64_range(rng, 1.5, 2.2, 0.15, 1.0, 3.0);
            let (ceb_lo, ceb_hi) = jitter_usize_range(rng, 32, 96, -6, 8, 8, 140);

            TunableParams {
                // Tabu core
                lambda: Some([6, 7, 8].choose(rng).copied().unwrap_or(7)),
                penalty_step: Some(2),
                decay_num: Some(rng.random_range(92..=96)),
                decay_den: Some(100),
                pulse_top_k: Some(rng.random_range(16..=24)),
                stagnation_rounds: Some(rng.random_range(6..=12)),
                max_local_steps: Some(rng.random_range(1700..=2200)),
                max_penalty: Some(1_000_000_000),
                hard_refetch_every: Some(rng.random_range(18..=36)),
                refetch_after_stale: Some(rng.random_range(48..=120)),

                tabu_tenure_lo: Some(rng.random_range(28..=44)),
                tabu_tenure_hi: Some(rng.random_range(48..=72)),
                tabu_samples_per_step: Some(rng.random_range(110..=170)),

                // Operators
                op_shift_earlier_lo: Some(se_lo),
                op_shift_earlier_hi: Some(se_hi),
                op_relocate_best_lo: Some(rb_lo),
                op_relocate_best_hi: Some(rb_hi),
                op_swap_pair_lo: Some(sp_lo),
                op_swap_pair_hi: Some(sp_hi),
                op_cross_exchange_lo: Some(ce_lo),
                op_cross_exchange_hi: Some(ce_hi),
                op_oropt_k_lo: Some(oo_k_lo),
                op_oropt_k_hi: Some(oo_k_hi),
                op_oropt_alpha_lo: Some(oo_a_lo),
                op_oropt_alpha_hi: Some(oo_a_hi),
                op_relocate_worsen_lo: Some(rw_lo),
                op_relocate_worsen_hi: Some(rw_hi),
                op_random_relocate_lo: Some(rr_lo),
                op_random_relocate_hi: Some(rr_hi),
                op_hc_relocate_lo: Some(hc_lo),
                op_hc_relocate_hi: Some(hc_hi),
                op_hc_swap_lo: Some(hs_lo),
                op_hc_swap_hi: Some(hs_hi),
                op_rgreedy_k_lo: Some(rg_k_lo),
                op_rgreedy_k_hi: Some(rg_k_hi),
                op_rgreedy_alpha_lo: Some(rg_a_lo),
                op_rgreedy_alpha_hi: Some(rg_a_hi),
                op_cross_exchange_best_lo: Some(ceb_lo),
                op_cross_exchange_best_hi: Some(ceb_hi),

                ..Default::default()
            }
        }
        StrategyKind::Ils => {
            // Local ops (same base as GLS/SA/TABU)
            let (se_lo, se_hi) = jitter_usize_range(rng, 18, 52, -4, 6, 6, 96);
            let (rb_lo, rb_hi) = jitter_usize_range(rng, 20, 64, -6, 8, 6, 96);
            let (sp_lo, sp_hi) = jitter_usize_range(rng, 36, 96, -8, 8, 8, 140);
            let (ce_lo, ce_hi) = jitter_usize_range(rng, 48, 128, -8, 10, 8, 160);
            let (oo_k_lo, oo_k_hi) = jitter_usize_range(rng, 6, 10, -1, 2, 2, 16);
            let (oo_a_lo, oo_a_hi) = jitter_f64_range(rng, 1.3, 1.8, 0.15, 1.0, 3.0);
            let (rw_lo, rw_hi) = jitter_usize_range(rng, 12, 24, -2, 2, 4, 40);
            let (rr_lo, rr_hi) = jitter_usize_range(rng, 12, 24, -2, 4, 4, 48);
            let (hc_lo, hc_hi) = jitter_usize_range(rng, 24, 72, -4, 6, 8, 120);
            let (hs_lo, hs_hi) = jitter_usize_range(rng, 48, 120, -8, 10, 12, 160);
            let (rg_k_lo, rg_k_hi) = jitter_usize_range(rng, 18, 48, -4, 4, 6, 64);
            let (rg_a_lo, rg_a_hi) = jitter_f64_range(rng, 1.5, 2.1, 0.15, 1.0, 3.0);
            let (ceb_lo, ceb_hi) = jitter_usize_range(rng, 32, 96, -6, 8, 8, 140);

            // Destroy/repair bases from ils_strategy
            let (d_random_lo, d_random_hi) = jitter_f64_range(rng, 0.26, 0.42, 0.04, 0.1, 0.9);
            let (d_worst_lo, d_worst_hi) = jitter_f64_range(rng, 0.28, 0.42, 0.04, 0.1, 0.9);
            let (shaw_ratio_lo, shaw_ratio_hi) = jitter_f64_range(rng, 0.24, 0.36, 0.04, 0.1, 0.9);
            let (shaw_greed_lo, shaw_greed_hi) = jitter_f64_range(rng, 1.6, 2.2, 0.2, 1.0, 3.0);
            let (tc_ratio_lo, tc_ratio_hi) = jitter_f64_range(rng, 0.28, 0.42, 0.04, 0.1, 0.9);
            let (tc_alpha_lo, tc_alpha_hi) = jitter_f64_range(rng, 1.5, 1.75, 0.2, 1.0, 3.0);
            let (tb_ratio_lo, tb_ratio_hi) = jitter_f64_range(rng, 0.44, 0.56, 0.05, 0.2, 0.9);
            let (tb_alpha_lo, tb_alpha_hi) = jitter_f64_range(rng, 1.4, 1.9, 0.2, 1.0, 3.0);
            let (bb_ratio_lo, bb_ratio_hi) = jitter_f64_range(rng, 0.26, 0.40, 0.05, 0.1, 0.9);
            let (bb_greed_lo, bb_greed_hi) = jitter_f64_range(rng, 1.4, 1.9, 0.2, 1.0, 3.0);
            let bb_span = [1usize, 2usize].choose(rng).copied().unwrap_or(1);
            let (sb_ratio_lo, sb_ratio_hi) = jitter_f64_range(rng, 0.32, 0.46, 0.05, 0.1, 0.9);
            let (sb_alpha_lo, sb_alpha_hi) = jitter_f64_range(rng, 1.5, 2.0, 0.2, 1.0, 3.0);
            let (bn_ratio_lo, bn_ratio_hi) = jitter_f64_range(rng, 0.28, 0.44, 0.05, 0.1, 0.9);
            let (bn_greedy_lo, bn_greedy_hi) = jitter_f64_range(rng, 1.4, 1.8, 0.2, 1.0, 3.0);
            let (pt_ratio_lo, pt_ratio_hi) = jitter_f64_range(rng, 0.22, 0.34, 0.05, 0.1, 0.9);
            let (pt_greedy_lo, pt_greedy_hi) = jitter_f64_range(rng, 1.7, 2.0, 0.2, 1.0, 3.0);

            TunableParams {
                local_lo: Some(rng.random_range(800..=1400)),
                local_hi: Some(rng.random_range(1400..=2000)),
                allow_sideways: Some(true),
                worsen_prob: Some(rng.random_range(8..=16) as f64 / 1000.0),
                destroy_attempts: Some(rng.random_range(12..=20)),
                repair_attempts: Some(rng.random_range(18..=36)),
                refetch_after_stale: Some(rng.random_range(40..=80)),
                hard_refetch_every: Some(rng.random_range(18..=36)),

                // Local ops
                op_shift_earlier_lo: Some(se_lo),
                op_shift_earlier_hi: Some(se_hi),
                op_relocate_best_lo: Some(rb_lo),
                op_relocate_best_hi: Some(rb_hi),
                op_swap_pair_lo: Some(sp_lo),
                op_swap_pair_hi: Some(sp_hi),
                op_cross_exchange_lo: Some(ce_lo),
                op_cross_exchange_hi: Some(ce_hi),
                op_oropt_k_lo: Some(oo_k_lo),
                op_oropt_k_hi: Some(oo_k_hi),
                op_oropt_alpha_lo: Some(oo_a_lo),
                op_oropt_alpha_hi: Some(oo_a_hi),
                op_relocate_worsen_lo: Some(rw_lo),
                op_relocate_worsen_hi: Some(rw_hi),
                op_random_relocate_lo: Some(rr_lo),
                op_random_relocate_hi: Some(rr_hi),
                op_hc_relocate_lo: Some(hc_lo),
                op_hc_relocate_hi: Some(hc_hi),
                op_hc_swap_lo: Some(hs_lo),
                op_hc_swap_hi: Some(hs_hi),
                op_rgreedy_k_lo: Some(rg_k_lo),
                op_rgreedy_k_hi: Some(rg_k_hi),
                op_rgreedy_alpha_lo: Some(rg_a_lo),
                op_rgreedy_alpha_hi: Some(rg_a_hi),
                op_cross_exchange_best_lo: Some(ceb_lo),
                op_cross_exchange_best_hi: Some(ceb_hi),

                // Destroy
                d_random_k_ratio_lo: Some(d_random_lo),
                d_random_k_ratio_hi: Some(d_random_hi),
                d_worst_cost_ratio_lo: Some(d_worst_lo),
                d_worst_cost_ratio_hi: Some(d_worst_hi),
                d_shaw_ratio_lo: Some(shaw_ratio_lo),
                d_shaw_ratio_hi: Some(shaw_ratio_hi),
                d_shaw_greedy_lo: Some(shaw_greed_lo),
                d_shaw_greedy_hi: Some(shaw_greed_hi),
                d_time_cluster_ratio_lo: Some(tc_ratio_lo),
                d_time_cluster_ratio_hi: Some(tc_ratio_hi),
                d_time_cluster_alpha_lo: Some(tc_alpha_lo),
                d_time_cluster_alpha_hi: Some(tc_alpha_hi),
                d_time_band_ratio_lo: Some(tb_ratio_lo),
                d_time_band_ratio_hi: Some(tb_ratio_hi),
                d_time_band_alpha_lo: Some(tb_alpha_lo),
                d_time_band_alpha_hi: Some(tb_alpha_hi),
                d_berth_band_ratio_lo: Some(bb_ratio_lo),
                d_berth_band_ratio_hi: Some(bb_ratio_hi),
                d_berth_band_greedy_lo: Some(bb_greed_lo),
                d_berth_band_greedy_hi: Some(bb_greed_hi),
                d_berth_band_span: Some(bb_span),
                d_string_block_ratio_lo: Some(sb_ratio_lo),
                d_string_block_ratio_hi: Some(sb_ratio_hi),
                d_string_block_alpha_lo: Some(sb_alpha_lo),
                d_string_block_alpha_hi: Some(sb_alpha_hi),
                d_berth_neighbors_ratio_lo: Some(bn_ratio_lo),
                d_berth_neighbors_ratio_hi: Some(bn_ratio_hi),
                d_berth_neighbors_greedy_lo: Some(bn_greedy_lo),
                d_berth_neighbors_greedy_hi: Some(bn_greedy_hi),
                d_ptime_cluster_ratio_lo: Some(pt_ratio_lo),
                d_ptime_cluster_ratio_hi: Some(pt_ratio_hi),
                d_ptime_cluster_greedy_lo: Some(pt_greedy_lo),
                d_ptime_cluster_greedy_hi: Some(pt_greedy_hi),

                // Repair
                r_kregret_k_lo: Some(rng.random_range(8..=10)),
                r_kregret_k_hi: Some(rng.random_range(10..=12)),
                r_rand_greedy_alpha_lo: Some(1.4),
                r_rand_greedy_alpha_hi: Some(2.1),

                ..Default::default()
            }
        }
        StrategyKind::Sa => {
            let (se_lo, se_hi) = jitter_usize_range(rng, 18, 52, -4, 6, 6, 96);
            let (rb_lo, rb_hi) = jitter_usize_range(rng, 20, 64, -6, 8, 6, 96);
            let (sp_lo, sp_hi) = jitter_usize_range(rng, 36, 96, -8, 8, 8, 140);
            let (ce_lo, ce_hi) = jitter_usize_range(rng, 48, 128, -8, 10, 8, 160);
            let (oo_k_lo, oo_k_hi) = jitter_usize_range(rng, 5, 9, -1, 2, 2, 16);
            let (oo_a_lo, oo_a_hi) = jitter_f64_range(rng, 1.4, 1.9, 0.15, 1.0, 3.0);
            let (rw_lo, rw_hi) = jitter_usize_range(rng, 12, 24, -2, 2, 4, 40);
            let (rr_lo, rr_hi) = jitter_usize_range(rng, 12, 24, -2, 4, 4, 48);
            let (hc_lo, hc_hi) = jitter_usize_range(rng, 24, 72, -4, 6, 8, 120);
            let (hs_lo, hs_hi) = jitter_usize_range(rng, 48, 120, -8, 10, 12, 160);
            let (rg_k_lo, rg_k_hi) = jitter_usize_range(rng, 18, 48, -4, 4, 6, 64);
            let (rg_a_lo, rg_a_hi) = jitter_f64_range(rng, 1.5, 2.2, 0.15, 1.0, 3.0);
            let (ceb_lo, ceb_hi) = jitter_usize_range(rng, 32, 96, -6, 8, 8, 140);

            TunableParams {
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

                // Local operators
                op_shift_earlier_lo: Some(se_lo),
                op_shift_earlier_hi: Some(se_hi),
                op_relocate_best_lo: Some(rb_lo),
                op_relocate_best_hi: Some(rb_hi),
                op_swap_pair_lo: Some(sp_lo),
                op_swap_pair_hi: Some(sp_hi),
                op_cross_exchange_lo: Some(ce_lo),
                op_cross_exchange_hi: Some(ce_hi),
                op_oropt_k_lo: Some(oo_k_lo),
                op_oropt_k_hi: Some(oo_k_hi),
                op_oropt_alpha_lo: Some(oo_a_lo),
                op_oropt_alpha_hi: Some(oo_a_hi),
                op_relocate_worsen_lo: Some(rw_lo),
                op_relocate_worsen_hi: Some(rw_hi),
                op_random_relocate_lo: Some(rr_lo),
                op_random_relocate_hi: Some(rr_hi),
                op_hc_relocate_lo: Some(hc_lo),
                op_hc_relocate_hi: Some(hc_hi),
                op_hc_swap_lo: Some(hs_lo),
                op_hc_swap_hi: Some(hs_hi),
                op_rgreedy_k_lo: Some(rg_k_lo),
                op_rgreedy_k_hi: Some(rg_k_hi),
                op_rgreedy_alpha_lo: Some(rg_a_lo),
                op_rgreedy_alpha_hi: Some(rg_a_hi),
                op_cross_exchange_best_lo: Some(ceb_lo),
                op_cross_exchange_best_hi: Some(ceb_hi),

                ..Default::default()
            }
        }
    }
}

// ------------------------ Strategy builders ------------------------

fn or_range_usize(
    lo: Option<usize>,
    hi: Option<usize>,
    default: RangeInclusive<usize>,
) -> RangeInclusive<usize> {
    match (lo, hi) {
        (Some(a), Some(b)) => a..=b,
        _ => default,
    }
}
fn or_range_f64(
    lo: Option<f64>,
    hi: Option<f64>,
    default: RangeInclusive<f64>,
) -> RangeInclusive<f64> {
    match (lo, hi) {
        (Some(a), Some(b)) => a..=b,
        _ => default,
    }
}
fn or_usize(v: Option<usize>, default: usize) -> usize {
    v.unwrap_or(default)
}
fn or_i64(v: Option<i64>, default: i64) -> i64 {
    v.unwrap_or(default)
}
fn or_bool(v: Option<bool>, default: bool) -> bool {
    v.unwrap_or(default)
}
fn or_f64(v: Option<f64>, default: f64) -> f64 {
    v.unwrap_or(default)
}
fn or_u32_from_usize(v: Option<usize>, default: u32) -> u32 {
    v.map(|x| x as u32).unwrap_or(default)
}

#[allow(clippy::type_complexity)]
fn build_gls_strategy<
    T: berth_alloc_solver::core::numeric::SolveNumeric + ToPrimitive + Copy + From<i32>,
>(
    model: &berth_alloc_solver::model::solver_model::SolverModel<T>,
    params: &TunableParams,
) -> gls::GuidedLocalSearchStrategy<
    T,
    ChaCha8Rng,
    gls::DefaultFeatureExtractor<T, fn(TimePoint<T>) -> i64>,
> {
    // Bucketizer like gls.rs
    let bucketizer: fn(TimePoint<T>) -> i64 = |t: TimePoint<T>| -> i64 {
        let v_i64 = t
            .value()
            .to_i64()
            .expect("TimePoint<T>::value() must be convertible to i64 for bucketing");
        v_i64 / 75
    };
    let feats = gls::DefaultFeatureExtractor::new(bucketizer)
        .set_include_req_berth(true)
        .set_include_time(true)
        .set_include_berth_time(true)
        .set_include_req_time(true)
        .set_include_berth(true)
        .set_include_request(true);
    let feats_arc = std::sync::Arc::new(feats);

    let proximity_map = model.proximity_map();
    let neighbors_any = berth_alloc_solver::engine::neighbors::any(proximity_map);
    let neighbors_direct_competitors =
        berth_alloc_solver::engine::neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = berth_alloc_solver::engine::neighbors::same_berth(proximity_map);

    let mut s = gls::GuidedLocalSearchStrategy::new(feats_arc);

    // Core params
    if let Some(v) = params.lambda {
        s = s.with_lambda(v);
    } else {
        s = s.with_lambda(9);
    }
    if let Some(v) = params.penalty_step {
        s = s.with_penalty_step(v);
    } else {
        s = s.with_penalty_step(2);
    }
    if let (Some(num), Some(den)) = (params.decay_num, params.decay_den) {
        s = s.with_decay(gls::DecayMode::Multiplicative { num, den });
    } else {
        s = s.with_decay(gls::DecayMode::Multiplicative { num: 95, den: 100 });
    }
    s = s.with_max_penalty(or_i64(params.max_penalty, 1_000_000_000));
    s = s.with_pulse_params(
        or_usize(params.stagnation_rounds, 8),
        or_usize(params.pulse_top_k, 20),
    );
    s = s.with_max_local_steps(or_usize(params.max_local_steps, 2100));
    s = s.with_refetch_after_stale(or_usize(params.refetch_after_stale, 60));
    s = s.with_hard_refetch_every(or_usize(params.hard_refetch_every, 24));
    s = s.with_hard_refetch_mode(gls::HardRefetchMode::IfBetter);
    s = s.with_restart_on_publish(true);
    s = s.with_reset_on_refetch(true);
    s = s.with_kick_steps_on_reset(6);

    // Local operators (ranges may be overridden by params)
    use berth_alloc_solver::search::operator_library::local as loc;
    s = s.with_local_op(Box::new(
        loc::ShiftEarlierOnSameBerth::new(or_range_usize(
            params.op_shift_earlier_lo,
            params.op_shift_earlier_hi,
            18..=52,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBest::new(or_range_usize(
            params.op_relocate_best_lo,
            params.op_relocate_best_hi,
            20..=64,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::SwapPairSameBerth::new(or_range_usize(
            params.op_swap_pair_lo,
            params.op_swap_pair_hi,
            36..=96,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_lo,
            params.op_cross_exchange_hi,
            48..=128,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::OrOptBlockRelocate::new(
            or_range_usize(params.op_oropt_k_lo, params.op_oropt_k_hi, 5..=9),
            or_range_f64(
                params.op_oropt_alpha_lo,
                params.op_oropt_alpha_hi,
                1.4..=1.9,
            ),
        )
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBestAllowWorsening::new(or_range_usize(
            params.op_relocate_worsen_lo,
            params.op_relocate_worsen_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomRelocateAnywhere::new(or_range_usize(
            params.op_random_relocate_lo,
            params.op_random_relocate_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbRelocateBest::new(or_range_usize(
            params.op_hc_relocate_lo,
            params.op_hc_relocate_hi,
            24..=72,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbBestSwapSameBerth::new(or_range_usize(
            params.op_hc_swap_lo,
            params.op_hc_swap_hi,
            48..=120,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomizedGreedyRelocateRcl::new(
            or_range_usize(params.op_rgreedy_k_lo, params.op_rgreedy_k_hi, 18..=48),
            or_range_f64(
                params.op_rgreedy_alpha_lo,
                params.op_rgreedy_alpha_hi,
                1.5..=2.2,
            ),
        )
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeBestAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_best_lo,
            params.op_cross_exchange_best_hi,
            32..=96,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));

    s
}

fn build_tabu_strategy<
    T: berth_alloc_solver::core::numeric::SolveNumeric + ToPrimitive + Copy + From<i32>,
>(
    model: &berth_alloc_solver::model::solver_model::SolverModel<T>,
    params: &TunableParams,
) -> tabu::TabuSearchStrategy<
    T,
    ChaCha8Rng,
    berth_alloc_solver::engine::feature_signal::prelude::DefaultFeatureExtractor<T>,
> {
    use berth_alloc_solver::engine::feature_signal::prelude::DecayMode;
    use berth_alloc_solver::engine::feature_signal::prelude::DefaultFeatureExtractor as Feat;
    // bucketizer like tabu.rs
    let bucketizer: fn(TimePoint<T>) -> i64 = |t: TimePoint<T>| {
        let v = t
            .value()
            .to_i64()
            .expect("TimePoint<T>::value() must be convertible to i64 for bucketing");
        v / 90
    };
    let feats = Feat::new(bucketizer)
        .set_include_req_berth(true)
        .set_include_time(true)
        .set_include_berth_time(true)
        .set_include_req_time(true)
        .set_include_berth(true)
        .set_include_request(true);
    let feats_arc = std::sync::Arc::new(feats);

    let proximity_map = model.proximity_map();
    let neighbors_any = berth_alloc_solver::engine::neighbors::any(proximity_map);
    let neighbors_direct_competitors =
        berth_alloc_solver::engine::neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = berth_alloc_solver::engine::neighbors::same_berth(proximity_map);

    let mut s = tabu::TabuSearchStrategy::new(feats_arc);

    s = s.with_lambda(or_i64(params.lambda, 7));
    s = s.with_penalty_step(or_i64(params.penalty_step, 2));
    if let (Some(num), Some(den)) = (params.decay_num, params.decay_den) {
        s = s.with_decay(DecayMode::Multiplicative { num, den });
    } else {
        s = s.with_decay(DecayMode::Multiplicative { num: 94, den: 100 });
    }
    s = s.with_max_penalty(or_i64(params.max_penalty, 1_000_000_000));
    s = s.with_max_local_steps(or_usize(params.max_local_steps, 2000));
    let tenure_lo = or_usize(params.tabu_tenure_lo, 36);
    let tenure_hi = or_usize(params.tabu_tenure_hi, 60);
    s = s.with_tabu_tenure(tenure_lo..=tenure_hi);
    s = s.with_samples_per_step(or_usize(params.tabu_samples_per_step, 140));
    s = s.with_refetch_after_stale(or_usize(params.refetch_after_stale, 60));
    s = s.with_hard_refetch_every(or_usize(params.hard_refetch_every, 24));
    s = s.with_hard_refetch_mode(tabu::HardRefetchMode::IfBetter);
    s = s.with_restart_on_publish(true);
    s = s.with_reset_on_refetch(true);
    s = s.with_kick_steps_on_reset(6);
    s = s.with_pulse_params(
        or_usize(params.stagnation_rounds, 8),
        or_usize(params.pulse_top_k, 18),
    );

    // Local operators
    use berth_alloc_solver::search::operator_library::local as loc;
    s = s.with_local_op(Box::new(
        loc::ShiftEarlierOnSameBerth::new(or_range_usize(
            params.op_shift_earlier_lo,
            params.op_shift_earlier_hi,
            18..=52,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBest::new(or_range_usize(
            params.op_relocate_best_lo,
            params.op_relocate_best_hi,
            20..=64,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::SwapPairSameBerth::new(or_range_usize(
            params.op_swap_pair_lo,
            params.op_swap_pair_hi,
            36..=96,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_lo,
            params.op_cross_exchange_hi,
            48..=128,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::OrOptBlockRelocate::new(
            or_range_usize(params.op_oropt_k_lo, params.op_oropt_k_hi, 5..=9),
            or_range_f64(
                params.op_oropt_alpha_lo,
                params.op_oropt_alpha_hi,
                1.4..=1.9,
            ),
        )
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBestAllowWorsening::new(or_range_usize(
            params.op_relocate_worsen_lo,
            params.op_relocate_worsen_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomRelocateAnywhere::new(or_range_usize(
            params.op_random_relocate_lo,
            params.op_random_relocate_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbRelocateBest::new(or_range_usize(
            params.op_hc_relocate_lo,
            params.op_hc_relocate_hi,
            24..=72,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbBestSwapSameBerth::new(or_range_usize(
            params.op_hc_swap_lo,
            params.op_hc_swap_hi,
            48..=120,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomizedGreedyRelocateRcl::new(
            or_range_usize(params.op_rgreedy_k_lo, params.op_rgreedy_k_hi, 18..=48),
            or_range_f64(
                params.op_rgreedy_alpha_lo,
                params.op_rgreedy_alpha_hi,
                1.5..=2.2,
            ),
        )
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeBestAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_best_lo,
            params.op_cross_exchange_best_hi,
            32..=96,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));

    s
}

fn build_ils_strategy<T: berth_alloc_solver::core::numeric::SolveNumeric + From<i32>>(
    model: &berth_alloc_solver::model::solver_model::SolverModel<T>,
    params: &TunableParams,
) -> ils::IteratedLocalSearchStrategy<T, ChaCha8Rng> {
    let proximity_map = model.proximity_map();
    let neighbors_any = berth_alloc_solver::engine::neighbors::any(proximity_map);
    let neighbors_direct_competitors =
        berth_alloc_solver::engine::neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = berth_alloc_solver::engine::neighbors::same_berth(proximity_map);

    let mut s = ils::IteratedLocalSearchStrategy::new();

    let lo = or_usize(params.local_lo, 900);
    let hi = or_usize(params.local_hi, 1600);
    s = s.with_local_steps_range(lo..=hi);
    s = s.with_local_sideways(or_bool(params.allow_sideways, true));
    s = s.with_local_worsening_prob(or_f64(params.worsen_prob, 0.015));
    s = s.with_destroy_attempts(or_usize(params.destroy_attempts, 12));
    s = s.with_repair_attempts(or_usize(params.repair_attempts, 28));
    s = s.with_shuffle_local_each_step(true);
    s = s.with_refetch_after_stale(or_usize(params.refetch_after_stale, 45));
    s = s.with_hard_refetch_every(or_usize(params.hard_refetch_every, 24));
    s = s.with_hard_refetch_mode(ils::HardRefetchMode::IfBetter);

    // Local ops
    use berth_alloc_solver::search::operator_library::local as loc;
    s = s.with_local_op(Box::new(
        loc::ShiftEarlierOnSameBerth::new(or_range_usize(
            params.op_shift_earlier_lo,
            params.op_shift_earlier_hi,
            18..=52,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBest::new(or_range_usize(
            params.op_relocate_best_lo,
            params.op_relocate_best_hi,
            20..=64,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::SwapPairSameBerth::new(or_range_usize(
            params.op_swap_pair_lo,
            params.op_swap_pair_hi,
            36..=96,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_lo,
            params.op_cross_exchange_hi,
            48..=128,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::OrOptBlockRelocate::new(
            or_range_usize(params.op_oropt_k_lo, params.op_oropt_k_hi, 6..=10),
            or_range_f64(
                params.op_oropt_alpha_lo,
                params.op_oropt_alpha_hi,
                1.3..=1.8,
            ),
        )
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBestAllowWorsening::new(or_range_usize(
            params.op_relocate_worsen_lo,
            params.op_relocate_worsen_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomRelocateAnywhere::new(or_range_usize(
            params.op_random_relocate_lo,
            params.op_random_relocate_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbRelocateBest::new(or_range_usize(
            params.op_hc_relocate_lo,
            params.op_hc_relocate_hi,
            24..=72,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbBestSwapSameBerth::new(or_range_usize(
            params.op_hc_swap_lo,
            params.op_hc_swap_hi,
            48..=120,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));

    // Destroy operators
    use berth_alloc_solver::search::operator_library::destroy as des;
    s = s.with_destroy_op(Box::new(
        des::RandomKRatioDestroy::new(or_range_f64(
            params.d_random_k_ratio_lo,
            params.d_random_k_ratio_hi,
            0.26..=0.42,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_destroy_op(Box::new(
        des::WorstCostDestroy::new(or_range_f64(
            params.d_worst_cost_ratio_lo,
            params.d_worst_cost_ratio_hi,
            0.28..=0.42,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_destroy_op(Box::new(
        des::ShawRelatedDestroy::new(
            or_range_f64(params.d_shaw_ratio_lo, params.d_shaw_ratio_hi, 0.24..=0.36),
            or_range_f64(params.d_shaw_greedy_lo, params.d_shaw_greedy_hi, 1.6..=2.2),
            1.into(),
            1.into(),
            5.into(),
        )
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_destroy_op(Box::new(
        des::TimeClusterDestroy::<T>::new(
            or_range_f64(
                params.d_time_cluster_ratio_lo,
                params.d_time_cluster_ratio_hi,
                0.28..=0.42,
            ),
            TimeDelta::new(24.into()),
        )
        .with_alpha(or_range_f64(
            params.d_time_cluster_alpha_lo,
            params.d_time_cluster_alpha_hi,
            1.5..=1.75,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_destroy_op(Box::new(
        des::TimeWindowBandDestroy::<T>::new(
            or_range_f64(
                params.d_time_band_ratio_lo,
                params.d_time_band_ratio_hi,
                0.44..=0.56,
            ),
            or_range_f64(
                params.d_time_band_alpha_lo,
                params.d_time_band_alpha_hi,
                1.4..=1.9,
            ),
            TimeDelta::new(16.into()),
        )
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_destroy_op(Box::new(des::BerthBandDestroy::new(
        or_range_f64(
            params.d_berth_band_ratio_lo,
            params.d_berth_band_ratio_hi,
            0.26..=0.40,
        ),
        or_range_f64(
            params.d_berth_band_greedy_lo,
            params.d_berth_band_greedy_hi,
            1.4..=1.9,
        ),
        or_u32_from_usize(params.d_berth_band_span, 1),
    )));
    s = s.with_destroy_op(Box::new(
        des::StringBlockDestroy::new(or_range_f64(
            params.d_string_block_ratio_lo,
            params.d_string_block_ratio_hi,
            0.32..=0.46,
        ))
        .with_alpha(or_range_f64(
            params.d_string_block_alpha_lo,
            params.d_string_block_alpha_hi,
            1.5..=2.0,
        )),
    ));
    s = s.with_destroy_op(Box::new(
        des::BerthNeighborsDestroy::new(
            or_range_f64(
                params.d_berth_neighbors_ratio_lo,
                params.d_berth_neighbors_ratio_hi,
                0.28..=0.44,
            ),
            or_range_f64(
                params.d_berth_neighbors_greedy_lo,
                params.d_berth_neighbors_greedy_hi,
                1.4..=1.8,
            ),
        )
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_destroy_op(Box::new(
        des::ProcessingTimeClusterDestroy::new(
            or_range_f64(
                params.d_ptime_cluster_ratio_lo,
                params.d_ptime_cluster_ratio_hi,
                0.22..=0.34,
            ),
            or_range_f64(
                params.d_ptime_cluster_greedy_lo,
                params.d_ptime_cluster_greedy_hi,
                1.7..=2.0,
            ),
        )
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));

    // Repair operators
    use berth_alloc_solver::search::operator_library::repair as rep;
    s = s.with_repair_op(Box::new(rep::KRegretInsertion::new(or_range_usize(
        params.r_kregret_k_lo,
        params.r_kregret_k_hi,
        9..=11,
    ))));
    s = s.with_repair_op(Box::new(rep::RandomizedGreedyInsertion::new(or_range_f64(
        params.r_rand_greedy_alpha_lo,
        params.r_rand_greedy_alpha_hi,
        1.4..=2.0,
    ))));
    s = s.with_repair_op(Box::new(rep::GreedyInsertion));

    // Post-repair shakers
    s = s.with_local_op(Box::new(
        loc::RandomizedGreedyRelocateRcl::new(
            or_range_usize(params.op_rgreedy_k_lo, params.op_rgreedy_k_hi, 18..=48),
            or_range_f64(
                params.op_rgreedy_alpha_lo,
                params.op_rgreedy_alpha_hi,
                1.5..=2.1,
            ),
        )
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeBestAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_best_lo,
            params.op_cross_exchange_best_hi,
            32..=96,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));

    s
}

fn build_sa_strategy<T: berth_alloc_solver::core::numeric::SolveNumeric + From<i32>>(
    model: &berth_alloc_solver::model::solver_model::SolverModel<T>,
    params: &TunableParams,
) -> sa::SimulatedAnnealingStrategy<T, ChaCha8Rng> {
    let proximity_map = model.proximity_map();
    let neighbors_any = berth_alloc_solver::engine::neighbors::any(proximity_map);
    let neighbors_direct_competitors =
        berth_alloc_solver::engine::neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = berth_alloc_solver::engine::neighbors::same_berth(proximity_map);

    let mut s = sa::SimulatedAnnealingStrategy::new();

    s = s.with_init_temp(or_f64(params.sa_init_temp, 1.7));
    s = s.with_cooling(or_f64(params.sa_cooling, 0.99925));
    s = s.with_steps_per_temp(or_usize(params.sa_steps_per_temp, 600));
    s = s.with_refetch_after_stale(or_usize(params.refetch_after_stale, 60));
    s = s.with_hard_refetch_every(or_usize(params.hard_refetch_every, 24));
    s = s.with_hard_refetch_mode(sa::HardRefetchMode::IfBetter);
    s = s.with_reheat_factor(or_f64(params.sa_reheat_factor, 0.6));
    s = s.with_op_ema_alpha(or_f64(params.sa_op_ema_alpha, 0.30));
    s = s.with_op_min_weight(or_f64(params.sa_op_min_weight, 0.10));
    let ar_lo = or_f64(params.sa_ar_low, 0.10);
    let ar_hi = or_f64(params.sa_ar_high, 0.52);
    s = s.with_acceptance_targets(ar_lo, ar_hi);
    s = s.with_big_m_for_energy(or_i64(params.sa_big_m, 1_250_000_000));

    use berth_alloc_solver::search::operator_library::local as loc;
    s = s.with_local_op(Box::new(
        loc::ShiftEarlierOnSameBerth::new(or_range_usize(
            params.op_shift_earlier_lo,
            params.op_shift_earlier_hi,
            18..=52,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBest::new(or_range_usize(
            params.op_relocate_best_lo,
            params.op_relocate_best_hi,
            20..=64,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::SwapPairSameBerth::new(or_range_usize(
            params.op_swap_pair_lo,
            params.op_swap_pair_hi,
            36..=96,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_lo,
            params.op_cross_exchange_hi,
            48..=128,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::OrOptBlockRelocate::new(
            or_range_usize(params.op_oropt_k_lo, params.op_oropt_k_hi, 5..=9),
            or_range_f64(
                params.op_oropt_alpha_lo,
                params.op_oropt_alpha_hi,
                1.4..=1.9,
            ),
        )
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RelocateSingleBestAllowWorsening::new(or_range_usize(
            params.op_relocate_worsen_lo,
            params.op_relocate_worsen_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomRelocateAnywhere::new(or_range_usize(
            params.op_random_relocate_lo,
            params.op_random_relocate_hi,
            12..=24,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbRelocateBest::new(or_range_usize(
            params.op_hc_relocate_lo,
            params.op_hc_relocate_hi,
            24..=72,
        ))
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::HillClimbBestSwapSameBerth::new(or_range_usize(
            params.op_hc_swap_lo,
            params.op_hc_swap_hi,
            48..=120,
        ))
        .with_neighbors(neighbors_same_berth.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::RandomizedGreedyRelocateRcl::new(
            or_range_usize(params.op_rgreedy_k_lo, params.op_rgreedy_k_hi, 18..=48),
            or_range_f64(
                params.op_rgreedy_alpha_lo,
                params.op_rgreedy_alpha_hi,
                1.5..=2.2,
            ),
        )
        .with_neighbors(neighbors_direct_competitors.clone()),
    ));
    s = s.with_local_op(Box::new(
        loc::CrossExchangeBestAcrossBerths::new(or_range_usize(
            params.op_cross_exchange_best_lo,
            params.op_cross_exchange_best_hi,
            32..=96,
        ))
        .with_neighbors(neighbors_any.clone()),
    ));

    s
}

fn build_boxed_strategy<
    T: berth_alloc_solver::core::numeric::SolveNumeric + ToPrimitive + Copy + From<i32>,
>(
    kind: StrategyKind,
    model: &berth_alloc_solver::model::solver_model::SolverModel<T>,
    params: &TunableParams,
) -> Box<dyn berth_alloc_solver::engine::search::SearchStrategy<T, ChaCha8Rng>> {
    match kind {
        StrategyKind::Gls => Box::new(build_gls_strategy(model, params)),
        StrategyKind::Tabu => Box::new(build_tabu_strategy(model, params)),
        StrategyKind::Ils => Box::new(build_ils_strategy(model, params)),
        StrategyKind::Sa => Box::new(build_sa_strategy(model, params)),
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

// ------------------------ Incremental runner over suite -------------------

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, fields(strategy = kind.as_str(), cfg_seed = cfg_seed))]
fn run_attempt_over_suite_incremental(
    suite_name: &str,
    kind: StrategyKind,
    params: &TunableParams,
    cfg_seed: u64,
    suite: &[(String, Problem<i64>)],
    time_secs_per_instance: u64,
    aggregate: AggregateKind,
    best_path: &Path,
    best_score: &mut Option<f64>,
) -> Option<(f64, usize)> {
    let cfg_id = format!("{}-{:08x}", kind.as_str(), (cfg_seed & 0xffff_ffff));

    let mut total: f64 = 0.0;
    let mut count: usize = 0;

    for (name, prob) in suite {
        let now = Instant::now();

        let model = berth_alloc_solver::model::solver_model::SolverModel::from_problem(prob)
            .expect("model");
        let strat = build_boxed_strategy::<i64>(kind, &model, params);

        let mut engine: SolverEngine<i64> = SolverEngineBuilder::default()
            .with_worker_count(2)
            .with_time_limit(Duration::from_secs(time_secs_per_instance))
            .with_strategy(strat)
            .build();

        match engine.solve(prob) {
            Ok(Some(sol)) => {
                let inst_cost = sol.cost() as f64;
                total += inst_cost;
                count += 1;
                let elapsed = now.elapsed();
                let score_so_far = match aggregate {
                    AggregateKind::Sum => total,
                    AggregateKind::Avg => total / (count as f64),
                };
                info!(
                    strategy = kind.as_str(),
                    cfg_id = cfg_id,
                    instance = name.as_str(),
                    inst_idx = count,
                    instances_total = suite.len(),
                    inst_cost,
                    elapsed_ms = elapsed.as_millis() as u64,
                    score_so_far,
                    "instance solved"
                );
            }
            other => {
                warn!(
                    strategy = kind.as_str(),
                    cfg_id = cfg_id,
                    instance = name.as_str(),
                    ?other,
                    "infeasible/failed config; aborting this config"
                );
                return None;
            }
        }
    }

    // AFTER the loop (i.e., all instances finished OK)
    let final_score = match aggregate {
        AggregateKind::Sum => total,
        AggregateKind::Avg => total / (count as f64),
    };

    // Only consider full-suite results for incumbent update
    if count == suite.len() {
        let better = match best_score {
            None => true,
            Some(s) => final_score < *s,
        };
        if better {
            let rec = BestRecord {
                updated_at: Utc::now().to_rfc3339(),
                strategy: kind.as_str().to_string(),
                suite: suite_name.to_string(),
                cfg_id: cfg_id.clone(),
                params: params.clone(),
                time_secs_per_instance,
                aggregate: aggregate.as_str().to_string(),
                instances: count, // equals suite.len()
                score: final_score,
            };
            if let Err(e) = write_best(best_path, &rec) {
                warn!(error = %e, path = ?best_path, "failed to write best record");
            } else {
                *best_score = Some(final_score);
                info!(
                    strategy = kind.as_str(),
                    cfg_id = cfg_id,
                    score = final_score,
                    instances = count,
                    "new incumbent (full-suite)"
                );
            }
        }
    }

    Some((final_score, count))
}

// ------------------------ Per-strategy sequential runner ------------------------

#[instrument(skip_all, fields(strategy = kind.as_str()))]
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
    if let Some(s) = best_score {
        info!(strategy = kind.as_str(), best_path = ?best_path, best_score = s, "resuming with prior incumbent");
    } else {
        info!(strategy = kind.as_str(), best_path = ?best_path, "no prior incumbent");
    }

    // RNG for this strategy thread; derive per-config seeds from it
    let mut rng = ChaCha8Rng::seed_from_u64(master_seed ^ (kind as u8 as u64));

    for i in 0..trials {
        if Instant::now() >= deadline {
            info!(
                strategy = kind.as_str(),
                trial = i,
                "budget reached, stopping trials"
            );
            break;
        }

        let cfg_seed: u64 = rng.random::<u64>() ^ ((i as u64) << 32);
        let mut sampler = ChaCha8Rng::seed_from_u64(cfg_seed);
        let params = sample_params(kind, &mut sampler);

        info!(
            strategy = kind.as_str(),
            trial = i,
            cfg_seed,
            "starting config"
        );
        let before = best_score;

        let _ = run_attempt_over_suite_incremental(
            suite_name,
            kind,
            &params,
            cfg_seed,
            suite,
            time_secs_per_instance,
            aggregate,
            &best_path,
            &mut best_score,
        );

        match (before, best_score) {
            (Some(b0), Some(b1)) if b1 < b0 => {
                info!(
                    strategy = kind.as_str(),
                    trial = i,
                    best_score = b1,
                    "trial improved incumbent"
                );
            }
            (None, Some(b1)) => {
                info!(
                    strategy = kind.as_str(),
                    trial = i,
                    best_score = b1,
                    "first incumbent established"
                );
            }
            _ => {
                debug!(
                    strategy = kind.as_str(),
                    trial = i,
                    "trial did not improve incumbent"
                );
            }
        }
    }
}

#[allow(dead_code)]
fn enable_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE)
        .init();
}

// ------------------------ Main ------------------------

fn main() -> Result<(), Box<dyn Error>> {
    enable_tracing();

    let cli = parse_cli();
    ensure_dir(&cli.out_dir);

    // Load suite
    let (suite_name, suite) = load_suite(&cli)?;
    if suite.is_empty() {
        return Err("no instances found in the suite".into());
    }

    // Quick banner
    let total_instances = suite.len();
    info!(
        suite = suite_name.as_str(),
        instances = total_instances,
        trials = cli.trials,
        time_secs_per_instance = cli.time_secs_per_instance,
        budget_secs = cli.budget_secs,
        aggregate = cli.aggregate.as_str(),
        filter_strategy = ?cli.strategy,
        "tune configuration"
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

    info!(
        out_dir = %cli.out_dir.display(),
        "tune finished; per-strategy best JSON files written incrementally"
    );
    Ok(())
}
