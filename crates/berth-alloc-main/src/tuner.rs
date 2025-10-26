//! Tuner: saturates CPU with many parallel tuners (one engine per thread, one strategy per engine).
//! - 20s time limit per trial, runs forever until you Ctrl-C.
//! - Wide param spaces for GLS / ILS / SA.
//! - Online archive per strategy: TPE-like exploit (mutate good configs) vs. explore (new samples),
//!   with Thompson sampling to choose mode.
//! - Writes JSONL trial logs + rolling best.json per strategy.
//!
//! Add to Cargo.toml if missing:
//! [dependencies]
//! rand = "0.8"
//! rand_chacha = "0.3"
//! rand_distr = "0.4"
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! chrono = { version = "0.4", features = ["serde"] }
//! parking_lot = "0.12"
//! once_cell = "1"
//! ctrlc = "3"
//! tracing = "0.1"
//! tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use tracing::{info, warn};

// ----- your solver crates -----
use berth_alloc_model::prelude::{Problem, SolutionView};
use berth_alloc_model::problem::loader::ProblemLoader;
use berth_alloc_solver::engine::solver_engine::{SolverEngineBuilder, SolverEngineConfig};
use berth_alloc_solver::engine::{gls, ils, sa};
use berth_alloc_solver::model::solver_model::SolverModel;

// ---------- simple app result ----------
type AppError = Box<dyn Error + Send + Sync + 'static>;
type AppResult<T> = Result<T, AppError>;

// ---------- filesystem helpers ----------
fn ensure_dir(p: &Path) -> AppResult<()> {
    if !p.exists() {
        fs::create_dir_all(p)?;
    }
    Ok(())
}

// ---------- instance loader (first instance in ./instances) ----------
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

fn load_first_instance() -> AppResult<(Problem<i64>, String)> {
    let inst_dir =
        find_instances_dir().ok_or_else(|| "Could not find instances/ directory".to_string())?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&inst_dir)?
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
        .next()
        .ok_or_else(|| format!("No .txt instance files in {}", inst_dir.display()))?;
    let loader = ProblemLoader::default();
    let problem = loader.from_path(&f)?;
    let name = f
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| f.to_string_lossy().into_owned());
    Ok((problem, name))
}

// ---------- per-strategy files + locks ----------
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum StratKind {
    Gls,
    Ils,
    Sa,
}
impl StratKind {
    fn as_str(&self) -> &'static str {
        match self {
            StratKind::Gls => "gls",
            StratKind::Ils => "ils",
            StratKind::Sa => "sa",
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct TrialRecord<P: Serialize + Clone> {
    strategy: String,
    params: P,
    seed: u64,
    start_ts: DateTime<Utc>,
    end_ts: DateTime<Utc>,
    runtime_ms: u128,
    cost: Option<i64>,
}

#[derive(Serialize, Deserialize, Clone)]
struct BestSnapshot<P: Serialize + Clone> {
    strategy: String,
    params: P,
    seed: u64,
    cost: i64,
    runtime_ms: u128,
    filename: String,
    at: DateTime<Utc>,
}

struct IoPaths {
    dir: PathBuf,
    trials_jsonl: PathBuf,
    best_json: PathBuf,
    top_json: PathBuf,
}
fn io_paths(kind: StratKind) -> IoPaths {
    let dir = PathBuf::from("tuning").join(kind.as_str());
    let trials_jsonl = dir.join(format!("tuning_{}_trials.jsonl", kind.as_str()));
    let best_json = dir.join(format!("tuning_{}_best.json", kind.as_str()));
    let top_json = dir.join(format!("tuning_{}_top.json", kind.as_str()));
    IoPaths {
        dir,
        trials_jsonl,
        best_json,
        top_json,
    }
}

static GLS_IO_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static ILS_IO_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static SA_IO_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

// ---------- archives (TPE-ish good pool + mutation) ----------
#[derive(Clone, Serialize, Deserialize)]
struct Scored<P> {
    params: P,
    seed: u64,
    cost: i64,
    runtime_ms: u128,
}
impl<P> PartialEq for Scored<P> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl<P> Eq for Scored<P> {}
impl<P> PartialOrd for Scored<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<P> Ord for Scored<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost).reverse()
    }
}

#[derive(Clone)]
struct Archive<P: Clone + Serialize> {
    best: Option<Scored<P>>,
    top: Vec<Scored<P>>,
    top_cap: usize,
    explore_beta: (f64, f64), // alpha, beta
    exploit_beta: (f64, f64),
}
impl<P: Clone + Serialize> Archive<P> {
    fn new(cap: usize) -> Self {
        Self {
            best: None,
            top: Vec::new(),
            top_cap: cap.max(8),
            explore_beta: (1.0, 1.0),
            exploit_beta: (1.0, 1.0),
        }
    }
    fn register(&mut self, scored: Scored<P>) -> (bool, bool) {
        let mut new_best = false;
        if let Some(b) = &self.best {
            if scored.cost < b.cost {
                self.best = Some(scored.clone());
                new_best = true;
            }
        } else {
            self.best = Some(scored.clone());
            new_best = true;
        }
        self.top.push(scored);
        self.top.sort_by_key(|s| s.cost);
        if self.top.len() > self.top_cap {
            self.top.truncate(self.top_cap);
        }
        let improved_top = new_best || self.top.len() < self.top_cap;
        (new_best, improved_top)
    }
    fn good_pool(&self) -> &[Scored<P>] {
        if self.top.is_empty() {
            return &self.top;
        }
        let q = ((self.top.len() as f64) * 0.2).ceil() as usize;
        let k = self.top.len().max(4);
        let upto = q.max(4).min(k);
        &self.top[0..upto]
    }
    fn exploit_weighted_pick<'a>(&'a self, rng: &mut ChaCha8Rng) -> Option<&'a Scored<P>> {
        let pool = self.good_pool();
        if pool.is_empty() {
            return None;
        }
        let mut total = 0.0;
        let weights: Vec<f64> = pool
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let w = 1.0 / ((i + 1) as f64);
                total += w;
                w
            })
            .collect();
        let mut r = rng.random::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return pool.get(i);
            }
            r -= *w;
        }
        pool.last()
    }
    fn choose_mode(&self, rng: &mut ChaCha8Rng) -> Mode {
        let (a_e, b_e) = self.explore_beta;
        let (a_x, b_x) = self.exploit_beta;
        let be = Beta::new(a_e.max(1e-6), b_e.max(1e-6)).unwrap();
        let bx = Beta::new(a_x.max(1e-6), b_x.max(1e-6)).unwrap();
        if bx.sample(rng) >= be.sample(rng) {
            Mode::Exploit
        } else {
            Mode::Explore
        }
    }
    fn reward(&mut self, mode: Mode, success: bool) {
        let ab = match mode {
            Mode::Explore => &mut self.explore_beta,
            Mode::Exploit => &mut self.exploit_beta,
        };
        if success {
            ab.0 += 1.0;
        } else {
            ab.1 += 1.0;
        }
    }
}
#[derive(Clone, Copy)]
enum Mode {
    Explore,
    Exploit,
}

// ---------- parameter spaces & mutations ----------
trait ParamSpace: Clone + Serialize + Send + Sync + 'static {
    fn sample_new(rng: &mut ChaCha8Rng) -> Self;
    fn mutate(&self, rng: &mut ChaCha8Rng) -> Self;
}
fn jitter_usize(rng: &mut ChaCha8Rng, base: usize, pct: f64, min: usize, max: usize) -> usize {
    let span = ((base as f64) * pct).round() as isize;
    let off = rng.random_range(-(span as i64)..=(span as i64)) as isize;
    (base as isize + off).clamp(min as isize, max as isize) as usize
}
fn jitter_f64(rng: &mut ChaCha8Rng, base: f64, pct: f64, lo: f64, hi: f64) -> f64 {
    let span = base.abs() * pct;
    let n = Normal::new(base, span.max(1e-9)).unwrap();
    n.sample(rng).clamp(lo, hi)
}
fn pick_usize(rng: &mut ChaCha8Rng, lo: usize, hi: usize) -> usize {
    rng.random_range(lo..=hi)
}
fn pick_f64(rng: &mut ChaCha8Rng, lo: f64, hi: f64) -> f64 {
    rng.random_range(lo..=hi)
}

// ---- GLS params ----
#[derive(Clone, Serialize, Deserialize)]
struct GlsParams {
    lambda: i64,
    penalty_step: i64,
    decay_num: u32,
    decay_den: u32,
    stagn_pulse: usize,
    pulse_top_k: usize,
    max_local_steps: usize,
    refetch_after_stale: usize,
    hard_refetch_every: usize,
    kicks_on_reset: usize,
    tgt_pen_low: f64,
    tgt_pen_high: f64,
    lambda_step_frac: f64,
}
impl ParamSpace for GlsParams {
    fn sample_new(rng: &mut ChaCha8Rng) -> Self {
        let dec = [
            (85, 100),
            (90, 100),
            (93, 100),
            (95, 100),
            (97, 100),
            (99, 100),
        ][pick_usize(rng, 0, 5)];
        let low = pick_f64(rng, 0.05, 0.30);
        let high = (low + pick_f64(rng, 0.1, 0.4)).clamp(low + 0.05, 0.85);
        Self {
            lambda: pick_usize(rng, 3, 40) as i64,
            penalty_step: pick_usize(rng, 1, 4) as i64,
            decay_num: dec.0,
            decay_den: dec.1,
            stagn_pulse: pick_usize(rng, 4, 24),
            pulse_top_k: pick_usize(rng, 8, 64),
            max_local_steps: pick_usize(rng, 800, 4000),
            refetch_after_stale: pick_usize(rng, 0, 80),
            hard_refetch_every: pick_usize(rng, 8, 64),
            kicks_on_reset: pick_usize(rng, 0, 24),
            tgt_pen_low: low,
            tgt_pen_high: high,
            lambda_step_frac: pick_f64(rng, 0.02, 0.15),
        }
    }
    fn mutate(&self, rng: &mut ChaCha8Rng) -> Self {
        let mut p = self.clone();
        p.lambda = jitter_usize(rng, p.lambda as usize, 0.35, 1, 2_000_000) as i64;
        p.penalty_step = jitter_usize(rng, p.penalty_step as usize, 0.40, 1, 16) as i64;
        p.decay_num =
            jitter_usize(rng, p.decay_num as usize, 0.05, 80, p.decay_den as usize) as u32;
        p.stagn_pulse = jitter_usize(rng, p.stagn_pulse, 0.50, 3, 64);
        p.pulse_top_k = jitter_usize(rng, p.pulse_top_k, 0.40, 6, 96);
        p.max_local_steps = jitter_usize(rng, p.max_local_steps, 0.40, 400, 6000);
        p.refetch_after_stale = jitter_usize(rng, p.refetch_after_stale, 0.60, 0, 120);
        p.hard_refetch_every = jitter_usize(rng, p.hard_refetch_every, 0.60, 4, 120);
        p.kicks_on_reset = jitter_usize(rng, p.kicks_on_reset, 0.60, 0, 32);
        p.tgt_pen_low = jitter_f64(rng, p.tgt_pen_low, 0.40, 0.0, 0.9);
        p.tgt_pen_high = (jitter_f64(rng, p.tgt_pen_high, 0.40, 0.0, 0.98))
            .max((p.tgt_pen_low + 0.02).min(0.98));
        p.lambda_step_frac = jitter_f64(rng, p.lambda_step_frac, 0.40, 0.01, 0.25);
        p
    }
}
fn build_gls<'p>(
    model: &SolverModel<'p, i64>,
    params: &GlsParams,
) -> Box<dyn berth_alloc_solver::engine::search::SearchStrategy<i64, ChaCha8Rng>> {
    let mut s = gls::gls_strategy::<i64, ChaCha8Rng>(model);
    s = s
        .with_lambda(params.lambda)
        .with_penalty_step(params.penalty_step)
        .with_decay(gls::DecayMode::Multiplicative {
            num: params.decay_num,
            den: params.decay_den,
        })
        .with_pulse_params(params.stagn_pulse, params.pulse_top_k)
        .with_max_local_steps(params.max_local_steps)
        .with_refetch_after_stale(params.refetch_after_stale)
        .with_hard_refetch_every(params.hard_refetch_every)
        .with_hard_refetch_mode(gls::HardRefetchMode::IfBetter)
        .with_restart_on_publish(true)
        .with_reset_on_refetch(true)
        .with_kick_steps_on_reset(params.kicks_on_reset)
        .with_adaptive_lambda(true)
        .with_target_penalty_share(params.tgt_pen_low, params.tgt_pen_high)
        .with_lambda_step_frac(params.lambda_step_frac)
        .with_lambda_bounds(1, 2_000_000);
    Box::new(s)
}

// ---- ILS params ----
#[derive(Clone, Serialize, Deserialize)]
struct IlsParams {
    local_lo: usize,
    local_hi: usize,
    sideways: bool,
    worsen_prob: f64,
    destroy_attempts: usize,
    repair_attempts: usize,
    refetch_after_stale: usize,
    hard_refetch_every: usize,
    kicks_after_refetch: usize,
    ewma_beta: f64,
    sr_low: f64,
    sr_high: f64,
    upd_period: usize,
    step_max: usize,
    bias_explore: bool,
}
impl ParamSpace for IlsParams {
    fn sample_new(rng: &mut ChaCha8Rng) -> Self {
        let lo = pick_usize(rng, 600, 2200);
        let hi = pick_usize(rng, lo + 100, (lo + 2400).min(3600));
        let sr_low = pick_f64(rng, 0.12, 0.35);
        let sr_high = (sr_low + pick_f64(rng, 0.18, 0.45)).clamp(sr_low + 0.05, 0.85);
        Self {
            local_lo: lo,
            local_hi: hi,
            sideways: rng.random_bool(0.7),
            worsen_prob: pick_f64(rng, 0.0, 0.05),
            destroy_attempts: pick_usize(rng, 6, 24),
            repair_attempts: pick_usize(rng, 12, 48),
            refetch_after_stale: pick_usize(rng, 24, 80),
            hard_refetch_every: pick_usize(rng, 8, 40),
            kicks_after_refetch: pick_usize(rng, 2, 16),
            ewma_beta: pick_f64(rng, 0.20, 0.60),
            sr_low,
            sr_high,
            upd_period: pick_usize(rng, 3, 8),
            step_max: pick_usize(rng, 1, 4),
            bias_explore: rng.random_bool(0.6),
        }
    }
    fn mutate(&self, rng: &mut ChaCha8Rng) -> Self {
        let mut p = self.clone();
        p.local_lo = jitter_usize(rng, p.local_lo, 0.40, 200, 4000);
        p.local_hi = jitter_usize(rng, p.local_hi, 0.40, p.local_lo + 50, 5000);
        p.sideways = if rng.random_bool(0.15) {
            !p.sideways
        } else {
            p.sideways
        };
        p.worsen_prob = jitter_f64(rng, p.worsen_prob, 0.8, 0.0, 0.15);
        p.destroy_attempts = jitter_usize(rng, p.destroy_attempts, 0.40, 3, 64);
        p.repair_attempts = jitter_usize(rng, p.repair_attempts, 0.40, 6, 80);
        p.refetch_after_stale = jitter_usize(rng, p.refetch_after_stale, 0.40, 8, 120);
        p.hard_refetch_every = jitter_usize(rng, p.hard_refetch_every, 0.60, 0, 64);
        p.kicks_after_refetch = jitter_usize(rng, p.kicks_after_refetch, 0.60, 0, 24);
        p.ewma_beta = jitter_f64(rng, p.ewma_beta, 0.40, 0.05, 0.95);
        p.sr_low = jitter_f64(rng, p.sr_low, 0.40, 0.0, 0.9);
        p.sr_high = (jitter_f64(rng, p.sr_high, 0.40, 0.0, 0.98)).max((p.sr_low + 0.05).min(0.98));
        p.upd_period = jitter_usize(rng, p.upd_period, 0.60, 1, 12);
        p.step_max = jitter_usize(rng, p.step_max, 0.60, 1, 6);
        p.bias_explore = if rng.random_bool(0.10) {
            !p.bias_explore
        } else {
            p.bias_explore
        };
        p
    }
}
fn build_ils<'p>(
    model: &SolverModel<'p, i64>,
    params: &IlsParams,
) -> Box<dyn berth_alloc_solver::engine::search::SearchStrategy<i64, ChaCha8Rng>> {
    let mut s = ils::ils_strategy::<i64, ChaCha8Rng>(model);
    s = s
        .with_local_steps_range(params.local_lo..=params.local_hi)
        .with_local_sideways(params.sideways)
        .with_local_worsening_prob(params.worsen_prob)
        .with_destroy_attempts(params.destroy_attempts)
        .with_repair_attempts(params.repair_attempts)
        .with_shuffle_local_each_step(true)
        .with_refetch_after_stale(params.refetch_after_stale)
        .with_hard_refetch_every(params.hard_refetch_every)
        .with_hard_refetch_mode(ils::HardRefetchMode::IfBetter)
        .with_kick_ops_after_refetch(params.kicks_after_refetch)
        .with_online_perturbation(true)
        .with_destroy_cap_bounds(4, 48)
        .with_repair_cap_bounds(8, 64)
        .with_online_ewma_beta(params.ewma_beta)
        .with_online_success_band(params.sr_low, params.sr_high)
        .with_online_cap_update_period(params.upd_period)
        .with_online_cap_step_max(params.step_max)
        .with_online_bias_explore_on_stagnation(params.bias_explore);
    Box::new(s)
}

// ---- SA params ----
#[derive(Clone, Serialize, Deserialize)]
struct SaParams {
    t0: f64,
    cooling: f64,
    steps_per_epoch: usize,
    refetch_every: usize,
    stale_epochs: usize,
    reheat: f64,
    kicks: usize,
    low: f64,
    high: f64,
    nudge_up: f64,
    extra_cool: f64,
    blend_geo: f64,
    blend_tgt: f64,
    beta: f64,
}
impl ParamSpace for SaParams {
    fn sample_new(rng: &mut ChaCha8Rng) -> Self {
        let low = pick_f64(rng, 0.12, 0.35);
        let high = (low + pick_f64(rng, 0.25, 0.5)).clamp(low + 0.2, 0.9);
        let blend_geo = pick_f64(rng, 0.4, 0.9);
        let blend_tgt = (1.0 - blend_geo).max(0.05);
        Self {
            t0: pick_f64(rng, 12.0, 90.0),
            cooling: pick_f64(rng, 0.9982, 0.99995),
            steps_per_epoch: pick_usize(rng, 600, 2000),
            refetch_every: pick_usize(rng, 60, 200),
            stale_epochs: pick_usize(rng, 30, 100),
            reheat: pick_f64(rng, 0.6, 0.95),
            kicks: pick_usize(rng, 4, 32),
            low,
            high,
            nudge_up: pick_f64(rng, 1.02, 1.12),
            extra_cool: pick_f64(rng, 0.990, 1.0),
            blend_geo,
            blend_tgt,
            beta: pick_f64(rng, 0.80, 0.96),
        }
    }
    fn mutate(&self, rng: &mut ChaCha8Rng) -> Self {
        let mut p = self.clone();
        p.t0 = jitter_f64(rng, p.t0, 0.40, 6.0, 120.0);
        p.cooling = jitter_f64(rng, p.cooling, 0.0006, 0.9975, 0.99999);
        p.steps_per_epoch = jitter_usize(rng, p.steps_per_epoch, 0.40, 300, 3000);
        p.refetch_every = jitter_usize(rng, p.refetch_every, 0.40, 40, 300);
        p.stale_epochs = jitter_usize(rng, p.stale_epochs, 0.40, 20, 160);
        p.reheat = jitter_f64(rng, p.reheat, 0.20, 0.5, 0.99);
        p.kicks = jitter_usize(rng, p.kicks, 0.40, 0, 48);
        p.low = jitter_f64(rng, p.low, 0.40, 0.05, 0.5);
        p.high = jitter_f64(rng, p.high, 0.40, p.low + 0.1, 0.95);
        p.nudge_up = jitter_f64(rng, p.nudge_up, 0.20, 1.01, 1.20);
        p.extra_cool = jitter_f64(rng, p.extra_cool, 0.05, 0.96, 1.0);
        let g = jitter_f64(rng, p.blend_geo, 0.30, 0.15, 0.95);
        p.blend_geo = g;
        p.blend_tgt = (1.0 - g).max(0.02);
        p.beta = jitter_f64(rng, p.beta, 0.10, 0.6, 0.98);
        p
    }
}
fn build_sa<'p>(
    model: &SolverModel<'p, i64>,
    params: &SaParams,
) -> Box<dyn berth_alloc_solver::engine::search::SearchStrategy<i64, ChaCha8Rng>> {
    let mut s = sa::sa_strategy::<i64, ChaCha8Rng>(model);
    s = s
        .with_init_temp(params.t0)
        .with_cooling(params.cooling)
        .with_min_temp(1e-4)
        .with_steps_per_epoch(params.steps_per_epoch)
        .with_hard_refetch_every(params.refetch_every)
        .with_hard_refetch_mode(sa::HardRefetchMode::IfBetter)
        .with_refetch_after_stale(params.stale_epochs)
        .with_reheat_factor(params.reheat)
        .with_kick_ops_after_refetch(params.kicks)
        .with_big_m_for_energy(900_000_000)
        .with_acceptance_targets(params.low, params.high)
        .with_ar_nudge_up(params.nudge_up)
        .with_ar_extra_cooling(params.extra_cool)
        .with_online_temp_blend(params.blend_geo, params.blend_tgt)
        .with_auto_temperatures(true)
        .with_ewma_beta(params.beta);
    Box::new(s)
}

// ---------- thread-safe archives & IO per strategy ----------
struct SharedSpace<P: ParamSpace> {
    archive: Mutex<Archive<P>>,
    paths: IoPaths,
    lock: &'static Mutex<()>, // for file IO
}
impl<P: ParamSpace> SharedSpace<P> {
    fn new(kind: StratKind, top_cap: usize) -> AppResult<Self> {
        let paths = io_paths(kind);
        ensure_dir(&paths.dir)?;
        Ok(Self {
            archive: Mutex::new(Archive::new(top_cap)),
            paths,
            lock: match kind {
                StratKind::Gls => &GLS_IO_LOCK,
                StratKind::Ils => &ILS_IO_LOCK,
                StratKind::Sa => &SA_IO_LOCK,
            },
        })
    }
    fn append_trial(&self, rec: &TrialRecord<P>) -> AppResult<()> {
        let _g = self.lock.lock();
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.trials_jsonl)?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer(&mut w, rec)?;
        w.write_all(b"\n")?;
        w.flush()?;
        Ok(())
    }
    fn write_best(&self, best: &BestSnapshot<P>) -> AppResult<()> {
        let _g = self.lock.lock();
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.paths.best_json)?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer_pretty(&mut w, best)?;
        w.flush()?;
        Ok(())
    }
    fn write_top(&self) -> AppResult<()> {
        let tops: Vec<Scored<P>> = {
            let arc = self.archive.lock();
            arc.top.clone()
        };
        let _g = self.lock.lock();
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.paths.top_json)?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer_pretty(&mut w, &tops)?;
        w.flush()?;
        Ok(())
    }
}

// ---------- core worker ----------
fn worker_loop<P, Bldr>(
    kind: StratKind,
    shared: Arc<SharedSpace<P>>,
    problem: Arc<Problem<i64>>,
    file_name: String,
    mut build_strategy: Bldr,
) where
    P: ParamSpace,
    Bldr: FnMut(
            &SolverModel<'_, i64>,
            &P,
        )
            -> Box<dyn berth_alloc_solver::engine::search::SearchStrategy<i64, ChaCha8Rng>>
        + Send
        + 'static,
{
    let tid: u64 = rand::random::<u64>();
    let mut rng = ChaCha8Rng::seed_from_u64(tid ^ 0xBADC0FFE_u64);
    loop {
        let params = {
            let arc = shared.archive.lock();
            let mode = arc.choose_mode(&mut rng);
            let candidate = match mode {
                Mode::Explore => P::sample_new(&mut rng),
                Mode::Exploit => {
                    if let Some(base) = arc.exploit_weighted_pick(&mut rng) {
                        base.params.clone().mutate(&mut rng)
                    } else {
                        P::sample_new(&mut rng)
                    }
                }
            };
            drop(arc);
            candidate
        };

        let model = match SolverModel::from_problem(&problem) {
            Ok(m) => m,
            Err(e) => {
                warn!("Model build failed: {e}");
                continue;
            }
        };

        let strategy = (build_strategy)(&model, &params);

        // One-strategy engine, one worker inside → lots of outer threads saturate CPU
        let mut engine = SolverEngineBuilder::<i64>::default()
            .with_config(SolverEngineConfig {
                num_workers: 1,
                time_limit: Duration::from_secs(20),
            })
            .with_strategy(strategy)
            .build();

        let start_ts = Utc::now();
        let t0 = Instant::now();
        let outcome = engine.solve(&problem);
        let runtime = t0.elapsed();
        let end_ts = Utc::now();

        let cost_opt: Option<i64> = match outcome {
            Ok(Some(sol)) => Some(sol.cost()),
            Ok(None) => None,
            Err(_) => None,
        };

        let rec = TrialRecord {
            strategy: kind.as_str().to_string(),
            params: params.clone(),
            seed: tid,
            start_ts,
            end_ts,
            runtime_ms: runtime.as_millis(),
            cost: cost_opt,
        };

        let mut new_best = false;
        if let Some(cost) = cost_opt {
            let scored = Scored {
                params: params.clone(),
                seed: tid,
                cost,
                runtime_ms: runtime.as_millis(),
            };
            let (is_best, _improved_top) = {
                let mut arc = shared.archive.lock();
                let (b, t) = arc.register(scored.clone());
                // reward by new global best (simple streaming signal)
                let mode_now = arc.choose_mode(&mut rng);
                arc.reward(mode_now, b);
                (b, t)
            };
            new_best = is_best;
            if is_best {
                let snap = BestSnapshot {
                    strategy: kind.as_str().to_string(),
                    params: params.clone(),
                    seed: tid,
                    cost,
                    runtime_ms: runtime.as_millis(),
                    filename: file_name.clone(),
                    at: end_ts,
                };
                let _ = shared.write_best(&snap);
                let _ = shared.write_top();
            }
        }

        let _ = shared.append_trial(&rec);
        if new_best {
            info!(
                "[{}] NEW BEST: {:?}",
                kind.as_str(),
                rec.cost.unwrap_or(i64::MAX)
            );
        }
    }
}

// ---------- main ----------
fn main() -> AppResult<()> {
    // tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .with_target(false)
        .compact()
        .init();

    ensure_dir(Path::new("tuning"))?;

    // load one instance (as requested)
    let (problem, filename) = load_first_instance()?;
    let problem = Arc::new(problem);

    // archives per strategy
    let gls_space: Arc<SharedSpace<GlsParams>> = Arc::new(SharedSpace::new(StratKind::Gls, 64)?);
    let ils_space: Arc<SharedSpace<IlsParams>> = Arc::new(SharedSpace::new(StratKind::Ils, 64)?);
    let sa_space: Arc<SharedSpace<SaParams>> = Arc::new(SharedSpace::new(StratKind::Sa, 64)?);

    // ctrl-c: exit (files are flushed on every write)
    ctrlc::set_handler(move || {
        eprintln!("\nCtrl-C: stopping tuners (logs already flushed).");
        std::process::exit(0);
    })?;

    // decide thread count and split across strategies to max out CPU
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(3);
    let base = total / 3;
    let mut rem = total % 3;
    let mut counts = [base, base, base];
    for item in &mut counts {
        if rem > 0 {
            *item += 1;
            rem -= 1;
        }
    }
    info!(
        "Launching {} tuner threads (GLS={}, ILS={}, SA={}) — 20s/trial, running forever.",
        total, counts[0], counts[1], counts[2]
    );

    // thread spawns
    let mut handles = Vec::new();

    for _ in 0..counts[0] {
        let space = gls_space.clone();
        let problem = problem.clone();
        let fname = filename.clone();
        handles.push(thread::spawn(move || {
            worker_loop(StratKind::Gls, space, problem, fname, |model, params| {
                build_gls(model, params)
            });
        }));
    }
    for _ in 0..counts[1] {
        let space = ils_space.clone();
        let problem = problem.clone();
        let fname = filename.clone();
        handles.push(thread::spawn(move || {
            worker_loop(StratKind::Ils, space, problem, fname, |model, params| {
                build_ils(model, params)
            });
        }));
    }
    for _ in 0..counts[2] {
        let space = sa_space.clone();
        let problem = problem.clone();
        let fname = filename.clone();
        handles.push(thread::spawn(move || {
            worker_loop(StratKind::Sa, space, problem, fname, |model, params| {
                build_sa(model, params)
            });
        }));
    }

    // park forever
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}
