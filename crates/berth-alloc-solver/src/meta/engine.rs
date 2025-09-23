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

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    framework::{
        err::SolverStatePlanApplyError,
        planning::{Plan, PlanningContext},
        solver::{ConstructionSolver, Solver},
        state::{SolverState, SolverStateView},
    },
    meta::{config::MetaConfig, operator::Operator},
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    prelude::{Problem, SolutionRef},
    solution::SolutionError,
};
use num_traits::{CheckedAdd, CheckedSub, ToPrimitive, Zero};
use rand::{
    SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::{debug, info, instrument, trace, warn};

#[inline]
fn acceptance_prob(delta: Cost, temp: f64) -> f64 {
    if delta < Cost::zero() {
        1.0
    } else if delta > Cost::zero() {
        let f = delta.to_f64().unwrap_or(f64::INFINITY);
        (-f / temp.max(1e-12)).exp()
    } else {
        0.0
    }
}

#[inline]
fn ewma(prev: f64, x: f64, alpha: f64) -> f64 {
    if prev == 0.0 {
        x
    } else {
        alpha * x + (1.0 - alpha) * prev
    }
}

#[derive(Debug)]
struct Candidate<'p, T: Ord + Copy> {
    op_idx: usize,
    plan: Plan<'p, T>,
    delta: Cost,
    unassigned: usize,
    feasible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperatorStats {
    attempts: u64,
    accepted: u64,
    ewma_reward: f64,
    total_improvement: Cost,
    emwa_gen_ns_per_proposal: f64,
    emwa_eval_ns_per_proposal: f64,
}

impl Default for OperatorStats {
    fn default() -> Self {
        Self {
            attempts: 0,
            accepted: 0,
            ewma_reward: 0.0,
            total_improvement: Cost::zero(),
            emwa_gen_ns_per_proposal: 0.0,
            emwa_eval_ns_per_proposal: 0.0,
        }
    }
}

impl OperatorStats {
    #[inline]
    pub fn on_attempt(&mut self) {
        self.attempts += 1;
    }

    #[inline]
    pub fn on_accept(&mut self, improvement: Cost, reward_alpha: f64) {
        self.accepted += 1;
        self.total_improvement += improvement;
        let r = improvement.to_f64().unwrap_or(0.0);
        self.ewma_reward = ewma(self.ewma_reward, r, reward_alpha);
    }

    #[inline]
    pub fn on_timing(&mut self, gen_ns: f64, eval_ns: f64, gen_alpha: f64, eval_alpha: f64) {
        if gen_ns > 0.0 {
            self.emwa_gen_ns_per_proposal = ewma(self.emwa_gen_ns_per_proposal, gen_ns, gen_alpha);
        }
        if eval_ns > 0.0 {
            self.emwa_eval_ns_per_proposal =
                ewma(self.emwa_eval_ns_per_proposal, eval_ns, eval_alpha);
        }
    }
}

#[derive(Debug)]
pub struct OperatorRecord<T: Copy + Ord> {
    operator: Box<dyn Operator<Time = T>>,
    stats: OperatorStats,
}

impl<T: Copy + Ord> OperatorRecord<T> {
    pub fn new(operator: Box<dyn Operator<Time = T>>) -> Self {
        Self {
            operator,
            stats: OperatorStats::default(),
        }
    }

    #[inline]
    pub fn operator(&self) -> &dyn Operator<Time = T> {
        self.operator.as_ref()
    }

    #[inline]
    pub fn stats(&self) -> &OperatorStats {
        &self.stats
    }

    #[inline]
    pub fn stats_mut(&mut self) -> &mut OperatorStats {
        &mut self.stats
    }
}

#[derive(Debug)]
pub struct OperatorPool<T: Copy + Ord> {
    records: Vec<OperatorRecord<T>>,
}

impl<T: Copy + Ord> OperatorPool<T> {
    fn new(records: Vec<OperatorRecord<T>>) -> Self {
        Self { records }
    }

    #[inline]
    fn len(&self) -> usize {
        self.records.len()
    }

    #[inline]
    fn get(&self, i: usize) -> &OperatorRecord<T> {
        &self.records[i]
    }

    #[inline]
    fn get_mut(&mut self, i: usize) -> &mut OperatorRecord<T> {
        &mut self.records[i]
    }

    #[inline]
    pub fn records(&self) -> &[OperatorRecord<T>] {
        &self.records
    }

    #[inline]
    fn raw_score_at(
        &self,
        i: usize,
        alloc: &crate::meta::config::AllocationConfig,
        stats: &crate::meta::config::StatsConfig,
    ) -> f64 {
        let s = &self.records[i].stats;
        let ns_per = (s.emwa_gen_ns_per_proposal + s.emwa_eval_ns_per_proposal)
            .max(stats.min_ns_per_proposal);
        let speed = 1.0 / ns_per;
        let succ = if s.attempts > 0 {
            s.accepted as f64 / s.attempts as f64
        } else {
            stats.bootstrap_success_rate
        };
        alloc.speed_weight * speed + alloc.success_weight * succ
    }

    fn apply_aggregates(&mut self, aggs: &[OpAgg], stats_cfg: &crate::meta::config::StatsConfig) {
        for (i, a) in aggs.iter().enumerate() {
            if a.attempts == 0 {
                continue;
            }
            let st = &mut self.records[i].stats;
            st.attempts += a.attempts;
            if a.gen_ns_count > 0 || a.eval_ns_count > 0 {
                let gene = if a.gen_ns_count > 0 {
                    a.gen_ns_sum / a.gen_ns_count as f64
                } else {
                    0.0
                };
                let eval = if a.eval_ns_count > 0 {
                    a.eval_ns_sum / a.eval_ns_count as f64
                } else {
                    0.0
                };
                st.on_timing(
                    gene,
                    eval,
                    stats_cfg.gen_time_alpha,
                    stats_cfg.eval_time_alpha,
                );
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct OpAgg {
    attempts: u64,
    gen_ns_sum: f64,
    eval_ns_sum: f64,
    gen_ns_count: u64,
    eval_ns_count: u64,
}

impl OpAgg {
    #[inline]
    fn add_attempt(&mut self) {
        self.attempts += 1;
    }
    #[inline]
    fn add_timing(&mut self, gen_ns: f64, eval_ns: f64) {
        if gen_ns > 0.0 {
            self.gen_ns_sum += gen_ns;
            self.gen_ns_count += 1;
        }
        if eval_ns > 0.0 {
            self.eval_ns_sum += eval_ns;
            self.eval_ns_count += 1;
        }
    }
}

#[derive(Debug)]
struct ThreadAccum<'p, T: Ord + Copy> {
    candidate: Option<Candidate<'p, T>>,
    per_op: Vec<OpAgg>,
}

impl<'p, T: Ord + Copy> ThreadAccum<'p, T> {
    #[inline]
    fn empty(n_ops: usize) -> Self {
        Self {
            candidate: None,
            per_op: vec![OpAgg::default(); n_ops],
        }
    }

    #[inline]
    fn merge(mut self, mut other: Self, temp: f64) -> Self {
        for (i, o) in other.per_op.iter_mut().enumerate() {
            let s = &mut self.per_op[i];
            s.attempts += o.attempts;
            s.gen_ns_sum += o.gen_ns_sum;
            s.eval_ns_sum += o.eval_ns_sum;
            s.gen_ns_count += o.gen_ns_count;
            s.eval_ns_count += o.eval_ns_count;
        }
        self.candidate = choose_sa(self.candidate, other.candidate, temp);
        self
    }
}

#[inline]
fn choose_sa<'p, T: Copy + Ord>(
    a: Option<Candidate<'p, T>>,
    b: Option<Candidate<'p, T>>,
    temp: f64,
) -> Option<Candidate<'p, T>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => {
            // 1) Prefer feasible over infeasible
            if x.feasible != y.feasible {
                return Some(if x.feasible { x } else { y });
            }
            // 2) Prefer fewer unassigned
            if x.unassigned != y.unassigned {
                return Some(if x.unassigned < y.unassigned { x } else { y });
            }
            // 3) Tie → SA on cost delta
            let d = y.delta - x.delta;
            let p = acceptance_prob(d, temp);
            if p > 0.0 && rand::random::<f64>() < p {
                Some(y)
            } else {
                Some(x)
            }
        }
    }
}

#[inline]
fn make_job_rng(base_seed: u64, j: usize) -> ChaCha8Rng {
    let s = base_seed ^ ((j as u64).rotate_left(17)) ^ 0x9E37_79B1_85EB_CA87u64;
    ChaCha8Rng::seed_from_u64(s)
}

#[derive(Debug)]
pub enum MetaEngineError<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    ConstructionError(S::Error),
    PlanApply(SolverStatePlanApplyError<T>),
    Solution(SolutionError),
}

impl<T, S> From<SolverStatePlanApplyError<T>> for MetaEngineError<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    fn from(e: SolverStatePlanApplyError<T>) -> Self {
        MetaEngineError::PlanApply(e)
    }
}

impl<T, S> From<SolutionError> for MetaEngineError<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    fn from(e: SolutionError) -> Self {
        MetaEngineError::Solution(e)
    }
}

impl<T, S> std::fmt::Display for MetaEngineError<T, S>
where
    T: Copy + Ord + std::fmt::Display,
    S: ConstructionSolver<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaEngineError::ConstructionError(_) => write!(f, "construction error"),
            MetaEngineError::PlanApply(e) => write!(f, "plan apply error: {e}"),
            MetaEngineError::Solution(e) => write!(f, "solution error: {e}"),
        }
    }
}

pub struct MetaEngine<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    config: MetaConfig,
    operator_pool: OperatorPool<T>,
    construction_solver: S,
    proposals_made: u64,
    weights_buf: Vec<f64>,
}

impl<T, S> MetaEngine<T, S>
where
    T: Copy + Ord + Send + Sync,
    S: ConstructionSolver<T>,
{
    pub fn new(
        config: MetaConfig,
        ops: impl IntoIterator<Item = Box<dyn Operator<Time = T>>>,
        construction_solver: S,
    ) -> Self {
        let records = ops.into_iter().map(|op| OperatorRecord::new(op)).collect();
        Self {
            config,
            operator_pool: OperatorPool::new(records),
            construction_solver,
            proposals_made: 0,
            weights_buf: Vec::new(),
        }
    }

    #[inline]
    pub fn construction_solver(&self) -> &S {
        &self.construction_solver
    }

    #[instrument(skip_all, fields(iteration, temp, tau), err(Display))]
    pub fn step<'state, 'p>(
        &mut self,
        state: &'state mut SolverState<'p, T>,
        iteration: usize,
    ) -> Result<Option<Cost>, SolverStatePlanApplyError<T>>
    where
        T: CheckedAdd + CheckedSub + std::fmt::Display,
    {
        let stats_cfg = &self.config.stats;
        let ledger = state.ledger();
        let terminal = state.terminal_occupancy();

        let anneal = &self.config.anneal;
        let alloc_cfg = &self.config.alloc;
        let rng_cfg = &self.config.random;

        let temp = (anneal.initial_temperature * anneal.cooling_rate.powi(iteration as i32))
            .max(anneal.min_temperature);
        let norm = (temp / anneal.initial_temperature).clamp(0.0, 1.0);
        let tau = alloc_cfg.softmax_tau_min
            + (alloc_cfg.softmax_tau_max - alloc_cfg.softmax_tau_min) * norm;

        tracing::Span::current().record("iteration", iteration);
        tracing::Span::current().record("temp", temp);
        tracing::Span::current().record("tau", tau);

        let n_ops = self.operator_pool.len();
        if n_ops == 0 {
            trace!("No operators available.");
            return Ok(None);
        }

        if self.weights_buf.len() != n_ops {
            self.weights_buf.resize(n_ops, 0.0);
        }

        let mut maxv = f64::NEG_INFINITY;
        for i in 0..n_ops {
            let s = self.operator_pool.raw_score_at(i, alloc_cfg, stats_cfg);
            if s > maxv {
                maxv = s;
            }
            self.weights_buf[i] = s;
        }

        let t = tau.max(1e-6);
        for w in &mut self.weights_buf {
            *w = ((*w - maxv) / t).exp();
        }

        if alloc_cfg.explore_frac > 0.0 {
            // average of current (softmaxed) weights
            let mut sum = 0.0;
            for &w in &self.weights_buf {
                sum += w;
            }
            // guard against pathological zero sum
            let avg = if sum > 0.0 {
                sum / n_ops as f64
            } else {
                1.0 / n_ops as f64
            };
            let e = alloc_cfg.explore_frac;
            for w in &mut self.weights_buf {
                *w = (1.0 - e) * *w + e * avg;
            }
        }

        let dist = Arc::new(
            WeightedIndex::new(self.weights_buf.iter().cloned())
                .expect("weights must be non-negative and finite"),
        );

        let total_draws = usize::max(n_ops, self.config.alloc.target_total_proposals_per_round);

        let base_seed = rng_cfg.seed_base_task ^ (iteration as u64);

        let reduced = (0..total_draws)
            .into_par_iter()
            .fold(
                || ThreadAccum::empty(n_ops), // one per worker
                |mut acc, j| {
                    let mut rng = make_job_rng(base_seed, j);
                    let op_idx = dist.sample(&mut rng);

                    let ctx = PlanningContext::new(ledger, terminal);
                    let t0 = Instant::now();
                    let plan = self
                        .operator_pool
                        .get(op_idx)
                        .operator()
                        .propose(iteration, ctx, &mut rng);
                    let gen_ns = t0.elapsed().as_nanos() as f64;

                    // Count attempt *always*, even if no-op
                    acc.per_op[op_idx].add_attempt();

                    if plan.is_none() {
                        return acc;
                    }
                    let plan = plan.unwrap();

                    let t1 = Instant::now();
                    let delta = plan.delta_cost();
                    let eval_ns = t1.elapsed().as_nanos() as f64;

                    // Fold timing into aggregates
                    acc.per_op[op_idx].add_timing(gen_ns, eval_ns);

                    // Maintain a per-worker candidate using SA reduction semantics
                    let unassigned = plan.ledger().iter_unassigned_requests().count();
                    let feasible = unassigned == 0;

                    let cand = Candidate {
                        op_idx,
                        plan,
                        delta,
                        unassigned,
                        feasible,
                    };
                    acc.candidate = choose_sa(acc.candidate, Some(cand), temp);
                    acc
                },
            )
            .reduce(|| ThreadAccum::empty(n_ops), |a, b| a.merge(b, temp));

        // If nothing useful was produced
        if reduced.per_op.iter().all(|agg| agg.attempts == 0) {
            trace!("All generated plans were no-ops; nothing to apply.");
            return Ok(None);
        }

        // Update per-operator attempts & timings (EWMA) once
        self.operator_pool
            .apply_aggregates(&reduced.per_op, stats_cfg);

        // Track proposals count accurately
        let mut attempts_sum: u64 = 0;
        for a in &reduced.per_op {
            attempts_sum += a.attempts;
        }
        self.proposals_made = self.proposals_made.saturating_add(attempts_sum);

        // Apply winner if present
        let Some(winner) = reduced.candidate else {
            trace!("No candidate survived the reduction.");
            return Ok(None);
        };

        let winner_op_name = self.operator_pool.get(winner.op_idx).operator().name();

        info!(
            op_index = winner.op_idx,
            op = winner_op_name,
            %temp,
            %tau,
            delta = %winner.delta,
            "Selected winner"
        );

        match state.apply_plan(winner.plan) {
            Ok(()) => {
                // Reward the winning operator
                let rec = self.operator_pool.get_mut(winner.op_idx);
                rec.stats_mut()
                    .on_accept(winner.delta, stats_cfg.reward_alpha);
                trace!("Applied plan successfully.");
                Ok(Some(winner.delta))
            }
            Err(e) => {
                warn!(error = %e, op = winner_op_name, "Plan application failed; skipping.");
                Err(e)
            }
        }
    }
}

impl<T, S> Solver<T> for MetaEngine<T, S>
where
    T: Copy + Ord + Send + Sync + std::fmt::Display + CheckedAdd + CheckedSub,
    S: ConstructionSolver<T>,
    S::Error: std::fmt::Debug,
{
    type Error = MetaEngineError<T, S>;

    #[instrument(skip_all, fields(max_ms = self.config.max_solver_time_ms), err(Display))]
    fn solve<'p>(
        &mut self,
        problem: &'p Problem<T>,
    ) -> Result<Option<SolutionRef<'p, T>>, Self::Error> {
        // 1) Build initial feasible state
        let mut state = self
            .construction_solver
            .construct(problem)
            .map_err(MetaEngineError::ConstructionError)?;

        // 2) Budget
        let budget = Duration::from_millis(self.config.max_solver_time_ms);
        let t0 = Instant::now();

        // 3) Best-so-far tracking
        let mut cum_delta = Cost::zero();
        let mut best_feasible_state: Option<_> = state.is_feasible().then(|| state.clone());
        let mut best_feasible_cum: Option<Cost> = state.is_feasible().then(|| cum_delta);

        let mut iter: usize = 0;
        while t0.elapsed() < budget {
            match self.step(&mut state, iter) {
                Ok(Some(delta)) => {
                    cum_delta += delta;
                    if state.is_feasible()
                        && best_feasible_cum.map_or(true, |best| cum_delta < best)
                    {
                        best_feasible_cum = Some(cum_delta);
                        best_feasible_state = Some(state.clone());
                        debug!(best = %cum_delta, "New best feasible cumulative delta");
                    }
                }
                Ok(None) => {
                    if state.is_feasible()
                        && best_feasible_cum.map_or(true, |best| cum_delta < best)
                    {
                        best_feasible_cum = Some(cum_delta);
                        best_feasible_state = Some(state.clone());
                        debug!(best = %cum_delta, "New best feasible (no-op step)");
                    }
                }
                Err(e) => warn!(error = %e, "Step failed; continuing."),
            }
            iter += 1;
            if (iter & 0xF) == 0 && t0.elapsed() >= budget {
                break;
            }
        }

        // logging unchanged…
        let final_temp = {
            let a = &self.config.anneal;
            (a.initial_temperature * a.cooling_rate.powi(iter as i32)).max(a.min_temperature)
        };

        info!(
            iterations = iter,
            proposals = self.proposals_made,
            temperature = final_temp,
            "Meta solve finished",
        );

        // prefer feasible incumbent
        let final_state = best_feasible_state.unwrap_or(state);
        if final_state.is_feasible() {
            Ok(Some(final_state.try_into()?))
        } else {
            Ok(None)
        }
    }
}
