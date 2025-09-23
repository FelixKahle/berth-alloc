use std::ops::Mul;
use std::sync::Arc;
use std::time::Duration;

use berth_alloc_core::prelude::Cost;
use berth_alloc_model::prelude::{Problem, SolutionRef};
use berth_alloc_model::solution::SolutionError;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use tracing::{debug, info, trace, warn};

use num_traits::{CheckedAdd, CheckedSub, ToPrimitive, Zero};

use crate::framework::state::SolverStateView;
use crate::framework::{
    err::SolverStatePlanApplyError, planning::PlanningContext, solver::ConstructionSolver,
    state::SolverState,
};

use crate::matheuristic::operatorpool::{OperatorPool, OperatorRecord};
use crate::matheuristic::policies::energy::EnergyModel;
use crate::matheuristic::policies::penalty::{PenaltyModel, RepairTrend};
use crate::matheuristic::policies::reducer::{Candidate, CandidateReducer};
use crate::matheuristic::policies::reheat::ReheatPolicy;
use crate::matheuristic::policies::scorer::OperatorScorer;
use crate::matheuristic::policies::selector::OperatorSelector;
use crate::matheuristic::policies::temp::TemperatureSchedule;
use crate::matheuristic::support::clock::Clock;
use crate::matheuristic::support::rng::SeedSequencer;

use crate::matheuristic::config::{MatheuristicConfig, PenaltyConfig};

#[derive(Debug)]
pub enum MatheuristicEngineError<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    ConstructionError(S::Error),
    PlanApply(SolverStatePlanApplyError<T>),
    Solution(SolutionError),
}

impl<T, S> From<SolverStatePlanApplyError<T>> for MatheuristicEngineError<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    fn from(e: SolverStatePlanApplyError<T>) -> Self {
        MatheuristicEngineError::PlanApply(e)
    }
}
impl<T, S> From<SolutionError> for MatheuristicEngineError<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    fn from(e: SolutionError) -> Self {
        MatheuristicEngineError::Solution(e)
    }
}

pub struct MatheuristicEngine<T, S>
where
    T: Copy + Ord,
    S: ConstructionSolver<T>,
{
    pub cfg: MatheuristicConfig,
    pub pool: OperatorPool<T>,
    pub constructor: S,

    pub temp_sched: Box<dyn TemperatureSchedule>,
    pub reheat: Box<dyn ReheatPolicy>,
    pub penalty: Box<dyn PenaltyModel>,
    pub energy: Box<dyn EnergyModel<T>>,
    pub selector: Box<dyn OperatorSelector>,
    pub scorer: Box<dyn OperatorScorer>,
    pub reducer: Box<dyn CandidateReducer<T>>,

    pub clock: Box<dyn Clock>,
    pub seeds: SeedSequencer,

    proposals_made: u64,
    iters_since_best_feasible: usize,
    temp_scale: f64,
    explore_boost_until: Option<usize>,
}

impl<T, S> MatheuristicEngine<T, S>
where
    T: Copy + Ord + Send + Sync,
    S: ConstructionSolver<T>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: MatheuristicConfig,
        ops: impl IntoIterator<Item = Box<dyn crate::matheuristic::operator::Operator<Time = T>>>,
        constructor: S,
        temp_sched: Box<dyn TemperatureSchedule>,
        reheat: Box<dyn ReheatPolicy>,
        penalty: Box<dyn PenaltyModel>,
        energy: Box<dyn EnergyModel<T>>,
        selector: Box<dyn OperatorSelector>,
        scorer: Box<dyn OperatorScorer>,
        reducer: Box<dyn CandidateReducer<T>>,
        clock: Box<dyn Clock>,
        seeds: SeedSequencer,
    ) -> Self {
        let records = ops.into_iter().map(OperatorRecord::new).collect();
        Self {
            cfg,
            pool: OperatorPool::new(records),
            constructor,
            temp_sched,
            reheat,
            penalty,
            energy,
            selector,
            scorer,
            reducer,
            clock,
            seeds,
            proposals_made: 0,
            iters_since_best_feasible: 0,
            temp_scale: 1.0,
            explore_boost_until: None,
        }
    }

    #[inline]
    pub fn construction_solver(&self) -> &S {
        &self.constructor
    }

    fn compute_initial_lambda(state: &SolverState<'_, T>, cfg: &PenaltyConfig) -> f64
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let total = state.problem().request_count();
        if total == 0 {
            return 0.0;
        }
        let init_cost = state.cost().to_f64().unwrap_or(f64::INFINITY);
        (init_cost * cfg.w) / (total as f64)
    }

    pub fn step<'state, 'p>(
        &mut self,
        state: &'state mut SolverState<'p, T>,
        iteration: usize,
        lambda: f64,
    ) -> Result<Option<Cost>, SolverStatePlanApplyError<T>>
    where
        T: CheckedAdd + CheckedSub + std::fmt::Display + Into<Cost> + Mul<Output = Cost>,
    {
        let stats_cfg = &self.cfg.stats;
        let alloc_cfg = &self.cfg.alloc;
        let anneal = &self.cfg.anneal;
        let rng_cfg = &self.cfg.random;

        let ledger = state.ledger();
        let terminal = state.terminal_occupancy();

        let temp_base = self.temp_sched.base_temp(iteration);
        let temp_eff = self
            .temp_sched
            .effective_temp(iteration, self.temp_scale)
            .max(anneal.min_temperature)
            .min(anneal.max_temperature);

        // softmax tau interpolation (same behavior as before)
        let norm = (temp_base / anneal.initial_temperature).clamp(0.0, 1.0);
        let tau = self.cfg.alloc.softmax_tau_min
            + (self.cfg.alloc.softmax_tau_max - self.cfg.alloc.softmax_tau_min) * norm;

        let mut explore_frac = alloc_cfg.explore_frac;
        if let Some(until) = self.explore_boost_until {
            if iteration < until {
                explore_frac = explore_frac.max(self.cfg.stagnation.explore_boost);
            } else {
                self.explore_boost_until = None;
            }
        }

        let n_ops = self.pool.len();
        if n_ops == 0 {
            trace!("No operators");
            return Ok(None);
        }

        // 1) scores -> weights
        let raw: Vec<f64> = self
            .pool
            .stats_slice()
            .map(|s| self.scorer.raw_score(s))
            .collect();
        let weights = self.scorer.to_weights(&raw, tau.max(1e-6), explore_frac);

        // 2) draw jobs with caps
        let total_draws = usize::max(n_ops, alloc_cfg.target_total_proposals_per_round);
        let iter_seed = self.seeds.for_iter(iteration) ^ rng_cfg.seed_base_select;
        let mut rng = crate::matheuristic::support::rng::SeedSequencer::rng(iter_seed);
        let jobs = self.selector.draw_jobs(
            &weights,
            alloc_cfg.min_per_op,
            alloc_cfg.max_per_op,
            total_draws,
            &mut rng,
        );
        let jobs = Arc::new(jobs);

        // 3) snapshot current penalized score
        let u_cur = state.ledger().iter_unassigned_requests().count();
        let cur_cost_f = state.cost().to_f64().unwrap_or(f64::INFINITY);

        // 4) parallel propose + evaluate + reduce
        #[derive(Clone, Default)]
        struct OpAgg {
            attempts: u64,
            gen_ns_sum: f64,
            eval_ns_sum: f64,
            gen_ns_count: u64,
            eval_ns_count: u64,
        }
        impl OpAgg {
            fn add_attempt(&mut self) {
                self.attempts += 1;
            }
            fn add_timing(&mut self, g: f64, e: f64) {
                if g > 0.0 {
                    self.gen_ns_sum += g;
                    self.gen_ns_count += 1;
                }
                if e > 0.0 {
                    self.eval_ns_sum += e;
                    self.eval_ns_count += 1;
                }
            }
        }

        struct ThreadAccum<'p, T: Copy + Ord> {
            cand: Option<Candidate<'p, T>>,
            per_op: Vec<OpAgg>,
        }

        impl<'p, T: Copy + Ord> ThreadAccum<'p, T> {
            fn empty(n: usize) -> Self {
                Self {
                    cand: None,
                    per_op: vec![OpAgg::default(); n],
                }
            }
            fn merge(
                mut self,
                other: Self,
                temp: f64,
                red: &dyn CandidateReducer<T>,
                rng: &mut dyn RngCore,
            ) -> Self {
                for (l, r) in self.per_op.iter_mut().zip(other.per_op.iter()) {
                    l.attempts += r.attempts;
                    l.gen_ns_sum += r.gen_ns_sum;
                    l.eval_ns_sum += r.eval_ns_sum;
                    l.gen_ns_count += r.gen_ns_count;
                    l.eval_ns_count += r.eval_ns_count;
                }
                self.cand = red.pick(self.cand, other.cand, temp, rng);
                self
            }
        }

        let base_seed = self.seeds.for_iter(iteration) ^ self.cfg.random.seed_base_task;

        let reduced = (0..jobs.len())
            .into_par_iter()
            .map(|j| j)
            .fold(
                || ThreadAccum::<'p, T>::empty(n_ops),
                |mut acc, j| {
                    let op_idx = jobs[j];
                    let job_seed = self.seeds.for_job(base_seed, j);
                    let mut job_rng = ChaCha8Rng::seed_from_u64(job_seed);

                    let ctx = PlanningContext::new(ledger, terminal);
                    let t0 = self.clock.now();
                    let plan =
                        self.pool
                            .get(op_idx)
                            .operator()
                            .propose(iteration, ctx, &mut job_rng);
                    let gen_ns = self.clock.now().duration_since(t0).as_nanos() as f64;

                    acc.per_op[op_idx].add_attempt();
                    if plan.is_none() {
                        return acc;
                    }
                    let plan = plan.unwrap();

                    // evaluation (delta only; your original also had an eval timer split)
                    let t1 = self.clock.now();
                    let delta = plan.delta_cost();
                    let eval_ns = self.clock.now().duration_since(t1).as_nanos() as f64;
                    acc.per_op[op_idx].add_timing(gen_ns, eval_ns);

                    let unassigned = plan.ledger().iter_unassigned_requests().count();
                    let feasible = unassigned == 0usize;

                    let delta_f = delta.to_f64().unwrap_or(f64::INFINITY);
                    let jitter_sample = (job_rng.next_u64() as f64) / (u64::MAX as f64);
                    let energy = self.energy.energy(
                        cur_cost_f,
                        u_cur,
                        delta_f,
                        unassigned,
                        &*self.penalty,
                        lambda,
                        self.cfg.anneal.jitter,
                        jitter_sample,
                    );

                    if delta.is_zero() && unassigned == u_cur {
                        return acc;
                    }

                    let cand = Candidate::new(op_idx, plan, delta, unassigned, feasible, energy);
                    acc.cand = self
                        .reducer
                        .pick(acc.cand, Some(cand), temp_eff, &mut job_rng);
                    acc
                },
            )
            .reduce(
                || ThreadAccum::<'p, T>::empty(n_ops),
                |a, b| {
                    // Need an RNG for the reducer during merge
                    let mut r = ChaCha8Rng::seed_from_u64(base_seed ^ 0xDEADBEEFCAFEBABEu64);
                    a.merge(b, temp_eff, &*self.reducer, &mut r)
                },
            );

        // 5) apply aggregates to stats
        for (i, agg) in reduced.per_op.iter().enumerate() {
            if agg.attempts == 0 {
                continue;
            }
            let st = self.pool.get_mut(i).stats_mut();
            st.attempts += agg.attempts;
            if agg.gen_ns_count > 0 || agg.eval_ns_count > 0 {
                let g = if agg.gen_ns_count > 0 {
                    agg.gen_ns_sum / agg.gen_ns_count as f64
                } else {
                    0.0
                };
                let e = if agg.eval_ns_count > 0 {
                    agg.eval_ns_sum / agg.eval_ns_count as f64
                } else {
                    0.0
                };
                st.on_timing(g, e, stats_cfg.gen_time_alpha, stats_cfg.eval_time_alpha);
            }
        }

        self.proposals_made = self
            .proposals_made
            .saturating_add(reduced.per_op.iter().map(|a| a.attempts).sum());

        let Some(winner) = reduced.cand else {
            trace!("No candidate");
            return Ok(None);
        };

        // Optional neutral guard
        let u_cur_now = u_cur;
        if winner.delta.is_zero() && winner.unassigned == u_cur_now {
            return Ok(None);
        }

        debug!(
            op = self.pool.get(winner.op_idx).operator().name(),
            delta = %winner.delta,
            energy = %winner.energy,
            unassigned = winner.unassigned,
            "Applying candidate"
        );

        match state.apply_plan(winner.plan) {
            Ok(()) => {
                let beneficial = (winner.energy < 0.0)
                    || (winner.unassigned < u_cur_now)
                    || (winner.delta < Cost::zero());
                if beneficial {
                    self.pool
                        .get_mut(winner.op_idx)
                        .stats_mut()
                        .on_accept(winner.delta, stats_cfg.reward_alpha);
                }
                Ok(Some(winner.delta))
            }
            Err(e) => {
                warn!(error=%e, op= self.pool.get(winner.op_idx).operator().name(), "Plan application failed");
                Err(e)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_defaults(
        cfg: MatheuristicConfig,
        ops: impl IntoIterator<Item = Box<dyn crate::matheuristic::operator::Operator<Time = T>>>,
        constructor: S,
    ) -> Self
    where
        T: Send + Sync,
    {
        use crate::matheuristic::policies::energy::RelativeEnergy;
        use crate::matheuristic::policies::penalty::DefaultPenalty;
        use crate::matheuristic::policies::reducer::SimulatedAnnealingReducer;
        use crate::matheuristic::policies::reheat::DefaultReheat;
        use crate::matheuristic::policies::scorer::LinearScorer;
        use crate::matheuristic::policies::selector::CappedWeightedSelector;
        use crate::matheuristic::policies::temp::GeometricSchedule;
        use crate::matheuristic::support::clock::SystemClock;

        // Temperature schedule from anneal config
        let temp_sched = Box::new(GeometricSchedule {
            initial: cfg.anneal.initial_temperature,
            rate: cfg.anneal.cooling_rate,
            min_t: cfg.anneal.min_temperature,
            max_t: cfg.anneal.max_temperature,
        });

        // Reheat policy from stagnation config
        let reheat = Box::new(DefaultReheat {
            iter_threshold: cfg.stagnation.iter_threshold,
            reheat_multiplier: cfg.stagnation.reheat_multiplier,
            explore_boost_iters: cfg.stagnation.explore_boost_iters,
            reset_operator_stats_on_reheat: cfg.stagnation.reset_operator_stats_on_reheat,
        });

        // Penalty model from penalty config
        let penalty = Box::new(DefaultPenalty {
            use_penalty: cfg.penalty.use_penalty,
            lambda_min: cfg.penalty.lambda_min,
            lambda_max: cfg.penalty.lambda_max,
            lambda_decay: cfg.penalty.lambda_decay,
            lambda_growth: cfg.penalty.lambda_growth,
        });

        // Relative penalized-energy model (no params)
        let energy = Box::new(RelativeEnergy);

        // Selection and scoring from alloc/stats config
        let selector = Box::new(CappedWeightedSelector);
        let scorer = Box::new(LinearScorer {
            speed_weight: cfg.alloc.speed_weight,
            success_weight: cfg.alloc.success_weight,
            min_ns_per_proposal: cfg.stats.min_ns_per_proposal,
            bootstrap_success_rate: cfg.stats.bootstrap_success_rate,
        });

        // SA reducer
        let reducer = Box::new(SimulatedAnnealingReducer);

        // Clock and seeds
        let clock = Box::new(SystemClock);
        let seeds = SeedSequencer::new(cfg.random.seed_base_select ^ cfg.random.seed_base_task);

        Self::new(
            cfg,
            ops,
            constructor,
            temp_sched,
            reheat,
            penalty,
            energy,
            selector,
            scorer,
            reducer,
            clock,
            seeds,
        )
    }
}

impl<T, S> crate::framework::solver::Solver<T> for MatheuristicEngine<T, S>
where
    T: Copy
        + Ord
        + Send
        + Sync
        + std::fmt::Display
        + CheckedAdd
        + CheckedSub
        + Mul<Output = Cost>
        + Into<Cost>,
    S: ConstructionSolver<T>,
    S::Error: std::fmt::Debug,
{
    type Error = MatheuristicEngineError<T, S>;

    fn solve<'p>(
        &mut self,
        problem: &'p Problem<T>,
    ) -> Result<Option<SolutionRef<'p, T>>, Self::Error> {
        let mut state = self
            .constructor
            .construct(problem)
            .map_err(MatheuristicEngineError::ConstructionError)?;

        let mut lambda = Self::compute_initial_lambda(&state, &self.cfg.penalty)
            .max(self.cfg.penalty.lambda_min);

        let budget = Duration::from_millis(self.cfg.max_solver_time_ms);
        let t0 = self.clock.now();

        let mut cum_delta = Cost::zero();
        let mut best_feasible_state: Option<_> = state.is_feasible().then(|| state.clone());
        let mut best_feasible_cum: Option<Cost> = state.is_feasible().then_some(cum_delta);

        self.iters_since_best_feasible = 0;
        self.temp_scale = 1.0;
        self.explore_boost_until = None;

        let mut prev_unassigned = state.ledger().iter_unassigned_requests().count();
        let mut iters_no_drop: usize = 0;

        let mut iter: usize = 0;
        while self.clock.now().duration_since(t0) < budget {
            match self.step(&mut state, iter, lambda) {
                Ok(Some(delta)) => {
                    cum_delta += delta;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error=%e, "Step failed; continuing");
                }
            }

            let cur_u = state.ledger().iter_unassigned_requests().count();
            let improved_feasible = if state.is_feasible() {
                match best_feasible_cum {
                    None => true,
                    Some(best) => cum_delta < best,
                }
            } else {
                false
            };

            if improved_feasible {
                best_feasible_cum = Some(cum_delta);
                best_feasible_state = Some(state.clone());
                self.iters_since_best_feasible = 0;
                self.temp_scale = 1.0;
                self.explore_boost_until = None;
                if self.cfg.penalty.use_penalty {
                    lambda = self.penalty.update_lambda(lambda, RepairTrend::Better);
                }
            } else {
                self.iters_since_best_feasible = self.iters_since_best_feasible.saturating_add(1);
                if self.cfg.penalty.use_penalty {
                    let trend = if cur_u < prev_unassigned {
                        RepairTrend::Better
                    } else if cur_u > prev_unassigned {
                        RepairTrend::Worse
                    } else {
                        RepairTrend::Unchanged
                    };
                    lambda = self.penalty.update_lambda(lambda, trend);
                    iters_no_drop = if matches!(trend, RepairTrend::Better) {
                        0
                    } else {
                        iters_no_drop.saturating_add(1)
                    };
                }
                let (new_scale, until, reset) =
                    self.reheat.on_stagnation(iters_no_drop, self.temp_scale);
                if new_scale != self.temp_scale || until.is_some() {
                    self.temp_scale = new_scale;
                    self.explore_boost_until = until;
                    if reset {
                        self.pool.reset_stats();
                    }
                }
            }

            prev_unassigned = cur_u;
            iter += 1;
            if (iter & 0xF) == 0 && self.clock.now().duration_since(t0) >= budget {
                break;
            }
        }

        info!(
            iterations = iter,
            proposals = self.proposals_made,
            temperature = self.temp_sched.effective_temp(iter, self.temp_scale),
            lambda = lambda,
            "Meta solve finished"
        );

        let final_state = best_feasible_state.unwrap_or(state);
        if final_state.is_feasible() {
            Ok(Some(final_state.try_into()?))
        } else {
            Ok(None)
        }
    }
}
