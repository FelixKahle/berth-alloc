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
    model::index::{BerthIndex, RequestIndex},
    search::{
        operator::{LocalMoveOperator, RepairOperator},
        planner::{CostEvaluator, PlanBuilder, PlanExplorer, PlanningContext},
    },
    state::{plan::Plan, terminal::terminalocc::FreeBerth},
};
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::{Rng, seq::SliceRandom};
use rand_distr::uniform::SampleUniform;
use std::{
    cmp::Ordering,
    ops::{Mul, RangeInclusive},
};

#[inline]
fn is_zero_delta_plan<T>(plan: &Plan<'_, T>) -> bool
where
    T: Copy + Ord,
{
    plan.delta_unassigned == 0 && plan.delta_cost == Cost::zero() && plan.terminal_delta.is_empty()
}

#[derive(Clone, Copy, Debug)]
struct CandidatePlacement<T> {
    request_index: RequestIndex,
    berth_index: BerthIndex,
    start_time: TimePoint<T>,
    end_time: TimePoint<T>,
    cost: Cost,
}

impl<T: Copy + PartialOrd> CandidatePlacement<T> {
    #[inline]
    fn interval(&self) -> TimeInterval<T> {
        TimeInterval::new(self.start_time, self.end_time)
    }
}

/// Generate K representative starts within [lo, hi) for processing time `pt`.
fn generate_candidate_starts<T>(
    interval_start: TimePoint<T>,
    interval_end: TimePoint<T>,
    processing_time: berth_alloc_core::prelude::TimeDelta<T>,
    number_of_candidate_starts: usize,
) -> Vec<TimePoint<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let latest_start = interval_end - processing_time;
    if latest_start < interval_start {
        return Vec::new();
    }

    let mut starts: Vec<TimePoint<T>> = Vec::with_capacity(number_of_candidate_starts.max(2));
    starts.push(interval_start);

    let mut t = interval_start + processing_time;
    while t <= latest_start && starts.len() + 1 < number_of_candidate_starts {
        starts.push(t);
        t += processing_time;
    }

    if Some(&latest_start) != starts.last() {
        starts.push(latest_start);
    }

    starts.sort();
    starts.dedup();
    starts
}

/// Enumerate candidate placements for a request using the current explorer snapshot.
fn enumerate_candidates_for_request<'e, 'c, 'm, 'p, T, C>(
    explorer: &PlanExplorer<'e, 'c, '_, 'm, 'p, T, C>,
    solver_model: &crate::model::solver_model::SolverModel<'m, T>,
    request_index: RequestIndex,
    number_of_candidate_starts_per_free_interval: usize,
    maximum_total_candidates_per_request: usize,
) -> Vec<CandidatePlacement<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
{
    let mut out: Vec<CandidatePlacement<T>> = Vec::new();

    for free in explorer.iter_free_for(request_index) {
        let bi = free.berth_index();
        let Some(pt) = solver_model.processing_time(request_index, bi) else {
            continue;
        };
        let iv = free.interval();

        let starts = generate_candidate_starts(
            iv.start(),
            iv.end(),
            pt,
            number_of_candidate_starts_per_free_interval,
        );

        for s in starts {
            let e = s + pt;
            if e > iv.end() {
                continue;
            }
            if let Some(cost) = explorer.peek_cost(request_index, s, bi) {
                out.push(CandidatePlacement {
                    request_index,
                    berth_index: bi,
                    start_time: s,
                    end_time: e,
                    cost,
                });
            }
            if out.len() >= maximum_total_candidates_per_request {
                break;
            }
        }
        if out.len() >= maximum_total_candidates_per_request {
            break;
        }
    }

    // Prefer cheaper first, then earlier start, then berth index for stability.
    out.sort_by(|a, b| match a.cost.cmp(&b.cost) {
        Ordering::Less => Ordering::Less,
        Ordering::Greater => Ordering::Greater,
        Ordering::Equal => match a.start_time.cmp(&b.start_time) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => a.berth_index.cmp(&b.berth_index),
        },
    });

    out
}

/* -------------------------------------------------------------------------- */
/*                    Single-thread HiGHS worker (non-blocking)               */
/* -------------------------------------------------------------------------- */

mod highs_worker {
    use good_lp::{Expression, Solution, SolverModel, default_solver, variable, variables};
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{
            OnceLock,
            atomic::{AtomicBool, Ordering},
            mpsc::{Receiver, Sender, channel},
        },
        thread,
    };

    /// Minimize `weights · x`
    /// s.t. for each request group: sum x_i == 1 (or <= 1),
    /// and for each segment: sum x_i <= 1.
    #[derive(Debug)]
    pub struct IlpJob {
        pub n_vars: usize,
        pub weights: Vec<f64>, // len = n_vars
        pub groups_by_request: Vec<Vec<usize>>,
        pub eq_one_per_request: bool,  // true => ==1, false => <=1
        pub segments: Vec<Vec<usize>>, // each segment: sum <= 1
    }

    enum Req {
        Solve(IlpJob, Sender<Resp>),
    }
    enum Resp {
        Ok(Vec<usize>),
        Err,
    }

    struct Worker {
        tx: Sender<Req>,
    }

    static WORKER: OnceLock<Worker> = OnceLock::new();
    static BUSY: AtomicBool = AtomicBool::new(false);

    /// Non-blocking check: `true` if the single HiGHS slot appears free.
    #[inline]
    pub fn is_free() -> bool {
        !BUSY.load(Ordering::Relaxed)
    }

    /// RAII guard that releases the BUSY flag even on panic/early-return.
    struct BusyGuard;
    impl BusyGuard {
        fn try_acquire() -> Option<Self> {
            BUSY.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .ok()
                .map(|_| BusyGuard)
        }
    }
    impl Drop for BusyGuard {
        fn drop(&mut self) {
            BUSY.store(false, Ordering::Release);
        }
    }

    fn spawn_worker() -> Worker {
        let (tx, rx): (Sender<Req>, Receiver<Req>) = channel();
        thread::Builder::new()
            .name("highs-worker".into())
            .spawn(move || worker_loop(rx))
            .expect("spawn highs-worker");

        Worker { tx }
    }

    fn worker_loop(rx: Receiver<Req>) {
        while let Ok(msg) = rx.recv() {
            match msg {
                Req::Solve(job, reply) => {
                    let res = catch_unwind(AssertUnwindSafe(|| run_job(job)));
                    let _ = match res {
                        Ok(Some(chosen)) => reply.send(Resp::Ok(chosen)),
                        _ => reply.send(Resp::Err),
                    };
                }
            }
        }
    }

    fn run_job(job: IlpJob) -> Option<Vec<usize>> {
        if job.n_vars == 0 || job.weights.len() != job.n_vars {
            return None;
        }

        // Vars & objective
        let mut vars = variables!();
        let xs: Vec<_> = (0..job.n_vars)
            .map(|i| vars.add(variable().binary().name(format!("x_{i}"))))
            .collect();

        let objective = xs
            .iter()
            .enumerate()
            .fold(Expression::from(0.0), |acc, (i, xi)| {
                acc + job.weights[i] * *xi
            });

        let mut prob = vars.minimise(objective).using(default_solver);

        // Request constraints
        for group in &job.groups_by_request {
            if group.is_empty() {
                continue;
            }
            let sum = group
                .iter()
                .fold(Expression::from(0.0), |acc, &i| acc + xs[i]);
            if job.eq_one_per_request {
                prob.add_constraint(sum.eq(1.0));
            } else {
                prob.add_constraint(sum.leq(1.0));
            }
        }

        // Overlap constraints
        for seg in &job.segments {
            if seg.len() <= 1 {
                continue;
            }
            let sum = seg
                .iter()
                .fold(Expression::from(0.0), |acc, &i| acc + xs[i]);
            prob.add_constraint(sum.leq(1.0));
        }

        let Ok(solution) = prob.solve() else {
            return None;
        };

        let mut chosen = Vec::new();
        for (i, xi) in xs.iter().enumerate() {
            if solution.value(*xi) >= 0.5 {
                chosen.push(i);
            }
        }
        Some(chosen)
    }

    /// Blocking solve **only if** the single HiGHS slot is free; otherwise return `None` immediately.
    pub fn try_run(job: IlpJob) -> Option<Vec<usize>> {
        let _guard = BusyGuard::try_acquire()?; // if busy, return None now

        let worker = WORKER.get_or_init(spawn_worker);
        let (tx, rx) = channel();

        // If send or recv fails, just return None; _guard drops to release BUSY.
        worker.tx.send(Req::Solve(job, tx)).ok()?;
        match rx.recv() {
            Ok(Resp::Ok(v)) => Some(v),
            _ => None,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                         Micro-MIP selector (repair)                        */
/* -------------------------------------------------------------------------- */

mod micro_mip {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    /// Solve the selection with ILP via the single-thread HiGHS worker.
    /// At most one candidate per request; segments prevent overlaps.
    pub fn solve_exact_with_ilp<T>(all_candidates: &[CandidatePlacement<T>]) -> Vec<usize>
    where
        T: Copy + Ord + CheckedAdd + CheckedSub,
    {
        if all_candidates.is_empty() {
            return Vec::new();
        }

        let n = all_candidates.len();

        // group by request
        let mut by_request: HashMap<RequestIndex, Vec<usize>> = HashMap::new();
        for (i, c) in all_candidates.iter().enumerate() {
            by_request.entry(c.request_index).or_default().push(i);
        }
        let groups_by_request: Vec<Vec<usize>> = by_request.into_values().collect();

        // segments per berth (partition time into elementary segments; sum ≤ 1 per segment)
        let mut by_berth: BTreeMap<BerthIndex, Vec<usize>> = BTreeMap::new();
        for (i, c) in all_candidates.iter().enumerate() {
            by_berth.entry(c.berth_index).or_default().push(i);
        }

        let mut segments: Vec<Vec<usize>> = Vec::new();
        for (_b, idxs) in by_berth {
            if idxs.is_empty() {
                continue;
            }
            let mut points: Vec<TimePoint<T>> = idxs
                .iter()
                .flat_map(|&i| [all_candidates[i].start_time, all_candidates[i].end_time])
                .collect();
            points.sort();
            points.dedup();

            for w in points.windows(2) {
                let (seg_start, seg_end) = (w[0], w[1]);
                if seg_end <= seg_start {
                    continue;
                }
                let seg: Vec<usize> = idxs
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let c = all_candidates[i];
                        !(c.end_time <= seg_start || c.start_time >= seg_end)
                    })
                    .collect();
                if seg.len() > 1 {
                    segments.push(seg);
                }
            }
        }

        // rank-based weights: cheaper → smaller rank
        let mut uniq: Vec<Cost> = all_candidates.iter().map(|c| c.cost).collect();
        uniq.sort();
        uniq.dedup();
        let mut rank_of: HashMap<Cost, f64> = HashMap::new();
        for (r, &cost) in uniq.iter().enumerate() {
            rank_of.insert(cost, r as f64 + 1.0);
        }
        let mut weights = vec![0.0; n];
        for i in 0..n {
            weights[i] = rank_of[&all_candidates[i].cost];
        }

        // Non-blocking solve on the dedicated worker
        let job = super::highs_worker::IlpJob {
            n_vars: n,
            weights,
            groups_by_request,
            eq_one_per_request: false, // repair: ≤ 1 per request
            segments,
        };

        super::highs_worker::try_run(job).unwrap_or_default()
    }
}

/* -------------------------------------------------------------------------- */
/*                    MatheuristicRepair operator (ILP)                       */
/* -------------------------------------------------------------------------- */

#[derive(Clone, Debug)]
pub struct MatheuristicRepair {
    /// Only optimize up to this many unassigned requests (sample if more).
    pub maximum_unassigned_to_optimize_range: RangeInclusive<usize>,
    /// Number of candidate starts generated per free interval.
    pub number_of_candidate_starts_per_free_interval_range: RangeInclusive<usize>,
    /// Hard cap per request to keep the micro model small.
    pub maximum_total_candidates_per_request_range: RangeInclusive<usize>,
}

impl Default for MatheuristicRepair {
    fn default() -> Self {
        Self {
            maximum_unassigned_to_optimize_range: 24..=64,
            number_of_candidate_starts_per_free_interval_range: 3..=6,
            maximum_total_candidates_per_request_range: 6..=10,
        }
    }
}

impl MatheuristicRepair {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn sample_from_range<R, N>(range: RangeInclusive<N>, rng: &mut R) -> N
    where
        R: Rng,
        N: Copy + PartialOrd + SampleUniform,
    {
        rng.random_range(range)
    }
}

impl<T, C, R> RepairOperator<T, C, R> for MatheuristicRepair
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost> + Send + Sync,
    C: CostEvaluator<T>,
    R: Rng,
{
    fn name(&self) -> &str {
        "MatheuristicRepair"
    }

    fn repair<'b, 'c, 's, 'm, 'p>(
        &self,
        planning_context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        // Non-blocking gate: if the single HiGHS slot is busy, bail out immediately.
        if !highs_worker::is_free() {
            return None;
        }

        let solver_model = planning_context.model();
        let mut builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = planning_context.builder();

        let max_unassigned =
            Self::sample_from_range(self.maximum_unassigned_to_optimize_range.clone(), rng);
        let cand_starts = Self::sample_from_range(
            self.number_of_candidate_starts_per_free_interval_range
                .clone(),
            rng,
        );
        let max_cands_per_req =
            Self::sample_from_range(self.maximum_total_candidates_per_request_range.clone(), rng);

        // collect unassigned
        let mut unassigned: Vec<RequestIndex> =
            builder.with_explorer(|ex| ex.iter_unassigned().collect());
        if unassigned.is_empty() {
            return None;
        }

        // prioritize constrained (few feasible positions)
        builder.with_explorer(|ex| {
            unassigned.sort_by(|&ra, &rb| {
                let positions = |r: RequestIndex| -> usize {
                    ex.iter_free_for(r)
                        .filter(|free| {
                            let bi = free.berth_index();
                            // Use solver_model from outer scope (PlanExplorer may not expose model())
                            solver_model.processing_time(r, bi).is_some_and(|pt| {
                                free.interval().start() + pt <= free.interval().end()
                            })
                        })
                        .count()
                };
                positions(ra).cmp(&positions(rb))
            });
        });

        if unassigned.len() > max_unassigned {
            unassigned.truncate(max_unassigned);
        }

        // enumerate candidates
        let all_candidates: Vec<CandidatePlacement<T>> = builder.with_explorer(|ex| {
            let mut out = Vec::new();
            for &ri in &unassigned {
                let mut v = enumerate_candidates_for_request(
                    ex,
                    solver_model,
                    ri,
                    cand_starts,
                    max_cands_per_req,
                );
                out.append(&mut v);
            }
            out
        });

        if all_candidates.is_empty() {
            return None;
        }

        // ILP select (single-worker, non-blocking)
        let chosen = micro_mip::solve_exact_with_ilp(&all_candidates);
        if chosen.is_empty() {
            return None;
        }

        // emit patches
        for &i in &chosen {
            let c = all_candidates[i];
            let fb = FreeBerth::new(c.interval(), c.berth_index);
            let _ = builder.propose_assignment(c.request_index, c.start_time, &fb);
        }

        let plan = builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

/* -------------------------------------------------------------------------- */
/*             MipBlockReoptimize (local micro-MIP block re-opt)              */
/* -------------------------------------------------------------------------- */

#[derive(Clone, Debug)]
pub struct MipBlockReoptimize {
    /// Contiguous block length (k) to re-optimize.
    pub block_length_to_reoptimize_range: RangeInclusive<usize>,
    /// Candidate starts generated per free interval.
    pub number_of_candidate_starts_per_free_interval_range: RangeInclusive<usize>,
    /// Hard cap per request to keep the micro model small.
    pub maximum_total_candidates_per_request_range: RangeInclusive<usize>,
    /// If true, restrict assignments to the same berth; otherwise allow all berths.
    pub restrict_to_same_berth: bool,
}

impl MipBlockReoptimize {
    pub fn same_berth(
        block_len: RangeInclusive<usize>,
        cand_starts_per_interval: RangeInclusive<usize>,
        max_cands_per_req: RangeInclusive<usize>,
    ) -> Self {
        assert!(!block_len.is_empty());
        assert!(!cand_starts_per_interval.is_empty());
        assert!(!max_cands_per_req.is_empty());
        Self {
            block_length_to_reoptimize_range: block_len,
            number_of_candidate_starts_per_free_interval_range: cand_starts_per_interval,
            maximum_total_candidates_per_request_range: max_cands_per_req,
            restrict_to_same_berth: true,
        }
    }

    pub fn across_all_berths(
        block_len: RangeInclusive<usize>,
        cand_starts_per_interval: RangeInclusive<usize>,
        max_cands_per_req: RangeInclusive<usize>,
    ) -> Self {
        assert!(!block_len.is_empty());
        assert!(!cand_starts_per_interval.is_empty());
        assert!(!max_cands_per_req.is_empty());
        Self {
            block_length_to_reoptimize_range: block_len,
            number_of_candidate_starts_per_free_interval_range: cand_starts_per_interval,
            maximum_total_candidates_per_request_range: max_cands_per_req,
            restrict_to_same_berth: false,
        }
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for MipBlockReoptimize
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "MipBlockReoptimize"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        use std::collections::{BTreeMap, HashMap};

        // Non-blocking gate: if the single HiGHS slot is busy, bail out immediately.
        if !highs_worker::is_free() {
            return None;
        }

        #[derive(Clone, Copy, Debug)]
        struct Cand<T> {
            req: RequestIndex,
            berth: BerthIndex,
            start: TimePoint<T>,
            end: TimePoint<T>,
            cost: Cost,
        }
        impl<T: Copy + PartialOrd> Cand<T> {
            #[inline]
            fn interval(&self) -> TimeInterval<T> {
                TimeInterval::new(self.start, self.end)
            }
        }

        #[inline]
        fn clamp_range_sample<Rng: rand::Rng>(
            rng: &mut Rng,
            range: &RangeInclusive<usize>,
            hard_cap: usize,
        ) -> usize {
            let (lo, hi) = (*range.start(), *range.end());
            let drawn = if lo == hi {
                lo
            } else {
                rng.random_range(lo..=hi)
            };
            drawn.min(hard_cap).max(1)
        }

        #[inline]
        fn gen_starts<U: Copy + Ord + CheckedAdd + CheckedSub>(
            lo: TimePoint<U>,
            hi: TimePoint<U>,
            pt: berth_alloc_core::prelude::TimeDelta<U>,
            k: usize,
        ) -> Vec<TimePoint<U>> {
            let latest = hi - pt;
            if latest < lo {
                return Vec::new();
            }
            let mut v = Vec::with_capacity(k.max(2));
            v.push(lo);
            let mut t = lo + pt;
            while t <= latest && v.len() + 1 < k {
                v.push(t);
                t += pt;
            }
            if v.last().copied() != Some(latest) {
                v.push(latest);
            }
            v.sort();
            v.dedup();
            v
        }

        // pick a berth + contiguous block
        let model = context.model();
        let seed: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

        let by_berth = seed.with_explorer(|ex| {
            let mut map: BTreeMap<BerthIndex, Vec<(RequestIndex, TimePoint<T>)>> = BTreeMap::new();
            for (i, dv) in ex.decision_vars().iter().enumerate() {
                if let crate::state::decisionvar::DecisionVar::Assigned(
                    crate::state::decisionvar::Decision {
                        berth_index,
                        start_time,
                    },
                ) = *dv
                {
                    map.entry(berth_index)
                        .or_default()
                        .push((RequestIndex::new(i), start_time));
                }
            }
            for v in map.values_mut() {
                v.sort_by_key(|&(_, s)| s.value());
            }
            map
        });
        if by_berth.is_empty() {
            return None;
        }

        let mut candidate_berths: Vec<_> = by_berth.iter().filter(|(_, v)| v.len() >= 2).collect();
        if candidate_berths.is_empty() {
            return None;
        }
        candidate_berths.shuffle(rng);
        let (&berth_choice, seq) = candidate_berths[0];

        let k = clamp_range_sample(rng, &self.block_length_to_reoptimize_range, seq.len())
            .min(seq.len())
            .max(2);
        let max_start = seq.len() - k;
        let start_idx = if max_start == 0 {
            0
        } else {
            rng.random_range(0..=max_start)
        };
        let block: Vec<(RequestIndex, TimePoint<T>)> = seq[start_idx..start_idx + k].to_vec();
        let block_req_indices: Vec<RequestIndex> = block.iter().map(|&(ri, _)| ri).collect();

        // original placements (on berth_choice)
        let mut original: HashMap<RequestIndex, (BerthIndex, TimePoint<T>)> = HashMap::new();
        for &(ri, s) in &block {
            original.insert(ri, (berth_choice, s));
        }

        // fresh builder: unassign block
        let mut builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        for &ri in &block_req_indices {
            if builder.propose_unassignment(ri).is_err() {
                return None;
            }
        }

        // enumerate candidates on updated snapshot
        let num_starts = clamp_range_sample(
            rng,
            &self.number_of_candidate_starts_per_free_interval_range,
            usize::MAX,
        );
        let max_cands_per_req = clamp_range_sample(
            rng,
            &self.maximum_total_candidates_per_request_range,
            usize::MAX,
        );

        const MAX_TOTAL_CANDS: usize = 2_000;

        let (all_candidates, by_req) = builder.with_explorer(|ex| {
            use std::collections::BTreeSet;

            let mut out: Vec<Cand<T>> = Vec::new();
            let mut by_req: HashMap<RequestIndex, Vec<usize>> = HashMap::new();
            let mut seen: BTreeSet<(RequestIndex, BerthIndex, T)> = BTreeSet::new();

            for &ri in &block_req_indices {
                let mut bag: Vec<Cand<T>> = Vec::new();

                for free in ex.iter_free_for(ri) {
                    if self.restrict_to_same_berth && free.berth_index() != berth_choice {
                        continue;
                    }
                    let bi = free.berth_index();
                    let Some(pt) = model.processing_time(ri, bi) else {
                        continue;
                    };
                    let iv = free.interval();

                    // suggested starts incl. original position (if feasible)
                    let mut starts = gen_starts(iv.start(), iv.end(), pt, num_starts);
                    if let Some(&(obi, os)) = original.get(&ri)
                        && (!self.restrict_to_same_berth || bi == obi)
                        && os >= iv.start()
                        && os + pt <= iv.end()
                    {
                        starts.push(os);
                    }
                    starts.sort();
                    starts.dedup();

                    for s in starts {
                        let e = s + pt;
                        if e <= s || e > iv.end() {
                            continue;
                        }
                        let key = (ri, bi, s.value());
                        if !seen.insert(key) {
                            continue;
                        }
                        if let Some(cost) = ex.peek_cost(ri, s, bi) {
                            bag.push(Cand {
                                req: ri,
                                berth: bi,
                                start: s,
                                end: e,
                                cost,
                            });
                            if bag.len() >= max_cands_per_req {
                                break;
                            }
                        }
                    }
                    if bag.len() >= max_cands_per_req {
                        break;
                    }
                }

                // prefer cheaper, then earlier
                bag.sort_by(|a, b| match a.cost.cmp(&b.cost) {
                    Ordering::Less => Ordering::Less,
                    Ordering::Greater => Ordering::Greater,
                    Ordering::Equal => a.start.value().cmp(&b.start.value()),
                });

                for c in bag {
                    let idx = out.len();
                    out.push(c);
                    by_req.entry(c.req).or_default().push(idx);
                }

                if out.len() > MAX_TOTAL_CANDS {
                    break;
                }
            }
            (out, by_req)
        });

        if all_candidates.is_empty() {
            return None;
        }
        if block_req_indices
            .iter()
            .any(|r| by_req.get(r).is_none_or(|v| v.is_empty()))
        {
            return None; // some request lost all candidates
        }

        // Build segments (clique cover) per berth for overlap ≤ 1
        let mut by_berth: BTreeMap<BerthIndex, Vec<usize>> = BTreeMap::new();
        for (i, c) in all_candidates.iter().enumerate() {
            by_berth.entry(c.berth).or_default().push(i);
        }
        let mut segments: Vec<Vec<usize>> = Vec::new();
        for (_b, idxs) in by_berth {
            if idxs.is_empty() {
                continue;
            }
            let mut points: Vec<TimePoint<T>> = idxs
                .iter()
                .flat_map(|&i| [all_candidates[i].start, all_candidates[i].end])
                .collect();
            points.sort();
            points.dedup();

            for w in points.windows(2) {
                let (seg_start, seg_end) = (w[0], w[1]);
                if seg_end <= seg_start {
                    continue;
                }
                let seg: Vec<usize> = idxs
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let c = all_candidates[i];
                        !(c.end <= seg_start || c.start >= seg_end)
                    })
                    .collect();
                if seg.len() > 1 {
                    segments.push(seg);
                }
            }
        }

        // Weights: rank (dominant) + small penalty for moving vs. original
        let mut uniq: Vec<Cost> = all_candidates.iter().map(|c| c.cost).collect();
        uniq.sort();
        uniq.dedup();
        let mut rank_w: std::collections::HashMap<Cost, f64> = std::collections::HashMap::new();
        for (r, &c) in uniq.iter().enumerate() {
            rank_w.insert(c, r as f64 + 1.0);
        }

        let moved_penalty: Vec<f64> = all_candidates
            .iter()
            .map(|c| {
                if let Some(&(obi, os)) = original.get(&c.req) {
                    if c.berth == obi && c.start == os {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            })
            .collect();

        let mut weights = vec![0.0; all_candidates.len()];
        for i in 0..all_candidates.len() {
            weights[i] = 1_000.0 * rank_w[&all_candidates[i].cost] + moved_penalty[i];
        }

        // Groups per request (exactly one per request in the block)
        let groups_by_request: Vec<Vec<usize>> = block_req_indices
            .iter()
            .map(|r| by_req[r].clone())
            .collect();

        // Solve on the dedicated HiGHS worker (non-blocking)
        let job = highs_worker::IlpJob {
            n_vars: all_candidates.len(),
            weights,
            groups_by_request,
            eq_one_per_request: true,
            segments,
        };

        let chosen_indices = highs_worker::try_run(job)?;
        if chosen_indices.is_empty() {
            return None;
        }

        // chosen per request
        let mut chosen_by_req: std::collections::HashMap<RequestIndex, usize> =
            std::collections::HashMap::new();
        for i in chosen_indices {
            chosen_by_req.insert(all_candidates[i].req, i);
        }
        if chosen_by_req.len() != block_req_indices.len() {
            return None;
        }

        // require a real change vs. original
        let any_change = chosen_by_req.iter().any(|(&ri, &i)| {
            if let Some(&(obi, os)) = original.get(&ri) {
                let c = all_candidates[i];
                c.berth != obi || c.start != os
            } else {
                true
            }
        });
        if !any_change {
            return None;
        }

        // apply
        for (&_ri, &i) in &chosen_by_req {
            let c = all_candidates[i];
            let fb = FreeBerth::new(c.interval(), c.berth);
            if builder.propose_assignment(c.req, c.start, &fb).is_err() {
                return None;
            }
        }

        let plan = builder.finalize();
        if plan.delta_unassigned != 0 || is_zero_delta_plan(&plan) {
            return None;
        }
        Some(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::planner::{DefaultCostEvaluator, PlanningContext},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};
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

    fn make_problem(n_reqs: usize, n_berths: usize, pt: i64) -> Problem<i64> {
        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();
        for b in 1..=n_berths {
            builder.add_berth(berth(b as u32, 0, 5_000));
        }
        for r in 1..=n_reqs {
            let pts = (1..=n_berths).map(|b| (b as u32, pt)).collect::<Vec<_>>();
            builder.add_flexible(flex_req(r as u32, (0, 2_000), &pts, 1));
        }
        builder.build().expect("valid problem")
    }

    fn make_unassigned_state<'p>(model: &SolverModel<'p, i64>) -> SolverState<'p, i64> {
        let dv = DecisionVarVec::from(vec![
            DecisionVar::unassigned();
            model.flexible_requests_len()
        ]);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let fit = Fitness::new(0, model.flexible_requests_len());
        SolverState::new(dv, term, fit)
    }

    fn make_ctx<'b, 'c, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        cost_eval: &'c DefaultCostEvaluator,
        state: &'s SolverState<'p, i64>,
        buffer: &'b mut [DecisionVar<i64>],
    ) -> PlanningContext<'b, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator> {
        PlanningContext::new(model, state, cost_eval, buffer)
    }

    #[test]
    fn matheuristic_repair_assigns_nontrivial_set_or_noop_if_solver_missing_or_busy() {
        let prob = make_problem(20, 3, 10);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_unassigned_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(42);

        let op = MatheuristicRepair::new();
        match op.repair(&mut ctx, &mut rng) {
            Some(plan) => {
                assert!(!plan.decision_var_patches.is_empty());
                assert!(plan.delta_unassigned < 0);
            }
            None => eprintln!("HiGHS/good_lp unavailable, infeasible, or busy; test no-op."),
        }
    }

    #[test]
    fn matheuristic_respects_request_uniqueness_or_noop_if_solver_missing_or_busy() {
        let prob = make_problem(5, 2, 12);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_unassigned_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(7);

        let op = MatheuristicRepair::new();
        if let Some(plan) = op.repair(&mut ctx, &mut rng) {
            use std::collections::HashSet;
            let mut seen: HashSet<_> = HashSet::new();
            for p in &plan.decision_var_patches {
                assert!(seen.insert(p.index));
            }
        } else {
            eprintln!("HiGHS/good_lp unavailable, infeasible, or busy; test no-op.");
        }
    }
}
