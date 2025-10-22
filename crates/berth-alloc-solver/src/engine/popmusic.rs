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
    engine::search::{SearchContext, SearchStrategy},
    model::index::{BerthIndex, RequestIndex},
    search::planner::{DefaultCostEvaluator, PlanningContext},
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::{SolverState, SolverStateView},
        terminal::terminalocc::{FreeBerth, TerminalRead},
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use good_lp::solvers::highs::highs;
use good_lp::*;
use rand::seq::SliceRandom;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering as AtomicOrdering;

/// One feasible placement option for a request.
#[derive(Clone, Debug)]
struct Placement<T: Copy + Ord> {
    req: RequestIndex,
    berth: BerthIndex,
    start: TimePoint<T>,
    end: TimePoint<T>,
    cost: Cost,
}

/// Parameterization of the POPMUSIC matheuristic.
/// Defaults chosen to match the paper:
///   t_parts=8, r_neighbors=1, k_per_free=6, per_subproblem_secs=20, rounds_without_gain_stop=8
#[derive(Clone)]
pub struct PopmusicParams {
    pub t_parts: usize,                  // number of time parts
    pub r_neighbors: usize,              // seed + r forward neighbor parts
    pub k_per_free: usize,               // placements sampled per free interval
    pub per_subproblem_secs: u64,        // MILP time limit for HiGHS
    pub rounds_without_gain_stop: usize, // global patience
}

impl Default for PopmusicParams {
    fn default() -> Self {
        Self {
            t_parts: 8,
            r_neighbors: 1,
            k_per_free: 6,
            per_subproblem_secs: 20,
            rounds_without_gain_stop: 8,
        }
    }
}

/// POPMUSIC strategy using a placement MILP solved with `good_lp` + HiGHS.
pub struct PopmusicStrategy<T, R> {
    params: PopmusicParams,
    _pd: std::marker::PhantomData<(T, R)>,
}

impl<T, R> PopmusicStrategy<T, R> {
    pub fn new(params: PopmusicParams) -> Self {
        Self {
            params,
            _pd: Default::default(),
        }
    }
}

impl<T, R> PopmusicStrategy<T, R>
where
    T: SolveNumeric + From<i32> + Copy + Ord + Into<Cost>,
    R: rand::Rng,
{
    #[inline]
    pub fn with_params(params: PopmusicParams) -> Self {
        Self::new(params)
    }

    /// Partition requests by assigned *start* time into `t` contiguous parts.
    /// (Unassigned requests are ignored here; they’ll be indirectly covered as neighbors shift
    /// across iterations.)
    fn partition_into_time_parts<'p>(
        &self,
        model: &crate::model::solver_model::SolverModel<'p, T>,
        state: &SolverState<'p, T>,
        t: usize,
    ) -> Vec<Vec<RequestIndex>> {
        if t == 0 {
            return vec![];
        }

        // Compute planning horizon H = max end over assigned
        let mut max_end: Option<TimePoint<T>> = None;
        for (i, dv) in state.decision_variables().iter().enumerate() {
            if let DecisionVar::Assigned(Decision {
                berth_index,
                start_time,
            }) = dv
            {
                if let Some(iv) = model.interval(RequestIndex::new(i), *berth_index, *start_time) {
                    max_end = Some(max_end.map_or(iv.end(), |m| m.max(iv.end())));
                }
            }
        }
        let h = max_end.unwrap_or_else(|| TimePoint::new(0.into()));

        // Assign each assigned request to a time bucket based on its start_time within [0, H)
        let mut parts = vec![Vec::new(); t];
        for (i, dv) in state.decision_variables().iter().enumerate() {
            if let DecisionVar::Assigned(Decision { start_time, .. }) = dv {
                let s_i64 = start_time.value().to_i64().unwrap_or(0);
                let h_i64 = h.value().to_i64().unwrap_or(1).max(1);
                let idx = ((s_i64 * t as i64) / h_i64).clamp(0, t as i64 - 1) as usize;
                parts[idx].push(RequestIndex::new(i));
            }
        }
        parts
    }

    /// Build and solve a small placement MILP for the given sub-request set.
    /// Returns chosen placements keyed by request index on success.
    fn solve_subproblem<'e, 'm, 'p>(
        &self,
        ctx: &mut SearchContext<'e, 'm, 'p, T, R>,
        sub_requests: &BTreeSet<usize>,
    ) -> Option<BTreeMap<usize, Placement<T>>> {
        let stop = ctx.stop();
        if stop.load(AtomicOrdering::Relaxed) {
            return None;
        }

        let model = ctx.model();
        let cost_eval = DefaultCostEvaluator;

        // Snapshot state to explore with a stable terminal & DV array.
        let snap = ctx.shared_incumbent().snapshot();
        if stop.load(AtomicOrdering::Relaxed) {
            return None;
        }

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pc = PlanningContext::new(model, &snap, &cost_eval, work_buf.as_mut_slice());

        // Collect candidate placements per request.
        let mut by_req: BTreeMap<usize, Vec<Placement<T>>> = BTreeMap::new();

        pc.builder().with_explorer(|ex| {
            'reqs: for &r_usize in sub_requests {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'reqs;
                }
                let r = RequestIndex::new(r_usize);

                // Always include a "keep" placement if currently assigned (feasibility anchor)
                if let DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                }) = ex.decision_vars()[r_usize]
                {
                    if let Some(iv) = model.interval(r, berth_index, start_time) {
                        if let Some(cost) = ex.peek_cost(r, start_time, berth_index) {
                            by_req.entry(r_usize).or_default().push(Placement {
                                req: r,
                                berth: berth_index,
                                start: start_time,
                                end: iv.end(),
                                cost,
                            });
                        }
                    }
                }

                // Sample placements from *currently free* intervals
                let mut count_for_req = 0usize;
                for free in ex.iter_free_for(r) {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    if count_for_req >= 5 * self.params.k_per_free {
                        break;
                    }

                    let b = free.berth_index();
                    if let Some(dur) = model.processing_time(r, b) {
                        let free_len_i64 = free.interval().length().value().to_i64().unwrap_or(0);
                        let dur_i64 = dur.value().to_i64().unwrap_or(0);
                        let slack = free_len_i64 - dur_i64;
                        if slack < 0 {
                            continue;
                        }

                        let start0 = free.interval().start();
                        let k = self.params.k_per_free.max(1);

                        for s_idx in 0..k {
                            if stop.load(AtomicOrdering::Relaxed) {
                                break;
                            }
                            let ofs_i64 = if k == 1 {
                                0
                            } else {
                                (slack * s_idx as i64) / (k as i64 - 1)
                            };
                            let ofs_t = num_traits::FromPrimitive::from_i64(ofs_i64)
                                .unwrap_or_else(T::zero);
                            let start = start0 + TimeDelta::new(ofs_t);
                            let end = start + dur;

                            if !free
                                .interval()
                                .contains_interval(&TimeInterval::new(start, end))
                            {
                                continue;
                            }
                            if let Some(cost) = ex.peek_cost(r, start, b) {
                                by_req.entry(r_usize).or_default().push(Placement {
                                    req: r,
                                    berth: b,
                                    start,
                                    end,
                                    cost,
                                });
                                count_for_req += 1;
                                if count_for_req >= 5 * self.params.k_per_free {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });

        if stop.load(AtomicOrdering::Relaxed) {
            return None;
        }

        // If any request lacks candidates, skip this subproblem.
        if by_req.values().any(|v| v.is_empty()) {
            return None;
        }

        // Map each candidate to a global binary var index.
        let mut index_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut all_vars: Vec<(usize, usize)> = Vec::new();
        for (r, vecp) in &by_req {
            for (k, _) in vecp.iter().enumerate() {
                let id = all_vars.len();
                index_of.insert((*r, k), id);
                all_vars.push((*r, k));
            }
        }

        if stop.load(AtomicOrdering::Relaxed) {
            return None;
        }

        // Build conflicts berth-by-berth (interval packing).
        let mut by_berth: BTreeMap<BerthIndex, Vec<(usize, usize)>> = BTreeMap::new();
        for (r, vecp) in &by_req {
            for (k, p) in vecp.iter().enumerate() {
                by_berth.entry(p.berth).or_default().push((*r, k));
            }
        }
        let mut conflicts: Vec<(usize, usize)> = Vec::new();
        for (_b, idxs) in by_berth {
            if stop.load(AtomicOrdering::Relaxed) {
                return None;
            }
            for i in 0..idxs.len() {
                if stop.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                for j in (i + 1)..idxs.len() {
                    let (r_i, k_i) = idxs[i];
                    let (r_j, k_j) = idxs[j];
                    let pi = &by_req[&r_i][k_i];
                    let pj = &by_req[&r_j][k_j];
                    let overlap = !(pi.end <= pj.start || pj.end <= pi.start);
                    if overlap {
                        conflicts.push((index_of[&(r_i, k_i)], index_of[&(r_j, k_j)]));
                    }
                }
            }
        }

        if stop.load(AtomicOrdering::Relaxed) {
            return None;
        }

        // === Build MILP in good_lp with HiGHS ===
        let mut vars = variables!();
        let y: Vec<Variable> = (0..all_vars.len())
            .map(|i| vars.add(variable().binary().name(format!("y_{i}"))))
            .collect();

        // Objective: minimize sum(cost_jk * y_jk)
        let objective = y
            .iter()
            .enumerate()
            .fold(Expression::from(0.0), |acc, (idx, _)| {
                let (r, k) = all_vars[idx];
                let p = &by_req[&r][k];
                acc + (p.cost as f64) * y[idx]
            });

        let mut prob = vars
            .minimise(objective)
            .using(highs)
            .with_time_limit(self.params.per_subproblem_secs as f64);

        // One placement per request: sum_k y[r,k] == 1
        for (r, vecp) in &by_req {
            if stop.load(AtomicOrdering::Relaxed) {
                return None;
            }
            let sum = (0..vecp.len())
                .map(|k| index_of[&(*r, k)])
                .fold(Expression::from(0.0), |acc, i| acc + y[i]);
            prob.add_constraint(sum.eq(1.0));
        }

        // Conflict constraints: y[i] + y[j] <= 1
        for (i, j) in conflicts {
            if stop.load(AtomicOrdering::Relaxed) {
                return None;
            }
            prob.add_constraint((y[i] + y[j]).leq(1.0));
        }

        if stop.load(AtomicOrdering::Relaxed) {
            return None;
        }

        // Solve (bounded by HiGHS time limit)
        let Ok(sol) = prob.solve() else {
            return None;
        };

        // Extract chosen placements
        let mut pick: BTreeMap<usize, Placement<T>> = BTreeMap::new();
        for (idx, &(r, k)) in all_vars.iter().enumerate() {
            if stop.load(AtomicOrdering::Relaxed) {
                return None;
            }
            if sol.value(y[idx]) >= 0.5 {
                pick.insert(r, by_req[&r][k].clone());
            }
        }
        Some(pick)
    }
}

impl<T, R> SearchStrategy<T, R> for PopmusicStrategy<T, R>
where
    T: SolveNumeric + From<i32> + Copy + Ord,
    R: rand::Rng + Send + Sync,
{
    fn name(&self) -> &str {
        "POPMUSIC (HiGHS matheuristic)"
    }

    fn run<'e, 'm, 'p>(&mut self, ctx: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = ctx.stop();
        let model = ctx.model();
        let t = self.params.t_parts.max(1);

        let mut best = ctx.shared_incumbent().snapshot();
        let mut rounds_no_gain = 0usize;

        while !stop.load(AtomicOrdering::Relaxed)
            && rounds_no_gain < self.params.rounds_without_gain_stop
        {
            let parts = self.partition_into_time_parts(model, &best, t);
            if stop.load(AtomicOrdering::Relaxed) {
                break;
            }

            let mut improved_any = false;

            // Randomize seed order for diversification
            let mut seed_ids: Vec<usize> = (0..t).collect();
            seed_ids.shuffle(ctx.rng());

            'seed_loop: for &seed in &seed_ids {
                if stop.load(AtomicOrdering::Relaxed) {
                    break;
                }

                // Seed + r forward neighbor parts (time-consistent block)
                let mut sub: BTreeSet<usize> = BTreeSet::new();
                for delta in 0..=self.params.r_neighbors {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let id = (seed + delta).min(t - 1);
                    for &r in &parts[id] {
                        if stop.load(AtomicOrdering::Relaxed) {
                            break;
                        }
                        sub.insert(r.get());
                    }
                }
                if sub.is_empty() || stop.load(AtomicOrdering::Relaxed) {
                    continue;
                }

                // Solve subproblem (MILP over sampled placements)
                if let Some(chosen) = self.solve_subproblem(ctx, &sub) {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break;
                    }

                    // Build and apply a plan with the chosen placements
                    let cost_eval = DefaultCostEvaluator;
                    let mut dv_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
                    let mut pc =
                        PlanningContext::new(model, &best, &cost_eval, dv_buf.as_mut_slice());

                    let plan = pc.with_builder(|pb| {
                        // Unassign all sub-requests (ignore errors if already unassigned)
                        for &r in &sub {
                            if stop.load(AtomicOrdering::Relaxed) {
                                break;
                            }
                            let ri = RequestIndex::new(r);
                            let _ = pb.propose_unassignment(ri);
                        }
                        // Assign chosen placements
                        for (r, p) in chosen.values().map(|pl| (pl.req, pl)).collect::<Vec<_>>() {
                            if stop.load(AtomicOrdering::Relaxed) {
                                break;
                            }
                            // Find a matching free segment (should exist by construction)
                            let mut picked: Option<FreeBerth<T>> = None;
                            for free in pb.sandbox().inner().iter_free_intervals_for_berths_in(
                                [p.berth],
                                model.feasible_interval(r),
                            ) {
                                if stop.load(AtomicOrdering::Relaxed) {
                                    break;
                                }
                                let iv = TimeInterval::new(p.start, p.end);
                                if free.interval().contains_interval(&iv) {
                                    picked = Some(free);
                                    break;
                                }
                            }
                            if let Some(free) = picked {
                                let _ = pb.propose_assignment(r, p.start, &free);
                            }
                        }
                    });

                    if stop.load(AtomicOrdering::Relaxed) {
                        break;
                    }

                    let mut candidate = best.clone();
                    candidate.apply_plan(plan);

                    if candidate.fitness() < best.fitness() {
                        best = candidate;
                        let succ = ctx.shared_incumbent().try_update(&best);
                        if succ {
                            tracing::info!("[POPMUSIC] Improved incumbent: cost={} ", best.cost());
                        }
                        improved_any = true;
                        // Restart POPMUSIC loop after an improvement
                        break 'seed_loop;
                    }
                }
            }

            if stop.load(AtomicOrdering::Relaxed) {
                break;
            }

            if improved_any {
                rounds_no_gain = 0;
            } else {
                rounds_no_gain += 1;
            }
        }

        let _ = ctx.shared_incumbent().try_update(&best);
    }
}
