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

use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::{RngCore, SeedableRng};
use std::time::{Duration, Instant};

use crate::{
    engine::context::SearchContext,
    eval::objective::Objective,
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    scheduling::traits::Scheduler,
    search::operator::runner::CandidateEvaluator,
    state::{
        chain_set::{
            index::{ChainIndex, NodeIndex},
            view::{ChainRef, ChainSetView},
        },
        search_state::SearchSnapshot,
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use rand_chacha::ChaCha8Rng;

/// SA parameters (time-capped).
#[derive(Debug, Clone)]
pub struct SAParams {
    pub time_limit: Duration,
    pub t_start: f64,
    pub t_end: f64,
    pub seed: u64,
    pub randomize_ops: f32,
}

impl Default for SAParams {
    fn default() -> Self {
        Self {
            time_limit: Duration::from_millis(1000),
            t_start: 1e2,
            t_end: 1e-3,
            seed: 0xC0FF_EE00_D15EA5ED,
            randomize_ops: 0.5,
        }
    }
}

pub struct Search<'engine, 'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: Scheduler<T>,
{
    context: SearchContext<'engine, 'model, 'problem, T, S>,
    best_snapshot: SearchSnapshot<'model, 'problem, T>,
}

impl<'engine, 'model, 'problem, T, S> Search<'engine, 'model, 'problem, T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Send + Sync + Into<Cost>,
    S: Scheduler<T>,
{
    pub fn new(context: SearchContext<'engine, 'model, 'problem, T, S>) -> Self {
        let best_snapshot = context.state().snapshot();
        Self {
            context,
            best_snapshot,
        }
    }

    pub fn run_sa_time_cap(mut self, params: SAParams) -> SearchSnapshot<'model, 'problem, T> {
        let start = Instant::now();
        let deadline = start + params.time_limit;

        let mut rng = chacha_from_u64(params.seed);

        let op_count = self.context.operators().get_operators().len();
        if op_count == 0 {
            return self.best_snapshot;
        }
        let mut rr = 0usize;
        let mut evaluator = CandidateEvaluator::<T>::new(self.context.state());

        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let frac = (now - start).as_secs_f64() / params.time_limit.as_secs_f64();
            let temp = temperature(params.t_start, params.t_end, frac);

            let op_idx =
                if params.randomize_ops > 0.0 && rng_f64(&mut rng) < params.randomize_ops as f64 {
                    rng_range_usize(&mut rng, op_count)
                } else {
                    let i = rr;
                    rr = (rr + 1) % op_count;
                    i
                };

            // Stats: attempt
            {
                let pool = self.context.operators_mut();
                pool.record_attempt(op_idx);
            }

            let op_exec_start = Instant::now();

            let delta_opt = {
                // Build a global ArcEvaluator that resolves the chain per (from,to)
                // and delegates to SearchContext::make_search_arc_eval(...)
                let arc_eval = self.make_global_arc_evaluator();
                let op = &self.context.operators().get_operators()[op_idx].operator;
                op.make_neighboor(self.context.state(), &arc_eval)
            };

            if delta_opt.is_some() {
                self.context.operators_mut().record_success(op_idx);
            } else {
                // still record execution time for the attempt
                let elapsed_ns = op_exec_start.elapsed().as_nanos() as f64;
                self.context
                    .operators_mut()
                    .record_exec_time_ns(op_idx, elapsed_ns);
                continue;
            }

            let delta = delta_opt.unwrap();
            if let Some(cand) = evaluator.evaluate(&self.context, self.context.state(), delta) {
                let d_search = cost_to_f64(cand.search_delta_cost);
                let accept =
                    d_search <= 0.0 || (temp > 0.0 && rng_f64(&mut rng) < (-d_search / temp).exp());
                let true_delta = cand.true_delta_cost;

                if accept {
                    //println!("Accept");
                    self.context
                        .operators_mut()
                        .record_accept(op_idx, true_delta);
                    self.context.accept_candidate(cand);

                    if self.context.state().current_true_cost() < self.best_snapshot.true_cost {
                        self.best_snapshot = self.context.state().snapshot();
                    }
                }
            }

            let elapsed_ns = op_exec_start.elapsed().as_nanos() as f64;
            self.context
                .operators_mut()
                .record_exec_time_ns(op_idx, elapsed_ns);
        }

        self.best_snapshot
    }

    #[inline]
    pub fn context(&self) -> &SearchContext<'engine, 'model, 'problem, T, S> {
        &self.context
    }

    #[inline]
    pub fn into_context(self) -> SearchContext<'engine, 'model, 'problem, T, S> {
        self.context
    }

    fn make_global_arc_evaluator(&self) -> impl Fn(NodeIndex, NodeIndex) -> Option<Cost> + '_ {
        move |u: NodeIndex, v: NodeIndex| {
            let cs = self.context.state().chain_set();
            let model = self.context.model();
            let search_obj = self.context.search_objective();

            // pick chain: prefer u's chain, else v's, else match sentinels
            let chain_of = |n: NodeIndex| -> Option<ChainIndex> {
                if let Some(ci) = cs.chain_of_node(n) {
                    return Some(ci);
                }
                for i in 0..cs.num_chains() {
                    let ci = ChainIndex(i);
                    if cs.start_of_chain(ci) == n || cs.end_of_chain(ci) == n {
                        return Some(ci);
                    }
                }
                None
            };

            let ci = chain_of(u).or_else(|| chain_of(v))?;
            let cr = ChainRef::new(cs, ci);
            let bi = BerthIndex(cr.chain_index().get());

            // If v is a sentinel, no assignment cost for an arc into it.
            if cr.is_sentinel_node(v) {
                return Some(0);
            }

            // 1) compute cursor = finish time up to and including `u`
            let mut cursor: Option<_> = None;
            if u != cr.start() {
                // start from first real node
                let mut n = cr.first_real_node(cr.start())?;
                let mut t_opt: Option<_> = None;
                let mut steps_left = model.flexible_requests_len();

                loop {
                    if steps_left == 0 {
                        return None;
                    }
                    steps_left -= 1;

                    // n must be a real request node (your NodeIndex == RequestIndex)
                    let ri = RequestIndex(n.get());
                    let req_tw = model.feasible_intervals()[ri.get()];
                    let dur = match model.processing_time(ri, bi) {
                        Some(Some(d)) => d,
                        _ => return None,
                    };

                    let base = t_opt.unwrap_or(req_tw.start());
                    let start = earliest_fit_after_in_calendar(model, bi, req_tw, dur, base)?;
                    t_opt = Some(start.checked_add(dur)?);

                    if n == u {
                        break;
                    }
                    n = cr.next_real(n)?;
                }
                cursor = t_opt;
            }

            // 2) schedule only `v` using that cursor and return objective cost
            let ri_v = RequestIndex(v.get());
            let req_tw_v = model.feasible_intervals()[ri_v.get()];
            let dur_v = match model.processing_time(ri_v, bi) {
                Some(Some(d)) => d,
                _ => return None,
            };
            let base_v = cursor.unwrap_or(req_tw_v.start());
            let start_v = earliest_fit_after_in_calendar(model, bi, req_tw_v, dur_v, base_v)?;
            search_obj.assignment_cost(model, ri_v, bi, start_v)
        }
    }
}

fn earliest_fit_after_in_calendar<T>(
    model: &SolverModel<'_, T>,
    berth: BerthIndex,
    req_tw: TimeInterval<T>,
    dur: TimeDelta<T>,
    cursor: TimePoint<T>,
) -> Option<TimePoint<T>>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let cal = model.calendar_for_berth(berth)?;
    for slot in cal.free_intervals() {
        // intersection of [req.start, req.end) ∩ [slot.start, slot.end) ∩ [cursor, +∞)
        let lo = max_tp(max_tp(req_tw.start(), slot.start()), cursor);
        let hi = min_tp(req_tw.end(), slot.end());
        let latest = hi.checked_sub(dur)?;
        if lo <= latest {
            return Some(lo);
        }
    }
    None
}

#[inline]
fn max_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a >= b { a } else { b }
}

#[inline]
fn min_tp<T: Copy + Ord>(a: TimePoint<T>, b: TimePoint<T>) -> TimePoint<T> {
    if a <= b { a } else { b }
}

#[inline]
fn temperature(t0: f64, tmin: f64, frac_01: f64) -> f64 {
    if !(t0.is_finite() && tmin.is_finite() && frac_01.is_finite()) {
        return 0.0;
    }
    if t0 <= 0.0 || tmin <= 0.0 {
        return 0.0;
    }
    if frac_01 <= 0.0 {
        return t0;
    }
    if frac_01 >= 1.0 {
        return tmin;
    }
    t0 * (tmin / t0).powf(frac_01)
}

#[inline]
fn cost_to_f64(c: Cost) -> f64 {
    c as f64
}

#[inline]
fn chacha_from_u64(seed: u64) -> ChaCha8Rng {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    ChaCha8Rng::from_seed(bytes)
}

#[inline]
fn rng_f64(rng: &mut ChaCha8Rng) -> f64 {
    let v = (rng.next_u64() >> 11) as u64;
    (v as f64) / ((1u64 << 53) as f64)
}

#[inline]
fn rng_range_usize(rng: &mut ChaCha8Rng, n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (rng.next_u64() as usize) % n
    }
}
