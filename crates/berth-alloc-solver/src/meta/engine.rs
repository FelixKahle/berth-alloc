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

use crate::{framework::planning::Plan, meta::operator::Operator};
use berth_alloc_core::prelude::Cost;
use num_traits::{ToPrimitive, Zero};

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

/// Small wrapper to encapsulate operator records & stats plumbing.
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
