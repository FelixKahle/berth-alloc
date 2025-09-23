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

use crate::matheuristic::support::acceptance::ewma;
use berth_alloc_core::prelude::Cost;
use num_traits::ToPrimitive;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OperatorStats {
    pub attempts: u64,
    pub accepted: u64,
    pub ewma_reward: f64,
    pub total_improvement: Cost,
    pub emwa_gen_ns_per_proposal: f64,
    pub emwa_eval_ns_per_proposal: f64,
}

impl OperatorStats {
    #[inline]
    pub fn on_accept(&mut self, delta: Cost, reward_alpha: f64) {
        let r = (-delta.to_f64().unwrap_or(0.0)).max(0.0);
        self.ewma_reward = ewma(self.ewma_reward, r, reward_alpha);
        if r > 0.0 {
            self.accepted = self.accepted.saturating_add(1);
            self.total_improvement += delta;
        }
    }
    #[inline]
    pub fn on_attempt(&mut self) {
        self.attempts = self.attempts.saturating_add(1);
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
    pub operator: Box<dyn crate::matheuristic::operator::Operator<Time = T>>,
    pub stats: OperatorStats,
}

impl<T: Copy + Ord> OperatorRecord<T> {
    pub fn new(operator: Box<dyn crate::matheuristic::operator::Operator<Time = T>>) -> Self {
        Self {
            operator,
            stats: OperatorStats::default(),
        }
    }
    #[inline]
    pub fn operator(&self) -> &dyn crate::matheuristic::operator::Operator<Time = T> {
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
    pub fn new(records: Vec<OperatorRecord<T>>) -> Self {
        Self { records }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    #[inline]
    pub fn get(&self, i: usize) -> &OperatorRecord<T> {
        &self.records[i]
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize) -> &mut OperatorRecord<T> {
        &mut self.records[i]
    }

    #[inline]
    pub fn records(&self) -> &[OperatorRecord<T>] {
        &self.records
    }

    #[inline]
    pub fn reset_stats(&mut self) {
        for r in &mut self.records {
            r.stats = OperatorStats::default();
        }
    }

    pub fn stats_slice(&self) -> impl Iterator<Item = &OperatorStats> {
        self.records.iter().map(|r| &r.stats)
    }
}
