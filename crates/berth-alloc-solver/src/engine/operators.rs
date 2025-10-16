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

use berth_alloc_core::{math::emwa::Ewma, prelude::Cost};
use num_traits::{CheckedAdd, CheckedSub};

use crate::search::operator::NeighborhoodOperator;

#[derive(Debug, Clone, PartialEq)]
pub struct NeighborhoodOperatorStats {
    pub attempt_count: u64,
    pub success_count: u64,
    pub accept_count: u64,
    pub execution_time_ns: Ewma<f64, f64>,
    pub cumulative_cost_delta: Cost, // negative if total improving
}

impl NeighborhoodOperatorStats {
    pub fn new() -> Self {
        let execution_time_ns = Ewma::from_half_life(20.0).expect("valid half life");
        Self {
            attempt_count: 0,
            success_count: 0,
            accept_count: 0,
            execution_time_ns,
            cumulative_cost_delta: 0,
        }
    }
}

pub struct OperatorRecord<'a, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    pub operator: Box<dyn NeighborhoodOperator<T> + 'a>,
    pub stats: NeighborhoodOperatorStats,
}

pub struct OperatorPool<'a, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    operators: Vec<OperatorRecord<'a, T>>,
}

impl<'a, T> OperatorPool<'a, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            operators: Vec::with_capacity(capacity),
        }
    }

    pub fn get_operators(&self) -> &Vec<OperatorRecord<'a, T>> {
        &self.operators
    }

    pub fn get_operators_mut(&mut self) -> &mut Vec<OperatorRecord<'a, T>> {
        &mut self.operators
    }

    pub fn add_operator(&mut self, op: Box<dyn NeighborhoodOperator<T> + 'a>) {
        self.operators.push(OperatorRecord {
            operator: op,
            stats: NeighborhoodOperatorStats::new(),
        });
    }

    pub fn clear(&mut self) {
        self.operators.clear();
    }

    pub fn len(&self) -> usize {
        self.operators.len()
    }

    #[inline]
    pub fn record_attempt(&mut self, idx: usize) {
        if let Some(r) = self.operators.get_mut(idx) {
            r.stats.attempt_count = r.stats.attempt_count.saturating_add(1);
        }
    }

    #[inline]
    pub fn record_success(&mut self, idx: usize) {
        if let Some(r) = self.operators.get_mut(idx) {
            r.stats.success_count = r.stats.success_count.saturating_add(1);
        }
    }

    #[inline]
    pub fn record_accept(&mut self, idx: usize, cost_delta_true: Cost) {
        if let Some(r) = self.operators.get_mut(idx) {
            r.stats.accept_count = r.stats.accept_count.saturating_add(1);
            r.stats.cumulative_cost_delta = r
                .stats
                .cumulative_cost_delta
                .saturating_add(cost_delta_true);
        }
    }

    #[inline]
    pub fn record_exec_time_ns(&mut self, idx: usize, nanos: f64) {
        if let Some(r) = self.operators.get_mut(idx) {
            r.stats.execution_time_ns.observe(nanos);
        }
    }
}
