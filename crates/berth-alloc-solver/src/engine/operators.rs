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

use crate::search::operator::traits::NeighborhoodOperator;
use berth_alloc_core::{math::emwa::Ewma, prelude::Cost};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone, PartialEq)]
pub struct NeighborhoodOperatorStats {
    pub attempt_count: u64,
    pub success_count: u64,
    pub accept_count: u64,
    pub execution_time_ns: Ewma<f64, f64>,
    pub cumulative_cost_delta: Cost, // negative if total improving
}

#[derive(Debug)]
pub struct OperatorRecord<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    pub operator: Box<dyn NeighborhoodOperator<T>>,
    pub stats: NeighborhoodOperatorStats,
}

#[derive(Debug)]
pub struct OperatorPool<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    operators: Vec<OperatorRecord<T>>,
}

impl<T> Default for OperatorPool<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OperatorPool<T>
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

    pub fn get_operators(&self) -> &Vec<OperatorRecord<T>> {
        &self.operators
    }

    pub fn get_operators_mut(&mut self) -> &mut Vec<OperatorRecord<T>> {
        &mut self.operators
    }
}
