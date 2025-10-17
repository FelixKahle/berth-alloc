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
    engine::search::SearchStrategy,
    search::operator::{DestroyOperator, LocalMoveOperator, RepairOperator},
};

pub struct IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    destroy_ops: Vec<Box<dyn DestroyOperator<T, R>>>,
    repair_ops: Vec<Box<dyn RepairOperator<T, R>>>,
    local_ops: Vec<Box<dyn LocalMoveOperator<T, R>>>,
    max_local_steps: usize,
}

impl<T, R> Default for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    pub fn new() -> Self {
        Self {
            destroy_ops: Vec::new(),
            repair_ops: Vec::new(),
            local_ops: Vec::new(),
            max_local_steps: 64,
        }
    }

    pub fn with_destroy_op(mut self, op: Box<dyn DestroyOperator<T, R>>) -> Self {
        self.destroy_ops.push(op);
        self
    }
    pub fn with_repair_op(mut self, op: Box<dyn RepairOperator<T, R>>) -> Self {
        self.repair_ops.push(op);
        self
    }
    pub fn with_local_op(mut self, op: Box<dyn LocalMoveOperator<T, R>>) -> Self {
        self.local_ops.push(op);
        self
    }
    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps;
        self
    }
}

impl<T, R> SearchStrategy<T, R> for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "Iterated Local Search"
    }

    #[tracing::instrument(name = "ILS Search", skip(self, context))]
    fn run<'e, 'p>(&mut self, context: &super::search::SearchContext<'e, 'p, T, R>) {
        todo!()
    }
}
