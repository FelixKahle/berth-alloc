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
    search::{
        eval::CostEvaluator,
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::plan::Plan,
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::SliceRandom;
use std::ops::Mul;

pub type OrderEvalFn = dyn Fn(usize, usize) -> i64 + Send + Sync;

#[inline(always)]
fn compound_no_restart(size: usize, active: usize, other: usize) -> i64 {
    if other < active {
        (size + other - active) as i64
    } else {
        (other - active) as i64
    }
}

pub struct CompoundOperator<'n, T, C, R> {
    name: String,
    ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    order: Vec<usize>,
    started: Vec<bool>,
    idx: usize,
    has_fragments: bool,
    evaluator: Box<OrderEvalFn>,
    last_yielded_child: Option<usize>,
}

impl<'n, T, C, R> CompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn with_evaluator<N: Into<String>>(
        name: N,
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
        evaluator: Box<OrderEvalFn>,
    ) -> Self {
        let has_fragments = ops.iter().any(|op| op.has_fragments());
        let n = ops.len();
        Self {
            name: name.into(),
            started: vec![false; n],
            order: (0..n).collect(),
            idx: 0,
            has_fragments,
            evaluator,
            ops,
            last_yielded_child: None,
        }
    }

    #[inline]
    pub fn concatenate_no_restart<N: Into<String>>(
        name: N,
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    ) -> Self {
        let n = ops.len();
        Self::with_evaluator(
            name,
            ops,
            Box::new(move |active, other| compound_no_restart(n, active, other)),
        )
    }

    #[inline]
    pub fn concatenate_restart<N: Into<String>>(
        name: N,
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    ) -> Self {
        Self::with_evaluator(name, ops, Box::new(|_, _| 0))
    }

    #[inline]
    fn reorder_from_active(&mut self, active_child: usize) {
        let n = self.ops.len();

        if n == 0 {
            return;
        }

        let mut keys = vec![0i64; n];
        for &i in &self.order {
            keys[i] = (self.evaluator)(active_child, i);
        }

        self.order
            .sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
        self.idx = 0;
    }
}

impl<'n, T, C, R> LocalSearchOperator<T, C, R> for CompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize<'b, 'r2, 'c, 's, 'm, 'p>(
        &mut self,
        _ctx: &mut OperatorContext<'b, 'r2, 'c, 's, 'm, 'p, T, C, R>,
    ) {
        let n = self.ops.len();
        self.started.fill(false);
        if n == 0 {
            return;
        }
        let active_child = self.order[self.idx];
        self.reorder_from_active(active_child);
    }

    fn reset(&mut self) {
        for op in &mut self.ops {
            op.reset();
        }
        self.last_yielded_child = None;
    }
    fn has_fragments(&self) -> bool {
        self.has_fragments
    }

    fn make_next_neighbor<'b, 'r2, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r2, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let n = self.ops.len();
        if n == 0 {
            return None;
        }

        let start = self.idx;
        loop {
            let child_idx = self.order[self.idx];

            if !self.started[child_idx] {
                self.ops[child_idx].synchronize(ctx);
                self.started[child_idx] = true;
            }

            if let Some(p) = self.ops[child_idx].make_next_neighbor(ctx) {
                self.last_yielded_child = Some(child_idx);
                self.reorder_from_active(child_idx);
                return Some(p);
            }

            self.idx += 1;
            if self.idx == n {
                self.idx = 0;
            }
            if self.idx == start {
                break;
            }
        }
        None
    }
}

pub struct RandomCompoundOperator<'n, T, C, R> {
    name: String,
    ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    has_fragments: bool,
}

impl<'n, T, C, R> RandomCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub fn new<N: Into<String>>(
        name: N,
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    ) -> Self {
        let has_fragments = ops.iter().any(|op| op.has_fragments());
        Self {
            name: name.into(),
            ops,
            has_fragments,
        }
    }
}

impl<'n, T, C, R> LocalSearchOperator<T, C, R> for RandomCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize<'b, 'r2, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r2, 'c, 's, 'm, 'p, T, C, R>,
    ) {
        for op in &mut self.ops {
            op.synchronize(ctx);
        }
    }

    fn reset(&mut self) {
        for op in &mut self.ops {
            op.reset();
        }
    }

    fn has_fragments(&self) -> bool {
        self.has_fragments
    }

    fn make_next_neighbor<'b, 'r2, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r2, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let n = self.ops.len();

        if n == 0 {
            return None;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(ctx.rng());

        for k in indices {
            if let Some(p) = self.ops[k].make_next_neighbor(ctx) {
                return Some(p);
            }
        }
        None
    }
}

pub struct MultiArmedBanditCompoundOperator<'n, T, C, R> {
    name: String,
    ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    order: Vec<usize>,
    started: Vec<bool>,
    idx: usize,
    has_fragments: bool,

    last_objective: Cost,
    avg_improvement: Vec<f64>,
    num_neighbors: i64,
    pulls_per_op: Vec<f64>,

    memory_coeff: f64,
    exploration_coeff: f64,
    last_yielded_child: Option<usize>,
}

impl<'n, T, C, R> MultiArmedBanditCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub fn new_min<N: Into<String>>(
        name: N,
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
        memory_coeff: f64,
        exploration_coeff: f64,
    ) -> Self {
        assert!((0.0..=1.0).contains(&memory_coeff));
        assert!(exploration_coeff >= 0.0);

        let has_fragments = ops.iter().any(|op| op.has_fragments());
        let n = ops.len();
        Self {
            name: name.into(),
            started: vec![false; n],
            order: (0..n).collect(),
            idx: 0,
            has_fragments,
            last_objective: i64::MAX,
            avg_improvement: vec![0.0; n],
            num_neighbors: 0,
            pulls_per_op: vec![0.0; n],
            memory_coeff,
            exploration_coeff,
            ops,
            last_yielded_child: None,
        }
    }

    #[inline]
    fn score(&self, i: usize) -> f64 {
        // UCB1 for minimization: larger positive improvement => better.
        let pulls = self.pulls_per_op[i];
        if pulls == 0.0 {
            f64::INFINITY
        } else {
            let bonus = self.exploration_coeff
                * (2.0 * (1.0 + self.num_neighbors as f64).ln() / (1.0 + pulls)).sqrt();
            self.avg_improvement[i] + bonus
        }
    }

    fn resort_by_score(&mut self) {
        let n = self.ops.len();
        if n == 0 {
            return;
        }
        let mut scores = vec![0.0f64; n];
        for &i in &self.order {
            scores[i] = self.score(i);
        }

        self.order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        self.idx = 0;
    }
}

impl<'n, T, C, R> LocalSearchOperator<T, C, R> for MultiArmedBanditCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) {
        self.started.fill(false);

        if self.ops.is_empty() {
            return;
        }

        let objective: Cost = ctx.state().fitness().cost;

        if objective == self.last_objective {
            return;
        }
        if self.last_objective == i64::MAX {
            self.last_objective = objective;
            return;
        }

        let improvement = self.last_objective - objective;
        self.last_objective = objective;

        if improvement >= 0 {
            let active = self.order[self.idx];
            let m = self.memory_coeff;
            self.avg_improvement[active] +=
                m * ((improvement as f64) - self.avg_improvement[active]);
            self.resort_by_score();
        }
    }

    fn reset(&mut self) {
        for op in &mut self.ops {
            op.reset();
        }
        self.idx = 0;
        self.last_yielded_child = None;
        self.avg_improvement.fill(0.0);
        self.pulls_per_op.fill(0.0);
        self.num_neighbors = 0;
        self.last_objective = i64::MAX;
    }

    fn has_fragments(&self) -> bool {
        self.has_fragments
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let n = self.ops.len();
        if n == 0 {
            return None;
        }

        let start = self.idx;
        loop {
            let k = self.order[self.idx];

            if !self.started[k] {
                self.ops[k].synchronize(ctx);
                self.started[k] = true;
            }

            if let Some(p) = self.ops[k].make_next_neighbor(ctx) {
                self.last_yielded_child = Some(k);

                self.num_neighbors += 1;
                self.pulls_per_op[k] += 1.0;

                return Some(p);
            }

            self.idx += 1;
            if self.idx == n {
                self.idx = 0;
            }
            if self.idx == start {
                break;
            }
        }
        None
    }
}
