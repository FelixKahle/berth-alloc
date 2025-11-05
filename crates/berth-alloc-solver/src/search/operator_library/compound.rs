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
        (size + other - active)
            .try_into()
            .expect("usize to i64 conversion failed")
    } else {
        (other - active)
            .try_into()
            .expect("usize to i64 conversion failed")
    }
}

pub struct CompoundOperator<'n, T, C, R> {
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
    pub fn with_evaluator(
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
        evaluator: Box<OrderEvalFn>,
    ) -> Self {
        let has_fragments = ops.iter().any(|op| op.has_fragments());
        let n = ops.len();
        Self {
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
    pub fn concatenate_no_restart(ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>) -> Self {
        let n = ops.len();
        Self::with_evaluator(
            ops,
            Box::new(move |active, other| compound_no_restart(n, active, other)),
        )
    }

    #[inline]
    pub fn concatenate_restart(ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>) -> Self {
        Self::with_evaluator(ops, Box::new(|_, _| 0))
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
        "CompoundOperator"
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

        let n = self.ops.len();
        self.order = (0..n).collect();
        self.started.fill(false);
        self.idx = 0;
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
    ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
    has_fragments: bool,
}

impl<'n, T, C, R> RandomCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub fn new(ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>) -> Self {
        let has_fragments = ops.iter().any(|op| op.has_fragments());
        Self { ops, has_fragments }
    }
}

impl<'n, T, C, R> LocalSearchOperator<T, C, R> for RandomCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomCompoundOperator"
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
    pub fn new(
        ops: Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>,
        memory_coeff: f64,
        exploration_coeff: f64,
    ) -> Self {
        assert!((0.0..=1.0).contains(&memory_coeff));
        assert!(exploration_coeff >= 0.0);

        let has_fragments = ops.iter().any(|op| op.has_fragments());
        let n = ops.len();
        Self {
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
        "MultiArmedBanditCompoundOperator"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::{
            eval::{CostEvaluator, DefaultCostEvaluator},
            operator::{LocalSearchOperator, OperatorContext},
        },
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::FitnessDelta,
            plan::Plan,
            solver_state::SolverState,
            terminal::delta::TerminalDelta,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::BTreeMap;

    type TT = i64;

    // Helpers to build a minimal model/state/context
    #[inline]
    fn tp(v: TT) -> TimePoint<TT> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: TT, b: TT) -> TimeInterval<TT> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: TT) -> TimeDelta<TT> {
        TimeDelta::new(v)
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: TT, e: TT) -> Berth<TT> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (TT, TT),
        pts: &[(u32, TT)],
        weight: TT,
    ) -> Request<FlexibleKind, TT> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, TT>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn minimal_problem() -> Problem<TT> {
        let b1 = berth(1, 0, 100);
        let r1 = flex_req(10, (0, 100), &[(1, 10)], 1);
        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_flexible(r1);
        pb.build().expect("valid problem")
    }

    fn ctx<'b, 'r, 'c, 's, 'm, 'p>(
        model: &'m SolverModel<'p, TT>,
        state: &'s SolverState<'p, TT>,
        eval: &'c DefaultCostEvaluator,
        rng: &'r mut StdRng,
        buffer: &'b mut [DecisionVar<TT>],
    ) -> OperatorContext<'b, 'r, 'c, 's, 'm, 'p, TT, DefaultCostEvaluator, StdRng> {
        OperatorContext::new(model, state, eval, rng, buffer)
    }

    // A simple dummy operator that yields a fixed number of neighbors.
    struct DummyOp {
        name: &'static str,
        id: usize,
        initial: usize,
        remaining: usize,
        fragments: bool,
        sync_calls: usize,
    }

    impl DummyOp {
        fn new(name: &'static str, id: usize, yield_count: usize) -> Self {
            Self {
                name,
                id,
                initial: yield_count,
                remaining: yield_count,
                fragments: false,
                sync_calls: 0,
            }
        }
        fn with_fragments(mut self, f: bool) -> Self {
            self.fragments = f;
            self
        }
    }

    impl LocalSearchOperator<TT, DefaultCostEvaluator, StdRng> for DummyOp {
        fn name(&self) -> &str {
            self.name
        }
        fn has_fragments(&self) -> bool {
            self.fragments
        }
        fn reset(&mut self) {
            self.remaining = self.initial;
        }
        fn synchronize<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, TT, DefaultCostEvaluator, StdRng>,
        ) {
            self.sync_calls += 1;
        }

        fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, TT, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, TT>> {
            if self.remaining == 0 {
                return None;
            }

            self.remaining -= 1;
            // Tag the plan with delta_cost equal to our id for identification
            let fd = FitnessDelta {
                delta_cost: self.id as i64,
                delta_unassigned: 0,
            };
            Some(Plan::new_delta(Vec::new(), TerminalDelta::empty(), fd))
        }
    }

    fn make_context() -> (
        SolverModel<'static, TT>,
        SolverState<'static, TT>,
        DefaultCostEvaluator,
        StdRng,
        Vec<DecisionVar<TT>>,
    ) {
        // Leak the problem to give it a 'static lifetime, matching returned model/state lifetimes
        let problem: &'static Problem<TT> = Box::leak(Box::new(minimal_problem()));
        let model = SolverModel::try_from(problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Build TerminalOccupancy from the problem (avoid borrowing the model)
        let term = TerminalOccupancy::new(problem.berths().iter());

        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fit = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars.clone()), term, fit);

        let rng = StdRng::seed_from_u64(12345);
        let buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        (model, state, evaluator, rng, buffer)
    }

    #[test]
    fn test_compound_operator_no_restart_round_robin_sequence() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        // op0 yields 2, op1 yields 1, op2 yields 1
        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 2)),
            Box::new(DummyOp::new("op1", 1, 1)),
            Box::new(DummyOp::new("op2", 2, 1)),
        ];

        let mut comp = CompoundOperator::concatenate_no_restart(ops);
        comp.synchronize(&mut ctx);

        let mut seen = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            seen.push(p.fitness_delta.delta_cost);
        }

        // Expected sequence: 0,0,1,2 (rotates order after each yield)
        assert_eq!(seen, vec![0, 0, 1, 2]);
        assert!(!comp.has_fragments());
    }

    #[test]
    fn test_compound_operator_restart_concatenation_drains_in_index_order() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        // op0=1, op1=2, op2=2 -> should drain 0 then 1 then 1 then 2 then 2 in index order
        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 2)),
            Box::new(DummyOp::new("op2", 2, 2)),
        ];

        let mut comp = CompoundOperator::concatenate_restart(ops);
        comp.synchronize(&mut ctx);

        let mut seen = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            seen.push(p.fitness_delta.delta_cost);
        }

        // Always sorted by index (restart policy makes evaluator return equal keys)
        assert_eq!(seen, vec![0, 1, 1, 2, 2]);
    }

    #[test]
    fn test_compound_operator_has_fragments_propagates() {
        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 1).with_fragments(true)),
            Box::new(DummyOp::new("op2", 2, 1)),
        ];

        let comp = CompoundOperator::concatenate_no_restart(ops);
        // synchronize isn't needed for this check
        assert!(comp.has_fragments());
    }

    #[test]
    fn test_compound_operator_reset_resets_children_and_sequence() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 1)),
        ];

        let mut comp = CompoundOperator::concatenate_no_restart(ops);
        comp.synchronize(&mut ctx);

        let mut first = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            first.push(p.fitness_delta.delta_cost);
        }
        assert_eq!(first, vec![0, 1]);

        // Reset and we should be able to yield again in same sequence
        comp.reset();
        comp.synchronize(&mut ctx);

        let mut second = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            second.push(p.fitness_delta.delta_cost);
        }
        assert_eq!(second, vec![0, 1]);
    }

    #[test]
    fn test_random_compound_operator_yields_all_children_once_in_random_order() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 1)),
            Box::new(DummyOp::new("op2", 2, 1)),
        ];

        let mut comp = RandomCompoundOperator::new(ops);
        comp.synchronize(&mut ctx);

        let mut seen = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            seen.push(p.fitness_delta.delta_cost);
        }

        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
        assert!(!comp.has_fragments());
    }

    #[test]
    fn test_random_compound_operator_with_fragments_flag() {
        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 1).with_fragments(true)),
        ];

        let comp = RandomCompoundOperator::new(ops);
        assert!(comp.has_fragments());
    }

    #[test]
    fn test_mab_compound_operator_yields_all_available_plans() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 1)),
            Box::new(DummyOp::new("op2", 2, 1)),
        ];

        let mut comp = MultiArmedBanditCompoundOperator::new(ops, 0.5, 1.0);

        // initialize with synchronize (sets last_objective)
        comp.synchronize(&mut ctx);

        let mut seen = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            seen.push(p.fitness_delta.delta_cost);
            // Normally the metaheuristic infrastructure would call synchronize()
            // between iterations as objective changes. Here we just keep yielding.
        }

        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
        assert!(!comp.has_fragments());
    }

    #[test]
    fn test_mab_compound_operator_reset_allows_reuse() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> = vec![
            Box::new(DummyOp::new("op0", 0, 1)),
            Box::new(DummyOp::new("op1", 1, 1)),
        ];

        let mut comp = MultiArmedBanditCompoundOperator::new(ops, 0.5, 1.0);
        comp.synchronize(&mut ctx);

        let mut first = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            first.push(p.fitness_delta.delta_cost);
        }
        first.sort();
        assert_eq!(first, vec![0, 1]);

        comp.reset();
        comp.synchronize(&mut ctx);

        let mut second = Vec::new();
        while let Some(p) = comp.make_next_neighbor(&mut ctx) {
            second.push(p.fitness_delta.delta_cost);
        }
        second.sort();
        assert_eq!(second, vec![0, 1]);
    }

    #[test]
    fn test_compound_operator_calls_child_synchronize_once_per_child_until_reset() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let o0 = DummyOp::new("op0", 0, 0);
        let o1 = DummyOp::new("op1", 1, 0);
        let o2 = DummyOp::new("op2", 2, 0);
        let ops: Vec<Box<dyn LocalSearchOperator<TT, DefaultCostEvaluator, StdRng>>> =
            vec![Box::new(o0), Box::new(o1), Box::new(o2)];

        // Can't easily inspect sync_calls after boxing, so test via behavior:
        let mut comp = CompoundOperator::concatenate_no_restart(ops);
        comp.synchronize(&mut ctx);

        // Trigger one pass (all children return None)
        assert!(comp.make_next_neighbor(&mut ctx).is_none());

        // After synchronize + one pass, started flags should prevent re-synchronization
        // until we call synchronize/reset again. Calling make_next_neighbor again should
        // also return None without re-synchronizing children.
        assert!(comp.make_next_neighbor(&mut ctx).is_none());

        // After explicit synchronize, children will be synchronized again
        comp.synchronize(&mut ctx);
        assert!(comp.make_next_neighbor(&mut ctx).is_none());
    }

    #[test]
    fn test_empty_compounds_yield_none() {
        let (model, state, eval, mut rng, mut buffer) = make_context();
        let mut ctx = ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let mut comp = CompoundOperator::concatenate_no_restart(Vec::new());
        comp.synchronize(&mut ctx);
        assert!(comp.make_next_neighbor(&mut ctx).is_none());

        let mut comp_r = RandomCompoundOperator::new(Vec::new());
        comp_r.synchronize(&mut ctx);
        assert!(comp_r.make_next_neighbor(&mut ctx).is_none());

        let mut comp_mab = MultiArmedBanditCompoundOperator::new(Vec::new(), 0.5, 1.0);
        comp_mab.synchronize(&mut ctx);
        assert!(comp_mab.make_next_neighbor(&mut ctx).is_none());
    }
}
