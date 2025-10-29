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

use crate::search::eval::CostEvaluator;
use crate::state::fitness::{Fitness, FitnessDelta};
use crate::state::terminal::terminalocc::TerminalWrite;
use crate::{
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    search::err::{
        BerthNotFreeError, NotAllowedOnBerthError, NotAssignedError, ProposeAssignmentError,
        ProposeUnassignmentError,
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::{DecisionVarPatch, Plan},
        terminal::{
            sandbox::TerminalSandbox,
            terminalocc::{FreeBerth, TerminalOccupancy, TerminalRead},
        },
    },
};
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Savepoint {
    undo_len: usize,
    patches_len: usize,
    saved_fitness_delta: FitnessDelta,
}

impl std::fmt::Display for Savepoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Savepoint(undo_len={}, patches_len={}, saved_fitness_delta={})",
            self.undo_len, self.patches_len, self.saved_fitness_delta
        )
    }
}

#[derive(Debug, Clone)]
enum TerminalUndoAction<T: Copy + Ord> {
    Occupy(BerthIndex, TimeInterval<T>),
    Release(BerthIndex, TimeInterval<T>),
}

impl<T: Copy + Ord + std::fmt::Display> std::fmt::Display for TerminalUndoAction<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminalUndoAction::Occupy(b, iv) => {
                write!(f, "Occupy(berth_index={}, interval={})", b, iv)
            }
            TerminalUndoAction::Release(b, iv) => {
                write!(f, "Release(berth_index={}, interval={})", b, iv)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecisionVarUndoAction<T: Copy + Ord> {
    request_index: RequestIndex,
    previous_value: DecisionVar<T>,
}

impl<T: Copy + Ord + std::fmt::Display> std::fmt::Display for DecisionVarUndoAction<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DecisionVarUndoAction(request_index={}, previous_value={})",
            self.request_index, self.previous_value
        )
    }
}

#[derive(Debug, Clone)]
enum UndoAction<T: Copy + Ord> {
    Terminal(TerminalUndoAction<T>),
    Decision(DecisionVarUndoAction<T>),
}

impl<T: Copy + Ord + std::fmt::Display> std::fmt::Display for UndoAction<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UndoAction::Terminal(action) => write!(f, "Terminal({})", action),
            UndoAction::Decision(action) => write!(f, "Decision({})", action),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanExplorer<'pb, 'c, 't, 'm, 'p, T: Copy + Ord, C: CostEvaluator<T>> {
    solver_model: &'m SolverModel<'p, T>,
    decision_vars: &'pb [DecisionVar<T>],
    cost_evaluator: &'c C,
    sandbox: &'pb TerminalSandbox<'t, 'p, T>,
}

impl<'pb, 'c, 't, 'm, 'p, T: Copy + Ord, C: CostEvaluator<T>>
    PlanExplorer<'pb, 'c, 't, 'm, 'p, T, C>
{
    #[inline]
    pub fn new(
        solver_model: &'m SolverModel<'p, T>,
        decision_vars: &'pb [DecisionVar<T>],
        cost_evaluator: &'c C,
        sandbox: &'pb TerminalSandbox<'t, 'p, T>,
    ) -> Self {
        Self {
            solver_model,
            decision_vars,
            cost_evaluator,
            sandbox,
        }
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.solver_model
    }

    #[inline]
    pub fn decision_vars(&self) -> &'pb [DecisionVar<T>] {
        self.decision_vars
    }

    #[inline]
    pub fn sandbox(&self) -> &'pb TerminalSandbox<'t, 'p, T> {
        self.sandbox
    }

    #[inline]
    pub fn iter_free_for(
        &self,
        request_index: RequestIndex,
    ) -> impl Iterator<Item = FreeBerth<T>> + 'pb
    where
        T: CheckedAdd + CheckedSub,
        'm: 'pb, // 'm lives at least as long as 'pb; the model will outlive the builder
    {
        let window = self.solver_model.feasible_interval(request_index);
        let allowed = self.solver_model.allowed_berth_indices(request_index);

        self.sandbox
            .inner()
            .iter_free_intervals_for_berths_in_slice(allowed, window)
    }

    #[inline]
    pub fn iter_unassigned(&self) -> impl Iterator<Item = RequestIndex>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.decision_vars()
            .iter()
            .enumerate()
            .filter_map(|(idx, dv)| {
                if !dv.is_assigned() {
                    Some(RequestIndex::new(idx))
                } else {
                    None
                }
            })
    }

    #[inline]
    pub fn iter_assigned_requests(&self) -> impl Iterator<Item = RequestIndex>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.decision_vars()
            .iter()
            .enumerate()
            .filter_map(|(idx, dv)| {
                if dv.is_assigned() {
                    Some(RequestIndex::new(idx))
                } else {
                    None
                }
            })
    }

    #[inline]
    pub fn iter_assignments(&self) -> impl Iterator<Item = &DecisionVar<T>> {
        self.decision_vars().iter().filter(|dv| dv.is_assigned())
    }

    #[inline]
    pub fn peek_cost(
        &self,
        request: RequestIndex,
        start_time: TimePoint<T>,
        berth_index: BerthIndex,
    ) -> Option<Cost>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        self.cost_evaluator
            .eval_request(self.model(), request, start_time, berth_index)
    }
}

#[derive(Debug)]
pub struct PlanBuilder<'b, 'c, 't, 'm, 'p, T: Copy + Ord, C: CostEvaluator<T>> {
    solver_model: &'m SolverModel<'p, T>,
    decision_vars: &'b mut [DecisionVar<T>],
    sandbox: TerminalSandbox<'t, 'p, T>,
    patches: Vec<DecisionVarPatch<T>>,
    cost_evaluator: &'c C,
    fitness_delta: FitnessDelta,
    undo: Vec<UndoAction<T>>,
}

impl<'b, 'c, 't, 'm, 'p, T: Copy + Ord, C> PlanBuilder<'b, 'c, 't, 'm, 'p, T, C>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
{
    #[inline]
    pub fn new(
        solver_model: &'m SolverModel<'p, T>,
        base_terminal: &'t TerminalOccupancy<'p, T>,
        cost_evaluator: &'c C,
        decision_vars: &'b mut [DecisionVar<T>],
    ) -> Self {
        Self {
            solver_model,
            decision_vars,
            sandbox: TerminalSandbox::new(base_terminal),
            cost_evaluator,
            fitness_delta: FitnessDelta::zero(),
            patches: Vec::with_capacity(32),
            undo: Vec::with_capacity(64),
        }
    }

    #[inline]
    pub fn delta_cost(&self) -> Cost {
        self.fitness_delta.delta_cost
    }

    #[inline]
    pub fn delta_unassigned(&self) -> i32 {
        self.fitness_delta.delta_unassigned
    }

    #[inline]
    pub fn fitness_delta(&self) -> FitnessDelta {
        self.fitness_delta
    }

    #[inline]
    pub fn sandbox(&self) -> &TerminalSandbox<'t, 'p, T> {
        &self.sandbox
    }

    #[inline]
    pub fn savepoint(&self) -> Savepoint {
        Savepoint {
            undo_len: self.undo.len(),
            patches_len: self.patches.len(),
            saved_fitness_delta: self.fitness_delta,
        }
    }

    #[inline]
    pub fn undo_to(&mut self, sp: Savepoint) {
        while self.undo.len() > sp.undo_len {
            let action = self.undo.pop().expect("undo stack underflow");
            match action {
                UndoAction::Terminal(TerminalUndoAction::Occupy(b, iv)) => {
                    self.sandbox
                        .occupy(b, iv)
                        .expect("undo occupy() must succeed");
                }
                UndoAction::Terminal(TerminalUndoAction::Release(b, iv)) => {
                    self.sandbox
                        .release(b, iv)
                        .expect("undo release() must succeed");
                }
                UndoAction::Decision(DecisionVarUndoAction {
                    request_index,
                    previous_value,
                }) => {
                    self.decision_vars[request_index.get()] = previous_value;
                }
            }
        }
        self.patches.truncate(sp.patches_len);
        self.fitness_delta = sp.saved_fitness_delta;
    }

    #[inline]
    pub fn propose_assignment(
        &mut self,
        request: RequestIndex,
        start_time: TimePoint<T>,
        free_berth: &FreeBerth<T>,
    ) -> Result<(), ProposeAssignmentError<T>>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let prev_dv = self.decision_vars[request.get()];

        if let DecisionVar::Assigned(old) = prev_dv {
            let old_iv = self
                .solver_model
                .interval(request, old.berth_index, old.start_time)
                .ok_or_else(|| {
                    ProposeAssignmentError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                        request,
                        old.berth_index,
                    ))
                })?;

            let old_cost = self
                .cost_evaluator
                .eval_request(self.solver_model, request, old.start_time, old.berth_index)
                .ok_or_else(|| {
                    ProposeAssignmentError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                        request,
                        old.berth_index,
                    ))
                })?;

            self.sandbox
                .release(old.berth_index, old_iv)
                .map_err(ProposeAssignmentError::from)?;
            self.undo
                .push(UndoAction::Terminal(TerminalUndoAction::Occupy(
                    old.berth_index,
                    old_iv,
                )));

            self.fitness_delta.delta_cost = self
                .fitness_delta
                .delta_cost
                .checked_sub(old_cost)
                .expect("Cost subtraction overflowed");
        }

        let cost = self
            .cost_evaluator
            .eval_request(
                self.solver_model,
                request,
                start_time,
                free_berth.berth_index(),
            )
            .ok_or_else(|| {
                ProposeAssignmentError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                    request,
                    free_berth.berth_index(),
                ))
            })?;

        let processing_time = self
            .solver_model
            .processing_time(request, free_berth.berth_index())
            .ok_or_else(|| {
                ProposeAssignmentError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                    request,
                    free_berth.berth_index(),
                ))
            })?;

        let end_time = start_time + processing_time;
        let asg_iv = TimeInterval::new(start_time, end_time);

        if !free_berth.interval().contains_interval(&asg_iv) {
            return Err(ProposeAssignmentError::BerthNotFree(
                BerthNotFreeError::new(free_berth.berth_index(), asg_iv, free_berth.interval()),
            ));
        }

        self.sandbox
            .occupy(free_berth.berth_index(), asg_iv)
            .map_err(ProposeAssignmentError::from)?;
        self.undo
            .push(UndoAction::Terminal(TerminalUndoAction::Release(
                free_berth.berth_index(),
                asg_iv,
            )));

        self.fitness_delta.delta_cost = self
            .fitness_delta
            .delta_cost
            .checked_add(cost)
            .expect("Cost addition overflowed");

        if !matches!(prev_dv, DecisionVar::Assigned(_)) {
            self.fitness_delta.delta_unassigned = self
                .fitness_delta
                .delta_unassigned
                .checked_sub(1)
                .expect("Unassigned requests addition overflowed");
        }

        self.undo.push(UndoAction::Decision(DecisionVarUndoAction {
            request_index: request,
            previous_value: prev_dv,
        }));

        let assigned = DecisionVar::Assigned(Decision {
            berth_index: free_berth.berth_index(),
            start_time,
        });
        self.decision_vars[request.get()] = assigned;
        self.patches.push(DecisionVarPatch::new(request, assigned));

        Ok(())
    }

    #[inline]
    pub fn propose_unassignment(
        &mut self,
        request: RequestIndex,
    ) -> Result<FreeBerth<T>, ProposeUnassignmentError<T>>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let prev_dv = self.decision_vars[request.get()];

        let dv = match prev_dv {
            DecisionVar::Unassigned => {
                return Err(ProposeUnassignmentError::NotAssigned(
                    NotAssignedError::new(request),
                ));
            }
            DecisionVar::Assigned(decision) => decision,
        };

        let iv = self
            .solver_model
            .interval(request, dv.berth_index, dv.start_time)
            .ok_or_else(|| {
                ProposeUnassignmentError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                    request,
                    dv.berth_index,
                ))
            })?;

        let cost = self
            .cost_evaluator
            .eval_request(self.solver_model, request, dv.start_time, dv.berth_index)
            .ok_or_else(|| {
                ProposeUnassignmentError::NotAllowedOnBerth(NotAllowedOnBerthError::new(
                    request,
                    dv.berth_index,
                ))
            })?;

        self.sandbox
            .release(dv.berth_index, iv)
            .map_err(ProposeUnassignmentError::from)?;
        self.undo
            .push(UndoAction::Terminal(TerminalUndoAction::Occupy(
                dv.berth_index,
                iv,
            )));

        self.fitness_delta.delta_cost = self
            .fitness_delta
            .delta_cost
            .checked_sub(cost)
            .expect("Cost subtraction overflowed");
        self.fitness_delta.delta_unassigned = self
            .fitness_delta
            .delta_unassigned
            .checked_add(1)
            .expect("Unassigned requests addition overflowed");

        self.undo.push(UndoAction::Decision(DecisionVarUndoAction {
            request_index: request,
            previous_value: prev_dv,
        }));

        self.decision_vars[request.get()] = DecisionVar::Unassigned;
        self.patches
            .push(DecisionVarPatch::new(request, DecisionVar::Unassigned));

        Ok(FreeBerth::new(iv, dv.berth_index))
    }

    #[inline]
    pub fn with_explorer<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&PlanExplorer<'_, 'c, 't, 'm, 'p, T, C>) -> R,
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let explorer: PlanExplorer<'_, 'c, 't, 'm, 'p, T, C> = PlanExplorer::new(
            self.solver_model,
            self.decision_vars,
            self.cost_evaluator,
            &self.sandbox,
        );
        f(&explorer)
    }

    #[inline(always)]
    pub fn discard(self) {
        // No-op for now, as we do not have external resources to clean up.
        // Ledger and sandbox will be dropped automatically.
    }

    #[inline]
    pub fn peek_cost(
        &self,
        request: RequestIndex,
        start_time: TimePoint<T>,
        free_berth: &FreeBerth<T>,
    ) -> Option<Cost>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        self.cost_evaluator.eval_request(
            self.solver_model,
            request,
            start_time,
            free_berth.berth_index(),
        )
    }

    #[inline]
    pub fn peek_fitness(&self) -> Option<Fitness>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let mut total_cost = Cost::zero();
        let mut unassigned = 0usize;

        for (i, dv) in self.decision_vars.iter().enumerate() {
            match *dv {
                DecisionVar::Unassigned => unassigned += 1,
                DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                }) => {
                    let c = self.cost_evaluator.eval_request(
                        self.solver_model,
                        RequestIndex::new(i),
                        start_time,
                        berth_index,
                    )?;
                    total_cost = total_cost.checked_add(c)?;
                }
            }
        }

        Some(Fitness::new(total_cost, unassigned))
    }

    #[inline]
    pub fn has_changes(&self) -> bool {
        !self.patches.is_empty()
    }

    #[inline]
    pub fn finalize(self) -> Plan<'p, T> {
        Plan::new_delta(self.patches, self.sandbox.delta(), self.fitness_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::DefaultCostEvaluator,
        state::{berth::berthocc::BerthRead, decisionvar::DecisionVar},
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
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

    fn problem_one_berth_two_flex() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn mk_occ<'b, I>(berths: I) -> TerminalOccupancy<'b, i64>
    where
        I: IntoIterator<Item = &'b Berth<i64>>,
    {
        TerminalOccupancy::new(berths)
    }

    #[test]
    fn test_plan_builder_propose_assignment_and_finalize() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        let free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .expect("free slot exists");

        pb.propose_assignment(r_ix, tp(0), &free)
            .expect("propose assignment should succeed");

        let pt = model.processing_time(r_ix, b_ix).expect("pt defined");
        let asg_iv = iv(0, pt.value());
        let occ = pb.sandbox().inner().berth(b_ix).expect("berth exists");
        assert!(!occ.is_free(asg_iv));

        assert!(pb.delta_cost() > 0);
        assert_eq!(pb.delta_unassigned(), -1);

        let plan = pb.finalize();
        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -1);
        assert!(!plan.terminal_delta.is_empty());
        assert_eq!(plan.decision_var_patches.len(), 1);
    }

    #[test]
    fn test_plan_builder_propose_unassignment_roundtrip() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        // NEW: working buffer
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        let free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();

        pb.propose_assignment(r_ix, tp(0), &free)
            .expect("assign ok");

        let fb = pb.propose_unassignment(r_ix).expect("unassign ok");

        let pt = model.processing_time(r_ix, b_ix).unwrap();
        let expected = iv(0, pt.value());
        assert_eq!(fb.interval(), expected);
        assert_eq!(fb.berth_index(), b_ix);

        assert_eq!(pb.delta_unassigned(), 0);
        assert_eq!(pb.delta_cost(), 0);
    }

    #[test]
    fn test_plan_explorer_iterators_reflect_builder_state() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_unassigned().count(), model.flexible_requests_len());
            assert_eq!(ex.iter_assigned_requests().count(), 0);

            let r_ix = model.index_manager().request_index(rid(1)).unwrap();
            assert!(ex.iter_free_for(r_ix).next().is_some());
        });

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();
        pb.propose_assignment(r_ix, tp(0), &free)
            .expect("assign ok");

        pb.with_explorer(|ex| {
            assert_eq!(
                ex.iter_unassigned().count(),
                model.flexible_requests_len() - 1
            );
            assert_eq!(ex.iter_assigned_requests().count(), 1);
            assert_eq!(ex.iter_assignments().count(), 1);
        });
    }

    #[test]
    fn test_propose_assignment_not_free_error() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        let narrow = FreeBerth::new(iv(5, 7), b_ix);

        let err = pb.propose_assignment(r_ix, tp(0), &narrow).unwrap_err();
        match err {
            ProposeAssignmentError::BerthNotFree(_) => {}
            other => panic!("expected BerthNotFree, got: {other:?}"),
        }

        assert_eq!(pb.delta_unassigned(), 0);
        assert_eq!(pb.delta_cost(), 0);
    }

    #[test]
    fn test_eval_fitness_all_unassigned() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let vars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fit = DefaultCostEvaluator.eval_fitness(&model, &vars);

        assert_eq!(fit.unassigned_requests, model.flexible_requests_len());
        assert_eq!(fit.cost, 0);
    }

    #[test]
    fn test_eval_fitness_one_assigned_one_unassigned() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        let mut vars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        vars[r1.get()] = DecisionVar::assigned(b1, tp(0));

        let expected_cost = model
            .cost_of_assignment(r1, b1, tp(0))
            .expect("valid assignment must have a cost");

        let fit = DefaultCostEvaluator.eval_fitness(&model, &vars);

        assert_eq!(fit.unassigned_requests, model.flexible_requests_len() - 1);
        assert_eq!(fit.cost, expected_cost);
    }

    #[test]
    fn test_eval_fitness_all_assigned_sums_costs() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let r2 = model.index_manager().request_index(rid(2)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // Both requests are feasible on berth 1 starting at time 0 in this problem setup
        let mut vars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        vars[r1.get()] = DecisionVar::assigned(b1, tp(0));
        vars[r2.get()] = DecisionVar::assigned(b1, tp(0));

        let expected_cost = model.cost_of_assignment(r1, b1, tp(0)).unwrap()
            + model.cost_of_assignment(r2, b1, tp(0)).unwrap();

        let fit = DefaultCostEvaluator.eval_fitness(&model, &vars);

        assert_eq!(fit.unassigned_requests, 0);
        assert_eq!(fit.cost, expected_cost);
    }

    #[test]
    fn test_savepoint_and_undo_assignment() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let sp0 = pb.savepoint();

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        let free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();

        pb.propose_assignment(r_ix, tp(0), &free).unwrap();

        let pt = model.processing_time(r_ix, b_ix).unwrap();
        let asg_iv = iv(0, pt.value());
        let occ = pb.sandbox().inner().berth(b_ix).unwrap();
        assert!(!occ.is_free(asg_iv));

        // Undo back to sp0: everything should be reset
        pb.undo_to(sp0);

        let occ2 = pb.sandbox().inner().berth(b_ix).unwrap();
        assert!(occ2.is_free(asg_iv));
        assert_eq!(pb.delta_cost(), 0);
        assert_eq!(pb.delta_unassigned(), 0);

        // With explorer: both requests unassigned
        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_assigned_requests().count(), 0);
            assert_eq!(ex.iter_unassigned().count(), model.flexible_requests_len());
        });
    }

    #[test]
    fn test_savepoint_and_undo_unassignment() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        let free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();

        pb.propose_assignment(r_ix, tp(0), &free).unwrap();

        let sp_after_assign = pb.savepoint();

        // Unassign
        let fb = pb.propose_unassignment(r_ix).unwrap();

        // After unassignment the interval is free
        let occ = pb.sandbox().inner().berth(b_ix).unwrap();
        assert!(occ.is_free(fb.interval()));

        // Undo to sp_after_assign -> should be assigned again
        pb.undo_to(sp_after_assign);

        let pt = model.processing_time(r_ix, b_ix).unwrap();
        let asg_iv = iv(0, pt.value());
        let occ2 = pb.sandbox().inner().berth(b_ix).unwrap();
        assert!(!occ2.is_free(asg_iv));

        // Deltas restored to after-assignment snapshot
        assert!(pb.delta_cost() > 0);
        assert_eq!(pb.delta_unassigned(), -1);

        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_assigned_requests().count(), 1);
        });
    }

    #[test]
    fn test_savepoint_and_undo_reassignment() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        // First assignment at t=0
        let free_initial = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();
        pb.propose_assignment(r_ix, tp(0), &free_initial).unwrap();

        let pt = model.processing_time(r_ix, b_ix).unwrap();
        let asg_iv_old = iv(0, pt.value());

        let sp_after_first = pb.savepoint();

        // Pick a free interval that is outside the old assignment to reassign
        let free_new = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .find(|fb| fb.interval().start() >= asg_iv_old.end())
            .expect("should have free interval after old assignment");

        let new_start = free_new.interval().start();
        pb.propose_assignment(r_ix, new_start, &free_new)
            .expect("reassign ok");

        // Now undo back to after-first-assignment
        pb.undo_to(sp_after_first);

        // Should be assigned back at t=0
        let occ = pb.sandbox().inner().berth(b_ix).unwrap();
        assert!(!occ.is_free(asg_iv_old));
        assert!(pb.delta_cost() > 0);
        assert_eq!(pb.delta_unassigned(), -1);
    }

    // NEW Comprehensive tests

    #[test]
    fn test_savepoint_multi_level_interleaved() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let r2 = model.index_manager().request_index(rid(2)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // Assign r1 at t=0
        let free1 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();
        pb.propose_assignment(r1, tp(0), &free1).unwrap();
        let pt1 = model.processing_time(r1, b1).unwrap();
        let r1_iv_0 = iv(0, pt1.value());
        assert!(!pb.sandbox().inner().berth(b1).unwrap().is_free(r1_iv_0));
        let sp1 = pb.savepoint();

        // Assign r2 at earliest available time (not necessarily t=0)
        let free2 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r2))
            .next()
            .unwrap();
        let start2 = free2.interval().start();
        pb.propose_assignment(r2, start2, &free2).unwrap();
        let pt2 = model.processing_time(r2, b1).unwrap();
        let end2 = start2 + pt2;
        let r2_iv = TimeInterval::new(start2, end2);

        assert!(!pb.sandbox().inner().berth(b1).unwrap().is_free(r2_iv));
        let sp2 = pb.savepoint();

        // Reassign r2 to a later time
        let later_free_for_r2 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r2))
            .find(|fb| fb.interval().start() >= r2_iv.end())
            .expect("later free exists");
        let later_start = later_free_for_r2.interval().start();
        pb.propose_assignment(r2, later_start, &later_free_for_r2)
            .unwrap();

        // Unassign r1
        let _fb = pb.propose_unassignment(r1).unwrap();

        // Undo to sp2 -> r1 assigned at 0 and r2 assigned back at start2
        pb.undo_to(sp2);
        let occ = pb.sandbox().inner().berth(b1).unwrap();
        assert!(!occ.is_free(r1_iv_0));
        assert!(!occ.is_free(r2_iv));

        // Undo to sp1 -> only r1 assigned at 0 and r2 unassigned
        pb.undo_to(sp1);
        let occ2 = pb.sandbox().inner().berth(b1).unwrap();
        assert!(!occ2.is_free(r1_iv_0));
        assert!(occ2.is_free(r2_iv));

        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_assigned_requests().count(), 1);
            assert_eq!(
                ex.iter_unassigned().count(),
                model.flexible_requests_len() - 1
            );
        });
    }

    #[test]
    fn test_undo_noop_behavior() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let sp0 = pb.savepoint();
        // Undo without changes -> no-op
        pb.undo_to(sp0);
        assert_eq!(pb.delta_cost(), 0);
        assert_eq!(pb.delta_unassigned(), 0);
        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_assigned_requests().count(), 0);
        });

        // Make a change then undo to sp0
        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();
        let free1 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();
        pb.propose_assignment(r1, tp(0), &free1).unwrap();
        assert!(pb.delta_cost() > 0);
        assert_eq!(pb.delta_unassigned(), -1);

        pb.undo_to(sp0);
        assert_eq!(pb.delta_cost(), 0);
        assert_eq!(pb.delta_unassigned(), 0);
        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_assigned_requests().count(), 0);
            assert_eq!(ex.iter_unassigned().count(), model.flexible_requests_len());
        });

        // Undo again to same sp0 -> still no-op
        pb.undo_to(sp0);
        assert_eq!(pb.delta_cost(), 0);
        assert_eq!(pb.delta_unassigned(), 0);
    }

    #[test]
    fn test_undo_safe_after_failed_propose() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let sp0 = pb.savepoint();

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // Deliberately choose a too-narrow free interval
        let bad = FreeBerth::new(iv(5, 6), b1);
        let err = pb.propose_assignment(r1, tp(0), &bad).unwrap_err();
        match err {
            ProposeAssignmentError::BerthNotFree(_) => {}
            other => panic!("expected BerthNotFree, got: {other:?}"),
        }

        // State must be unchanged and undo_to must be safe
        assert_eq!(pb.delta_cost(), 0);
        assert_eq!(pb.delta_unassigned(), 0);
        pb.undo_to(sp0);
        assert_eq!(pb.delta_cost(), 0);
        assert_eq!(pb.delta_unassigned(), 0);
    }

    #[test]
    fn test_finalize_after_undo_reflects_current_state() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).unwrap();
        let term = mk_occ(prob.berths().iter());
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let r2 = model.index_manager().request_index(rid(2)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // Assign r1 at t=0
        let free1 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();
        pb.propose_assignment(r1, tp(0), &free1).unwrap();
        let sp1 = pb.savepoint();

        // Assign r2 at earliest available time (not necessarily t=0)
        let free2 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r2))
            .next()
            .unwrap();
        let start2 = free2.interval().start();
        pb.propose_assignment(r2, start2, &free2).unwrap();

        // Now unassign r1
        let _fb = pb.propose_unassignment(r1).unwrap();

        // Undo to sp1 -> only r1 is assigned
        pb.undo_to(sp1);
        pb.with_explorer(|ex| {
            assert_eq!(ex.iter_assigned_requests().count(), 1);
        });

        // Finalize and check deltas > 0 and unassigned decreased by 1
        let delta_cost_now = pb.delta_cost();
        let delta_unassigned_now = pb.delta_unassigned();
        assert!(delta_cost_now > 0);
        assert_eq!(delta_unassigned_now, -1);

        let plan = pb.finalize();
        assert_eq!(plan.fitness_delta.delta_cost, delta_cost_now);
        assert_eq!(plan.fitness_delta.delta_unassigned, delta_unassigned_now);
        assert_eq!(plan.decision_var_patches.len(), 1);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_explorer_peek_cost_matches_model() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        pb.with_explorer(|ex| {
            let start = tp(0);
            let peek = ex.peek_cost(r1, start, b1).expect("peek must have cost");
            let expected = model.cost_of_assignment(r1, b1, start).expect("model cost");
            assert_eq!(peek, expected);
        });
    }

    #[test]
    fn test_peek_fitness_some_matches_model() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // Assign r1 at t=0
        let free1 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();
        pb.propose_assignment(r1, tp(0), &free1).unwrap();

        // peek_fitness should be Some and match model’s cost
        let fit = pb.peek_fitness().expect("peek_fitness should return Some");
        let expected_cost = model
            .cost_of_assignment(r1, b1, tp(0))
            .expect("valid assignment must have a cost");

        assert_eq!(fit.cost, expected_cost);
        assert_eq!(fit.unassigned_requests, model.flexible_requests_len() - 1);
    }

    #[test]
    fn test_propose_unassignment_not_assigned_error() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let err = pb.propose_unassignment(r1).unwrap_err();
        match err {
            ProposeUnassignmentError::NotAssigned(_) => {}
            other => panic!("expected NotAssigned, got: {other:?}"),
        }
        assert_eq!(pb.delta_unassigned(), 0);
        assert_eq!(pb.delta_cost(), 0);
    }

    fn problem_two_berths_flex_only_on_first() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 100)); // allowed
        berths.insert(berth(2, 0, 100)); // not allowed

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // Only berth 1 allowed
        flex.insert(flex_req(1, (0, 100), &[(1, 10)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    #[test]
    fn test_propose_assignment_not_allowed_on_berth_error() {
        let prob = problem_two_berths_flex_only_on_first();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b2 = model.index_manager().berth_index(bid(2)).unwrap();

        // Craft a free berth interval on b2 (exists), but the request isn't allowed there
        let free_b2 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b2], model.feasible_interval(r1))
            .next()
            .expect("free window on berth 2 exists");

        let err = pb
            .propose_assignment(r1, free_b2.interval().start(), &free_b2)
            .unwrap_err();
        match err {
            ProposeAssignmentError::NotAllowedOnBerth(e) => {
                assert_eq!(e.request_index(), r1);
                assert_eq!(e.berth_index(), b2);
            }
            other => panic!("expected NotAllowedOnBerth, got: {other:?}"),
        }
        assert_eq!(pb.delta_unassigned(), 0);
        assert_eq!(pb.delta_cost(), 0);
    }

    #[test]
    fn test_has_changes_toggles_after_propose() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        assert!(!pb.has_changes(), "initially no changes");

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();
        let free1 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();
        pb.propose_assignment(r1, tp(0), &free1).unwrap();

        assert!(pb.has_changes(), "after propose, changes should be present");
    }

    #[test]
    fn test_reassign_keeps_delta_unassigned_constant() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // First assignment at t=0
        let free1 = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .next()
            .unwrap();
        pb.propose_assignment(r1, tp(0), &free1).unwrap();
        let delta_unassigned_after_first = pb.delta_unassigned();
        assert_eq!(delta_unassigned_after_first, -1);

        // Compute end of the assigned interval
        let pt1 = model.processing_time(r1, b1).unwrap();
        let assigned_end = tp(0) + pt1;

        // Reassign to a later free slot (start >= end of assigned interval)
        let later_free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b1], model.feasible_interval(r1))
            .find(|fb| fb.interval().start() >= assigned_end)
            .expect("later free exists");
        pb.propose_assignment(r1, later_free.interval().start(), &later_free)
            .unwrap();

        // Still one net assignment (unassigned delta unchanged)
        assert_eq!(pb.delta_unassigned(), -1);
        assert!(
            pb.delta_cost() != 0.into(),
            "cost should reflect reassignment"
        );
    }

    #[test]
    fn test_finalize_no_changes_yields_empty_plan() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        let plan = pb.finalize();
        assert_eq!(plan.fitness_delta.delta_cost, 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, 0);
        assert!(plan.decision_var_patches.is_empty());
        assert!(plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_discard_noop() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let pb = PlanBuilder::new(
            &model,
            &term,
            &DefaultCostEvaluator,
            work_buf.as_mut_slice(),
        );

        // Just ensure it compiles and is a no-op
        pb.discard();
    }
}
