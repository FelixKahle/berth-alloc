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

use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

use crate::state::solver_state::SolverStateView;
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
        solver_state::SolverState,
        terminal::{
            sandbox::TerminalSandbox,
            terminalocc::{FreeBerth, TerminalOccupancy, TerminalRead},
        },
    },
};

#[derive(Debug, Clone)]
pub struct PlanExplorer<'pb, 't, 'm, 'p, T: Copy + Ord> {
    solver_model: &'m SolverModel<'p, T>,
    decision_vars: &'pb [DecisionVar<T>],
    sandbox: &'pb TerminalSandbox<'t, 'p, T>,
}

impl<'pb, 't, 'm, 'p, T: Copy + Ord> PlanExplorer<'pb, 't, 'm, 'p, T> {
    #[inline]
    pub fn new(
        solver_model: &'m SolverModel<'p, T>,
        decision_vars: &'pb [DecisionVar<T>],
        sandbox: &'pb TerminalSandbox<'t, 'p, T>,
    ) -> Self {
        Self {
            solver_model,
            decision_vars,
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
        'm: 'pb, // m lives at least as long as pb; the model will outlive the builder
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
        self.solver_model
            .cost_of_assignment(request, berth_index, start_time)
    }
}

#[derive(Debug)]
pub struct PlanBuilder<'b, 't, 'm, 'p, T: Copy + Ord> {
    solver_model: &'m SolverModel<'p, T>,
    decision_vars: &'b mut [DecisionVar<T>],
    sandbox: TerminalSandbox<'t, 'p, T>,
    patches: Vec<DecisionVarPatch<T>>,
    delta_cost: Cost,
    delta_unassigned: i32,
}

impl<'b, 't, 'm, 'p, T: Copy + Ord> PlanBuilder<'b, 't, 'm, 'p, T> {
    #[inline]
    pub fn new(
        solver_model: &'m SolverModel<'p, T>,
        base_terminal: &'t TerminalOccupancy<'p, T>,
        decision_vars: &'b mut [DecisionVar<T>],
    ) -> Self {
        Self {
            solver_model,
            decision_vars,
            sandbox: TerminalSandbox::new(base_terminal),
            delta_cost: Cost::zero(),
            delta_unassigned: 0,
            patches: Vec::with_capacity(32),
        }
    }

    #[inline]
    pub fn delta_cost(&self) -> Cost {
        self.delta_cost
    }

    #[inline]
    pub fn delta_unassigned(&self) -> i32 {
        self.delta_unassigned
    }

    #[inline]
    pub fn sandbox(&self) -> &TerminalSandbox<'t, 'p, T> {
        &self.sandbox
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
        let cost = self
            .solver_model
            .cost_of_assignment(request, free_berth.berth_index(), start_time)
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

        self.delta_cost += cost;
        self.delta_unassigned -= 1;

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
        let dv = match self.decision_vars[request.get()] {
            DecisionVar::Unassigned => {
                return Err(ProposeUnassignmentError::NotAssigned(
                    NotAssignedError::new(request),
                ));
            }
            DecisionVar::Assigned(decision) => decision,
        };

        let iv = match self
            .solver_model
            .interval(request, dv.berth_index, dv.start_time)
        {
            Some(interval) => interval,
            None => {
                return Err(ProposeUnassignmentError::NotAllowedOnBerth(
                    NotAllowedOnBerthError::new(request, dv.berth_index),
                ));
            }
        };

        let cost =
            match self
                .solver_model
                .cost_of_assignment(request, dv.berth_index, dv.start_time)
            {
                Some(cost) => cost,
                None => {
                    return Err(ProposeUnassignmentError::NotAllowedOnBerth(
                        NotAllowedOnBerthError::new(request, dv.berth_index),
                    ));
                }
            };

        self.sandbox
            .release(dv.berth_index, iv)
            .map_err(ProposeUnassignmentError::from)?;

        self.delta_cost -= cost;
        self.delta_unassigned += 1;

        self.decision_vars[request.get()] = DecisionVar::Unassigned;
        self.patches
            .push(DecisionVarPatch::new(request, DecisionVar::Unassigned));

        Ok(FreeBerth::new(iv, dv.berth_index))
    }

    #[inline]
    pub fn with_explorer<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&PlanExplorer<'_, 't, 'm, 'p, T>) -> R,
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        let explorer: PlanExplorer<'_, 't, 'm, 'p, T> =
            PlanExplorer::new(self.solver_model, self.decision_vars, &self.sandbox);
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
        self.solver_model
            .cost_of_assignment(request, free_berth.berth_index(), start_time)
    }

    #[inline]
    pub fn finalize(self) -> Plan<'p, T> {
        Plan::new_delta(
            self.patches,
            self.sandbox.delta(),
            self.delta_cost,
            self.delta_unassigned,
        )
    }
}

#[derive(Debug)]
pub struct PlanningContext<'b, 's, 'm, 'p, T: Copy + Ord> {
    model: &'m SolverModel<'p, T>,
    state: &'s SolverState<'p, T>,
    buffer: &'b mut [DecisionVar<T>],
}

impl<'b, 's, 'm, 'p, T: Copy + Ord> PlanningContext<'b, 's, 'm, 'p, T> {
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        state: &'s SolverState<'p, T>,
        buffer: &'b mut [DecisionVar<T>],
    ) -> Self {
        Self {
            model,
            state,
            buffer,
        }
    }

    pub fn state(&self) -> &'s SolverState<'p, T> {
        self.state
    }

    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    // Build a plan using a closure to configure the builder.
    #[inline]
    pub fn with_builder<F>(&mut self, f: F) -> Plan<'p, T>
    where
        F: FnOnce(&mut PlanBuilder<'_, 's, 'm, 'p, T>),
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        // Seed the work buffer with the current decision variables.
        self.buffer.copy_from_slice(self.state.decision_variables());

        // For now we do not use specialized overlays. Just the cloned ledger and terminal,
        // so proposals can be made on them independently from the master state.
        let mut pb = PlanBuilder::new(self.model, self.state.terminal_occupancy(), self.buffer);
        f(&mut pb);
        pb.finalize()
    }

    #[inline]
    pub fn builder(&mut self) -> PlanBuilder<'_, 's, 'm, 'p, T>
    where
        T: CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    {
        // Seed the work buffer with the current decision variables.
        self.buffer.copy_from_slice(self.state.decision_variables());
        PlanBuilder::new(self.model, self.state.terminal_occupancy(), self.buffer)
    }
}

impl<'b, 's, 'm, 'p, T: Copy + Ord + std::fmt::Display> std::fmt::Display
    for PlanningContext<'b, 's, 'm, 'p, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PlanningContext(state: {})", self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        state::{
            berth::berthocc::BerthRead,
            decisionvar::{DecisionVar, DecisionVarVec}, // keep DecisionVarVec for the context test
        },
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

        // NEW: working buffer
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(&model, &term, work_buf.as_mut_slice());

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
        assert!(plan.delta_cost > 0);
        assert_eq!(plan.delta_unassigned, -1);
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
        let mut pb = PlanBuilder::new(&model, &term, work_buf.as_mut_slice());

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

        // NEW: working buffer
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(&model, &term, work_buf.as_mut_slice());

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
    fn test_planning_context_with_builder_isolated_from_master() {
        use crate::state::fitness::Fitness;
        use crate::state::solver_state::SolverState;

        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        // State still uses DecisionVarVec
        let dv = DecisionVarVec::from(vec![
            DecisionVar::unassigned();
            model.flexible_requests_len()
        ]);
        let fit = Fitness::new(0, model.flexible_requests_len());
        let state = SolverState::new(dv, term, fit);

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = PlanningContext::new(&model, &state, &mut work_buf);

        // NEW: provide a working buffer to the context
        let plan = ctx.with_builder(|pb| {
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
        });

        assert_eq!(
            state
                .decision_variables()
                .iter()
                .filter(|dv| dv.is_assigned())
                .count(),
            0
        );

        assert_eq!(plan.decision_var_patches.len(), 1);
        assert!(!plan.terminal_delta.is_empty());
        assert!(plan.delta_cost > 0);
        assert_eq!(plan.delta_unassigned, -1);
    }

    #[test]
    fn test_propose_assignment_not_free_error() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let term = mk_occ(prob.berths().iter());

        // NEW: working buffer
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut pb = PlanBuilder::new(&model, &term, work_buf.as_mut_slice());

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
}
