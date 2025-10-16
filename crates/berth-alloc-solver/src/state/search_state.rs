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
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    eval::objective::Objective,
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    search::candidate::NeighborhoodCandidate,
    state::chain_set::base::ChainSet,
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    common::{FixedKind, FlexibleKind},
    prelude::{AssignmentContainer, SolutionRef},
    problem::asg::AssignmentRef,
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone)]
pub struct SearchSnapshot<'model, 'problem, T: Copy + Ord> {
    pub model: &'model SolverModel<'problem, T>,
    pub chain_set: ChainSet,
    pub interval_vars: Vec<IntervalVar<T>>,
    pub decision_vars: Vec<DecisionVar<T>>,
    pub true_cost: Cost,
}

#[derive(Debug, Clone)]
pub struct SolverSearchState<'model, 'problem, T: Copy + Ord + CheckedAdd + CheckedSub> {
    model: &'model SolverModel<'problem, T>,
    chain_set: ChainSet,
    interval_vars: Vec<IntervalVar<T>>,
    decision_vars: Vec<DecisionVar<T>>,
    current_true_cost: Cost,
    current_search_cost: Cost,
}

impl<'problem, 'model, T> SolverSearchState<'model, 'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
{
    pub fn new(
        model: &'model SolverModel<'problem, T>,
        chain_set: ChainSet,
        interval_vars: Vec<IntervalVar<T>>,
        decision_vars: Vec<DecisionVar<T>>,
        initial_true_cost: Cost,
        initial_search_cost: Cost,
    ) -> Self {
        Self {
            model,
            chain_set,
            interval_vars,
            decision_vars,
            current_true_cost: initial_true_cost,
            current_search_cost: initial_search_cost,
        }
    }

    /// Helper to compute total costs for a DV vector using caller-provided objectives.
    /// Keeps this module independent of any specific objective type.
    #[inline]
    pub fn compute_total_costs<TrueObjective, SearchObjective>(
        model: &SolverModel<'problem, T>,
        obj_true: &TrueObjective,
        obj_search: &SearchObjective,
        decision_vars: &[DecisionVar<T>],
    ) -> (Cost, Cost)
    where
        TrueObjective: Objective<T>,
        SearchObjective: Objective<T>,
    {
        let mut true_acc: Cost = 0;
        let mut search_acc: Cost = 0;

        for (i, dv) in decision_vars.iter().enumerate() {
            let ri = RequestIndex(i);
            match *dv {
                DecisionVar::Unassigned => {
                    true_acc = true_acc.saturating_add(obj_true.unassignment_cost(model, ri));
                    search_acc = search_acc.saturating_add(obj_search.unassignment_cost(model, ri));
                }
                DecisionVar::Assigned(dec) => {
                    if let Some(c) =
                        obj_true.assignment_cost(model, ri, dec.berth_index, dec.start_time)
                    {
                        true_acc = true_acc.saturating_add(c);
                    }
                    if let Some(c) =
                        obj_search.assignment_cost(model, ri, dec.berth_index, dec.start_time)
                    {
                        search_acc = search_acc.saturating_add(c);
                    }
                }
            }
        }
        (true_acc, search_acc)
    }

    #[inline]
    pub fn recompute_costs<TrueObjective, SearchObjective>(
        &mut self,
        obj_true: &TrueObjective,
        obj_search: &SearchObjective,
    ) where
        TrueObjective: Objective<T>,
        SearchObjective: Objective<T>,
    {
        let (tc, sc) =
            Self::compute_total_costs(self.model, obj_true, obj_search, &self.decision_vars);
        self.current_true_cost = tc;
        self.current_search_cost = sc;
    }

    /// Accept an evaluated candidate (already scheduled & scored) and update costs in O(1).
    #[inline]
    pub fn apply_candidate(&mut self, cand: NeighborhoodCandidate<T>) {
        self.chain_set.apply_delta(cand.delta);
        for p in &cand.interval_var_patch {
            self.interval_vars[p.index()] = *p.patch();
        }
        for p in &cand.decision_vars_patch {
            self.decision_vars[p.index()] = *p.patch();
        }
        self.current_true_cost = self.current_true_cost.saturating_add(cand.true_delta_cost);
        self.current_search_cost = self
            .current_search_cost
            .saturating_add(cand.search_delta_cost);
    }

    #[inline]
    pub fn model(&self) -> &'model SolverModel<'problem, T> {
        self.model
    }
    #[inline]
    pub fn chain_set(&self) -> &ChainSet {
        &self.chain_set
    }
    #[inline]
    pub fn chain_set_mut(&mut self) -> &mut ChainSet {
        &mut self.chain_set
    }
    #[inline]
    pub fn interval_vars(&self) -> &[IntervalVar<T>] {
        &self.interval_vars
    }
    #[inline]
    pub fn interval_vars_mut(&mut self) -> &mut [IntervalVar<T>] {
        &mut self.interval_vars
    }
    #[inline]
    pub fn decision_vars(&self) -> &[DecisionVar<T>] {
        &self.decision_vars
    }
    #[inline]
    pub fn decision_vars_mut(&mut self) -> &mut [DecisionVar<T>] {
        &mut self.decision_vars
    }
    #[inline]
    pub fn current_true_cost(&self) -> Cost {
        self.current_true_cost
    }
    #[inline]
    pub fn current_search_cost(&self) -> Cost {
        self.current_search_cost
    }

    #[inline]
    pub fn snapshot(&self) -> SearchSnapshot<'model, 'problem, T> {
        SearchSnapshot {
            model: self.model,
            chain_set: self.chain_set.clone(),
            interval_vars: self.interval_vars.clone(),
            decision_vars: self.decision_vars.clone(),
            true_cost: self.current_true_cost,
        }
    }

    #[inline]
    pub fn into_snapshot(self) -> SearchSnapshot<'model, 'problem, T> {
        SearchSnapshot {
            model: self.model,
            chain_set: self.chain_set,
            interval_vars: self.interval_vars,
            decision_vars: self.decision_vars,
            true_cost: self.current_true_cost,
        }
    }
}

#[derive(Debug)]
pub enum ExportError {
    MissingRequest(RequestIndex),
    MissingBerth(BerthIndex),
    InvalidDecisionVar(RequestIndex),
    BadAssignmentRef,
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::MissingRequest(ri) => write!(f, "missing request for {:?}", ri),
            ExportError::MissingBerth(bi) => write!(f, "missing berth for {:?}", bi),
            ExportError::InvalidDecisionVar(ri) => write!(f, "invalid decision var at {:?}", ri),
            ExportError::BadAssignmentRef => write!(f, "failed to construct AssignmentRef"),
        }
    }
}
impl std::error::Error for ExportError {}

impl<'model, 'problem, T> TryInto<SolutionRef<'problem, T>>
    for SolverSearchState<'model, 'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    type Error = ExportError;

    fn try_into(self) -> Result<SolutionRef<'problem, T>, Self::Error> {
        let model = self.model;
        let problem = model.problem();
        let im = model.index_manager();

        let fixed_iter = problem.iter_fixed_assignments().map(|a| a.to_ref());
        let fixed_container: AssignmentContainer<
            FixedKind,
            T,
            AssignmentRef<'problem, 'problem, FixedKind, T>,
        > = fixed_iter.collect();
        let mut flex_vec: Vec<AssignmentRef<'problem, 'problem, FlexibleKind, T>> =
            Vec::with_capacity(self.decision_vars.len());

        for (i, dv) in self.decision_vars.into_iter().enumerate() {
            let ri = RequestIndex(i);
            match dv {
                DecisionVar::Unassigned => {
                    return Err(ExportError::InvalidDecisionVar(ri));
                }
                DecisionVar::Assigned(dec) => {
                    let rid = im.request_id(ri).ok_or(ExportError::MissingRequest(ri))?;
                    let bid = im
                        .berth_id(dec.berth_index)
                        .ok_or(ExportError::MissingBerth(dec.berth_index))?;
                    let req_ref = problem
                        .flexible_requests()
                        .get(rid)
                        .ok_or(ExportError::MissingRequest(ri))?;
                    let berth_ref = problem
                        .berths()
                        .get(bid)
                        .ok_or(ExportError::MissingBerth(dec.berth_index))?;
                    let aref =
                        AssignmentRef::<FlexibleKind, T>::new(req_ref, berth_ref, dec.start_time)
                            .map_err(|_| ExportError::BadAssignmentRef)?;

                    flex_vec.push(aref);
                }
            }
        }
        let flexible_container: AssignmentContainer<
            FlexibleKind,
            T,
            AssignmentRef<'problem, 'problem, FlexibleKind, T>,
        > = flex_vec.into_iter().collect();
        Ok(SolutionRef::new(fixed_container, flexible_container))
    }
}

impl<'model, 'problem, T> TryInto<SolutionRef<'problem, T>> for SearchSnapshot<'model, 'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    type Error = ExportError;

    fn try_into(self) -> Result<SolutionRef<'problem, T>, Self::Error> {
        let model = self.model;
        let problem = model.problem();
        let im = model.index_manager();

        // Collect fixed assignments first
        let fixed_iter = problem.iter_fixed_assignments().map(|a| a.to_ref());
        let fixed_container: AssignmentContainer<
            FixedKind,
            T,
            AssignmentRef<'problem, 'problem, FixedKind, T>,
        > = fixed_iter.collect();

        // Build flexible assignment refs
        let mut flex_vec: Vec<AssignmentRef<'problem, 'problem, FlexibleKind, T>> =
            Vec::with_capacity(self.decision_vars.len());

        for (i, dv) in self.decision_vars.into_iter().enumerate() {
            let ri = RequestIndex(i);
            match dv {
                DecisionVar::Unassigned => {
                    return Err(ExportError::InvalidDecisionVar(ri));
                }
                DecisionVar::Assigned(dec) => {
                    let rid = im.request_id(ri).ok_or(ExportError::MissingRequest(ri))?;
                    let bid = im
                        .berth_id(dec.berth_index)
                        .ok_or(ExportError::MissingBerth(dec.berth_index))?;

                    // Validate berth feasibility (processing time is defined)
                    match model.processing_time(ri, dec.berth_index) {
                        Some(Some(_)) => {}
                        _ => return Err(ExportError::InvalidDecisionVar(ri)),
                    }

                    let req_ref = problem
                        .flexible_requests()
                        .get(rid)
                        .ok_or(ExportError::MissingRequest(ri))?;
                    let berth_ref = problem
                        .berths()
                        .get(bid)
                        .ok_or(ExportError::MissingBerth(dec.berth_index))?;

                    let aref =
                        AssignmentRef::<FlexibleKind, T>::new(req_ref, berth_ref, dec.start_time)
                            .map_err(|_| ExportError::BadAssignmentRef)?;
                    flex_vec.push(aref);
                }
            }
        }

        let flexible_container: AssignmentContainer<
            FlexibleKind,
            T,
            AssignmentRef<'problem, 'problem, FlexibleKind, T>,
        > = flex_vec.into_iter().collect();

        Ok(SolutionRef::new(fixed_container, flexible_container))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
        state::chain_set::base::ChainSet,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier, SolutionRef, SolutionView},
        problem::{asg::AssignmentView, builder::ProblemBuilder, req::Request},
    };
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex(n)
    }
    #[inline]
    fn ri(n: usize) -> RequestIndex {
        RequestIndex(n)
    }

    // Build a Problem with weights carried in flexible requests and per-berth processing times.
    fn build_problem_with_weights(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        weights: &[i64],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let _b_len = berths_windows.len();
        let r_len = request_windows.len();
        assert_eq!(processing.len(), r_len);
        assert_eq!(weights.len(), r_len);

        let mut builder = ProblemBuilder::new();

        for (i, windows) in berths_windows.iter().enumerate() {
            let b = Berth::from_windows(bid(i), windows.iter().map(|&(s, e)| iv(s, e)));
            builder.add_berth(b);
        }

        for (i, &(ws, we)) in request_windows.iter().enumerate() {
            let mut map = BTreeMap::new();
            for (j, p) in processing[i].iter().copied().enumerate() {
                if let Some(dur) = p {
                    map.insert(bid(j), td(dur));
                }
            }
            let req = Request::<FlexibleKind, i64>::new(rid(i), iv(ws, we), weights[i], map)
                .expect("request should be well-formed");
            builder.add_flexible(req);
        }

        builder.build().expect("problem should build")
    }

    fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
        m.feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect()
    }

    // Simple objective: unassigned cost equals model weight; assigned cost is zero.
    struct WeightOnlyObjective;
    impl Objective<i64> for WeightOnlyObjective {
        fn assignment_cost(
            &self,
            _model: &SolverModel<'_, i64>,
            _request_index: RequestIndex,
            _berth_index: BerthIndex,
            _start_time: TimePoint<i64>,
        ) -> Option<Cost> {
            Some(0)
        }

        fn unassignment_cost(
            &self,
            model: &SolverModel<'_, i64>,
            request_index: RequestIndex,
        ) -> Cost {
            model.weights()[request_index.get()]
        }
    }

    #[test]
    fn test_compute_and_recompute_costs_with_weight_objective() {
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        // Start with explicit state: first Unassigned, second Assigned (cost 0 for assigned)
        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let mut dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dv[1] = DecisionVar::assigned(bi(0), tp(0));

        let mut state = SolverSearchState::new(&m, chain_set, iv, dv, 0, 0);

        // Compute totals with the objective
        let (t, s) = SolverSearchState::compute_total_costs(
            &m,
            &WeightOnlyObjective,
            &WeightOnlyObjective,
            state.decision_vars(),
        );
        assert_eq!(t, 3);
        assert_eq!(s, 3);

        // Recompute into state
        state.recompute_costs(&WeightOnlyObjective, &WeightOnlyObjective);
        assert_eq!(state.current_true_cost(), 3);
        assert_eq!(state.current_search_cost(), 3);
    }

    #[test]
    fn test_export_solution_success_and_errors() {
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        // Success: both assigned on valid berth 0
        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let mut dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dv[0] = DecisionVar::assigned(bi(0), tp(0));
        dv[1] = DecisionVar::assigned(bi(0), tp(10));

        let state_ok = SolverSearchState::new(&m, chain_set.clone(), iv.clone(), dv, 0, 0);
        let sol: Result<SolutionRef<'_, i64>, ExportError> = state_ok.try_into();
        assert!(sol.is_ok());

        // Error: one unassigned -> InvalidDecisionVar
        let mut dv_err = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dv_err[0] = DecisionVar::assigned(bi(0), tp(0));
        dv_err[1] = DecisionVar::Unassigned;

        let state_err = SolverSearchState::new(&m, chain_set.clone(), iv.clone(), dv_err, 0, 0);
        let res_err: Result<SolutionRef<'_, i64>, ExportError> = state_err.try_into();
        let err = res_err.unwrap_err();
        match err {
            ExportError::InvalidDecisionVar(ri_) => assert_eq!(ri_, ri(1)),
            x => panic!("expected InvalidDecisionVar, got {:?}", x),
        }

        // Error: invalid berth index -> MissingBerth
        let mut dv_bad_berth = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dv_bad_berth[0] = DecisionVar::assigned(BerthIndex(999), tp(0));
        dv_bad_berth[1] = DecisionVar::assigned(bi(0), tp(10));
        let state_badb = SolverSearchState::new(&m, chain_set, iv, dv_bad_berth, 0, 0);
        let res_err2: Result<SolutionRef<'_, i64>, ExportError> = state_badb.try_into();
        let err2 = res_err2.unwrap_err();
        match err2 {
            ExportError::MissingBerth(bi_) => assert_eq!(bi_, BerthIndex(999)),
            x => panic!("expected MissingBerth, got {:?}", x),
        }
    }

    #[test]
    fn test_export_solution_includes_fixed_assignments() {
        use berth_alloc_model::{
            common::{FixedKind, FlexibleKind},
            prelude::{Assignment, SolutionRef},
        };

        // Build a problem with:
        // - 1 berth [0, 100)
        // - 1 fixed assignment on that berth (start=10, duration=10 => [10,20))
        // - 1 flexible request allowed on that berth (we'll assign it arbitrarily)
        let mut builder = ProblemBuilder::new();

        let b0 = Berth::from_windows(bid(0), [iv(0, 100)]);
        builder.add_berth(b0.clone());

        // Fixed request data
        let mut fixed_pt = BTreeMap::new();
        fixed_pt.insert(bid(0), td(10));
        let r_fixed = Request::<FixedKind, i64>::new(rid(1_000), iv(0, 100), 0, fixed_pt).unwrap();
        let a_fixed =
            Assignment::<FixedKind, i64>::new(r_fixed.clone(), b0.clone(), tp(10)).unwrap();
        builder.add_fixed(a_fixed);

        // One flexible request allowed on berth 0
        let mut flex_pt = BTreeMap::new();
        flex_pt.insert(bid(0), td(5));
        let r_flex = Request::<FlexibleKind, i64>::new(rid(0), iv(0, 100), 1, flex_pt).unwrap();
        builder.add_flexible(r_flex);

        let p = builder.build().expect("problem builds with fixed + flex");
        let m = SolverModel::from_problem(&p).unwrap();

        // Build a trivial assigned state for the single flexible request
        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let ivars = default_ivars(&m);
        let mut dvars = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dvars[0] = DecisionVar::assigned(bi(0), tp(0));

        let state = SolverSearchState::new(&m, chain_set, ivars, dvars, 0, 0);

        // Export and assert fixed assignments are present as in the problem
        let sol: SolutionRef<'_, i64> = state.try_into().expect("export should succeed");
        let fixed: Vec<_> = sol.fixed_assignments().iter().collect();
        assert_eq!(
            sol.fixed_assignments().len(),
            1,
            "exactly one fixed assignment expected"
        );

        let fa = fixed[0];
        assert_eq!(fa.request_id(), rid(1_000));
        assert_eq!(fa.berth_id(), bid(0));
        assert_eq!(fa.start_time(), tp(10));
    }
}
