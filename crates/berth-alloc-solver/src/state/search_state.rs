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
    search::operator::runner::NeighborhoodCandidate,
    state::chain_set::base::ChainSet,
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    common::{FixedKind, FlexibleKind},
    prelude::{AssignmentContainer, SolutionRef},
    problem::asg::AssignmentRef,
};
use num_traits::{CheckedAdd, CheckedSub};
use std::vec;

#[derive(Debug, Clone)]
pub struct SearchSnapshot<'model, 'problem, T: Copy + Ord> {
    pub model: &'model SolverModel<'problem, T>,
    pub chain_set: ChainSet,
    pub interval_vars: Vec<IntervalVar<T>>,
    pub decision_vars: Vec<DecisionVar<T>>,
    pub true_cost: Cost,
}

pub struct SolverSearchState<'model, 'problem, T: Copy + Ord + CheckedAdd + CheckedSub> {
    model: &'model SolverModel<'problem, T>,
    chain_set: ChainSet,
    interval_vars: Vec<IntervalVar<T>>,
    decision_vars: Vec<DecisionVar<T>>,
    current_true_cost: Cost,
    current_search_cost: Cost,
    best_true_cost: Option<Cost>,
    best: Option<SearchSnapshot<'model, 'problem, T>>,
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
        let mut state = Self {
            model,
            chain_set,
            interval_vars,
            decision_vars,
            current_true_cost: initial_true_cost,
            current_search_cost: initial_search_cost,
            best_true_cost: None,
            best: None,
        };

        // Seed best = current
        state.best_true_cost = Some(state.current_true_cost);
        state.best = Some(SearchSnapshot {
            model,
            chain_set: state.chain_set.clone(),
            interval_vars: state.interval_vars.clone(),
            decision_vars: state.decision_vars.clone(),
            true_cost: state.current_true_cost,
        });

        state
    }

    /// Construct a fresh state with all requests Unassigned.
    /// `initial_true_cost` and `initial_search_cost` are computed by the opening/engine.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn new_unassigned(
        model: &'model SolverModel<'problem, T>,
        initial_true_cost: Cost,
        initial_search_cost: Cost,
    ) -> Self {
        let num_chains = model.berths_len();
        let num_nodes = model.flexible_requests_len();

        let interval_vars = model
            .feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect::<Vec<_>>();
        let decision_vars = vec![DecisionVar::Unassigned; num_nodes];
        let chain_set = ChainSet::new(num_nodes, num_chains);
        Self::new(
            model,
            chain_set,
            interval_vars,
            decision_vars,
            initial_true_cost,
            initial_search_cost,
        )
    }

    #[inline]
    pub fn is_improved_true(&self) -> bool {
        match self.best_true_cost {
            None => true,
            Some(best) => self.current_true_cost < best,
        }
    }

    #[inline]
    pub fn restore_best_into_current<TrueObjective, SearchObjective>(
        &mut self,
        obj_true: &TrueObjective,
        obj_search: &SearchObjective,
    ) -> bool
    where
        TrueObjective: Objective<T>,
        SearchObjective: Objective<T>,
    {
        if let Some(best) = &self.best {
            self.chain_set = best.chain_set.clone();
            self.interval_vars = best.interval_vars.clone();
            self.decision_vars = best.decision_vars.clone();
            self.current_true_cost = best.true_cost;

            let (true_acc, search_acc) =
                Self::compute_total_costs(self.model, obj_true, obj_search, &self.decision_vars);
            self.current_true_cost = true_acc;
            self.current_search_cost = search_acc;

            true
        } else {
            false
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

    /// Recompute `current_true_cost` and `current_search_cost` exactly (e.g., after λ change).
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
        // 1) mutate chain structure
        self.chain_set.apply_delta(cand.delta);

        // 2) apply patches
        for p in &cand.interval_var_patch {
            self.interval_vars[p.index()] = *p.patch();
        }
        for p in &cand.decision_vars_patch {
            self.decision_vars[p.index()] = *p.patch();
        }

        // 3) update running costs
        self.current_true_cost = self.current_true_cost.saturating_add(cand.true_delta_cost);
        self.current_search_cost = self
            .current_search_cost
            .saturating_add(cand.search_delta_cost);

        // 4) track best-by-true-objective
        let better = self
            .best_true_cost
            .map(|b| self.current_true_cost < b)
            .unwrap_or(true);

        if better {
            self.best_true_cost = Some(self.current_true_cost);
            self.best = Some(SearchSnapshot {
                model: self.model,
                chain_set: self.chain_set.clone(),
                interval_vars: self.interval_vars.clone(),
                decision_vars: self.decision_vars.clone(),
                true_cost: self.current_true_cost,
            });
        }
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
    pub fn best_true_cost(&self) -> Option<Cost> {
        self.best_true_cost
    }

    pub fn take_best(&mut self) -> Option<SearchSnapshot<'model, 'problem, T>> {
        self.best_true_cost = self.best.as_ref().map(|s| s.true_cost);
        self.best.take()
    }

    pub fn best_snapshot(&self) -> Option<&SearchSnapshot<'model, 'problem, T>> {
        self.best.as_ref()
    }

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
        search::operator::patch::VarPatch,
        state::chain_set::{base::ChainSet, delta::ChainSetDelta, view::ChainSetView},
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
    fn test_new_unassigned_seeds_best_and_dimensions() {
        // 1 berth, 2 requests with weights 3 and 7
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let state = SolverSearchState::new_unassigned(&m, 0, 0);

        // Best is seeded and matches current
        assert_eq!(state.best_true_cost(), Some(0));
        let best = state.best_snapshot().expect("best snapshot must exist");
        assert_eq!(best.true_cost, 0);

        // Dimensions of chain set and variables
        assert_eq!(state.chain_set().num_chains(), m.berths_len());
        assert_eq!(state.chain_set().num_nodes(), m.flexible_requests_len());
        assert_eq!(state.interval_vars().len(), m.flexible_requests_len());
        assert_eq!(state.decision_vars().len(), m.flexible_requests_len());
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
    fn test_apply_candidate_updates_state_and_best_tracking() {
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];

        let mut state = SolverSearchState::new(&m, chain_set, iv, dv, 10, 10);

        // Candidate that assigns request 0 and improves cost by 5 (delta = -5)
        // Note: we assume Cost supports negative deltas (as is typical for signed Cost).
        let cand1 = NeighborhoodCandidate::new(
            ChainSetDelta::new(), // no structural change
            vec![],               // no interval var changes
            vec![VarPatch::new(DecisionVar::assigned(bi(0), tp(5)), 0)], // assign r0
            -5,                   // true delta cost
            -5,                   // search delta cost
        );

        state.apply_candidate(cand1);
        assert_eq!(
            state.decision_vars()[0].as_assigned().unwrap().berth_index,
            bi(0)
        );
        assert_eq!(state.current_true_cost(), 5);
        assert_eq!(state.current_search_cost(), 5);

        // Best should now reflect the improved state
        let best = state
            .best_snapshot()
            .expect("best snapshot exists after improvement");
        assert_eq!(best.true_cost, 5);
        assert_eq!(
            best.decision_vars[0].as_assigned().unwrap().berth_index,
            bi(0)
        );

        // A worsening candidate should not replace best
        let cand2 = NeighborhoodCandidate::new(ChainSetDelta::new(), vec![], vec![], 3, 3);
        state.apply_candidate(cand2);
        assert_eq!(state.current_true_cost(), 8);
        assert_eq!(state.best_true_cost(), Some(5)); // best remains 5
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
    fn test_restore_best_into_current() {
        // Problem: 1 berth, 2 requests (weights 3 and 7)
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];

        // Start at cost 10 (arbitrary), then apply an improvement that assigns r0
        // and reduces cost by exactly -3 (the weight of r0 in our objective).
        let mut state = SolverSearchState::new(&m, chain_set, iv, dv, 10, 10);
        let cand_improve = NeighborhoodCandidate::new(
            ChainSetDelta::new(),
            vec![],
            vec![VarPatch::new(
                DecisionVar::assigned(BerthIndex(0), tp(1)),
                0,
            )],
            -3,
            -3,
        );
        state.apply_candidate(cand_improve);
        assert_eq!(state.best_true_cost(), Some(7)); // 10 + (-3) = 7

        // Make the current state worse to exercise restore
        let cand_worse = NeighborhoodCandidate::new(ChainSetDelta::new(), vec![], vec![], 10, 10);
        state.apply_candidate(cand_worse);
        assert_eq!(state.current_true_cost(), 17);

        // Restore best snapshot and recompute costs using the objective
        let restored = state.restore_best_into_current(&WeightOnlyObjective, &WeightOnlyObjective);
        assert!(restored);

        // After restore, DV has r0 assigned and r1 unassigned => true/search cost 7
        assert_eq!(state.current_true_cost(), 7);
        assert_eq!(state.current_search_cost(), 7);
        assert!(state.decision_vars()[0].is_assigned());
        assert!(matches!(state.decision_vars()[1], DecisionVar::Unassigned));
    }

    #[test]
    fn test_take_best_returns_and_clears_best() {
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        let mut state = SolverSearchState::new(&m, chain_set, iv, dv, 10, 10);

        // Improve cost by -3 by assigning r0
        let cand_improve = NeighborhoodCandidate::new(
            ChainSetDelta::new(),
            vec![],
            vec![VarPatch::new(
                DecisionVar::assigned(BerthIndex(0), tp(1)),
                0,
            )],
            -3,
            -3,
        );
        state.apply_candidate(cand_improve);
        assert_eq!(state.best_true_cost(), Some(7));

        // Take best snapshot out; this clears internal best
        let snap = state.take_best().expect("should take best");
        assert_eq!(snap.true_cost, 7);
        assert!(state.best_snapshot().is_none());
        assert_eq!(state.best_true_cost(), Some(7));

        // restore_best_into_current would now return false (no internal best)
        assert!(!state.restore_best_into_current(&WeightOnlyObjective, &WeightOnlyObjective));
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

    #[test]
    fn test_snapshot_try_into_solution_success() {
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        // Build a fully assigned state
        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let mut dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dv[0] = DecisionVar::assigned(bi(0), tp(0));
        dv[1] = DecisionVar::assigned(bi(0), tp(10));

        let state = SolverSearchState::new(&m, chain_set, iv, dv, 0, 0);
        let snap = state.best_snapshot().expect("snapshot seeded");
        let sol: Result<SolutionRef<'_, i64>, ExportError> = snap.clone().try_into();
        assert!(sol.is_ok());
        let sr = sol.unwrap();
        assert_eq!(sr.flexible_assignments().iter().count(), 2);
        assert_eq!(
            sr.fixed_assignments().iter().count(),
            p.fixed_assignments().iter().count()
        );
    }

    #[test]
    fn test_snapshot_try_into_solution_unassigned_error() {
        let p = build_problem_with_weights(
            &[vec![(0, 100)]],
            &[(0, 50), (10, 60)],
            &[3, 7],
            &[vec![Some(5)], vec![Some(6)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        // One assigned, one unassigned
        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let mut dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        dv[0] = DecisionVar::assigned(bi(0), tp(0));
        dv[1] = DecisionVar::Unassigned;

        let state = SolverSearchState::new(&m, chain_set, iv, dv, 0, 0);
        let snap = state.best_snapshot().expect("snapshot seeded");
        let res: Result<SolutionRef<'_, i64>, ExportError> = snap.clone().try_into();
        match res {
            Err(ExportError::InvalidDecisionVar(ri_)) => assert_eq!(ri_, ri(1)),
            other => panic!("expected InvalidDecisionVar, got {:?}", other),
        }
    }

    #[test]
    fn test_snapshot_try_into_solution_missing_berth_error() {
        let p = build_problem_with_weights(&[vec![(0, 100)]], &[(0, 50)], &[3], &[vec![Some(5)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let chain_set = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        let iv = default_ivars(&m);
        let mut dv = vec![DecisionVar::Unassigned; m.flexible_requests_len()];
        // Intentionally use an invalid berth index
        dv[0] = DecisionVar::assigned(BerthIndex(999), tp(0));

        let state = SolverSearchState::new(&m, chain_set, iv, dv, 0, 0);
        let snap = state.best_snapshot().expect("snapshot seeded");
        let res: Result<SolutionRef<'_, i64>, ExportError> = snap.clone().try_into();
        match res {
            Err(ExportError::MissingBerth(bi_)) => assert_eq!(bi_, BerthIndex(999)),
            other => panic!("expected MissingBerth, got {:?}", other),
        }
    }
}
