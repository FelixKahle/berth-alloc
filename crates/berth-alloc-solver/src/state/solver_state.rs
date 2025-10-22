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
    model::solver_model::SolverModel,
    state::{
        decisionvar::{DecisionVar, DecisionVarVec},
        fitness::Fitness,
        plan::{DecisionVarPatch, Plan},
        terminal::terminalocc::{TerminalOccupancy, TerminalWrite},
    },
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    common::FlexibleKind,
    prelude::{AssignmentContainer, SolutionRef},
    problem::asg::AssignmentRef,
    solution::SolutionError,
};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

pub trait SolverStateView<'p, T: Copy + Ord> {
    fn decision_variables(&self) -> &[DecisionVar<T>];
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T>;
    fn fitness(&self) -> &Fitness;

    fn is_feasible(&self) -> bool {
        self.fitness().unassigned_requests == 0
    }

    fn cost(&self) -> Cost {
        self.fitness().cost
    }
}

#[derive(Debug, Clone)]
pub struct SolverState<'p, T: Copy + Ord> {
    decision_variables: DecisionVarVec<T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
    fitness: Fitness,
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>>
    SolverState<'p, T>
{
    #[inline]
    pub fn new(
        decision_variables: DecisionVarVec<T>,
        terminal_occupancy: TerminalOccupancy<'p, T>,
        fitness: Fitness,
    ) -> Self {
        Self {
            decision_variables,
            terminal_occupancy,
            fitness,
        }
    }

    #[inline]
    pub fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    #[inline]
    pub fn fitness(&self) -> &Fitness {
        &self.fitness
    }

    #[inline]
    pub fn apply_plan(&mut self, plan: Plan<'p, T>)
    where
        T: std::fmt::Debug,
    {
        #[cfg(debug_assertions)]
        let prev_cost = self.fitness.cost;
        #[cfg(debug_assertions)]
        let prev_unassigned = self.fitness.unassigned_requests;

        // Apply DV patches and terminal delta first
        self.apply_decision_var_patches(plan.decision_var_patches);
        let res = self.terminal_occupancy.apply_delta(plan.terminal_delta);
        debug_assert!(res.is_ok(), "Failed to apply terminal delta: {:?}", res);

        // Apply fitness deltas to keep the state consistent with the plan
        self.fitness.cost += plan.delta_cost;

        if plan.delta_unassigned != 0 {
            // Use checked_add_signed to handle +/- deltas safely
            let new_unassigned = self
                .fitness
                .unassigned_requests
                .checked_add_signed(isize::try_from(plan.delta_unassigned).unwrap())
                .expect("unassigned delta overflow");
            self.fitness.unassigned_requests = new_unassigned;
        }

        #[cfg(debug_assertions)]
        {
            debug_assert!(
                self.fitness.cost == prev_cost + plan.delta_cost,
                "fitness.cost did not match delta: prev={prev_cost}, delta={}, new={}",
                plan.delta_cost,
                self.fitness.cost
            );
            debug_assert!(
                self.fitness.cost.is_positive(),
                "fitness.cost became negative"
            );

            let expected_unassigned = prev_unassigned
                .checked_add_signed(isize::try_from(plan.delta_unassigned).unwrap())
                .unwrap();

            debug_assert_eq!(
                self.fitness.unassigned_requests, expected_unassigned,
                "fitness.unassigned_requests did not match delta"
            );

            let dv_unassigned = self.count_unassigned_requests();
            debug_assert_eq!(
                dv_unassigned, self.fitness.unassigned_requests,
                "decision-variables unassigned count ({dv_unassigned}) differs from fitness.unassigned_requests ({})",
                self.fitness.unassigned_requests
            );
        }
    }

    #[inline]
    pub fn into_solution(
        self,
        solver_model: &SolverModel<'p, T>,
    ) -> Result<SolutionRef<'p, T>, SolutionError>
    where
        T: std::fmt::Display + std::fmt::Debug,
    {
        tracing::info!("Cost of solution: {}", self.fitness.cost);

        let flexible_assignments = self.make_flexible_assignments(solver_model);
        let fixed_assignments = solver_model
            .problem()
            .iter_fixed_assignments()
            .map(|a| a.to_ref())
            .collect();

        SolutionRef::new(
            fixed_assignments,
            flexible_assignments,
            solver_model.problem(),
        )
    }

    pub fn make_flexible_assignments(
        &self,
        solver_model: &SolverModel<'p, T>,
    ) -> AssignmentContainer<FlexibleKind, T, AssignmentRef<'p, 'p, FlexibleKind, T>>
    where
        T: std::fmt::Display + std::fmt::Debug,
    {
        let index_manager = solver_model.index_manager();
        let problem = solver_model.problem();

        let mut assignments = AssignmentContainer::<
            FlexibleKind,
            T,
            AssignmentRef<'p, 'p, FlexibleKind, T>,
        >::with_capacity(self.decision_variables.len());

        for (request_index, decision_var) in self.decision_variables.enumerate() {
            let DecisionVar::Assigned(decision) = decision_var else {
                continue;
            };

            let Some(request_id) = index_manager.request_id(request_index) else {
                debug_assert!(false, "request_id missing for {}", request_index);
                continue;
            };

            let Some(berth_id) = index_manager.berth_id(decision.berth_index) else {
                debug_assert!(false, "berth_id missing for {}", decision.berth_index);
                continue;
            };

            let Some(request_ref) = problem.flexible_requests().get(request_id) else {
                debug_assert!(false, "request {} not found in Problem", request_id);
                continue;
            };
            let Some(berth_ref) = problem.berths().get(berth_id) else {
                debug_assert!(false, "berth {} not found in Problem", berth_id);
                continue;
            };

            match AssignmentRef::new(request_ref, berth_ref, decision.start_time) {
                Ok(asg_ref) => {
                    assignments.insert(asg_ref);
                }
                Err(_) => {
                    debug_assert!(
                        false,
                        "failed to create AssignmentRef for request {} and berth {}",
                        request_id, berth_id
                    );
                    continue;
                }
            }
        }

        assignments
    }

    #[inline(always)]
    fn apply_decision_var_patches(&mut self, patches: Vec<DecisionVarPatch<T>>) {
        for patch in patches {
            self.decision_variables[patch.index] = patch.patch;
        }
    }

    #[inline]
    #[cfg(debug_assertions)]
    fn count_unassigned_requests(&self) -> usize {
        self.decision_variables
            .as_slice()
            .iter()
            .filter(|dv| !dv.is_assigned())
            .count()
    }
}

impl<'p, T: Copy + Ord> SolverStateView<'p, T> for SolverState<'p, T> {
    #[inline]
    fn terminal_occupancy(&self) -> &TerminalOccupancy<'p, T> {
        &self.terminal_occupancy
    }

    #[inline]
    fn fitness(&self) -> &Fitness {
        &self.fitness
    }

    #[inline]
    fn decision_variables(&self) -> &[DecisionVar<T>] {
        &self.decision_variables
    }
}

impl<'p, T: Copy + Ord + std::fmt::Display> std::fmt::Display for SolverState<'p, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SolverState {{ cost: {}, unassigned_requests: {} }}",
            self.fitness.cost, self.fitness.unassigned_requests
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{
            index::{BerthIndex, RequestIndex},
            solver_model::SolverModel,
        },
        state::{
            berth::berthocc::{BerthRead, BerthWrite},
            decisionvar::{DecisionVar, DecisionVarVec},
            plan::{DecisionVarPatch, Plan},
            terminal::{
                delta::TerminalDelta,
                terminalocc::{TerminalOccupancy, TerminalRead},
            },
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier, SolutionView},
        problem::{builder::ProblemBuilder, req::Request},
    };
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
    fn td(v: i64) -> TimeDelta<i64> {
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
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex::new(n)
    }
    #[inline]
    fn ri(n: usize) -> RequestIndex {
        RequestIndex::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pts: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn make_problem_simple() -> Problem<i64> {
        // Two berths and two flexible requests available on berth 1
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);

        let r10 = flex_req(10, (0, 100), &[(1, 5)], 1);
        let r20 = flex_req(20, (0, 100), &[(1, 5)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r10);
        builder.add_flexible(r20);
        builder.build().expect("valid problem")
    }

    #[test]
    fn test_new_and_accessors() {
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);
        let dv = DecisionVarVec::from(vec![DecisionVar::unassigned(), DecisionVar::unassigned()]);
        let fit = Fitness::new(100, 2);

        let st = SolverState::new(dv, term, fit);
        assert_eq!(st.fitness().cost, 100);
        assert_eq!(st.fitness().unassigned_requests, 2);
        assert!(!st.is_feasible());
        assert_eq!(st.cost(), 100);

        // Accessor returns the same slice semantics
        assert_eq!(st.decision_variables().len(), 2);
        assert!(matches!(
            st.decision_variables()[0],
            DecisionVar::Unassigned
        ));
    }

    #[test]
    fn test_display_format() {
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);
        let dv = DecisionVarVec::from(vec![DecisionVar::unassigned()]);
        let fit = Fitness::new(123, 1);
        let st = SolverState::new(dv, term, fit);

        let s = format!("{}", st);
        assert!(s.contains("cost: 123"), "fmt must include cost; got: {s}");
        assert!(
            s.contains("unassigned_requests: 1"),
            "fmt must include unassigned; got: {s}"
        );
    }

    #[test]
    fn test_apply_decision_var_patches_private() {
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);
        // 3 vars: U, U, A
        let mut st = SolverState::new(
            DecisionVarVec::from(vec![
                DecisionVar::unassigned(),
                DecisionVar::unassigned(),
                DecisionVar::assigned(BerthIndex::new(0), tp(50)),
            ]),
            term,
            Fitness::new(10, 2),
        );

        // Apply patches directly (private method) to avoid delta assertions
        st.apply_decision_var_patches(vec![
            DecisionVarPatch::new(ri(0), DecisionVar::assigned(bi(0), tp(0))), // U -> A
            DecisionVarPatch::new(ri(1), DecisionVar::unassigned()),           // stays U
            DecisionVarPatch::new(ri(2), DecisionVar::unassigned()),           // A -> U
        ]);

        assert!(st.decision_variables()[0].is_assigned());
        assert!(!st.decision_variables()[1].is_assigned());
        assert!(!st.decision_variables()[2].is_assigned());
    }

    #[test]
    fn test_apply_plan_balanced_patches_and_terminal_delta() {
        // Initial: [A(0,0), U] → unassigned=1; cost positive
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);

        // Build occupancy delta to occupy [0,5) on berth index 0
        let mut occ0 = term.berth(BerthIndex(0)).cloned().expect("berth exists");
        occ0.occupy(iv(0, 5)).expect("occupy in delta must succeed");
        let delta = TerminalDelta::from_updates(vec![(BerthIndex(0), occ0)]);

        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(bi(0), tp(0)),
            DecisionVar::unassigned(),
        ]);
        let mut st = SolverState::new(dv, term, Fitness::new(50, 1));

        // Balanced patches: swap A<->U; delta_unassigned=0; delta_cost=0
        let patches = vec![
            DecisionVarPatch::new(ri(0), DecisionVar::unassigned()),
            DecisionVarPatch::new(ri(1), DecisionVar::assigned(bi(0), tp(10))),
        ];

        let plan = Plan::new_delta(patches, delta, 0, 0);
        st.apply_plan(plan);

        // Decision variables swapped
        assert!(!st.decision_variables()[0].is_assigned());
        assert!(st.decision_variables()[1].is_assigned());

        // Terminal occupancy reflects [0,5) occupied on index 0
        let b0_view = st
            .terminal_occupancy()
            .berth(BerthIndex(0))
            .expect("berth 0 must exist");
        assert!(b0_view.is_occupied(iv(0, 5)), "terminal delta not applied");
    }

    #[test]
    fn test_make_flexible_assignments_happy_path() {
        let prob = make_problem_simple();
        let model = SolverModel::try_from(&prob).expect("build model");

        // Build DV with both assigned on berth index of id 1
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();
        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(bi1, tp(0)),
            DecisionVar::assigned(bi1, tp(10)),
        ]);
        let term = TerminalOccupancy::new(std::iter::empty::<&Berth<i64>>());
        let st = SolverState::new(dv, term, Fitness::new(10, 0));

        let flex = st.make_flexible_assignments(&model);
        assert_eq!(
            flex.len(),
            model.flexible_requests_len(),
            "all assigned should be materialized"
        );
    }

    #[test]
    fn test_into_solution_success() {
        let prob = make_problem_simple();
        let model = SolverModel::try_from(&prob).expect("build model");

        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();
        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(bi1, tp(0)),
            DecisionVar::assigned(bi1, tp(10)),
        ]);
        let term = TerminalOccupancy::new(std::iter::empty::<&Berth<i64>>());
        let st = SolverState::new(dv, term, Fitness::new(10, 0));

        let sol = st.into_solution(&model).expect("into_solution Ok");
        assert_eq!(sol.flexible_assignments_len(), 2);
    }

    #[test]
    fn test_decision_variables_exposure_slice_semantics() {
        let dv = DecisionVarVec::from(vec![
            DecisionVar::unassigned(),
            DecisionVar::assigned(bi(0), tp(1)),
        ]);
        let st = SolverState::new(
            dv,
            TerminalOccupancy::new(&[] as &[Berth<i64>]),
            Fitness::new(5, 1),
        );

        let slice = st.decision_variables();
        assert_eq!(slice.len(), 2);
        match slice[0] {
            DecisionVar::Unassigned => {}
            _ => panic!("expected Unassigned at 0"),
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_count_unassigned_requests_debug_only() {
        let dv = DecisionVarVec::from(vec![
            DecisionVar::unassigned(),
            DecisionVar::assigned(bi(0), tp(1)),
            DecisionVar::unassigned(),
        ]);
        let st = SolverState::new(
            dv,
            TerminalOccupancy::new(&[] as &[Berth<i64>]),
            Fitness::new(7, 2),
        );
        let cnt = st.count_unassigned_requests();
        assert_eq!(cnt, 2);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_make_flexible_assignments_invalid_entries_debug_panics() {
        let prob = make_problem_simple();
        let model = SolverModel::try_from(&prob).expect("build model");

        // Invalid berth index (out of range) for index 0
        let invalid_bi = BerthIndex::new(999);
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();

        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(invalid_bi, tp(0)), // triggers debug_assert! false
            DecisionVar::assigned(bi1, tp(10)),
        ]);

        // Empty terminal occupancy using a typed empty slice to avoid E0716
        let term = TerminalOccupancy::new(&[] as &[Berth<i64>]);

        let st = SolverState::new(dv, term, Fitness::new(10, 0));

        // This should panic in debug mode due to debug_assert!(false, ...)
        let _ = st.make_flexible_assignments(&model);
    }

    // In release builds, invalid entries are skipped (no panic).
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_make_flexible_assignments_skips_invalid_entries_release() {
        let prob = make_problem_simple();
        let model = SolverModel::try_from(&prob).expect("build model");

        let invalid_bi = BerthIndex::new(999);
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();

        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(invalid_bi, tp(0)), // will be skipped in release
            DecisionVar::assigned(bi1, tp(10)),       // ok
        ]);

        // Empty terminal occupancy using a typed empty slice to avoid E0716
        let term = TerminalOccupancy::new(&[] as &[Berth<i64>]);

        let st = SolverState::new(dv, term, Fitness::new(10, 0));

        let flex = st.make_flexible_assignments(&model);
        assert_eq!(
            flex.len(),
            1,
            "invalid mapping should be skipped in release"
        );
    }

    #[test]
    fn test_make_flexible_assignments_empty() {
        let prob = make_problem_simple();
        let model = SolverModel::try_from(&prob).expect("build model");

        let dv = DecisionVarVec::from(Vec::<DecisionVar<i64>>::new());
        let term = TerminalOccupancy::new(&[] as &[Berth<i64>]);
        let st = SolverState::new(dv, term, Fitness::new(0, 0));

        let flex = st.make_flexible_assignments(&model);
        assert_eq!(flex.len(), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_make_flexible_assignments_request_index_missing_debug_panics() {
        let prob = make_problem_simple(); // 2 flex requests
        let model = SolverModel::try_from(&prob).expect("build model");
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();

        // 3 DVs → index 2 has no RequestIdentifier mapping
        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(bi1, tp(0)),
            DecisionVar::assigned(bi1, tp(10)),
            DecisionVar::assigned(bi1, tp(20)), // triggers missing request_id
        ]);
        let term = TerminalOccupancy::new(&[] as &[Berth<i64>]);
        let st = SolverState::new(dv, term, Fitness::new(0, 0));

        let _ = st.make_flexible_assignments(&model); // debug panics
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn test_make_flexible_assignments_request_index_missing_release_skips() {
        let prob = make_problem_simple();
        let model = SolverModel::try_from(&prob).expect("build model");
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();

        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(bi1, tp(0)),
            DecisionVar::assigned(bi1, tp(10)),
            DecisionVar::assigned(bi1, tp(20)), // skipped in release
        ]);
        let term = TerminalOccupancy::new(&[] as &[Berth<i64>]);
        let st = SolverState::new(dv, term, Fitness::new(0, 0));

        let flex = st.make_flexible_assignments(&model);
        assert_eq!(flex.len(), 2);
    }

    #[test]
    fn test_into_solution_includes_fixed_assignments() {
        // Build a problem with one fixed assignment plus two flex
        use berth_alloc_model::common::FixedKind;
        use berth_alloc_model::prelude::Assignment;

        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();

        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        builder.add_berth(b1.clone());
        builder.add_berth(b2.clone());

        let r_fixed = Request::<FixedKind, i64>::new(
            rid(999),
            iv(0, 100),
            1,
            [(bid(1), td(10))].into_iter().collect(),
        )
        .unwrap();
        let a_fixed =
            Assignment::<FixedKind, i64>::new(r_fixed.clone(), b1.clone(), tp(0)).unwrap();
        builder.add_fixed(a_fixed);

        let r10 = flex_req(10, (0, 100), &[(1, 5)], 1);
        let r20 = flex_req(20, (0, 100), &[(1, 5)], 1);
        builder.add_flexible(r10);
        builder.add_flexible(r20);

        let prob = builder.build().expect("valid problem");
        let model = SolverModel::try_from(&prob).expect("build model");

        // Assign both flex on berth 1
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();
        let dv = DecisionVarVec::from(vec![
            DecisionVar::assigned(bi1, tp(0)),
            DecisionVar::assigned(bi1, tp(10)),
        ]);
        let term = TerminalOccupancy::new(std::iter::empty::<&Berth<i64>>());
        let st = SolverState::new(dv, term, Fitness::new(0, 0));

        let sol = st.into_solution(&model).expect("into_solution Ok");
        assert_eq!(sol.flexible_assignments_len(), 2);
        assert_eq!(
            sol.fixed_assignments_len(),
            1,
            "fixed assignments preserved"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_apply_plan_unassigned_delta_mismatch_debug_panics() {
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);

        // Start with both Unassigned → unassigned=2
        let dv = DecisionVarVec::from(vec![DecisionVar::unassigned(), DecisionVar::unassigned()]);
        let mut st = SolverState::new(dv, term, Fitness::new(0, 2));

        // Patch assigns one request (U -> A), so real delta_unassigned = -1,
        // but we lie in the plan and set 0 → should panic in debug.
        let patches = vec![DecisionVarPatch::new(
            ri(0),
            DecisionVar::assigned(bi(0), tp(0)),
        )];
        let delta = crate::state::terminal::delta::TerminalDelta::empty();

        let plan = Plan::new_delta(patches, delta, 0, 0); // incorrect delta_unassigned
        st.apply_plan(plan);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_apply_plan_cost_delta_updates_fitness_debug() {
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);
        let dv = DecisionVarVec::from(vec![DecisionVar::unassigned()]);
        // fitness.cost = 10; we'll claim delta_cost = +7 (no-op patches)
        let mut st = SolverState::new(dv, term, Fitness::new(10, 1));

        let patches = vec![DecisionVarPatch::new(ri(0), DecisionVar::unassigned())]; // no-op
        let delta = crate::state::terminal::delta::TerminalDelta::empty();

        let plan = Plan::new_delta(patches, delta, 7, 0);
        st.apply_plan(plan);

        // With apply_plan now applying fitness deltas, cost must update and unassigned remain unchanged
        assert_eq!(st.fitness().cost, 17);
        assert_eq!(st.fitness().unassigned_requests, 1);
        assert!(matches!(
            st.decision_variables()[0],
            DecisionVar::Unassigned
        ));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_apply_plan_terminal_delta_error_debug_panics() {
        let base = vec![berth(1, 0, 100)];
        let term = TerminalOccupancy::new(&base);
        let dv = DecisionVarVec::from(vec![DecisionVar::unassigned()]);
        let mut st = SolverState::new(dv, term.clone(), Fitness::new(0, 1));

        // Create an invalid delta by taking a snapshot of berth 1 and
        // releasing an interval that isn't occupied in the snapshot.
        let mut occ = term.berth(BerthIndex(1)).cloned().expect("exists");
        // Release [0,5) which is not occupied → should error when applied
        occ.release(iv(0, 5))
            .expect_err("release on free interval must error");
        // But we have a mutated copy with an illegal state; feed it as an update
        let delta =
            crate::state::terminal::delta::TerminalDelta::from_updates(vec![(BerthIndex(1), occ)]);

        let plan = Plan::new_delta(Vec::new(), delta, 0, 0);
        st.apply_plan(plan); // debug_assert!(res.is_ok()) panics
    }
}
