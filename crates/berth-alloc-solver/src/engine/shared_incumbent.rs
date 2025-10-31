// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to do so, subject to the following conditions:
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
    model::solver_model::SolverModel,
    state::{
        fitness::Fitness,
        solver_state::{SolverState, SolverStateView},
    },
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

#[derive(Debug)]
pub struct SharedIncumbent<'p, T>
where
    T: Copy + Ord,
{
    best_state: Mutex<SolverState<'p, T>>,
    best_unassigned: AtomicUsize, // Avoid locking for simple reads
    best_cost: AtomicI64,         // Avoid locking for simple reads
}

impl<'p, T> PartialEq for SharedIncumbent<'p, T>
where
    T: Copy + Ord,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let self_guard = self.best_state.lock();
        let other_guard = other.best_state.lock();
        *self_guard == *other_guard
    }
}

impl<'p, T> Eq for SharedIncumbent<'p, T> where T: Copy + Ord {}

impl<'p, T> SharedIncumbent<'p, T>
where
    T: Copy + Ord,
{
    #[inline]
    pub fn new(initial_state: SolverState<'p, T>) -> Self {
        Self {
            best_cost: AtomicI64::new(initial_state.fitness().cost),
            best_unassigned: AtomicUsize::new(initial_state.fitness().unassigned_requests),
            best_state: Mutex::new(initial_state),
        }
    }

    /// Lightweight best-known snapshot without locking the big state.
    #[inline]
    pub fn peek(&self) -> Fitness {
        Fitness::new(
            self.best_cost.load(Ordering::Acquire),
            self.best_unassigned.load(Ordering::Acquire),
        )
    }

    /// Full cloned snapshot (requires the state to be `Clone`).
    #[inline]
    pub fn snapshot(&self) -> SolverState<'p, T>
    where
        SolverState<'p, T>: Clone,
    {
        self.best_state.lock().clone()
    }

    #[tracing::instrument(level = "debug", skip(self, candidate_state, model))]
    pub fn try_update(
        &self,
        candidate_state: &SolverState<'p, T>,
        model: &SolverModel<'p, T>,
    ) -> bool
    where
        T: SolveNumeric,
        SolverState<'p, T>: Clone,
    {
        let cand_fit = *candidate_state.fitness();
        let best_atomic = self.peek();
        if cand_fit >= best_atomic {
            return false;
        }

        let mut best_guard = self.best_state.lock();
        let best_locked = *best_guard.fitness();
        if cand_fit >= best_locked {
            return false;
        }

        let snapshot = candidate_state.clone();
        let snap_fit = *snapshot.fitness();

        if snap_fit >= best_locked {
            return false;
        }

        tracing::info!(
            old_unassigned = best_locked.unassigned_requests,
            old_cost = best_locked.cost,
            new_unassigned = snap_fit.unassigned_requests,
            new_cost = snap_fit.cost,
            "New incumbent"
        );

        *best_guard = snapshot;
        self.best_unassigned
            .store(snap_fit.unassigned_requests, Ordering::Release);
        self.best_cost.store(snap_fit.cost, Ordering::Release);

        #[cfg(debug_assertions)]
        {
            use berth_alloc_model::prelude::SolutionView;

            let installed = best_guard.fitness();
            debug_assert_eq!(
                *installed, snap_fit,
                "installed state fitness differs from atomics/logs"
            );

            if installed.unassigned_requests == 0 {
                let true_cost = best_guard.clone().into_solution(model).unwrap().cost();
                debug_assert_eq!(
                    installed.cost, true_cost,
                    "installed cost differs from recomputed solution cost"
                );
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::{CostEvaluator, DefaultCostEvaluator},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier, SolutionView},
        problem::{builder::ProblemBuilder, req::Request},
    };
    use std::collections::BTreeMap;
    use std::thread;

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

    // One-berth, one-flexible request problem; useful for feasible candidate tests.
    fn make_problem_one_flex() -> Problem<i64> {
        // Berth 1 with window [0, 100)
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);

        // Request 10 with window [0, 100), weight 1, pt on berth 1 is 10
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 1, pt).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.build().unwrap()
    }

    // One-berth, two-flexible requests; useful for infeasible improvements (avoid debug recompute).
    fn make_problem_two_flex() -> Problem<i64> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);

        // r10 pt=10, r20 pt=20, both on berth 1
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(20));
        let r20 = Request::<FlexibleKind, i64>::new(rid(20), iv(0, 100), 1, pt2).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.add_flexible(r20);
        builder.build().unwrap()
    }

    // Build a SolverModel + initial state (all Unassigned) with computed fitness.
    fn make_model_and_initial_state(
        problem: &Problem<i64>,
    ) -> (SolverModel<'_, i64>, SolverState<'_, i64>) {
        let model = SolverModel::try_from(problem).expect("model should build");
        let term = TerminalOccupancy::new(problem.berths().iter());

        let dvars_vec = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fitness = DefaultCostEvaluator.eval_fitness(&model, &dvars_vec);

        let state = SolverState::new(DecisionVarVec::from(dvars_vec), term, fitness);
        (model, state)
    }

    #[test]
    fn test_new_peek_and_snapshot() {
        let problem = make_problem_two_flex();
        let (_model, init_state) = make_model_and_initial_state(&problem);

        // initial: cost 0, unassigned = 2
        assert_eq!(init_state.fitness().cost, 0);
        assert_eq!(init_state.fitness().unassigned_requests, 2);

        let inc = SharedIncumbent::new(init_state.clone());

        let peek = inc.peek();
        assert_eq!(peek.cost, 0);
        assert_eq!(peek.unassigned_requests, 2);

        // snapshot requires SolverState: Clone
        let snap = inc.snapshot();
        assert_eq!(snap, init_state);
    }

    #[test]
    fn test_try_update_rejects_worse_or_equal() {
        let problem = make_problem_two_flex();
        let (model, init_state) = make_model_and_initial_state(&problem);
        let inc = SharedIncumbent::new(init_state.clone());

        // Worse: same unassigned, higher cost
        let worse = {
            let dv = DecisionVarVec::from_slice(init_state.decision_variables());
            let term = init_state.terminal_occupancy().clone();
            // Note: Fitness here does not need to match DVs for this test (infeasible -> no debug recompute)
            let f = Fitness::new(999, init_state.fitness().unassigned_requests);
            SolverState::new(dv, term, f)
        };
        assert!(!inc.try_update(&worse, &model));
        assert_eq!(inc.peek(), *init_state.fitness());

        // Equal: same fitness (should not update)
        let equal = init_state.clone();
        assert!(!inc.try_update(&equal, &model));
        assert_eq!(inc.peek(), *init_state.fitness());
    }

    #[test]
    fn test_try_update_accepts_improved_infeasible() {
        let problem = make_problem_two_flex();
        let (model, init_state) = make_model_and_initial_state(&problem);
        let inc = SharedIncumbent::new(init_state.clone());

        // Better: fewer unassigned (2 -> 1), cost arbitrary (infeasible so no debug recompute)
        let better = {
            let dv = DecisionVarVec::from_slice(init_state.decision_variables());
            let term = init_state.terminal_occupancy().clone();
            let f = Fitness::new(123, init_state.fitness().unassigned_requests - 1);
            SolverState::new(dv, term, f)
        };

        let updated = inc.try_update(&better, &model);
        assert!(updated);
        assert_eq!(inc.peek(), *better.fitness());

        // snapshot equals installed
        let snap = inc.snapshot();
        assert_eq!(snap, better);
    }

    #[test]
    fn test_try_update_accepts_feasible_and_debug_cost_matches_solution() {
        // This test ensures that in debug builds, the recomputed solution cost equals installed fitness cost.
        // Setup a feasible candidate with cost computed from model/evaluator.
        let problem = make_problem_one_flex();
        let (model, init_state) = make_model_and_initial_state(&problem);
        let inc = SharedIncumbent::new(init_state.clone());

        let im = model.index_manager();
        let r_ix = im.request_index(rid(10)).expect("request exists");
        let b_ix = im.berth_index(bid(1)).expect("berth exists");

        // Assign the only request at time 0 (inside window)
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r_ix.get()] = DecisionVar::assigned(b_ix, tp(0));

        let fitness = DefaultCostEvaluator.eval_fitness(&model, &dvars);
        assert_eq!(fitness.unassigned_requests, 0, "must be feasible");

        let term = TerminalOccupancy::new(problem.berths().iter());
        let candidate = SolverState::new(DecisionVarVec::from(dvars), term, fitness);

        let updated = inc.try_update(&candidate, &model);
        assert!(updated);

        // In both debug and release, the installed fitness equals peek and matches solution cost.
        let peek = inc.peek();
        assert_eq!(peek, fitness);

        // Double-check via full solution path
        let sol_cost = inc.snapshot().into_solution(&model).unwrap().cost();
        assert_eq!(peek.cost, sol_cost);
    }

    #[test]
    fn test_partial_eq_compares_underlying_states() {
        let problem = make_problem_two_flex();
        let (model, init_state) = make_model_and_initial_state(&problem);

        let a = SharedIncumbent::new(init_state.clone());
        let b = SharedIncumbent::new(init_state.clone());

        assert_eq!(a, b, "same initial state");

        // Improve b; then a != b
        let better = {
            let dv = DecisionVarVec::from_slice(init_state.decision_variables());
            let term = init_state.terminal_occupancy().clone();
            let f = Fitness::new(0, init_state.fitness().unassigned_requests - 1);
            SolverState::new(dv, term, f)
        };
        assert!(b.try_update(&better, &model));
        assert_ne!(a, b);
    }

    #[test]
    fn test_concurrent_updates_best_wins() {
        // Multiple threads attempt to improve; ensure final incumbent is the best.
        let problem = make_problem_one_flex();
        let (model, init_state) = make_model_and_initial_state(&problem);
        let incumbent = SharedIncumbent::new(init_state.clone());

        // Build several candidates; include a feasible optimal one.
        let im = model.index_manager();
        let r_ix = im.request_index(rid(10)).unwrap();
        let b_ix = im.berth_index(bid(1)).unwrap();

        // Feasible candidate (best by unassigned=0)
        let best_candidate = {
            let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
            dvars[r_ix.get()] = DecisionVar::assigned(b_ix, tp(0));
            let fit = DefaultCostEvaluator.eval_fitness(&model, &dvars);
            let term = TerminalOccupancy::new(problem.berths().iter());
            SolverState::new(DecisionVarVec::from(dvars), term, fit)
        };

        // Two worse candidates (infeasible, so arbitrary consistent fitness)
        let worse1 = {
            let dv = DecisionVarVec::from_slice(init_state.decision_variables());
            let term = init_state.terminal_occupancy().clone();
            let f = Fitness::new(50, init_state.fitness().unassigned_requests); // equal unassigned, higher cost
            SolverState::new(dv, term, f)
        };
        let worse2 = {
            let dv = DecisionVarVec::from_slice(init_state.decision_variables()); // all Unassigned
            let term = init_state.terminal_occupancy().clone();
            // same unassigned as init (1), arbitrary cost
            let f = Fitness::new(10, init_state.fitness().unassigned_requests);
            SolverState::new(dv, term, f)
        };

        let threads = vec![best_candidate.clone(), worse1, worse2];

        thread::scope(|scope| {
            let inc_ref = &incumbent;
            let model_ref = &model;

            for cand in threads {
                let inc_ref = inc_ref;
                let model_ref = model_ref;
                scope.spawn(move || {
                    // Try update from multiple threads
                    let _ = inc_ref.try_update(&cand, model_ref);
                });
            }
        });

        // Best must win: unassigned=0
        let final_fit = incumbent.peek();
        assert_eq!(final_fit.unassigned_requests, 0);
        // And snapshot equals the best candidate state we constructed
        let snap = incumbent.snapshot();
        assert_eq!(snap, best_candidate);
    }
}
