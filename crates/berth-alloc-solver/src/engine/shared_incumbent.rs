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

use parking_lot::Mutex;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use crate::state::{
    fitness::Fitness,
    solver_state::{SolverState, SolverStateView},
};

#[derive(Debug)]
pub struct SharedIncumbent<'p, T>
where
    T: Copy + Ord,
{
    best_state: Mutex<SolverState<'p, T>>,
    best_unassigned: AtomicUsize, // Avoid locking for simple reads
    best_cost: AtomicI64,         // Avoid locking for simple reads
}

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

    /// Attempt to install an owned candidate as the new incumbent.
    /// Uses a quick non-blocking pre-check via `peek()`, then re-validates under the lock.
    /// Prefers `<` on `Fitness` (unassigned primary, cost secondary).
    #[tracing::instrument(level = "debug", skip(self, candidate_state))]
    pub fn try_update_owned(&self, candidate_state: SolverState<'p, T>) -> bool {
        // Extract fields *before* moving the candidate to avoid borrow-after-move issues.
        let candidate_fitness_ref = candidate_state.fitness();
        let candidate_unassigned_count = candidate_fitness_ref.unassigned_requests;
        let candidate_cost_value = candidate_fitness_ref.cost;

        // Fast pre-check using atomics (best-effort, race-tolerant).
        let best_atomic_snapshot_fitness = self.peek();
        let candidate_fitness_value =
            Fitness::new(candidate_cost_value, candidate_unassigned_count);
        if (candidate_fitness_value >= best_atomic_snapshot_fitness) {
            return false;
        }

        // Definitive check under the lock.
        let mut best_state_guard = self.best_state.lock();
        let current_best_fitness_locked = best_state_guard.fitness();

        if candidate_fitness_value < *current_best_fitness_locked {
            *best_state_guard = candidate_state; // move, no clone
            self.best_unassigned
                .store(candidate_unassigned_count, Ordering::Release);
            self.best_cost
                .store(candidate_cost_value, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Attempt to install a borrowed candidate as the new incumbent.
    /// Uses the same logic as the owned version, but clones on install.
    #[tracing::instrument(level = "debug", skip(self, candidate_state))]
    pub fn try_update(&self, candidate_state: &SolverState<'p, T>) -> bool
    where
        SolverState<'p, T>: Clone,
    {
        let candidate_fitness_ref = candidate_state.fitness();
        let candidate_fitness_value = candidate_fitness_ref.clone(); // Fitness is cheap to clone

        // Fast pre-check using atomics (best-effort, race-tolerant).
        let best_atomic_snapshot_fitness = self.peek();
        if (candidate_fitness_value >= best_atomic_snapshot_fitness) {
            return false;
        }

        // Definitive check under the lock.
        let mut best_state_guard = self.best_state.lock();
        let current_best_fitness_locked = best_state_guard.fitness().clone();

        if candidate_fitness_value < current_best_fitness_locked {
            *best_state_guard = candidate_state.clone();
            self.best_unassigned.store(
                candidate_fitness_value.unassigned_requests,
                Ordering::Release,
            );
            self.best_cost
                .store(candidate_fitness_value.cost, Ordering::Release);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{registry::ledger::Ledger, terminal::terminalocc::TerminalOccupancy};
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::req::RequestView};
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
        // r1 pt=10 on b1, r2 pt=5 on b1 (same weight=1)
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    // Build a SolverState with given set of assigned flexible request IDs at given starts.
    // Requests are assumed to be feasible and non-overlapping.
    fn state_with_assignments<'p>(
        prob: &'p Problem<i64>,
        assigns: &[(u32, i64)], // (req_id, start)
    ) -> SolverState<'p, i64> {
        let mut ledger = Ledger::new(prob);
        // Borrow berths from the problem so the occupancy lives as long as `prob`
        let terminal_occupancy = TerminalOccupancy::new(prob.iter_berths());

        for (rid_u32, start) in assigns {
            let req = prob
                .flexible_requests()
                .get(RequestIdentifier::new(*rid_u32))
                .expect("request exists");
            let b1 = prob
                .berths()
                .get(BerthIdentifier::new(1))
                .expect("berth exists");
            ledger
                .commit_assignment(req, b1, TimePoint::new(*start))
                .expect("commit must succeed");
        }

        SolverState::new(ledger, terminal_occupancy)
    }

    fn assigned_ids_vec(state: &SolverState<'_, i64>) -> Vec<RequestIdentifier> {
        state
            .ledger()
            .iter_assigned_requests()
            .map(|r| r.id())
            .collect()
    }

    // ---------- borrowed-path tests (baseline) ----------

    #[test]
    fn test_peek_reflects_initial_state() {
        let prob = problem_one_berth_two_flex();
        let initial = state_with_assignments(&prob, &[]); // 0 assigned => 2 unassigned, cost 0
        let incumbent = SharedIncumbent::new(initial);

        let fitness_snapshot = incumbent.peek();
        assert_eq!(fitness_snapshot.unassigned_requests, 2);
        assert_eq!(fitness_snapshot.cost, 0);
    }

    #[test]
    fn test_try_update_rejects_worse_unassigned_borrowed() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: 1 assigned (rid 2) => 1 unassigned, lower cost
        let inc_state = state_with_assignments(&prob, &[(2, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: 0 assigned => 2 unassigned (worse)
        let candidate = state_with_assignments(&prob, &[]);

        assert!(!incumbent.try_update(&candidate));
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 1);
        assert!(peek.cost > 0);
    }

    #[test]
    fn test_try_update_rejects_higher_cost_same_unassigned_borrowed() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: assign cheaper (rid 2, pt=5) => 1 unassigned, lower cost
        let inc_state = state_with_assignments(&prob, &[(2, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: assign more expensive (rid 1, pt=10) => 1 unassigned, higher cost
        let candidate = state_with_assignments(&prob, &[(1, 0)]);

        assert!(!incumbent.try_update(&candidate));
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 1);
        let cand_cost = candidate.cost();
        assert!(peek.cost < cand_cost);
    }

    #[test]
    fn test_try_update_accepts_lower_cost_same_unassigned_borrowed() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: assign expensive (rid 1) => 1 unassigned, higher cost
        let inc_state = state_with_assignments(&prob, &[(1, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: assign cheaper (rid 2) => 1 unassigned, lower cost
        let candidate = state_with_assignments(&prob, &[(2, 0)]);

        assert!(incumbent.try_update(&candidate));

        // Snapshot should reflect candidate’s assignment set
        let snapshot = incumbent.snapshot();
        assert_eq!(assigned_ids_vec(&snapshot), vec![rid(2)]);

        // Peek should match candidate’s fitness
        let peek = incumbent.peek();
        assert_eq!(
            peek.unassigned_requests,
            candidate.fitness().unassigned_requests
        );
        assert_eq!(peek.cost, candidate.fitness().cost);
    }

    #[test]
    fn test_try_update_accepts_fewer_unassigned_even_if_cost_higher_borrowed() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: 1 assigned (rid 2) => cost low, 1 unassigned
        let inc_state = state_with_assignments(&prob, &[(2, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: 2 assigned (rid 1 at 0, rid 2 at 20) => 0 unassigned, cost higher
        let candidate = state_with_assignments(&prob, &[(1, 0), (2, 20)]);

        assert!(incumbent.try_update(&candidate));
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 0);
        assert_eq!(assigned_ids_vec(&incumbent.snapshot()).len(), 2);
    }

    // ---------- owned-path tests (focus) ----------

    #[test]
    fn test_try_update_owned_rejects_equal_fitness() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: assign rid 2 at 0 => 1 unassigned
        let inc_state = state_with_assignments(&prob, &[(2, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate with identical assignment set (equal fitness)
        let candidate = state_with_assignments(&prob, &[(2, 0)]);

        assert!(!incumbent.try_update_owned(candidate));
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 1);
    }

    #[test]
    fn test_try_update_owned_rejects_worse_unassigned() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: 1 assigned (rid 2) => 1 unassigned
        let inc_state = state_with_assignments(&prob, &[(2, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: 0 assigned => 2 unassigned (worse)
        let candidate = state_with_assignments(&prob, &[]);

        assert!(!incumbent.try_update_owned(candidate));
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 1);
    }

    #[test]
    fn test_try_update_owned_accepts_lower_cost_same_unassigned() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: assign expensive (rid 1) => 1 unassigned, higher cost
        let inc_state = state_with_assignments(&prob, &[(1, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: assign cheaper (rid 2) => 1 unassigned, lower cost
        let candidate = state_with_assignments(&prob, &[(2, 0)]);

        assert!(incumbent.try_update_owned(candidate));
        let snapshot = incumbent.snapshot();
        assert_eq!(assigned_ids_vec(&snapshot), vec![rid(2)]);
    }

    #[test]
    fn test_try_update_owned_accepts_fewer_unassigned_even_if_cost_higher() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: 1 assigned (rid 2) => 1 unassigned
        let inc_state = state_with_assignments(&prob, &[(2, 0)]);
        let incumbent = SharedIncumbent::new(inc_state);

        // Candidate: assign both => 0 unassigned (better on primary objective)
        let candidate = state_with_assignments(&prob, &[(1, 0), (2, 20)]);

        assert!(incumbent.try_update_owned(candidate));
        let snapshot = incumbent.snapshot();
        assert_eq!(assigned_ids_vec(&snapshot).len(), 2);
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 0);
    }

    #[test]
    fn test_try_update_owned_basic_update_and_peek_consistency() {
        let prob = problem_one_berth_two_flex();

        // Incumbent: none assigned => 2 unassigned
        let initial = state_with_assignments(&prob, &[]);
        let incumbent = SharedIncumbent::new(initial);

        // Candidate: assign one => 1 unassigned
        let candidate = state_with_assignments(&prob, &[(2, 0)]);

        assert!(incumbent.try_update_owned(candidate));
        let peek = incumbent.peek();
        assert_eq!(peek.unassigned_requests, 1);
        assert!(peek.cost > 0);
    }
}
