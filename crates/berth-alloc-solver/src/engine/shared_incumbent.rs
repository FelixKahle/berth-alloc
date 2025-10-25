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
        if candidate_fitness_value >= best_atomic_snapshot_fitness {
            return false;
        }

        // Definitive check under the lock.
        let mut best_state_guard = self.best_state.lock();
        let current_best_fitness_locked = best_state_guard.fitness();

        if candidate_fitness_value < *current_best_fitness_locked {
            tracing::debug!(
                "New incumbent found: old fitness={}, new fitness={}",
                current_best_fitness_locked,
                candidate_fitness_value
            );

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
    pub fn try_update(
        &self,
        candidate_state: &SolverState<'p, T>,
        model: &SolverModel<'p, T>,
    ) -> bool
    where
        T: SolveNumeric,
        SolverState<'p, T>: Clone,
    {
        // Snapshot first to avoid TOCTOU with a mutating candidate.
        let snapshot = candidate_state.clone();
        let candidate_fitness_value = *snapshot.fitness(); // Fitness is cheap to clone

        // Fast pre-check using atomics (best-effort, race-tolerant).
        let best_atomic_snapshot_fitness = self.peek();
        if candidate_fitness_value >= best_atomic_snapshot_fitness {
            return false;
        }

        // Definitive check under the lock.
        let mut best_state_guard = self.best_state.lock();
        let current_best_fitness_locked = *best_state_guard.fitness();

        if candidate_fitness_value < current_best_fitness_locked {
            tracing::info!(
                "New incumbent found: old fitness={}, new fitness={}",
                current_best_fitness_locked,
                candidate_fitness_value
            );

            // Install exactly the snapshot we evaluated/logged.
            *best_state_guard = snapshot;
            self.best_unassigned.store(
                candidate_fitness_value.unassigned_requests,
                Ordering::Release,
            );
            self.best_cost
                .store(candidate_fitness_value.cost, Ordering::Release);

            #[cfg(debug_assertions)]
            {
                use berth_alloc_model::prelude::SolutionView;

                let installed = best_state_guard.fitness();
                debug_assert_eq!(
                    *installed, candidate_fitness_value,
                    "installed state fitness differs from atomics/logs"
                );

                if installed.unassigned_requests == 0 {
                    let true_calculated_cost =
                        candidate_state.clone().into_solution(model).unwrap().cost();
                    debug_assert_eq!(
                        installed.cost, true_calculated_cost,
                        "installed state cost differs from recalculated solution cost"
                    );
                }
            }

            true
        } else {
            false
        }
    }
}
