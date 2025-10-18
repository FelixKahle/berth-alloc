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

use berth_alloc_core::prelude::TimeDelta;
use rand::seq::SliceRandom;

use crate::{
    core::numeric::SolveNumeric,
    engine::{
        acceptor::{Acceptor, LexStrictAcceptor, RepairAcceptor},
        neighbors,
        search::{SearchContext, SearchStrategy},
    },
    model::solver_model::SolverModel,
    search::{
        operator::{DestroyOperator, LocalMoveOperator, RepairOperator},
        operator_library::{
            destroy::{
                RandomKRatioDestroy, ShawRelatedDestroy, StringBlockDestroy, TimeClusterDestroy,
                TimeWindowBandDestroy, WorstCostDestroy,
            },
            local::{
                CrossExchangeAcrossBerths, OrOptBlockRelocate, RelocateSingleBest,
                ShiftEarlierOnSameBerth, SwapPairSameBerth,
            },
            repair::{GreedyInsertion, KRegretInsertion, RandomizedGreedyInsertion},
        },
        planner::PlanningContext,
    },
    state::solver_state::{SolverState, SolverStateView},
};
use std::sync::atomic::Ordering as AtomicOrdering;

/// Periodic hard-refetch policy.
#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    /// Replace working state with the shared incumbent only if the incumbent is strictly better.
    IfBetter,
    /// Replace unconditionally on the period (for stronger convergence/sync).
    Always,
}

pub struct IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    destroy_ops: Vec<Box<dyn DestroyOperator<T, R>>>,
    repair_ops: Vec<Box<dyn RepairOperator<T, R>>>,
    local_ops: Vec<Box<dyn LocalMoveOperator<T, R>>>,

    local_acceptor: LexStrictAcceptor, // Phase A
    repair_acceptor: RepairAcceptor,   // Phase C

    max_local_steps: usize,

    /// After this many *outer* iterations without improvement, try refetching the
    /// shared incumbent (only if strictly better than our working state).
    /// 0 => **disabled**.
    refetch_after_stale: usize,

    /// Perform a *hard* refetch every N outer rounds.
    /// 0 => **disabled**.
    hard_refetch_every: usize,

    /// Policy for the periodic hard refetch.
    hard_refetch_mode: HardRefetchMode,
}

impl<T, R> Default for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    pub fn new() -> Self {
        Self {
            destroy_ops: Vec::new(),
            repair_ops: Vec::new(),
            local_ops: Vec::new(),
            local_acceptor: LexStrictAcceptor, // Phase A
            repair_acceptor: RepairAcceptor,   // Phase C
            max_local_steps: 64,
            refetch_after_stale: 8,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
        }
    }

    pub fn with_destroy_op(mut self, op: Box<dyn DestroyOperator<T, R>>) -> Self {
        self.destroy_ops.push(op);
        self
    }

    pub fn with_repair_op(mut self, op: Box<dyn RepairOperator<T, R>>) -> Self {
        self.repair_ops.push(op);
        self
    }

    pub fn with_local_op(mut self, op: Box<dyn LocalMoveOperator<T, R>>) -> Self {
        self.local_ops.push(op);
        self
    }

    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps;
        self
    }

    /// Set stale refetch threshold. 0 disables staleness-based refetch.
    pub fn with_refetch_after_stale(mut self, rounds: usize) -> Self {
        self.refetch_after_stale = rounds;
        self
    }

    /// Set periodic hard-refetch cadence. 0 disables periodic hard refetch.
    pub fn with_hard_refetch_every(mut self, period: usize) -> Self {
        self.hard_refetch_every = period;
        self
    }

    pub fn with_hard_refetch_mode(mut self, mode: HardRefetchMode) -> Self {
        self.hard_refetch_mode = mode;
        self
    }

    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    #[inline]
    fn maybe_apply_periodic_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        outer_rounds: usize,
    ) {
        if !self.should_hard_refetch(outer_rounds) {
            return;
        }
        let best_now = context.shared_incumbent().peek();
        let do_fetch = match self.hard_refetch_mode {
            HardRefetchMode::IfBetter => best_now < *current.fitness(),
            HardRefetchMode::Always => true,
        };
        if do_fetch {
            tracing::debug!(
                "ILS: periodic refetch at round {} (current {}, incumbent {})",
                outer_rounds,
                current.fitness(),
                best_now
            );
            *current = context.shared_incumbent().snapshot();
        }
    }

    #[inline]
    fn maybe_apply_stale_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        stale_rounds: usize,
    ) {
        if self.refetch_after_stale == 0 || stale_rounds < self.refetch_after_stale {
            return;
        }
        let best_now = context.shared_incumbent().peek();
        if best_now < *current.fitness() {
            tracing::debug!(
                "ILS: staleness refetch after {} stale rounds ({} -> {})",
                stale_rounds,
                current.fitness(),
                best_now
            );
            *current = context.shared_incumbent().snapshot();
        } else {
            tracing::trace!(
                "ILS: staleness refetch skipped; incumbent ({}) not better than current ({})",
                best_now,
                current.fitness()
            );
        }
    }
}

impl<T, R> SearchStrategy<T, R> for IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "Iterated Local Search"
    }

    #[tracing::instrument(name = "ILS Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let model = context.model();
        let stop = context.stop();

        if self.local_ops.is_empty() && (self.destroy_ops.is_empty() || self.repair_ops.is_empty())
        {
            tracing::warn!(
                "ILS: no operators configured (local={}, destroy={}, repair={})",
                self.local_ops.len(),
                self.destroy_ops.len(),
                self.repair_ops.len()
            );
            return;
        }

        // Seed working state from the global incumbent once at start.
        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();

        debug_assert_eq!(
            current.decision_variables().len(),
            model.flexible_requests_len(),
            "incumbent DV vector length must match model"
        );

        // Pre-allocate scratch buffer for PlanningContext.
        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // Bookkeeping.
        let mut stale_rounds = 0usize;
        let mut outer_rounds = 0usize;

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }

            // Periodic hard refetch policy.
            self.maybe_apply_periodic_refetch(&mut current, context, outer_rounds);

            // --------------- Phase A: Local improvement (first-improvement) ---------------
            let mut improved_in_round = false;

            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                let mut accepted_this_step = false;

                // Shuffle the visiting order of local operators
                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(context.rng()); // uses the strategy's RNG

                for &i in &order {
                    let op = &self.local_ops[i];

                    let mut pc = PlanningContext::new(model, &current, dv_buf.as_mut_slice());
                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        let mut tmp = current.clone();
                        tmp.apply_plan(plan);

                        if self.local_acceptor.accept(current.fitness(), tmp.fitness()) {
                            current = tmp;
                            accepted_this_step = true;
                            improved_in_round = true;

                            let _ = context.shared_incumbent().try_update(&current);
                            tracing::trace!("ILS: accepted local op {}", op.name());
                            break; // restart local climb from updated state
                        }
                    }
                }

                if !accepted_this_step {
                    break; // no improving local move found right now
                }
            }

            if !improved_in_round {
                // ---------------------- Phase B: Destroy + Phase C: Repair ----------------------
                if self.destroy_ops.is_empty() || self.repair_ops.is_empty() {
                    tracing::debug!("ILS: no perturbation operators configured; stopping.");
                    break 'outer;
                }

                let baseline = current.clone();

                // Try a random destroy operator (up to |destroy_ops| attempts).
                let mut destroyed = false;
                for _ in 0..self.destroy_ops.len() {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }
                    let idx = if self.destroy_ops.len() == 1 {
                        0
                    } else {
                        context.rng().random_range(0..self.destroy_ops.len())
                    };
                    let d = &self.destroy_ops[idx];

                    let mut pc = PlanningContext::new(model, &current, dv_buf.as_mut_slice());
                    if let Some(plan) = d.propose(&mut pc, context.rng()) {
                        current.apply_plan(plan);
                        destroyed = true;
                        tracing::trace!("ILS: applied destroy op {}", d.name());
                        break;
                    }
                }

                if !destroyed {
                    tracing::debug!("ILS: no destroy operator produced a plan; stopping.");
                    break 'outer;
                }

                // Try repairs; accept only if strictly better than the pre-destroy baseline.
                let mut repaired_and_accepted = false;
                for r in &self.repair_ops {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break 'outer;
                    }

                    let mut temp = current.clone();
                    let mut pc = PlanningContext::new(model, &temp, dv_buf.as_mut_slice());
                    if let Some(plan) = r.repair(&mut pc, context.rng()) {
                        temp.apply_plan(plan);

                        if self
                            .repair_acceptor
                            .accept(baseline.fitness(), temp.fitness())
                        {
                            current = temp;
                            repaired_and_accepted = true;

                            // Publish improvement vs baseline (may or may not beat global best).
                            let _ = context.shared_incumbent().try_update(&current);
                            tracing::trace!("ILS: accepted repair op {}", r.name());
                            break;
                        }
                    }
                }

                if repaired_and_accepted {
                    stale_rounds = 0;
                } else {
                    // Revert to baseline if repair couldn't beat it.
                    current = baseline;
                    stale_rounds = stale_rounds.saturating_add(1);
                    tracing::debug!(
                        "ILS: repair failed to beat baseline (stale_rounds={})",
                        stale_rounds
                    );
                }
            } else {
                stale_rounds = 0;
            }

            // Staleness-triggered refetch (only if incumbent strictly better).
            self.maybe_apply_stale_refetch(&mut current, context, stale_rounds);

            outer_rounds = outer_rounds.saturating_add(1);
        }

        // Final publish (harmless if we didn't beat the incumbent).
        let _ = context.shared_incumbent().try_update(&current);
    }
}

pub fn ils_strategy<T, R>(model: &SolverModel<T>) -> IteratedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    // Neighbor scopes
    let proximity_map = model.proximity_map();
    let neighbors_any = neighbors::any(proximity_map);
    let neighbors_direct_competitors = neighbors::direct_competitors(proximity_map);
    let neighbors_same_berth = neighbors::same_berth(proximity_map);

    // tuned for ~250 vessels, 15–20 berths, PT in [8, 20]
    IteratedLocalSearchStrategy::new()
        // cadence & refetch
        .with_max_local_steps(1024)
        .with_refetch_after_stale(128)
        .with_hard_refetch_every(0)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        // ------------------------- Phase A: Local improvement -------------------------
        .with_local_op(Box::new(ShiftEarlierOnSameBerth {
            // try a small random batch each step
            number_of_candidates_to_try_range: 8..=24,
        }))
        .with_local_op(Box::new(RelocateSingleBest {
            number_of_candidates_to_try_range: 8..=24,
        }))
        .with_local_op(Box::new(SwapPairSameBerth {
            number_of_pair_attempts_to_try_range: 10..=40,
        }))
        .with_local_op(Box::new(CrossExchangeAcrossBerths {
            number_of_pair_attempts_to_try_range: 12..=48,
        }))
        .with_local_op(Box::new(OrOptBlockRelocate::new(
            2..=4,     // block length to relocate
            1.4..=2.0, // RCL alpha (seed bias)
        )))
        // ---------------------- Phase B: Destroy (neighbor-aware) ---------------------
        // random subset (exploration)
        .with_destroy_op(Box::new(
            RandomKRatioDestroy::new(0.15..=0.55).with_neighbors(neighbors_any.clone()),
        ))
        // worst-cost (exploit)
        .with_destroy_op(Box::new(
            WorstCostDestroy::new(0.20..=0.40).with_neighbors(neighbors_direct_competitors.clone()),
        ))
        // time cluster around long job
        .with_destroy_op(Box::new(
            TimeClusterDestroy::<T>::new(0.20..=0.35, TimeDelta::new(24.into()))
                .with_alpha(1.6..=2.2)
                .with_neighbors(neighbors_any.clone()),
        ))
        // Shaw relatedness (temporal + berth penalty)
        .with_destroy_op(Box::new(
            ShawRelatedDestroy::new(
                0.20..=0.40, // ratio
                1.6..=2.2,   // alpha
                1.into(),    // |Δstart| weight
                1.into(),    // |Δend| weight
                4.into(),    // berth mismatch penalty
            )
            .with_neighbors(neighbors_same_berth.clone()),
        ))
        // contiguous block on a berth
        .with_destroy_op(Box::new(
            StringBlockDestroy::new(0.25..=0.45).with_alpha(1.4..=2.0),
        ))
        // time band around seed interval
        .with_destroy_op(Box::new(
            TimeWindowBandDestroy::<T>::new(0.30..=0.50, 1.4..=1.8, TimeDelta::new(12.into()))
                .with_neighbors(neighbors_any),
        ))
        // -------------------------- Phase C: Repair operators -------------------------
        .with_repair_op(Box::new(KRegretInsertion::new(4..=4))) // keep k=4 deterministically
        .with_repair_op(Box::new(RandomizedGreedyInsertion::new(1.6..=2.2)))
        .with_repair_op(Box::new(GreedyInsertion))
}
