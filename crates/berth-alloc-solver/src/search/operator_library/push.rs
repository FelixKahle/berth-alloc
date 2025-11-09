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
    model::index::RequestIndex,
    search::{
        eval::CostEvaluator,
        operator::{LocalSearchOperator, OperatorContext},
        planner::PlanBuilder,
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::Plan,
        solver_state::SolverStateView,
        terminal::terminalocc::FreeBerth,
    },
};
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

/// Implements a "cascade" or "push" insertion.
///
/// This operator iterates through each unassigned request (`r_insert`).
/// For each, it iterates through each assigned request (`r_block`).
/// It attempts to:
/// 1. Unassign `r_block`.
/// 2. Assign `r_insert` into `r_block`'s *exact* previous slot.
/// 3. Assign `r_block` into the *earliest available* slot on the same
///    berth, starting *after* `r_insert`'s new position.
///
/// This creates a 2-request "cascade" and results in one fewer
/// unassigned request.
#[derive(Debug, Default)]
pub struct PushInsertOp {
    /// Index for the unassigned request (`r_insert`).
    i: usize,
    /// Index for the assigned request to be "pushed" (`r_block`).
    j: usize,
}

impl PushInsertOp {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Helper to find the earliest fitting slot for a request on a target berth
    /// within a specified search window.
    #[inline]
    fn find_earliest_slot_in<'b, 'c, 't, 'm, 'p, T, C>(
        pb: &PlanBuilder<'b, 'c, 't, 'm, 'p, T, C>,
        r: RequestIndex,
        target_berth: crate::model::index::BerthIndex,
        search_window: TimeInterval<T>,
    ) -> Option<(TimePoint<T>, FreeBerth<T>)>
    where
        T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        C: CostEvaluator<T>,
    {
        let model = pb.model();
        let pt = model.processing_time(r, target_berth)?;

        pb.iter_free_for_on_berth_in(r, target_berth, search_window)
            .filter_map(|fb| {
                let start = fb.interval().start();
                let end = start.checked_add(pt)?;
                if end <= fb.interval().end() {
                    Some((start, fb))
                } else {
                    None
                }
            })
            .min_by_key(|(start, _)| *start)
    }
}

impl<T, C, R> LocalSearchOperator<T, C, R> for PushInsertOp
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "PushInsertOp"
    }

    fn reset(&mut self) {
        self.i = 0;
        self.j = 0;
    }

    fn has_fragments(&self) -> bool {
        false
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let dvars = ctx.state().decision_variables();
        let n = dvars.len();
        let model = ctx.model();

        while self.i < n {
            let r_insert = RequestIndex::new(self.i);

            // 1. `r_insert` must be unassigned.
            if dvars.get(r_insert.get())?.is_assigned() {
                self.i += 1;
                self.j = 0;
                continue;
            }

            while self.j < n {
                if self.i == self.j {
                    self.j += 1;
                    continue;
                }
                let r_block = RequestIndex::new(self.j);

                // 2. `r_block` must be assigned.
                let (b_target, s_target) = match dvars.get(r_block.get()).copied() {
                    Some(DecisionVar::Assigned(Decision {
                        berth_index,
                        start_time,
                    })) => (berth_index, start_time),
                    _ => {
                        self.j += 1;
                        continue;
                    }
                };

                // Increment `j` *before* any potential 'continue'
                self.j += 1;

                // 3. `r_insert` must be allowed on `b_target`.
                if !model.allowed_berth_indices(r_insert).contains(&b_target) {
                    continue;
                }

                // 4. `r_insert` must be feasible at `s_target`.
                let pt_insert = model.processing_time(r_insert, b_target)?;
                let w_insert = model.feasible_interval(r_insert);
                let iv_insert_new = TimeInterval::new(s_target, s_target.checked_add(pt_insert)?);
                if !w_insert.contains_interval(&iv_insert_new) {
                    continue;
                }

                // 5. Get `r_block`'s info for re-insertion.
                let w_block = model.feasible_interval(r_block);

                // --- Start Transaction ---
                let mut pb = ctx.builder();
                let sp = pb.savepoint();

                // 6. Unassign `r_block`. This fails if `r_block` wasn't assigned in the builder's
                //    view (which it is), so we unwrap.
                let fb_block_old = match pb.propose_unassignment(r_block) {
                    Ok(fb) => fb,
                    Err(_) => {
                        pb.undo_to(sp);
                        continue;
                    }
                };

                // Sanity check: old slot must match s_target
                if fb_block_old.interval().start() != s_target {
                    pb.undo_to(sp);
                    continue;
                }

                // 7. Find the exact free slot for `r_insert`.
                // We use iter_free_for_on_berth_in to be robust, even though
                // we "know" the slot [s_target, s_target + pt_block_old] is free.
                let fb_insert_opt = pb
                    .iter_free_for_on_berth_in(r_insert, b_target, iv_insert_new)
                    .next();

                let fb_insert = match fb_insert_opt {
                    Some(fb) if fb.interval().contains_interval(&iv_insert_new) => fb,
                    _ => {
                        pb.undo_to(sp);
                        continue;
                    } // Slot not actually free or doesn't fit
                };

                // 8. Assign `r_insert` at `s_target`.
                if pb
                    .propose_assignment(r_insert, s_target, &fb_insert)
                    .is_err()
                {
                    pb.undo_to(sp);
                    continue;
                }

                // 9. Find new slot for `r_block`.
                // Search window starts right after `r_insert` finishes.
                let search_start_block = iv_insert_new.end();
                let search_window_block = match w_block
                    .intersection(&TimeInterval::new(search_start_block, w_block.end()))
                {
                    Some(iv) => iv,
                    None => {
                        pb.undo_to(sp);
                        continue;
                    } // No feasible window remains
                };

                let slot_block_opt =
                    Self::find_earliest_slot_in(&pb, r_block, b_target, search_window_block);

                if let Some((s_block_new, fb_block_new)) = slot_block_opt {
                    // 10. Assign `r_block` to its new slot.
                    if pb
                        .propose_assignment(r_block, s_block_new, &fb_block_new)
                        .is_ok()
                    {
                        // Success!
                        return Some(pb.finalize());
                    }
                }

                // Failed to re-assign `r_block`. Rollback entire operation.
                pb.undo_to(sp);
            }

            // Exhausted `j` for this `i`.
            self.i += 1;
            self.j = 0;
        }

        None
    }
}
