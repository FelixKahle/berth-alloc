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
    model::index::{BerthIndex, RequestIndex},
    search::{
        eval::CostEvaluator,
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::SolverStateView,
    },
};
use berth_alloc_core::prelude::{Cost, TimeInterval};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

/// Relocates an adjacent pair of requests as a single block (OR-Opt 2).
///
/// This operator scans all pairs of assigned requests (r1, r2). If they are
/// adjacent on the same berth (end(r1) == start(r2)), it attempts to relocate
/// the entire contiguous block [r1, r2] to the earliest feasible position on
/// any berth (including the source berth), while respecting each request's
/// feasible window and berth permissions.
///
/// The operator yields at most one plan per call (first found feasible move).
#[derive(Debug, Default)]
pub struct RelocateAdjacentPairOp {
    i: usize, // index for r1
    j: usize, // index for r2
    k: usize, // index for destination berth
}

impl RelocateAdjacentPairOp {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, C, R> LocalSearchOperator<T, C, R> for RelocateAdjacentPairOp
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RelocateAdjacentPairOp"
    }

    fn reset(&mut self) {
        self.i = 0;
        self.j = 0;
        self.k = 0;
    }

    fn has_fragments(&self) -> bool {
        false
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let dvars = ctx.state().decision_variables();
        let n = dvars.len();
        let model = ctx.model();
        let num_berths = model.berths().len();

        while self.i < n {
            let r1 = RequestIndex::new(self.i);

            // Get r1 assignment and processing time on its source berth
            let (b_src, s1, pt1_src) = match dvars.get(r1.get()).copied() {
                Some(DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                })) => {
                    let pt1 = match model.processing_time(r1, berth_index) {
                        Some(pt) => pt,
                        None => {
                            // Should not happen for a valid assignment; skip r1
                            self.i += 1;
                            self.j = 0;
                            self.k = 0;
                            continue;
                        }
                    };
                    (berth_index, start_time, pt1)
                }
                _ => {
                    // Unassigned r1: skip to next i
                    self.i += 1;
                    self.j = 0;
                    self.k = 0;
                    continue;
                }
            };

            let s1_end = match s1.checked_add(pt1_src) {
                Some(t) => t,
                None => {
                    self.i += 1;
                    self.j = 0;
                    self.k = 0;
                    continue;
                }
            };

            while self.j < n {
                if self.j == self.i {
                    self.j += 1;
                    continue;
                }

                let r2 = RequestIndex::new(self.j);

                // Get r2 assignment
                let (b2, s2) = match dvars.get(r2.get()).copied() {
                    Some(DecisionVar::Assigned(Decision {
                        berth_index,
                        start_time,
                    })) => (berth_index, start_time),
                    _ => {
                        self.j += 1;
                        self.k = 0;
                        continue;
                    }
                };

                // Adjacency condition: same berth and r1.end == r2.start
                if b_src != b2 || s1_end != s2 {
                    self.j += 1;
                    self.k = 0;
                    continue;
                }

                // Feasible windows of r1, r2
                let w1 = model.feasible_interval(r1);
                let w2 = model.feasible_interval(r2);

                // Try relocating the block [r1, r2] onto berths (including possibly the source)
                while self.k < num_berths {
                    let b_dest = BerthIndex::new(self.k);
                    self.k += 1;

                    // Check both requests are allowed on b_dest and get processing times there
                    let pt1_dest = match model.processing_time(r1, b_dest) {
                        Some(pt) => pt,
                        None => continue,
                    };
                    let pt2_dest = match model.processing_time(r2, b_dest) {
                        Some(pt) => pt,
                        None => continue,
                    };

                    // Prepare a PlanBuilder
                    let mut pb = ctx.builder();
                    // Savepoint before any changes on this berth attempt
                    let sp0 = pb.savepoint();

                    // Unassign both requests to free their slots in the sandbox
                    if pb.propose_unassignment(r1).is_err() || pb.propose_unassignment(r2).is_err()
                    {
                        // Roll back and skip this berth
                        pb.undo_to(sp0);
                        continue;
                    }

                    // Savepoint right after unassignments; we will revert to this between slot attempts
                    let sp_unassigned = pb.savepoint();

                    // Compute a conservative search window for the new first start:
                    // s_new_1 >= w1.start
                    // s_new_1 >= w2.start - pt1_dest  (so that s_new_2 = s_new_1 + pt1_dest >= w2.start)
                    // and s_new_1 <= min(w1.end, w2.end) (we'll additionally enforce block length in checks)
                    let mut lower_bound = w1.start();
                    if let Some(lb2) = w2.start().checked_sub(pt1_dest)
                        && lb2 > lower_bound
                    {
                        lower_bound = lb2;
                    }
                    let upper_cap = if w1.end() < w2.end() {
                        w1.end()
                    } else {
                        w2.end()
                    };

                    // If the window is degenerate, no possible placement
                    if lower_bound >= upper_cap {
                        pb.undo_to(sp0);
                        continue;
                    }

                    let search_window = TimeInterval::new(lower_bound, upper_cap);

                    // Collect free intervals first to avoid borrowing pb immutably while mutating it
                    let slots: Vec<_> = pb
                        .iter_free_for_in_berths([b_dest], search_window)
                        .collect();

                    for fb in slots {
                        // Earliest viable start within this free interval
                        let fb_start = fb.interval().start();
                        let mut s_new_1 = fb_start;
                        if lower_bound > s_new_1 {
                            s_new_1 = lower_bound;
                        }

                        // Compute consecutive ends for the block
                        let s_new_mid = match s_new_1.checked_add(pt1_dest) {
                            Some(t) => t,
                            None => continue,
                        };
                        let s_new_end = match s_new_mid.checked_add(pt2_dest) {
                            Some(t) => t,
                            None => continue,
                        };

                        // Must fit into the free interval and both windows
                        if s_new_end > fb.interval().end() {
                            continue;
                        }
                        // r1 interval
                        let iv1_new = TimeInterval::new(s_new_1, s_new_mid);
                        // r2 interval
                        let iv2_new = TimeInterval::new(s_new_mid, s_new_end);

                        if !w1.contains_interval(&iv1_new) || !w2.contains_interval(&iv2_new) {
                            continue;
                        }

                        // Skip no-op (same berth and same start time as original block)
                        if b_dest == b_src && s_new_1 == s1 {
                            continue;
                        }

                        // Re-validate exact free berth slices for both placements.
                        // Put the iterator in a temporary block so its borrow ends
                        let fb1_new_opt =
                            { pb.iter_free_for_on_berth_in(r1, b_dest, iv1_new).next() };
                        let fb2_new_opt =
                            { pb.iter_free_for_on_berth_in(r2, b_dest, iv2_new).next() };

                        let (Some(fb1_new), Some(fb2_new)) = (fb1_new_opt, fb2_new_opt) else {
                            continue;
                        };

                        // Attempt assignments transactionally
                        let mut ok = true;
                        if pb.propose_assignment(r1, s_new_1, &fb1_new).is_err() {
                            ok = false;
                        }
                        if ok && pb.propose_assignment(r2, s_new_mid, &fb2_new).is_err() {
                            ok = false;
                        }

                        if ok {
                            // Success: return the plan
                            return Some(pb.finalize());
                        } else {
                            // Reset to unassigned state and try next free slot
                            pb.undo_to(sp_unassigned);
                        }
                    }

                    // No slot on this destination berth worked; undo to original state
                    pb.undo_to(sp0);
                }

                // Exhausted berths for this r1-r2 pair
                self.j += 1;
                self.k = 0;
            }

            // Exhausted r2 for current r1
            self.i += 1;
            self.j = 0;
            self.k = 0;
        }

        None
    }
}
