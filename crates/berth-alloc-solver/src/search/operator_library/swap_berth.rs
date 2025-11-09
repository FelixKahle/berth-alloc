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
        neighboors::NeighborFn,
        operator::{LocalSearchOperator, OperatorContext},
        planner::PlanBuilder,
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::SolverStateView,
        terminal::terminalocc::FreeBerth,
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

enum CurrentList<'n> {
    None,
    Slice {
        slice: &'n [RequestIndex],
        pos: usize,
    },
    Vec {
        vec: Vec<RequestIndex>,
        pos: usize,
    },
    Full {
        pos: usize,
        n: usize,
    },
}
impl<'n> CurrentList<'n> {
    #[inline]
    fn reset(&mut self) {
        *self = CurrentList::None;
    }
    #[inline]
    fn from_optional_neighbors(i: RequestIndex, nf: &Option<NeighborFn<'n>>, n: usize) -> Self {
        match nf {
            Some(NeighborFn::Slice(f)) => CurrentList::Slice {
                slice: f(i),
                pos: 0,
            },
            Some(NeighborFn::Vec(f)) => CurrentList::Vec { vec: f(i), pos: 0 },
            None => CurrentList::Full {
                pos: i.get().saturating_add(1),
                n,
            },
        }
    }
}
impl<'n> Iterator for CurrentList<'n> {
    type Item = RequestIndex;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CurrentList::None => None,
            CurrentList::Slice { slice, pos } => {
                if *pos >= slice.len() {
                    return None;
                }
                let v = slice[*pos];
                *pos += 1;
                Some(v)
            }
            CurrentList::Vec { vec, pos } => {
                if *pos >= vec.len() {
                    return None;
                }
                let v = vec[*pos];
                *pos += 1;
                Some(v)
            }
            CurrentList::Full { pos, n } => {
                if *pos >= *n {
                    return None;
                }
                let v = RequestIndex::new(*pos);
                *pos += 1;
                Some(v)
            }
        }
    }
}

/// Swaps two assigned requests (r1, r2) between their berths (b1, b2).
///
/// This operator finds the *earliest* feasible slot for r1 on b2 and
/// the *earliest* feasible slot for r2 on b1.
///
/// This is different from `SwapSlotOp`, which swaps their exact start times.
pub struct SwapBerthOp<'n> {
    i: usize,
    current: CurrentList<'n>,
    neighbor_function: Option<NeighborFn<'n>>,
}

impl<'n> Default for SwapBerthOp<'n> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'n> SwapBerthOp<'n> {
    #[inline]
    pub fn new() -> Self {
        Self {
            i: 0,
            current: CurrentList::None,
            neighbor_function: None,
        }
    }

    #[inline]
    pub fn with_neighbors(neigh: NeighborFn<'n>) -> Self {
        Self {
            i: 0,
            current: CurrentList::None,
            neighbor_function: Some(neigh),
        }
    }

    /// Helper to find the earliest fitting slot for a request on a target berth.
    #[inline]
    fn find_earliest_slot<'b, 'c, 't, 'm, 'p, T, C>(
        pb: &PlanBuilder<'b, 'c, 't, 'm, 'p, T, C>,
        r: RequestIndex,
        target_berth: crate::model::index::BerthIndex,
    ) -> Option<(TimePoint<T>, FreeBerth<T>)>
    where
        T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        C: CostEvaluator<T>,
    {
        let model = pb.model();
        let pt = model.processing_time(r, target_berth)?;

        pb.iter_free_for_on_berth(r, target_berth)
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

impl<'n, T, C, R> LocalSearchOperator<T, C, R> for SwapBerthOp<'n>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "SwapBerthOp"
    }

    fn reset(&mut self) {
        self.i = 0;
        self.current.reset();
    }

    fn has_fragments(&self) -> bool {
        false
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let model = ctx.model();
        let dvars = ctx.state().decision_variables();
        let n = dvars.len();

        while self.i < n {
            // Initialize candidate list for r1 if needed
            if matches!(self.current, CurrentList::None) {
                let r1 = RequestIndex::new(self.i);
                self.current = CurrentList::from_optional_neighbors(r1, &self.neighbor_function, n);
            }

            let r1 = RequestIndex::new(self.i);

            // r1 must be assigned
            let b1 = match dvars.get(r1.get()).copied() {
                Some(DecisionVar::Assigned(Decision { berth_index, .. })) => berth_index,
                _ => {
                    self.i += 1;
                    self.current.reset();
                    continue;
                }
            };

            for r2 in self.current.by_ref() {
                if r2.get() >= n || r2 == r1 {
                    continue;
                }

                // r2 must be assigned
                let b2 = match dvars.get(r2.get()).copied() {
                    Some(DecisionVar::Assigned(Decision { berth_index, .. })) => berth_index,
                    _ => continue,
                };

                // This operator swaps *between* berths
                if b1 == b2 {
                    continue;
                }

                // allowed-berth pruning
                let a1 = model.allowed_berth_indices(r1);
                let a2 = model.allowed_berth_indices(r2);
                if !a1.contains(&b2) || !a2.contains(&b1) {
                    continue;
                }

                // Build plan atomically with rollback on failure.
                let mut pb = ctx.builder();
                let sp = pb.savepoint();

                // Make the two holes first
                if pb.propose_unassignment(r1).is_err() {
                    pb.undo_to(sp);
                    continue;
                }
                if pb.propose_unassignment(r2).is_err() {
                    pb.undo_to(sp);
                    continue;
                }

                // Find earliest slot for r1 on b2
                let slot1_opt = Self::find_earliest_slot(&pb, r1, b2);
                // Find earliest slot for r2 on b1
                let slot2_opt = Self::find_earliest_slot(&pb, r2, b1);

                let ((s1_new, fb1), (s2_new, fb2)) = match (slot1_opt, slot2_opt) {
                    (Some(a), Some(b)) => (a, b),
                    _ => {
                        // One or both requests couldn't find a new slot
                        pb.undo_to(sp);
                        continue;
                    }
                };

                // Commit the two new assignments
                if pb.propose_assignment(r1, s1_new, &fb1).is_err() {
                    pb.undo_to(sp);
                    continue;
                }
                if pb.propose_assignment(r2, s2_new, &fb2).is_err() {
                    pb.undo_to(sp);
                    continue;
                }

                // Yield the neighbor
                return Some(pb.finalize());
            }

            // exhausted r2 candidates for this r1
            self.i += 1;
            self.current.reset();
        }

        None
    }
}
