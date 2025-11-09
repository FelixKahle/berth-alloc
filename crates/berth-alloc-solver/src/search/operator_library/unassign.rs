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
    },
    state::{decisionvar::DecisionVar, plan::Plan, solver_state::SolverStateView},
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

/// Unassigns an assigned request.
///
/// This operator iterates through all requests. For the first *assigned*
/// request it finds, it yields a plan to move it to the unassigned state.
///
/// This is typically a "shake" or diversification move, as it
/// increases the unassigned count.
#[derive(Debug, Default)]
pub struct UnassignOp {
    /// The index of the next request to check for assignment.
    i: usize,
}

impl UnassignOp {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, C, R> LocalSearchOperator<T, C, R> for UnassignOp
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "UnassignOp"
    }

    fn reset(&mut self) {
        self.i = 0;
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

        while self.i < n {
            let r = RequestIndex::new(self.i);
            self.i += 1;

            // Skip if this request is already unassigned
            if let Some(DecisionVar::Unassigned) = dvars.get(r.get()) {
                continue;
            }

            // This request is Assigned. Try to unassign it.
            let mut pb = ctx.builder();
            let sp = pb.savepoint();

            if pb.propose_unassignment(r).is_ok() {
                // Yield the unassignment plan.
                return Some(pb.finalize());
            }

            // Unassignment failed (this shouldn't happen if it was assigned).
            // Discard changes and try the next request.
            pb.undo_to(sp);
        }

        None
    }
}
