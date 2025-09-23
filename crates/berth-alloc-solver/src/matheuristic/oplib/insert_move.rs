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

use crate::matheuristic::operator::Operator;
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::problem::asg::AssignmentView;
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::{IteratorRandom, SliceRandom};
use rand_chacha::ChaCha8Rng;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct InsertMoveOperator<T> {
    pub try_randomize: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for InsertMoveOperator<T> {
    fn default() -> Self {
        Self {
            try_randomize: true,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for InsertMoveOperator<T>
where
    T: Copy
        + Ord
        + Send
        + Sync
        + std::fmt::Debug
        + CheckedAdd
        + CheckedSub
        + Mul<Output = Cost>
        + Into<Cost>,
{
    type Time = T;

    fn name(&self) -> &'static str {
        "InsertMove"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut placed = false;

        let res = ctx.with_builder(|builder| {
            let victim = builder.with_explorer(|ex| ex.iter_assigned_requests().choose(rng));
            let Some(v) = victim else {
                return;
            };
            let rid = v.asg().request_id();

            if builder.propose_unassignment(&v).is_err() {
                return;
            }

            let req = builder
                .with_explorer(|ex| ex.iter_unassigned_requests().find(|r| r.req().id() == rid));
            let Some(req) = req else {
                return;
            };

            let mut candidates: Vec<_> = builder.with_explorer(|ex| {
                let mut out = Vec::new();
                for fb in ex.iter_free_for(req.clone()) {
                    let bid = fb.berth().id();
                    if let Some(pt) = req.req().processing_time_for(bid) {
                        let w = req.req().feasible_window();
                        let iv = *fb.interval();
                        let lo = std::cmp::max(iv.start(), w.start());
                        let hi_iv = iv.end().checked_sub(pt);
                        let hi_w = w.end().checked_sub(pt);
                        if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                out.push((fb.clone(), lo));
                                if hi != lo {
                                    out.push((fb.clone(), hi));
                                }
                            }
                        }
                    }
                }
                out
            });

            if self.try_randomize {
                candidates.shuffle(rng);
            }

            for (free, start) in candidates {
                if builder
                    .propose_assignment(req.clone(), start, &free)
                    .is_ok()
                {
                    placed = true;
                    break;
                }
            }
        });

        match (placed, res) {
            (true, Ok(plan)) => Some(plan),
            _ => None,
        }
    }
}
