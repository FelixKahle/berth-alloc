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

use std::ops::Mul;

use berth_alloc_core::prelude::{Cost, TimePoint};
use berth_alloc_model::problem::asg::AssignmentView;
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::SliceRandom;

use crate::meta::operator::Operator;

#[derive(Debug, Clone)]
pub struct SwapPairOperator<T> {
    pub attempts_per_call: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for SwapPairOperator<T> {
    fn default() -> Self {
        Self {
            attempts_per_call: 10,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for SwapPairOperator<T>
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
        "SwapPair"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let aps = self.attempts_per_call.max(1);

        // helper: given (req, free window), compute a feasible start (prefer right-packed/hi).
        let mut pick_start = |builder: &crate::framework::planning::PlanBuilder<'_, 'p, T>,
                              req: crate::framework::planning::BrandedRequest<
            '_,
            'p,
            berth_alloc_model::common::FlexibleKind,
            T,
        >,
                              free: &crate::framework::planning::BrandedFreeBerth<'_, 'p, T>|
         -> Option<TimePoint<T>> {
            builder.with_explorer(|ex| {
                let r = req.req();
                let w = r.feasible_window();
                let bid = free.berth().id();
                let Some(pt) = r.processing_time_for(bid) else {
                    return None;
                };
                let iv = *free.interval();
                let lo = std::cmp::max(iv.start(), w.start());
                let hi_iv = iv.end().checked_sub(pt)?;
                let hi_w = w.end().checked_sub(pt)?;
                let hi = std::cmp::min(hi_iv, hi_w);
                if lo <= hi { Some(hi) } else { None }
            })
        };

        // Try a few random pairs; stop on first successful full swap.
        for _ in 0..aps {
            let plan = ctx.with_builder(|builder| {
                // sample two distinct assigned
                let mut pool: Vec<_> =
                    builder.with_explorer(|ex| ex.iter_assigned_requests().collect());
                if pool.len() < 2 {
                    return;
                }
                pool.shuffle(rng);
                let a = pool[0].clone();
                let b = pool[1].clone();

                // read ids first (immutable)
                let (rid_a, rid_b) =
                    builder.with_explorer(|ex| (a.asg().request_id(), b.asg().request_id()));

                // unassign both -> we get branded free windows fa, fb
                let fa = match builder.propose_unassignment(&a) {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let fb = match builder.propose_unassignment(&b) {
                    Ok(x) => x,
                    Err(_) => return,
                };

                // fetch now-unassigned branded requests
                let (ra, rb) = builder.with_explorer(|ex| {
                    let ra = ex
                        .iter_unassigned_requests()
                        .find(|r| r.req().id() == rid_a);
                    let rb = ex
                        .iter_unassigned_requests()
                        .find(|r| r.req().id() == rid_b);
                    (ra, rb)
                });
                let (Some(ra), Some(rb)) = (ra, rb) else {
                    return;
                };

                // try insert B into fa, then A into fb
                if let Some(start_b) = pick_start(builder, rb.clone(), &fa) {
                    if let Ok(_ab) = builder.propose_assignment(rb.clone(), start_b, &fa) {
                        if let Some(start_a) = pick_start(builder, ra.clone(), &fb) {
                            if builder.propose_assignment(ra.clone(), start_a, &fb).is_ok() {
                                // success: both swapped
                                return;
                            }
                        }
                        // rollback partial insert B
                        let last = builder.with_explorer(|ex| {
                            ex.iter_assigned_requests()
                                .find(|x| x.asg().request_id() == rid_b)
                        });
                        if let Some(basg) = last {
                            let _ = builder.propose_unassignment(&basg);
                        }
                    }
                }
                // If we get here we leave builder with only the two unassignments
                // -> meta engine will discard the plan when we return None below.
            });

            if let Ok(plan) = plan {
                return Some(plan);
            }
        }
        None
    }
}
