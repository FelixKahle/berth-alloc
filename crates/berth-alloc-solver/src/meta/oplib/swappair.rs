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

use crate::meta::operator::Operator;
use berth_alloc_core::prelude::{Cost, TimePoint};
use berth_alloc_model::problem::asg::AssignmentView;
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::SliceRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct SwapPairOperator<T> {
    pub attempts_per_call: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for SwapPairOperator<T> {
    fn default() -> Self {
        Self {
            attempts_per_call: 60,
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
        let attempts = self.attempts_per_call.max(1);

        #[inline]
        fn feasible_starts<'brand, 'p, T2>(
            req: &crate::framework::planning::BrandedRequest<
                'brand,
                'p,
                berth_alloc_model::common::FlexibleKind,
                T2,
            >,
            free: &crate::framework::planning::BrandedFreeBerth<'brand, 'p, T2>,
        ) -> Option<(TimePoint<T2>, Option<TimePoint<T2>>)>
        where
            T2: Copy + Ord + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        {
            let r = req.req();
            let w = r.feasible_window();
            let bid = free.berth().id();
            let pt = r.processing_time_for(bid)?;
            let iv = *free.interval();

            let lo = std::cmp::max(iv.start(), w.start());
            let hi_iv = iv.end().checked_sub(pt)?;
            let hi_w = w.end().checked_sub(pt)?;
            let hi = std::cmp::min(hi_iv, hi_w);

            if lo > hi {
                None
            } else if hi == lo {
                Some((hi, None))
            } else {
                Some((hi, Some(lo)))
            }
        }

        #[inline]
        fn place_req<'brand, 'p, T2>(
            builder: &mut crate::framework::planning::PlanBuilder<'brand, 'p, T2>,
            req: &crate::framework::planning::BrandedRequest<
                'brand,
                'p,
                berth_alloc_model::common::FlexibleKind,
                T2,
            >,
            free: &crate::framework::planning::BrandedFreeBerth<'brand, 'p, T2>,
        ) -> bool
        where
            T2: Copy + Ord + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        {
            if let Some((hi, lo_opt)) = feasible_starts(req, free) {
                if builder.propose_assignment(req.clone(), hi, free).is_ok() {
                    return true;
                }
                if let Some(lo) = lo_opt
                    && builder.propose_assignment(req.clone(), lo, free).is_ok() {
                        return true;
                    }
            }
            false
        }

        for _ in 0..attempts {
            let plan_res = ctx.with_builder(|builder| {
                let mut pool =
                    builder.with_explorer(|ex| ex.iter_assignments().collect::<Vec<_>>());
                if pool.len() < 2 {
                    return;
                }
                pool.shuffle(rng);
                let a = pool[0].clone();
                let b = pool[1].clone();

                let (rid_a, rid_b) = (a.asg().request_id(), b.asg().request_id());

                let fa = match builder.propose_unassignment(&a) {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let fb = match builder.propose_unassignment(&b) {
                    Ok(x) => x,
                    Err(_) => return,
                };

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

                let mut swapped = false;
                if place_req(builder, &rb, &fa) {
                    if place_req(builder, &ra, &fb) {
                        swapped = true;
                    } else {
                        // rollback partial insert of B
                        if let Some(basg) = builder.with_explorer(|ex| {
                            ex.iter_assignments()
                                .find(|x| x.asg().request_id() == rid_b)
                        }) {
                            let _ = builder.propose_unassignment(&basg);
                        }
                    }
                }

                if !swapped
                    && place_req(builder, &ra, &fb) {
                        if place_req(builder, &rb, &fa) {
                            swapped = true;
                        } else if let Some(aasg) = builder.with_explorer(|ex| {
                            ex.iter_assignments()
                                .find(|x| x.asg().request_id() == rid_a)
                        }) {
                            let _ = builder.propose_unassignment(&aasg);
                        }
                    }

                if !swapped {
                    let sa = fa.interval().start();
                    let sb = fb.interval().start();
                    let _ = builder.propose_assignment(ra.clone(), sa, &fa);
                    let _ = builder.propose_assignment(rb.clone(), sb, &fb);
                }
            });

            if let Ok(plan) = plan_res {
                return Some(plan);
            }
        }

        None
    }
}
