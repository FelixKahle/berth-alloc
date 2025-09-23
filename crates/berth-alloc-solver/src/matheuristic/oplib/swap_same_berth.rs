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
    framework::planning::{BrandedRequest, PlanBuilder},
    matheuristic::operator::Operator,
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    common::FlexibleKind, prelude::BerthIdentifier, problem::asg::AssignmentView,
};
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::{IteratorRandom, SliceRandom};
use rand_chacha::ChaCha8Rng;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct SwapOrderSameBerthOperator<T> {
    /// Shuffle candidate start choices (lo/hi) for diversification.
    pub try_randomize_starts: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for SwapOrderSameBerthOperator<T> {
    fn default() -> Self {
        Self {
            try_randomize_starts: true,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for SwapOrderSameBerthOperator<T>
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
        "SwapOrderSameBerth"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut placed_both = false;

        // Helper that *does not capture* `builder`: it receives everything via args.
        fn try_assign_on_berth<'brand, 'p, T>(
            builder: &mut PlanBuilder<'brand, 'p, T>,
            req: BrandedRequest<'brand, 'p, FlexibleKind, T>,
            berth_id: BerthIdentifier,
            rng: &mut ChaCha8Rng,
            try_randomize_starts: bool,
        ) -> bool
        where
            T: Copy + Ord + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
        {
            let mut options = builder.with_explorer(|ex| {
                let mut out = Vec::new();
                for fb in ex
                    .iter_free_for(req.clone())
                    .filter(|fb| fb.berth().id() == berth_id)
                {
                    let w = req.req().feasible_window();
                    if let Some(pt) = req.req().processing_time_for(berth_id) {
                        let iv = *fb.interval();
                        let lo = std::cmp::max(iv.start(), w.start());
                        let hi_iv = iv.end().checked_sub(pt);
                        let hi_w = w.end().checked_sub(pt);
                        if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                out.push((fb.clone(), lo));
                                if lo != hi {
                                    out.push((fb.clone(), hi));
                                }
                            }
                        }
                    }
                }
                out
            });

            if try_randomize_starts {
                options.shuffle(rng);
            }

            for (free, start) in options.into_iter() {
                if builder
                    .propose_assignment(req.clone(), start, &free)
                    .is_ok()
                {
                    return true;
                }
            }
            false
        }

        let res = ctx.with_builder(|builder| {
            let assigned: Vec<_> =
                builder.with_explorer(|ex| ex.iter_assigned_requests().collect());

            let pair = assigned
                .iter()
                .flat_map(|a| {
                    let bid = a.asg().berth_id();
                    assigned
                        .iter()
                        .filter(move |b| b.asg().berth_id() == bid)
                        .map(move |b| (a, b))
                })
                .filter(|(a, b)| a.asg().request_id() != b.asg().request_id())
                .choose(rng);

            let Some((a0, b0)) = pair else {
                return;
            };

            let berth_id = a0.asg().berth_id();
            let rid_a = a0.asg().request_id();
            let rid_b = b0.asg().request_id();

            if builder.propose_unassignment(a0).is_err() {
                return;
            }
            if builder.propose_unassignment(b0).is_err() {
                return;
            }

            let req_a = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .find(|r| r.req().id() == rid_a)
            });
            let req_b = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .find(|r| r.req().id() == rid_b)
            });
            let (Some(req_a), Some(req_b)) = (req_a, req_b) else {
                return;
            };

            let ok_b = try_assign_on_berth(
                builder,
                req_b.clone(),
                berth_id,
                rng,
                self.try_randomize_starts,
            );
            if !ok_b {
                return;
            }

            let ok_a = try_assign_on_berth(
                builder,
                req_a.clone(),
                berth_id,
                rng,
                self.try_randomize_starts,
            );
            if !ok_a {
                if let Some(xb) = builder.with_explorer(|ex| {
                    ex.iter_assigned_requests()
                        .find(|x| x.asg().request_id() == rid_b)
                }) {
                    let _ = builder.propose_unassignment(&xb);
                }

                let ok_a2 =
                    try_assign_on_berth(builder, req_a, berth_id, rng, self.try_randomize_starts);
                if !ok_a2 {
                    return;
                }
                let ok_b2 =
                    try_assign_on_berth(builder, req_b, berth_id, rng, self.try_randomize_starts);
                if !ok_b2 {
                    return;
                }
            }

            placed_both = true;
        });

        match (placed_both, res) {
            (true, Ok(plan)) => Some(plan),
            _ => None,
        }
    }
}
