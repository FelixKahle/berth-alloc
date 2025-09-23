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
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::IteratorRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct HoleFillerOperator<T> {
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for HoleFillerOperator<T> {
    fn default() -> Self {
        Self {
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for HoleFillerOperator<T>
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
        "HoleFiller"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut placed = false;

        let plan_res = ctx.with_builder(|builder| {
            let Some(req_for_view) =
                builder.with_explorer(|ex| ex.iter_unassigned_requests().choose(rng))
            else {
                return;
            };

            let Some(fb) =
                builder.with_explorer(|ex| ex.iter_free_for(req_for_view.clone()).choose(rng))
            else {
                return;
            };

            let iv = *fb.interval();
            let bid = fb.berth().id();
            let span = iv.length();

            let mut cands = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .filter(|r| r.req().processing_time_for(bid).is_some())
                    .collect::<Vec<_>>()
            });
            if cands.is_empty() {
                return;
            }

            cands.sort_by(|a, b| {
                let pa = a.req().processing_time_for(bid).unwrap();
                let pb = b.req().processing_time_for(bid).unwrap();
                let fa = pa <= span;
                let fb = pb <= span;

                fa.cmp(&fb)
                    .reverse()
                    .then_with(|| pa.cmp(&pb).reverse())
                    .then_with(|| a.req().weight().cmp(&b.req().weight()).reverse())
            });

            let feasible_starts = |r: &crate::framework::planning::BrandedRequest<'_, 'p, _, T>| {
                let rr = r.req();
                let w = rr.feasible_window();
                let pt = rr.processing_time_for(bid)?;

                let lo = std::cmp::max(iv.start(), w.start());
                let hi_iv = iv.end().checked_sub(pt)?;
                let hi_w = w.end().checked_sub(pt)?;
                let hi = std::cmp::min(hi_iv, hi_w);

                if lo <= hi {
                    Some(if hi != lo { vec![hi, lo] } else { vec![hi] })
                } else {
                    None
                }
            };

            for r in cands {
                if let Some(starts) = feasible_starts(&r) {
                    for s in starts {
                        if builder.propose_assignment(r.clone(), s, &fb).is_ok() {
                            placed = true;
                            return;
                        }
                    }
                }
            }
        });

        match (placed, plan_res) {
            (true, Ok(plan)) => Some(plan),
            _ => None,
        }
    }
}
