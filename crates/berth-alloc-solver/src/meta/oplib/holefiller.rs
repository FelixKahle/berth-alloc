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
        let plan = ctx.with_builder(|builder| {
            // Grab a random free interval from any request’s perspective:
            // We synthesize free intervals by iterating a dummy unassigned request set—cheaper: pick a berth, derive free windows via sandbox? We must use explorer API.
            let berths: Vec<_> = builder
                .ledger()
                .problem()
                .berths()
                .iter()
                .map(|b| b.id())
                .collect();
            // we’ll approximate by sampling one unassigned request and using its view:
            let some_req = builder.with_explorer(|ex| ex.iter_unassigned_requests().choose(rng));
            let Some(req) = some_req else {
                return;
            };

            let free_any: Vec<_> =
                builder.with_explorer(|ex| ex.iter_free_for(req.clone()).collect());
            if free_any.is_empty() {
                return;
            }
            let fb = free_any.into_iter().choose(rng).unwrap();

            // choose best-fit unassigned for this specific (berth, interval)
            let bid = fb.berth().id();
            let iv = *fb.interval();

            // collect unassigned that allow this berth
            let mut candidates = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .filter(|r| r.req().processing_time_for(bid).is_some())
                    .collect::<Vec<_>>()
            });
            if candidates.is_empty() {
                return;
            }

            // best-fit by processing time (largest ≤ span), then weight desc
            candidates.sort_by(|a, b| {
                let pa = a.req().processing_time_for(bid).unwrap();
                let pb = b.req().processing_time_for(bid).unwrap();
                let span = iv.length();
                let fits_a = pa <= span;
                let fits_b = pb <= span;
                fits_b
                    .cmp(&fits_a) // prefer fitting
                    .then_with(|| pa.cmp(&pb)) // larger first use reverse if you prefer
                    .then_with(|| {
                        std::cmp::Reverse(a.req().weight())
                            .cmp(&std::cmp::Reverse(b.req().weight()))
                    })
            });

            for r in candidates {
                let Some(pt) = r.req().processing_time_for(bid) else {
                    continue;
                };
                // compute hi/lo feasible starts in this gap
                let w = r.req().feasible_window();
                let lo = std::cmp::max(iv.start(), w.start());
                let hi_iv = iv.end().checked_sub(pt);
                if hi_iv.is_none() {
                    continue;
                }
                let hi_w = w.end().checked_sub(pt);
                if hi_w.is_none() {
                    continue;
                }
                let hi = std::cmp::min(hi_iv.unwrap(), hi_w.unwrap());
                if lo <= hi {
                    // prefer right-packed
                    if builder.propose_assignment(r.clone(), hi, &fb).is_ok() {
                        placed = true;
                        break;
                    }
                    if hi != lo {
                        let _ = builder.propose_assignment(r.clone(), lo, &fb).map(|_| {
                            placed = true;
                        });
                        if placed {
                            break;
                        }
                    }
                }
            }
        });
        if placed { plan.ok() } else { None }
    }
}
