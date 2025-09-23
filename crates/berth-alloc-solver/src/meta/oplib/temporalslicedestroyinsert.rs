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
use berth_alloc_model::problem::asg::AssignmentView;
use num_traits::{CheckedAdd, CheckedSub, ToPrimitive};
use rand::Rng;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct TemporalSliceDestroyInsertOperator<T> {
    pub window_len: i64, // measured in whatever 'Cost' maps to; meaningful for T=i64
    _p: std::marker::PhantomData<T>,
}
impl<T> Default for TemporalSliceDestroyInsertOperator<T> {
    fn default() -> Self {
        Self {
            window_len: 20,
            _p: Default::default(),
        }
    }
}

impl<T> Operator for TemporalSliceDestroyInsertOperator<T>
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
        "TemporalSliceDestroyInsert"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut ok = true;

        let plan = ctx.with_builder(|builder| {
            // Helper: TimePoint<T> -> Cost (via T)
            let tp_to_cost = |tp: berth_alloc_core::prelude::TimePoint<T>| -> Cost {
                let t: T = tp.value();
                Into::<Cost>::into(t)
            };

            // 1) Get min/max time (from assigned intervals)
            let (min_t, max_t) = builder.with_explorer(|ex| {
                let mut lo: Option<berth_alloc_core::prelude::TimePoint<T>> = None;
                let mut hi: Option<berth_alloc_core::prelude::TimePoint<T>> = None;
                for a in ex.iter_assignments() {
                    let iv = a.asg().interval();
                    lo = Some(lo.map_or(
                        iv.start(),
                        |x: berth_alloc_core::prelude::TimePoint<T>| {
                            if x <= iv.start() { x } else { iv.start() }
                        },
                    ));
                    hi = Some(hi.map_or(
                        iv.end(),
                        |x: berth_alloc_core::prelude::TimePoint<T>| {
                            if x >= iv.end() { x } else { iv.end() }
                        },
                    ));
                }
                (lo, hi)
            });

            let (Some(lo_tp), Some(hi_tp)) = (min_t, max_t) else {
                ok = false;
                return;
            };

            let lo_c = tp_to_cost(lo_tp);
            let hi_c = tp_to_cost(hi_tp);

            // 2) Random slice [slice_lo, slice_hi) in Cost space
            let span_c = (hi_c - lo_c);
            let span_i64 = span_c.to_i64().unwrap_or(0).max(1);
            let off = rng.random_range(0..span_i64);
            let slice_lo = lo_c + Cost::from(off);
            let slice_hi = slice_lo + Cost::from(self.window_len.max(1));

            // 3) Pick victims whose assigned interval overlaps the slice
            let victims: Vec<_> = builder.with_explorer(|ex| {
                ex.iter_assignments()
                    .filter(|a| {
                        let iv = a.asg().interval();
                        let s = tp_to_cost(iv.start());
                        let e = tp_to_cost(iv.end());
                        // overlap test: [s,e) intersects [slice_lo, slice_hi)
                        e > slice_lo && s < slice_hi
                    })
                    .collect::<Vec<_>>()
            });

            if victims.is_empty() {
                ok = false;
                return;
            }

            for v in victims {
                let _ = builder.propose_unassignment(&v);
            }

            // 4) Greedy repair all currently unassigned requests
            let mut unassigned =
                builder.with_explorer(|ex| ex.iter_unassigned_requests().collect::<Vec<_>>());

            for req in unassigned.drain(..) {
                let mut opts = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        let bid = fb.berth().id();
                        if let Some(pt) = r.processing_time_for(bid) {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = iv.end().checked_sub(pt).expect("end-pt ok");
                            let hi_w = w.end().checked_sub(pt).expect("win-pt ok");
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                // right-packed first, then earliest if distinct
                                out.push((fb.clone(), hi));
                                if hi != lo {
                                    out.push((fb.clone(), lo));
                                }
                            }
                        }
                    }
                    // optional: try latest first
                    out.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
                    out
                });

                let mut placed = false;
                for (free, start) in opts.drain(..) {
                    if builder
                        .propose_assignment(req.clone(), start, &free)
                        .is_ok()
                    {
                        placed = true;
                        break;
                    }
                }

                if !placed {
                    ok = false;
                    break;
                }
            }
        });

        if ok { plan.ok() } else { None }
    }
}
