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
use num_traits::{CheckedAdd, CheckedSub, ToPrimitive};
use rand::Rng;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct TemporalSliceDestroyInsertOperator<T> {
    /// Slice length in the same numeric space as `Cost` (meaningful e.g. for T=i64).
    pub window_len: i64,
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
        let mut succeeded = true;

        #[inline]
        fn tp_to_cost<T: Copy + Into<Cost>>(tp: berth_alloc_core::prelude::TimePoint<T>) -> Cost {
            let v: T = tp.value();
            v.into()
        }

        // Build and run the plan.
        let plan = ctx.with_builder(|builder| {
            let (lo_opt, hi_opt) = builder.with_explorer(|ex| {
                let mut lo: Option<berth_alloc_core::prelude::TimePoint<T>> = None;
                let mut hi: Option<berth_alloc_core::prelude::TimePoint<T>> = None;

                for a in ex.iter_assignments() {
                    let iv = a.asg().interval();
                    let s = iv.start();
                    let e = iv.end();
                    lo = Some(lo.map_or(s, |cur| if s < cur { s } else { cur }));
                    hi = Some(hi.map_or(e, |cur| if e > cur { e } else { cur }));
                }
                (lo, hi)
            });

            let (Some(lo_tp), Some(hi_tp)) = (lo_opt, hi_opt) else {
                succeeded = false;
                return;
            };

            let lo_c = tp_to_cost(lo_tp);
            let hi_c = tp_to_cost(hi_tp);
            if hi_c <= lo_c {
                succeeded = false;
                return;
            }

            let span_cost = hi_c - lo_c;
            let Some(span_i64) = span_cost.to_i64() else {
                succeeded = false;
                return;
            };
            if span_i64 <= 0 {
                succeeded = false;
                return;
            }

            let slice_len = self.window_len.max(1);
            let offset = rng.random_range(0..span_i64);
            let slice_lo = lo_c + Cost::from(offset);
            let slice_hi = slice_lo + Cost::from(slice_len);

            let victims: Vec<_> = builder.with_explorer(|ex| {
                ex.iter_assignments()
                    .filter(|a| {
                        let iv = a.asg().interval();
                        let s = tp_to_cost(iv.start());
                        let e = tp_to_cost(iv.end());
                        e > slice_lo && s < slice_hi
                    })
                    .collect()
            });

            if victims.is_empty() {
                succeeded = false;
                return;
            }

            for v in victims {
                let _ = builder.propose_unassignment(&v);
            }

            let mut todo =
                builder.with_explorer(|ex| ex.iter_unassigned_requests().collect::<Vec<_>>());

            for req in todo.drain(..) {
                let mut opts = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        let bid = fb.berth().id();
                        if let Some(pt) = r.processing_time_for(bid) {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = iv.end().checked_sub(pt);
                            let hi_w = w.end().checked_sub(pt);
                            let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) else {
                                continue;
                            };
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                out.push((fb.clone(), hi));
                                if hi != lo {
                                    out.push((fb.clone(), lo));
                                }
                            }
                        }
                    }
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
                    succeeded = false;
                    break;
                }
            }
        });

        if succeeded { plan.ok() } else { None }
    }
}
