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
use rand::seq::IteratorRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct PackLeftOnBerthOperator<T> {
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for PackLeftOnBerthOperator<T> {
    fn default() -> Self {
        Self {
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for PackLeftOnBerthOperator<T>
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
        "PackLeftOnBerth"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut ok = true;

        let plan = ctx.with_builder(|builder| {
            let maybe_berth = builder.with_explorer(|ex| {
                let mut by_berth: std::collections::BTreeMap<_, Vec<_>> =
                    std::collections::BTreeMap::new();
                for a in ex.iter_assignments() {
                    by_berth.entry(a.asg().berth_id()).or_default().push(a);
                }
                by_berth.into_iter().choose(rng)
            });

            let Some((bid, mut list)) = maybe_berth else {
                ok = false;
                return;
            };

            list.sort_by_key(|a| a.asg().start_time());
            for a in &list {
                let _ = builder.propose_unassignment(a);
            }

            let mut reqs = builder.with_explorer(|ex| {
                let ids: Vec<_> = list.iter().map(|a| a.asg().request_id()).collect();
                let mut v: Vec<_> = ex
                    .iter_unassigned_requests()
                    .filter(|r| ids.contains(&r.req().id()))
                    .collect();
                v.sort_by_key(|r| r.req().id());
                v
            });

            'all: for req in reqs.drain(..) {
                let mut opts = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        if fb.berth().id() != bid {
                            continue;
                        }
                        let Some(pt) = r.processing_time_for(bid) else {
                            continue;
                        };
                        let iv = *fb.interval();
                        let lo = std::cmp::max(iv.start(), w.start());
                        let hi_iv = iv.end().checked_sub(pt).expect("end-pt ok");
                        let hi_w = w.end().checked_sub(pt).expect("win-pt ok");
                        let hi = std::cmp::min(hi_iv, hi_w);
                        if lo <= hi {
                            out.push((fb.clone(), lo));
                            if hi != lo {
                                out.push((fb.clone(), hi));
                            }
                        }
                    }
                    out.sort_by_key(|(_, s)| *s);
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
                    break 'all;
                }
            }
        });

        if ok { plan.ok() } else { None }
    }
}
