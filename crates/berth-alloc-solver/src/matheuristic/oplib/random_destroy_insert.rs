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
use std::{collections::HashSet, ops::Mul};

#[derive(Debug, Clone)]
pub struct RandomDestroyInsertOperator<T> {
    pub fraction: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for RandomDestroyInsertOperator<T> {
    fn default() -> Self {
        Self {
            fraction: 0.7,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for RandomDestroyInsertOperator<T>
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
        "RandomDestroyInsert"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, Self::Time>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, Self::Time>> {
        let mut fully_repaired = true;
        let frac = self.fraction.clamp(0.0, 1.0);

        let plan_res = ctx.with_builder(|builder| {
            let (victims, destroyed_ids): (Vec<_>, HashSet<_>) = builder.with_explorer(|ex| {
                let n = ex.iter_assignments().count();
                let mut k = (frac * n as f64).round() as usize;
                if frac > 0.0 {
                    k = k.max(1);
                }
                let v: Vec<_> = ex.iter_assignments().choose_multiple(rng, k);
                let ids = v.iter().map(|a| a.asg().request_id()).collect();
                (v, ids)
            });

            if victims.is_empty() {
                fully_repaired = false;
                return;
            }
            for v in &victims {
                let _ = builder.propose_unassignment(v);
            }

            let mut destroyed_reqs = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .filter(|r| destroyed_ids.contains(&r.req().id()))
                    .collect::<Vec<_>>()
            });
            destroyed_reqs.shuffle(rng);

            for req in destroyed_reqs {
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
                            if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                                let hi = std::cmp::min(hi_iv, hi_w);
                                if lo <= hi {
                                    out.push((fb.clone(), hi));
                                    if hi != lo {
                                        out.push((fb.clone(), lo));
                                    }
                                }
                            }
                        }
                    }
                    out
                });

                opts.shuffle(rng);

                let mut placed = false;
                for (free, start) in opts {
                    if builder
                        .propose_assignment(req.clone(), start, &free)
                        .is_ok()
                    {
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    fully_repaired = false;
                    return;
                }
            }

            let mut others = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .filter(|r| !destroyed_ids.contains(&r.req().id()))
                    .collect::<Vec<_>>()
            });
            others.shuffle(rng);

            for req in others {
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
                            if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                                let hi = std::cmp::min(hi_iv, hi_w);
                                if lo <= hi {
                                    out.push((fb.clone(), hi));
                                    if hi != lo {
                                        out.push((fb.clone(), lo));
                                    }
                                }
                            }
                        }
                    }
                    out
                });

                opts.shuffle(rng);

                for (free, start) in opts {
                    if builder
                        .propose_assignment(req.clone(), start, &free)
                        .is_ok()
                    {
                        break;
                    }
                }
            }
        });

        match (fully_repaired, plan_res) {
            (true, Ok(plan)) => Some(plan),
            _ => None,
        }
    }
}
