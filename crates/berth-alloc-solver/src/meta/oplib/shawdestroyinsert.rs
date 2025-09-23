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
use rand::seq::{IteratorRandom, SliceRandom};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct ShawDestroyInsertOperator<T> {
    /// Number of assignments to destroy (seed + k-1 most related).
    pub k: usize,
    /// Weight of temporal distance in relatedness score.
    pub alpha_time: f64,
    /// Weight for same-berth “bonus” (0 if same berth, 1 otherwise).
    pub alpha_berth: f64,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for ShawDestroyInsertOperator<T> {
    fn default() -> Self {
        Self {
            k: 4,
            alpha_time: 1.0,
            alpha_berth: 1.0,
            _p: Default::default(),
        }
    }
}

impl<T> Operator for ShawDestroyInsertOperator<T>
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
        "ShawDestroyInsert"
    }

    fn propose<'s, 'p>(
        &self,
        _iter: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut fully_repaired = true;

        let plan_res = ctx.with_builder(|builder| {
            let mut assigned =
                builder.with_explorer(|ex| ex.iter_assignments().collect::<Vec<_>>());
            if assigned.is_empty() {
                fully_repaired = false;
                return;
            }

            let Some(seed) = assigned.iter().choose(rng) else {
                fully_repaired = false;
                return;
            };
            let sbid = seed.asg().berth_id();
            let sstart = seed.asg().start_time();

            let time_dist = |tp: berth_alloc_core::prelude::TimePoint<T>| -> f64 {
                let a: Cost = tp.value().into();
                let b: Cost = sstart.value().into();
                (a - b).abs().to_f64().unwrap_or(0.0)
            };

            assigned.sort_by(|a, b| {
                let (abid, astart) = (a.asg().berth_id(), a.asg().start_time());
                let (bbid, bstart) = (b.asg().berth_id(), b.asg().start_time());

                let rd_a = self.alpha_berth * if abid == sbid { 0.0 } else { 1.0 }
                    + self.alpha_time * time_dist(astart);
                let rd_b = self.alpha_berth * if bbid == sbid { 0.0 } else { 1.0 }
                    + self.alpha_time * time_dist(bstart);

                rd_a.partial_cmp(&rd_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            let k = self.k.min(assigned.len());
            let victims = assigned.into_iter().take(k).collect::<Vec<_>>();
            if victims.is_empty() {
                fully_repaired = false;
                return;
            }
            for v in &victims {
                let _ = builder.propose_unassignment(v);
            }

            let mut todo =
                builder.with_explorer(|ex| ex.iter_unassigned_requests().collect::<Vec<_>>());
            todo.shuffle(rng);

            'repair: for req in todo.into_iter() {
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
                    break 'repair;
                }
            }
        });

        match (fully_repaired, plan_res) {
            (true, Ok(plan)) => Some(plan),
            _ => None,
        }
    }
}
