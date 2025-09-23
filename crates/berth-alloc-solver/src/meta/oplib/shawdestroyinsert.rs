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
    pub k: usize,
    pub alpha_time: f64,  // weight for temporal distance
    pub alpha_berth: f64, // weight for same-berth bonus
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
        let mut ok = true;

        let plan = ctx.with_builder(|builder| {
            let assigned: Vec<_> = builder.with_explorer(|ex| ex.iter_assignments().collect());
            if assigned.is_empty() {
                ok = false;
                return;
            }

            let seed = assigned.iter().choose(rng).unwrap().clone();
            let (sbid, sstart) =
                builder.with_explorer(|ex| (seed.asg().berth_id(), seed.asg().start_time()));

            // Score relatedness
            let mut cand: Vec<_> = assigned;

            // helper: TimePoint<T> -> Cost (via T)
            let tp_to_cost = |tp: berth_alloc_core::prelude::TimePoint<T>| -> Cost {
                let t: T = tp.value();
                Into::<Cost>::into(t)
            };

            cand.sort_by(|a, b| {
                let (abid, astart) = (a.asg().berth_id(), a.asg().start_time());
                let (bbid, bstart) = (b.asg().berth_id(), b.asg().start_time());

                let rd_a = (self.alpha_berth * if abid == sbid { 0.0 } else { 1.0 })
                    + self.alpha_time
                        * (tp_to_cost(astart) - tp_to_cost(sstart))
                            .abs()
                            .to_f64()
                            .unwrap_or(0.0);

                let rd_b = (self.alpha_berth * if bbid == sbid { 0.0 } else { 1.0 })
                    + self.alpha_time
                        * (tp_to_cost(bstart) - tp_to_cost(sstart))
                            .abs()
                            .to_f64()
                            .unwrap_or(0.0);

                rd_a.partial_cmp(&rd_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Pick top-k related (excluding the seed itself appears near top)
            let k = self.k.min(cand.len());
            let victims: Vec<_> = cand.into_iter().take(k).collect();

            for v in &victims {
                let _ = builder.propose_unassignment(v);
            }

            // greedy repair all currently unassigned
            let mut todo =
                builder.with_explorer(|ex| ex.iter_unassigned_requests().collect::<Vec<_>>());
            for req in todo.drain(..) {
                let mut opts = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut o = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        let bid = fb.berth().id();
                        if let Some(pt) = r.processing_time_for(bid) {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = iv.end().checked_sub(pt).expect("end-pt ok");
                            let hi_w = w.end().checked_sub(pt).expect("win-pt ok");
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                o.push((fb.clone(), hi));
                                if hi != lo {
                                    o.push((fb.clone(), lo));
                                }
                            }
                        }
                    }
                    o
                });
                // randomized order for diversification
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
                    ok = false;
                    break;
                }
            }
        });

        if ok { plan.ok() } else { None }
    }
}
