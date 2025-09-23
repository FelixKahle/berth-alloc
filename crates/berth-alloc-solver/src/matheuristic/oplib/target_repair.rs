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
pub struct TargetedRepairOperator<T> {
    /// When true, pick the unassigned request with the highest weight; else pick random.
    pub pick_heaviest: bool,
    /// Max number of overlapping assignments to remove on the chosen berth (use 0 for "all").
    pub max_victims: usize,
    /// Try right-packed first when inserting the target request.
    pub pack_right: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for TargetedRepairOperator<T> {
    fn default() -> Self {
        Self {
            pick_heaviest: true,
            max_victims: 0, // 0 => no cap
            pack_right: true,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for TargetedRepairOperator<T>
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
        "TargetedRepair"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut succeeded = true;

        let plan = ctx.with_builder(|builder| {
            // --- pick an unassigned target request
            let target_opt = builder.with_explorer(|ex| {
                if self.pick_heaviest {
                    ex.iter_unassigned_requests()
                        .max_by_key(|r| r.req().weight())
                } else {
                    ex.iter_unassigned_requests().choose(rng)
                }
            });
            let Some(target) = target_opt else {
                succeeded = false;
                return;
            };

            let r = target.req();
            let window = r.feasible_window();

            // Pick one allowed berth to try on (random among allowed)
            let allowed: Vec<_> = r.iter_allowed_berths_ids().collect();
            if allowed.is_empty() {
                succeeded = false;
                return;
            }
            let bid = *allowed.iter().choose(rng).unwrap();

            // --- collect victims: assignments on that berth overlapping the target window
            let mut victim_ids: Vec<_> = builder.with_explorer(|ex| {
                ex.iter_assignments()
                    .filter(|a| a.asg().berth_id() == bid)
                    .filter(|a| {
                        let iv = a.asg().interval();
                        // overlap test: [iv.start, iv.end) ∩ [window.start, window.end)
                        iv.end() > window.start() && iv.start() < window.end()
                    })
                    .map(|a| a.asg().request_id())
                    .collect()
            });

            if victim_ids.is_empty() {
                // maybe it already fits without removing; try to place directly
                let mut fit = false;
                let mut opts = builder.with_explorer(|ex| {
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(target.clone()) {
                        if fb.berth().id() != bid {
                            continue;
                        }
                        if let Some(pt) = r.processing_time_for(bid) {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), window.start());
                            let hi_iv = iv.end().checked_sub(pt);
                            let hi_w = window.end().checked_sub(pt);
                            if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                                let hi = std::cmp::min(hi_iv, hi_w);
                                if lo <= hi {
                                    if self.pack_right {
                                        out.push((fb.clone(), hi));
                                        if hi != lo {
                                            out.push((fb.clone(), lo));
                                        }
                                    } else {
                                        out.push((fb.clone(), lo));
                                        if hi != lo {
                                            out.push((fb.clone(), hi));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out
                });
                for (free, start) in opts.drain(..) {
                    if builder
                        .propose_assignment(target.clone(), start, &free)
                        .is_ok()
                    {
                        fit = true;
                        break;
                    }
                }
                if !fit {
                    succeeded = false;
                }
                return;
            }

            // optionally cap victims to keep the ruin limited
            if self.max_victims > 0 && victim_ids.len() > self.max_victims {
                victim_ids.truncate(self.max_victims);
            }

            // unassign all chosen victims (remember their request ids)
            for rid in &victim_ids {
                // Find the current assignment by id
                if let Some(asg) = builder.with_explorer(|ex| {
                    ex.iter_assignments().find(|a| a.asg().request_id() == *rid)
                }) {
                    let _ = builder.propose_unassignment(&asg);
                }
            }

            // try to place the target on the selected berth
            let mut placed_target = false;
            let mut target_opts = builder.with_explorer(|ex| {
                let mut out = Vec::new();
                for fb in ex.iter_free_for(target.clone()) {
                    if fb.berth().id() != bid {
                        continue;
                    }
                    if let Some(pt) = r.processing_time_for(bid) {
                        let iv = *fb.interval();
                        let lo = std::cmp::max(iv.start(), window.start());
                        let hi_iv = iv.end().checked_sub(pt);
                        let hi_w = window.end().checked_sub(pt);
                        if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                if self.pack_right {
                                    out.push((fb.clone(), hi));
                                    if hi != lo {
                                        out.push((fb.clone(), lo));
                                    }
                                } else {
                                    out.push((fb.clone(), lo));
                                    if hi != lo {
                                        out.push((fb.clone(), hi));
                                    }
                                }
                            }
                        }
                    }
                }
                out
            });

            for (free, start) in target_opts.drain(..) {
                if builder
                    .propose_assignment(target.clone(), start, &free)
                    .is_ok()
                {
                    placed_target = true;
                    break;
                }
            }
            if !placed_target {
                succeeded = false;
                return;
            }

            // greedily reinsert the victims anywhere
            for rid in victim_ids {
                // fetch the now-unassigned branded request by id
                let req_opt = builder.with_explorer(|ex| {
                    ex.iter_unassigned_requests()
                        .find(|r2| r2.req().id() == rid)
                });
                let Some(req2) = req_opt else {
                    // already reinserted by chance? treat as fine
                    continue;
                };

                let mut opts = builder.with_explorer(|ex| {
                    let r2 = req2.req();
                    let w2 = r2.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req2.clone()) {
                        let bid2 = fb.berth().id();
                        if let Some(pt2) = r2.processing_time_for(bid2) {
                            let iv2 = *fb.interval();
                            let lo2 = std::cmp::max(iv2.start(), w2.start());
                            let hi2_iv = iv2.end().checked_sub(pt2);
                            let hi2_w = w2.end().checked_sub(pt2);
                            if let (Some(hiv), Some(hww)) = (hi2_iv, hi2_w) {
                                let hi2 = std::cmp::min(hiv, hww);
                                if lo2 <= hi2 {
                                    // try both directions for diversity
                                    out.push((fb.clone(), hi2));
                                    if hi2 != lo2 {
                                        out.push((fb.clone(), lo2));
                                    }
                                }
                            }
                        }
                    }
                    out
                });

                let mut placed = false;
                for (free, start) in opts.drain(..) {
                    if builder
                        .propose_assignment(req2.clone(), start, &free)
                        .is_ok()
                    {
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    succeeded = false;
                    return;
                }
            }
        });

        if succeeded { plan.ok() } else { None }
    }
}
