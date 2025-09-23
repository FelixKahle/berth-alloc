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
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use berth_alloc_model::problem::asg::AssignmentView;
use num_traits::Zero;
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::SliceRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct HillClimbRelocateOperator<T> {
    /// How many random victims to try per call (first improving move is returned).
    pub attempts_per_call: usize,
    /// Shuffle candidate placements for diversification (keeps heuristics light).
    pub shuffle_options: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for HillClimbRelocateOperator<T> {
    fn default() -> Self {
        Self {
            attempts_per_call: 16,
            shuffle_options: true,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for HillClimbRelocateOperator<T>
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
        "HillClimbRelocate"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let aps = self.attempts_per_call.max(1);

        for _ in 0..aps {
            let plan = ctx.with_builder(|builder| {
                let mut pool: Vec<_> =
                    builder.with_explorer(|ex| ex.iter_assigned_requests().collect());
                if pool.is_empty() {
                    return;
                }
                pool.shuffle(rng);
                let victim = pool[0].clone();

                let (rid, bid0, start0) = (
                    victim.asg().request_id(),
                    victim.asg().berth_id(),
                    victim.asg().start_time(),
                );

                let base_before = builder.delta_cost();

                if builder.propose_unassignment(&victim).is_err() {
                    return;
                }
                let base_after_unassign = builder.delta_cost();
                debug_assert!(base_after_unassign <= base_before);

                let req = builder.with_explorer(|ex| {
                    ex.iter_unassigned_requests().find(|r| r.req().id() == rid)
                });
                let Some(req) = req else {
                    let _ = restore_original(builder, rid, bid0, start0);
                    return;
                };

                let mut options = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        let b = fb.berth().id();
                        if let Some(pt) = r.processing_time_for(b) {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = match iv.end().checked_sub(pt) {
                                Some(x) => x,
                                None => continue,
                            };
                            let hi_w = match w.end().checked_sub(pt) {
                                Some(x) => x,
                                None => continue,
                            };
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                out.push((iv, b, hi));
                                if hi != lo {
                                    out.push((iv, b, lo));
                                }
                            }
                        }
                    }
                    out
                });

                if self.shuffle_options {
                    options.shuffle(rng);
                } else {
                    options.sort_by_key(|&(_, _, s)| std::cmp::Reverse(s));
                }

                let mut best: Option<(TimeInterval<T>, _, TimePoint<T>, Cost)> = None;

                for (iv, b, s) in options {
                    let free_opt = builder.with_explorer(|ex| {
                        ex.iter_free_for(req.clone())
                            .find(|fb| fb.berth().id() == b && *fb.interval() == iv)
                    });
                    let Some(free) = free_opt else {
                        continue;
                    };

                    if builder.propose_assignment(req.clone(), s, &free).is_ok() {
                        let delta_total = builder.delta_cost();
                        if delta_total < base_before {
                            match &mut best {
                                None => best = Some((iv, b, s, delta_total)),
                                Some((_, _, _, best_dt)) => {
                                    if delta_total < *best_dt {
                                        *best_dt = delta_total;
                                        best = Some((iv, b, s, delta_total));
                                    }
                                }
                            }
                        }

                        let maybe_asg = builder.with_explorer(|ex| {
                            ex.iter_assigned_requests()
                                .find(|a| a.asg().request_id() == rid)
                        });
                        if let Some(a_ref) = maybe_asg {
                            let _ = builder.propose_unassignment(&a_ref);
                        }
                    }
                }

                if let Some((iv_best, b_best, s_best, _)) = best {
                    let free_final = builder.with_explorer(|ex| {
                        ex.iter_free_for(req.clone())
                            .find(|fb| fb.berth().id() == b_best && *fb.interval() == iv_best)
                    });
                    if let Some(fb_final) = free_final {
                        let _ = builder.propose_assignment(req.clone(), s_best, &fb_final);
                        return;
                    }
                }

                let _ = restore_original(builder, rid, bid0, start0);
            });

            if let Ok(p) = plan
                && p.delta_cost() < Cost::zero() {
                    return Some(p);
                }
        }

        None
    }
}

/// Try to put the victim back exactly where it was.
/// If the exact slot has merged, find a free interval on the same berth that contains [start, start+pt).
fn restore_original<'p, T>(
    builder: &mut crate::framework::planning::PlanBuilder<'_, 'p, T>,
    rid: berth_alloc_model::prelude::RequestIdentifier,
    bid0: berth_alloc_model::prelude::BerthIdentifier,
    start0: TimePoint<T>,
) -> Result<(), ()>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
{
    let req =
        builder.with_explorer(|ex| ex.iter_unassigned_requests().find(|r| r.req().id() == rid));
    let Some(req) = req else {
        return Err(());
    };

    let Some(pt) = req.req().processing_time_for(bid0) else {
        return Err(());
    };

    let end0 = match start0.checked_add(pt) {
        Some(e) => e,
        None => return Err(()),
    };
    let target = TimeInterval::new(start0, end0);

    let free = builder.with_explorer(|ex| {
        ex.iter_free_for(req.clone())
            .find(|fb| fb.berth().id() == bid0 && fb.interval().contains_interval(&target))
    });

    if let Some(fb) = free {
        builder
            .propose_assignment(req, start0, &fb)
            .map(|_| ())
            .map_err(|_| ())
    } else {
        Err(())
    }
}
