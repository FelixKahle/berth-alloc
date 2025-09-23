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
use rand::seq::SliceRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct LongestRuinRecreateOperator<T> {
    /// Fraction of assigned requests to ruin (0..=1). If >0, at least one is removed.
    pub fraction: f64,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for LongestRuinRecreateOperator<T> {
    fn default() -> Self {
        Self {
            fraction: 0.15,
            _p: Default::default(),
        }
    }
}

impl<T> Operator for LongestRuinRecreateOperator<T>
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
        "LongestRuinRecreate"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut repaired_all = true;

        let plan = ctx.with_builder(|builder| {
            // Collect all assignments
            let mut assigned: Vec<_> = builder.with_explorer(|ex| ex.iter_assignments().collect());
            if assigned.is_empty() {
                repaired_all = false;
                return;
            }

            assigned.sort_by_key(|a| std::cmp::Reverse(a.asg().interval().length()));

            let n = assigned.len();
            let f = self.fraction.clamp(0.0, 1.0);
            let mut k = ((f * n as f64).round() as usize).min(n);
            if f > 0.0 {
                k = k.max(1);
            }

            let victims = assigned.into_iter().take(k).collect::<Vec<_>>();
            if victims.is_empty() {
                repaired_all = false;
                return;
            }

            for v in &victims {
                let _ = builder.propose_unassignment(v);
            }

            let victim_ids: std::collections::HashSet<_> =
                victims.iter().map(|a| a.asg().request_id()).collect();
            let mut todo = builder.with_explorer(|ex| {
                ex.iter_unassigned_requests()
                    .filter(|r| victim_ids.contains(&r.req().id()))
                    .collect::<Vec<_>>()
            });
            todo.shuffle(rng);

            'outer: for req in todo {
                let mut opts = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        let bid = fb.berth().id();
                        if let Some(pt) = r.processing_time_for(bid) {
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
                                out.push((fb.clone(), hi));
                                if hi != lo {
                                    out.push((fb.clone(), lo));
                                }
                            }
                        }
                    }
                    out.shuffle(rng);
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
                    repaired_all = false;
                    break 'outer;
                }
            }
        });

        match plan {
            Ok(p) if repaired_all => Some(p),
            _ => None,
        }
    }
}
