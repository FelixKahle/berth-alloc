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
use rand::seq::SliceRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct ReconstructGreedyOperator<T> {
    /// If true, unassign everything first; otherwise only (re)insert currently unassigned.
    pub wipe_all_first: bool,
    /// Randomize request order for diversification.
    pub shuffle_requests: bool,
    /// Randomize options (free windows) per request.
    pub shuffle_options: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for ReconstructGreedyOperator<T> {
    fn default() -> Self {
        Self {
            wipe_all_first: true,
            shuffle_requests: true,
            shuffle_options: true,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for ReconstructGreedyOperator<T>
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
        "ReconstructGreedy"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut touched = false;

        let plan = ctx.with_builder(|builder| {
            // 1) Optional full wipe
            if self.wipe_all_first {
                // collect first to avoid borrowing issues while mutating
                let to_unassign =
                    builder.with_explorer(|ex| ex.iter_assignments().collect::<Vec<_>>());
                for a in to_unassign {
                    if builder.propose_unassignment(&a).is_ok() {
                        touched = true;
                    }
                }
            }

            // 2) Gather the current unassigned set (may be all requests after wipe)
            let mut reqs =
                builder.with_explorer(|ex| ex.iter_unassigned_requests().collect::<Vec<_>>());

            // A simple heuristic: sort by earliest deadline-like criterion,
            // then by higher weight; optionally shuffle for diversification.
            if self.shuffle_requests {
                reqs.shuffle(rng);
            } else {
                reqs.sort_by_key(|r| {
                    (
                        r.req().feasible_window().end(),
                        std::cmp::Reverse(r.req().weight()),
                    )
                });
            }

            // 3) Greedy (right-packed first, then earliest) across all free gaps/berths
            for req in reqs {
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
                                // right-packed and earliest as fallback
                                out.push((fb.clone(), hi));
                                if hi != lo {
                                    out.push((fb.clone(), lo));
                                }
                            }
                        }
                    }
                    out
                });

                if self.shuffle_options {
                    opts.shuffle(rng);
                } else {
                    // try right-packed first overall
                    opts.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
                }

                for (free, start) in opts {
                    if builder
                        .propose_assignment(req.clone(), start, &free)
                        .is_ok()
                    {
                        touched = true;
                        break;
                    }
                }
                // If none fit, we simply leave this req unassigned; plan can be infeasible.
            }
        });

        match plan {
            Ok(p) if touched => Some(p),
            _ => None,
        }
    }
}
