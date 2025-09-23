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
use num_traits::{CheckedAdd, CheckedSub};
use rand::{Rng, seq::IteratorRandom};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct BerthBlockRuinRecreateOperator<T> {
    /// Max number of consecutive assignments on a berth to ruin.
    pub max_block_len: usize,
    /// Pack-left or pack-right preference on rebuild.
    pub pack_left: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for BerthBlockRuinRecreateOperator<T> {
    fn default() -> Self {
        Self {
            max_block_len: 6,
            pack_left: true,
            _p: Default::default(),
        }
    }
}

impl<T> Operator for BerthBlockRuinRecreateOperator<T>
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
        "BerthBlockRuinRecreate"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut ok = true;

        let plan = ctx.with_builder(|builder| {
            // Pick a random berth with assignments
            let maybe = builder.with_explorer(|ex| {
                let mut by_berth: std::collections::BTreeMap<_, Vec<_>> =
                    std::collections::BTreeMap::new();
                for a in ex.iter_assignments() {
                    by_berth.entry(a.asg().berth_id()).or_default().push(a);
                }
                // shuffle keys by sampling one
                by_berth
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .choose(rng)
            });

            let Some((bid, mut list)) = maybe else {
                ok = false;
                return;
            };

            // Sort by start time to get contiguous block
            list.sort_by_key(|a| a.asg().start_time());
            if list.is_empty() {
                ok = false;
                return;
            }

            let len = list.len();
            let bl = self.max_block_len.min(len).max(1);
            // choose start index uniformly
            let start_idx = rng.random_range(0..=len - 1);
            let end_idx = (start_idx + bl).min(len);
            if start_idx >= end_idx {
                ok = false;
                return;
            }

            let victims = list[start_idx..end_idx].to_vec();
            if victims.is_empty() {
                ok = false;
                return;
            }

            for v in &victims {
                let _ = builder.propose_unassignment(v);
            }

            // Reinsert only these victims, prefer same berth and pack-left/right
            let victim_ids: std::collections::HashSet<_> =
                victims.iter().map(|a| a.asg().request_id()).collect();

            let mut reqs = builder.with_explorer(|ex| {
                let mut v: Vec<_> = ex
                    .iter_unassigned_requests()
                    .filter(|r| victim_ids.contains(&r.req().id()))
                    .collect();
                // keep original order by id (stable) to reduce variance
                v.sort_by_key(|r| r.req().id());
                v
            });

            'outer: for req in reqs.drain(..) {
                let mut opts = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();
                    let mut out = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        if fb.berth().id() != bid {
                            continue;
                        }
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
                                if self.pack_left {
                                    out.push((fb.clone(), lo));
                                    if hi != lo {
                                        out.push((fb.clone(), hi));
                                    }
                                } else {
                                    out.push((fb.clone(), hi));
                                    if hi != lo {
                                        out.push((fb.clone(), lo));
                                    }
                                }
                            }
                        }
                    }
                    // ensure across gaps we try earlier (or later) starts first on this berth
                    if self.pack_left {
                        out.sort_by_key(|(_, s)| *s);
                    } else {
                        out.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
                    }
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
                    break 'outer;
                }
            }
        });

        if ok { plan.ok() } else { None }
    }
}
