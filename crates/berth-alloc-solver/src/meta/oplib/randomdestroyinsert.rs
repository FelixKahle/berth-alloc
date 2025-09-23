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

use std::ops::Mul;

use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::{IteratorRandom, SliceRandom};

use crate::meta::operator::Operator;

#[derive(Debug, Clone)]
pub struct RandomDestroyInsertOperator<T> {
    pub fraction: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for RandomDestroyInsertOperator<T> {
    fn default() -> Self {
        Self {
            fraction: 0.3,
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
        context: crate::framework::planning::PlanningContext<'s, 'p, Self::Time>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, Self::Time>> {
        // We'll mark success/failure inside the builder closure and decide afterwards.
        let mut all_inserted = true;

        let plan_res = context.with_builder(|builder| {
            // ----- Random destroy phase
            let victims = builder.with_explorer(|explorer| {
                let n = explorer.iter_assigned_requests().count();
                let f = self.fraction.clamp(0.0, 1.0);
                let mut k = (f * n as f64).round() as usize;
                if f > 0.0 {
                    k = k.max(1);
                }
                explorer.iter_assigned_requests().choose_multiple(rng, k)
            });

            // IMPORTANT: consume the iterator so unassignments actually happen
            for v in victims {
                // Best-effort unassignment; if something races, just continue
                let _ = builder.propose_unassignment(&v);
            }

            // ----- Reinsert *all* currently unassigned requests
            // (includes ones that were already unassigned + freshly destroyed)
            // We'll randomize the order to diversify search.
            let mut todo = builder
                .with_explorer(|explorer| explorer.iter_unassigned_requests().collect::<Vec<_>>());
            todo.shuffle(rng);

            // Greedy repair: for each unassigned request, try all current free options
            // (right-packed first, then earliest in the same gap).
            'repair: for req in todo {
                // Snapshot current free options for this request
                let mut options = builder.with_explorer(|ex| {
                    let r = req.req();
                    let w = r.feasible_window();

                    // build (free_berth, start_time) pairs in the preferred order
                    let mut opts = Vec::new();
                    for fb in ex.iter_free_for(req.clone()) {
                        let bid = fb.berth().id();
                        if let Some(pt) = r.processing_time_for(bid) {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = iv.end().checked_sub(pt).expect("end-pt ok");
                            let hi_w = w.end().checked_sub(pt).expect("win-pt ok");
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                // try right-packed first, then earliest if distinct
                                opts.push((fb.clone(), hi));
                                if hi != lo {
                                    opts.push((fb.clone(), lo));
                                }
                            }
                        }
                    }
                    opts
                });

                // Optionally randomize within the same heuristic ordering for diversification.
                // (Keeps the hi/lo pairing but shuffles among different gaps/berths.)
                options.shuffle(rng);

                // Try to place this request
                let mut placed = false;
                for (free, start) in options {
                    match builder.propose_assignment(req.clone(), start, &free) {
                        Ok(_a) => {
                            placed = true;
                            break;
                        }
                        Err(crate::framework::err::ProposeAssignmentError::NotFree(_)) => {
                            continue;
                        }
                        Err(_other) => {
                            continue;
                        }
                    }
                }

                if !placed {
                    all_inserted = false;
                    break 'repair;
                }
            }
        });

        let Ok(plan) = plan_res else {
            return None;
        };

        if all_inserted { Some(plan) } else { None }
    }
}
