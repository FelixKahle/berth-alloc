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
use rand::seq::{IteratorRandom, SliceRandom};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct RelocateOneOperator<T> {
    pub try_randomize_options: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for RelocateOneOperator<T> {
    fn default() -> Self {
        Self {
            try_randomize_options: true,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for RelocateOneOperator<T>
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
        "RelocateOne"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let plan = ctx.with_builder(|builder| {
            // pick one assigned uniformly at random
            let victim = builder.with_explorer(|ex| ex.iter_assigned_requests().choose(rng));
            let Some(v) = victim else {
                return;
            };

            let rid = builder.with_explorer(|ex| v.asg().request_id());

            // unassign -> free hole (we may or may not re-use it)
            let _freed = match builder.propose_unassignment(&v) {
                Ok(x) => x,
                Err(_) => return,
            };

            // fetch branded request (now unassigned)
            let req = builder
                .with_explorer(|ex| ex.iter_unassigned_requests().find(|r| r.req().id() == rid));
            let Some(req) = req else {
                return;
            };

            // gather all current options for this request
            let mut opts = builder.with_explorer(|ex| {
                let r = req.req();
                let w = r.feasible_window();
                let mut out = Vec::new();
                for fb in ex.iter_free_for(req.clone()) {
                    let bid = fb.berth().id();
                    if let Some(pt) = r.processing_time_for(bid) {
                        let iv = *fb.interval();
                        let lo = std::cmp::max(iv.start(), w.start());
                        let hi_iv = iv.end().checked_sub(pt).expect("end-pt ok");
                        let hi_w = w.end().checked_sub(pt).expect("win-pt ok");
                        let hi = std::cmp::min(hi_iv, hi_w);
                        if lo <= hi {
                            out.push((fb.clone(), hi));
                            if hi != lo {
                                out.push((fb.clone(), lo));
                            }
                        }
                    }
                }
                out
            });

            if self.try_randomize_options {
                opts.shuffle(rng);
            }

            for (free, start) in opts {
                if builder
                    .propose_assignment(req.clone(), start, &free)
                    .is_ok()
                {
                    return; // success
                }
            }
            // else: leave as unassigned ⇒ propose() will return None.
        });

        match plan {
            Ok(p) => Some(p),
            Err(_) => None,
        }
    }
}
