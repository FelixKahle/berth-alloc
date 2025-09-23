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
use rand::seq::IteratorRandom;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct NukeOperator<T> {
    pub fraction: f64,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for NukeOperator<T> {
    fn default() -> Self {
        Self {
            fraction: 0.1,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for NukeOperator<T>
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
        "Nuke"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: crate::framework::planning::PlanningContext<'s, 'p, T>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        let mut made_change = false;
        let f = self.fraction.clamp(0.0, 1.0);

        let plan = ctx.with_builder(|builder| {
            let victims = builder.with_explorer(|ex| {
                let n = ex.iter_assignments().count();
                let mut k = (f * n as f64).round() as usize;
                if f > 0.0 {
                    k = k.max(1);
                }
                ex.iter_assignments().choose_multiple(rng, k)
            });

            if victims.is_empty() {
                return;
            }

            for a in victims {
                if builder.propose_unassignment(&a).is_ok() {
                    made_change = true;
                }
            }
        });

        match plan {
            Ok(p) if made_change => Some(p),
            _ => None,
        }
    }
}
