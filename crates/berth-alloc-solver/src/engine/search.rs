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

use crate::{engine::shared_incumbent::SharedIncumbent, model::solver_model::SolverModel};
use berth_alloc_model::prelude::Problem;
use std::sync::atomic::AtomicBool;

#[derive(Clone)]
pub struct SearchContext<'e, 'm, 'p, T: Copy + Ord, R: rand::Rng> {
    problem: &'p Problem<T>,
    model: &'m SolverModel<'p, T>,
    shared_incumbent: &'e SharedIncumbent<'p, T>,
    stop: &'e AtomicBool,
    rng: R,
}

impl<'e, 'm, 'p, T: Copy + Ord, R: rand::Rng> SearchContext<'e, 'm, 'p, T, R> {
    pub fn new(
        problem: &'p Problem<T>,
        model: &'m SolverModel<'p, T>,
        shared_incumbent: &'e SharedIncumbent<'p, T>,
        stop: &'e AtomicBool,
        rng: R,
    ) -> Self {
        Self {
            problem,
            model,
            shared_incumbent,
            stop,
            rng,
        }
    }

    pub fn problem(&self) -> &'p Problem<T> {
        self.problem
    }

    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    pub fn shared_incumbent(&self) -> &'e SharedIncumbent<'p, T> {
        self.shared_incumbent
    }

    pub fn stop(&self) -> &'e AtomicBool {
        self.stop
    }

    pub fn rng(&mut self) -> &mut R {
        &mut self.rng
    }
}

impl<'e, 'm, 'p, T: Copy + Ord, R: rand::Rng> std::fmt::Debug for SearchContext<'e, 'm, 'p, T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearcherContext")
            .field("problem", &"Problem { .. }")
            .field("model", &"SolverModel { .. }")
            .field("shared_incumbent", &"SharedIncumbent { .. }")
            .field("stop", &self.stop)
            .field("rng", &"Rng { .. }")
            .finish()
    }
}

pub trait SearchStrategy<T, R>: Send + Sync
where
    T: Copy + Ord,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>);
}
