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

use crate::state::{
    chain::delta::ChainDelta, filter::filter_trait::Filter, solver::solver_state::SolverState,
};

pub struct FilterStack<T> {
    filters: Vec<Box<dyn Filter<T>>>,
}

impl<T: Copy + Ord> FilterStack<T> {
    #[inline]
    pub fn new() -> Self {
        Self { filters: vec![] }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            filters: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn add_filter<F: Filter<T> + 'static>(&mut self, filter: F) {
        // Keep simplest first, most complex last
        self.filters.push(Box::new(filter));
        self.filters.sort_by_key(|f| f.complexity());
    }
}

impl<T: Copy + Ord> Filter<T> for FilterStack<T> {
    #[inline]
    fn check(&self, delta: &ChainDelta, state: &SolverState<T>) -> bool {
        self.filters.iter().all(|f| f.check(delta, state))
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for FilterStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterStack")
            .field("filters", &self.filters.len())
            .finish()
    }
}

impl<T: std::fmt::Display> std::fmt::Display for FilterStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FilterStack with {} filters", self.filters.len())
    }
}

impl<T: Copy + Ord> Default for FilterStack<T> {
    fn default() -> Self {
        Self::new()
    }
}
