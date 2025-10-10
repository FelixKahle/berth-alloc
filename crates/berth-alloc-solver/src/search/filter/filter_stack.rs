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

use crate::search::filter::traits::FeasibilityFilter;
use num_traits::{CheckedAdd, CheckedSub};

pub struct FilterStack<'model, 'problem, T: Copy + Ord> {
    filters: Vec<Box<dyn FeasibilityFilter<'model, 'problem, T> + 'model>>,
}

impl<'model, 'problem, T> Default for FilterStack<'model, 'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'model, 'problem, T: Copy + Ord> std::fmt::Debug for FilterStack<'model, 'problem, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterStack")
            .field("num_filters", &self.filters.len())
            .finish()
    }
}

impl<'model, 'problem, T> FilterStack<'model, 'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
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
    pub fn with_filters(
        filters: Vec<Box<dyn FeasibilityFilter<'model, 'problem, T> + 'model>>,
    ) -> Self {
        let mut stack = Self { filters };
        stack
            .filters
            .sort_by_key(|f| std::cmp::Reverse(f.complexity()));
        stack
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    #[inline]
    pub fn add_filter(&mut self, filter: Box<dyn FeasibilityFilter<'model, 'problem, T> + 'model>) {
        self.filters.push(filter);
        self.filters.sort_by_key(|f| f.complexity());
    }
}

impl<'model, 'problem, T> FeasibilityFilter<'model, 'problem, T>
    for FilterStack<'model, 'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn is_feasible(
        &self,
        delta: &crate::state::chain_set::prelude::ChainSetDelta,
        search_state: &crate::state::search_state::SolverSearchState<'model, 'problem, T>,
    ) -> bool {
        for filter in &self.filters {
            if !filter.is_feasible(delta, search_state) {
                return false;
            }
        }
        true
    }

    fn complexity(&self) -> usize {
        1
    }
}
