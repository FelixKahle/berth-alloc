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

use crate::{
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    search::filter::traits::FeasibilityFilter,
    state::{chain_set::delta::ChainSetDelta, search_state::SolverSearchState},
};
use num_traits::{CheckedAdd, CheckedSub};

pub struct FilterStack<T: Copy + Ord> {
    filters: Vec<Box<dyn FeasibilityFilter<T>>>,
}

impl<T> Default for FilterStack<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord> std::fmt::Debug for FilterStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterStack")
            .field("num_filters", &self.filters.len())
            .finish()
    }
}

impl<T> FilterStack<T>
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
    pub fn with_filters(filters: Vec<Box<dyn FeasibilityFilter<T>>>) -> Self {
        let mut stack = Self { filters };
        stack
            .filters
            .sort_by_key(|f| std::cmp::Reverse(f.complexity()));
        stack
    }

    #[inline]
    pub fn empty() -> Self {
        Self { filters: vec![] }
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
    pub fn add_filter(&mut self, filter: Box<dyn FeasibilityFilter<T>>) {
        self.filters.push(filter);
        self.filters.sort_by_key(|f| f.complexity());
    }
}

impl<T> FeasibilityFilter<T> for FilterStack<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn is_feasible<'model, 'problem>(
        &self,
        delta: &ChainSetDelta,
        search_state: &SolverSearchState<'model, 'problem, T>,
        iv: &[IntervalVar<T>],
        dv: &[DecisionVar<T>],
        touched: &[usize],
    ) -> bool {
        for filter in &self.filters {
            if !filter.is_feasible(delta, search_state, iv, dv, touched) {
                return false;
            }
        }
        true
    }

    fn complexity(&self) -> usize {
        1
    }
}
