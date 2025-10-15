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

use crate::state::{chain_set::delta::ChainSetDelta, search_state::SolverSearchState};
use num_traits::{CheckedAdd, CheckedSub};

pub trait FeasibilityFilter<T>: Send + Sync
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn complexity(&self) -> usize;

    fn is_feasible<'model, 'problem>(
        &self,
        delta: &ChainSetDelta,
        search_state: &SolverSearchState<T>,
    ) -> bool;
}

impl<T> PartialEq for dyn FeasibilityFilter<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn eq(&self, other: &Self) -> bool {
        self.complexity() == other.complexity()
    }
}

impl<T> Eq for dyn FeasibilityFilter<T> where T: Copy + Ord + CheckedAdd + CheckedSub {}

impl<T> PartialOrd for dyn FeasibilityFilter<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for dyn FeasibilityFilter<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.complexity().cmp(&other.complexity())
    }
}

impl<T> std::fmt::Debug for dyn FeasibilityFilter<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filter")
            .field("name", &self.name())
            .field("complexity", &self.complexity())
            .finish()
    }
}

impl<T> std::fmt::Display for dyn FeasibilityFilter<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (complexity: {})", self.name(), self.complexity())
    }
}
