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
    eval::ArcEvaluator,
    state::{
        chain_set::{delta::ChainSetDelta, index::NodeIndex},
        search_state::SolverSearchState,
    },
};
use num_traits::{CheckedAdd, CheckedSub};

/// A function that, given two node indices, returns a vector of neighboring node indices.
pub type NeighborAccessor = dyn Fn(NodeIndex, NodeIndex) -> Vec<NodeIndex>;

pub trait NeighborhoodOperator<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn make_neighboor<'state, 'model, 'problem>(
        &self,
        search_state: &'state SolverSearchState<'model, 'problem, T>,
        arc_evaluator: &ArcEvaluator,
    ) -> Option<ChainSetDelta>;
}

impl<T> std::fmt::Debug for dyn NeighborhoodOperator<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name().fmt(f)
    }
}

impl<T> std::fmt::Display for dyn NeighborhoodOperator<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name().fmt(f)
    }
}

pub trait NeighborhoodOperatorFactory<T>: Send + Sync {
    fn name(&self) -> &'static str;
    fn create(&self) -> Box<dyn NeighborhoodOperator<T>>;
}
