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
    eval::arc_evaluator::ArcEvaluator,
    search::operator::context::OperatorContext,
    state::{chain_set::delta::ChainSetDelta, search_state::SolverSearchState},
};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait OperatorTask<'eval, 'state, 'model, 'problem, T, A>: Send + Sync
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero,
    A: ArcEvaluator<T>,
{
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn key(&self) -> u64;

    fn run(
        &self,
        ctx: &OperatorContext<'eval, 'state, 'model, 'problem, T, A>,
    ) -> Option<ChainSetDelta>;
}

impl<'eval, 'state, 'model, 'problem, T, A> std::fmt::Debug
    for dyn OperatorTask<'eval, 'state, 'model, 'problem, T, A> + Send + Sync
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero,
    A: ArcEvaluator<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorTask")
            .field("name", &self.name())
            .field("key", &self.key())
            .finish()
    }
}

pub trait NeighborhoodOperator<'eval, 'state, 'model, 'problem, T, A>: Send + Sync
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero,
    A: ?Sized + ArcEvaluator<T>,
{
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn plan(
        &self,
        state: &'state SolverSearchState<'model, 'problem, T>,
    ) -> Vec<Box<dyn OperatorTask<'eval, 'state, 'model, 'problem, T, A>>>;
}

impl<'eval, 'state, 'model, 'problem, T, A> std::fmt::Debug
    for dyn NeighborhoodOperator<'eval, 'state, 'model, 'problem, T, A> + Send + Sync
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero,
    A: ?Sized + ArcEvaluator<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeighborhoodOperator")
            .field("name", &self.name())
            .finish()
    }
}

#[cfg(test)]
mod static_assert {
    use super::*;
    use crate::eval::wtt::WeightedTurnaroundTimeObjective;

    static_assertions::assert_obj_safe!(
        OperatorTask<'static, 'static, 'static, 'static, i32, WeightedTurnaroundTimeObjective>
    );

    static_assertions::assert_obj_safe!(
        NeighborhoodOperator<
            'static,
            'static,
            'static,
            'static,
            i32,
            WeightedTurnaroundTimeObjective,
        >
    );
}
