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
    search::eval::CostEvaluator,
    state::{plan::Plan, solver_state::SolverState},
};

pub trait LocalSearchOperator<'p, T, C, R>: Send + Sync
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;
    fn synchronize<'s>(&mut self, state: &'s SolverState<'p, T>);
    fn enter_search(&mut self, random: &mut R);
    fn make_next_neighbor(&mut self) -> Option<Plan<'p, T>>;
    fn reset(&mut self);
    fn has_fragments(&self) -> bool;

    #[inline]
    fn iter_neighbors<'o>(&'o mut self) -> LocalSearchOperatorIterator<'o, 'p, T, C, R, Self>
    where
        Self: Sized,
    {
        LocalSearchOperatorIterator::new(self)
    }
}

#[derive(Debug)]
pub struct LocalSearchOperatorIterator<'o, 'p, T, C, R, O>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
    O: LocalSearchOperator<'p, T, C, R>,
{
    operator: &'o mut O,
    _phantom: std::marker::PhantomData<(&'p T, C, R)>,
}

impl<'o, 'p, T, C, R, O> LocalSearchOperatorIterator<'o, 'p, T, C, R, O>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
    O: LocalSearchOperator<'p, T, C, R>,
{
    pub fn new(operator: &'o mut O) -> Self {
        Self {
            operator,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'o, 'p, T, C, R, O> Iterator for LocalSearchOperatorIterator<'o, 'p, T, C, R, O>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
    O: LocalSearchOperator<'p, T, C, R>,
{
    type Item = Plan<'p, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.operator.make_next_neighbor()
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use crate::search::eval::DefaultCostEvaluator;
    use ::static_assertions::{assert_impl_all, assert_obj_safe};
    use rand_chacha::ChaCha8Rng;

    macro_rules! test_integer_types {
        ($($t:ty),* $(,)?) => {
            $(
                assert_obj_safe!(LocalSearchOperator<$t, DefaultCostEvaluator, ChaCha8Rng>);
                assert_impl_all!(dyn LocalSearchOperator<$t, DefaultCostEvaluator, ChaCha8Rng> + Send + Sync: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
