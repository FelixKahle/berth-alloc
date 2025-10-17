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

use crate::{search::planner::PlanningContext, state::plan::Plan};
use berth_alloc_model::prelude::RequestIdentifier;

pub trait Operator<T: Copy + Ord, R: rand::Rng>: Send + Sync {
    fn name(&self) -> &str;
    fn propose<'s, 'p>(
        &self,
        context: &PlanningContext<'s, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>>;
}

pub trait LocalMoveOperator<T: Copy + Ord, R: rand::Rng>: Send + Sync {
    fn name(&self) -> &str;
    fn propose<'s, 'p>(&self, ctx: &PlanningContext<'s, 'p, T>, rng: &mut R)
    -> Option<Plan<'p, T>>;
}

pub trait DestroyOperator<T: Copy + Ord, R: rand::Rng>: Send + Sync {
    fn name(&self) -> &str;
    fn select<'s, 'p>(
        &self,
        ctx: &PlanningContext<'s, 'p, T>,
        rng: &mut R,
    ) -> Vec<RequestIdentifier>;
}

pub trait RepairOperator<T: Copy + Ord, R: rand::Rng>: Send + Sync {
    fn name(&self) -> &str;
    fn repair<'s, 'p>(
        &self,
        ctx: &PlanningContext<'s, 'p, T>,
        removed: &[RequestIdentifier],
        rng: &mut R,
    ) -> Option<Plan<'p, T>>;
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::{assert_impl_all, assert_obj_safe};
    use rand_chacha::ChaCha8Rng;

    macro_rules! test_integer_types {
        ($($t:ty),* $(,)?) => {
            $(
                assert_obj_safe!(Operator<$t, ChaCha8Rng>);
                assert_impl_all!(dyn Operator<$t, ChaCha8Rng> + Send + Sync: Send, Sync);
                assert_obj_safe!(LocalMoveOperator<$t, ChaCha8Rng>);
                assert_impl_all!(dyn LocalMoveOperator<$t, ChaCha8Rng> + Send + Sync: Send, Sync);
                assert_obj_safe!(DestroyOperator<$t, ChaCha8Rng>);
                assert_impl_all!(dyn DestroyOperator<$t, ChaCha8Rng> + Send + Sync: Send, Sync);
                assert_obj_safe!(RepairOperator<$t, ChaCha8Rng>);
                assert_impl_all!(dyn RepairOperator<$t, ChaCha8Rng> + Send + Sync: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
