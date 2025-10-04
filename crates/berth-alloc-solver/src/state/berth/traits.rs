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

use crate::state::berth::err::BerthUpdateError;
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::prelude::Berth;
use num_traits::Zero;

pub trait BerthRead<'b, T: Copy + Ord> {
    fn is_free(&self, interval: TimeInterval<T>) -> bool;
    fn is_occupied(&self, interval: TimeInterval<T>) -> bool;
    fn berth(&self) -> &'b Berth<T>;
    fn iter_free_intervals_in(
        &self,
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = TimeInterval<T>>;

    #[inline]
    fn iter_free_intervals(&self) -> impl Iterator<Item = TimeInterval<T>>
    where
        T: Zero + 'b,
    {
        self.iter_free_intervals_in(self.berth().horizon_interval())
    }

    #[inline]
    fn first_free_interval_in(&self, window: TimeInterval<T>) -> Option<TimeInterval<T>>
    where
        T: Zero + 'b,
    {
        self.iter_free_intervals_in(window).next()
    }
}

pub trait BerthWrite<'b, T: Copy + Ord>: BerthRead<'b, T> {
    fn occupy(&mut self, interval: TimeInterval<T>) -> Result<(), BerthUpdateError<T>>;
    fn release(&mut self, interval: TimeInterval<T>) -> Result<(), BerthUpdateError<T>>;
}
