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

use crate::{model::index::BerthIndex, state::berth::berthocc::BerthOccupancy};

#[derive(Debug, Clone)]
pub struct TerminalDelta<'b, T: Copy + Ord> {
    updates: Vec<(BerthIndex, BerthOccupancy<'b, T>)>,
}

impl<'b, T: Copy + Ord> TerminalDelta<'b, T> {
    #[inline]
    pub fn from_updates(updates: Vec<(BerthIndex, BerthOccupancy<'b, T>)>) -> Self {
        Self { updates }
    }

    #[inline]
    pub fn empty() -> Self {
        Self {
            updates: Vec::new(),
        }
    }

    pub fn updates(&self) -> &[(BerthIndex, BerthOccupancy<'b, T>)] {
        &self.updates
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }
}

impl<'b, T: Copy + Ord> Default for TerminalDelta<'b, T> {
    #[inline]
    fn default() -> Self {
        Self {
            updates: Vec::new(),
        }
    }
}

impl<'b, T: Copy + Ord> IntoIterator for TerminalDelta<'b, T> {
    type Item = (BerthIndex, BerthOccupancy<'b, T>);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.updates.into_iter()
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    macro_rules! test_integer_types {
        ($($t:ty),*) => {
            $(
                assert_impl_all!(TerminalDelta<$t>: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
