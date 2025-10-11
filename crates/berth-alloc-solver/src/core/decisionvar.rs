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

use crate::state::index::BerthIndex;
use berth_alloc_core::prelude::TimePoint;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Decision<T> {
    pub berth_index: BerthIndex,
    pub start_time: TimePoint<T>,
}

impl<T> Decision<T> {
    #[inline]
    pub fn new(berth_index: BerthIndex, start_time: TimePoint<T>) -> Self {
        Self {
            berth_index,
            start_time,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Decision<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth: {}, Start Time: {}",
            self.berth_index, self.start_time
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DecisionVar<T> {
    Unassigned,
    Assigned(Decision<T>),
}

impl<T> DecisionVar<T> {
    #[inline]
    pub fn unassigned() -> Self {
        Self::Unassigned
    }

    #[inline]
    pub fn assigned(berth_index: BerthIndex, start_time: TimePoint<T>) -> Self {
        Self::Assigned(Decision::new(berth_index, start_time))
    }

    #[inline]
    pub fn is_assigned(&self) -> bool {
        matches!(self, Self::Assigned(_))
    }

    #[inline]
    pub fn as_assigned(&self) -> Option<&Decision<T>> {
        match self {
            Self::Assigned(decision) => Some(decision),
            Self::Unassigned => None,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for DecisionVar<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unassigned => write!(f, "Unassigned"),
            Self::Assigned(decision) => write!(f, "{}", decision),
        }
    }
}
