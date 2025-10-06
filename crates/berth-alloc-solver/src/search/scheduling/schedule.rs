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

use crate::state::index::{BerthIndex, RequestIndex};
use berth_alloc_core::prelude::TimeInterval;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Schedule<T> {
    request_index: RequestIndex,
    berth_index: BerthIndex,
    assigned_time_interval: TimeInterval<T>,
}

impl<T> Schedule<T> {
    #[inline]
    pub fn new(
        request_index: RequestIndex,
        berth_index: BerthIndex,
        assigned_time_interval: TimeInterval<T>,
    ) -> Self {
        Self {
            request_index,
            berth_index,
            assigned_time_interval,
        }
    }

    #[inline]
    pub fn request_index(&self) -> RequestIndex {
        self.request_index
    }

    #[inline]
    pub fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    #[inline]
    pub fn assigned_time_interval(&self) -> TimeInterval<T>
    where
        T: Copy,
    {
        self.assigned_time_interval
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Schedule<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {} assigned to Berth {} during {}",
            self.request_index, self.berth_index, self.assigned_time_interval
        )
    }
}
