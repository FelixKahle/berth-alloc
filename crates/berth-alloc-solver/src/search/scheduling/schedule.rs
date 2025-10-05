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
pub struct RequestSchedule<T> {
    request: RequestIndex,
    berth: BerthIndex,
    interval: TimeInterval<T>,
}

impl<T> RequestSchedule<T> {
    #[inline]
    pub fn new(request: RequestIndex, berth: BerthIndex, interval: TimeInterval<T>) -> Self {
        Self {
            request,
            berth,
            interval,
        }
    }

    #[inline]
    pub fn request(&self) -> RequestIndex {
        self.request
    }

    #[inline]
    pub fn berth(&self) -> BerthIndex {
        self.berth
    }

    #[inline]
    pub fn interval(&self) -> &TimeInterval<T> {
        &self.interval
    }
}

impl<T: std::fmt::Display> std::fmt::Display for RequestSchedule<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RequestSchedule(request: {}, berth: {}, interval: {})",
            self.request, self.berth, self.interval
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule<T> {
    schedules: Vec<Vec<RequestSchedule<T>>>,
}

impl<T: Copy + Ord> Schedule<T> {
    #[inline]
    pub fn new(num_berths: usize) -> Self {
        Self {
            schedules: vec![Vec::new(); num_berths],
        }
    }

    #[inline]
    pub fn add_schedule(&mut self, schedule: RequestSchedule<T>) {
        let berth_idx = schedule.berth().0;
        if berth_idx >= self.schedules.len() {
            self.schedules.resize(berth_idx + 1, Vec::new());
        }
        self.schedules[berth_idx].push(schedule);
    }

    #[inline]
    pub fn schedules_for_berth(&self, berth: BerthIndex) -> Option<&[RequestSchedule<T>]> {
        self.schedules.get(berth.0).map(|v| v.as_slice())
    }

    #[inline]
    pub fn iter_schedules(&self) -> impl Iterator<Item = (BerthIndex, &[RequestSchedule<T>])> {
        self.schedules
            .iter()
            .enumerate()
            .map(|(i, v)| (BerthIndex(i), v.as_slice()))
    }
}
