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

use crate::model::index::{BerthIndex, RequestIndex};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NotAllowedOnBerthError {
    request: RequestIndex,
    berth: BerthIndex,
}

impl NotAllowedOnBerthError {
    #[inline]
    pub fn new(request: RequestIndex, berth: BerthIndex) -> Self {
        Self { request, berth }
    }

    #[inline]
    pub fn request(&self) -> RequestIndex {
        self.request
    }

    #[inline]
    pub fn berth(&self) -> BerthIndex {
        self.berth
    }
}

impl std::fmt::Display for NotAllowedOnBerthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request with index {} is not allowed on berth {}",
            self.request, self.berth
        )
    }
}

impl std::error::Error for NotAllowedOnBerthError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeasiblyWindowViolationError {
    request: RequestIndex,
}

impl FeasiblyWindowViolationError {
    #[inline]
    pub fn new(request: RequestIndex) -> Self {
        Self { request }
    }

    #[inline]
    pub fn request(&self) -> RequestIndex {
        self.request
    }
}

impl std::fmt::Display for FeasiblyWindowViolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request with index {} cannot be scheduled within its feasible window",
            self.request
        )
    }
}

impl std::error::Error for FeasiblyWindowViolationError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OverlapError {
    berth: BerthIndex,
    left: RequestIndex,
    right: RequestIndex,
}

impl OverlapError {
    #[inline]
    pub fn new(berth: BerthIndex, left: RequestIndex, right: RequestIndex) -> Self {
        Self { berth, left, right }
    }

    #[inline]
    pub fn berth(&self) -> BerthIndex {
        self.berth
    }

    #[inline]
    pub fn left(&self) -> RequestIndex {
        self.left
    }

    #[inline]
    pub fn right(&self) -> RequestIndex {
        self.right
    }
}

impl std::fmt::Display for OverlapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Requests with indices {} and {} overlap on berth with index {}",
            self.left, self.right, self.berth
        )
    }
}

impl std::error::Error for OverlapError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SchedulingError {
    NotAllowedOnBerth(NotAllowedOnBerthError),
    FeasiblyWindowViolation(FeasiblyWindowViolationError),
    Overlap(OverlapError),
}

impl std::fmt::Display for SchedulingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulingError::NotAllowedOnBerth(e) => e.fmt(f),
            SchedulingError::FeasiblyWindowViolation(e) => e.fmt(f),
            SchedulingError::Overlap(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SchedulingError {}

impl From<NotAllowedOnBerthError> for SchedulingError {
    fn from(e: NotAllowedOnBerthError) -> Self {
        SchedulingError::NotAllowedOnBerth(e)
    }
}

impl From<FeasiblyWindowViolationError> for SchedulingError {
    fn from(e: FeasiblyWindowViolationError) -> Self {
        SchedulingError::FeasiblyWindowViolation(e)
    }
}

impl From<OverlapError> for SchedulingError {
    fn from(e: OverlapError) -> Self {
        SchedulingError::Overlap(e)
    }
}
