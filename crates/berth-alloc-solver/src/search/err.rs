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
    model::index::{BerthIndex, RequestIndex},
    state::berth::err::BerthUpdateError,
};
use berth_alloc_core::prelude::TimeInterval;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BerthNotFreeError<T> {
    index: BerthIndex,
    requested: TimeInterval<T>,
    available: TimeInterval<T>,
}

impl<T> BerthNotFreeError<T> {
    #[inline]
    pub fn new(index: BerthIndex, requested: TimeInterval<T>, actual: TimeInterval<T>) -> Self {
        Self {
            index,
            requested,
            available: actual,
        }
    }

    #[inline]
    pub fn berth_index(&self) -> BerthIndex {
        self.index
    }

    #[inline]
    pub fn requested(&self) -> TimeInterval<T>
    where
        T: Copy,
    {
        self.requested
    }

    #[inline]
    pub fn available(&self) -> TimeInterval<T>
    where
        T: Copy,
    {
        self.available
    }
}

impl<T> std::fmt::Display for BerthNotFreeError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth {} not free for requested {} (available window: {})",
            self.index, self.requested, self.available
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NotAllowedOnBerthError {
    request_index: RequestIndex,
    berth_index: BerthIndex,
}

impl NotAllowedOnBerthError {
    #[inline]
    pub fn new(request_index: RequestIndex, berth_index: BerthIndex) -> Self {
        Self {
            request_index,
            berth_index,
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
}

impl std::fmt::Display for NotAllowedOnBerthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {} is not allowed on berth {}",
            self.request_index, self.berth_index
        )
    }
}

impl std::error::Error for NotAllowedOnBerthError {}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for BerthNotFreeError<T> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProposeAssignmentError<T> {
    Berth(BerthUpdateError<T>),
    NotAllowedOnBerth(NotAllowedOnBerthError),
    BerthNotFree(BerthNotFreeError<T>),
}

impl<T: std::fmt::Debug + std::fmt::Display> std::fmt::Display for ProposeAssignmentError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProposeAssignmentError::Berth(e) => write!(f, "{}", e),
            ProposeAssignmentError::NotAllowedOnBerth(e) => {
                write!(f, "{}", e)
            }
            ProposeAssignmentError::BerthNotFree(e) => {
                write!(f, "{}", e)
            }
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for ProposeAssignmentError<T> {}

impl<T> From<BerthUpdateError<T>> for ProposeAssignmentError<T> {
    fn from(err: BerthUpdateError<T>) -> Self {
        ProposeAssignmentError::Berth(err)
    }
}

impl<T> From<NotAllowedOnBerthError> for ProposeAssignmentError<T> {
    fn from(err: NotAllowedOnBerthError) -> Self {
        ProposeAssignmentError::NotAllowedOnBerth(err)
    }
}

impl<T> From<BerthNotFreeError<T>> for ProposeAssignmentError<T> {
    fn from(err: BerthNotFreeError<T>) -> Self {
        ProposeAssignmentError::BerthNotFree(err)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NotAssignedError {
    request_index: RequestIndex,
}

impl NotAssignedError {
    #[inline]
    pub fn new(request_index: RequestIndex) -> Self {
        Self { request_index }
    }

    #[inline]
    pub fn request_index(&self) -> RequestIndex {
        self.request_index
    }
}

impl std::fmt::Display for NotAssignedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {} is not assigned", self.request_index)
    }
}

impl std::error::Error for NotAssignedError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProposeUnassignmentError<T> {
    NotAssigned(NotAssignedError),
    Berth(BerthUpdateError<T>),
    NotAllowedOnBerth(NotAllowedOnBerthError),
}

impl<T: std::fmt::Debug + std::fmt::Display> std::fmt::Display for ProposeUnassignmentError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProposeUnassignmentError::NotAssigned(e) => write!(f, "{}", e),
            ProposeUnassignmentError::Berth(e) => write!(f, "{}", e),
            ProposeUnassignmentError::NotAllowedOnBerth(e) => {
                write!(f, "{}", e)
            }
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for ProposeUnassignmentError<T> {}

impl<T> From<NotAssignedError> for ProposeUnassignmentError<T> {
    fn from(err: NotAssignedError) -> Self {
        ProposeUnassignmentError::NotAssigned(err)
    }
}

impl<T> From<BerthUpdateError<T>> for ProposeUnassignmentError<T> {
    fn from(err: BerthUpdateError<T>) -> Self {
        ProposeUnassignmentError::Berth(err)
    }
}

impl<T> From<NotAllowedOnBerthError> for ProposeUnassignmentError<T> {
    fn from(err: NotAllowedOnBerthError) -> Self {
        ProposeUnassignmentError::NotAllowedOnBerth(err)
    }
}
