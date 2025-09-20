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

use berth_alloc_core::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutsideAvailabilityError<T> {
    requested: TimeInterval<T>,
}

impl<T> OutsideAvailabilityError<T> {
    pub fn new(requested: TimeInterval<T>) -> Self {
        Self { requested }
    }

    pub fn requested(&self) -> TimeInterval<T>
    where
        T: Copy,
    {
        self.requested
    }
}

impl<T> std::fmt::Display for OutsideAvailabilityError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Requested interval {} is outside of berth availability",
            self.requested
        )
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for OutsideAvailabilityError<T> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NotFreeError<T> {
    requested: TimeInterval<T>,
}

impl<T> NotFreeError<T> {
    pub fn new(requested: TimeInterval<T>) -> Self {
        Self { requested }
    }

    pub fn requested(&self) -> TimeInterval<T>
    where
        T: Copy,
    {
        self.requested
    }
}

impl<T> std::fmt::Display for NotFreeError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Requested interval {} is not free", self.requested)
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for NotFreeError<T> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BerthUpdateError<T> {
    OutsideAvailability(OutsideAvailabilityError<T>),
    NotFree(NotFreeError<T>),
}

impl<T> std::fmt::Display for BerthUpdateError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BerthUpdateError::OutsideAvailability(e) => write!(f, "{}", e),
            BerthUpdateError::NotFree(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for BerthUpdateError<T> {}

impl<T> From<OutsideAvailabilityError<T>> for BerthUpdateError<T> {
    fn from(e: OutsideAvailabilityError<T>) -> Self {
        BerthUpdateError::OutsideAvailability(e)
    }
}

impl<T> From<NotFreeError<T>> for BerthUpdateError<T> {
    fn from(e: NotFreeError<T>) -> Self {
        BerthUpdateError::NotFree(e)
    }
}
