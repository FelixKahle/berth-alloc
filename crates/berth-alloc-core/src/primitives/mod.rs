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

pub mod affine;
pub mod interval;

pub use affine::*;
pub use interval::Interval;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeMarker;

impl MarkerName for TimeMarker {
    const NAME_POINT: &'static str = "TimePoint";
    const NAME_DELTA: &'static str = "TimeDelta";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositionMarker;

impl MarkerName for PositionMarker {
    const NAME_POINT: &'static str = "PositionPoint";
    const NAME_DELTA: &'static str = "PositionDelta";
}

pub type TimePoint<T> = Point<T, TimeMarker>;
pub type PositionPoint<T> = Point<T, PositionMarker>;
pub type TimeDelta<T> = Delta<T, TimeMarker>;
pub type PositionDelta<T> = Delta<T, PositionMarker>;
pub type TimeInterval<T> = Interval<TimePoint<T>>;
pub type PositionInterval<T> = Interval<PositionPoint<T>>;
