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

use berth_alloc_core::prelude::{TimeInterval, TimePoint};

use crate::problem::{BerthIdentifier, req::RequestIdentifier};
use std::num::ParseIntError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmptyBerthMapError;

impl std::fmt::Display for EmptyBerthMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "The berth map is empty.")
    }
}

impl std::error::Error for EmptyBerthMapError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoFeasibleAssignmentError {
    id: RequestIdentifier,
}

impl NoFeasibleAssignmentError {
    pub fn new(id: RequestIdentifier) -> Self {
        Self { id }
    }

    pub fn id(&self) -> RequestIdentifier {
        self.id
    }
}

impl std::fmt::Display for NoFeasibleAssignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No feasible assignment for request {}", self.id)
    }
}

impl std::error::Error for NoFeasibleAssignmentError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssignmentOverlapError {
    first: RequestIdentifier,
    second: RequestIdentifier,
}

impl AssignmentOverlapError {
    pub fn new(a: RequestIdentifier, b: RequestIdentifier) -> Self {
        Self {
            first: a,
            second: b,
        }
    }

    pub fn first(&self) -> RequestIdentifier {
        self.first
    }

    pub fn second(&self) -> RequestIdentifier {
        self.second
    }
}

impl std::fmt::Display for AssignmentOverlapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assignments for requests {} and {} overlap",
            self.first, self.second
        )
    }
}

impl std::error::Error for AssignmentOverlapError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RequestError {
    EmptyBerthMap(EmptyBerthMapError),
    NoFeasibleAssignment(NoFeasibleAssignmentError),
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::EmptyBerthMap(e) => write!(f, "{}", e),
            RequestError::NoFeasibleAssignment(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for RequestError {}

impl From<EmptyBerthMapError> for RequestError {
    fn from(err: EmptyBerthMapError) -> Self {
        RequestError::EmptyBerthMap(err)
    }
}

impl From<NoFeasibleAssignmentError> for RequestError {
    fn from(err: NoFeasibleAssignmentError) -> Self {
        RequestError::NoFeasibleAssignment(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IncomatibleBerthError {
    request: RequestIdentifier,
    berth: BerthIdentifier,
}

impl IncomatibleBerthError {
    pub fn new(request: RequestIdentifier, berth: BerthIdentifier) -> Self {
        Self { request, berth }
    }

    pub fn request(&self) -> RequestIdentifier {
        self.request
    }

    pub fn berth(&self) -> BerthIdentifier {
        self.berth
    }
}

impl std::fmt::Display for IncomatibleBerthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {} is incompatible with berth {}",
            self.request, self.berth
        )
    }
}

impl std::error::Error for IncomatibleBerthError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BerthNotFoundError {
    request: RequestIdentifier,
    requested_berth: BerthIdentifier,
}

impl BerthNotFoundError {
    pub fn new(request: RequestIdentifier, requested_berth: BerthIdentifier) -> Self {
        Self {
            request,
            requested_berth,
        }
    }

    pub fn request(&self) -> RequestIdentifier {
        self.request
    }

    pub fn requested_berth(&self) -> BerthIdentifier {
        self.requested_berth
    }
}

impl std::fmt::Display for BerthNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Requested berth {} was not found for request {}",
            self.requested_berth, self.request
        )
    }
}

impl std::error::Error for BerthNotFoundError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProblemError {
    FixedAssignmentOverlap(AssignmentOverlapError),
    BerthNotFound(BerthNotFoundError),
}

impl std::fmt::Display for ProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProblemError::FixedAssignmentOverlap(e) => write!(f, "{}", e),
            ProblemError::BerthNotFound(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ProblemError {}

impl From<AssignmentOverlapError> for ProblemError {
    fn from(err: AssignmentOverlapError) -> Self {
        ProblemError::FixedAssignmentOverlap(err)
    }
}

impl From<BerthNotFoundError> for ProblemError {
    fn from(err: BerthNotFoundError) -> Self {
        ProblemError::BerthNotFound(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssignmenStartsBeforeFeasibleWindowError<T> {
    id: RequestIdentifier,
    start_time: TimePoint<T>,
    assigned_time: TimePoint<T>,
}

impl<T> AssignmenStartsBeforeFeasibleWindowError<T> {
    pub fn new(
        id: RequestIdentifier,
        start_time: TimePoint<T>,
        assigned_time: TimePoint<T>,
    ) -> Self {
        Self {
            id,
            start_time,
            assigned_time,
        }
    }

    pub fn id(&self) -> RequestIdentifier {
        self.id
    }

    pub fn start_time(&self) -> TimePoint<T>
    where
        T: Copy,
    {
        self.start_time
    }

    pub fn assigned_time(&self) -> TimePoint<T>
    where
        T: Copy,
    {
        self.assigned_time
    }
}

impl<T> std::fmt::Display for AssignmenStartsBeforeFeasibleWindowError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assignment for request {} starts before its allowed time window (assigned: {}, window start: {})",
            self.id, self.assigned_time, self.start_time
        )
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error
    for AssignmenStartsBeforeFeasibleWindowError<T>
{
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssignmentEndsAfterFeasibleWindowError<T> {
    id: RequestIdentifier,
    end_time: TimePoint<T>,
    window: TimeInterval<T>,
}

impl<T> AssignmentEndsAfterFeasibleWindowError<T> {
    pub fn new(id: RequestIdentifier, end_time: TimePoint<T>, window: TimeInterval<T>) -> Self {
        Self {
            id,
            end_time,
            window,
        }
    }

    pub fn id(&self) -> RequestIdentifier {
        self.id
    }

    pub fn end_time(&self) -> TimePoint<T>
    where
        T: Copy,
    {
        self.end_time
    }

    pub fn window(&self) -> TimeInterval<T>
    where
        T: Copy,
    {
        self.window
    }
}

impl<T> std::fmt::Display for AssignmentEndsAfterFeasibleWindowError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assignment for request {} ends after its feasible time window (assigned end: {}, window: {})",
            self.id, self.end_time, self.window
        )
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error
    for AssignmentEndsAfterFeasibleWindowError<T>
{
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssignmentError<T> {
    Incompatible(IncomatibleBerthError),
    AssignmentStartsBeforeFeasibleWindow(AssignmenStartsBeforeFeasibleWindowError<T>),
    AssignmentEndsAfterFeasibleWindow(AssignmentEndsAfterFeasibleWindowError<T>),
}

impl<T> std::fmt::Display for AssignmentError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssignmentError::Incompatible(e) => write!(f, "{}", e),
            AssignmentError::AssignmentStartsBeforeFeasibleWindow(e) => write!(f, "{}", e),
            AssignmentError::AssignmentEndsAfterFeasibleWindow(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for AssignmentError<T> {}

impl<T> From<IncomatibleBerthError> for AssignmentError<T> {
    fn from(err: IncomatibleBerthError) -> Self {
        AssignmentError::Incompatible(err)
    }
}

impl<T> From<AssignmenStartsBeforeFeasibleWindowError<T>> for AssignmentError<T> {
    fn from(err: AssignmenStartsBeforeFeasibleWindowError<T>) -> Self {
        AssignmentError::AssignmentStartsBeforeFeasibleWindow(err)
    }
}

impl<T> From<AssignmentEndsAfterFeasibleWindowError<T>> for AssignmentError<T> {
    fn from(err: AssignmentEndsAfterFeasibleWindowError<T>) -> Self {
        AssignmentError::AssignmentEndsAfterFeasibleWindow(err)
    }
}

#[derive(Debug)]
pub enum ProblemLoaderError {
    Io(std::io::Error),
    ParseInt(ParseIntError),
    UnexpectedEof,
    NonPositiveCounts,
    NoFeasibleRequest(RequestIdentifier),
    Request(RequestError),
    Problem(ProblemError),
}

impl From<std::io::Error> for ProblemLoaderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParseIntError> for ProblemLoaderError {
    fn from(e: ParseIntError) -> Self {
        Self::ParseInt(e)
    }
}

impl From<RequestError> for ProblemLoaderError {
    fn from(e: RequestError) -> Self {
        Self::Request(e)
    }
}

impl From<ProblemError> for ProblemLoaderError {
    fn from(e: ProblemError) -> Self {
        Self::Problem(e)
    }
}

impl std::fmt::Display for ProblemLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ProblemLoaderError::*;
        match self {
            Io(e) => write!(f, "I/O error: {e}"),
            ParseInt(e) => write!(f, "parse-int error: {e}"),
            UnexpectedEof => write!(f, "unexpected end of file while parsing instance"),
            NonPositiveCounts => write!(f, "N and M must be positive"),
            NoFeasibleRequest(id) => write!(f, "no feasible berth remains for request {id}"),
            Request(e) => write!(f, "request error: {e}"),
            Problem(e) => write!(f, "problem error: {e}"),
        }
    }
}

impl std::error::Error for ProblemLoaderError {}
