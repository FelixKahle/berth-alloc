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
    prelude::IncompatibleBerthError,
    problem::{
        err::{AssignmentOverlapError, BerthNotFoundError},
        req::RequestIdentifier,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MissingFixedAssignmentError {
    request_id: RequestIdentifier,
}

impl MissingFixedAssignmentError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for MissingFixedAssignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Missing fixed assignment for request {}",
            self.request_id
        )
    }
}

impl std::error::Error for MissingFixedAssignmentError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MissingFlexibleAssignmentError {
    request_id: RequestIdentifier,
}

impl MissingFlexibleAssignmentError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for MissingFlexibleAssignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Missing flexible assignment for request {}",
            self.request_id
        )
    }
}

impl std::error::Error for MissingFlexibleAssignmentError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtraFlexibleRequestError {
    request_id: RequestIdentifier,
}
impl ExtraFlexibleRequestError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for ExtraFlexibleRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Extra flexible assignment for unknown request {} (not in problem)",
            self.request_id
        )
    }
}

impl std::error::Error for ExtraFlexibleRequestError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtraFixedRequestError {
    request_id: RequestIdentifier,
}

impl ExtraFixedRequestError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for ExtraFixedRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Extra fixed assignment for unknown request {} (not in problem)",
            self.request_id
        )
    }
}

impl std::error::Error for ExtraFixedRequestError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtraFixedAssignmentError {
    request_id: RequestIdentifier,
}
impl ExtraFixedAssignmentError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for ExtraFixedAssignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Multiple fixed assignments for request {} (only one allowed)",
            self.request_id
        )
    }
}
impl std::error::Error for ExtraFixedAssignmentError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtraFlexibleAssignmentError {
    request_id: RequestIdentifier,
}

impl ExtraFlexibleAssignmentError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for ExtraFlexibleAssignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Multiple flexible assignments for request {} (only one allowed)",
            self.request_id
        )
    }
}

impl std::error::Error for ExtraFlexibleAssignmentError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestIdNotUniqueError {
    request_id: RequestIdentifier,
}

impl RequestIdNotUniqueError {
    #[inline]
    pub fn new(request_id: RequestIdentifier) -> Self {
        Self { request_id }
    }
    #[inline]
    pub fn request_id(&self) -> RequestIdentifier {
        self.request_id
    }
}

impl std::fmt::Display for RequestIdNotUniqueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {} must be unique across fixed and flexible assignments",
            self.request_id
        )
    }
}

impl std::error::Error for RequestIdNotUniqueError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CrossValidationError {
    UnknownBerth(BerthNotFoundError),
    Overlap(AssignmentOverlapError),
    IncompatibleBerth(IncompatibleBerthError),
}

impl std::fmt::Display for CrossValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossValidationError::UnknownBerth(e) => write!(f, "{e}"),
            CrossValidationError::Overlap(e) => write!(f, "{e}"),
            CrossValidationError::IncompatibleBerth(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CrossValidationError {}

impl From<BerthNotFoundError> for CrossValidationError {
    fn from(e: BerthNotFoundError) -> Self {
        Self::UnknownBerth(e)
    }
}

impl From<AssignmentOverlapError> for CrossValidationError {
    fn from(e: AssignmentOverlapError) -> Self {
        Self::Overlap(e)
    }
}

impl From<IncompatibleBerthError> for CrossValidationError {
    fn from(e: IncompatibleBerthError) -> Self {
        Self::IncompatibleBerth(e)
    }
}
