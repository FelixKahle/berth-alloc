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

use berth_alloc_model::{
    prelude::{
        CrossValidationError, ExtraFlexibleAssignmentError, ExtraFlexibleRequestError,
        RequestIdNotUniqueError, RequestIdentifier,
    },
    problem::err::AssignmentError,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LedgerCommitError<T> {
    Assignment(AssignmentError<T>),
    CrossValidation(CrossValidationError),
    ExtraFlexibleAssignment(ExtraFlexibleAssignmentError),
    ExtraFlexibleRequest(ExtraFlexibleRequestError),
    RequestIdNotUnique(RequestIdNotUniqueError),
}

impl<T: std::fmt::Display> std::fmt::Display for LedgerCommitError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Assignment(e) => write!(f, "{}", e),
            Self::CrossValidation(e) => write!(f, "{}", e),
            Self::ExtraFlexibleAssignment(e) => write!(f, "{}", e),
            Self::ExtraFlexibleRequest(e) => write!(f, "{}", e),
            Self::RequestIdNotUnique(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Display + std::fmt::Debug> std::error::Error for LedgerCommitError<T> {}

impl<T> From<AssignmentError<T>> for LedgerCommitError<T> {
    fn from(err: AssignmentError<T>) -> Self {
        LedgerCommitError::Assignment(err)
    }
}

impl<T> From<CrossValidationError> for LedgerCommitError<T> {
    fn from(err: CrossValidationError) -> Self {
        LedgerCommitError::CrossValidation(err)
    }
}

impl<T> From<ExtraFlexibleAssignmentError> for LedgerCommitError<T> {
    fn from(err: ExtraFlexibleAssignmentError) -> Self {
        LedgerCommitError::ExtraFlexibleAssignment(err)
    }
}

impl<T> From<ExtraFlexibleRequestError> for LedgerCommitError<T> {
    fn from(err: ExtraFlexibleRequestError) -> Self {
        LedgerCommitError::ExtraFlexibleRequest(err)
    }
}

impl<T> From<RequestIdNotUniqueError> for LedgerCommitError<T> {
    fn from(err: RequestIdNotUniqueError) -> Self {
        LedgerCommitError::RequestIdNotUnique(err)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LedgerUncomitError(RequestIdentifier);

impl LedgerUncomitError {
    pub fn new(id: RequestIdentifier) -> Self {
        Self(id)
    }

    pub fn id(&self) -> RequestIdentifier {
        self.0
    }
}

impl std::fmt::Display for LedgerUncomitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {} is not committed", self.0)
    }
}

impl std::error::Error for LedgerUncomitError {}
