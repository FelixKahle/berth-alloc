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

use crate::validation::err::{
    CrossValidationError, ExtraFixedAssignmentError, ExtraFixedRequestError,
    ExtraFlexibleAssignmentError, ExtraFlexibleRequestError, MissingFixedAssignmentError,
    MissingFlexibleAssignmentError, RequestIdNotUniqueError,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolutionError {
    MissingFixedAssignment(MissingFixedAssignmentError),
    MissingFlexibleAssignment(MissingFlexibleAssignmentError),
    ExtraFlexibleRequest(ExtraFlexibleRequestError),
    ExtraFixedRequest(ExtraFixedRequestError),
    ExtraFixedAssignment(ExtraFixedAssignmentError),
    ExtraFlexibleAssignment(ExtraFlexibleAssignmentError),
    RequestIdNotUnique(RequestIdNotUniqueError),
    CrossValidation(CrossValidationError),
}

impl std::fmt::Display for SolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolutionError::MissingFixedAssignment(e) => write!(f, "{}", e),
            SolutionError::MissingFlexibleAssignment(e) => write!(f, "{}", e),
            SolutionError::ExtraFlexibleRequest(e) => write!(f, "{}", e),
            SolutionError::ExtraFixedRequest(e) => write!(f, "{}", e),
            SolutionError::ExtraFixedAssignment(e) => write!(f, "{}", e),
            SolutionError::ExtraFlexibleAssignment(e) => write!(f, "{}", e),
            SolutionError::RequestIdNotUnique(e) => write!(f, "{}", e),
            SolutionError::CrossValidation(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for SolutionError {}

impl From<MissingFixedAssignmentError> for SolutionError {
    fn from(err: MissingFixedAssignmentError) -> Self {
        SolutionError::MissingFixedAssignment(err)
    }
}

impl From<MissingFlexibleAssignmentError> for SolutionError {
    fn from(err: MissingFlexibleAssignmentError) -> Self {
        SolutionError::MissingFlexibleAssignment(err)
    }
}

impl From<ExtraFlexibleRequestError> for SolutionError {
    fn from(err: ExtraFlexibleRequestError) -> Self {
        SolutionError::ExtraFlexibleRequest(err)
    }
}

impl From<ExtraFixedRequestError> for SolutionError {
    fn from(err: ExtraFixedRequestError) -> Self {
        SolutionError::ExtraFixedRequest(err)
    }
}

impl From<ExtraFixedAssignmentError> for SolutionError {
    fn from(err: ExtraFixedAssignmentError) -> Self {
        SolutionError::ExtraFixedAssignment(err)
    }
}

impl From<ExtraFlexibleAssignmentError> for SolutionError {
    fn from(err: ExtraFlexibleAssignmentError) -> Self {
        SolutionError::ExtraFlexibleAssignment(err)
    }
}

impl From<RequestIdNotUniqueError> for SolutionError {
    fn from(err: RequestIdNotUniqueError) -> Self {
        SolutionError::RequestIdNotUnique(err)
    }
}

impl From<CrossValidationError> for SolutionError {
    fn from(err: CrossValidationError) -> Self {
        SolutionError::CrossValidation(err)
    }
}
