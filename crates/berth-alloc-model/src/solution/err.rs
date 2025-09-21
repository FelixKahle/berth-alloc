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

use crate::problem::err::{
    AssignmenStartsBeforeFeasibleWindowError, AssignmentEndsAfterFeasibleWindowError,
    AssignmentOverlapError, BerthNotFoundError, IncomatibleBerthError,
};
use crate::problem::req::RequestIdentifier;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolutionValidationError<T> {
    MissingFixed(RequestIdentifier),
    MissingFlexible(RequestIdentifier),
    ExtraFlexible(RequestIdentifier),
    ExtraFixed(RequestIdentifier),
    UnknownBerth(BerthNotFoundError),
    Incompatible(IncomatibleBerthError),
    AssignmentStartsBeforeFeasibleWindow(AssignmenStartsBeforeFeasibleWindowError<T>),
    AssignmentEndsAfterFeasibleWindow(AssignmentEndsAfterFeasibleWindowError<T>),
    Overlap(AssignmentOverlapError),
}

impl<T> std::fmt::Display for SolutionValidationError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolutionValidationError::MissingFixed(id) => {
                write!(f, "Missing fixed assignment for request {}", id)
            }
            SolutionValidationError::MissingFlexible(id) => {
                write!(f, "Missing flexible assignment for request {}", id)
            }
            SolutionValidationError::ExtraFlexible(id) => {
                write!(f, "Extra flexible assignment for request {}", id)
            }
            SolutionValidationError::ExtraFixed(id) => {
                write!(f, "Extra fixed assignment for request {}", id)
            }
            SolutionValidationError::UnknownBerth(err) => write!(f, "{}", err),
            SolutionValidationError::Incompatible(err) => write!(f, "{}", err),
            SolutionValidationError::AssignmentStartsBeforeFeasibleWindow(err) => {
                write!(f, "{}", err)
            }
            SolutionValidationError::AssignmentEndsAfterFeasibleWindow(err) => write!(f, "{}", err),
            SolutionValidationError::Overlap(err) => write!(f, "{}", err),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for SolutionValidationError<T> {}
