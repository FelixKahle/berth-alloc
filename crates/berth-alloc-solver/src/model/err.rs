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

use berth_alloc_model::{prelude::RequestIdentifier, problem::err::BerthNotFoundError};

#[derive(Debug, Clone)]
pub struct MissingRequestError {
    id: RequestIdentifier,
}

impl MissingRequestError {
    pub fn new(id: RequestIdentifier) -> Self {
        Self { id }
    }

    pub fn id(&self) -> RequestIdentifier {
        self.id
    }
}

impl std::fmt::Display for MissingRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request with ID {} is missing", self.id)
    }
}

impl std::error::Error for MissingRequestError {}

#[derive(Debug, Clone)]
pub enum SolverModelBuildError {
    MissingRequest(MissingRequestError),
    BerthNotFound(BerthNotFoundError),
}

impl std::fmt::Display for SolverModelBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverModelBuildError::MissingRequest(err) => write!(f, "{}", err),
            SolverModelBuildError::BerthNotFound(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for SolverModelBuildError {}

impl From<MissingRequestError> for SolverModelBuildError {
    fn from(err: MissingRequestError) -> Self {
        SolverModelBuildError::MissingRequest(err)
    }
}

impl From<BerthNotFoundError> for SolverModelBuildError {
    fn from(err: BerthNotFoundError) -> Self {
        SolverModelBuildError::BerthNotFound(err)
    }
}
