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

use crate::state::berth::err::{BerthApplyError, BerthUpdateError};
use berth_alloc_model::prelude::BerthIdentifier;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BerthIdentifierNotFoundError {
    identifier: BerthIdentifier,
}

impl BerthIdentifierNotFoundError {
    pub fn new(identifier: BerthIdentifier) -> Self {
        Self { identifier }
    }

    pub fn identifier(&self) -> BerthIdentifier {
        self.identifier
    }
}

impl std::fmt::Display for BerthIdentifierNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Berth identifier {} not found", self.identifier)
    }
}

impl std::error::Error for BerthIdentifierNotFoundError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TerminalUpdateError<T> {
    BerthIdentifierNotFound(BerthIdentifierNotFoundError),
    BerthUpdate(BerthUpdateError<T>),
}

impl<T> std::fmt::Display for TerminalUpdateError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminalUpdateError::BerthIdentifierNotFound(e) => write!(f, "{}", e),
            TerminalUpdateError::BerthUpdate(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for TerminalUpdateError<T> {}

impl<T> From<BerthIdentifierNotFoundError> for TerminalUpdateError<T> {
    fn from(err: BerthIdentifierNotFoundError) -> Self {
        TerminalUpdateError::BerthIdentifierNotFound(err)
    }
}

impl<T> From<BerthUpdateError<T>> for TerminalUpdateError<T> {
    fn from(err: BerthUpdateError<T>) -> Self {
        TerminalUpdateError::BerthUpdate(err)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TerminalApplyError<T> {
    BerthIdentifierNotFound(BerthIdentifierNotFoundError),
    BerthApply(BerthApplyError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for TerminalApplyError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminalApplyError::BerthIdentifierNotFound(e) => write!(f, "{}", e),
            TerminalApplyError::BerthApply(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for TerminalApplyError<T> {}

impl<T> From<BerthIdentifierNotFoundError> for TerminalApplyError<T> {
    fn from(err: BerthIdentifierNotFoundError) -> Self {
        TerminalApplyError::BerthIdentifierNotFound(err)
    }
}

impl<T> From<BerthApplyError<T>> for TerminalApplyError<T> {
    fn from(err: BerthApplyError<T>) -> Self {
        TerminalApplyError::BerthApply(err)
    }
}
