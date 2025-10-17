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

use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::{prelude::BerthIdentifier, problem::err::AssignmentError};

use crate::state::{
    registry::err::{LedgerCommitError, LedgerUncomitError},
    terminal::err::TerminalUpdateError,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BerthNotFreeError<T> {
    id: BerthIdentifier,
    requested: TimeInterval<T>,
    available: TimeInterval<T>,
}

impl<T> BerthNotFreeError<T> {
    #[inline]
    pub fn new(id: BerthIdentifier, requested: TimeInterval<T>, actual: TimeInterval<T>) -> Self {
        Self {
            id,
            requested,
            available: actual,
        }
    }

    #[inline]
    pub fn id(&self) -> BerthIdentifier {
        self.id
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
            self.id, self.requested, self.available
        )
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for BerthNotFreeError<T> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProposeAssignmentError<T> {
    Ledger(LedgerCommitError<T>),
    Terminal(TerminalUpdateError<T>),
    NotFree(BerthNotFreeError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for ProposeAssignmentError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProposeAssignmentError::Ledger(e) => write!(f, "{}", e),
            ProposeAssignmentError::Terminal(e) => write!(f, "{}", e),
            ProposeAssignmentError::NotFree(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for ProposeAssignmentError<T> {}

impl<T> From<LedgerCommitError<T>> for ProposeAssignmentError<T> {
    fn from(err: LedgerCommitError<T>) -> Self {
        ProposeAssignmentError::Ledger(err)
    }
}

impl<T> From<TerminalUpdateError<T>> for ProposeAssignmentError<T> {
    fn from(err: TerminalUpdateError<T>) -> Self {
        ProposeAssignmentError::Terminal(err)
    }
}

impl<T> From<BerthNotFreeError<T>> for ProposeAssignmentError<T> {
    fn from(err: BerthNotFreeError<T>) -> Self {
        ProposeAssignmentError::NotFree(err)
    }
}

impl<T> From<AssignmentError<T>> for ProposeAssignmentError<T> {
    fn from(err: AssignmentError<T>) -> Self {
        ProposeAssignmentError::Ledger(LedgerCommitError::from(err))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProposeUnassignmentError<T> {
    Ledger(LedgerUncomitError),
    Terminal(TerminalUpdateError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for ProposeUnassignmentError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProposeUnassignmentError::Ledger(e) => write!(f, "{}", e),
            ProposeUnassignmentError::Terminal(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for ProposeUnassignmentError<T> {}

impl<T> From<LedgerUncomitError> for ProposeUnassignmentError<T> {
    fn from(err: LedgerUncomitError) -> Self {
        ProposeUnassignmentError::Ledger(err)
    }
}

impl<T> From<TerminalUpdateError<T>> for ProposeUnassignmentError<T> {
    fn from(err: TerminalUpdateError<T>) -> Self {
        ProposeUnassignmentError::Terminal(err)
    }
}
