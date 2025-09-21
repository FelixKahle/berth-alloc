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
    registry::err::{LedgerComitError, LedgerUncomitError},
    terminal::err::{TerminalApplyError, TerminalUpdateError},
};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::{
    prelude::{BerthIdentifier, RequestIdentifier},
    problem::err::{AssignmentError, AssignmentOverlapError},
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
    Ledger(LedgerComitError<T>),
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

impl<T> From<LedgerComitError<T>> for ProposeAssignmentError<T> {
    fn from(err: LedgerComitError<T>) -> Self {
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
        ProposeAssignmentError::Ledger(LedgerComitError::from(err))
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnassignedRequestError {
    id: RequestIdentifier,
}

impl UnassignedRequestError {
    #[inline]
    pub fn new(id: RequestIdentifier) -> Self {
        Self { id }
    }

    #[inline]
    pub fn id(&self) -> RequestIdentifier {
        self.id
    }
}

impl std::fmt::Display for UnassignedRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {} is not assigned", self.id)
    }
}

impl std::error::Error for UnassignedRequestError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnassignedRequestsError {
    errors: Vec<UnassignedRequestError>,
}

impl UnassignedRequestsError {
    #[inline]
    pub fn new(errors: Vec<UnassignedRequestError>) -> Self {
        Self { errors }
    }

    #[inline]
    pub fn errors(&self) -> &[UnassignedRequestError] {
        &self.errors
    }
}

impl std::fmt::Display for UnassignedRequestsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ids: Vec<String> = self.errors.iter().map(|e| e.id().to_string()).collect();
        write!(f, "Requests not assigned: {}", ids.join(", "))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeasibilityError<T> {
    Unassigned(UnassignedRequestsError),
    Overlap(AssignmentOverlapError),
    Terminal(TerminalApplyError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for FeasibilityError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeasibilityError::Unassigned(e) => write!(f, "{e}"),
            FeasibilityError::Overlap(e) => write!(f, "{e}"),
            FeasibilityError::Terminal(e) => write!(f, "{e}"),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for FeasibilityError<T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanRejectionError<T> {
    Unassigned(UnassignedRequestsError),
    Overlap(AssignmentOverlapError),
    Terminal(TerminalApplyError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for PlanRejectionError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanRejectionError::Unassigned(e) => write!(f, "{e}"),
            PlanRejectionError::Overlap(e) => write!(f, "{e}"),
            PlanRejectionError::Terminal(e) => write!(f, "{e}"),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for PlanRejectionError<T> {}
