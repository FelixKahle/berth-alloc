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
    registry::err::{LedgerCommitError, LedgerUncomitError},
    terminal::err::{TerminalApplyError, TerminalUpdateError},
};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::{
    prelude::{
        BerthIdentifier, CrossValidationError, ExtraFlexibleAssignmentError,
        ExtraFlexibleRequestError, MissingFlexibleAssignmentError, RequestIdNotUniqueError,
    },
    problem::err::AssignmentError,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IncompleteSolverStatePlanApplyError<T> {
    CrossValidation(CrossValidationError),
    ExtraFlexibleAssignment(ExtraFlexibleAssignmentError),
    ExtraFlexibleRequest(ExtraFlexibleRequestError),
    RequestIdNotUnique(RequestIdNotUniqueError),
    TerminalApply(TerminalApplyError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for IncompleteSolverStatePlanApplyError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncompleteSolverStatePlanApplyError::CrossValidation(e) => write!(f, "{}", e),
            IncompleteSolverStatePlanApplyError::ExtraFlexibleAssignment(e) => write!(f, "{}", e),
            IncompleteSolverStatePlanApplyError::ExtraFlexibleRequest(e) => write!(f, "{}", e),
            IncompleteSolverStatePlanApplyError::RequestIdNotUnique(e) => write!(f, "{}", e),
            IncompleteSolverStatePlanApplyError::TerminalApply(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Display + std::fmt::Debug> std::error::Error
    for IncompleteSolverStatePlanApplyError<T>
{
}

impl<T> From<CrossValidationError> for IncompleteSolverStatePlanApplyError<T> {
    fn from(err: CrossValidationError) -> Self {
        IncompleteSolverStatePlanApplyError::CrossValidation(err)
    }
}

impl<T> From<ExtraFlexibleRequestError> for IncompleteSolverStatePlanApplyError<T> {
    fn from(err: ExtraFlexibleRequestError) -> Self {
        IncompleteSolverStatePlanApplyError::ExtraFlexibleRequest(err)
    }
}

impl<T> From<ExtraFlexibleAssignmentError> for IncompleteSolverStatePlanApplyError<T> {
    fn from(err: ExtraFlexibleAssignmentError) -> Self {
        IncompleteSolverStatePlanApplyError::ExtraFlexibleAssignment(err)
    }
}

impl<T> From<RequestIdNotUniqueError> for IncompleteSolverStatePlanApplyError<T> {
    fn from(err: RequestIdNotUniqueError) -> Self {
        IncompleteSolverStatePlanApplyError::RequestIdNotUnique(err)
    }
}

impl<T> From<TerminalApplyError<T>> for IncompleteSolverStatePlanApplyError<T> {
    fn from(err: TerminalApplyError<T>) -> Self {
        IncompleteSolverStatePlanApplyError::TerminalApply(err)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeasibilityError {
    CrossValidation(CrossValidationError),
    MissingFlexibleAssignment(MissingFlexibleAssignmentError),
    ExtraFlexibleAssignment(ExtraFlexibleAssignmentError),
    ExtraFlexibleRequest(ExtraFlexibleRequestError),
    RequestIdNotUnique(RequestIdNotUniqueError),
}

impl std::fmt::Display for FeasibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeasibilityError::CrossValidation(e) => write!(f, "{}", e),
            FeasibilityError::MissingFlexibleAssignment(e) => write!(f, "{}", e),
            FeasibilityError::ExtraFlexibleAssignment(e) => write!(f, "{}", e),
            FeasibilityError::ExtraFlexibleRequest(e) => write!(f, "{}", e),
            FeasibilityError::RequestIdNotUnique(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for FeasibilityError {}

impl From<CrossValidationError> for FeasibilityError {
    fn from(err: CrossValidationError) -> Self {
        FeasibilityError::CrossValidation(err)
    }
}
impl From<MissingFlexibleAssignmentError> for FeasibilityError {
    fn from(err: MissingFlexibleAssignmentError) -> Self {
        FeasibilityError::MissingFlexibleAssignment(err)
    }
}
impl From<ExtraFlexibleAssignmentError> for FeasibilityError {
    fn from(err: ExtraFlexibleAssignmentError) -> Self {
        FeasibilityError::ExtraFlexibleAssignment(err)
    }
}
impl From<ExtraFlexibleRequestError> for FeasibilityError {
    fn from(err: ExtraFlexibleRequestError) -> Self {
        FeasibilityError::ExtraFlexibleRequest(err)
    }
}
impl From<RequestIdNotUniqueError> for FeasibilityError {
    fn from(err: RequestIdNotUniqueError) -> Self {
        FeasibilityError::RequestIdNotUnique(err)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlanRejectionError<T> {
    CrossValidation(CrossValidationError),
    MissingFlexibleAssignment(MissingFlexibleAssignmentError),
    ExtraFlexibleAssignment(ExtraFlexibleAssignmentError),
    ExtraFlexibleRequest(ExtraFlexibleRequestError),
    RequestIdNotUnique(RequestIdNotUniqueError),
    Terminal(TerminalApplyError<T>),
}

impl<T: std::fmt::Display> std::fmt::Display for PlanRejectionError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanRejectionError::CrossValidation(e) => write!(f, "{}", e),
            PlanRejectionError::MissingFlexibleAssignment(e) => write!(f, "{}", e),
            PlanRejectionError::ExtraFlexibleAssignment(e) => write!(f, "{}", e),
            PlanRejectionError::ExtraFlexibleRequest(e) => write!(f, "{}", e),
            PlanRejectionError::RequestIdNotUnique(e) => write!(f, "{}", e),
            PlanRejectionError::Terminal(e) => write!(f, "{}", e),
        }
    }
}

impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for PlanRejectionError<T> {}

impl<T> From<CrossValidationError> for PlanRejectionError<T> {
    fn from(err: CrossValidationError) -> Self {
        PlanRejectionError::CrossValidation(err)
    }
}
impl<T> From<MissingFlexibleAssignmentError> for PlanRejectionError<T> {
    fn from(err: MissingFlexibleAssignmentError) -> Self {
        PlanRejectionError::MissingFlexibleAssignment(err)
    }
}
impl<T> From<ExtraFlexibleAssignmentError> for PlanRejectionError<T> {
    fn from(err: ExtraFlexibleAssignmentError) -> Self {
        PlanRejectionError::ExtraFlexibleAssignment(err)
    }
}
impl<T> From<ExtraFlexibleRequestError> for PlanRejectionError<T> {
    fn from(err: ExtraFlexibleRequestError) -> Self {
        PlanRejectionError::ExtraFlexibleRequest(err)
    }
}
impl<T> From<RequestIdNotUniqueError> for PlanRejectionError<T> {
    fn from(err: RequestIdNotUniqueError) -> Self {
        PlanRejectionError::RequestIdNotUnique(err)
    }
}
impl<T> From<TerminalApplyError<T>> for PlanRejectionError<T> {
    fn from(err: TerminalApplyError<T>) -> Self {
        PlanRejectionError::Terminal(err)
    }
}
