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
    framework::err::{FeasibilityError, IncompleteSolverStatePlanApplyError},
    terminal::err::BerthIdentifierNotFoundError,
};
use berth_alloc_model::solution::SolutionError;

#[derive(Debug)]
pub enum GreedyError<T> {
    PlanApply(IncompleteSolverStatePlanApplyError<T>),
    Feasible(FeasibilityError),
    Solution(SolutionError),
    BerthIdentifierNotFound(BerthIdentifierNotFoundError),
}

impl<T> From<IncompleteSolverStatePlanApplyError<T>> for GreedyError<T> {
    fn from(e: IncompleteSolverStatePlanApplyError<T>) -> Self {
        GreedyError::PlanApply(e)
    }
}

impl<T> From<FeasibilityError> for GreedyError<T> {
    fn from(e: FeasibilityError) -> Self {
        GreedyError::Feasible(e)
    }
}

impl<T> From<berth_alloc_model::solution::SolutionError> for GreedyError<T> {
    fn from(e: berth_alloc_model::solution::SolutionError) -> Self {
        GreedyError::Solution(e)
    }
}

impl<T> From<BerthIdentifierNotFoundError> for GreedyError<T> {
    fn from(e: BerthIdentifierNotFoundError) -> Self {
        GreedyError::BerthIdentifierNotFound(e)
    }
}

impl<T: std::fmt::Display> std::fmt::Display for GreedyError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GreedyError::PlanApply(e) => write!(f, "plan apply: {e}"),
            GreedyError::Feasible(e) => write!(f, "feasibility: {e}"),
            GreedyError::Solution(e) => write!(f, "solution: {e}"),
            GreedyError::BerthIdentifierNotFound(e) => write!(f, "berth identifier not found: {e}"),
        }
    }
}
impl<T: std::fmt::Debug + std::fmt::Display> std::error::Error for GreedyError<T> {}
