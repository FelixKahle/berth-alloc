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
    search::scheduling::err::SchedulingError,
    state::{
        chain_set::view::ChainSetView,
        index::{BerthIndex, RequestIndex},
        model::SolverModel,
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::TimeInterval;
use berth_alloc_model::{
    common::FlexibleKind,
    prelude::{
        CrossValidationError, IncompatibleBerthError, MissingFlexibleAssignmentError, SolutionRef,
    },
    problem::asg::AssignmentRef,
    solution::SolutionError,
};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait Scheduler<T: Copy + Ord + CheckedAdd + CheckedSub + Zero + std::fmt::Debug> {
    fn name(&self) -> &str;

    #[inline]
    fn solution<'problem, C: ChainSetView>(
        &self,
        solver_state: &'problem SolverSearchState<T>,
        chains: &C,
    ) -> Result<berth_alloc_model::prelude::SolutionRef<'problem, T>, SolutionError> {
        let model = solver_state.model();
        let problem = model.problem();
        let im = model.index_manager();

        let fixed_refs = problem
            .fixed_assignments()
            .iter()
            .map(|a| a.to_ref())
            .collect();

        let mut flex_refs = berth_alloc_model::problem::asg::AssignmentContainer::<
            FlexibleKind,
            T,
            AssignmentRef<'problem, 'problem, FlexibleKind, T>,
        >::new();

        self.process_schedule(solver_state, chains, |req, berth, iv| {
            let req_id = im.request_id(req).expect("valid request index");
            let berth_id = im.berth_id(berth).expect("valid berth index");

            let req_ref = problem
                .flexible_requests()
                .get(req_id)
                .expect("flex request present");
            let berth_ref = problem.berths().get(berth_id).expect("berth present");

            let start = iv.start();

            let aref = AssignmentRef::<FlexibleKind, T>::new_flexible(req_ref, berth_ref, start)
                .expect("greedy placement must be feasible for assignment ref");

            flex_refs.insert(aref);
        })
        .map_err(|e| map_sched_to_solution(e, model))?;
        SolutionRef::new(fixed_refs, flex_refs, problem)
    }

    #[inline]
    fn check_schedule<C: ChainSetView>(
        &self,
        solver_state: &SolverSearchState<T>,
        chains: &C,
    ) -> Result<(), SchedulingError> {
        self.process_schedule(solver_state, chains, |_, _, _| {})
    }

    fn process_schedule<C, F>(
        &self,
        solver_state: &SolverSearchState<T>,
        chains: &C,
        on_scheduled_item: F,
    ) -> Result<(), SchedulingError>
    where
        C: ChainSetView,
        F: FnMut(RequestIndex, BerthIndex, TimeInterval<T>);
}

fn map_sched_to_solution<T: Copy + Ord + CheckedAdd + CheckedSub>(
    err: SchedulingError,
    model: &SolverModel<T>,
) -> SolutionError {
    let im = model.index_manager();
    match err {
        SchedulingError::NotAllowedOnBerth(e) => {
            let rid = im.request_id(e.request()).expect("map rid");
            let bid = im.berth_id(e.berth()).expect("map bid");
            SolutionError::CrossValidation(CrossValidationError::IncompatibleBerth(
                IncompatibleBerthError::new(rid, bid),
            ))
        }
        SchedulingError::FeasiblyWindowViolation(e) => {
            let rid = im.request_id(e.request()).expect("map rid");
            SolutionError::MissingFlexibleAssignment(MissingFlexibleAssignmentError::new(rid))
        }
        SchedulingError::Overlap(e) => {
            let a = im.request_id(e.right()).expect("map a");
            let b = im.request_id(e.left()).expect("map b");
            SolutionError::CrossValidation(CrossValidationError::Overlap(
                berth_alloc_model::problem::err::AssignmentOverlapError::new(a, b),
            ))
        }
    }
}
