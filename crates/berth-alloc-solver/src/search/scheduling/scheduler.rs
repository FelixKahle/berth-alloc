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
    search::scheduling::{err::SchedulingError, schedule::Schedule},
    state::{chain_set::view::ChainSetView, search_state::SolverSearchState},
};
use berth_alloc_model::{common::FlexibleKind, prelude::SolutionRef, problem::asg::AssignmentRef};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait Scheduler<T: Copy + Ord + CheckedAdd + CheckedSub + Zero> {
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    #[inline]
    fn solution<'problem, C: ChainSetView>(
        &self,
        solver_state: &'problem SolverSearchState<T>,
        chains: &C,
    ) -> Result<berth_alloc_model::prelude::SolutionRef<'problem, T>, SchedulingError>
    where
        T: std::fmt::Debug + std::fmt::Display,
    {
        let model = solver_state.model();
        let problem = model.problem();
        let index_manager = model.index_manager();

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

        self.process_schedule(solver_state, chains, |schedule| {
            let req = schedule.request_index();
            let berth = schedule.berth_index();
            let iv = schedule.assigned_time_interval();
            let req_id = index_manager
                .request_id(req)
                .expect("request_id map is total after preprocessing");

            let berth_id = index_manager
                .berth_id(berth)
                .expect("berth_id map is total after preprocessing");
            let req_ref = problem
                .flexible_requests()
                .get(req_id)
                .expect("flexible request must exist");

            let berth_ref = problem.berths().get(berth_id).expect("berth must exist");

            let start = iv.start();

            let aref = AssignmentRef::<FlexibleKind, T>::new_flexible(req_ref, berth_ref, start)
                .expect("greedy placement must construct a feasible AssignmentRef");

            flex_refs.insert(aref);
        })?;
        Ok(SolutionRef::new(fixed_refs, flex_refs))
    }

    #[inline]
    fn check_schedule<C: ChainSetView>(
        &self,
        solver_state: &SolverSearchState<T>,
        chains: &C,
    ) -> Result<(), SchedulingError> {
        self.process_schedule(solver_state, chains, |_| {})
    }

    fn process_schedule<C, F>(
        &self,
        solver_state: &SolverSearchState<T>,
        chains: &C,
        on_scheduled_item: F,
    ) -> Result<(), SchedulingError>
    where
        C: ChainSetView,
        F: FnMut(&Schedule<T>);
}
