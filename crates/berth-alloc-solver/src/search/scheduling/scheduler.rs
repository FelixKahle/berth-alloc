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
    state::{chain_set::view::ChainSetView, model::SolverModel},
};
use berth_alloc_model::{common::FlexibleKind, prelude::SolutionRef, problem::asg::AssignmentRef};
use num_traits::{CheckedAdd, CheckedSub, Zero};

pub trait Scheduler<T: Copy + Ord + CheckedAdd + CheckedSub + Zero> {
    type ScheduleIter<'run, C>: Iterator<Item = Result<Schedule<T>, SchedulingError>> + 'run
    where
        C: ChainSetView + 'run,
        Self: 'run,
        T: 'run;

    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn schedules<'run, C>(
        &'run self,
        solver_model: &'run SolverModel<'run, T>,
        chains: &'run C,
    ) -> Self::ScheduleIter<'run, C>
    where
        C: ChainSetView + 'run;

    #[inline]
    fn check_schedule<C: ChainSetView>(
        &self,
        solver_model: &SolverModel<T>,
        chains: &C,
    ) -> Result<(), SchedulingError> {
        self.schedules(solver_model, chains)
            .try_for_each(|r| r.map(|_| ()))
    }

    #[inline]
    fn solution<'problem, C: ChainSetView>(
        &self,
        solver_model: &SolverModel<'problem, T>,
        chains: &C,
    ) -> Result<SolutionRef<'problem, T>, SchedulingError>
    where
        T: std::fmt::Debug,
    {
        let problem = solver_model.problem();
        let index_manager = solver_model.index_manager();

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

        for sched in self.schedules(solver_model, chains) {
            let s = sched?;
            let req_id = index_manager.request_id(s.request_index()).expect("total");
            let berth_id = index_manager.berth_id(s.berth_index()).expect("total");
            let req_ref = problem.flexible_requests().get(req_id).expect("exists");
            let berth_ref = problem.berths().get(berth_id).expect("exists");
            let start = s.assigned_time_interval().start();
            let aref = AssignmentRef::<FlexibleKind, T>::new_flexible(req_ref, berth_ref, start)
                .expect("greedy placement must be feasible");
            flex_refs.insert(aref);
        }

        Ok(SolutionRef::new(fixed_refs, flex_refs))
    }
}
