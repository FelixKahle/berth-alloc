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
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    scheduling::err::SchedulingError,
    state::{
        chain_set::{
            index::NodeIndex,
            view::{ChainRef, ChainSetView, ChainViewDyn},
        },
        model::SolverModel,
    },
};
use num_traits::CheckedAdd;

pub trait CalendarScheduler<T: Copy + Ord + CheckedAdd> {
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn schedule_chain<'a, C: ChainSetView>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, C>,
        interval_vars: &mut [IntervalVar<T>], // aligned by RequestIndex = node.get()
        decision_vars: &mut [DecisionVar<T>], // aligned by RequestIndex = node.get()
    ) -> Result<(), SchedulingError> {
        let start = chain.start();
        self.schedule_chain_slice(model, chain, start, None, interval_vars, decision_vars)
    }

    fn schedule_chain_slice<'a, C: ChainSetView>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, C>,
        start_node: NodeIndex,
        end_node_exclusive: Option<NodeIndex>, // None => chain end
        interval_vars: &mut [IntervalVar<T>],  // aligned by RequestIndex = node.get()
        decision_vars: &mut [DecisionVar<T>],  // aligned by RequestIndex = node.get()
    ) -> Result<(), SchedulingError>;

    fn valid_schedule_slice<'a, C: ChainSetView>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, C>,
        start_node: NodeIndex,
        end_node_exclusive: Option<NodeIndex>, // None => chain end
        interval_vars: &[IntervalVar<T>],      // aligned by RequestIndex = node.get()
    ) -> Result<(), SchedulingError>;

    fn valid_schedule<'a, C: ChainSetView>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, C>,
        interval_vars: &[IntervalVar<T>], // aligned by RequestIndex = node.get()
    ) -> Result<(), SchedulingError> {
        let start = chain.start();
        self.valid_schedule_slice(model, chain, start, None, interval_vars)
    }
}

pub trait Propagator<T: Copy + Ord + CheckedAdd> {
    #[inline]
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn propagate(
        &self,
        solver_model: &SolverModel<'_, T>,
        chain: &dyn ChainViewDyn,
        iv: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError>;
}
