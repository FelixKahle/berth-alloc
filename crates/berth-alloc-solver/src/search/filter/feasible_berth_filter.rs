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
    model::index::{BerthIndex, RequestIndex},
    search::filter::traits::FeasibilityFilter,
    state::{
        chain_set::{
            delta::ChainSetDelta, index::NodeIndex, overlay::ChainSetOverlay, view::ChainSetView,
        },
        search_state::SolverSearchState,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub, Zero};

#[derive(Debug, Clone, Default)]
pub struct FeasibleBerthFilter;

impl FeasibleBerthFilter {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FeasibilityFilter<T> for FeasibleBerthFilter
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Zero + Into<Cost>,
{
    #[inline]
    fn complexity(&self) -> usize {
        10
    }

    #[inline]
    fn is_feasible<'model, 'problem>(
        &self,
        delta: &ChainSetDelta,
        search_state: &SolverSearchState<'model, 'problem, T>,
        _iv: &[IntervalVar<T>],
        dv: &[DecisionVar<T>],
        touched: &[usize],
    ) -> bool {
        let cs = search_state.chain_set();
        let model = search_state.model();
        let overlay = ChainSetOverlay::new(cs, delta);

        for &req_idx in touched {
            let ri = RequestIndex(req_idx);

            // Find which chain the node sits on *after* applying delta (use overlay).
            // NodeIndex == RequestIndex in your setup.
            let n = NodeIndex(ri.get());
            let Some(ci) = overlay.chain_of_node(n).or_else(|| cs.chain_of_node(n)) else {
                // If it’s not on any chain, skip (or return false if that should be illegal)
                continue;
            };
            let bi = BerthIndex(ci.get());

            // 1) Request must be processable on this chain’s berth
            if !matches!(model.processing_time(ri, bi), Some(Some(_))) {
                return false;
            }

            // 2) And DV must target that berth (since DV carries berth index)
            match dv[ri.get()] {
                DecisionVar::Assigned(dec) if dec.berth_index == bi => {}
                _ => return false,
            }
        }
        true
    }
}
