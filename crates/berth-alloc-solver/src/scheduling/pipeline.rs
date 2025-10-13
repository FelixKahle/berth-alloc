// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to do so, subject to the following conditions:
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

use crate::core::{decisionvar::DecisionVar, intervalvar::IntervalVar};
use crate::scheduling::{
    err::SchedulingError,
    traits::{Propagator, Scheduler},
};
use crate::state::chain_set::base::ChainSet;
use crate::state::chain_set::index::NodeIndex;
use crate::state::chain_set::overlay::ChainSetOverlay;
use crate::state::chain_set::view::ChainRef;
use crate::state::model::SolverModel;
use num_traits::{CheckedAdd, CheckedSub};
use std::marker::PhantomData;

#[allow(dead_code)]
trait PropagatorObject<T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    fn propagate_overlay<'a>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, ChainSetOverlay>,
        interval_vars: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError> {
        self.propagate_overlay_slice(model, chain, chain.start(), None, interval_vars)
    }

    fn propagate_overlay_slice<'a>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, ChainSetOverlay>,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
        interval_vars: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError>;

    fn propagate_base<'a>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, ChainSet>,
        interval_vars: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError> {
        self.propagate_base_slice(model, chain, chain.start(), None, interval_vars)
    }

    fn propagate_base_slice<'a>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, ChainSet>,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
        interval_vars: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError>;
}

impl<T, P> PropagatorObject<T> for P
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    P: Propagator<T>,
{
    fn propagate_overlay_slice<'a>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, ChainSetOverlay>,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
        interval_vars: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError> {
        self.propagate_slice(model, chain, start_inclusive, end_exclusive, interval_vars)
    }

    fn propagate_base_slice<'a>(
        &self,
        model: &SolverModel<'a, T>,
        chain: ChainRef<'_, ChainSet>,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
        interval_vars: &mut [IntervalVar<T>],
    ) -> Result<(), SchedulingError> {
        self.propagate_slice(model, chain, start_inclusive, end_exclusive, interval_vars)
    }
}

pub struct SchedulingPipeline<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: Scheduler<T>,
{
    propagators: Vec<Box<dyn PropagatorObject<T>>>,
    placer: S,
    _phantom: PhantomData<T>,
}

impl<T, S> SchedulingPipeline<T, S>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
    S: Scheduler<T>,
{
    #[inline]
    pub fn from_propagators<I, P>(props: I, placer: S) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Propagator<T> + 'static,
    {
        let propagators: Vec<Box<dyn PropagatorObject<T>>> = props
            .into_iter()
            .map(|p| Box::new(p) as Box<dyn PropagatorObject<T>>)
            .collect();

        Self {
            propagators,
            placer,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn empty(placer: S) -> Self {
        Self {
            propagators: Vec::new(),
            placer,
            _phantom: PhantomData,
        }
    }

    /// Run on a `ChainSet` (base).
    #[inline]
    pub fn run_slice_base(
        &self,
        model: &SolverModel<'_, T>,
        chain: ChainRef<'_, ChainSet>,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
        iv: &mut [IntervalVar<T>],
        dv: &mut [DecisionVar<T>],
    ) -> Result<(), SchedulingError> {
        for p in &self.propagators {
            p.propagate_base_slice(model, chain, start_inclusive, end_exclusive, iv)?;
        }
        self.placer
            .schedule_chain_slice(model, chain, start_inclusive, end_exclusive, iv, dv)
    }

    #[inline]
    pub fn run_base(
        &self,
        model: &SolverModel<'_, T>,
        chain: ChainRef<'_, ChainSet>,
        iv: &mut [IntervalVar<T>],
        dv: &mut [DecisionVar<T>],
    ) -> Result<(), SchedulingError> {
        self.run_slice_base(model, chain, chain.start(), None, iv, dv)
    }

    /// Run on a `ChainSetOverlay`.
    #[inline]
    pub fn run_slice_overlay(
        &self,
        model: &SolverModel<'_, T>,
        chain: ChainRef<'_, ChainSetOverlay<'_, '_>>,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
        iv: &mut [IntervalVar<T>],
        dv: &mut [DecisionVar<T>],
    ) -> Result<(), SchedulingError> {
        for p in &self.propagators {
            p.propagate_overlay_slice(model, chain, start_inclusive, end_exclusive, iv)?;
        }
        self.placer
            .schedule_chain_slice(model, chain, start_inclusive, end_exclusive, iv, dv)
    }

    #[inline]
    pub fn run_overlay(
        &self,
        model: &SolverModel<'_, T>,
        chain: ChainRef<'_, ChainSetOverlay<'_, '_>>,
        iv: &mut [IntervalVar<T>],
        dv: &mut [DecisionVar<T>],
    ) -> Result<(), SchedulingError> {
        self.run_slice_overlay(model, chain, chain.start(), None, iv, dv)
    }

    /// Add another concrete propagator later.
    pub fn add_propagator<P>(&mut self, p: P)
    where
        P: Propagator<T> + 'static,
    {
        self.propagators.push(Box::new(p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{decisionvar::DecisionVar, intervalvar::IntervalVar};
    use crate::scheduling::{greedy::GreedyScheduler, tightener::BoundsTightener};
    use crate::state::{
        chain_set::{
            base::ChainSet,
            delta::{ChainNextRewire, ChainSetDelta},
            index::{ChainIndex, NodeIndex},
            view::ChainSetView,
        },
        index::BerthIndex,
        model::SolverModel,
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex(n)
    }

    // Build a Problem:
    // - berths_windows[b] = vec![(s,e), ...] availability windows for berth b (ids 0..B-1).
    // - request_windows[r] = (s,e) feasible window for request r (ids 0..R-1).
    // - processing[r][b] = Some(dur) if r allowed on berth b with PT=dur; None otherwise.
    fn build_problem(
        berths_windows: &[Vec<(i64, i64)>],
        request_windows: &[(i64, i64)],
        processing: &[Vec<Option<i64>>],
    ) -> Problem<i64> {
        let b_len = berths_windows.len();
        let r_len = request_windows.len();
        assert_eq!(processing.len(), r_len);
        for row in processing {
            assert_eq!(
                row.len(),
                b_len,
                "processing times per request must match number of berths"
            );
        }

        let mut builder = ProblemBuilder::new();

        for (i, windows) in berths_windows.iter().enumerate() {
            let b = Berth::from_windows(bid(i), windows.iter().map(|&(s, e)| iv(s, e)));
            builder.add_berth(b);
        }

        for (i, &(ws, we)) in request_windows.iter().enumerate() {
            let mut map = BTreeMap::new();
            for (j, p) in processing[i].iter().copied().enumerate() {
                if let Some(dur) = p {
                    map.insert(bid(j), td(dur));
                }
            }
            let req = Request::<FlexibleKind, i64>::new(rid(i), iv(ws, we), 1, map)
                .expect("request should be well-formed");
            builder.add_flexible(req);
        }

        builder.build().expect("problem should build")
    }

    fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
        m.feasible_intervals()
            .iter()
            .map(|w| IntervalVar::new(w.start(), w.end()))
            .collect()
    }
    fn default_dvars(m: &SolverModel<'_, i64>) -> Vec<DecisionVar<i64>> {
        vec![DecisionVar::Unassigned; m.flexible_requests_len()]
    }

    // Link nodes onto chain c: start -> nodes[0] -> ... -> nodes[k] -> end
    fn link_chain(cs: &mut ChainSet, c: usize, nodes: &[usize]) {
        let s = cs.start_of_chain(ChainIndex(c));
        let e = cs.end_of_chain(ChainIndex(c));
        if nodes.is_empty() {
            return;
        }
        let mut delta = ChainSetDelta::new();
        delta.push_rewire(ChainNextRewire::new(s, NodeIndex(nodes[0])));
        for w in nodes.windows(2) {
            delta.push_rewire(ChainNextRewire::new(NodeIndex(w[0]), NodeIndex(w[1])));
        }
        delta.push_rewire(ChainNextRewire::new(NodeIndex(*nodes.last().unwrap()), e));
        cs.apply_delta(delta);
    }

    #[test]
    fn test_pipeline_end_to_end_basic_greedy_with_tightener() {
        // One berth with [0,100), three requests, PTs 5,7,3 => starts 0,5,12
        let p = build_problem(
            &[vec![(0, 100)]],
            &[(0, 100), (0, 100), (0, 100)],
            &[vec![Some(5)], vec![Some(7)], vec![Some(3)]],
        );
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0, 1, 2]);
        let c0 = cs.chain(ChainIndex(0));

        let pipeline = SchedulingPipeline::from_propagators([(BoundsTightener)], GreedyScheduler);

        pipeline
            .run_base(&m, c0, &mut ivars, &mut dvars)
            .expect("pipeline should schedule");

        let d0 = dvars[0].as_assigned().unwrap();
        let d1 = dvars[1].as_assigned().unwrap();
        let d2 = dvars[2].as_assigned().unwrap();
        assert_eq!(d0.berth_index, bi(0));
        assert_eq!(d0.start_time, tp(0));
        assert_eq!(d1.start_time, tp(5));
        assert_eq!(d2.start_time, tp(12));
    }

    #[test]
    fn test_pipeline_tightener_updates_bounds_before_greedy() {
        // Design:
        // - First segment [0,3) is too short for PT=4, so Tightener must raise LB to 10 (next segment).
        // - Second segment [10,30) and window [0,30) ensure feasibility:
        //   Tightener UB becomes latest start 26; Greedy uses finish-cap: UB' = min(30-4, 26-4) = 22 >= 10.
        let p = build_problem(&[vec![(0, 3), (10, 30)]], &[(0, 30)], &[vec![Some(4)]]);
        let m = SolverModel::from_problem(&p).unwrap();

        let mut ivars = default_ivars(&m);
        let mut dvars = default_dvars(&m);

        let mut cs = ChainSet::new(m.flexible_requests_len(), m.berths_len());
        link_chain(&mut cs, 0, &[0]);
        let c0 = cs.chain(ChainIndex(0));

        let pipeline = SchedulingPipeline::from_propagators([(BoundsTightener)], GreedyScheduler);

        pipeline
            .run_base(&m, c0, &mut ivars, &mut dvars)
            .expect("pipeline should schedule");

        // BoundsTightener should raise LB to 10 (first segment can't fit PT=4), and Greedy should assign at 10.
        let d0 = dvars[0].as_assigned().unwrap();
        assert_eq!(d0.start_time, tp(10));
        assert_eq!(ivars[0].start_time_lower_bound, tp(10));
        // UB should be tightened to a latest-start bound and remain >= LB.
        assert!(ivars[0].start_time_upper_bound >= tp(10));
    }
}
