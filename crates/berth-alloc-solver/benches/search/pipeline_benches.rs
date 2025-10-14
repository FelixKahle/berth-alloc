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

use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
use berth_alloc_model::common::FlexibleKind;
use berth_alloc_model::prelude::*;
use berth_alloc_model::problem::builder::ProblemBuilder;
use berth_alloc_model::problem::req::Request;
use berth_alloc_solver::{
    core::{decisionvar::DecisionVar, intervalvar::IntervalVar},
    model::solver_model::SolverModel,
    scheduling::{greedy::GreedyScheduler, tightener::BoundsTightener, traits::Propagator},
    state::chain_set::{
        base::ChainSet,
        delta::{ChainNextRewire, ChainSetDelta},
        index::{ChainIndex, NodeIndex},
        view::ChainSetView,
    },
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::{collections::BTreeMap, hint::black_box};

/// --- helpers ---
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

/// Build a simple 1-berth problem with identical request windows and PTs.
fn build_problem(
    num_requests: usize,
    berth_window: (i64, i64),
    request_window: (i64, i64),
    pt: i64,
) -> Problem<i64> {
    let mut builder = ProblemBuilder::new();
    let berth = Berth::from_windows(bid(0), [iv(berth_window.0, berth_window.1)]);
    builder.add_berth(berth);

    for i in 0..num_requests {
        let mut map = BTreeMap::new();
        map.insert(bid(0), td(pt));
        let req = Request::<FlexibleKind, i64>::new(
            rid(i),
            iv(request_window.0, request_window.1),
            1,
            map,
        )
        .expect("request ok");
        builder.add_flexible(req);
    }

    builder.build().expect("problem ok")
}

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

/// Fresh IVs from model windows.
fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
    m.feasible_intervals()
        .iter()
        .map(|w| IntervalVar::new(w.start(), w.end()))
        .collect()
}

/// Fresh DVs (all Unassigned).
fn default_dvars(m: &SolverModel<'_, i64>) -> Vec<DecisionVar<i64>> {
    vec![DecisionVar::Unassigned; m.flexible_requests_len()]
}

fn bench_pipeline_scheduler(c: &mut Criterion) {
    // --- setup ---
    let num_requests = 25;
    let berth_window = (0, 500);
    let request_window = (0, 500);
    let processing_time = 10;

    let problem = build_problem(num_requests, berth_window, request_window, processing_time);
    let model = SolverModel::from_problem(&problem).unwrap();

    let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
    link_chain(&mut cs, 0, &(0..num_requests).collect::<Vec<_>>());
    let c0 = cs.chain(ChainIndex(0));

    // explicit propagator to mirror your docs bench
    let tightener = BoundsTightener;
    let pipeline = berth_alloc_solver::scheduling::pipeline::SchedulingPipeline::from_propagators(
        [tightener],
        GreedyScheduler,
    );

    c.bench_function(
        "PipelineScheduler (Tightener + Greedy) schedule_chain (25 nodes)",
        |b| {
            b.iter(|| {
                let mut ivars = default_ivars(&model);
                let mut dvars = default_dvars(&model);

                // (Optional) show the propagator alone; mostly to match your narrative.
                // Not required because PipelineScheduler will call it again; comment out if you want pure pipeline cost.
                black_box(BoundsTightener.propagate(&model, c0, ivars.as_mut_slice()))
                    .expect("propagation ok");

                black_box(
                    pipeline
                        .run_base(&model, c0, &mut ivars, &mut dvars)
                        .expect("pipeline schedule ok"),
                );

                // sanity: monotone increasing, contiguous placement
                let d0 = dvars[0].as_assigned().unwrap();
                let d1 = dvars[1].as_assigned().unwrap();
                assert!(d1.start_time >= d0.start_time);
            });
        },
    );
}

criterion_group!(benches, bench_pipeline_scheduler);
criterion_main!(benches);
