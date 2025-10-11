use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
use berth_alloc_model::common::FlexibleKind;
use berth_alloc_model::prelude::*;
use berth_alloc_model::problem::builder::ProblemBuilder;
use berth_alloc_model::problem::req::Request;
use berth_alloc_solver::{
    core::intervalvar::IntervalVar,
    scheduling::{tightener::BoundsTightener, traits::Propagator},
    state::{
        chain_set::{
            base::ChainSet,
            delta::{ChainNextRewire, ChainSetDelta},
            index::{ChainIndex, NodeIndex},
            view::{ChainSetView, ChainViewDynAdapter},
        },
        model::SolverModel,
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
        .expect("req ok");
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

/// Build fresh interval variables with identical windows.
fn default_ivars(m: &SolverModel<'_, i64>) -> Vec<IntervalVar<i64>> {
    m.feasible_intervals()
        .iter()
        .map(|w| IntervalVar::new(w.start(), w.end()))
        .collect::<Vec<_>>() // <-- explicit type
}

fn bench_bounds_tightener(c: &mut Criterion) {
    // --- setup ---
    let num_requests = 25;
    let berth_window = (0, 500);
    let request_window = (0, 500);
    let processing_time = 10;

    let problem = build_problem(num_requests, berth_window, request_window, processing_time);
    let model = SolverModel::from_problem(&problem).unwrap();

    let ivars = default_ivars(&model);

    let mut cs = ChainSet::new(model.flexible_requests_len(), model.berths_len());
    link_chain(&mut cs, 0, &(0..num_requests).collect::<Vec<_>>());
    let c0 = cs.chain(ChainIndex(0));
    let dyn_c0 = ChainViewDynAdapter(c0);

    let propagator = BoundsTightener;

    // --- benchmark ---
    c.bench_function("BoundsTightener propagate (25 nodes, 1 berth)", |b| {
        b.iter(|| {
            let mut ivars_copy = ivars.clone();
            black_box(propagator.propagate(&model, &dyn_c0, ivars_copy.as_mut_slice()))
                .expect("valid schedule");
        });
    });
}

criterion_group!(benches, bench_bounds_tightener);
criterion_main!(benches);
