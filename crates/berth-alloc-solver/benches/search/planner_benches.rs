use std::hint::black_box;
use std::time::{Duration, Instant};

use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
use berth_alloc_model::prelude::*;
use berth_alloc_solver::{
    model::solver_model::SolverModel,
    search::planner::{DefaultCostEvaluator, PlanBuilder},
    state::{
        decisionvar::DecisionVar,
        terminal::terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite},
    },
};
use criterion::{Criterion, criterion_group, criterion_main};

#[inline]
fn tp(v: i64) -> TimePoint<i64> {
    TimePoint::new(v)
}
#[inline]
fn iv(a: i64, b: i64) -> TimeInterval<i64> {
    TimeInterval::new(tp(a), tp(b))
}
#[inline]
fn bid(n: u32) -> BerthIdentifier {
    BerthIdentifier::new(n)
}
#[inline]
fn rid(n: u32) -> RequestIdentifier {
    RequestIdentifier::new(n)
}

fn flex_req(
    id: u32,
    window: (i64, i64),
    pt: &[(u32, i64)],
    weight: i64,
) -> Request<FlexibleKind, i64> {
    let mut m = std::collections::BTreeMap::new();
    for (b, d) in pt {
        m.insert(bid(*b), TimeDelta::new(*d));
    }
    Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
}

fn make_problem_one_berth_n_flex(n: usize, pt: i64, window_end: i64) -> Problem<i64> {
    let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
    // One opening time per berth: a single window [0, window_end)
    berths.insert(Berth::from_windows(bid(1), [iv(0, window_end)]));

    let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

    let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
    for i in 0..n {
        // every request is allowed on berth 1 with the same processing time
        flex.insert(flex_req((i + 1) as u32, (0, window_end), &[(1, pt)], 1));
    }

    Problem::new(berths, fixed, flex).unwrap()
}

fn prefill_one_berth(
    term: &mut TerminalOccupancy<'_, i64>,
    b_ix: berth_alloc_solver::model::index::BerthIndex,
    count: usize,
    pt: i64,
    gap: i64,
) {
    // Occupy sequential slots: [0,pt), [pt+gap, 2*pt+gap), ...
    let mut start = 0i64;
    for _ in 0..count {
        let end = start + pt;
        term.occupy(b_ix, iv(start, end)).expect("occupy ok");
        start = end + gap;
    }
}

pub fn planner_benches(c: &mut Criterion) {
    // Parameters for the scenario
    let pt: i64 = 100; // processing time per vessel
    let gap: i64 = 0; // no gaps between scheduled vessels
    let scheduled: usize = 25; // 25 vessels scheduled on the (single) berth
    let total_requests: usize = 26; // 25 scheduled + 1 we will operate on
    let window_end: i64 = 3000; // room for all scheduled + free tail

    // Build model and base terminal occupancy (with 25 already scheduled)
    let prob = make_problem_one_berth_n_flex(total_requests, pt, window_end);
    let model = SolverModel::try_from(&prob).expect("model ok");

    let b_ix = model
        .index_manager()
        .berth_index(bid(1))
        .expect("berth index exists");

    // Base terminal with scheduled vessels
    let mut base_term = TerminalOccupancy::new(prob.berths().iter());
    prefill_one_berth(&mut base_term, b_ix, scheduled, pt, gap);

    // We'll operate on the last request (id=26)
    let ri = model
        .index_manager()
        .request_index(rid(total_requests as u32))
        .expect("request index exists");

    // Working buffer for decision variables (alive across iterations)
    let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

    // Common free slot for the request on the allowed berth(s)
    // We derive this using a temporary builder to honor the same sandbox logic.
    let tmp_pb = PlanBuilder::new(&model, &base_term, &DefaultCostEvaluator, &mut work_buf);
    let free_slot = tmp_pb
        .sandbox()
        .inner()
        .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(ri))
        .next()
        .expect("some free slot must exist");
    let start_time = free_slot.interval().start();
    drop(tmp_pb);
    // Reset buffer since tmp_pb may have modified it (it did not, but keep explicit)
    work_buf.fill(DecisionVar::unassigned());

    // 1) Benchmark discovery: PlanExplorer::iter_free_for in isolation
    c.bench_function(
        "planner/explorer_iter_free_for_one_berth_25_scheduled",
        |b| {
            b.iter_custom(|iters| {
                // Fresh builder for this benchmark, buffer reused across iterations
                work_buf.fill(DecisionVar::unassigned());
                let pb = PlanBuilder::new(&model, &base_term, &DefaultCostEvaluator, &mut work_buf);

                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let cnt = pb.with_explorer(|ex| ex.iter_free_for(ri).count());
                    total += t0.elapsed();

                    // Prevent optimizer from removing the loop
                    black_box(cnt);
                }
                total
            })
        },
    );

    // 2) Benchmark placing: PlanBuilder::propose_assignment in isolation
    c.bench_function("planner/propose_assignment_one_berth_25_scheduled", |b| {
        b.iter_custom(|iters| {
            work_buf.fill(DecisionVar::unassigned());
            let mut pb = PlanBuilder::new(&model, &base_term, &DefaultCostEvaluator, &mut work_buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Measure only propose_assignment
                let t0 = Instant::now();
                pb.propose_assignment(ri, start_time, &free_slot)
                    .expect("assign ok");
                total += t0.elapsed();

                // Reset outside the timed region for the next iteration
                pb.propose_unassignment(ri).expect("unassign ok");
            }
            total
        })
    });

    // 3) Benchmark freeing: PlanBuilder::propose_unassignment in isolation
    c.bench_function("planner/propose_unassignment_one_berth_25_scheduled", |b| {
        b.iter_custom(|iters| {
            work_buf.fill(DecisionVar::unassigned());
            let mut pb = PlanBuilder::new(&model, &base_term, &DefaultCostEvaluator, &mut work_buf);

            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Prepare outside timed region
                pb.propose_assignment(ri, start_time, &free_slot)
                    .expect("assign ok");

                // Measure only propose_unassignment
                let t0 = Instant::now();
                let fb = pb.propose_unassignment(ri).expect("unassign ok");
                total += t0.elapsed();

                // Prevent optimizer elision
                black_box(fb);
            }
            total
        })
    });
}

criterion_group!(benches, planner_benches);
criterion_main!(benches);
