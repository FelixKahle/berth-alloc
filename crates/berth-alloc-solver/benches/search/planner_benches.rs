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
use berth_alloc_model::prelude::*;
use berth_alloc_model::problem::builder::ProblemBuilder;
use berth_alloc_solver::model::solver_model::SolverModel;
use berth_alloc_solver::search::eval::DefaultCostEvaluator;
use berth_alloc_solver::search::planner::{PlanBuilder, PlanExplorer};
use berth_alloc_solver::state::decisionvar::DecisionVar;
use berth_alloc_solver::state::terminal::sandbox::TerminalSandbox;
use berth_alloc_solver::state::terminal::terminalocc::{
    TerminalOccupancy, TerminalRead, TerminalWrite,
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

const N_BERTHS: usize = 20;
const PER_BERTH: usize = 25; // 25 assigned requests per berth
const TOTAL_REQUESTS: usize = N_BERTHS * PER_BERTH;
const PT: i64 = 100; // processing time
const GAP: i64 = 10; // free gap between assignments on a berth
const HORIZON_START: i64 = 0;
const HORIZON_END: i64 = 3000; // roomy enough horizon

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
fn bid(n: u32) -> BerthIdentifier {
    BerthIdentifier::new(n)
}
#[inline]
fn rid(n: u32) -> RequestIdentifier {
    RequestIdentifier::new(n)
}

// Deterministic mapping for the 20x25 schedule
#[inline]
fn assigned_berth_index_for_req(req_idx: usize) -> usize {
    req_idx / PER_BERTH
}
#[inline]
fn assigned_start_for_req(req_idx: usize) -> TimePoint<i64> {
    let k = (req_idx % PER_BERTH) as i64;
    tp(k * (PT + GAP))
}

fn build_problem_20x25() -> Problem<i64> {
    let mut builder = ProblemBuilder::new();

    // 20 berths with identical availability windows
    for b in 0..N_BERTHS {
        let b_id = bid((b + 1) as u32);
        let berth = Berth::from_windows(b_id, [iv(HORIZON_START, HORIZON_END)]);
        builder.add_berth(berth);
    }

    // 500 flexible requests, each allowed on all berths with identical PT
    for r in 0..TOTAL_REQUESTS {
        let r_id = rid((r + 1) as u32);
        let mut pt_map = BTreeMap::new();
        for b in 0..N_BERTHS {
            pt_map.insert(bid((b + 1) as u32), td(PT));
        }
        let req =
            Request::<FlexibleKind, i64>::new(r_id, iv(HORIZON_START, HORIZON_END), 1, pt_map)
                .expect("request must be valid");
        builder.add_flexible(req);
    }

    builder.build().expect("problem must build")
}

struct Fixture {
    model: &'static SolverModel<'static, i64>,
    base: &'static TerminalOccupancy<'static, i64>,
    template_vars: Vec<DecisionVar<i64>>, // all 500 pre-assigned
}

fn build_fixture() -> Fixture {
    // Build and leak problem to 'static
    let problem_box = Box::new(build_problem_20x25());
    let problem: &'static Problem<i64> = Box::leak(problem_box);

    // Build and leak model to 'static
    let model = SolverModel::try_from(problem).expect("model must build");
    let model: &'static SolverModel<'static, i64> = Box::leak(Box::new(model));

    // Base terminal occupancy seeded with deterministic assignments
    let mut base = TerminalOccupancy::new(problem.berths().iter());

    // Prepare decision vars (all assigned) and occupy base accordingly
    let mut vars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

    for req_idx in 0..TOTAL_REQUESTS {
        let berth_idx = assigned_berth_index_for_req(req_idx);
        let start = assigned_start_for_req(req_idx);

        let ri = model
            .index_manager()
            .request_index(rid((req_idx + 1) as u32))
            .expect("request index");
        let bi = model
            .index_manager()
            .berth_index(bid((berth_idx + 1) as u32))
            .expect("berth index");

        vars[ri.get()] = DecisionVar::assigned(bi, start);

        let iv_assigned = model.interval(ri, bi, start).expect("interval exists");
        base.occupy(bi, iv_assigned).expect("occupy must succeed");
    }

    // Leak base
    let base: &'static TerminalOccupancy<'static, i64> = Box::leak(Box::new(base));

    Fixture {
        model,
        base,
        template_vars: vars,
    }
}

fn bench_plan_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner/plan_builder");
    let fixture = build_fixture();

    // 1) PlanBuilder::new
    group.bench_function("PlanBuilder::new", |b| {
        b.iter_batched(
            || fixture.template_vars.clone(),
            |mut work| {
                let pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );
                black_box(pb);
            },
            BatchSize::SmallInput,
        )
    });

    // 2) savepoint()
    group.bench_function("PlanBuilder::savepoint", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut work = fixture.template_vars.clone();
            let pb = PlanBuilder::new(
                fixture.model,
                fixture.base,
                &DefaultCostEvaluator,
                work.as_mut_slice(),
            );
            for _ in 0..iters {
                let start = Instant::now();
                let sp = pb.savepoint();
                black_box(sp);
                total += start.elapsed();
            }
            total
        });
    });

    // 3) undo_to()
    group.bench_function("PlanBuilder::undo_to", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // setup state outside timing
                let mut work = fixture.template_vars.clone();
                let mut pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );
                let sp0 = pb.savepoint();

                // Reassign r=1 to another free slot on its berth
                let r_ix = fixture.model.index_manager().request_index(rid(1)).unwrap();
                let req_idx = r_ix.get();
                let berth_idx = assigned_berth_index_for_req(req_idx);
                let bi = fixture
                    .model
                    .index_manager()
                    .berth_index(bid((berth_idx + 1) as u32))
                    .unwrap();
                let pt = fixture.model.processing_time(r_ix, bi).unwrap();

                let free = pb
                    .sandbox()
                    .inner()
                    .iter_free_intervals_for_berths_in([bi], fixture.model.feasible_interval(r_ix))
                    .find(|fb| fb.interval().length().value() >= pt.value())
                    .unwrap();
                let new_start = free.interval().start();
                pb.propose_assignment(r_ix, new_start, &free).unwrap();

                // measured
                let start = Instant::now();
                pb.undo_to(sp0);
                total += start.elapsed();

                black_box(&pb);
            }
            total
        });
    });

    // 4) propose_unassignment()
    group.bench_function("PlanBuilder::propose_unassignment", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut work = fixture.template_vars.clone();
                let mut pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );
                let r_ix = fixture.model.index_manager().request_index(rid(1)).unwrap();

                let start = Instant::now();
                let fb = pb.propose_unassignment(r_ix).unwrap();
                black_box(fb);
                total += start.elapsed();
            }
            total
        });
    });

    // 5) propose_assignment() (reassign path)
    group.bench_function("PlanBuilder::propose_assignment (reassign)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut work = fixture.template_vars.clone();
                let mut pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );

                let r_ix = fixture.model.index_manager().request_index(rid(2)).unwrap();
                let req_idx = r_ix.get();
                let berth_idx = assigned_berth_index_for_req(req_idx);
                let bi = fixture
                    .model
                    .index_manager()
                    .berth_index(bid((berth_idx + 1) as u32))
                    .unwrap();
                let pt = fixture.model.processing_time(r_ix, bi).unwrap();

                let free = pb
                    .sandbox()
                    .inner()
                    .iter_free_intervals_for_berths_in([bi], fixture.model.feasible_interval(r_ix))
                    .find(|fb| fb.interval().length().value() >= pt.value())
                    .unwrap();
                let new_start = free.interval().start();

                let start = Instant::now();
                pb.propose_assignment(r_ix, new_start, &free).unwrap();
                total += start.elapsed();
                black_box(&pb);
            }
            total
        });
    });

    // 6) with_explorer()
    group.bench_function("PlanBuilder::with_explorer", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut work = fixture.template_vars.clone();
            let pb = PlanBuilder::new(
                fixture.model,
                fixture.base,
                &DefaultCostEvaluator,
                work.as_mut_slice(),
            );
            for _ in 0..iters {
                let start = Instant::now();
                pb.with_explorer(|_ex| {});
                total += start.elapsed();
            }
            black_box(&pb);
            total
        });
    });

    // 7) peek_cost()
    group.bench_function("PlanBuilder::peek_cost", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut work = fixture.template_vars.clone();
                let pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );

                let r_ix = fixture.model.index_manager().request_index(rid(3)).unwrap();
                let req_idx = r_ix.get();
                let berth_idx = assigned_berth_index_for_req(req_idx);
                let bi = fixture
                    .model
                    .index_manager()
                    .berth_index(bid((berth_idx + 1) as u32))
                    .unwrap();
                let pt = fixture.model.processing_time(r_ix, bi).unwrap();

                let free = pb
                    .sandbox()
                    .inner()
                    .iter_free_intervals_for_berths_in([bi], fixture.model.feasible_interval(r_ix))
                    .find(|fb| fb.interval().length().value() >= pt.value())
                    .unwrap();
                let start_time = free.interval().start();

                let start = Instant::now();
                let c = pb.peek_cost(r_ix, start_time, &free);
                black_box(c);
                total += start.elapsed();
            }
            total
        });
    });

    // 8) peek_fitness()
    group.bench_function("PlanBuilder::peek_fitness", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut work = fixture.template_vars.clone();
                let pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );

                let start = Instant::now();
                let fit = pb.peek_fitness();
                black_box(fit);
                total += start.elapsed();
            }
            total
        });
    });

    // 9) finalize()
    group.bench_function("PlanBuilder::finalize", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut work = fixture.template_vars.clone();
                let mut pb = PlanBuilder::new(
                    fixture.model,
                    fixture.base,
                    &DefaultCostEvaluator,
                    work.as_mut_slice(),
                );

                // Ensure there is a patch to finalize
                let r_ix = fixture.model.index_manager().request_index(rid(4)).unwrap();
                let req_idx = r_ix.get();
                let berth_idx = assigned_berth_index_for_req(req_idx);
                let bi = fixture
                    .model
                    .index_manager()
                    .berth_index(bid((berth_idx + 1) as u32))
                    .unwrap();
                let pt = fixture.model.processing_time(r_ix, bi).unwrap();
                let free = pb
                    .sandbox()
                    .inner()
                    .iter_free_intervals_for_berths_in([bi], fixture.model.feasible_interval(r_ix))
                    .find(|fb| fb.interval().length().value() >= pt.value())
                    .unwrap();
                let start_time = free.interval().start();
                pb.propose_assignment(r_ix, start_time, &free).unwrap();

                let start = Instant::now();
                let plan = pb.finalize();
                black_box(plan);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn bench_plan_explorer(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner/plan_explorer");
    let fixture = build_fixture();

    // iter_unassigned()
    group.bench_function("PlanExplorer::iter_unassigned", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ex_vars = fixture.template_vars.clone();
                let sandbox = TerminalSandbox::new(fixture.base);
                let explorer = PlanExplorer::new(
                    fixture.model,
                    ex_vars.as_slice(),
                    &DefaultCostEvaluator,
                    &sandbox,
                );
                let start = Instant::now();
                let count = explorer.iter_unassigned().count();
                black_box(count);
                total += start.elapsed();
            }
            total
        });
    });

    // iter_assigned_requests()
    group.bench_function("PlanExplorer::iter_assigned_requests", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ex_vars = fixture.template_vars.clone();
                let sandbox = TerminalSandbox::new(fixture.base);
                let explorer = PlanExplorer::new(
                    fixture.model,
                    ex_vars.as_slice(),
                    &DefaultCostEvaluator,
                    &sandbox,
                );
                let start = Instant::now();
                let count = explorer.iter_assigned_requests().count();
                black_box(count);
                total += start.elapsed();
            }
            total
        });
    });

    // iter_assignments()
    group.bench_function("PlanExplorer::iter_assignments", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ex_vars = fixture.template_vars.clone();
                let sandbox = TerminalSandbox::new(fixture.base);
                let explorer = PlanExplorer::new(
                    fixture.model,
                    ex_vars.as_slice(),
                    &DefaultCostEvaluator,
                    &sandbox,
                );
                let start = Instant::now();
                let count = explorer.iter_assignments().count();
                black_box(count);
                total += start.elapsed();
            }
            total
        });
    });

    // iter_free_for()
    group.bench_function("PlanExplorer::iter_free_for", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ex_vars = fixture.template_vars.clone();
                let sandbox = TerminalSandbox::new(fixture.base);
                let explorer = PlanExplorer::new(
                    fixture.model,
                    ex_vars.as_slice(),
                    &DefaultCostEvaluator,
                    &sandbox,
                );
                let r_ix = fixture.model.index_manager().request_index(rid(5)).unwrap();
                let start = Instant::now();
                let count = explorer.iter_free_for(r_ix).count();
                black_box(count);
                total += start.elapsed();
            }
            total
        });
    });

    // peek_cost()
    group.bench_function("PlanExplorer::peek_cost", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let ex_vars = fixture.template_vars.clone();
                let sandbox = TerminalSandbox::new(fixture.base);
                let explorer = PlanExplorer::new(
                    fixture.model,
                    ex_vars.as_slice(),
                    &DefaultCostEvaluator,
                    &sandbox,
                );

                let r_ix = fixture.model.index_manager().request_index(rid(6)).unwrap();
                let req_idx = r_ix.get();
                let berth_idx = assigned_berth_index_for_req(req_idx);
                let bi = fixture
                    .model
                    .index_manager()
                    .berth_index(bid((berth_idx + 1) as u32))
                    .unwrap();
                let start_time = assigned_start_for_req(req_idx);

                let start = Instant::now();
                let c = explorer.peek_cost(r_ix, start_time, bi);
                black_box(c);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn bench_terminal_read_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner/terminal_read_write");
    let fixture = build_fixture();

    // Baseline occupy/release on TerminalOccupancy
    group.bench_function("TerminalOccupancy::occupy/release (baseline)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut base = fixture.base.clone();
                let bi = fixture.model.index_manager().berth_index(bid(1)).unwrap();
                let iv = iv(90_000, 90_000 + PT);

                let start = Instant::now();
                base.occupy(bi, iv).expect("occupy ok");
                base.release(bi, iv).expect("release ok");
                black_box(&base);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn bench_planner(c: &mut Criterion) {
    bench_plan_builder(c);
    bench_plan_explorer(c);
    bench_terminal_read_write(c);
}

criterion_group!(benches, bench_planner);
criterion_main!(benches);
