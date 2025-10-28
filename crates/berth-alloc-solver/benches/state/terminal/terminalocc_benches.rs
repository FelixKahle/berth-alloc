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

use berth_alloc_core::prelude::{TimeInterval, TimePoint};
use berth_alloc_model::prelude::*;
use berth_alloc_solver::model::index::BerthIndex;
use berth_alloc_solver::state::berth::berthocc::{BerthOccupancy, BerthWrite};
use berth_alloc_solver::state::terminal::delta::TerminalDelta;
use berth_alloc_solver::state::terminal::terminalocc::{
    TerminalOccupancy, TerminalRead, TerminalWrite,
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// Scenario constants: 20 berths, 25 assignments each
const N_BERTHS: usize = 20;
const PER_BERTH: usize = 25;
const PT: i64 = 100; // processing time
const GAP: i64 = 10; // gap between assignments
const HORIZON_START: i64 = 0;
const HORIZON_END: i64 = 3000;

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

// Deterministic mapping for the 20x25 schedule
#[inline]
fn assigned_start_for_req_on_berth(k: usize) -> TimePoint<i64> {
    // k in 0..PER_BERTH on that berth
    tp((k as i64) * (PT + GAP))
}

struct ApplyFixture {
    // Leaked so we can keep references in TerminalOccupancy stable across iterations.
    base: &'static TerminalOccupancy<'static, i64>,
}

// Build a base terminal occupancy with 20 berths and 25 assigned requests per berth,
// placed at non-overlapping intervals with a GAP between them.
fn build_fixture() -> ApplyFixture {
    // Build and leak the berths
    let mut berths: Vec<Berth<i64>> = Vec::with_capacity(N_BERTHS);
    for b in 0..N_BERTHS {
        berths.push(Berth::from_windows(
            bid((b + 1) as u32),
            [iv(HORIZON_START, HORIZON_END)],
        ));
    }
    let leaked_berths: &'static mut Vec<Berth<i64>> = Box::leak(Box::new(berths));

    // Base terminal occupancy
    let mut term = TerminalOccupancy::new(leaked_berths.iter());

    // Occupy PER_BERTH assignments per berth with PT and GAP spacing
    for b in 0..N_BERTHS {
        let bi = BerthIndex::new(b);
        for k in 0..PER_BERTH {
            let start = assigned_start_for_req_on_berth(k);
            let end = tp(start.value() + PT);
            term.occupy(bi, TimeInterval::new(start, end))
                .expect("initial occupy must succeed");
        }
    }

    // Leak base for 'static lifetime
    let base: &'static TerminalOccupancy<'static, i64> = Box::leak(Box::new(term));
    ApplyFixture { base }
}

// Build a delta that, for each berth, moves the first assignment by +GAP/2.
// This keeps the interval non-overlapping: [GAP/2, GAP/2 + PT) ends at PT + GAP/2 < PT + GAP.
fn build_delta_change_one_on_all_berths(
    base: &TerminalOccupancy<'static, i64>,
) -> TerminalDelta<'static, i64> {
    let mut updates: Vec<(BerthIndex, BerthOccupancy<'static, i64>)> =
        Vec::with_capacity(base.berths_len());

    for b in 0..base.berths_len() {
        let bi = BerthIndex::new(b);

        // Clone current occupancy for this berth and adjust one assignment
        let mut occ = base.berths()[b].clone();

        // Old interval is the first assignment on this berth: [0, PT)
        let old_start = assigned_start_for_req_on_berth(0);
        let old_iv = TimeInterval::new(old_start, tp(old_start.value() + PT));

        // New interval shifted by half the gap: [GAP/2, GAP/2 + PT)
        let shift = GAP / 2;
        let new_start = tp(old_start.value() + shift);
        let new_iv = TimeInterval::new(new_start, tp(new_start.value() + PT));

        // Apply change on the cloned occupancy
        occ.release(old_iv).expect("release must succeed");
        occ.occupy(new_iv).expect("occupy must succeed");

        updates.push((bi, occ));
    }

    TerminalDelta::from_updates(updates)
}

fn build_delta_change_one_on_single_berth(
    base: &TerminalOccupancy<'static, i64>,
    touch_idx: usize,
) -> TerminalDelta<'static, i64> {
    let mut updates: Vec<(BerthIndex, BerthOccupancy<'static, i64>)> = Vec::with_capacity(1);

    let bi = BerthIndex::new(touch_idx);
    let mut occ = base.berths()[touch_idx].clone();

    // Old interval is the first assignment on this berth: [0, PT)
    let old_start = assigned_start_for_req_on_berth(0);
    let old_iv = TimeInterval::new(old_start, tp(old_start.value() + PT));

    // New interval shifted by half the gap: [GAP/2, GAP/2 + PT)
    let shift = GAP / 2;
    let new_start = tp(old_start.value() + shift);
    let new_iv = TimeInterval::new(new_start, tp(new_start.value() + PT));

    occ.release(old_iv).expect("release must succeed");
    occ.occupy(new_iv).expect("occupy must succeed");

    updates.push((bi, occ));
    TerminalDelta::from_updates(updates)
}

fn bench_terminal_apply_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("state/apply");

    // Build fixture once
    let fixture = build_fixture();

    group.bench_function("TerminalOccupancy::apply_delta (20x, 1 change each)", |b| {
        b.iter_batched(
            || {
                // Setup per-iteration outside timing:
                // - clone base target
                // - build a delta that changes one assignment on each berth
                let target = fixture.base.clone();
                let delta = build_delta_change_one_on_all_berths(fixture.base);
                (target, delta)
            },
            |(mut target, delta)| {
                // Measured: apply the delta
                target.apply_delta(delta).expect("apply_delta must succeed");
                black_box(target);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_terminal_apply_delta_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("state/apply");

    // Build fixture once
    let fixture = build_fixture();

    group.bench_function("TerminalOccupancy::apply_delta (1x, 1 change)", |b| {
        b.iter_batched(
            || {
                // Per-iteration setup (outside timing)
                let target = fixture.base.clone();
                let delta = build_delta_change_one_on_single_berth(fixture.base, 0);
                (target, delta)
            },
            |(mut target, delta)| {
                // Measured: apply the single-berth delta
                target.apply_delta(delta).expect("apply_delta must succeed");
                black_box(target);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_terminal_apply_delta,
    bench_terminal_apply_delta_single
);
criterion_main!(benches);
