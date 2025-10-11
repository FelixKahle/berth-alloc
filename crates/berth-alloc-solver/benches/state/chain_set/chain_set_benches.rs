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

use berth_alloc_solver::state::chain_set::{
    base::ChainSet,
    delta::{ChainNextRewire, ChainSetDelta},
    index::{ChainIndex, NodeIndex},
    view::ChainSetView,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Instant};

// -----------------------
// Problem size constants
// -----------------------
const NUM_NODES: usize = 250;
const NUM_CHAINS: usize = 10;
const NODES_PER_CHAIN: usize = NUM_NODES / NUM_CHAINS; // 25

// Build: for chain c, start(c) -> c*25 .. c*25+24 -> end(c)
fn build_chainset() -> ChainSet {
    let mut cs = ChainSet::new(NUM_NODES, NUM_CHAINS);
    for c in 0..NUM_CHAINS {
        let chain = ChainIndex(c);
        let s = cs.start_of_chain(chain);
        let e = cs.end_of_chain(chain);
        let mut tail = s;
        for j in 0..NODES_PER_CHAIN {
            let n = NodeIndex(c * NODES_PER_CHAIN + j);
            let mut d = ChainSetDelta::new();
            d.push_rewire(ChainNextRewire::new(tail, n));
            cs.apply_delta(d);
            tail = n;
        }
        let mut d_end = ChainSetDelta::new();
        d_end.push_rewire(ChainNextRewire::new(tail, e));
        cs.apply_delta(d_end);
    }
    cs
}

// -----------------------
// 1) Single next_node()
// -----------------------
fn bench_single_next(c: &mut Criterion) {
    let cs = build_chainset();
    // pick a stable, on-chain node (avoid sentinels / self-loops)
    let mid = NodeIndex(5 * NODES_PER_CHAIN + (NODES_PER_CHAIN / 2)); // 5*25+12

    c.bench_function("chainset/single_next_node", |b| {
        b.iter(|| {
            // black_box both input and output so LLVM can't elide / hoist
            let out = cs.next_node(black_box(mid));
            black_box(out)
        })
    });
}

// ---------------------------------------------------------
// 2) Single apply_delta (measure ONLY the commit)
//    Use exactly one rewire per delta and alternate A/B.
// ---------------------------------------------------------
fn bench_single_apply_delta(c: &mut Criterion) {
    c.bench_function("chainset/single_apply_delta_one_rewire", |b| {
        b.iter_custom(|iters| {
            // ---------- setup (NOT timed) ----------
            let mut cs = build_chainset();

            // Choose a tail and two different successors within the same chain.
            // Using adjacent nodes keeps the path small and predictable.
            let tail = NodeIndex(0 * NODES_PER_CHAIN + 10);
            let s1 = NodeIndex(0 * NODES_PER_CHAIN + 11);
            let s2 = NodeIndex(0 * NODES_PER_CHAIN + 12);

            // Build two one-rewire deltas that set next[tail] = s1 (A) and = s2 (B).
            // Note: ChainSet::set_next will handle detaching the prior successor.
            let mut delta_a = ChainSetDelta::new();
            delta_a.push_rewire(ChainNextRewire::new(tail, s1));

            let mut delta_b = ChainSetDelta::new();
            delta_b.push_rewire(ChainNextRewire::new(tail, s2));

            // Pre-create the sequence of deltas so construction isn't measured.
            let mut deltas: Vec<ChainSetDelta> = Vec::with_capacity(iters as usize);
            for i in 0..iters {
                // Clone is cheap (very small vec); not timed.
                deltas.push(if i % 2 == 0 {
                    delta_a.clone()
                } else {
                    delta_b.clone()
                });
            }

            // ---------- timed region ----------
            let start = Instant::now();
            for d in deltas {
                // This is what we want to measure: the commit path (stores + detach).
                cs.apply_delta(d);
                black_box(&mut cs); // consume the mutation to avoid elision
            }
            start.elapsed()
        })
    });
}

// ---------------------------------------------------------
// 3) Iterate over all chains (consume all 250 nodes)
// ---------------------------------------------------------
fn bench_iter_all_chains(c: &mut Criterion) {
    let cs = build_chainset();

    c.bench_function("chainset/iterate_all_chains_250n_10c", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for ci in 0..NUM_CHAINS {
                for n in cs.iter_chain(ChainIndex(ci)) {
                    // Consume so the loop can't be optimized away
                    acc = acc.wrapping_add(n.get());
                }
            }
            black_box(acc)
        })
    });
}

criterion_group!(
    benches,
    bench_single_next,
    bench_single_apply_delta,
    bench_iter_all_chains
);
criterion_main!(benches);
