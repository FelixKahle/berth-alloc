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
    overlay::ChainSetOverlay,
    view::ChainSetView,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

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

// Prepare a tiny swap in chain 0: swap successors of p and q,
// where p = node 10 (succ = 11) and q = node 11 (succ = 12) initially.
// After two rewires, overlay yields: ... -> 10 -> 12 -> 11 -> ...
fn make_swap_delta_for_chain0(_: &ChainSet) -> (NodeIndex, NodeIndex, ChainSetDelta) {
    let p = NodeIndex(0 * NODES_PER_CHAIN + 10);
    let a = NodeIndex(0 * NODES_PER_CHAIN + 11);
    let q = a;
    let b = NodeIndex(0 * NODES_PER_CHAIN + 12);

    let mut d = ChainSetDelta::new();
    // Effectively: next[p] = b; next[q] = a  (two rewires),
    // but we write them as two simple next-overrides:
    d.push_rewire(ChainNextRewire::new(p, b));
    d.push_rewire(ChainNextRewire::new(q, a));
    (p, a, d)
}

// -----------------------
// 1) Overlay next() on an UNaffected edge
// -----------------------
fn bench_overlay_next_unaffected(c: &mut Criterion) {
    let base = build_chainset();
    let (_p, _a, delta) = make_swap_delta_for_chain0(&base);
    // Build overlay once (not timed)
    let overlay = ChainSetOverlay::new(&base, &delta);

    // Choose a node far from the swap (chain 5, middle)
    let mid = NodeIndex(5 * NODES_PER_CHAIN + (NODES_PER_CHAIN / 2)); // 5*25+12

    c.bench_function("overlay/single_next_node_unaffected", |b| {
        b.iter(|| {
            let out = overlay.next_node(black_box(mid));
            black_box(out)
        })
    });
}

// -----------------------
// 2) Overlay next() on an overridden edge (the swapped pair)
// -----------------------
fn bench_overlay_next_overridden(c: &mut Criterion) {
    let base = build_chainset();
    let (p, a, delta) = make_swap_delta_for_chain0(&base);
    let overlay = ChainSetOverlay::new(&base, &delta);

    // Measure next() on the tails whose successor was overridden by the delta:
    // - p's successor was changed from a(=11) to b(=12)
    // - a's successor now points to itself in overlay (detached) or to new target depending on your overlay rules
    c.bench_function("overlay/single_next_node_overridden_p", |b| {
        b.iter(|| {
            let out = overlay.next_node(black_box(p));
            black_box(out)
        })
    });

    c.bench_function("overlay/single_next_node_overridden_a", |b| {
        b.iter(|| {
            let out = overlay.next_node(black_box(a));
            black_box(out)
        })
    });
}

// -----------------------
// 3) Iterate all chains via overlay (delta visible)
// -----------------------
fn bench_overlay_iter_all_chains(c: &mut Criterion) {
    let base = build_chainset();
    let (_p, _a, delta) = make_swap_delta_for_chain0(&base);
    let overlay = ChainSetOverlay::new(&base, &delta);

    c.bench_function("overlay/iterate_all_chains_250n_10c", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for ci in 0..NUM_CHAINS {
                for n in overlay.iter_chain(ChainIndex(ci)) {
                    acc = acc.wrapping_add(n.get());
                }
            }
            black_box(acc)
        })
    });
}

criterion_group!(
    benches_overlay,
    bench_overlay_next_unaffected,
    bench_overlay_next_overridden,
    bench_overlay_iter_all_chains
);
criterion_main!(benches_overlay);
