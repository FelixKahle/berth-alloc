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
    model::index::{BerthIndex, RequestIndex},
    search::{
        eval::CostEvaluator,
        lns::{RuinOutcome, RuinProcedure, RuinProcedureContext},
        neighboors::NeighborFn,
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::Plan,
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use rand::Rng;
use std::{
    cmp::{max, min},
    collections::HashSet,
};
use std::{collections::VecDeque, ops::Mul};

// A compact record of an assigned request in the current state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AssignedRec<T: Copy + Ord> {
    r: RequestIndex,
    b: BerthIndex,
    s: TimePoint<T>,
}

// Collect all currently assigned requests with their berth and start time.
// Output order is arbitrary; callers may sort as needed.
fn collect_assigned<'p, T: Copy + Ord>(state: &SolverState<'p, T>) -> Vec<AssignedRec<T>> {
    let dvars = state.decision_variables();
    let mut out = Vec::with_capacity(dvars.len());
    for (i, dv) in dvars.iter().enumerate() {
        if let DecisionVar::Assigned(Decision {
            berth_index,
            start_time,
        }) = dv
        {
            out.push(AssignedRec {
                r: RequestIndex::new(i),
                b: *berth_index,
                s: *start_time,
            });
        }
    }
    out
}

#[inline(always)]
fn build_unassign_plan<'b, 'r, 'c, 's, 'm, 'p, T, C, R>(
    ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    victims: &[RequestIndex],
) -> RuinOutcome<'p, T>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    if victims.is_empty() {
        return RuinOutcome::new(Plan::empty(), Vec::new());
    }
    let mut pb = ctx.builder();
    for &r in victims {
        let _ = pb.propose_unassignment(r);
    }
    RuinOutcome::new(pb.finalize(), victims.to_vec())
}

pub struct RandomSubsetRuin {
    k: usize,
}

impl RandomSubsetRuin {
    #[inline]
    pub fn new(k: usize) -> Self {
        Self { k: max(1, k) }
    }
}

// Floyd’s algorithm to sample k distinct indices from 0..n in expected O(k)
#[inline(always)]
fn sample_k_distinct<R: Rng>(rng: &mut R, n: usize, k: usize) -> Vec<usize> {
    use std::collections::HashSet;
    let k = k.min(n);
    let mut chosen: HashSet<usize> = HashSet::with_capacity(k);
    let mut result = Vec::with_capacity(k);
    for j in (n - k)..n {
        let r = rng.random_range(0..=j);
        let x = if chosen.contains(&r) { j } else { r };
        if chosen.insert(x) {
            result.push(x);
        }
    }
    result
}

impl<T, C, R> RuinProcedure<T, C, R> for RandomSubsetRuin
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomSubsetRuin"
    }

    fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T> {
        let all = collect_assigned(ctx.state());
        if all.is_empty() {
            return build_unassign_plan(ctx, &[]);
        }

        let picks = sample_k_distinct(ctx.rng(), all.len(), self.k);
        let victims: Vec<RequestIndex> = picks.into_iter().map(|i| all[i].r).collect();
        build_unassign_plan(ctx, &victims)
    }
}

pub struct TimeBandRuin {
    band_len: usize,
}

impl TimeBandRuin {
    #[inline]
    pub fn new(band_len: usize) -> Self {
        Self {
            band_len: max(1, band_len),
        }
    }
}

impl<T, C, R> RuinProcedure<T, C, R> for TimeBandRuin
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "TimeBandRuin"
    }

    fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T> {
        let mut all = collect_assigned(ctx.state());
        if all.is_empty() {
            return build_unassign_plan(ctx, &[]);
        }

        // Sort by start time
        all.sort_by_key(|a| a.s);
        let n = all.len();
        let win = self.band_len.min(n);

        // Choose a pivot and take a contiguous window of length `win` around it
        let pivot = ctx.rng().random_range(0..n);
        let half = win / 2;
        let start = pivot.saturating_sub(half);
        let start = start.min(n - win);
        let end = start + win;

        let victims: Vec<RequestIndex> = all[start..end].iter().map(|r| r.r).collect();
        build_unassign_plan(ctx, &victims)
    }
}

pub struct SameBerthBlockRuin {
    block_len: usize,
}

impl SameBerthBlockRuin {
    #[inline]
    pub fn new(block_len: usize) -> Self {
        Self {
            block_len: max(1, block_len),
        }
    }
}

impl<T, C, R> RuinProcedure<T, C, R> for SameBerthBlockRuin
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "SameBerthBlockRuin"
    }

    fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T> {
        let mut all = collect_assigned(ctx.state());
        if all.is_empty() {
            return build_unassign_plan(ctx, &[]);
        }

        // Sort by (berth, start) so per-berth blocks are contiguous
        all.sort_by_key(|a| (a.b.get(), a.s));

        // Compute ranges [start, end) per berth in one pass
        let mut ranges: Vec<(usize, usize)> = vec![(0, 0); ctx.model().berths_len()];
        {
            let mut i = 0;
            while i < all.len() {
                let b = all[i].b.get();
                let start = i;
                while i < all.len() && all[i].b.get() == b {
                    i += 1;
                }
                ranges[b] = (start, i);
            }
        }

        // Collect non-empty berth indices
        let mut non_empty: Vec<usize> = Vec::new();
        for (b, (s, e)) in ranges.iter().enumerate() {
            if e > s {
                non_empty.push(b);
            }
        }
        if non_empty.is_empty() {
            return build_unassign_plan(ctx, &[]);
        }

        // Pick a berth uniformly among non-empty ones
        let pick_bi = non_empty[ctx.rng().random_range(0..non_empty.len())];
        let (start_b, end_b) = ranges[pick_bi];
        let len_b = end_b - start_b;

        let win = self.block_len.min(len_b);
        let start_off = if len_b > win {
            ctx.rng().random_range(0..=(len_b - win))
        } else {
            0
        };
        let start = start_b + start_off;
        let end = start + win;

        let victims: Vec<RequestIndex> = all[start..end].iter().map(|r| r.r).collect();
        build_unassign_plan(ctx, &victims)
    }
}

pub struct RandomWalkRuin {
    steps: usize,
    same_berth_bias: f64,
}

impl RandomWalkRuin {
    pub fn new(steps: usize, same_berth_bias: f64) -> Self {
        let bias = same_berth_bias.clamp(0.0, 1.0);
        Self {
            steps: max(1, steps),
            same_berth_bias: bias,
        }
    }
}

impl<T, C, R> RuinProcedure<T, C, R> for RandomWalkRuin
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomWalkRuin"
    }

    fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T> {
        let mut all = collect_assigned(ctx.state());
        if all.is_empty() {
            return build_unassign_plan(ctx, &[]);
        }

        // Sort by (berth, time) and compute berth ranges
        all.sort_by_key(|a| (a.b.get(), a.s));
        let mut ranges: Vec<(usize, usize)> = vec![(0, 0); ctx.model().berths_len()];
        {
            let mut i = 0;
            while i < all.len() {
                let b = all[i].b.get();
                let start = i;
                while i < all.len() && all[i].b.get() == b {
                    i += 1;
                }
                ranges[b] = (start, i);
            }
        }

        let max_steps = min(self.steps, all.len());
        let mut victims: Vec<RequestIndex> = Vec::with_capacity(max_steps);
        let mut removed = vec![false; all.len()];

        // Start anywhere
        let mut cur = ctx.rng().random_range(0..all.len());

        for _ in 0..max_steps {
            if !removed[cur] {
                victims.push(all[cur].r);
                removed[cur] = true;
            }

            let go_same = ctx.rng().random_bool(self.same_berth_bias);
            let next = if go_same {
                // Search within the same berth range: expand left/right until a remaining index is found
                let (start, end) = ranges[all[cur].b.get()];
                if start == end {
                    // empty bucket; fallback global
                    let mut j = ctx.rng().random_range(0..all.len());
                    while removed[j] {
                        j = (j + 1) % all.len();
                    }
                    j
                } else {
                    let mut left = cur;
                    let mut right = cur;
                    let mut found = None;
                    for _ in 0..(end - start) {
                        if left > start {
                            left -= 1;
                            if !removed[left] {
                                found = Some(left);
                                break;
                            }
                        }
                        if right + 1 < end {
                            right += 1;
                            if !removed[right] {
                                found = Some(right);
                                break;
                            }
                        }
                    }
                    found.unwrap_or_else(|| {
                        let mut j = ctx.rng().random_range(0..all.len());
                        while removed[j] {
                            j = (j + 1) % all.len();
                        }
                        j
                    })
                }
            } else {
                // Choose any globally not yet removed (linear probe from a random spot)
                let mut j = ctx.rng().random_range(0..all.len());
                let mut guard = 0usize;
                while removed[j] && guard < all.len() {
                    j = (j + 1) % all.len();
                    guard += 1;
                }
                j
            };

            cur = next;
        }

        build_unassign_plan(ctx, &victims)
    }
}

/// Ruin procedure that removes a cluster of "related" requests.
///
/// It starts from a single random, assigned request and performs a
/// Breadth-First Search (BFS) using the provided `NeighborFn` to find
/// a cluster of other *assigned* requests. It stops after finding `k`
/// victims or exhausting the connected component.
pub struct RelatedRuin<'a> {
    k: usize,
    neighbor_fn: NeighborFn<'a>,
}

impl<'a> RelatedRuin<'a> {
    #[inline]
    pub fn new(k: usize, neighbor_fn: NeighborFn<'a>) -> Self {
        Self {
            k: max(1, k),
            neighbor_fn,
        }
    }
}

impl<'a, T, C, R> RuinProcedure<T, C, R> for RelatedRuin<'a>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RelatedRuin"
    }

    #[tracing::instrument(skip(self, ctx), level = "debug")]
    fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> RuinOutcome<'p, T> {
        let all_assigned = collect_assigned(ctx.state());
        let assigned_set: HashSet<RequestIndex> = all_assigned.iter().map(|rec| rec.r).collect();

        if assigned_set.is_empty() {
            return build_unassign_plan(ctx, &[]);
        }

        let seed_rec = all_assigned[ctx.rng().random_range(0..all_assigned.len())];
        let mut victims: Vec<RequestIndex> = Vec::with_capacity(self.k);
        let mut queue: VecDeque<RequestIndex> = VecDeque::new();
        let mut visited: HashSet<RequestIndex> = HashSet::new();

        queue.push_back(seed_rec.r);
        visited.insert(seed_rec.r);

        while victims.len() < self.k {
            let Some(current_r) = queue.pop_front() else {
                break; // Cluster is exhausted
            };

            victims.push(current_r);
            let add_neighbors =
                |neighbors: &[RequestIndex],
                 queue: &mut VecDeque<RequestIndex>,
                 visited: &mut HashSet<RequestIndex>| {
                    for &neighbor_r in neighbors {
                        if !visited.contains(&neighbor_r) && assigned_set.contains(&neighbor_r) {
                            visited.insert(neighbor_r);
                            queue.push_back(neighbor_r);
                        }
                    }
                };

            match &self.neighbor_fn {
                NeighborFn::Vec(f) => {
                    let neighbors = f(current_r); // This creates a new Vec
                    add_neighbors(&neighbors, &mut queue, &mut visited);
                }
                NeighborFn::Slice(f) => {
                    let neighbors = f(current_r); // This is just a slice
                    add_neighbors(neighbors, &mut queue, &mut visited);
                }
            }
        }

        build_unassign_plan(ctx, &victims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::DefaultCostEvaluator,
        state::{decisionvar::DecisionVarVec, terminal::terminalocc::TerminalWrite},
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use once_cell::sync::Lazy;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng as StdRng;
    use std::{
        collections::{BTreeMap, HashMap},
        sync::Arc,
    };

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pts: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    /// Build a small problem with 2 berths and 4 flexible requests allowed on both,
    /// and return a state where all four are assigned (two per berth).
    fn make_assigned_state() -> (
        SolverModel<'static, i64>,
        SolverState<'static, i64>,
        DefaultCostEvaluator,
    ) {
        let b1 = berth(1, 0, 200);
        let b2 = berth(2, 0, 200);

        let r1 = flex_req(1, (0, 200), &[(1, 20), (2, 20)], 1);
        let r2 = flex_req(2, (0, 200), &[(1, 15), (2, 15)], 1);
        let r3 = flex_req(3, (0, 200), &[(1, 25), (2, 25)], 1);
        let r4 = flex_req(4, (0, 200), &[(1, 10), (2, 10)], 1);

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        pb.add_flexible(r2);
        pb.add_flexible(r3);
        pb.add_flexible(r4);
        // Leak the problem to give it a 'static lifetime for returned model/state in tests
        let problem: &'static Problem<i64> = Box::leak(Box::new(pb.build().unwrap()));

        let model = SolverModel::try_from(problem).unwrap();
        let eval = DefaultCostEvaluator;

        let im = model.index_manager();
        let r1 = im.request_index(rid(1)).unwrap();
        let r2 = im.request_index(rid(2)).unwrap();
        let r3 = im.request_index(rid(3)).unwrap();
        let r4 = im.request_index(rid(4)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        // Assign two per berth in non-overlapping slots.
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1.get()] = DecisionVar::assigned(b1, tp(0));
        dvars[r2.get()] = DecisionVar::assigned(b1, tp(25));
        dvars[r3.get()] = DecisionVar::assigned(b2, tp(0));
        dvars[r4.get()] = DecisionVar::assigned(b2, tp(30));

        // Build TerminalOccupancy from problem berths (avoid borrowing model)
        use crate::state::terminal::terminalocc::TerminalOccupancy;
        let mut term = TerminalOccupancy::new(problem.berths().iter());
        for (r, (b, s)) in [
            (r1, (b1, tp(0))),
            (r2, (b1, tp(25))),
            (r3, (b2, tp(0))),
            (r4, (b2, tp(30))),
        ] {
            let iv = model.interval(r, b, s).unwrap();
            term.occupy(b, iv).unwrap();
        }

        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    fn ruin_ctx<'b, 'r, 'c, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        state: &'s SolverState<'p, i64>,
        eval: &'c DefaultCostEvaluator,
        rng: &'r mut StdRng,
        buf: &'b mut [DecisionVar<i64>],
    ) -> RuinProcedureContext<'b, 'r, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng> {
        RuinProcedureContext::new(model, state, eval, rng, buf)
    }

    // Helper to create a state with r1, r3 assigned and r2, r4 unassigned
    fn make_partially_assigned_state() -> (
        SolverModel<'static, i64>,
        SolverState<'static, i64>,
        DefaultCostEvaluator,
    ) {
        // ... (setup is identical to make_assigned_state) ...
        let b1 = berth(1, 0, 200);
        let b2 = berth(2, 0, 200);
        let r1 = flex_req(1, (0, 200), &[(1, 20), (2, 20)], 1);
        let r2 = flex_req(2, (0, 200), &[(1, 15), (2, 15)], 1);
        let r3 = flex_req(3, (0, 200), &[(1, 25), (2, 25)], 1);
        let r4 = flex_req(4, (0, 200), &[(1, 10), (2, 10)], 1);
        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        pb.add_flexible(r2);
        pb.add_flexible(r3);
        pb.add_flexible(r4);
        let problem: &'static Problem<i64> = Box::leak(Box::new(pb.build().unwrap()));
        let model = SolverModel::try_from(problem).unwrap();
        let eval = DefaultCostEvaluator;
        let im = model.index_manager();

        // --- Dvars ---
        // r1, r3 assigned; r2, r4 unassigned
        let r1_ix = im.request_index(rid(1)).unwrap();
        let _r2_ix = im.request_index(rid(2)).unwrap();
        let r3_ix = im.request_index(rid(3)).unwrap();
        let _r4_ix = im.request_index(rid(4)).unwrap();
        let b1_ix = im.berth_index(bid(1)).unwrap();
        let b2_ix = im.berth_index(bid(2)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1_ix.get()] = DecisionVar::assigned(b1_ix, tp(0));
        dvars[r3_ix.get()] = DecisionVar::assigned(b2_ix, tp(0));

        // --- Terminal ---
        use crate::state::terminal::terminalocc::TerminalOccupancy;
        let mut term = TerminalOccupancy::new(problem.berths().iter());
        for (r, (b, s)) in [(r1_ix, (b1_ix, tp(0))), (r3_ix, (b2_ix, tp(0)))] {
            let iv = model.interval(r, b, s).unwrap();
            term.occupy(b, iv).unwrap();
        }

        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    // Helper to create a state with no requests assigned
    fn make_unassigned_state() -> (
        SolverModel<'static, i64>,
        SolverState<'static, i64>,
        DefaultCostEvaluator,
    ) {
        // ... (setup is identical to make_assigned_state) ...
        let b1 = berth(1, 0, 200);
        let b2 = berth(2, 0, 200);
        let r1 = flex_req(1, (0, 200), &[(1, 20), (2, 20)], 1);
        let r2 = flex_req(2, (0, 200), &[(1, 15), (2, 15)], 1);
        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1);
        pb.add_berth(b2);
        pb.add_flexible(r1);
        pb.add_flexible(r2);
        let problem: &'static Problem<i64> = Box::leak(Box::new(pb.build().unwrap()));
        let model = SolverModel::try_from(problem).unwrap();
        let eval = DefaultCostEvaluator;

        // --- Dvars (all unassigned) ---
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // --- Terminal (empty) ---
        use crate::state::terminal::terminalocc::TerminalOccupancy;
        let term = TerminalOccupancy::new(problem.berths().iter());

        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    // A static neighbor graph for testing.
    // r1 <-> r2 <-> r3 <-> r4
    static TEST_NEIGHBORS: Lazy<HashMap<RequestIndex, Vec<RequestIndex>>> = Lazy::new(|| {
        let mut m = HashMap::new();
        let r1 = RequestIndex::new(0);
        let r2 = RequestIndex::new(1);
        let r3 = RequestIndex::new(2);
        let r4 = RequestIndex::new(3);
        m.insert(r1, vec![r2]);
        m.insert(r2, vec![r1, r3]);
        m.insert(r3, vec![r2, r4]);
        m.insert(r4, vec![r3]);
        m
    });

    // Helper to create a NeighborFn::Vec
    fn create_neighbor_fn_vec() -> NeighborFn<'static> {
        NeighborFn::Vec(Arc::new(|r| {
            TEST_NEIGHBORS.get(&r).cloned().unwrap_or_default()
        }))
    }

    // Helper to create a NeighborFn::Slice
    fn create_neighbor_fn_slice() -> NeighborFn<'static> {
        NeighborFn::Slice(Arc::new(|r| {
            TEST_NEIGHBORS.get(&r).map_or(&[], |v| v.as_slice())
        }))
    }

    #[test]
    fn random_subset_removes_k_or_less() {
        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(7);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let mut ruin = RandomSubsetRuin::new(3);
        let out = ruin.ruin(&mut ctx);
        assert_eq!(out.ruined.len(), 3);
        assert_eq!(out.ruined_plan.decision_var_patches.len(), 3);
        assert!(!out.ruined_plan.terminal_delta.is_empty());
    }

    #[test]
    fn time_band_removes_contiguous_by_time() {
        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(8);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let mut ruin = TimeBandRuin::new(2);
        let out = ruin.ruin(&mut ctx);
        assert!(out.ruined.len() >= 1 && out.ruined.len() <= 2);
        assert_eq!(out.ruined_plan.decision_var_patches.len(), out.ruined.len());
    }

    #[test]
    fn same_berth_block_targets_single_berth() {
        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(9);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let mut ruin = SameBerthBlockRuin::new(2);
        let out = ruin.ruin(&mut ctx);
        // Expect 1..=2 removals
        assert!(out.ruined.len() >= 1 && out.ruined.len() <= 2);
    }

    #[test]
    fn random_walk_removes_up_to_steps() {
        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(10);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let mut ruin = RandomWalkRuin::new(3, 0.7);
        let out = ruin.ruin(&mut ctx);
        assert!(out.ruined.len() >= 1 && out.ruined.len() <= 3);
        assert_eq!(out.ruined_plan.decision_var_patches.len(), out.ruined.len());
    }

    #[test]
    fn sample_k_distinct_returns_unique_indices() {
        // Access private helper from parent module
        let mut rng = StdRng::seed_from_u64(123);
        let n = 10;

        for k in 0..=n {
            let picks = super::sample_k_distinct(&mut rng, n, k);
            assert_eq!(picks.len(), k, "must return exactly k picks");

            // All distinct and within range
            let mut seen = std::collections::HashSet::new();
            for &i in &picks {
                assert!(i < n, "pick out of range: {i} >= {n}");
                assert!(seen.insert(i), "duplicate pick {i} for k={k}");
            }
        }

        // k > n clamps to n
        let picks = super::sample_k_distinct(&mut rng, n, n + 5);
        assert_eq!(picks.len(), n, "k > n should clamp to n");
    }

    #[test]
    fn time_band_respects_exact_band_len() {
        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(12345);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        // We have 4 assigned in make_assigned_state; request band_len=3 should yield exactly 3 removals.
        let mut ruin = TimeBandRuin::new(3);
        let out = ruin.ruin(&mut ctx);
        assert_eq!(
            out.ruined.len(),
            3,
            "time band should remove exactly band_len elements"
        );
        assert_eq!(
            out.ruined_plan.decision_var_patches.len(),
            out.ruined.len(),
            "plan patches should match number of victims"
        );
    }

    #[test]
    fn same_berth_block_is_contiguous_within_berth() {
        use std::collections::{BTreeMap, HashMap};

        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(9876);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        // Remove a block of length 2; verify all victims share one berth and are contiguous in that berth by start time.
        let mut ruin = SameBerthBlockRuin::new(2);
        let out = ruin.ruin(&mut ctx);
        assert!(
            (1..=2).contains(&out.ruined.len()),
            "block ruin should remove between 1 and block_len"
        );

        // Build map RequestIndex -> (berth, start) from current assignments
        let assigned = super::collect_assigned(&state);
        let mut map: HashMap<usize, (u32, i64)> = HashMap::new();
        for rec in &assigned {
            map.insert(rec.r.get(), (rec.b.get() as u32, rec.s.value()));
        }

        // All victims share same berth
        let mut berth_ids: BTreeMap<u32, usize> = BTreeMap::new();
        for &r in &out.ruined {
            let (b, _s) = map.get(&r.get()).expect("victim must be assigned");
            *berth_ids.entry(*b).or_default() += 1;
        }
        assert_eq!(berth_ids.len(), 1, "all victims must be on the same berth");

        // Extract the specific berth
        let (shared_berth, _) = berth_ids.iter().next().unwrap();

        // Build the per-berth ordered sequence by start time
        let mut per_berth_seq: Vec<(usize, i64)> = assigned
            .iter()
            .filter(|rec| (rec.b.get() as u32) == *shared_berth)
            .map(|rec| (rec.r.get(), rec.s.value()))
            .collect();
        per_berth_seq.sort_by_key(|&(_r, s)| s);

        // Map request -> position in per_berth_seq
        let mut pos_by_req: HashMap<usize, usize> = HashMap::new();
        for (idx, &(r_id, _)) in per_berth_seq.iter().enumerate() {
            pos_by_req.insert(r_id, idx);
        }

        // Positions of victims in that berth sequence must form a consecutive range
        let mut positions: Vec<usize> = out.ruined.iter().map(|r| pos_by_req[&r.get()]).collect();
        positions.sort_unstable();
        let min_pos = positions.first().copied().unwrap();
        let max_pos = positions.last().copied().unwrap();
        assert_eq!(
            max_pos - min_pos + 1,
            positions.len(),
            "victim positions within berth must be a contiguous block"
        );
    }

    #[test]
    fn build_unassign_plan_unassigns_exactly_victims() {
        let (model, state, eval) = make_assigned_state();
        let mut rng = StdRng::seed_from_u64(2024);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        // Pick first two assigned as victims deterministically
        let assigned = super::collect_assigned(&state);
        assert!(assigned.len() >= 2);
        let victims = vec![assigned[0].r, assigned[1].r];

        let out = super::build_unassign_plan(&mut ctx, &victims);
        assert_eq!(
            out.ruined, victims,
            "ruined list must mirror requested victims ordering"
        );
        assert_eq!(
            out.ruined_plan.decision_var_patches.len(),
            victims.len(),
            "plan must contain one DV patch per victim"
        );
        assert!(
            !out.ruined_plan.terminal_delta.is_empty(),
            "terminal delta should reflect releases"
        );
    }

    #[test]
    fn related_ruin_on_empty_state_returns_empty() {
        let (model, state, eval) = make_unassigned_state();
        let mut rng = StdRng::seed_from_u64(30);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let mut ruin = RelatedRuin::new(5, create_neighbor_fn_vec());
        let out = ruin.ruin(&mut ctx);

        assert!(
            out.ruined.is_empty(),
            "victim list should be empty when state is empty"
        );
        assert!(
            out.ruined_plan.is_empty(),
            "plan should be empty when state is empty"
        );
    }

    #[test]
    fn related_ruin_vec_respects_k_limit() {
        let (model, state, eval) = make_assigned_state(); // All 4 requests are assigned
        let mut rng = StdRng::seed_from_u64(31);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        // Graph is r1-r2-r3-r4 (all assigned). Cluster size is 4.
        // We request k=3.
        let mut ruin = RelatedRuin::new(3, create_neighbor_fn_vec());
        let out = ruin.ruin(&mut ctx);

        assert_eq!(
            out.ruined.len(),
            3,
            "should remove exactly k=3 victims when cluster is larger"
        );
        assert_eq!(
            out.ruined_plan.decision_var_patches.len(),
            3,
            "plan should match victim count"
        );
    }

    #[test]
    fn related_ruin_slice_stops_at_unassigned_and_exhausts_cluster() {
        // This state has r1, r3 assigned and r2, r4 unassigned.
        let (model, state, eval) = make_partially_assigned_state();
        let mut rng = StdRng::seed_from_u64(32);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ruin_ctx(&model, &state, &eval, &mut rng, &mut buffer);

        let im = model.index_manager();
        let r1_ix = im.request_index(rid(1)).unwrap(); // Assigned
        let r3_ix = im.request_index(rid(3)).unwrap(); // Assigned

        // Graph is r1-r2-r3-r4.
        // We request k=5 (larger than any possible cluster).
        // The BFS should get stuck at the unassigned requests (r2, r4).
        // The two valid, disjoint clusters are {r1} and {r3}.
        let mut ruin = RelatedRuin::new(5, create_neighbor_fn_slice());
        let out = ruin.ruin(&mut ctx);

        // The seed must be either r1 or r3. In either case, the cluster size is 1.
        assert_eq!(out.ruined.len(), 1, "should only find a cluster of size 1");
        let victim = out.ruined[0];
        assert!(
            victim == r1_ix || victim == r3_ix,
            "victim must be one of the two assigned requests"
        );
    }
}
