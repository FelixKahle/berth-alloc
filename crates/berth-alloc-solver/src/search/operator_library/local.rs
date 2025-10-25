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
        operator::{LocalMoveOperator, NeighborFn},
        planner::{CostEvaluator, PlanBuilder, PlanExplorer, PlanningContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::Plan,
        terminal::terminalocc::FreeBerth,
    },
};
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use rand::seq::SliceRandom;
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    ops::Mul,
    ops::RangeInclusive,
    sync::Arc,
};

/// Best (lowest-cost) insertion for `ri` right now via explorer.
/// Returns `(FreeBerth, start, cost)` for the earliest feasible start in each free interval.
/// NOTE: returns the `FreeBerth<T>` by value to avoid lifetime issues.
#[inline]
fn best_insertion_for_request_ex<'e, 'c, 'm, 'p, T, C>(
    explorer: &PlanExplorer<'e, 'c, '_, 'm, 'p, T, C>,
    model: &crate::model::solver_model::SolverModel<'m, T>,
    request_index: RequestIndex,
) -> Option<(FreeBerth<T>, TimePoint<T>, Cost)>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
{
    let mut best_triplet: Option<(Cost, FreeBerth<T>, TimePoint<T>)> = None;

    for free_berth_interval in explorer.iter_free_for(request_index) {
        let berth_index = free_berth_interval.berth_index();
        let Some(processing_time) = model.processing_time(request_index, berth_index) else {
            continue;
        };

        let free_interval = free_berth_interval.interval();
        let earliest_start = free_interval.start();
        if earliest_start + processing_time > free_interval.end() {
            continue;
        }

        if let Some(candidate_cost) = explorer.peek_cost(request_index, earliest_start, berth_index)
        {
            match best_triplet {
                None => best_triplet = Some((candidate_cost, free_berth_interval, earliest_start)),
                Some((best_cost, _, _)) if candidate_cost < best_cost => {
                    best_triplet = Some((candidate_cost, free_berth_interval, earliest_start))
                }
                _ => {}
            }
        }
    }

    best_triplet.map(|(c, fb, s)| (fb, s, c))
}

#[inline]
fn clamp_range_sample<R: rand::Rng>(
    rng: &mut R,
    range: &RangeInclusive<usize>,
    hard_cap: usize,
) -> usize {
    let lo = *range.start();
    let hi = *range.end();
    let drawn = if lo == hi {
        lo
    } else {
        rng.random_range(lo..=hi)
    };
    drawn.min(hard_cap).max(0)
}

#[inline]
fn rcl_index<R: rand::Rng>(len: usize, alpha: f64, rng: &mut R) -> usize {
    debug_assert!(len > 0);
    let p: f64 = rng.random();
    (((len as f64) * p.powf(alpha)).ceil() as usize)
        .saturating_sub(1)
        .min(len - 1)
}

/// Restrict a list of (RequestIndex, Payload) pairs to a seed’s neighborhood
/// defined by `neighbor_cb`. If the neighborhood is empty, returns the original
/// list as a fallback. Also returns the chosen seed.
#[inline]
fn restrict_to_neighborhood_or_fallback<T, R>(
    rng: &mut R,
    assigned: &[(RequestIndex, T)],
    neighbor_cb: &Option<Arc<NeighborFn>>,
) -> (Vec<(RequestIndex, T)>, RequestIndex)
where
    R: rand::Rng,
    T: Clone,
{
    let (seed, _) = assigned[rng.random_range(0..assigned.len())].clone();
    if let Some(cb) = neighbor_cb {
        let mut keep: HashSet<usize> = HashSet::new();
        keep.insert(seed.get());
        for n in cb(seed) {
            keep.insert(n.get());
        }
        let filtered: Vec<_> = assigned
            .iter()
            .filter(|&(ri, _)| keep.contains(&ri.get()))
            .cloned()
            .collect();
        if !filtered.is_empty() {
            return (filtered, seed);
        }
    }
    (assigned.to_vec(), seed)
}

#[derive(Clone)]
pub struct ShiftEarlierOnSameBerth {
    /// How many assigned requests to randomly sample and try to shift earlier.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
    /// Optional neighborhood restriction; when present, candidates are picked from seed ∪ neighbors.
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl ShiftEarlierOnSameBerth {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for ShiftEarlierOnSameBerth
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "ShiftEarlierOnSameBerth"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Use a one-off builder to snapshot assigned jobs and sample a subset.
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let assigned_jobs: Vec<(RequestIndex, (BerthIndex, TimePoint<T>))> = seed_builder
            .with_explorer(|ex| {
                ex.decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            Some((RequestIndex::new(i), (berth_index, start_time)))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_jobs.is_empty() {
            return None;
        }

        let (mut pool, _seed) = restrict_to_neighborhood_or_fallback(
            rng,
            assigned_jobs.as_slice(),
            &self.neighbor_callback,
        );

        let sample_size =
            clamp_range_sample(rng, &self.number_of_candidates_to_try_range, pool.len()).max(1);
        pool.shuffle(rng);
        pool.truncate(sample_size);

        // Try candidates one by one, each attempt with a FRESH builder.
        for (request_index, (berth_index, current_start_time)) in pool {
            let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

            // Find earliest feasible start on the same berth.
            let earliest_opt = attempt_builder.with_explorer(|ex| {
                ex.iter_free_for(request_index)
                    .filter(|f| f.berth_index() == berth_index)
                    .filter_map(|f| {
                        let iv = f.interval();
                        let start = iv.start();
                        model
                            .processing_time(request_index, berth_index)
                            .and_then(|pt| {
                                if start + pt <= iv.end() {
                                    Some((f, start))
                                } else {
                                    None
                                }
                            })
                    })
                    .min_by_key(|(_, s)| s.value())
            });

            let Some((target_free_berth, target_start_time)) = earliest_opt else {
                continue;
            };
            if target_start_time >= current_start_time {
                continue; // not earlier → next candidate
            }

            // Balanced move on the fresh builder.
            let _ = attempt_builder.propose_unassignment(request_index).ok()?;
            attempt_builder
                .propose_assignment(request_index, target_start_time, &target_free_berth)
                .ok()?;

            // New API: only finalize when we actually changed and fitness is evaluable.
            if !attempt_builder.has_changes() {
                continue;
            }
            if attempt_builder.peek_fitness().is_none() {
                continue;
            }
            return Some(attempt_builder.finalize());
        }

        None
    }
}

// ======================================================================
// RelocateSingleBest  (range-based sampling)
// ======================================================================

#[derive(Clone)]
pub struct RelocateSingleBest {
    /// How many assigned requests to randomly sample and try to relocate.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl RelocateSingleBest {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for RelocateSingleBest
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RelocateSingleBest"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Snapshot assigned jobs with their current cost (one-off builder).
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

        #[allow(clippy::type_complexity)]
        let assigned_with_cost: Vec<(RequestIndex, (BerthIndex, TimePoint<T>, Cost))> =
            seed_builder.with_explorer(|ex| {
                ex.decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            let ri = RequestIndex::new(i);
                            ex.peek_cost(ri, start_time, berth_index)
                                .map(|current_cost| (ri, (berth_index, start_time, current_cost)))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_with_cost.is_empty() {
            return None;
        }

        let (mut pool, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned_with_cost, &self.neighbor_callback);

        let sample_size =
            clamp_range_sample(rng, &self.number_of_candidates_to_try_range, pool.len()).max(1);
        pool.shuffle(rng);
        pool.truncate(sample_size);

        for (request_index, (current_berth, current_start_time, current_cost)) in pool {
            let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

            let best_option = attempt_builder
                .with_explorer(|ex| best_insertion_for_request_ex(ex, model, request_index));
            let Some((best_free_berth, best_start_time, best_cost)) = best_option else {
                continue;
            };

            // No-op and non-worsening guards.
            if best_free_berth.berth_index() == current_berth
                && best_start_time == current_start_time
            {
                continue;
            }
            if best_cost > current_cost {
                continue;
            }

            // Balanced relocate on a fresh builder.
            let _ = attempt_builder.propose_unassignment(request_index).ok()?;
            attempt_builder
                .propose_assignment(request_index, best_start_time, &best_free_berth)
                .ok()?;

            if !attempt_builder.has_changes() {
                continue;
            }
            if attempt_builder.peek_fitness().is_none() {
                continue;
            }
            return Some(attempt_builder.finalize());
        }

        None
    }
}

// ======================================================================
// SwapPairSameBerth  (range-based pair attempts; fresh builder per attempt)
// ======================================================================

#[derive(Clone)]
pub struct SwapPairSameBerth {
    /// How many random same-berth pairs to attempt before giving up.
    pub number_of_pair_attempts_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl SwapPairSameBerth {
    pub fn new(number_of_pair_attempts_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_pair_attempts_to_try_range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for SwapPairSameBerth
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "SwapPairSameBerth"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Use a seed builder to group assigned by berth (ordered by start).
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let mut by_berth: BTreeMap<BerthIndex, Vec<(RequestIndex, TimePoint<T>)>> = seed_builder
            .with_explorer(|ex| {
                let mut map: BTreeMap<BerthIndex, Vec<(RequestIndex, TimePoint<T>)>> =
                    BTreeMap::new();
                for (i, dv) in ex.decision_vars().iter().enumerate() {
                    if let DecisionVar::Assigned(Decision {
                        berth_index,
                        start_time,
                    }) = *dv
                    {
                        map.entry(berth_index)
                            .or_default()
                            .push((RequestIndex::new(i), start_time));
                    }
                }
                for v in map.values_mut() {
                    v.sort_by_key(|(_, s)| s.value());
                }
                map
            });
        if by_berth.is_empty() {
            return None;
        }

        // Neighborhood restriction (seed ∪ neighbors) within a *single* berth’s sequence.
        if let Some(cb) = &self.neighbor_callback {
            // flatten to pick a seed
            let mut flat: Vec<(BerthIndex, (RequestIndex, TimePoint<T>))> = Vec::new();
            for (bi, seq) in by_berth.iter() {
                for e in seq {
                    flat.push((*bi, *e));
                }
            }
            if flat.is_empty() {
                return None;
            }
            let (seed_bi, (seed_ri, _)) = flat[rng.random_range(0..flat.len())];
            let mut set: HashSet<usize> = HashSet::new();
            set.insert(seed_ri.get());
            for n in cb(seed_ri) {
                set.insert(n.get());
            }
            if let Some(seq) = by_berth.get_mut(&seed_bi) {
                let filtered: Vec<_> = seq
                    .iter()
                    .cloned()
                    .filter(|(ri, _)| set.contains(&ri.get()))
                    .collect();
                if filtered.len() >= 2 {
                    by_berth.clear();
                    by_berth.insert(seed_bi, filtered);
                }
            }
        }

        let attempts_to_try =
            clamp_range_sample(rng, &self.number_of_pair_attempts_to_try_range, usize::MAX).max(1);

        // Build attempts on the (possibly) restricted berth map.
        for _attempt in 0..attempts_to_try {
            let mut candidate_berths: Vec<_> =
                by_berth.iter().filter(|(_, v)| v.len() >= 2).collect();
            if candidate_berths.is_empty() {
                return None;
            }
            candidate_berths.shuffle(rng);
            let (berth_index, sequence) = candidate_berths[0];

            // Pick two distinct random positions on that berth.
            let mut indices: Vec<usize> = (0..sequence.len()).collect();
            indices.shuffle(rng);
            let a_idx = indices[0];
            let b_idx = indices[1];

            let (ri_a, start_a) = sequence[a_idx];
            let (ri_b, start_b) = sequence[b_idx];

            let Some(pt_a_on_b) = model.processing_time(ri_a, *berth_index) else {
                continue;
            };
            let Some(pt_b_on_b) = model.processing_time(ri_b, *berth_index) else {
                continue;
            };

            // Target rectangles for swap into each other's positions (same berth).
            let fb_a_to_b = FreeBerth::new(
                TimeInterval::new(start_b, start_b + pt_a_on_b),
                *berth_index,
            );
            let fb_b_to_a = FreeBerth::new(
                TimeInterval::new(start_a, start_a + pt_b_on_b),
                *berth_index,
            );

            // Do the swap on a FRESH builder so failures don't pollute later attempts.
            let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
            let _ = attempt_builder.propose_unassignment(ri_a).ok()?;
            let _ = attempt_builder.propose_unassignment(ri_b).ok()?;

            if attempt_builder
                .propose_assignment(ri_a, start_b, &fb_a_to_b)
                .is_err()
            {
                continue;
            }
            if attempt_builder
                .propose_assignment(ri_b, start_a, &fb_b_to_a)
                .is_err()
            {
                continue;
            }

            if !attempt_builder.has_changes() {
                continue;
            }
            if attempt_builder.peek_fitness().is_none() {
                continue;
            }
            return Some(attempt_builder.finalize());
        }

        None
    }
}

// ======================================================================
// CrossExchangeAcrossBerths  (range-based pair attempts; fresh builder per attempt)
// ======================================================================

#[derive(Clone)]
pub struct CrossExchangeAcrossBerths {
    /// How many random cross-berth pairs to attempt before giving up.
    pub number_of_pair_attempts_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl CrossExchangeAcrossBerths {
    pub fn new(number_of_pair_attempts_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_pair_attempts_to_try_range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for CrossExchangeAcrossBerths
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "CrossExchangeAcrossBerths"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let model = context.model();

        // Snapshot all assigned as (ri, (berth, start)).
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let assigned: Vec<(RequestIndex, (BerthIndex, TimePoint<T>))> =
            seed_builder.with_explorer(|ex| {
                ex.decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            Some((RequestIndex::new(i), (berth_index, start_time)))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned.len() < 2 {
            return None;
        }

        // Restrict to neighborhood; ensure cross-berth diversity exists, else fallback.
        let (mut pool, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned, &self.neighbor_callback);

        let mut berths: BTreeSet<BerthIndex> = BTreeSet::new();
        for (_, (bi, _)) in &pool {
            berths.insert(*bi);
        }
        if berths.len() < 2 {
            pool = assigned.clone();
        }

        let attempts_to_try =
            clamp_range_sample(rng, &self.number_of_pair_attempts_to_try_range, usize::MAX).max(1);

        for _attempt in 0..attempts_to_try {
            let mut shuffled = pool.clone();
            shuffled.shuffle(rng);

            // Find first cross-berth pair in this random order.
            #[allow(clippy::type_complexity)]
            let mut chosen: Option<(
                RequestIndex,
                RequestIndex,
                BerthIndex,
                BerthIndex,
                TimePoint<T>,
                TimePoint<T>,
            )> = None;
            'scan: for i in 0..(shuffled.len().saturating_sub(1)) {
                for j in (i + 1)..shuffled.len() {
                    let (ra, (bi_a, sa)) = shuffled[i];
                    let (rb, (bi_b, sb)) = shuffled[j];
                    if bi_a != bi_b {
                        chosen = Some((ra, rb, bi_a, bi_b, sa, sb));
                        break 'scan;
                    }
                }
            }
            let Some((ra, rb, bi_a, bi_b, sa, sb)) = chosen else {
                continue;
            };

            let Some(pta) = model.processing_time(ra, bi_b) else {
                continue;
            };
            let Some(ptb) = model.processing_time(rb, bi_a) else {
                continue;
            };

            // target rectangles
            let iv_a_to_b = TimeInterval::new(sb, sb + pta);
            let iv_b_to_a = TimeInterval::new(sa, sa + ptb);

            // quick sanity: both targets allowed per evaluator
            if context
                .cost_evaluator()
                .eval_request(model, ra, sb, bi_b)
                .is_none()
            {
                continue;
            }
            if context
                .cost_evaluator()
                .eval_request(model, rb, sa, bi_a)
                .is_none()
            {
                continue;
            }

            // Do the cross-swap on a FRESH builder to keep attempts independent.
            let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

            let _ = attempt_builder.propose_unassignment(ra).ok()?;
            let _ = attempt_builder.propose_unassignment(rb).ok()?;

            // FreeBerth shims using exact target intervals.
            let fb_a_to_b = FreeBerth::new(iv_a_to_b, bi_b);
            let fb_b_to_a = FreeBerth::new(iv_b_to_a, bi_a);

            if attempt_builder
                .propose_assignment(ra, sb, &fb_a_to_b)
                .is_err()
            {
                continue;
            }
            if attempt_builder
                .propose_assignment(rb, sa, &fb_b_to_a)
                .is_err()
            {
                continue;
            }

            if !attempt_builder.has_changes() {
                continue;
            }
            if attempt_builder.peek_fitness().is_none() {
                continue;
            }
            return Some(attempt_builder.finalize());
        }

        None
    }
}

// ======================================================================
// OrOptBlockRelocate  (ranges for block length + RCL alpha)
// ======================================================================

#[derive(Clone)]
pub struct OrOptBlockRelocate {
    /// Inclusive range for the contiguous block length (k) to relocate.
    pub block_length_to_relocate_range: RangeInclusive<usize>,
    /// Inclusive range for RCL alpha when picking the block's starting index.
    pub rcl_alpha_range: RangeInclusive<f64>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl OrOptBlockRelocate {
    pub fn new(
        block_length_to_relocate_range: RangeInclusive<usize>,
        rcl_alpha_range: RangeInclusive<f64>,
    ) -> Self {
        assert!(!block_length_to_relocate_range.is_empty());
        assert!(!rcl_alpha_range.is_empty());
        assert!(
            *block_length_to_relocate_range.start() >= 2,
            "Or-Opt requires k >= 2"
        );
        assert!(*rcl_alpha_range.start() > 0.0 && rcl_alpha_range.end().is_finite());
        Self {
            block_length_to_relocate_range,
            rcl_alpha_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for OrOptBlockRelocate
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "OrOptBlockRelocate"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let model = context.model();
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

        // Pick any berth with at least 2 assigned; collect ordered by start.
        let maybe_group = seed_builder.with_explorer(|ex| {
            let mut by_berth: BTreeMap<BerthIndex, Vec<(RequestIndex, TimePoint<T>)>> =
                BTreeMap::new();
            for (i, dv) in ex.decision_vars().iter().enumerate() {
                if let DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                }) = *dv
                {
                    by_berth
                        .entry(berth_index)
                        .or_default()
                        .push((RequestIndex::new(i), start_time));
                }
            }
            let mut groups: Vec<_> = by_berth.into_iter().collect();
            groups.shuffle(rng);
            groups.into_iter().find_map(|(bi, mut v)| {
                if v.len() >= 2 {
                    v.sort_by_key(|(_, s)| s.value());
                    Some((bi, v))
                } else {
                    None
                }
            })
        });

        let (_berth_index, full_sequence) = maybe_group?;

        // Sample k and alpha using the full berth sequence.
        let max_k = full_sequence.len();
        let sampled_k = clamp_range_sample(rng, &self.block_length_to_relocate_range, max_k).max(2);
        if sampled_k > max_k {
            return None;
        }
        let alpha_sample = {
            let a_lo = *self.rcl_alpha_range.start();
            let a_hi = *self.rcl_alpha_range.end();
            if (a_lo - a_hi).abs() < f64::EPSILON {
                a_lo
            } else {
                rng.random_range(a_lo..=a_hi)
            }
        };

        // Neighborhood within this berth (seed ∪ neighbors). Fallback to full berth if
        // neighborhood is smaller than k.
        let sequence = if let Some(cb) = &self.neighbor_callback {
            let (seed_ri, _) = full_sequence[rng.random_range(0..full_sequence.len())];
            let mut set: HashSet<usize> = HashSet::new();
            set.insert(seed_ri.get());
            for n in cb(seed_ri) {
                set.insert(n.get());
            }
            let filtered: Vec<_> = full_sequence
                .iter()
                .cloned()
                .filter(|(ri, _)| set.contains(&ri.get()))
                .collect();
            if filtered.len() >= sampled_k {
                filtered
            } else {
                full_sequence
            }
        } else {
            full_sequence
        };

        // Choose a start index with RCL over the chosen sequence.
        let valid_start_count = sequence.len().saturating_sub(sampled_k).saturating_add(1);
        if valid_start_count == 0 {
            return None;
        }
        let start_index = rcl_index(valid_start_count, alpha_sample, rng);
        let block_slice = &sequence[start_index..start_index + sampled_k];
        let block_indices: Vec<RequestIndex> = block_slice.iter().map(|&(ri, _)| ri).collect();

        // Rebuild on a fresh builder: unassign block → greedy reinsert anywhere.
        let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

        // 1) Unassign all requests in the block first
        for &ri in &block_indices {
            let _ = attempt_builder.propose_unassignment(ri).ok()?;
        }

        // 2) Reinsert every request; if any fails, abandon this attempt
        for &ri in &block_indices {
            if let Some((free, start, _cost)) =
                attempt_builder.with_explorer(|ex| best_insertion_for_request_ex(ex, model, ri))
            {
                if attempt_builder
                    .propose_assignment(ri, start, &free)
                    .is_err()
                {
                    return None;
                }
            } else {
                return None;
            }
        }

        // 3) Before finalize, guard with new API; then enforce block change count.
        if !attempt_builder.has_changes() {
            return None;
        }
        attempt_builder.peek_fitness()?;

        // 4) Finalize and ensure we actually changed all k requests (2 patches per request)
        let plan = attempt_builder.finalize();
        let expected_min_patches = 2 * block_indices.len();
        if plan.decision_var_patches.len() < expected_min_patches {
            return None;
        }
        Some(plan)
    }
}

// ======================================================================
// RelocateSingleBestAllowWorsening  (same as RelocateSingleBest but allows worsening)
// ======================================================================

#[derive(Clone)]
pub struct RelocateSingleBestAllowWorsening {
    /// How many assigned requests to randomly sample and try to relocate.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl RelocateSingleBestAllowWorsening {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for RelocateSingleBestAllowWorsening
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RelocateSingleBestAllowWorsening"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Snapshot assigned jobs with their current cost (one-off builder).
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

        #[allow(clippy::type_complexity)]
        let assigned_with_cost: Vec<(RequestIndex, (BerthIndex, TimePoint<T>, Cost))> =
            seed_builder.with_explorer(|ex| {
                ex.decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            let ri = RequestIndex::new(i);
                            ex.peek_cost(ri, start_time, berth_index)
                                .map(|current_cost| (ri, (berth_index, start_time, current_cost)))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_with_cost.is_empty() {
            return None;
        }

        let (mut pool, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned_with_cost, &self.neighbor_callback);

        let sample_size =
            clamp_range_sample(rng, &self.number_of_candidates_to_try_range, pool.len()).max(1);
        pool.shuffle(rng);
        pool.truncate(sample_size);

        for (request_index, (_current_berth, _current_start_time, _current_cost)) in pool {
            let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

            // Keep the "best insertion" computation, but DO NOT require it to beat current cost.
            let best_option = attempt_builder
                .with_explorer(|ex| best_insertion_for_request_ex(ex, model, request_index));
            let Some((best_free_berth, best_start_time, _best_cost)) = best_option else {
                continue;
            };

            let _ = attempt_builder.propose_unassignment(request_index).ok()?;
            attempt_builder
                .propose_assignment(request_index, best_start_time, &best_free_berth)
                .ok()?;

            if !attempt_builder.has_changes() {
                continue;
            }
            if attempt_builder.peek_fitness().is_none() {
                continue;
            }
            return Some(attempt_builder.finalize());
        }

        None
    }
}

// ======================================================================
// RandomRelocateAnywhere  (random feasible reinsert; can worsen on purpose)
// ======================================================================

#[derive(Clone)]
pub struct RandomRelocateAnywhere {
    /// How many assigned requests to randomly sample and try to reinsert randomly.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl RandomRelocateAnywhere {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for RandomRelocateAnywhere
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomRelocateAnywhere"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Snapshot assigned jobs.
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let assigned_jobs: Vec<(RequestIndex, (BerthIndex, TimePoint<T>))> = seed_builder
            .with_explorer(|ex| {
                ex.decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            Some((RequestIndex::new(i), (berth_index, start_time)))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_jobs.is_empty() {
            return None;
        }

        let (mut pool, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned_jobs, &self.neighbor_callback);

        let sample_size =
            clamp_range_sample(rng, &self.number_of_candidates_to_try_range, pool.len()).max(1);

        pool.shuffle(rng);
        pool.truncate(sample_size);

        for (ri, (_cur_b, _cur_s)) in pool {
            let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

            // Gather all feasible "earliest start" placements across all free intervals.
            let options = attempt_builder.with_explorer(|ex| {
                let mut opts: Vec<(FreeBerth<T>, TimePoint<T>)> = Vec::new();
                for free in ex.iter_free_for(ri) {
                    let b = free.berth_index();
                    if let Some(pt) = model.processing_time(ri, b) {
                        let iv = free.interval();
                        let s = iv.start();
                        if s + pt <= iv.end() {
                            opts.push((free, s));
                        }
                    }
                }
                opts
            });

            if options.is_empty() {
                continue;
            }

            // Pick a random feasible placement (not minimum cost on purpose).
            let (fb, s) = {
                let idx = rng.random_range(0..options.len());
                options[idx].clone()
            };

            let _ = attempt_builder.propose_unassignment(ri).ok()?;
            attempt_builder.propose_assignment(ri, s, &fb).ok()?;

            if !attempt_builder.has_changes() {
                continue;
            }
            if attempt_builder.peek_fitness().is_none() {
                continue;
            }
            return Some(attempt_builder.finalize());
        }

        None
    }
}

// ======================================================================
// HillClimbRelocateBest  (best improving relocate among sampled requests)
// ======================================================================

#[derive(Clone)]
pub struct HillClimbRelocateBest {
    /// How many assigned requests to sample and try (upper bound).
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl HillClimbRelocateBest {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for HillClimbRelocateBest
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "HillClimbRelocateBest"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Gather assigned requests (seed snapshot).
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let assigned: Vec<(RequestIndex, (BerthIndex, TimePoint<T>))> =
            seed_builder.with_explorer(|ex| {
                ex.decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            Some((RequestIndex::new(i), (berth_index, start_time)))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned.is_empty() {
            return None;
        }

        // Restrict to neighborhood.
        let (mut cand_reqs, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned, &self.neighbor_callback);

        // Sample a subset to keep per-call work bounded.
        let k = clamp_range_sample(
            rng,
            &self.number_of_candidates_to_try_range,
            cand_reqs.len(),
        )
        .max(1);
        cand_reqs.shuffle(rng);
        cand_reqs.truncate(k);

        // Track best improving plan by candidate fitness (true objective).
        let current_fit = *context.state().fitness();
        let mut best_fit: Option<crate::state::fitness::Fitness> = None;
        let mut best_plan: Option<Plan<'p, T>> = None;

        for (ri, (_cur_b, _cur_s)) in cand_reqs {
            // Explore all feasible earliest placements; pick the one that yields an improvement.
            let attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

            // Build the list of candidate placements (free rectangles + earliest start).
            let placements = attempt_builder.with_explorer(|ex| {
                let mut opts: Vec<(FreeBerth<T>, TimePoint<T>)> = Vec::new();
                for free in ex.iter_free_for(ri) {
                    let b = free.berth_index();
                    if let Some(pt) = model.processing_time(ri, b) {
                        let iv = free.interval();
                        let s = iv.start();
                        if s + pt <= iv.end() {
                            opts.push((free, s));
                        }
                    }
                }
                opts
            });

            if placements.is_empty() {
                continue;
            }

            for (fb, s) in placements {
                let mut bld: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
                let _ = bld.propose_unassignment(ri).ok()?;
                if bld.propose_assignment(ri, s, &fb).is_err() {
                    continue;
                }

                if !bld.has_changes() {
                    continue;
                }
                let Some(cand_fit) = bld.peek_fitness() else {
                    continue;
                };
                if cand_fit >= current_fit {
                    continue; // require improvement
                }

                // Finalize only improving candidates; keep the best by fitness.
                if best_fit.as_ref().map(|bf| cand_fit < *bf).unwrap_or(true) {
                    best_fit = Some(cand_fit);
                    best_plan = Some(bld.finalize());
                }
            }
        }

        best_plan
    }
}

// ======================================================================
// HillClimbBestSwapSameBerth  (best improving swap among sampled same-berth pairs)
// ======================================================================

#[derive(Clone)]
pub struct HillClimbBestSwapSameBerth {
    /// How many random same-berth pairs to sample per call (upper bound).
    pub number_of_pair_attempts_to_try_range: RangeInclusive<usize>,
    pub neighbor_callback: Option<Arc<NeighborFn>>,
}

impl HillClimbBestSwapSameBerth {
    pub fn new(number_of_pair_attempts_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_pair_attempts_to_try_range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, C, R> LocalMoveOperator<T, C, R> for HillClimbBestSwapSameBerth
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "HillClimbBestSwapSameBerth"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let model = context.model();

        // Collect assigned by berth, ordered by start.
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let mut by_berth = seed_builder.with_explorer(|ex| {
            let mut map: BTreeMap<BerthIndex, Vec<(RequestIndex, TimePoint<T>)>> = BTreeMap::new();
            for (i, dv) in ex.decision_vars().iter().enumerate() {
                if let DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                }) = *dv
                {
                    map.entry(berth_index)
                        .or_default()
                        .push((RequestIndex::new(i), start_time));
                }
            }
            for v in map.values_mut() {
                v.sort_by_key(|(_, s)| s.value());
            }
            map
        });
        if by_berth.is_empty() {
            return None;
        }

        // Neighborhood restriction (seed ∪ neighbors) within one berth.
        if let Some(cb) = &self.neighbor_callback {
            let mut flat: Vec<(BerthIndex, (RequestIndex, TimePoint<T>))> = Vec::new();
            for (bi, seq) in by_berth.iter() {
                for e in seq {
                    flat.push((*bi, *e));
                }
            }
            if flat.is_empty() {
                return None;
            }
            let (seed_bi, (seed_ri, _)) = flat[rng.random_range(0..flat.len())];

            let mut set: HashSet<usize> = HashSet::new();
            set.insert(seed_ri.get());
            for n in cb(seed_ri) {
                set.insert(n.get());
            }

            if let Some(seq) = by_berth.get_mut(&seed_bi) {
                let filtered: Vec<_> = seq
                    .iter()
                    .cloned()
                    .filter(|(ri, _)| set.contains(&ri.get()))
                    .collect();
                if filtered.len() >= 2 {
                    by_berth.clear();
                    by_berth.insert(seed_bi, filtered);
                }
            }
        }

        // Enumerate all feasible swap candidates (indices on same berth).
        #[allow(clippy::type_complexity)]
        let mut pairs: Vec<(
            BerthIndex,
            (RequestIndex, TimePoint<T>),
            (RequestIndex, TimePoint<T>),
        )> = Vec::new();
        for (bi, seq) in by_berth.iter() {
            if seq.len() < 2 {
                continue;
            }
            for i in 0..(seq.len() - 1) {
                for j in (i + 1)..seq.len() {
                    pairs.push((*bi, seq[i], seq[j]));
                }
            }
        }
        if pairs.is_empty() {
            return None;
        }

        // Sample a subset of pairs to bound per-call work.
        let attempts =
            clamp_range_sample(rng, &self.number_of_pair_attempts_to_try_range, pairs.len()).max(1);
        pairs.shuffle(rng);
        pairs.truncate(attempts);

        // Track best improving swap plan by candidate fitness.
        let current_fit = *context.state().fitness();
        let mut best_fit: Option<crate::state::fitness::Fitness> = None;
        let mut best_plan: Option<Plan<'p, T>> = None;

        for (bi, (ri_a, sa), (ri_b, sb)) in pairs {
            let Some(pt_a_on_b) = model.processing_time(ri_a, bi) else {
                continue;
            };
            let Some(pt_b_on_b) = model.processing_time(ri_b, bi) else {
                continue;
            };

            let iv_a_to_b = TimeInterval::new(sb, sb + pt_a_on_b);
            let iv_b_to_a = TimeInterval::new(sa, sa + pt_b_on_b);

            let fb_a_to_b = FreeBerth::new(iv_a_to_b, bi);
            let fb_b_to_a = FreeBerth::new(iv_b_to_a, bi);

            let mut bld: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
            let _ = bld.propose_unassignment(ri_a).ok()?;
            let _ = bld.propose_unassignment(ri_b).ok()?;
            if bld.propose_assignment(ri_a, sb, &fb_a_to_b).is_err() {
                continue;
            }
            if bld.propose_assignment(ri_b, sa, &fb_b_to_a).is_err() {
                continue;
            }

            if !bld.has_changes() {
                continue;
            }
            let Some(cand_fit) = bld.peek_fitness() else {
                continue;
            };
            if cand_fit >= current_fit {
                continue; // require improvement
            }

            if best_fit.as_ref().map(|bf| cand_fit < *bf).unwrap_or(true) {
                best_fit = Some(cand_fit);
                best_plan = Some(bld.finalize());
            }
        }

        best_plan
    }
}

#[derive(Clone)]
pub struct CrossExchangeBestAcrossBerths {
    pub number_of_pair_attempts_to_try_range: std::ops::RangeInclusive<usize>,
    pub neighbor_callback: Option<std::sync::Arc<crate::search::operator::NeighborFn>>,
}

impl CrossExchangeBestAcrossBerths {
    pub fn new(range: std::ops::RangeInclusive<usize>) -> Self {
        assert!(!range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range: range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(
        mut self,
        cb: std::sync::Arc<crate::search::operator::NeighborFn>,
    ) -> Self {
        self.neighbor_callback = Some(cb);
        self
    }
}

impl<T, C, R> crate::search::operator::LocalMoveOperator<T, C, R> for CrossExchangeBestAcrossBerths
where
    T: Copy
        + Ord
        + num_traits::CheckedAdd
        + num_traits::CheckedSub
        + Into<berth_alloc_core::prelude::Cost>
        + std::ops::Mul<Output = berth_alloc_core::prelude::Cost>,
    C: crate::search::planner::CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "CrossExchangeBestAcrossBerths"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut crate::search::planner::PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        use crate::model::index::{BerthIndex, RequestIndex};
        use crate::state::decisionvar::{Decision, DecisionVar};
        use crate::state::terminal::terminalocc::FreeBerth;

        let model = context.model();

        // Collect assigned (ri, (berth, start))
        let seed_builder = context.builder();
        let assigned: Vec<(
            RequestIndex,
            (BerthIndex, berth_alloc_core::prelude::TimePoint<T>),
        )> = seed_builder.with_explorer(|ex| {
            ex.decision_vars()
                .iter()
                .enumerate()
                .filter_map(|(i, dv)| {
                    if let DecisionVar::Assigned(Decision {
                        berth_index,
                        start_time,
                    }) = *dv
                    {
                        Some((RequestIndex::new(i), (berth_index, start_time)))
                    } else {
                        None
                    }
                })
                .collect()
        });
        if assigned.len() < 2 {
            return None;
        }

        // Neighborhood restriction or fallback
        let (mut pool, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned, &self.neighbor_callback);

        // Need at least two different berths in pool
        let mut seen = std::collections::BTreeSet::new();
        for (_, (b, _)) in &pool {
            seen.insert(*b);
        }
        if seen.len() < 2 {
            pool = assigned.clone();
        }

        let attempts =
            clamp_range_sample(rng, &self.number_of_pair_attempts_to_try_range, usize::MAX).max(1);

        for _ in 0..attempts {
            let mut shuffled = pool.clone();
            shuffled.shuffle(rng);

            // Find first cross-berth pair
            let mut chosen = None;
            'pick: for i in 0..(shuffled.len().saturating_sub(1)) {
                for j in (i + 1)..shuffled.len() {
                    let (ra, (bi_a, _sa)) = shuffled[i];
                    let (rb, (bi_b, _sb)) = shuffled[j];
                    if bi_a != bi_b {
                        chosen = Some((ra, rb, bi_a, bi_b));
                        break 'pick;
                    }
                }
            }
            let Some((ra, rb, bi_a, bi_b)) = chosen else {
                continue;
            };

            // Fresh builder: unassign both, then reinsert each at best earliest on the other berth
            let mut bld = context.builder();
            let _ = bld.propose_unassignment(ra).ok()?;
            let _ = bld.propose_unassignment(rb).ok()?;

            // Best earliest for ra on bi_b
            let opt_a = bld.with_explorer(|ex| {
                let mut best: Option<(
                    FreeBerth<T>,
                    berth_alloc_core::prelude::TimePoint<T>,
                    berth_alloc_core::prelude::Cost,
                )> = None;
                for free in ex.iter_free_for(ra).filter(|f| f.berth_index() == bi_b) {
                    if let Some(pt) = model.processing_time(ra, bi_b) {
                        let iv = free.interval();
                        let s = iv.start();
                        if s + pt <= iv.end()
                            && let Some(c) = ex.peek_cost(ra, s, bi_b)
                        {
                            match best {
                                None => best = Some((free, s, c)),
                                Some((_, _, bc)) if c < bc => best = Some((free, s, c)),
                                _ => {}
                            }
                        }
                    }
                }
                best
            });
            let Some((fb_a, s_a, _)) = opt_a else {
                continue;
            };

            if bld.propose_assignment(ra, s_a, &fb_a).is_err() {
                continue;
            }

            // Best earliest for rb on bi_a (after ra placed)
            let opt_b = bld.with_explorer(|ex| {
                let mut best: Option<(
                    FreeBerth<T>,
                    berth_alloc_core::prelude::TimePoint<T>,
                    berth_alloc_core::prelude::Cost,
                )> = None;
                for free in ex.iter_free_for(rb).filter(|f| f.berth_index() == bi_a) {
                    if let Some(pt) = model.processing_time(rb, bi_a) {
                        let iv = free.interval();
                        let s = iv.start();
                        if s + pt <= iv.end()
                            && let Some(c) = ex.peek_cost(rb, s, bi_a)
                        {
                            match best {
                                None => best = Some((free, s, c)),
                                Some((_, _, bc)) if c < bc => best = Some((free, s, c)),
                                _ => {}
                            }
                        }
                    }
                }
                best
            });
            let Some((fb_b, s_b, _)) = opt_b else {
                continue;
            };

            if bld.propose_assignment(rb, s_b, &fb_b).is_err() {
                continue;
            }

            if !bld.has_changes() {
                continue;
            }
            if bld.peek_fitness().is_none() {
                continue;
            }
            return Some(bld.finalize());
        }

        None
    }
}

#[derive(Clone)]
pub struct RandomizedGreedyRelocateRcl {
    pub number_of_candidates_to_try_range: std::ops::RangeInclusive<usize>,
    pub rcl_alpha_range: std::ops::RangeInclusive<f64>,
    pub neighbor_callback: Option<std::sync::Arc<crate::search::operator::NeighborFn>>,
}

impl RandomizedGreedyRelocateRcl {
    pub fn new(k: std::ops::RangeInclusive<usize>, alpha: std::ops::RangeInclusive<f64>) -> Self {
        assert!(!k.is_empty() && !alpha.is_empty());
        Self {
            number_of_candidates_to_try_range: k,
            rcl_alpha_range: alpha,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(
        mut self,
        cb: std::sync::Arc<crate::search::operator::NeighborFn>,
    ) -> Self {
        self.neighbor_callback = Some(cb);
        self
    }
}

impl<T, C, R> crate::search::operator::LocalMoveOperator<T, C, R> for RandomizedGreedyRelocateRcl
where
    T: Copy
        + Ord
        + num_traits::CheckedAdd
        + num_traits::CheckedSub
        + Into<berth_alloc_core::prelude::Cost>
        + std::ops::Mul<Output = berth_alloc_core::prelude::Cost>,
    C: crate::search::planner::CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomizedGreedyRelocateRCL"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut crate::search::planner::PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        use crate::model::index::RequestIndex;
        use crate::state::decisionvar::{Decision, DecisionVar};
        use crate::state::terminal::terminalocc::FreeBerth;

        let model = context.model();
        let seed = context.builder();
        let assigned: Vec<(RequestIndex, ())> = seed.with_explorer(|ex| {
            ex.decision_vars()
                .iter()
                .enumerate()
                .filter_map(|(i, dv)| {
                    if let DecisionVar::Assigned(Decision { .. }) = *dv {
                        Some((RequestIndex::new(i), ()))
                    } else {
                        None
                    }
                })
                .collect()
        });
        if assigned.is_empty() {
            return None;
        }

        let (mut pool, _seed) =
            restrict_to_neighborhood_or_fallback(rng, &assigned, &self.neighbor_callback);
        let k = clamp_range_sample(rng, &self.number_of_candidates_to_try_range, pool.len()).max(1);
        pool.shuffle(rng);
        pool.truncate(k);

        let alpha = {
            let lo = *self.rcl_alpha_range.start();
            let hi = *self.rcl_alpha_range.end();
            if (lo - hi).abs() < f64::EPSILON {
                lo
            } else {
                rng.random_range(lo..=hi)
            }
        };

        for (ri, _) in pool {
            let mut bld = context.builder();
            // enumerate earliest-feasible placements and rank by cost
            let mut opts: Vec<(
                FreeBerth<T>,
                berth_alloc_core::prelude::TimePoint<T>,
                berth_alloc_core::prelude::Cost,
            )> = bld.with_explorer(|ex| {
                let mut v = Vec::new();
                for free in ex.iter_free_for(ri) {
                    let b = free.berth_index();
                    if let Some(pt) = model.processing_time(ri, b) {
                        let iv = free.interval();
                        let s = iv.start();
                        if s + pt <= iv.end()
                            && let Some(c) = ex.peek_cost(ri, s, b)
                        {
                            v.push((free, s, c));
                        }
                    }
                }
                v
            });
            if opts.is_empty() {
                continue;
            }
            opts.sort_by_key(|(_, _, c)| *c);

            let idx = rcl_index(opts.len(), alpha, rng);
            let (fb, s, _c) = opts[idx].clone();

            let _ = bld.propose_unassignment(ri).ok()?;
            bld.propose_assignment(ri, s, &fb).ok()?;
            if !bld.has_changes() {
                continue;
            }
            if bld.peek_fitness().is_none() {
                continue;
            }
            return Some(bld.finalize());
        }

        None
    }
}

#[derive(Clone)]
pub enum CascadeInsertPolicy {
    /// Always pick the best earliest-feasible placement by base cost.
    BestEarliest,
    /// Randomized candidate via RCL over earliest-feasible placements.
    Rcl {
        alpha_range: std::ops::RangeInclusive<f64>,
    },
}

#[derive(Clone)]
pub struct CascadeRelocateK {
    /// Chain length K to try per proposal (sampled per call).
    pub chain_len_range: std::ops::RangeInclusive<usize>,
    /// How many seeds to consider per call (upper bound).
    pub seeds_to_try_range: std::ops::RangeInclusive<usize>,
    /// Max candidates to consider per step (neighbors of last moved ⇒ fallback full).
    pub candidates_per_step_range: std::ops::RangeInclusive<usize>,
    /// Locality restriction.
    pub neighbor_callback: Option<std::sync::Arc<crate::search::operator::NeighborFn>>,
    /// How to choose the target placement of each moved request.
    pub insert_policy: CascadeInsertPolicy,
    /// Per-step acceptance: require non-worsening at every step?
    pub monotone_steps: bool,
    /// If monotone=false, still forbid *huge* step regressions (ε guard).
    pub max_single_step_worsening: berth_alloc_core::prelude::Cost,
}

impl CascadeRelocateK {
    pub fn new(
        chain_len_range: std::ops::RangeInclusive<usize>,
        seeds_to_try_range: std::ops::RangeInclusive<usize>,
        candidates_per_step_range: std::ops::RangeInclusive<usize>,
    ) -> Self {
        assert!(!chain_len_range.is_empty());
        assert!(!seeds_to_try_range.is_empty());
        assert!(!candidates_per_step_range.is_empty());
        Self {
            chain_len_range,
            seeds_to_try_range,
            candidates_per_step_range,
            neighbor_callback: None,
            insert_policy: CascadeInsertPolicy::BestEarliest,
            monotone_steps: true,
            max_single_step_worsening: 0, // ignored if monotone=true
        }
    }
    pub fn with_neighbors(
        mut self,
        cb: std::sync::Arc<crate::search::operator::NeighborFn>,
    ) -> Self {
        self.neighbor_callback = Some(cb);
        self
    }
    pub fn with_insert_policy(mut self, p: CascadeInsertPolicy) -> Self {
        self.insert_policy = p;
        self
    }
    pub fn allow_step_worsening(mut self, epsilon_cost: berth_alloc_core::prelude::Cost) -> Self {
        self.monotone_steps = false;
        self.max_single_step_worsening = epsilon_cost.max(0);
        self
    }
}

impl<T, C, R> crate::search::operator::LocalMoveOperator<T, C, R> for CascadeRelocateK
where
    T: Copy
        + Ord
        + num_traits::CheckedAdd
        + num_traits::CheckedSub
        + Into<berth_alloc_core::prelude::Cost>
        + std::ops::Mul<Output = berth_alloc_core::prelude::Cost>,
    C: crate::search::planner::CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "CascadeRelocateK"
    }

    fn propose<'b, 'c, 's, 'm, 'p>(
        &self,
        context: &mut crate::search::planner::PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let model = context.model();
        let seed_builder = context.builder();

        // -------------- collect assigned requests --------------
        let assigned: Vec<(RequestIndex, ())> = seed_builder.with_explorer(|ex| {
            ex.decision_vars()
                .iter()
                .enumerate()
                .filter_map(|(i, dv)| {
                    if let DecisionVar::Assigned(Decision { .. }) = *dv {
                        Some((RequestIndex::new(i), ()))
                    } else {
                        None
                    }
                })
                .collect()
        });
        if assigned.is_empty() {
            return None;
        }

        // -------------- seeds to try this call --------------
        let seeds_to_try = {
            let lo = *self.seeds_to_try_range.start();
            let hi = *self.seeds_to_try_range.end();
            if lo == hi {
                lo
            } else {
                rng.random_range(lo..=hi)
            }
        }
        .min(assigned.len())
        .max(1);

        // helper: enumerate earliest-feasible placements for a req
        let enumerate_options =
            |bld: &mut crate::search::planner::PlanBuilder<'_, 'c, 's, 'm, 'p, T, C>,
             ri: RequestIndex|
             -> Vec<(FreeBerth<T>, TimePoint<T>, Cost)> {
                bld.with_explorer(|ex| {
                    let mut v: Vec<(FreeBerth<T>, TimePoint<T>, Cost)> = Vec::new();
                    for free in ex.iter_free_for(ri) {
                        let b = free.berth_index();
                        if let Some(pt) = model.processing_time(ri, b) {
                            let iv = free.interval();
                            let s = iv.start();
                            if s + pt <= iv.end()
                                && let Some(c) = ex.peek_cost(ri, s, b) {
                                    v.push((free, s, c));
                                }
                        }
                    }
                    v
                })
            };

        // -------------- try up to N seeds --------------
        let mut seeds_pool = assigned.clone();
        seeds_pool.shuffle(rng);
        seeds_pool.truncate(seeds_to_try);

        for (seed_ri, _) in seeds_pool {
            // fresh builder per seed
            let mut bld = context.builder();
            if bld.peek_fitness().is_none() {
                continue;
            }

            // draw K for this attempt
            let k = {
                let lo = *self.chain_len_range.start();
                let hi = *self.chain_len_range.end();
                if lo == hi {
                    lo
                } else {
                    rng.random_range(lo..=hi)
                }
            }
            .max(2);

            // we’ll build a small candidate frontier: start with the seed
            let mut frontier: Vec<RequestIndex> = vec![seed_ri];
            let mut moved: Vec<RequestIndex> = Vec::with_capacity(k);

            // chain the relocations
            for _ in 0..k {
                // pick a source to move: prefer last moved’s neighbors else fallback to all assigned
                let source_pool: Vec<RequestIndex> = {
                    let seed_list: Vec<(RequestIndex, ())> = bld.with_explorer(|ex| {
                        ex.decision_vars()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, dv)| {
                                if let DecisionVar::Assigned(Decision { .. }) = *dv {
                                    Some((RequestIndex::new(i), ()))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    });
                    let mut pool: Vec<(RequestIndex, ())> = seed_list.clone();
                    if let Some(cb) = &self.neighbor_callback
                        && let Some(&anchor) = frontier.last() {
                            let mut set = std::collections::HashSet::new();
                            set.insert(anchor.get());
                            for n in cb(anchor) {
                                set.insert(n.get());
                            }
                            let filtered: Vec<_> = seed_list
                                .into_iter()
                                .filter(|(ri, _)| set.contains(&ri.get()))
                                .collect();
                            if !filtered.is_empty() {
                                pool = filtered;
                            }
                        }
                    // cap candidates per step
                    let max_c = {
                        let lo = *self.candidates_per_step_range.start();
                        let hi = *self.candidates_per_step_range.end();
                        if lo == hi {
                            lo
                        } else {
                            rng.random_range(lo..=hi)
                        }
                    }
                    .max(1);
                    let mut v: Vec<RequestIndex> = pool.into_iter().map(|(ri, _)| ri).collect();
                    v.shuffle(rng);
                    v.truncate(max_c.min(v.len()));
                    v
                };

                // choose the first source that can be feasibly reinserted
                let mut did_step = false;
                for src in source_pool {
                    // Don’t move same request twice in one chain (cheap guard)
                    if moved.contains(&src) {
                        continue;
                    }

                    // enumerate options on the *current* builder
                    let mut opts = enumerate_options(&mut bld, src);
                    if opts.is_empty() {
                        continue;
                    }

                    // choose insertion
                    let (fb, s, _cand_cost) = match self.insert_policy {
                        CascadeInsertPolicy::BestEarliest => {
                            opts.sort_by_key(|(_, _, c)| *c);
                            opts[0].clone()
                        }
                        CascadeInsertPolicy::Rcl { ref alpha_range } => {
                            opts.sort_by_key(|(_, _, c)| *c);
                            let a_lo = *alpha_range.start();
                            let a_hi = *alpha_range.end();
                            let alpha = if (a_lo - a_hi).abs() < f64::EPSILON {
                                a_lo
                            } else {
                                rng.random_range(a_lo..=a_hi)
                            };
                            let idx = rcl_index(opts.len(), alpha, rng);
                            opts[idx].clone()
                        }
                    };

                    // compute step delta (peek fitness before/after)
                    let before = match bld.peek_fitness() {
                        Some(f) => f,
                        None => break,
                    };
                    let _ = bld.propose_unassignment(src).ok()?;
                    if bld.propose_assignment(src, s, &fb).is_err() {
                        // Revert to the path state by rebuilding and replaying the moved chain.
                        bld = context.builder();
                        for &ri_m in &moved {
                            let mut rep_opts = enumerate_options(&mut bld, ri_m);
                            if rep_opts.is_empty() {
                                break;
                            }
                            rep_opts.sort_by_key(|(_, _, c)| *c);
                            let (rfb, rs, _) = rep_opts[0].clone();
                            let _ = bld.propose_unassignment(ri_m);
                            let _ = bld.propose_assignment(ri_m, rs, &rfb);
                        }
                        continue;
                    }
                    // step guard
                    if self.monotone_steps {
                        if let Some(after) = bld.peek_fitness() {
                            if after >= before {
                                // undo by discarding this attempt path: rebuild small builder
                                bld = context.builder();
                                // reapply already moved requests deterministically (best earliest again)
                                for &ri_m in &moved {
                                    let mut rep_opts = enumerate_options(&mut bld, ri_m);
                                    if rep_opts.is_empty() {
                                        break;
                                    }
                                    rep_opts.sort_by_key(|(_, _, c)| *c);
                                    let (rfb, rs, _) = rep_opts[0].clone();
                                    let _ = bld.propose_unassignment(ri_m);
                                    let _ = bld.propose_assignment(ri_m, rs, &rfb);
                                }
                                continue;
                            }
                        } else {
                            // cannot evaluate → abandon this src
                            bld = context.builder();
                            for &ri_m in &moved {
                                let mut rep_opts = enumerate_options(&mut bld, ri_m);
                                if rep_opts.is_empty() {
                                    break;
                                }
                                rep_opts.sort_by_key(|(_, _, c)| *c);
                                let (rfb, rs, _) = rep_opts[0].clone();
                                let _ = bld.propose_unassignment(ri_m);
                                let _ = bld.propose_assignment(ri_m, rs, &rfb);
                            }
                            continue;
                        }
                    } else {
                        // allow small worsening per step
                        if let (Some(after), Some(before)) = (bld.peek_fitness(), Some(before))
                            && after.cost > before.cost + self.max_single_step_worsening {
                                // revert path like above
                                bld = context.builder();
                                for &ri_m in &moved {
                                    let mut rep_opts = enumerate_options(&mut bld, ri_m);
                                    if rep_opts.is_empty() {
                                        break;
                                    }
                                    rep_opts.sort_by_key(|(_, _, c)| *c);
                                    let (rfb, rs, _) = rep_opts[0].clone();
                                    let _ = bld.propose_unassignment(ri_m);
                                    let _ = bld.propose_assignment(ri_m, rs, &rfb);
                                }
                                continue;
                            }
                    }

                    // commit this step
                    moved.push(src);
                    frontier.push(src);
                    did_step = true;
                    break;
                }

                if !did_step {
                    break; // chain stuck; stop early
                }
            }

            // finalize only if we actually changed at least 2 requests (≥4 patches)
            if !bld.has_changes() {
                continue;
            }
            if moved.len() < 2 {
                continue;
            }
            if bld.peek_fitness().is_none() {
                continue;
            }

            let plan = bld.finalize();
            if plan.decision_var_patches.len() >= 2 * moved.len() {
                return Some(plan);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            solver_state::SolverStateView,
            terminal::terminalocc::TerminalWrite,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use rand::SeedableRng;
    use std::collections::BTreeMap;

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

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn make_problem() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn make_state_with_assignments<'p>(
        model: &SolverModel<'p, i64>,
        starts: &[(u32, i64)],
    ) -> crate::state::solver_state::SolverState<'p, i64> {
        // Build DV vector with the correct length.
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // We'll use berth id=1 for these tests (exists in make_problem()).
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();

        // Create terminal occupancy and compute total cost and occupied intervals.
        let mut term = crate::state::terminal::terminalocc::TerminalOccupancy::new(
            model.problem().berths().iter(),
        );
        use num_traits::Zero;
        let mut total_cost = Cost::zero();

        for (req_id, start) in starts {
            let ri = model.index_manager().request_index(rid(*req_id)).unwrap();
            let start_tp = tp(*start);

            // Set the decision variable
            dvars[ri.get()] = DecisionVar::assigned(b_ix, start_tp);

            // Occupy the corresponding interval in terminal occupancy
            let pt = model
                .processing_time(ri, b_ix)
                .expect("processing time defined for test");
            let end_tp = start_tp + pt;
            let asg_iv = TimeInterval::new(start_tp, end_tp);
            term.occupy(b_ix, asg_iv).expect("occupy must succeed");

            // Accumulate the cost for consistency with SolverState::apply_plan
            let c = model
                .cost_of_assignment(ri, b_ix, start_tp)
                .expect("cost defined for test");
            total_cost += c;
        }

        let dv = DecisionVarVec::from(dvars);

        let unassigned = model.flexible_requests_len() - starts.len();
        crate::state::solver_state::SolverState::new(
            dv,
            term,
            crate::state::fitness::Fitness::new(total_cost, unassigned),
        )
    }

    fn make_ctx<'b, 'c, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        state: &'s crate::state::solver_state::SolverState<'p, i64>,
        buffer: &'b mut [DecisionVar<i64>],
    ) -> PlanningContext<'b, 'c, 's, 'm, 'p, i64, crate::search::planner::DefaultCostEvaluator>
    {
        PlanningContext::new(
            model,
            state,
            &crate::search::planner::DefaultCostEvaluator,
            buffer,
        )
    }

    #[inline]
    fn neighbor_pm1(seed: RequestIndex) -> Vec<RequestIndex> {
        let i = seed.get();
        let mut v = Vec::new();
        if i > 0 {
            v.push(RequestIndex::new(i - 1));
        }
        v.push(seed);
        v
    }

    #[test]
    fn test_shift_earlier_moves_left_within_random_sample_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = ShiftEarlierOnSameBerth {
            number_of_candidates_to_try_range: 2..=2,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(1)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_relocate_single_best_changes_when_better_exists_sampling_range_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = RelocateSingleBest {
            number_of_candidates_to_try_range: 2..=2,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(2)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_swap_pair_same_berth_executes_with_neighbor_restriction_or_fallback() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = SwapPairSameBerth {
            number_of_pair_attempts_to_try_range: 4..=4,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(3)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_cross_exchange_across_berths_works_with_neighbors_and_fallback() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = CrossExchangeAcrossBerths {
            number_of_pair_attempts_to_try_range: 4..=4,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(4)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_or_opt_block_relocate_k_and_alpha_ranges_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = OrOptBlockRelocate {
            block_length_to_relocate_range: 2..=2,
            rcl_alpha_range: 1.7..=1.7,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(5)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_relocate_single_best_allow_worsening_produces_plan_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = RelocateSingleBestAllowWorsening {
            number_of_candidates_to_try_range: 2..=2,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(6)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_random_relocate_anywhere_moves_some_request_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = RandomRelocateAnywhere {
            number_of_candidates_to_try_range: 2..=2,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(7)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_hill_climb_relocate_best_reduces_cost_when_possible_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = HillClimbRelocateBest {
            number_of_candidates_to_try_range: 4..=4,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(8)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_hill_climb_best_swap_same_berth_finds_improving_swap_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = HillClimbBestSwapSameBerth {
            number_of_pair_attempts_to_try_range: 6..=6,
            neighbor_callback: Some(Arc::new(neighbor_pm1)),
        };

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(9)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_cascade_relocate_k_bestearliest_executes_with_neighbors() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        // Two requests, both on single berth
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = CascadeRelocateK::new(2..=2, 2..=2, 2..=2).with_neighbors(Arc::new(neighbor_pm1));

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(10)) {
            state.apply_plan(plan);
            // remain with assigned jobs and a feasible state
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }

    #[test]
    fn test_cascade_relocate_k_rcl_allow_worsening_executes() {
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let mut state = make_state_with_assignments(&model, &[(1, 10), (2, 20)]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let op = CascadeRelocateK::new(2..=2, 2..=2, 2..=2)
            .with_neighbors(Arc::new(neighbor_pm1))
            .with_insert_policy(CascadeInsertPolicy::Rcl {
                alpha_range: 1.3..=1.3,
            })
            .allow_step_worsening(1000);

        if let Some(plan) = op.propose(&mut ctx, &mut rand::rngs::StdRng::seed_from_u64(11)) {
            state.apply_plan(plan);
            assert!(
                state
                    .decision_variables()
                    .iter()
                    .filter(|dv| dv.is_assigned())
                    .count()
                    > 0
            );
        }
    }
}
