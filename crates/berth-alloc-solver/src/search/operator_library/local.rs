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
        operator::LocalMoveOperator,
        planner::{CostEvaluator, PlanBuilder, PlanExplorer, PlanningContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::Plan,
        terminal::terminalocc::FreeBerth,
    },
};
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::seq::SliceRandom;
use std::{ops::Mul, ops::RangeInclusive};

#[inline]
fn is_zero_delta_plan<T>(plan: &Plan<'_, T>) -> bool
where
    T: Copy + Ord,
{
    plan.delta_unassigned == 0 && plan.delta_cost == Cost::zero() && plan.terminal_delta.is_empty()
}

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

#[derive(Clone, Debug)]
pub struct ShiftEarlierOnSameBerth {
    /// How many assigned requests to randomly sample and try to shift earlier.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
}

impl ShiftEarlierOnSameBerth {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
        }
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
        let assigned_jobs: Vec<(RequestIndex, BerthIndex, TimePoint<T>)> = seed_builder
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
                            Some((RequestIndex::new(i), berth_index, start_time))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_jobs.is_empty() {
            return None;
        }

        let sample_size = clamp_range_sample(
            rng,
            &self.number_of_candidates_to_try_range,
            assigned_jobs.len(),
        )
        .max(1);
        let mut sampled_jobs = assigned_jobs.clone();
        sampled_jobs.shuffle(rng);
        sampled_jobs.truncate(sample_size);

        // Try candidates one by one, each attempt with a FRESH builder.
        for (request_index, berth_index, current_start_time) in sampled_jobs {
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

            let plan = attempt_builder.finalize();
            if !is_zero_delta_plan(&plan) {
                return Some(plan);
            }
        }

        None
    }
}

// ======================================================================
// RelocateSingleBest  (range-based sampling)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RelocateSingleBest {
    /// How many assigned requests to randomly sample and try to relocate.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
}

impl RelocateSingleBest {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
        }
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
        let assigned_with_cost: Vec<(RequestIndex, BerthIndex, TimePoint<T>, Cost)> = seed_builder
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
                            let ri = RequestIndex::new(i);
                            ex.peek_cost(ri, start_time, berth_index)
                                .map(|current_cost| (ri, berth_index, start_time, current_cost))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_with_cost.is_empty() {
            return None;
        }

        let sample_size = clamp_range_sample(
            rng,
            &self.number_of_candidates_to_try_range,
            assigned_with_cost.len(),
        )
        .max(1);
        let mut sampled = assigned_with_cost.clone();
        sampled.shuffle(rng);
        sampled.truncate(sample_size);

        for (request_index, current_berth, current_start_time, current_cost) in sampled {
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

            let plan = attempt_builder.finalize();
            if !is_zero_delta_plan(&plan) {
                return Some(plan);
            }
        }

        None
    }
}

// ======================================================================
// SwapPairSameBerth  (range-based pair attempts; fresh builder per attempt)
// ======================================================================

#[derive(Clone, Debug)]
pub struct SwapPairSameBerth {
    /// How many random same-berth pairs to attempt before giving up.
    pub number_of_pair_attempts_to_try_range: RangeInclusive<usize>,
}

impl SwapPairSameBerth {
    pub fn new(number_of_pair_attempts_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_pair_attempts_to_try_range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range,
        }
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
        let by_berth = seed_builder.with_explorer(|ex| {
            use std::collections::BTreeMap;
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

        let attempts_to_try =
            clamp_range_sample(rng, &self.number_of_pair_attempts_to_try_range, usize::MAX).max(1);

        // Build a flattened list of (berth, idx_a, idx_b) choices lazily by sampling.
        for _attempt in 0..attempts_to_try {
            // Pick a random berth that has at least two assignments.
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

            let plan = attempt_builder.finalize();
            if !plan.decision_var_patches.is_empty() {
                return Some(plan);
            }
        }

        None
    }
}

// ======================================================================
// CrossExchangeAcrossBerths  (range-based pair attempts; fresh builder per attempt)
// ======================================================================

#[derive(Clone, Debug)]
pub struct CrossExchangeAcrossBerths {
    /// How many random cross-berth pairs to attempt before giving up.
    pub number_of_pair_attempts_to_try_range: RangeInclusive<usize>,
}

impl CrossExchangeAcrossBerths {
    pub fn new(number_of_pair_attempts_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_pair_attempts_to_try_range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range,
        }
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

        // Snapshot all assigned as (ri, berth, start).
        let seed_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();
        let assigned: Vec<(RequestIndex, BerthIndex, TimePoint<T>)> =
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
                            Some((RequestIndex::new(i), berth_index, start_time))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned.len() < 2 {
            return None;
        }

        let attempts_to_try =
            clamp_range_sample(rng, &self.number_of_pair_attempts_to_try_range, usize::MAX).max(1);

        for _attempt in 0..attempts_to_try {
            // Randomize a pair; keep only cross-berth pairs.
            let mut shuffled = assigned.clone();
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
                    let (ra, bi_a, sa) = shuffled[i];
                    let (rb, bi_b, sb) = shuffled[j];
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

            // quick sanity: both targets allowed by model
            if model.cost_of_assignment(ra, bi_b, sb).is_none() {
                continue;
            }
            if model.cost_of_assignment(rb, bi_a, sa).is_none() {
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

            let plan = attempt_builder.finalize();
            if !plan.decision_var_patches.is_empty() {
                return Some(plan);
            }
        }

        None
    }
}

// ======================================================================
// OrOptBlockRelocate  (ranges for block length + RCL alpha)
// ======================================================================

#[derive(Clone, Debug)]
pub struct OrOptBlockRelocate {
    /// Inclusive range for the contiguous block length (k) to relocate.
    pub block_length_to_relocate_range: RangeInclusive<usize>,
    /// Inclusive range for RCL alpha when picking the block's starting index.
    pub rcl_alpha_range: RangeInclusive<f64>,
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
        }
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
            use std::collections::BTreeMap;
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

        let (_berth_index, ordered_sequence) = maybe_group?;

        // Sample k and alpha.
        let sampled_k = clamp_range_sample(
            rng,
            &self.block_length_to_relocate_range,
            ordered_sequence.len(),
        )
        .max(2);
        if sampled_k > ordered_sequence.len() {
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

        // Choose a start index with RCL.
        let valid_start_count = ordered_sequence.len() - sampled_k + 1;
        let start_index = rcl_index(valid_start_count, alpha_sample, rng);
        let block_slice = &ordered_sequence[start_index..start_index + sampled_k];
        let block_indices: Vec<RequestIndex> = block_slice.iter().map(|&(ri, _)| ri).collect();

        // Rebuild on a fresh builder: unassign block → greedy reinsert anywhere.
        let mut attempt_builder: PlanBuilder<'_, 'c, 's, 'm, 'p, T, C> = context.builder();

        for &ri in &block_indices {
            let _ = attempt_builder.propose_unassignment(ri).ok()?;
        }
        for &ri in &block_indices {
            if let Some((free, start, _cost)) =
                attempt_builder.with_explorer(|ex| best_insertion_for_request_ex(ex, model, ri))
            {
                let _ = attempt_builder.propose_assignment(ri, start, &free).ok();
            }
        }

        let plan = attempt_builder.finalize();
        if plan.decision_var_patches.is_empty() {
            None
        } else {
            Some(plan)
        }
    }
}

// ======================================================================
// RelocateSingleBestAllowWorsening  (same as RelocateSingleBest but allows worsening)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RelocateSingleBestAllowWorsening {
    /// How many assigned requests to randomly sample and try to relocate.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
}

impl RelocateSingleBestAllowWorsening {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
        }
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
        let assigned_with_cost: Vec<(RequestIndex, BerthIndex, TimePoint<T>, Cost)> = seed_builder
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
                            let ri = RequestIndex::new(i);
                            ex.peek_cost(ri, start_time, berth_index)
                                .map(|current_cost| (ri, berth_index, start_time, current_cost))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_with_cost.is_empty() {
            return None;
        }

        let sample_size = clamp_range_sample(
            rng,
            &self.number_of_candidates_to_try_range,
            assigned_with_cost.len(),
        )
        .max(1);
        let mut sampled = assigned_with_cost.clone();
        sampled.shuffle(rng);
        sampled.truncate(sample_size);

        for (request_index, _current_berth, _current_start_time, _current_cost) in sampled {
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

            let plan = attempt_builder.finalize();
            if !is_zero_delta_plan(&plan) {
                return Some(plan);
            }
        }

        None
    }
}

// ======================================================================
// RandomRelocateAnywhere  (random feasible reinsert; can worsen on purpose)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RandomRelocateAnywhere {
    /// How many assigned requests to randomly sample and try to reinsert randomly.
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
}

impl RandomRelocateAnywhere {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
        }
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
        let assigned_jobs: Vec<(RequestIndex, BerthIndex, TimePoint<T>)> = seed_builder
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
                            Some((RequestIndex::new(i), berth_index, start_time))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned_jobs.is_empty() {
            return None;
        }

        let sample_size = clamp_range_sample(
            rng,
            &self.number_of_candidates_to_try_range,
            assigned_jobs.len(),
        )
        .max(1);

        // Shuffle & truncate candidates.
        let mut sampled = assigned_jobs.clone();
        sampled.shuffle(rng);
        sampled.truncate(sample_size);

        for (ri, _cur_b, _cur_s) in sampled {
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

            let plan = attempt_builder.finalize();
            if !is_zero_delta_plan(&plan) {
                return Some(plan);
            }
        }

        None
    }
}

// ======================================================================
// HillClimbRelocateBest  (best improving relocate among sampled requests)
// ======================================================================

#[derive(Clone, Debug)]
pub struct HillClimbRelocateBest {
    /// How many assigned requests to sample and try (upper bound).
    pub number_of_candidates_to_try_range: RangeInclusive<usize>,
}

impl HillClimbRelocateBest {
    pub fn new(number_of_candidates_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_candidates_to_try_range.is_empty());
        Self {
            number_of_candidates_to_try_range,
        }
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
        let assigned: Vec<(RequestIndex, BerthIndex, TimePoint<T>)> =
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
                            Some((RequestIndex::new(i), berth_index, start_time))
                        } else {
                            None
                        }
                    })
                    .collect()
            });
        if assigned.is_empty() {
            return None;
        }

        // Sample a subset to keep per-call work bounded.
        let k =
            clamp_range_sample(rng, &self.number_of_candidates_to_try_range, assigned.len()).max(1);
        let mut cand_reqs = assigned.clone();
        cand_reqs.shuffle(rng);
        cand_reqs.truncate(k);

        // Track best improving plan by lexicographic delta.
        let mut best_plan: Option<Plan<'p, T>> = None;
        let mut best_delta_unassigned: i64 = 0;
        let mut best_delta_cost: Cost = Cost::zero();

        // Lex compare on deltas: smaller is better; unassigned first, then cost.
        #[inline]
        fn lex_better(du_a: i64, dc_a: Cost, du_b: i64, dc_b: Cost) -> bool {
            (du_a < du_b) || (du_a == du_b && dc_a < dc_b)
        }

        for (ri, _cur_b, _cur_s) in cand_reqs {
            // Explore all feasible earliest placements; pick the one that yields the best improvement.
            // We will construct a plan for each feasible "best_insertion_for_request_ex" and evaluate deltas.
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
                let plan = bld.finalize();

                if is_zero_delta_plan(&plan) {
                    continue;
                }

                // Improvement check (lex on deltas).
                let du = plan.delta_unassigned as i64;
                let dc = plan.delta_cost;

                // Must be an improving move.
                if du < 0 || (du == 0 && dc < 0) {
                    match &best_plan {
                        None => {
                            best_delta_unassigned = du;
                            best_delta_cost = dc;
                            best_plan = Some(plan);
                        }
                        Some(_) => {
                            if lex_better(du, dc, best_delta_unassigned, best_delta_cost) {
                                best_delta_unassigned = du;
                                best_delta_cost = dc;
                                best_plan = Some(plan);
                            }
                        }
                    }
                }
            }
        }

        best_plan
    }
}

// ======================================================================
// HillClimbBestSwapSameBerth  (best improving swap among sampled same-berth pairs)
// ======================================================================

#[derive(Clone, Debug)]
pub struct HillClimbBestSwapSameBerth {
    /// How many random same-berth pairs to sample per call (upper bound).
    pub number_of_pair_attempts_to_try_range: RangeInclusive<usize>,
}

impl HillClimbBestSwapSameBerth {
    pub fn new(number_of_pair_attempts_to_try_range: RangeInclusive<usize>) -> Self {
        assert!(!number_of_pair_attempts_to_try_range.is_empty());
        Self {
            number_of_pair_attempts_to_try_range,
        }
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
        let by_berth = seed_builder.with_explorer(|ex| {
            use std::collections::BTreeMap;
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

        // Track best improving swap plan (lex on deltas).
        let mut best_plan: Option<Plan<'p, T>> = None;
        let mut best_du: i64 = 0;
        let mut best_dc: Cost = Cost::zero();

        #[inline]
        fn lex_better(du_a: i64, dc_a: Cost, du_b: i64, dc_b: Cost) -> bool {
            (du_a < du_b) || (du_a == du_b && dc_a < dc_b)
        }

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
            let plan = bld.finalize();
            if is_zero_delta_plan(&plan) {
                continue;
            }

            // Improvement?
            let du = plan.delta_unassigned as i64;
            let dc = plan.delta_cost;
            if du < 0 || (du == 0 && dc < 0) {
                match &best_plan {
                    None => {
                        best_du = du;
                        best_dc = dc;
                        best_plan = Some(plan);
                    }
                    Some(_) => {
                        if lex_better(du, dc, best_du, best_dc) {
                            best_du = du;
                            best_dc = dc;
                            best_plan = Some(plan);
                        }
                    }
                }
            }
        }

        best_plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::planner::{DefaultCostEvaluator, PlanningContext},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    // ------------ helpers ------------
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

    fn make_problem(n: usize) -> Problem<i64> {
        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();
        builder.add_berth(berth(1, 0, 1000));
        for i in 1..=n {
            builder.add_flexible(flex_req(i as u32, (0, 200), &[(1, 10)], 1));
        }
        builder.build().expect("valid problem")
    }

    fn make_state_with_assignments<'p>(
        model: &SolverModel<'p, i64>,
        starts: &[i64],
    ) -> SolverState<'p, i64> {
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let mut dvars = Vec::with_capacity(model.flexible_requests_len());
        for (i, _) in (0..model.flexible_requests_len()).enumerate() {
            if i < starts.len() {
                dvars.push(DecisionVar::assigned(b_ix, tp(starts[i])));
            } else {
                dvars.push(DecisionVar::unassigned());
            }
        }
        let dv = DecisionVarVec::from(dvars);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let unassigned = model.flexible_requests_len().saturating_sub(starts.len());
        let fit = Fitness::new(0, unassigned);
        SolverState::new(dv, term, fit)
    }

    fn make_ctx<'b, 'c, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        cost_evaluator: &'c DefaultCostEvaluator,
        state: &'s SolverState<'p, i64>,
        buffer: &'b mut [DecisionVar<i64>],
    ) -> PlanningContext<'b, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator> {
        PlanningContext::new(model, state, cost_evaluator, buffer)
    }

    // ------------ tests ------------

    #[test]
    fn shift_earlier_moves_left_within_random_sample() {
        let prob = make_problem(1);
        let model = SolverModel::try_from(&prob).unwrap();
        // Starts later than earliest; operator should move to earliest (=0)
        let state = make_state_with_assignments(&model, &[50]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);

        let mut rng = ChaCha8Rng::from_seed([7; 32]);
        let op = ShiftEarlierOnSameBerth::new(1..=3);
        let plan = op.propose(&mut ctx, &mut rng).expect("expected a plan");

        assert_eq!(plan.decision_var_patches.len(), 2); // unassign + assign
        assert_eq!(plan.delta_unassigned, 0);
        assert!(!plan.terminal_delta.is_empty());

        // Verify new start ≤ old start
        let new_start = plan
            .decision_var_patches
            .iter()
            .rev()
            .find_map(|p| {
                if let DecisionVar::Assigned(dec) = p.patch {
                    Some(dec.start_time)
                } else {
                    None
                }
            })
            .expect("assigned patch must exist");
        assert!(new_start <= tp(50));
    }

    #[test]
    fn relocate_single_best_changes_when_better_exists_sampling_range() {
        let prob = make_problem(2);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_state_with_assignments(&model, &[60, 80]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);

        let mut rng = ChaCha8Rng::from_seed([9; 32]);
        let op = RelocateSingleBest::new(1..=2);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert_eq!(plan.delta_unassigned, 0);
        assert!(plan.decision_var_patches.len() >= 2); // unassign + assign
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn swap_pair_same_berth_executes_with_fresh_builder_attempts() {
        let prob = make_problem(2);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_state_with_assignments(&model, &[0, 40]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);

        let mut rng = ChaCha8Rng::from_seed([3; 32]);
        let op = SwapPairSameBerth::new(1..=4);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert_eq!(plan.decision_var_patches.len(), 4); // unassign both + assign both
        assert_eq!(plan.delta_unassigned, 0);
        assert!(!plan.terminal_delta.is_empty());

        let mut starts = plan
            .decision_var_patches
            .iter()
            .filter_map(|p| {
                if let DecisionVar::Assigned(dec) = p.patch {
                    Some(dec.start_time.value())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        starts.sort_unstable();
        assert_eq!(starts, vec![0, 40]); // swapped positions
    }

    #[test]
    fn cross_exchange_across_berths_works_with_fresh_builder_attempts() {
        // Two berths, one request on each to allow cross-exchange.
        let mut pb = berth_alloc_model::problem::builder::ProblemBuilder::new();
        pb.add_berth(berth(1, 0, 1_000));
        pb.add_berth(berth(2, 0, 1_000));
        pb.add_flexible(flex_req(1, (0, 200), &[(1, 10), (2, 10)], 1));
        pb.add_flexible(flex_req(2, (0, 200), &[(1, 10), (2, 10)], 1));
        let prob = pb.build().expect("ok");
        let model = SolverModel::try_from(&prob).unwrap();

        // Assign R1@B1 at 0, R2@B2 at 40
        let b1 = model.index_manager().berth_index(bid(1)).unwrap();
        let b2 = model.index_manager().berth_index(bid(2)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[0] = DecisionVar::assigned(b1, tp(0));
        dvars[1] = DecisionVar::assigned(b2, tp(40));
        let dv = DecisionVarVec::from(dvars);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let fit = Fitness::new(0, 0);
        let state = SolverState::new(dv, term, fit);

        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);

        let mut rng = ChaCha8Rng::from_seed([11; 32]);
        let op = CrossExchangeAcrossBerths::new(1..=3);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert_eq!(plan.delta_unassigned, 0);
        assert!(plan.decision_var_patches.len() >= 4);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn or_opt_block_relocate_k_and_alpha_ranges() {
        let prob = make_problem(6);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_state_with_assignments(&model, &[0, 20, 40, 60, 80, 100]);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);

        let mut rng = ChaCha8Rng::from_seed([13; 32]);
        // Fix k=3 and alpha=1.7 for determinism in this test.
        let op = OrOptBlockRelocate::new(3..=3, 1.7..=1.7);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert!(plan.decision_var_patches.len() >= 6); // unassign 3 + assign ≤3
        assert_eq!(plan.delta_unassigned, 0);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn operators_return_none_when_no_assigned() {
        let prob = make_problem(2);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_state_with_assignments(&model, &[]); // nobody assigned
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);

        let mut rng = ChaCha8Rng::from_seed([1; 32]);

        let op_shift = ShiftEarlierOnSameBerth::new(1..=2);
        let op_reloc = RelocateSingleBest::new(1..=2);
        let op_swap = SwapPairSameBerth::new(1..=2);
        let op_cross = CrossExchangeAcrossBerths::new(1..=2);
        let op_oropt = OrOptBlockRelocate::new(2..=3, 1.4..=1.8);

        assert!(op_shift.propose(&mut ctx, &mut rng).is_none());
        assert!(op_reloc.propose(&mut ctx, &mut rng).is_none());
        assert!(op_swap.propose(&mut ctx, &mut rng).is_none());
        assert!(op_cross.propose(&mut ctx, &mut rng).is_none());
        assert!(op_oropt.propose(&mut ctx, &mut rng).is_none());
    }

    #[test]
    fn test_relocate_single_best_allow_worsening_produces_plan() {
        let prob = make_problem(2);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_state_with_assignments(&model, &[60, 80]);

        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);
        let mut rng = ChaCha8Rng::from_seed([17; 32]);

        let op = RelocateSingleBestAllowWorsening::new(1..=3);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert_eq!(plan.delta_unassigned, 0);
        assert!(
            plan.decision_var_patches.len() >= 2,
            "should perform an unassign+assign"
        );
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_random_relocate_anywhere_moves_some_request() {
        let prob = make_problem(3);
        let model = SolverModel::try_from(&prob).unwrap();
        // Three assigned at spaced times to give room for relocations
        let state = make_state_with_assignments(&model, &[0, 30, 70]);

        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);
        let mut rng = ChaCha8Rng::from_seed([19; 32]);

        let op = RandomRelocateAnywhere::new(2..=4);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert_eq!(plan.delta_unassigned, 0);
        assert!(
            plan.decision_var_patches.len() >= 2,
            "expect an unassign + assign"
        );
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_hill_climb_relocate_best_reduces_cost_when_possible() {
        let prob = make_problem(1);
        let model = SolverModel::try_from(&prob).unwrap();
        // Start later than arrival; moving earlier should reduce base cost
        let state = make_state_with_assignments(&model, &[50]);

        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);
        let mut rng = ChaCha8Rng::from_seed([23; 32]);

        let op = HillClimbRelocateBest::new(1..=3);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        // Expect strict cost improvement on the base objective
        assert!(
            plan.delta_cost < 0,
            "hill-climb should reduce base cost when earlier start is possible; got delta_cost={}",
            plan.delta_cost
        );
        assert_eq!(plan.delta_unassigned, 0);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_hill_climb_best_swap_same_berth_finds_improving_swap() {
        // Build a problem where swapping yields a clear improvement due to different weights.
        let mut pb = berth_alloc_model::problem::builder::ProblemBuilder::new();
        pb.add_berth(berth(1, 0, 1_000));
        // Two requests allowed on the same berth, different weights
        // PT = 10 for both; arrival = 0 (window start)
        pb.add_flexible(flex_req(1, (0, 200), &[(1, 10)], 10)); // heavy
        pb.add_flexible(flex_req(2, (0, 200), &[(1, 10)], 1)); // light
        let prob = pb.build().expect("ok");
        let model = SolverModel::try_from(&prob).unwrap();

        let b1 = model.index_manager().berth_index(bid(1)).unwrap();

        // Assign heavy late and light early: [R1@40, R2@0]
        // Swapping them should greatly reduce cost.
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[0] = DecisionVar::assigned(b1, tp(40)); // heavy
        dvars[1] = DecisionVar::assigned(b1, tp(0)); // light
        let dv = DecisionVarVec::from(dvars);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let state = SolverState::new(dv, term, Fitness::new(0, 0));

        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &DefaultCostEvaluator, &state, &mut buffer);
        let mut rng = ChaCha8Rng::from_seed([29; 32]);

        let op = HillClimbBestSwapSameBerth::new(1..=5);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        // Expect a strict improvement
        assert!(
            plan.delta_cost < 0,
            "swap should reduce base cost with asymmetric weights; got delta_cost={}",
            plan.delta_cost
        );
        assert_eq!(plan.delta_unassigned, 0);
        assert!(
            plan.decision_var_patches.len() >= 4,
            "expect unassign+assign for both requests"
        );
        assert!(!plan.terminal_delta.is_empty());
    }
}
