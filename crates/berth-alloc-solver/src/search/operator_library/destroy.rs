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
        operator::{DestroyOperator, NeighborFn},
        planner::{PlanBuilder, PlanExplorer, PlanningContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        plan::Plan,
    },
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::{
    Rng,
    seq::{IteratorRandom, SliceRandom},
};
use std::{ops::RangeInclusive, sync::Arc};

// ---------------------------------------------------------------------
// Type aliases for nicer, explicit APIs
// ---------------------------------------------------------------------

type RatioRange = RangeInclusive<f64>;
type AlphaRange = RangeInclusive<f64>;

/// Sample a value from an inclusive f64 range using the provided RNG.
#[inline]
fn sample_f64_inclusive<R: Rng>(range: &RangeInclusive<f64>, rng: &mut R) -> f64 {
    rng.random_range(range.clone())
}

/// Returns `true` when a `Plan` contains no changes (no patches, no terminal delta, zero deltas).
#[inline]
fn is_zero_delta_plan<T>(plan: &Plan<'_, T>) -> bool
where
    T: Copy + Ord,
{
    plan.delta_unassigned == 0 && plan.delta_cost == Cost::zero() && plan.terminal_delta.is_empty()
}

/// Randomized–greedy index selector (RCL) used in Shaw-style operators.
/// For `len > 0`, returns an index in `[0, len-1]`.
/// `greediness_alpha = 1.0` ≈ uniform; larger values bias toward lower indices after sorting.
#[inline]
fn randomized_greedy_index<R: Rng>(length: usize, greediness_alpha: f64, rng: &mut R) -> usize {
    debug_assert!(length > 0, "randomized_greedy_index called with len=0");
    let p: f64 = rng.random_range(0.0..1.0);
    let idx = ((length as f64) * p.powf(greediness_alpha)).ceil() as usize;
    idx.saturating_sub(1).min(length - 1)
}

// ======================================================================
// Small explorer-based utilities used by operators
// ======================================================================

#[inline]
fn triples_from_explorer<'e, 'm, 'p, T>(
    explorer: &PlanExplorer<'e, '_, 'm, 'p, T>,
    model: &crate::model::solver_model::SolverModel<'m, T>,
) -> Vec<(RequestIndex, BerthIndex, TimeInterval<T>)>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let decision_vars = explorer.decision_vars();
    let mut out = Vec::new();
    for (i, dv) in decision_vars.iter().enumerate() {
        if let DecisionVar::Assigned(Decision {
            berth_index,
            start_time,
        }) = *dv
        {
            let request_index = RequestIndex::new(i);
            if let Some(processing_time) = model.processing_time(request_index, berth_index) {
                out.push((
                    request_index,
                    berth_index,
                    TimeInterval::new(start_time, start_time + processing_time),
                ));
            }
        }
    }
    out
}

#[inline]
fn neighborhood_triples_from_explorer<'e, 'm, 'p, T>(
    explorer: &PlanExplorer<'e, '_, 'm, 'p, T>,
    model: &crate::model::solver_model::SolverModel<'m, T>,
    seed_request: RequestIndex,
    neighbor_cb: &NeighborFn,
) -> Vec<(RequestIndex, BerthIndex, TimeInterval<T>)>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    let decision_vars = explorer.decision_vars();
    let mut out = Vec::new();

    let mut push_if_assigned = |request_index: RequestIndex| {
        if let DecisionVar::Assigned(Decision {
            berth_index,
            start_time,
        }) = decision_vars[request_index.get()]
            && let Some(processing_time) = model.processing_time(request_index, berth_index)
        {
            let interval = TimeInterval::new(start_time, start_time + processing_time);
            if !out.iter().any(|(r, _, _)| *r == request_index) {
                out.push((request_index, berth_index, interval));
            }
        }
    };

    push_if_assigned(seed_request);
    for n in neighbor_cb(seed_request) {
        if n.get() < decision_vars.len() {
            push_if_assigned(n);
        }
    }
    out
}

// ======================================================================
// Destroy Operators (range-based configuration)
// ======================================================================

/// Randomly removes a `ratio` fraction of currently assigned requests (ratio sampled from range).
/// If `neighbor_callback` is provided and returns assigned neighbors for a random seed,
/// the removal is sampled from that localized pool; otherwise falls back to all assigned.
#[derive(Clone)]
pub struct RandomKRatioDestroy {
    ratio_range: RatioRange,
    neighbor_callback: Option<Arc<NeighborFn>>,
}

impl RandomKRatioDestroy {
    pub fn new(ratio_range: RatioRange) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        Self {
            ratio_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for RandomKRatioDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "RandomKRatioDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        // Assigned indices from explorer snapshot
        let assigned_indices: Vec<RequestIndex> = plan_builder.with_explorer(|explorer| {
            explorer
                .decision_vars()
                .iter()
                .enumerate()
                .filter_map(|(i, dv)| {
                    if dv.is_assigned() {
                        Some(RequestIndex::new(i))
                    } else {
                        None
                    }
                })
                .collect()
        });

        if assigned_indices.is_empty() {
            return None;
        }

        let target_removal_count =
            ((assigned_indices.len() as f64) * sampled_ratio).ceil() as usize;
        let target_removal_count = target_removal_count.clamp(1, assigned_indices.len());

        let &seed_request = assigned_indices.iter().choose(rng)?;

        // Prefer a localized explorer pool when possible.
        let mut candidate_triples = plan_builder.with_explorer(|explorer| {
            if let Some(callback) = &self.neighbor_callback {
                let local = neighborhood_triples_from_explorer(
                    explorer,
                    model,
                    seed_request,
                    callback.as_ref(),
                );
                if local.is_empty() {
                    triples_from_explorer(explorer, model)
                } else {
                    local
                }
            } else {
                triples_from_explorer(explorer, model)
            }
        });

        if candidate_triples.is_empty() {
            return None;
        }

        candidate_triples.shuffle(rng);
        let mut selected_for_removal: Vec<RequestIndex> = candidate_triples
            .into_iter()
            .map(|t| t.0)
            .take(target_removal_count)
            .collect();

        // Top up from global if the local neighborhood is too small.
        if selected_for_removal.len() < target_removal_count {
            for request_index in assigned_indices.into_iter() {
                if selected_for_removal.len() == target_removal_count {
                    break;
                }
                if !selected_for_removal.contains(&request_index) {
                    selected_for_removal.push(request_index);
                }
            }
        }

        // Apply removals via builder
        for request_index in selected_for_removal {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

/// Removes the top `ratio` fraction of assigned requests by *current assignment cost*,
/// optionally restricted to a local neighborhood. Ratio is sampled from range.
/// If the callback yields no candidates, falls back to all assigned.
#[derive(Clone)]
pub struct WorstCostDestroy {
    ratio_range: RatioRange,
    neighbor_callback: Option<Arc<NeighborFn>>,
}

impl WorstCostDestroy {
    pub fn new(ratio_range: RatioRange) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        Self {
            ratio_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for WorstCostDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "WorstCostDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        // Global assigned indices (for target K and seed selection).
        let assigned_indices: Vec<RequestIndex> = plan_builder.with_explorer(|explorer| {
            explorer
                .decision_vars()
                .iter()
                .enumerate()
                .filter_map(|(i, dv)| {
                    if dv.is_assigned() {
                        Some(RequestIndex::new(i))
                    } else {
                        None
                    }
                })
                .collect()
        });

        if assigned_indices.is_empty() {
            return None;
        }

        let global_target_count = ((assigned_indices.len() as f64) * sampled_ratio).ceil() as usize;
        let global_target_count = global_target_count.clamp(1, assigned_indices.len());

        // Seed for neighborhood restriction (when a callback exists).
        let &seed_request = assigned_indices.iter().choose(rng)?;

        // Candidate pool (local if provided & non-empty; else full), cost-sorted desc.
        let pool = plan_builder.with_explorer(|explorer| {
            let mut triples = if let Some(callback) = &self.neighbor_callback {
                let local = neighborhood_triples_from_explorer(
                    explorer,
                    model,
                    seed_request,
                    callback.as_ref(),
                );
                if local.is_empty() {
                    triples_from_explorer(explorer, model)
                } else {
                    local
                }
            } else {
                triples_from_explorer(explorer, model)
            };

            // sort by cost desc using explorer peek
            triples.sort_by(|a, b| {
                let cost_a = explorer
                    .peek_cost(a.0, a.2.start(), a.1)
                    .unwrap_or_else(Cost::zero);
                let cost_b = explorer
                    .peek_cost(b.0, b.2.start(), b.1)
                    .unwrap_or_else(Cost::zero);
                cost_b.cmp(&cost_a)
            });
            triples
        });

        if pool.is_empty() {
            return None;
        }

        let mut selected: Vec<RequestIndex> =
            pool.iter().take(global_target_count).map(|t| t.0).collect();

        // TOP-UP: if local neighborhood was too small, add the best remaining from the full pool.
        if selected.len() < global_target_count {
            let mut global_pool = plan_builder.with_explorer(|explorer| {
                let mut g = triples_from_explorer(explorer, model);
                g.sort_by(|a, b| {
                    let cost_a = explorer
                        .peek_cost(a.0, a.2.start(), a.1)
                        .unwrap_or_else(Cost::zero);
                    let cost_b = explorer
                        .peek_cost(b.0, b.2.start(), b.1)
                        .unwrap_or_else(Cost::zero);
                    cost_b.cmp(&cost_a)
                });
                g
            });
            for (ri, _, _) in global_pool.drain(..) {
                if selected.len() == global_target_count {
                    break;
                }
                if !selected.contains(&ri) {
                    selected.push(ri);
                }
            }
        }

        for request_index in selected {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

/// Shaw-style relatedness removal around a seed, guided by
/// weighted absolute start/end time gaps and a berth mismatch penalty.
/// Uses randomized–greedy selection with `alpha` sampled from range.
#[derive(Clone)]
pub struct ShawRelatedDestroy {
    ratio_range: RatioRange,
    alpha_range: AlphaRange,
    weight_abs_start_gap: Cost,
    weight_abs_end_gap: Cost,
    penalty_berth_mismatch: Cost,
    neighbor_callback: Option<Arc<NeighborFn>>,
}

impl ShawRelatedDestroy {
    pub fn new(
        ratio_range: RatioRange,
        alpha_range: AlphaRange,
        weight_abs_start_gap: Cost,
        weight_abs_end_gap: Cost,
        penalty_berth_mismatch: Cost,
    ) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        assert!(*alpha_range.start() > 0.0 && alpha_range.end().is_finite());
        Self {
            ratio_range,
            alpha_range,
            weight_abs_start_gap,
            weight_abs_end_gap,
            penalty_berth_mismatch,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for ShawRelatedDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "ShawRelatedDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        let sampled_alpha = sample_f64_inclusive(&self.alpha_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        let full_pool = plan_builder.with_explorer(|explorer| {
            let mut v = triples_from_explorer(explorer, model);
            v.sort_by(|a, b| {
                let len_a = (a.2.end() - a.2.start()).value();
                let len_b = (b.2.end() - b.2.start()).value();
                len_a.cmp(&len_b)
            });
            v
        });

        if full_pool.is_empty() {
            return None;
        }

        let target_removal_count = ((full_pool.len() as f64) * sampled_ratio).ceil() as usize;

        let seed_idx = randomized_greedy_index(full_pool.len(), sampled_alpha, rng);
        let (seed_request, seed_berth, seed_interval) = full_pool[seed_idx];

        let mut candidate_pool = plan_builder.with_explorer(|explorer| {
            if let Some(callback) = &self.neighbor_callback {
                let local = neighborhood_triples_from_explorer(
                    explorer,
                    model,
                    seed_request,
                    callback.as_ref(),
                );
                if local.is_empty() {
                    full_pool.clone()
                } else {
                    local
                }
            } else {
                full_pool.clone()
            }
        });

        if !candidate_pool.iter().any(|(ri, _, _)| *ri == seed_request) {
            candidate_pool.push((seed_request, seed_berth, seed_interval));
        }
        if candidate_pool.is_empty() {
            return None;
        }

        let mut selected_for_removal: Vec<RequestIndex> =
            Vec::with_capacity(target_removal_count.max(1));
        selected_for_removal.push(seed_request);

        let mut current_berth = seed_berth;
        let mut current_interval = seed_interval;

        while selected_for_removal.len() < target_removal_count && !candidate_pool.is_empty() {
            // score by relatedness
            let mut scored: Vec<(usize, Cost)> = candidate_pool
                .iter()
                .enumerate()
                .filter(|(_, (ri, _, _))| !selected_for_removal.contains(ri))
                .map(|(idx, (_, berth, iv))| {
                    let start_gap_cost = {
                        let dt = if iv.start() >= current_interval.start() {
                            (iv.start() - current_interval.start()).value()
                        } else {
                            (current_interval.start() - iv.start()).value()
                        };
                        Into::<Cost>::into(dt)
                    };
                    let end_gap_cost = {
                        let dt = if iv.end() >= current_interval.end() {
                            (iv.end() - current_interval.end()).value()
                        } else {
                            (current_interval.end() - iv.end()).value()
                        };
                        Into::<Cost>::into(dt)
                    };
                    let berth_penalty = if *berth == current_berth {
                        Cost::zero()
                    } else {
                        self.penalty_berth_mismatch
                    };
                    let score = self
                        .weight_abs_start_gap
                        .saturating_mul(start_gap_cost)
                        .saturating_add(self.weight_abs_end_gap.saturating_mul(end_gap_cost))
                        .saturating_add(berth_penalty);
                    (idx, score)
                })
                .collect();

            if scored.is_empty() {
                break;
            }

            scored.sort_by(|a, b| a.1.cmp(&b.1));
            let pick = randomized_greedy_index(scored.len(), sampled_alpha, rng);
            let idx_in_pool = scored[pick].0;
            let (ri, berth, iv) = candidate_pool[idx_in_pool];
            selected_for_removal.push(ri);

            if rng.random_bool(0.5) {
                current_berth = berth;
                current_interval = iv;
            }
        }

        selected_for_removal.truncate(target_removal_count.max(1));

        for request_index in selected_for_removal {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

/// Removes a seed and its **temporal or same-berth neighbors**.
/// If a neighbor callback exists, it defines the candidate neighborhood; otherwise,
/// the neighborhood defaults to “same berth as the seed, sorted by |Δstart|”.
#[derive(Clone)]
pub struct BerthNeighborsDestroy {
    ratio_range: RatioRange,
    alpha_range: AlphaRange,
    neighbor_callback: Option<Arc<NeighborFn>>,
}

impl BerthNeighborsDestroy {
    pub fn new(ratio_range: RatioRange, alpha_range: AlphaRange) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        assert!(*alpha_range.start() > 0.0 && alpha_range.end().is_finite());
        Self {
            ratio_range,
            alpha_range,
            neighbor_callback: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbor_callback = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for BerthNeighborsDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "BerthNeighborsDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        let sampled_alpha = sample_f64_inclusive(&self.alpha_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        let full_pool = plan_builder.with_explorer(|explorer| {
            let mut v = triples_from_explorer(explorer, model);
            v.sort_by(|a, b| {
                let la = (a.2.end() - a.2.start()).value();
                let lb = (b.2.end() - b.2.start()).value();
                la.cmp(&lb)
            });
            v
        });
        if full_pool.is_empty() {
            return None;
        }

        let target_removal_count = ((full_pool.len() as f64) * sampled_ratio).ceil() as usize;
        let target_removal_count = target_removal_count.clamp(1, full_pool.len());

        let seed_idx = randomized_greedy_index(full_pool.len(), sampled_alpha, rng);
        let (seed_request, seed_berth, seed_interval) = full_pool[seed_idx];

        let mut candidate_pool = plan_builder.with_explorer(|explorer| {
            if let Some(callback) = &self.neighbor_callback {
                let local = neighborhood_triples_from_explorer(
                    explorer,
                    model,
                    seed_request,
                    callback.as_ref(),
                );
                if local.is_empty() {
                    full_pool.clone()
                } else {
                    local
                }
            } else {
                full_pool
                    .clone()
                    .into_iter()
                    .filter(|(_, berth, _)| *berth == seed_berth)
                    .collect::<Vec<_>>()
            }
        });

        if !candidate_pool.iter().any(|(ri, _, _)| *ri == seed_request) {
            candidate_pool.push((seed_request, seed_berth, seed_interval));
        }

        // Sort by |Δstart| to seed (closest-in-time first).
        candidate_pool.sort_by(|a, b| {
            let da = {
                let dt = if a.2.start() >= seed_interval.start() {
                    (a.2.start() - seed_interval.start()).value()
                } else {
                    (seed_interval.start() - a.2.start()).value()
                };
                Into::<Cost>::into(dt)
            };
            let db = {
                let dt = if b.2.start() >= seed_interval.start() {
                    (b.2.start() - seed_interval.start()).value()
                } else {
                    (seed_interval.start() - b.2.start()).value()
                };
                Into::<Cost>::into(dt)
            };
            da.cmp(&db)
        });

        let selected_for_removal: Vec<RequestIndex> = candidate_pool
            .into_iter()
            .map(|t| t.0)
            .take(target_removal_count)
            .collect();

        for request_index in selected_for_removal {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

// -------------------------
// TimeWindowBandDestroy
// -------------------------

#[derive(Clone)]
pub struct TimeWindowBandDestroy<T> {
    /// fraction of currently assigned to remove (0,1]; clamped to [1, |pool|]
    pub ratio_range: RatioRange,
    /// Shaw-style randomized greediness for picking the seed among longer rectangles
    pub alpha_for_seed_range: AlphaRange,
    /// Half-width of the time band to remove around the seed's [start,end]
    pub half_band: berth_alloc_core::prelude::TimeDelta<T>,
    /// Optional neighborhood restriction
    pub neighbors: Option<Arc<NeighborFn>>,
}

impl<T> TimeWindowBandDestroy<T> {
    pub fn new(
        ratio_range: RatioRange,
        alpha_for_seed_range: AlphaRange,
        half_band: berth_alloc_core::prelude::TimeDelta<T>,
    ) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        assert!(*alpha_for_seed_range.start() > 0.0 && alpha_for_seed_range.end().is_finite());
        Self {
            ratio_range,
            alpha_for_seed_range,
            half_band,
            neighbors: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbors = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for TimeWindowBandDestroy<T>
where
    T: Copy
        + Ord
        + CheckedAdd
        + CheckedSub
        + Into<Cost>
        + std::ops::Mul<Output = Cost>
        + Send
        + Sync,
    R: Rng,
{
    fn name(&self) -> &str {
        "TimeWindowBandDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        let sampled_alpha = sample_f64_inclusive(&self.alpha_for_seed_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        let full = plan_builder.with_explorer(|explorer| {
            let mut v = triples_from_explorer(explorer, model);
            v.sort_by(|a, b| {
                let la = (a.2.end() - a.2.start()).value();
                let lb = (b.2.end() - b.2.start()).value();
                la.cmp(&lb)
            });
            v
        });
        if full.is_empty() {
            return None;
        }

        let target_count = ((full.len() as f64) * sampled_ratio).ceil() as usize;

        let seed_idx = randomized_greedy_index(full.len(), sampled_alpha, rng);
        let (seed_req, seed_berth, seed_iv) = full[seed_idx];

        // Candidate pool = neighbor(seed) if provided/non-empty else full
        let mut pool = plan_builder.with_explorer(|explorer| {
            if let Some(cb) = &self.neighbors {
                let loc =
                    neighborhood_triples_from_explorer(explorer, model, seed_req, cb.as_ref());
                if loc.is_empty() { full.clone() } else { loc }
            } else {
                full.clone()
            }
        });

        // Ensure seed triple is in the pool:
        if !pool.iter().any(|(ri, _, _)| *ri == seed_req) {
            pool.push((seed_req, seed_berth, seed_iv));
        }

        // Compute the time band around the seed:
        let band_start = seed_iv.start() - self.half_band;
        let band_end = seed_iv.end() + self.half_band;

        // Keep triples whose interval intersects the band:
        let mut removed: Vec<RequestIndex> = pool
            .into_iter()
            .filter(|(_, _, iv)| !(iv.end() <= band_start || iv.start() >= band_end))
            .map(|t| t.0)
            .collect();

        // Truncate or top-up to the target
        removed.truncate(target_count.max(1));
        if removed.is_empty() {
            removed.push(seed_req);
        }

        for request_index in removed {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

// -------------------------
// BerthBandDestroy
// -------------------------
#[derive(Clone)]
pub struct BerthBandDestroy {
    /// fraction of assigned to remove (0,1]; clamped
    pub ratio_range: RatioRange,
    /// seed greediness among longer rectangles
    pub alpha_for_seed_range: AlphaRange,
    /// half-width in berth-index units (inclusive)
    pub half_berth_span: u32,
    /// optional neighbor restriction
    pub neighbors: Option<Arc<NeighborFn>>,
}

impl BerthBandDestroy {
    pub fn new(
        ratio_range: RatioRange,
        alpha_for_seed_range: AlphaRange,
        half_berth_span: u32,
    ) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        assert!(*alpha_for_seed_range.start() > 0.0 && alpha_for_seed_range.end().is_finite());
        Self {
            ratio_range,
            alpha_for_seed_range,
            half_berth_span,
            neighbors: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbors = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for BerthBandDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "BerthBandDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        let sampled_alpha = sample_f64_inclusive(&self.alpha_for_seed_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        let full = plan_builder.with_explorer(|explorer| {
            let mut v = triples_from_explorer(explorer, model);
            v.sort_by(|a, b| {
                let la = (a.2.end() - a.2.start()).value();
                let lb = (b.2.end() - b.2.start()).value();
                la.cmp(&lb)
            });
            v
        });
        if full.is_empty() {
            return None;
        }

        let target_count = ((full.len() as f64) * sampled_ratio).ceil() as usize;

        let seed_idx = randomized_greedy_index(full.len(), sampled_alpha, rng);
        let (seed_req, seed_berth, seed_iv) = full[seed_idx];

        // Candidate pool: neighbors(seed) else same-berth
        let mut pool = plan_builder.with_explorer(|explorer| {
            if let Some(cb) = &self.neighbors {
                let loc =
                    neighborhood_triples_from_explorer(explorer, model, seed_req, cb.as_ref());
                if loc.is_empty() { full.clone() } else { loc }
            } else {
                full.clone()
                    .into_iter()
                    .filter(|(_, bi, _)| *bi == seed_berth)
                    .collect::<Vec<_>>()
            }
        });

        if !pool.iter().any(|(ri, _, _)| *ri == seed_req) {
            pool.push((seed_req, seed_berth, seed_iv));
        }

        // Same-berth first, closest |Δstart| to seed
        pool.sort_by(|a, b| {
            let da = {
                let dt = if a.2.start() >= seed_iv.start() {
                    (a.2.start() - seed_iv.start()).value()
                } else {
                    (seed_iv.start() - a.2.start()).value()
                };
                Into::<Cost>::into(dt)
            };
            let db = {
                let dt = if b.2.start() >= seed_iv.start() {
                    (b.2.start() - seed_iv.start()).value()
                } else {
                    (seed_iv.start() - b.2.start()).value()
                };
                Into::<Cost>::into(dt)
            };
            da.cmp(&db)
        });

        let removed: Vec<RequestIndex> = pool
            .into_iter()
            .map(|t| t.0)
            .take(target_count.max(1))
            .collect();

        for request_index in removed {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

// -------------------------
// ProcessingTimeClusterDestroy
// -------------------------
#[derive(Clone)]
pub struct ProcessingTimeClusterDestroy {
    /// fraction of assigned to remove (0,1]; clamped
    pub ratio_range: RatioRange,
    /// seed greediness among longer rectangles
    pub alpha_for_seed_range: AlphaRange,
    /// optional neighbor restriction
    pub neighbors: Option<Arc<NeighborFn>>,
}

impl ProcessingTimeClusterDestroy {
    pub fn new(ratio_range: RatioRange, alpha_for_seed_range: AlphaRange) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        assert!(*alpha_for_seed_range.start() > 0.0 && alpha_for_seed_range.end().is_finite());
        Self {
            ratio_range,
            alpha_for_seed_range,
            neighbors: None,
        }
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbors = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for ProcessingTimeClusterDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "ProcessingTimeClusterDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        _rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let sampled_ratio = *self.ratio_range.start(); // deterministic if caller wants; or sample externally
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        let full = plan_builder.with_explorer(|explorer| {
            let mut v = triples_from_explorer(explorer, model);
            v.sort_by(|a, b| {
                let la = (a.2.end() - a.2.start()).value();
                let lb = (b.2.end() - b.2.start()).value();
                la.cmp(&lb)
            });
            v
        });
        if full.is_empty() {
            return None;
        }

        let target_count = ((full.len() as f64) * sampled_ratio).ceil() as usize;

        let (seed_ri, seed_bi, seed_iv) = full[0]; // longest rectangle bias already applied above
        let seed_pt_val = (seed_iv.end() - seed_iv.start()).value();

        // Candidate pool: neighbors(seed) if any; else full
        let mut pool = full.clone();
        if let Some(cb) = &self.neighbors {
            pool = plan_builder.with_explorer(|explorer| {
                let loc = neighborhood_triples_from_explorer(explorer, model, seed_ri, cb.as_ref());
                if loc.is_empty() { full.clone() } else { loc }
            });
        }
        if !pool.iter().any(|(ri, _, _)| *ri == seed_ri) {
            pool.push((seed_ri, seed_bi, seed_iv));
        }

        // Sort by |PT - seed_PT| ascending
        pool.sort_by(|a, b| {
            let pa = (a.2.end() - a.2.start()).value();
            let pb = (b.2.end() - b.2.start()).value();
            let da = if pa >= seed_pt_val {
                pa - seed_pt_val
            } else {
                seed_pt_val - pa
            };
            let db = if pb >= seed_pt_val {
                pb - seed_pt_val
            } else {
                seed_pt_val - pb
            };
            da.cmp(&db)
        });

        let removed: Vec<RequestIndex> = pool
            .into_iter()
            .map(|t| t.0)
            .take(target_count.max(1))
            .collect();

        for request_index in removed {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

/// Remove up to `ratio` of *assigned* requests whose [start,end) overlaps
/// a time band centered at a random pivot start time.
///
/// This is a classic ALNS "time-oriented removal": it clears a temporal zone
/// to let repair re-pack many jobs tightly.
#[derive(Clone)]
pub struct TimeClusterDestroy<T> {
    ratio_range: RatioRange,          // (0,1]
    half_window: TimeDelta<T>,        // band half-width
    alpha_for_seed_range: AlphaRange, // RCL greediness for seed among longer rectangles
    neighbors: Option<Arc<NeighborFn>>,
}

impl<T> TimeClusterDestroy<T> {
    pub fn new(ratio_range: RatioRange, half_window: TimeDelta<T>) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        Self {
            ratio_range,
            half_window,
            alpha_for_seed_range: 1.7..=1.7,
            neighbors: None,
        }
    }
    pub fn with_alpha(mut self, alpha_range: AlphaRange) -> Self {
        assert!(*alpha_range.start() > 0.0 && alpha_range.end().is_finite());
        self.alpha_for_seed_range = alpha_range;
        self
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbors = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for TimeClusterDestroy<T>
where
    T: Copy
        + Ord
        + CheckedAdd
        + CheckedSub
        + Into<Cost>
        + std::ops::Mul<Output = Cost>
        + Send
        + Sync,
    R: Rng,
{
    fn name(&self) -> &str {
        "TimeClusterDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        let sampled_alpha = sample_f64_inclusive(&self.alpha_for_seed_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        // Collect assigned triples (ri, bi, [start,end))
        let assigned: Vec<(RequestIndex, BerthIndex, TimeInterval<T>)> = plan_builder
            .with_explorer(|explorer| {
                explorer
                    .decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            let ri = RequestIndex::new(i);
                            model.processing_time(ri, berth_index).map(|pt| {
                                (
                                    ri,
                                    berth_index,
                                    TimeInterval::new(start_time, start_time + pt),
                                )
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            });

        if assigned.is_empty() {
            return None;
        }

        // Prefer longer rectangles as seed (they’re good pivots); RCL pick.
        let mut sorted = assigned.clone();
        sorted.sort_by_key(|(_, _, iv)| (iv.end() - iv.start()).value());
        let seed_index = randomized_greedy_index(sorted.len(), sampled_alpha, rng);
        let (seed_request, _, seed_interval) = sorted[seed_index];

        // Choose pool = neighbors(seed) if provided/non-empty; else full.
        let pool: Vec<(RequestIndex, BerthIndex, TimeInterval<T>)> =
            plan_builder.with_explorer(|explorer| {
                if let Some(cb) = &self.neighbors {
                    let neighborhood = {
                        let dsv = explorer.decision_vars();
                        cb(seed_request)
                            .into_iter()
                            .filter_map(|ri| {
                                if let DecisionVar::Assigned(Decision {
                                    berth_index,
                                    start_time,
                                }) = dsv[ri.get()]
                                {
                                    model.processing_time(ri, berth_index).map(|pt| {
                                        (
                                            ri,
                                            berth_index,
                                            TimeInterval::new(start_time, start_time + pt),
                                        )
                                    })
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    if neighborhood.is_empty() {
                        assigned.clone()
                    } else {
                        neighborhood
                    }
                } else {
                    assigned.clone()
                }
            });

        if pool.is_empty() {
            return None;
        }

        // Band around the seed's midpoint (or just seed start); we use seed interval center.
        let mid = {
            let half = seed_interval.end() - seed_interval.start();
            seed_interval.start() + half
        };
        let band_lo = mid - self.half_window;
        let band_hi = mid + self.half_window;

        let mut removed_count = 0usize;
        let target_count =
            (((pool.len() as f64) * sampled_ratio).ceil() as usize).clamp(1, pool.len());

        for (ri, _bi, iv) in pool.into_iter() {
            if removed_count >= target_count {
                break;
            }
            let outside = iv.end() <= band_lo || iv.start() >= band_hi;
            if !outside {
                // builder handles occupancy + accounting
                let _ = plan_builder.propose_unassignment(ri);
                removed_count += 1;
            }
        }

        let plan = plan_builder.finalize();
        if plan.decision_var_patches.is_empty() {
            None
        } else {
            Some(plan)
        }
    }
}

/// Remove a **contiguous block** of assignments centered on a seed, restricted
/// to the seed's berth. Great for "opening a long gap" on one resource to let
/// regret/greedy repair rebuild a better pack.
///
/// `ratio` controls the block length relative to the number of assigned on that berth.
/// Both ratio and alpha are sampled from ranges.
#[derive(Clone)]
pub struct StringBlockDestroy {
    ratio_range: RatioRange,
    alpha_for_seed_range: AlphaRange,
    neighbors: Option<Arc<NeighborFn>>,
}

impl StringBlockDestroy {
    pub fn new(ratio_range: RatioRange) -> Self {
        assert!(*ratio_range.start() >= 0.0 && *ratio_range.end() <= 1.0);
        Self {
            ratio_range,
            alpha_for_seed_range: 1.7..=1.7,
            neighbors: None,
        }
    }
    pub fn with_alpha(mut self, alpha_range: AlphaRange) -> Self {
        assert!(*alpha_range.start() > 0.0 && alpha_range.end().is_finite());
        self.alpha_for_seed_range = alpha_range;
        self
    }
    pub fn with_neighbors(mut self, callback: Arc<NeighborFn>) -> Self {
        self.neighbors = Some(callback);
        self
    }
}

impl<T, R> DestroyOperator<T, R> for StringBlockDestroy
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + std::ops::Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "StringBlockDestroy"
    }

    fn propose<'b, 's, 'm, 'p>(
        &self,
        ctx: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let sampled_ratio = sample_f64_inclusive(&self.ratio_range, rng);
        let sampled_alpha = sample_f64_inclusive(&self.alpha_for_seed_range, rng);
        if sampled_ratio == 0.0 {
            return None;
        }

        let model = ctx.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = ctx.builder();

        // Collect (ri, bi, start, end) for currently assigned.
        let assigned: Vec<(RequestIndex, BerthIndex, TimeInterval<T>)> = plan_builder
            .with_explorer(|explorer| {
                explorer
                    .decision_vars()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dv)| {
                        if let DecisionVar::Assigned(Decision {
                            berth_index,
                            start_time,
                        }) = *dv
                        {
                            let ri = RequestIndex::new(i);
                            model.processing_time(ri, berth_index).map(|pt| {
                                (
                                    ri,
                                    berth_index,
                                    TimeInterval::new(start_time, start_time + pt),
                                )
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            });

        if assigned.is_empty() {
            return None;
        }

        // Prefer longer rectangles for the seed; pick via RCL.
        let mut sorted = assigned.clone();
        sorted.sort_by_key(|(_, _, iv)| (iv.end() - iv.start()).value());
        let seed_idx = randomized_greedy_index(sorted.len(), sampled_alpha, rng);
        let (seed_request, seed_berth, _) = sorted[seed_idx];

        // Limit pool to same berth as seed (or a neighbor fn if provided & non-empty).
        let mut same_berth: Vec<(RequestIndex, TimeInterval<T>)> =
            plan_builder.with_explorer(|explorer| {
                if let Some(cb) = &self.neighbors {
                    let dsv = explorer.decision_vars();
                    let loc: Vec<(RequestIndex, TimeInterval<T>)> = cb(seed_request)
                        .into_iter()
                        .filter_map(|ri| {
                            if let DecisionVar::Assigned(Decision {
                                berth_index,
                                start_time,
                            }) = dsv[ri.get()]
                                && berth_index == seed_berth
                            {
                                model
                                    .processing_time(ri, berth_index)
                                    .map(|pt| (ri, TimeInterval::new(start_time, start_time + pt)))
                            } else {
                                None
                            }
                        })
                        .collect();

                    if loc.is_empty() {
                        assigned
                            .iter()
                            .filter(|(_, bi, _)| *bi == seed_berth)
                            .map(|(ri, _, iv)| (*ri, *iv))
                            .collect()
                    } else {
                        loc
                    }
                } else {
                    assigned
                        .iter()
                        .filter(|(_, bi, _)| *bi == seed_berth)
                        .map(|(ri, _, iv)| (*ri, *iv))
                        .collect()
                }
            });

        if same_berth.is_empty() {
            return None;
        }

        // Order by start time ascending to define the “string”
        same_berth.sort_by_key(|(_, iv)| iv.start().value());

        // Locate seed position
        let seed_pos = same_berth
            .iter()
            .position(|(ri, _)| *ri == seed_request)
            .unwrap_or(0);

        // Block length target
        let target_count = (((same_berth.len() as f64) * sampled_ratio).ceil() as usize)
            .clamp(1, same_berth.len());

        // Center block around seed (shift if needed to stay within bounds)
        let half = target_count / 2;
        let mut start = seed_pos.saturating_sub(half);
        if start + target_count > same_berth.len() {
            start = same_berth.len() - target_count;
        }

        let block = &same_berth[start..(start + target_count)];
        for (request_index, _iv) in block.iter().copied() {
            let _ = plan_builder.propose_unassignment(request_index);
        }

        let plan = plan_builder.finalize();
        if plan.decision_var_patches.is_empty() {
            None
        } else {
            Some(plan)
        }
    }
}

// ======================================================================
// Tests (updated for range-based API: use x..=x for determinism)
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::planner::PlanningContext,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            solver_state::SolverState,
            terminal::terminalocc::{TerminalOccupancy, TerminalRead},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};
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

    fn make_problem(n_requests: usize) -> Problem<i64> {
        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();
        builder.add_berth(berth(1, 0, 1000));
        for i in 1..=n_requests {
            builder.add_flexible(flex_req(i as u32, (0, 200), &[(1, 10)], 1));
        }
        builder.build().expect("valid problem")
    }

    fn make_assigned_state<'p>(
        model: &SolverModel<'p, i64>,
        k_assigned: usize,
    ) -> SolverState<'p, i64> {
        let mut dvars = Vec::with_capacity(model.flexible_requests_len());
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        for i in 0..model.flexible_requests_len() {
            if i < k_assigned {
                dvars.push(DecisionVar::assigned(b_ix, tp(0)));
            } else {
                dvars.push(DecisionVar::unassigned());
            }
        }
        let dv = DecisionVarVec::from(dvars);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let fit =
            crate::state::fitness::Fitness::new(0, model.flexible_requests_len() - k_assigned);
        SolverState::new(dv, term, fit)
    }

    fn make_ctx<'b, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        state: &'s SolverState<'p, i64>,
        buffer: &'b mut [DecisionVar<i64>],
    ) -> PlanningContext<'b, 's, 'm, 'p, i64> {
        PlanningContext::new(model, state, buffer)
    }

    // ------------- helper tests

    #[test]
    fn test_is_zero_delta_plan_true_and_false() {
        let prob = make_problem(2);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 0);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        // Plan with no edits → zero-delta (via builder)
        let pb = ctx.builder();
        let nop_plan = pb.finalize();
        assert!(is_zero_delta_plan(&nop_plan));

        // Plan with an edit → non-zero-delta
        let mut pb = ctx.builder();
        let r_ix = model.index_manager().request_index(rid(1)).unwrap();
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let free = pb
            .sandbox()
            .inner()
            .iter_free_intervals_for_berths_in([b_ix], model.feasible_interval(r_ix))
            .next()
            .unwrap();
        pb.propose_assignment(r_ix, tp(0), &free).unwrap();
        let plan_with_edit = pb.finalize();
        assert!(!is_zero_delta_plan(&plan_with_edit));
    }

    #[test]
    fn test_randomized_greedy_index_bounds() {
        let mut rng = StdRng::seed_from_u64(123);
        for len in [1_usize, 2, 5, 10] {
            for &alpha in &[1.0, 2.0, 5.0] {
                for _ in 0..100 {
                    let idx = randomized_greedy_index(len, alpha, &mut rng);
                    assert!(idx < len, "idx {} not in [0, {})", idx, len);
                }
            }
        }
    }

    // ------------- operator tests

    #[test]
    fn test_random_k_ratio_destroy_none_when_no_assigned() {
        let prob = make_problem(3);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 0);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(42);

        let op = RandomKRatioDestroy::new(1.0..=1.0);
        assert!(op.propose(&mut ctx, &mut rng).is_none());
    }

    #[test]
    fn test_random_k_ratio_destroy_zero_ratio_is_none() {
        let prob = make_problem(4);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 3);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(7);

        let op = RandomKRatioDestroy::new(0.0..=0.0);
        assert!(op.propose(&mut ctx, &mut rng).is_none());
    }

    #[test]
    fn test_random_k_ratio_destroy_one_removes_all_assigned() {
        let prob = make_problem(5);
        let model = SolverModel::try_from(&prob).unwrap();
        let k_assigned = 3;
        let state = make_assigned_state(&model, k_assigned);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(123);

        let op = RandomKRatioDestroy::new(1.0..=1.0);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");

        assert_eq!(plan.decision_var_patches.len(), k_assigned);
        assert_eq!(plan.delta_unassigned as usize, k_assigned);
        assert!(plan.delta_cost < Cost::zero());
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn test_random_k_ratio_destroy_respects_neighbor_callback_topup() {
        let prob = make_problem(6);
        let model = SolverModel::try_from(&prob).unwrap();
        // assign 6
        let state = make_assigned_state(&model, 6);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(55);

        // Neighbor callback that returns only itself (very tiny local pool) → operator will top up.
        let cb: Arc<NeighborFn> = Arc::new(|seed| vec![seed]);

        let ratio = 0.5..=0.5; // remove ceil(6 * 0.5) = 3
        let op = RandomKRatioDestroy::new(ratio).with_neighbors(cb);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");
        assert_eq!(plan.decision_var_patches.len(), 3);
    }

    #[test]
    fn test_worst_cost_destroy_sorts_by_cost() {
        let prob = make_problem(5);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 5);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(1);

        let op = WorstCostDestroy::new(0.4..=0.4);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");
        // removes ceil(5 * 0.4) = 2
        assert_eq!(plan.decision_var_patches.len(), 2);
        assert_eq!(plan.delta_unassigned, 2);
    }

    #[test]
    fn test_shaw_related_destroy_basic() {
        let prob = make_problem(7);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 7);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(9);

        let op = ShawRelatedDestroy::new(
            0.3..=0.3, // ceil(7*0.3) = 3
            2.0..=2.0,
            1.into(),
            1.into(),
            5.into(),
        );
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");
        assert_eq!(plan.decision_var_patches.len(), 3);
        assert_eq!(plan.delta_unassigned, 3);
    }

    #[test]
    fn test_berth_neighbors_destroy_without_callback_uses_same_berth() {
        let prob = make_problem(6);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 6);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(77);

        let op = BerthNeighborsDestroy::new(0.5..=0.5, 2.0..=2.0); // ceil(6*0.5)=3
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");
        assert_eq!(plan.decision_var_patches.len(), 3);
    }

    #[test]
    fn test_neighbor_callback_variants_change_pool() {
        let prob = make_problem(6);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 6);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(101);

        // Return two neighbors (seed±1 if valid).
        let cb: Arc<NeighborFn> = Arc::new(|seed| {
            let i = seed.get();
            let mut v = Vec::new();
            if i > 0 {
                v.push(RequestIndex::new(i - 1));
            }
            v.push(seed);
            v.push(RequestIndex::new(i + 1));
            v
        });

        let op = WorstCostDestroy::new(0.5..=0.5).with_neighbors(cb);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan expected");
        assert_eq!(plan.delta_unassigned, 3); // ceil(6*0.5)
    }

    #[test]
    fn time_window_band_destroy_removes_band() {
        let prob = make_problem(8);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 6);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let mut rng = StdRng::seed_from_u64(1);
        let op = TimeWindowBandDestroy::new(0.5..=0.5, 1.7..=1.7, TimeDelta::new(5));
        let plan = op.propose(&mut ctx, &mut rng).expect("plan");
        let expected = (((6_usize) as f64) * 0.5_f64).ceil() as usize;
        assert_eq!(plan.decision_var_patches.len(), expected);
        assert!(plan.delta_unassigned > 0);
    }

    #[test]
    fn berth_band_destroy_respects_span_and_ratio() {
        let prob = make_problem(10);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 7);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let mut rng = StdRng::seed_from_u64(2);
        let op = BerthBandDestroy::new(0.4..=0.4, 1.3..=1.3, 0); // same-berth only
        let plan = op.propose(&mut ctx, &mut rng).expect("plan");
        let expected = (((7_usize) as f64) * 0.4_f64).ceil() as usize;
        assert_eq!(plan.decision_var_patches.len(), expected);
    }

    #[test]
    fn processing_time_cluster_destroy_clusters_by_pt() {
        let prob = make_problem(9);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 9);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        // deterministic via degenerate range
        let mut rng = StdRng::seed_from_u64(3);
        let op = ProcessingTimeClusterDestroy::new(0.33..=0.33, 2.0..=2.0);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan");
        let expected = (((9_usize) as f64) * 0.33_f64).ceil() as usize;
        assert_eq!(plan.decision_var_patches.len(), expected);
    }

    #[test]
    fn time_window_band_destroy_uses_neighbor_callback_when_nonempty() {
        use std::sync::Arc;
        let prob = make_problem(6);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 6);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        // NeighborFn: each request neighbors the next 2 indices (wrapping disabled)
        let cb: Arc<NeighborFn> = Arc::new(|ri: RequestIndex| {
            let i = ri.get();
            let mut v = Vec::new();
            if i + 1 < 6 {
                v.push(RequestIndex::new(i + 1));
            }
            if i + 2 < 6 {
                v.push(RequestIndex::new(i + 2));
            }
            v
        });

        let mut rng = StdRng::seed_from_u64(4);
        let op =
            TimeWindowBandDestroy::new(0.5..=0.5, 1.2..=1.2, TimeDelta::new(5)).with_neighbors(cb);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan");

        // Pool restricted by callback should still reach the requested ratio (topped-up if needed)
        let expected = (((6_usize) as f64) * 0.5_f64).ceil() as usize;
        assert_eq!(plan.decision_var_patches.len(), expected);
    }

    #[test]
    fn time_cluster_destroy_with_ranged_alpha_and_ratio() {
        let prob = make_problem(8);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 8);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(11);

        let op = TimeClusterDestroy::new(0.25..=0.25, TimeDelta::new(5)).with_alpha(1.7..=1.7);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan");
        let expected = (((8_usize) as f64) * 0.25_f64).ceil() as usize;
        assert_eq!(plan.decision_var_patches.len(), expected);
    }

    #[test]
    fn string_block_destroy_with_ranges() {
        let prob = make_problem(10);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_assigned_state(&model, 10);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(22);

        let op = StringBlockDestroy::new(0.3..=0.3).with_alpha(2.0..=2.0);
        let plan = op.propose(&mut ctx, &mut rng).expect("plan");
        let expected = (((10_usize) as f64) * 0.3_f64).ceil() as usize;
        assert_eq!(plan.decision_var_patches.len(), expected);
    }
}
