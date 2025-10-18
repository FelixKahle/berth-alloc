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
    model::index::RequestIndex,
    search::{
        operator::RepairOperator,
        planner::{PlanBuilder, PlanExplorer, PlanningContext},
    },
    state::{plan::Plan, terminal::terminalocc::FreeBerth},
};
use berth_alloc_core::prelude::{Cost, TimePoint};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::Rng;
use std::{
    cmp::Ordering,
    ops::{Mul, RangeInclusive},
};

// Type aliases to make intent explicit
type AlphaRange = RangeInclusive<f64>;
type KRange = RangeInclusive<usize>;

/// Returns `true` when a `Plan` contains no changes (no patches, no terminal delta, zero deltas).
#[inline]
fn is_zero_delta_plan<T>(plan: &Plan<'_, T>) -> bool
where
    T: Copy + Ord,
{
    plan.delta_unassigned == 0 && plan.delta_cost == Cost::zero() && plan.terminal_delta.is_empty()
}

/// Randomized–greedy index selector used by RCL-style insertion.
/// For `len > 0`, returns an index in `[0, len-1]`.
/// `greediness_alpha = 1.0` ≈ uniform; larger values bias toward lower indices.
#[inline]
fn randomized_greedy_index<R: Rng>(len: usize, greediness_alpha: f64, rng: &mut R) -> usize {
    debug_assert!(len > 0, "randomized_greedy_index called with len=0");
    let random_unit: f64 = rng.random_range(0.0..1.0);
    let raw_index = ((len as f64) * random_unit.powf(greediness_alpha)).ceil() as usize;
    raw_index.saturating_sub(1).min(len - 1)
}

/// Count coarse feasible insertion “slots” for request `request_index` using the explorer:
/// number of free intervals that can host at least one feasible start.
#[inline]
fn count_feasible_positions_ex<T>(
    plan_explorer: &PlanExplorer<'_, '_, '_, '_, T>,
    model: &crate::model::solver_model::SolverModel<'_, T>,
    request_index: RequestIndex,
) -> usize
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    plan_explorer
        .iter_free_for(request_index)
        .filter(|free_window_for_berth| {
            if let Some(processing_time) =
                model.processing_time(request_index, free_window_for_berth.berth_index())
            {
                let free_interval = free_window_for_berth.interval();
                (free_interval.start() + processing_time) <= free_interval.end()
            } else {
                false
            }
        })
        .count()
}

/// Best (lowest-cost) insertion for `request_index` right now via explorer.
/// Returns `(FreeBerth, start, cost)` for the *earliest* feasible start in each free interval.
#[inline]
fn best_insertion_for_request_ex<T>(
    plan_explorer: &PlanExplorer<'_, '_, '_, '_, T>,
    model: &crate::model::solver_model::SolverModel<'_, T>,
    request_index: RequestIndex,
) -> Option<(FreeBerth<T>, TimePoint<T>, Cost)>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
{
    let mut best_triplet: Option<(Cost, FreeBerth<T>, TimePoint<T>)> = None;

    for free_window_for_berth in plan_explorer.iter_free_for(request_index) {
        let berth_index = free_window_for_berth.berth_index();
        let Some(processing_time) = model.processing_time(request_index, berth_index) else {
            continue;
        };

        let free_interval = free_window_for_berth.interval();
        let earliest_start = free_interval.start();
        if earliest_start + processing_time > free_interval.end() {
            continue;
        }

        if let Some(cost_if_inserted) =
            plan_explorer.peek_cost(request_index, earliest_start, berth_index)
        {
            match best_triplet {
                None => {
                    best_triplet = Some((cost_if_inserted, free_window_for_berth, earliest_start))
                }
                Some((best_cost, _, _)) if cost_if_inserted < best_cost => {
                    best_triplet = Some((cost_if_inserted, free_window_for_berth, earliest_start))
                }
                _ => {}
            }
        }
    }

    best_triplet.map(|(c, fb, s)| (fb, s, c))
}

// ======================================================================
// RandomizedGreedyInsertion (now with alpha range)
// ======================================================================

#[derive(Clone, Debug)]
pub struct RandomizedGreedyInsertion {
    /// Randomized greediness exponent range (α ≥ 1.0). α=1 → uniform; larger → greedier.
    pub greediness_alpha_range: AlphaRange,
}

impl RandomizedGreedyInsertion {
    pub fn new(greediness_alpha_range: AlphaRange) -> Self {
        assert!(greediness_alpha_range.start().is_finite());
        assert!(greediness_alpha_range.end().is_finite());
        assert!(*greediness_alpha_range.start() >= 1.0);
        Self {
            greediness_alpha_range,
        }
    }
}

impl<T, R> RepairOperator<T, R> for RandomizedGreedyInsertion
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "RandomizedGreedyInsertion"
    }

    fn repair<'b, 's, 'm, 'p>(
        &self,
        planning_context: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let solver_model = planning_context.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = planning_context.builder();

        // Sample α once for this repair phase to keep behavior coherent within a step.
        let sampled_greediness_alpha: f64 = rng.random_range(self.greediness_alpha_range.clone());

        loop {
            let mut unassigned_request_indices: Vec<RequestIndex> = plan_builder
                .with_explorer(|plan_explorer| plan_explorer.iter_unassigned().collect());
            if unassigned_request_indices.is_empty() {
                break;
            }

            // Sort by “most constrained” first, then by best cost, then by request index (stable).
            plan_builder.with_explorer(|plan_explorer| {
                unassigned_request_indices.sort_by(|&ra, &rb| {
                    let feasible_a = count_feasible_positions_ex(plan_explorer, solver_model, ra);
                    let feasible_b = count_feasible_positions_ex(plan_explorer, solver_model, rb);
                    match feasible_a.cmp(&feasible_b) {
                        Ordering::Less => Ordering::Less,
                        Ordering::Greater => Ordering::Greater,
                        Ordering::Equal => {
                            let best_cost_a =
                                best_insertion_for_request_ex(plan_explorer, solver_model, ra)
                                    .map(|(_, _, c)| c);
                            let best_cost_b =
                                best_insertion_for_request_ex(plan_explorer, solver_model, rb)
                                    .map(|(_, _, c)| c);
                            match (best_cost_a, best_cost_b) {
                                (Some(x), Some(y)) => match x.cmp(&y) {
                                    Ordering::Equal => ra.get().cmp(&rb.get()),
                                    ord => ord,
                                },
                                (Some(_), None) => Ordering::Less, // feasible beats infeasible
                                (None, Some(_)) => Ordering::Greater,
                                (None, None) => ra.get().cmp(&rb.get()),
                            }
                        }
                    }
                });
            });

            let chosen_index = randomized_greedy_index(
                unassigned_request_indices.len(),
                sampled_greediness_alpha,
                rng,
            );
            let request_index = unassigned_request_indices.remove(chosen_index);

            if let Some((best_free_window, best_start_time, _best_cost)) = plan_builder
                .with_explorer(|plan_explorer| {
                    best_insertion_for_request_ex(plan_explorer, solver_model, request_index)
                })
            {
                let _ = plan_builder.propose_assignment(
                    request_index,
                    best_start_time,
                    &best_free_window,
                );
            }
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

// ======================================================================
// KRegretInsertion (now with K range)
// ======================================================================

#[derive(Clone, Debug)]
pub struct KRegretInsertion {
    /// Number of candidate insertions to consider for regret computation.
    pub k_choice_range: KRange,
}

impl KRegretInsertion {
    pub fn new(k_choice_range: KRange) -> Self {
        assert!(*k_choice_range.start() >= 1);
        Self { k_choice_range }
    }
}

impl<T, R> RepairOperator<T, R> for KRegretInsertion
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "KRegretInsertion"
    }

    fn repair<'b, 's, 'm, 'p>(
        &self,
        planning_context: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let solver_model = planning_context.model();
        let mut plan_builder: PlanBuilder<'_, 's, 'm, 'p, T> = planning_context.builder();

        // Sample k once for this repair phase.
        let sampled_k: usize = rng.random_range(self.k_choice_range.clone());

        loop {
            let unassigned_request_indices: Vec<RequestIndex> = plan_builder
                .with_explorer(|plan_explorer| plan_explorer.iter_unassigned().collect());
            if unassigned_request_indices.is_empty() {
                break;
            }

            #[allow(clippy::type_complexity)]
            let regret_table: Vec<(
                RequestIndex,
                Vec<(Cost, FreeBerth<T>, TimePoint<T>)>,
            )> = plan_builder.with_explorer(|plan_explorer| {
                #[allow(clippy::type_complexity)]
                let mut local_table: Vec<(
                    RequestIndex,
                    Vec<(Cost, FreeBerth<T>, TimePoint<T>)>,
                )> = Vec::with_capacity(unassigned_request_indices.len());

                for &request_index in &unassigned_request_indices {
                    let mut candidate_insertions: Vec<(Cost, FreeBerth<T>, TimePoint<T>)> =
                        Vec::new();

                    for free_window_for_berth in plan_explorer.iter_free_for(request_index) {
                        let berth_index = free_window_for_berth.berth_index();
                        let Some(processing_time) =
                            solver_model.processing_time(request_index, berth_index)
                        else {
                            continue;
                        };

                        let free_interval = free_window_for_berth.interval();
                        let earliest_start = free_interval.start();
                        if earliest_start + processing_time > free_interval.end() {
                            continue;
                        }
                        if let Some(cost_if_inserted) =
                            plan_explorer.peek_cost(request_index, earliest_start, berth_index)
                        {
                            candidate_insertions.push((
                                cost_if_inserted,
                                free_window_for_berth,
                                earliest_start,
                            ));
                        }
                    }

                    candidate_insertions.sort_by(|a, b| a.0.cmp(&b.0));
                    if candidate_insertions.len() > sampled_k {
                        candidate_insertions.truncate(sampled_k);
                    }
                    if !candidate_insertions.is_empty() {
                        local_table.push((request_index, candidate_insertions));
                    }
                }
                local_table
            });

            if regret_table.is_empty() {
                break;
            }

            // Pick the request with maximal k-regret (best_kth - best_1st).
            let mut best_row_index = 0usize;
            let mut best_regret_value = Cost::zero();
            for (row_idx, (_, options)) in regret_table.iter().enumerate() {
                let best_cost = options[0].0;
                let kth_cost = if options.len() >= sampled_k {
                    options[sampled_k - 1].0
                } else {
                    options[options.len().saturating_sub(1)].0
                };
                let regret_value = kth_cost.saturating_sub(best_cost);
                if regret_value > best_regret_value {
                    best_regret_value = regret_value;
                    best_row_index = row_idx;
                }
            }

            let (chosen_request_index, options_for_request) = &regret_table[best_row_index];
            let (_best_cost, best_free_window, best_start_time) = &options_for_request[0];
            let _ = plan_builder.propose_assignment(
                *chosen_request_index,
                *best_start_time,
                best_free_window,
            );
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

// ======================================================================
// GreedyInsertion (unchanged API)
// ======================================================================

/// At each step, finds the single best insertion (lowest cost) across all
/// unassigned requests and inserts it. Repeats until no more insertions are possible.
/// Tie-break: lower cost → earlier start → lower request index.
#[derive(Clone, Debug)]
pub struct GreedyInsertion;

impl<T, R> RepairOperator<T, R> for GreedyInsertion
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "GreedyInsertion"
    }

    fn repair<'b, 's, 'm, 'p>(
        &self,
        planning_context: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        _rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let solver_model = planning_context.model();
        let mut plan_builder = planning_context.builder();

        loop {
            let unassigned_request_indices: Vec<RequestIndex> = plan_builder
                .with_explorer(|plan_explorer| plan_explorer.iter_unassigned().collect());
            if unassigned_request_indices.is_empty() {
                break;
            }

            let best_move = plan_builder.with_explorer(|plan_explorer| {
                unassigned_request_indices
                    .iter()
                    .filter_map(|&request_index| {
                        best_insertion_for_request_ex(plan_explorer, solver_model, request_index)
                            .map(|(free_window, start_time, cost)| {
                                (request_index, free_window, start_time, cost)
                            })
                    })
                    .min_by(
                        |(ria, _fba, sta, ca), (rib, _fbb, stb, cb)| match ca.cmp(cb) {
                            Ordering::Less => Ordering::Less,
                            Ordering::Greater => Ordering::Greater,
                            Ordering::Equal => match sta.cmp(stb) {
                                Ordering::Less => Ordering::Less,
                                Ordering::Greater => Ordering::Greater,
                                Ordering::Equal => ria.get().cmp(&rib.get()),
                            },
                        },
                    )
            });

            if let Some((request_index, best_free_window, best_start_time, _cost)) = best_move {
                let _ = plan_builder.propose_assignment(
                    request_index,
                    best_start_time,
                    &best_free_window,
                );
            } else {
                // No feasible insertions left for any request
                break;
            }
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

// ======================================================================
// EarliestWindowInsertion (unchanged API)
// ======================================================================

/// Sorts all unassigned requests by their feasible window start time and attempts
/// to insert them in that order into their best available position.
#[derive(Clone, Debug)]
pub struct EarliestWindowInsertion;

impl<T, R> RepairOperator<T, R> for EarliestWindowInsertion
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + Mul<Output = Cost>,
    R: Rng,
{
    fn name(&self) -> &str {
        "EarliestWindowInsertion"
    }

    fn repair<'b, 's, 'm, 'p>(
        &self,
        planning_context: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        _rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        let solver_model = planning_context.model();
        let mut plan_builder = planning_context.builder();

        let mut unassigned_request_indices: Vec<RequestIndex> =
            plan_builder.with_explorer(|plan_explorer| plan_explorer.iter_unassigned().collect());

        // Sort requests chronologically by their window start time.
        unassigned_request_indices.sort_by_key(|&ri| solver_model.feasible_interval(ri).start());

        for request_index in unassigned_request_indices {
            if let Some((best_free_window, best_start_time, _best_cost)) = plan_builder
                .with_explorer(|plan_explorer| {
                    best_insertion_for_request_ex(plan_explorer, solver_model, request_index)
                })
            {
                let _ = plan_builder.propose_assignment(
                    request_index,
                    best_start_time,
                    &best_free_window,
                );
            }
        }

        let plan = plan_builder.finalize();
        if is_zero_delta_plan(&plan) {
            None
        } else {
            Some(plan)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::planner::PlanningContext,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
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

    fn make_problem_sequential_windows(n: usize, pt: i64, spacing: i64) -> Problem<i64> {
        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();
        builder.add_berth(berth(1, 0, 10_000));
        for i in 0..n {
            let start = (i as i64) * spacing;
            builder.add_flexible(flex_req(
                (i + 1) as u32,
                (start, start + 500),
                &[(1, pt)],
                1,
            ));
        }
        builder.build().expect("valid problem")
    }

    fn make_unassigned_state<'p>(model: &SolverModel<'p, i64>) -> SolverState<'p, i64> {
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let dv = DecisionVarVec::from(dvars);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let fit = Fitness::new(0, model.flexible_requests_len());
        SolverState::new(dv, term, fit)
    }

    fn make_ctx<'b, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        state: &'s SolverState<'p, i64>,
        buffer: &'b mut [DecisionVar<i64>],
    ) -> PlanningContext<'b, 's, 'm, 'p, i64> {
        PlanningContext::new(model, state, buffer)
    }

    #[test]
    fn randomized_greedy_insertion_assigns_some() {
        let prob = make_problem_sequential_windows(5, 10, 10);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_unassigned_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(1337);

        let op = RandomizedGreedyInsertion::new(1.6..=2.2);
        let plan = op.repair(&mut ctx, &mut rng).expect("plan expected");

        assert!(!plan.decision_var_patches.is_empty());
        assert!(plan.delta_unassigned < 0);
    }

    #[test]
    fn k_regret_insertion_assigns_some() {
        let prob = make_problem_sequential_windows(4, 12, 15);
        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_unassigned_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(4242);

        let op = KRegretInsertion::new(2..=3);
        let plan = op.repair(&mut ctx, &mut rng).expect("plan expected");

        assert!(!plan.decision_var_patches.is_empty());
    }

    #[test]
    fn greedy_insertion_picks_best_first() {
        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();
        builder.add_berth(berth(1, 0, 1000));
        // r1: costly, window starts at 50. best insertion cost is high.
        builder.add_flexible(flex_req(1, (50, 200), &[(1, 10)], 1));
        // r2: cheap, window starts at 0. best insertion cost is low.
        builder.add_flexible(flex_req(2, (0, 200), &[(1, 10)], 1));
        let prob = builder.build().unwrap();

        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_unassigned_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); 2];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(1);

        let op = GreedyInsertion;
        let plan = op.repair(&mut ctx, &mut rng).unwrap();

        // The first request inserted should be r2 (index 1), as it has the lowest cost (tie-broken by start/index).
        let first_patch = &plan.decision_var_patches[0];
        let r2_index = model.index_manager().request_index(rid(2)).unwrap();
        assert_eq!(first_patch.index, r2_index);
    }

    #[test]
    fn earliest_window_insertion_respects_order() {
        let mut builder = berth_alloc_model::problem::builder::ProblemBuilder::new();
        builder.add_berth(berth(1, 0, 1000));
        // Order: r2 (start 20), r3 (start 50), r1 (start 100)
        builder.add_flexible(flex_req(1, (100, 200), &[(1, 10)], 1));
        builder.add_flexible(flex_req(2, (20, 200), &[(1, 10)], 1));
        builder.add_flexible(flex_req(3, (50, 200), &[(1, 10)], 1));
        let prob = builder.build().unwrap();

        let model = SolverModel::try_from(&prob).unwrap();
        let state = make_unassigned_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); 3];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(1);

        let op = EarliestWindowInsertion;
        let plan = op.repair(&mut ctx, &mut rng).unwrap();

        // The plan should contain patches in chronological order of window start
        let r2_index = model.index_manager().request_index(rid(2)).unwrap();
        let r3_index = model.index_manager().request_index(rid(3)).unwrap();
        let r1_index = model.index_manager().request_index(rid(1)).unwrap();

        assert_eq!(plan.decision_var_patches[0].index, r2_index);
        assert_eq!(plan.decision_var_patches[1].index, r3_index);
        assert_eq!(plan.decision_var_patches[2].index, r1_index);
    }

    #[test]
    fn repair_returns_none_when_all_assigned() {
        let prob = make_problem_sequential_windows(2, 10, 10);
        let model = SolverModel::try_from(&prob).unwrap();
        // Build a fully-assigned state
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let dvars: Vec<_> = (0..2).map(|_| DecisionVar::assigned(b_ix, tp(0))).collect();
        let dv = DecisionVarVec::from(dvars);
        let term = TerminalOccupancy::new(model.problem().berths().iter());
        let fit = Fitness::new(0, 0);
        let state = SolverState::new(dv, term, fit);

        let mut buffer = vec![DecisionVar::unassigned(); 2];
        let mut ctx = make_ctx(&model, &state, &mut buffer);
        let mut rng = StdRng::seed_from_u64(7);

        assert!(
            RandomizedGreedyInsertion::new(1.2..=1.2)
                .repair(&mut ctx, &mut rng)
                .is_none()
        );
        assert!(
            KRegretInsertion::new(3..=3)
                .repair(&mut ctx, &mut rng)
                .is_none()
        );
        assert!(GreedyInsertion.repair(&mut ctx, &mut rng).is_none());
        assert!(EarliestWindowInsertion.repair(&mut ctx, &mut rng).is_none());
    }
}
