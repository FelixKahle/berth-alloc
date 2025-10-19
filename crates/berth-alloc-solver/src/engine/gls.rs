// src/search/strategy_gls.rs

use crate::{
    core::numeric::SolveNumeric,
    engine::search::{SearchContext, SearchStrategy},
    search::{
        operator::LocalMoveOperator,
        operator_library::{
            local::{
                CrossExchangeAcrossBerths, OrOptBlockRelocate, RelocateSingleBest,
                ShiftEarlierOnSameBerth, SwapPairSameBerth,
            },
            math::MipBlockReoptimize,
        },
        planner::PlanningContext,
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::SolverState,
    },
};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::sync::atomic::Ordering as AtomicOrdering;

type FeatureKey = (usize, usize); // (request_raw, berth_raw)

#[derive(Clone, Copy)]
pub enum HardRefetchMode {
    IfBetter,
    Always,
}

pub struct GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    local_ops: Vec<Box<dyn LocalMoveOperator<T, R>>>,
    lambda: i64,
    penalty_step: i64,
    stagnation_rounds_before_pulse: usize,
    pulse_top_k: usize,
    max_local_steps: usize,
    penalties: HashMap<FeatureKey, i64>,

    // ILS-like sync/refetch knobs
    refetch_after_stale: usize, // 0 => disabled
    hard_refetch_every: usize,  // 0 => disabled
    hard_refetch_mode: HardRefetchMode,
}

impl<T, R> Default for GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, R> GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    pub fn new() -> Self {
        Self {
            local_ops: Vec::new(),
            lambda: 4,
            penalty_step: 1,
            stagnation_rounds_before_pulse: 16,
            pulse_top_k: 8,
            max_local_steps: 512,
            penalties: HashMap::new(),
            refetch_after_stale: 128,
            hard_refetch_every: 0,
            hard_refetch_mode: HardRefetchMode::IfBetter,
        }
    }
    pub fn with_local_op(mut self, op: Box<dyn LocalMoveOperator<T, R>>) -> Self {
        self.local_ops.push(op);
        self
    }
    pub fn with_lambda(mut self, lambda: i64) -> Self {
        self.lambda = lambda;
        self
    }
    pub fn with_max_local_steps(mut self, steps: usize) -> Self {
        self.max_local_steps = steps;
        self
    }
    pub fn with_refetch_after_stale(mut self, rounds: usize) -> Self {
        self.refetch_after_stale = rounds;
        self
    }
    pub fn with_hard_refetch_every(mut self, period: usize) -> Self {
        self.hard_refetch_every = period;
        self
    }
    pub fn with_hard_refetch_mode(mut self, mode: HardRefetchMode) -> Self {
        self.hard_refetch_mode = mode;
        self
    }

    #[inline]
    fn feature_of(&self, ri_raw: usize, berth_raw: usize) -> FeatureKey {
        (ri_raw, berth_raw)
    }

    fn augmented_delta_for_plan<'b, 's, 'm, 'p>(
        &self,
        _context: &mut PlanningContext<'b, 's, 'm, 'p, T>,
        plan: &crate::state::plan::Plan<'p, T>,
    ) -> i64 {
        let mut delta_penalty: i64 = 0;
        for p in &plan.decision_var_patches {
            let rid = p.index.get();
            match p.patch {
                DecisionVar::Assigned(Decision { berth_index, .. }) => {
                    let key = self.feature_of(rid, berth_index.get());
                    let pen = *self.penalties.get(&key).unwrap_or(&0);
                    delta_penalty = delta_penalty.saturating_add(self.lambda.saturating_mul(pen));
                }
                DecisionVar::Unassigned => {
                    // No-op in our model; paired assign handles net change.
                }
            }
        }
        plan.delta_cost.saturating_add(delta_penalty)
    }

    fn pulse_increase_penalties<'b, 's, 'm, 'p>(
        &mut self,
        context: &mut PlanningContext<'b, 's, 'm, 'p, T>,
    ) {
        use crate::model::index::RequestIndex;
        let builder = context.builder();
        let utils = builder.with_explorer(|ex| {
            let mut tmp: Vec<(FeatureKey, i64, i64)> = Vec::new();
            for (i, dv) in ex.decision_vars().iter().enumerate() {
                if let DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                }) = *dv
                    && let Some(base) = ex.peek_cost(RequestIndex::new(i), start_time, berth_index)
                    {
                        let key = (i, berth_index.get());
                        let p = *self.penalties.get(&key).unwrap_or(&0);
                        let util = base / (1 + p);
                        tmp.push((key, util, p));
                    }
            }
            tmp.sort_by_key(|(_, util, _)| -(*util)); // descending
            tmp
        });

        for (idx, (key, _, _)) in utils.into_iter().enumerate() {
            if idx >= self.pulse_top_k {
                break;
            }
            *self.penalties.entry(key).or_insert(0) += self.penalty_step;
        }
    }

    #[inline]
    fn should_hard_refetch(&self, outer_rounds: usize) -> bool {
        self.hard_refetch_every > 0
            && outer_rounds > 0
            && outer_rounds.is_multiple_of(self.hard_refetch_every)
    }

    #[inline]
    fn maybe_apply_periodic_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        outer_rounds: usize,
    ) {
        if !self.should_hard_refetch(outer_rounds) {
            return;
        }
        let best_now = context.shared_incumbent().peek();
        let do_fetch = match self.hard_refetch_mode {
            HardRefetchMode::IfBetter => best_now < *current.fitness(),
            HardRefetchMode::Always => true,
        };
        if do_fetch {
            tracing::debug!(
                "GLS: periodic refetch at round {} (curr {}, inc {})",
                outer_rounds,
                current.fitness(),
                best_now
            );
            *current = context.shared_incumbent().snapshot();
        }
    }

    #[inline]
    fn maybe_apply_stale_refetch<'e, 'm, 'p>(
        &self,
        current: &mut SolverState<'p, T>,
        context: &SearchContext<'e, 'm, 'p, T, R>,
        stale_rounds: usize,
    ) -> bool {
        if self.refetch_after_stale == 0 || stale_rounds < self.refetch_after_stale {
            return false;
        }
        let best_now = context.shared_incumbent().peek();
        if best_now < *current.fitness() {
            tracing::debug!(
                "GLS: staleness refetch after {} rounds ({} -> {})",
                stale_rounds,
                current.fitness(),
                best_now
            );
            *current = context.shared_incumbent().snapshot();
            true
        } else {
            false
        }
    }
}

impl<T, R> SearchStrategy<T, R> for GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "Guided Local Search"
    }

    #[tracing::instrument(name = "GLS Search", skip(self, context))]
    fn run<'e, 'm, 'p>(&mut self, context: &mut SearchContext<'e, 'm, 'p, T, R>) {
        let stop = context.stop();
        let model = context.model();
        if self.local_ops.is_empty() {
            tracing::warn!("GLS: no local operators configured");
            return;
        }

        let mut current: SolverState<'p, T> = context.shared_incumbent().snapshot();
        let mut stale = 0usize;
        let mut outer_rounds = 0usize;

        use crate::state::decisionvar::DecisionVar;
        let mut dv_buf: Vec<DecisionVar<T>> =
            vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        'outer: loop {
            if stop.load(AtomicOrdering::Relaxed) {
                break 'outer;
            }
            outer_rounds = outer_rounds.saturating_add(1);

            // Periodic refetch like ILS
            self.maybe_apply_periodic_refetch(&mut current, context, outer_rounds);

            let mut accepted_any = false;

            for _ in 0..self.max_local_steps {
                if stop.load(AtomicOrdering::Relaxed) {
                    break 'outer;
                }

                // operator order
                let mut order: Vec<usize> = (0..self.local_ops.len()).collect();
                order.shuffle(context.rng());

                let mut step_taken = false;

                for &i in &order {
                    let op = &self.local_ops[i];
                    let mut pc = PlanningContext::new(model, &current, dv_buf.as_mut_slice());
                    if let Some(plan) = op.propose(&mut pc, context.rng()) {
                        // augmented acceptance
                        let aug_delta = self.augmented_delta_for_plan(&mut pc, &plan);
                        if aug_delta < 0 {
                            let mut tmp = current.clone();
                            tmp.apply_plan(plan);

                            // Publish improvement vs global (true objective)
                            let _ = context.shared_incumbent().try_update(&tmp);

                            current = tmp;
                            step_taken = true;
                            accepted_any = true;
                            break;
                        }
                    }
                }
                if !step_taken {
                    break;
                }
            }

            if !accepted_any {
                stale = stale.saturating_add(1);

                if stale >= self.stagnation_rounds_before_pulse {
                    // Pulse penalties on highest-utility features
                    let mut pc = PlanningContext::new(model, &current, dv_buf.as_mut_slice());
                    self.pulse_increase_penalties(&mut pc);
                    stale = 0;
                    tracing::trace!("GLS: penalty pulse (top_k={})", self.pulse_top_k);
                } else {
                    // Try staleness refetch; if not better, keep looping (don't exit early)
                    if self.maybe_apply_stale_refetch(&mut current, context, stale) {
                        stale = 0;
                    }
                }
            } else {
                stale = 0;
            }
        }

        let _ = context.shared_incumbent().try_update(&current);
    }
}

pub fn gls_strategy<T, R>(
    _: &crate::model::solver_model::SolverModel<T>,
) -> GuidedLocalSearchStrategy<T, R>
where
    T: SolveNumeric + From<i32>,
    R: rand::Rng,
{
    GuidedLocalSearchStrategy::new()
        .with_lambda(4)
        .with_max_local_steps(1024)
        .with_refetch_after_stale(128)
        .with_hard_refetch_every(0)
        .with_hard_refetch_mode(HardRefetchMode::IfBetter)
        .with_local_op(Box::new(ShiftEarlierOnSameBerth {
            number_of_candidates_to_try_range: 8..=24,
        }))
        .with_local_op(Box::new(RelocateSingleBest {
            number_of_candidates_to_try_range: 8..=24,
        }))
        .with_local_op(Box::new(SwapPairSameBerth {
            number_of_pair_attempts_to_try_range: 10..=40,
        }))
        .with_local_op(Box::new(CrossExchangeAcrossBerths {
            number_of_pair_attempts_to_try_range: 12..=48,
        }))
        .with_local_op(Box::new(OrOptBlockRelocate::new(2..=4, 1.4..=2.0)))
        .with_local_op(Box::new(MipBlockReoptimize::same_berth(
            2..=5,  // block length k
            3..=6,  // candidate starts per free interval
            6..=10, // max candidates per request
        )))
        // (B) across-all-berths block reopt — heavier, keep caps tighter
        .with_local_op(Box::new(MipBlockReoptimize::across_all_berths(
            2..=4, // k a bit smaller across berths
            2..=4, // fewer starts per interval
            4..=8, // tighter cap per request
        )))
}
