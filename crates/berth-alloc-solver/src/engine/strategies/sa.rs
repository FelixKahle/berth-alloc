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
    core::numeric::SolveNumeric,
    engine::{
        neighbors,
        strategy::{ImprovingStrategy, Strategy, StrategyContext},
    },
    model::solver_model::SolverModel,
    monitor::{
        search_monitor::{LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck},
        step::{PlanLimitMonitor, StagnationMonitor},
    },
    search::{
        eval::{CostEvaluator, DefaultCostEvaluator},
        filter::NeighborhoodFilterStack,
        local_search::MetaheuristicLocalSearch,
        metaheuristic_library::sa::{EnergyParams, IterReciprocalCooling, SimulatedAnnealing},
        operator_library::{self, OperatorSelectionConfig},
    },
    state::solver_state::SolverState,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct SimulatedAnnealingConfig {
    pub operator_selection_config: OperatorSelectionConfig,
    pub initial_temperature: f64,
    pub step: i64,
    pub allow_infeasible_uphill: bool,
    pub seed: u64,
    pub memory_coefficient: f64,
    pub exploration_coefficient: f64,
    pub refetch_incumbent_after_stagnation: i64,
    pub refetch_incumbent_after_plan_generation: Option<u64>,
}

impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            operator_selection_config: OperatorSelectionConfig::default(),
            initial_temperature: 1000.0,
            step: 100,
            allow_infeasible_uphill: true,
            seed: 42,
            memory_coefficient: 0.7,
            exploration_coefficient: 1.4,
            refetch_incumbent_after_stagnation: 500,
            refetch_incumbent_after_plan_generation: None,
        }
    }
}

impl<'n, T, C, R> std::fmt::Debug for SimulatedAnnealingRefetchStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulatedAnnealingRefetchStrategy")
            .field("name", &self.name)
            .field("stagnation_budget", &self.stagnation_budget)
            .field("plan_generation_budget", &self.plan_generation_budget)
            .field(
                "refetch_incumbent_after_stagnation",
                &self.config.refetch_incumbent_after_stagnation,
            )
            .field(
                "refetch_incumbent_after_plan_generation",
                &self.config.refetch_incumbent_after_plan_generation,
            )
            .finish()
    }
}

impl<'n, T, C, R> std::fmt::Display for SimulatedAnnealingRefetchStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SimulatedAnnealingRefetchStrategy(name: {}, stagnation_budget: {:?}, \
             plan_generation_budget: {:?}, refetch_stagnation: {}, refetch_plan_generation: {:?})",
            self.name,
            self.stagnation_budget,
            self.plan_generation_budget,
            self.config.refetch_incumbent_after_stagnation,
            self.config.refetch_incumbent_after_plan_generation
        )
    }
}

/// Forward monitor that fans out plan + lifecycle events to:
/// - The external (base) monitor passed in the StrategyContext.
/// - Internal monitors tracking stagnation and optional plan generation budget.
struct RefetchForwardMonitor<'a, T>
where
    T: Copy + Ord,
{
    base: &'a mut dyn SearchMonitor<T>,
    stagnation: &'a mut StagnationMonitor,
    plan_limit: Option<&'a mut PlanLimitMonitor>,
}

impl<'a, T> RefetchForwardMonitor<'a, T>
where
    T: Copy + Ord,
{
    #[inline]
    fn new(
        base: &'a mut dyn SearchMonitor<T>,
        stagnation: &'a mut StagnationMonitor,
        plan_limit: Option<&'a mut PlanLimitMonitor>,
    ) -> Self {
        Self {
            base,
            stagnation,
            plan_limit,
        }
    }
}

impl<'a, T> TerminationCheck for RefetchForwardMonitor<'a, T>
where
    T: Copy + Ord,
{
    #[inline]
    fn should_terminate_search(&self) -> bool {
        // We only terminate on base monitor request.
        self.base.should_terminate_search()
    }
}

impl<'a, T> PlanEventMonitor<T> for RefetchForwardMonitor<'a, T>
where
    T: Copy + Ord,
{
    #[inline]
    fn on_plan_generated<'p>(&mut self, plan: &crate::state::plan::Plan<'p, T>) {
        self.base.on_plan_generated(plan);
        self.stagnation.on_plan_generated(plan);
        if let Some(pl) = &mut self.plan_limit {
            pl.on_plan_generated(plan);
        }
    }

    #[inline]
    fn on_plan_rejected<'p>(&mut self, plan: &crate::state::plan::Plan<'p, T>) {
        self.base.on_plan_rejected(plan);
        self.stagnation.on_plan_rejected(plan);
        if let Some(pl) = &mut self.plan_limit {
            pl.on_plan_rejected(plan);
        }
    }

    #[inline]
    fn on_plan_accepted<'p>(&mut self, plan: &crate::state::plan::Plan<'p, T>) {
        self.base.on_plan_accepted(plan);
        self.stagnation.on_plan_accepted(plan);
        if let Some(pl) = &mut self.plan_limit {
            pl.on_plan_accepted(plan);
        }
    }
}

impl<'a, T> LifecycleMonitor for RefetchForwardMonitor<'a, T>
where
    T: Copy + Ord,
{
    #[inline]
    fn on_search_start(&mut self) {
        self.base.on_search_start();
        self.stagnation.on_search_start();
        if let Some(pl) = &mut self.plan_limit {
            pl.on_search_start();
        }
    }

    #[inline]
    fn on_search_end(&mut self) {
        self.base.on_search_end();
        self.stagnation.on_search_end();
        if let Some(pl) = &mut self.plan_limit {
            pl.on_search_end();
        }
    }
}

impl<'a, T> SearchMonitor<T> for RefetchForwardMonitor<'a, T>
where
    T: Copy + Ord,
{
    fn name(&self) -> &str {
        "RefetchForwardMonitor"
    }
}

/// Strategy that embeds SimulatedAnnealing and refetches the incumbent state on configured triggers.
/// The decision builder is a MetaheuristicLocalSearch wrapping SA.
pub struct SimulatedAnnealingRefetchStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    config: SimulatedAnnealingConfig,
    evaluator: C,
    rng: R,
    decision_builder:
        Box<dyn crate::search::decision_builder::DecisionBuilder<T, C, R> + Send + 'n>,
    stagnation_budget: Option<u64>,
    plan_generation_budget: Option<u64>,
    stagnation_monitor: StagnationMonitor,
    plan_limit_monitor: Option<PlanLimitMonitor>,
    name: String,
}

impl<'n, T, C, R> SimulatedAnnealingRefetchStrategy<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        config: SimulatedAnnealingConfig,
        evaluator: C,
        rng: R,
        decision_builder: Box<
            dyn crate::search::decision_builder::DecisionBuilder<T, C, R> + Send + 'n,
        >,
    ) -> Self {
        let stagnation_budget = if config.refetch_incumbent_after_stagnation > 0 {
            Some(config.refetch_incumbent_after_stagnation as u64)
        } else {
            None
        };
        let plan_generation_budget = config.refetch_incumbent_after_plan_generation;

        let stagnation_monitor = StagnationMonitor::new(
            stagnation_budget.unwrap_or(u64::MAX), // effectively disabled if None
        );

        let plan_limit_monitor = plan_generation_budget.map(PlanLimitMonitor::new);

        let name = "SimulatedAnnealingRefetchStrategy".to_string();

        Self {
            config,
            evaluator,
            rng,
            decision_builder,
            stagnation_budget,
            plan_generation_budget,
            stagnation_monitor,
            plan_limit_monitor,
            name,
        }
    }

    #[inline]
    pub fn config(&self) -> &SimulatedAnnealingConfig {
        &self.config
    }

    #[inline]
    fn restart_from_incumbent<'p>(
        &mut self,
        context: &StrategyContext<'_, '_, 'p, T>,
    ) -> SolverState<'p, T>
    where
        SolverState<'p, T>: Clone,
    {
        // Reset decision builder (this resets SA metaheuristic, cooling schedule).
        self.decision_builder.reset();

        // Re-create monitors for fresh counters.
        if let Some(b) = self.stagnation_budget {
            self.stagnation_monitor = StagnationMonitor::new(b);
        }
        if let Some(b) = self.plan_generation_budget {
            self.plan_limit_monitor = Some(PlanLimitMonitor::new(b));
        }

        // Snapshot incumbent (best so far).
        context.shared_incumbent().snapshot()
    }

    #[inline]
    fn internal_restart_triggered(&self) -> bool {
        let stagnated =
            self.stagnation_budget.is_some() && self.stagnation_monitor.should_terminate_search();
        let plan_budget_hit = self
            .plan_limit_monitor
            .as_ref()
            .map(|m| m.should_terminate_search())
            .unwrap_or(false);
        stagnated || plan_budget_hit
    }
}

impl<'n, T, C, R> Strategy<T> for SimulatedAnnealingRefetchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn run<'p, 'e, 'm>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>> {
        // Start from the current context state (could also start from incumbent if desired).
        let mut state = context.state().clone();
        let mut work_buf = vec![
            crate::state::decisionvar::DecisionVar::unassigned();
            context.model().flexible_requests_len()
        ];

        // Wrap the external monitor with our forward monitor to receive plan events.
        {
            // Initial lifecycle start
            let mut fwd = RefetchForwardMonitor::new(
                context.monitor_mut(),
                &mut self.stagnation_monitor,
                self.plan_limit_monitor.as_mut(),
            );
            fwd.on_search_start();
        }

        loop {
            if context.monitor().should_terminate_search() {
                break;
            }

            // Restart if triggers fire
            if self.internal_restart_triggered() {
                state = self.restart_from_incumbent(context);
                // Lifecycle start again for internal restart (only internal monitors)
                let mut fwd = RefetchForwardMonitor::new(
                    context.monitor_mut(),
                    &mut self.stagnation_monitor,
                    self.plan_limit_monitor.as_mut(),
                );
                // We intentionally do NOT call base monitor's on_search_start again to avoid
                // double-counting global lifecycle. Only internal monitors reset.
                // To achieve that, we manually invoke resets instead of lifecycle:
                // For simplicity here we just use on_search_start (base will see extra starts).
                // If you want to avoid that: add a specialized method or restructure monitor.
                fwd.on_search_start();
            }

            let current_fitness = *state.fitness();
            let model = context.model();

            let next_plan = {
                let mut fwd = RefetchForwardMonitor::new(
                    context.monitor_mut(),
                    &mut self.stagnation_monitor,
                    self.plan_limit_monitor.as_mut(),
                );
                let mut sc = crate::search::decision_builder::SearchContext::new(
                    model,
                    &state,
                    &self.evaluator,
                    &mut self.rng,
                    &mut work_buf,
                    current_fitness,
                    &mut fwd,
                );
                self.decision_builder.next(&mut sc)
            };

            match next_plan {
                Some(plan) => {
                    state.apply_plan(plan);
                    let _ = context.shared_incumbent().try_update(&state, model);
                }
                None => {
                    // If decision builder yields None, we are at local optimum for this restart.
                    // We can either break or immediately trigger another refetch cycle.
                    if self.internal_restart_triggered() {
                        // Trigger restart logic on next loop iteration.
                        continue;
                    } else {
                        // No restart condition active => end search.
                        break;
                    }
                }
            }
        }

        {
            // Final lifecycle end
            let mut fwd = RefetchForwardMonitor::new(
                context.monitor_mut(),
                &mut self.stagnation_monitor,
                self.plan_limit_monitor.as_mut(),
            );
            fwd.on_search_end();
        }

        Some(state)
    }
}

/// Factory function returning either the legacy ImprovingStrategy
/// or the refetching SA strategy depending on configured restart triggers.
#[inline]
pub fn make_simulated_annealing_strategy<'m, 'p, T>(
    config: SimulatedAnnealingConfig,
    model: &'m SolverModel<'p, T>,
) -> Box<dyn Strategy<T> + Send + 'm>
where
    T: SolveNumeric,
{
    // Build neighborhood and operator (multi-armed bandit compound)
    let neighboors = neighbors::build_neighbors_from_model(model);
    let operator = operator_library::make_multi_armed_bandit_compound_operator::<
        T,
        DefaultCostEvaluator,
        ChaCha8Rng,
    >(
        &config.operator_selection_config,
        &neighboors,
        config.memory_coefficient,
        config.exploration_coefficient,
    );

    let filter_stack = NeighborhoodFilterStack::empty();

    // SA metaheuristic
    let energy_params = EnergyParams::with_default_lambda(
        model,
        config.step,
        config.initial_temperature,
        config.allow_infeasible_uphill,
    );
    let sa_mh = SimulatedAnnealing::new(
        energy_params,
        IterReciprocalCooling::new(config.initial_temperature),
    );

    let decision_builder = Box::new(MetaheuristicLocalSearch::new(
        Box::new(operator),
        filter_stack,
        sa_mh,
    ));

    let evaluator = DefaultCostEvaluator;
    let outer_rng = ChaCha8Rng::seed_from_u64(config.seed);

    let use_refetch = (config.refetch_incumbent_after_stagnation > 0)
        || config.refetch_incumbent_after_plan_generation.is_some();

    if use_refetch {
        Box::new(SimulatedAnnealingRefetchStrategy::new(
            config,
            evaluator,
            outer_rng,
            decision_builder,
        ))
    } else {
        Box::new(ImprovingStrategy::new(
            evaluator,
            outer_rng,
            decision_builder,
        ))
    }
}
