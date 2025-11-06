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
        strategy::{ImprovingStrategy, Strategy},
    },
    model::solver_model::SolverModel,
    search::{
        eval::DefaultCostEvaluator,
        filter::NeighborhoodFilterStack,
        local_search::MetaheuristicLocalSearch,
        operator_library::{self, OperatorSelectionConfig},
        sa::{EnergyParams, IterReciprocalCooling, SimulatedAnnealing},
    },
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
        }
    }
}

impl std::fmt::Display for SimulatedAnnealingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SimulatedAnnealingConfig {{ \
            initial_temperature: {}, \
            step: {}, \
            allow_infeasible_uphill: {}, \
            seed: {}, \
            memory_coefficient: {}, \
            exploration_coefficient: {} \
            }}",
            self.initial_temperature,
            self.step,
            self.allow_infeasible_uphill,
            self.seed,
            self.memory_coefficient,
            self.exploration_coefficient,
        )
    }
}

#[inline]
pub fn make_simulated_annealing_strategy<'m, 'p, T>(
    config: SimulatedAnnealingConfig,
    model: &'m SolverModel<'p, T>,
) -> Box<dyn Strategy<T> + Send + 'm>
where
    T: SolveNumeric,
{
    // Build neighborhood and operator (MAB compound)
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

    // No extra neighborhood filters by default
    let filter_stack = NeighborhoodFilterStack::empty();

    // SA metaheuristic with reciprocal cooling and its own RNG
    let energy_params = EnergyParams::with_default_lambda(
        model,
        config.step,
        config.initial_temperature,
        config.allow_infeasible_uphill,
    );
    let mh = SimulatedAnnealing::new(
        energy_params,
        IterReciprocalCooling::new(config.initial_temperature),
    );

    // Decision builder = Local search + filters + metaheuristic
    let decision_builder = Box::new(MetaheuristicLocalSearch::new(
        Box::new(operator),
        filter_stack,
        mh,
    ));

    // Evaluator and outer RNG owned by the strategy
    let evaluator = DefaultCostEvaluator;
    let outer_rng = ChaCha8Rng::seed_from_u64(config.seed);

    Box::new(ImprovingStrategy::new(
        evaluator,
        outer_rng,
        decision_builder,
    ))
}
