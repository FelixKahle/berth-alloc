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

use berth_alloc_model::prelude::{Problem, SolutionView};
use berth_alloc_model::problem::loader::ProblemLoader;
use berth_alloc_solver::engine::solver::Solver;
use berth_alloc_solver::engine::{neighbors, strategies};
use berth_alloc_solver::search::eval::DefaultCostEvaluator;
use berth_alloc_solver::search::filter::NeighborhoodFilterStack;
use berth_alloc_solver::search::lns_library::{self, RepairSelectionConfig, RuinSelectionConfig};
use berth_alloc_solver::search::local_search::MetaheuristicLocalSearch;
use berth_alloc_solver::search::metaheuristic_library::greedy_descent::GreedyDescentMetaheuristic;
use berth_alloc_solver::search::operator_library::{self, OperatorSelectionConfig};
use chrono::{DateTime, Utc};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

fn find_instances_dir() -> Option<PathBuf> {
    let mut cur: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(p) = cur {
        let cand = p.join("instances");
        if cand.is_dir() {
            return Some(cand);
        }
        cur = p.parent();
    }
    None
}

#[allow(dead_code)]
fn instances() -> impl Iterator<Item = (Problem<i64>, String)> {
    let inst_dir = find_instances_dir()
        .expect("Could not find an `instances/` directory in any ancestor of CARGO_MANIFEST_DIR");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&inst_dir)
        .expect("read_dir(instances) failed")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && e.path().extension().map(|x| x == "txt").unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    files.sort();
    files.into_iter().filter_map(|f| {
        let loader = ProblemLoader::default();
        match loader.from_path(&f) {
            Ok(problem) => {
                let name = f
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| f.to_string_lossy().into_owned());
                Some((problem, name))
            }
            Err(_) => None,
        }
    })
}

#[allow(dead_code)]
fn enable_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE)
        .init();
}

#[derive(Serialize)]
struct RunRecord {
    iteration: usize,
    filename: String,
    start_ts: DateTime<Utc>,
    end_ts: DateTime<Utc>,
    runtime_ms: u128,
    cost: Option<i64>,
}

fn main() {
    enable_tracing();

    let mut results: Vec<RunRecord> = Vec::new();

    for (iter, (problem, file)) in instances().enumerate().take(1) {
        let iteration = iter + 1;

        tracing::info!(
            "Solving [{}] {} with {} berths and {} flexible requests",
            iteration,
            file,
            problem.berths().len(),
            problem.flexible_requests().len()
        );

        let start_ts = Utc::now();
        let t0 = Instant::now();

        // Build a Solver with 8 distinct strategies
        let mut solver = Solver::new()
            .with_time_limit(std::time::Duration::from_secs(120))
            // Strategy 1: SA (hot start, large step) with aggressive restart on stagnation
            .with_strategy_fn(|model| {
                let cfg = strategies::sa::SimulatedAnnealingConfig {
                    operator_selection_config: OperatorSelectionConfig::default(),
                    initial_temperature: 2000.0,
                    step: 150,
                    allow_infeasible_uphill: true,
                    seed: 10,
                    memory_coefficient: 0.6,
                    exploration_coefficient: 1.8,
                    refetch_incumbent_after_stagnation: 4_000,
                    refetch_incumbent_after_plan_generation: Some(40_000),
                };
                strategies::sa::make_simulated_annealing_strategy(cfg, model)
            })
            // Strategy 2: SA (cool start) more deterministic, small step, fewer uphill moves
            .with_strategy_fn(|model| {
                let cfg = strategies::sa::SimulatedAnnealingConfig {
                    operator_selection_config: OperatorSelectionConfig {
                        use_relocate: true,
                        use_slide: true,
                        use_swap: true,
                        use_push_insert: false,
                        use_assign: true,
                        use_unassign: true,
                        use_or_opt: false,
                    },
                    initial_temperature: 250.0,
                    step: 60,
                    allow_infeasible_uphill: false,
                    seed: 11,
                    memory_coefficient: 0.75,
                    exploration_coefficient: 1.2,
                    refetch_incumbent_after_stagnation: 3_000,
                    refetch_incumbent_after_plan_generation: Some(25_000),
                };
                strategies::sa::make_simulated_annealing_strategy(cfg, model)
            })
            // Strategy 3: SA (balanced temperature) high exploration coefficient
            .with_strategy_fn(|model| {
                let cfg = strategies::sa::SimulatedAnnealingConfig {
                    operator_selection_config: OperatorSelectionConfig::default(),
                    initial_temperature: 1200.0,
                    step: 80,
                    allow_infeasible_uphill: true,
                    seed: 12,
                    memory_coefficient: 0.7,
                    exploration_coefficient: 2.2,
                    refetch_incumbent_after_stagnation: 6_000,
                    refetch_incumbent_after_plan_generation: Some(60_000),
                };
                strategies::sa::make_simulated_annealing_strategy(cfg, model)
            })
            // Strategy 4: SA (very small step) fine-grained local refinement
            .with_strategy_fn(|model| {
                let cfg = strategies::sa::SimulatedAnnealingConfig {
                    operator_selection_config: OperatorSelectionConfig {
                        use_relocate: true,
                        use_slide: false,
                        use_swap: true,
                        use_push_insert: true,
                        use_assign: true,
                        use_unassign: false,
                        use_or_opt: true,
                    },
                    initial_temperature: 800.0,
                    step: 20,
                    allow_infeasible_uphill: true,
                    seed: 13,
                    memory_coefficient: 0.65,
                    exploration_coefficient: 1.6,
                    refetch_incumbent_after_stagnation: 5_000,
                    refetch_incumbent_after_plan_generation: Some(35_000),
                };
                strategies::sa::make_simulated_annealing_strategy(cfg, model)
            })
            // Strategy 5: ILS (Greedy descent improving) with limited ruin set (random subset + time band)
            .with_strategy_fn(|model| {
                let neigh = neighbors::build_neighbors_from_model(model);
                let operator =
                    operator_library::make_multi_armed_bandit_compound_operator::<
                        i64,
                        DefaultCostEvaluator,
                        ChaCha8Rng,
                    >(&OperatorSelectionConfig::default(), &neigh, 0.8, 2.0);

                let filter_stack = NeighborhoodFilterStack::empty();
                let mh =
                    GreedyDescentMetaheuristic::<i64, DefaultCostEvaluator, ChaCha8Rng>::default();
                let improving_db = Box::new(MetaheuristicLocalSearch::new(
                    Box::new(operator),
                    filter_stack,
                    mh,
                ));

                let ruin_cfg = RuinSelectionConfig {
                    use_random_subset: true,
                    use_time_band: true,
                    use_same_berth_block: false,
                    use_random_walk: false,
                    use_related_neighbors: false,
                    random_subset_k: 25,
                    time_band_len: 120,
                    same_berth_block_len: 0,
                    random_walk_steps: 0,
                    random_walk_same_berth_bias: 0.0,
                    related_k: 0,
                };
                let repair_cfg = RepairSelectionConfig {
                    use_cheapest_insertion: true,
                    use_best_fit_by_slack: false,
                    use_regret_insertion: true,
                };
                let perturb_db = lns_library::make_random_ruin_repair_perturb_pair(
                    &ruin_cfg,
                    &repair_cfg,
                    &neigh,
                );

                let evaluator = DefaultCostEvaluator;
                let rng = ChaCha8Rng::seed_from_u64(200);

                Box::new(strategies::ils::IteratedLocalSearchStrategy::new(
                    strategies::ils::IteratedLocalSearchConfig {
                        max_local_stagnation_steps: 8_000,
                        max_local_steps: None,
                        acceptance_criterion: Box::new(
                            strategies::ils::GreedyDescentAcceptanceCriterion::new(),
                        ),
                        improving_decision_builder: improving_db,
                        perturbing_decision_builder: Box::new(perturb_db),
                        evaluator,
                        rng,
                        refetch_incumbent_after_outer_stagnation: Some(12),
                        refetch_incumbent_after_plan_generation: Some(1_500),
                        reset_builders_on_refetch: false,
                    },
                ))
            })
            // Strategy 6: ILS (Always accept) with full ruin mix but reduced operator set
            .with_strategy_fn(|model| {
                let neigh = neighbors::build_neighbors_from_model(model);
                let operator = operator_library::make_multi_armed_bandit_compound_operator::<
                    i64,
                    DefaultCostEvaluator,
                    ChaCha8Rng,
                >(
                    &OperatorSelectionConfig {
                        use_relocate: true,
                        use_slide: true,
                        use_swap: false,
                        use_push_insert: true,
                        use_assign: true,
                        use_unassign: true,
                        use_or_opt: false,
                    },
                    &neigh,
                    0.9,
                    1.3,
                );

                let filter_stack = NeighborhoodFilterStack::empty();
                let mh =
                    GreedyDescentMetaheuristic::<i64, DefaultCostEvaluator, ChaCha8Rng>::default();
                let improving_db = Box::new(MetaheuristicLocalSearch::new(
                    Box::new(operator),
                    filter_stack,
                    mh,
                ));

                let ruin_cfg = RuinSelectionConfig::default(); // enable all
                let repair_cfg = RepairSelectionConfig {
                    use_cheapest_insertion: true,
                    use_best_fit_by_slack: true,
                    use_regret_insertion: false,
                };

                let perturb_db = lns_library::make_random_ruin_repair_perturb_pair(
                    &ruin_cfg,
                    &repair_cfg,
                    &neigh,
                );

                let evaluator = DefaultCostEvaluator;
                let rng = ChaCha8Rng::seed_from_u64(201);

                Box::new(strategies::ils::IteratedLocalSearchStrategy::new(
                    strategies::ils::IteratedLocalSearchConfig {
                        max_local_stagnation_steps: 9_000,
                        max_local_steps: None,
                        acceptance_criterion: Box::new(
                            strategies::ils::AlwaysAcceptAcceptanceCriterion::new(),
                        ),
                        improving_decision_builder: improving_db,
                        perturbing_decision_builder: Box::new(perturb_db),
                        evaluator,
                        rng,
                        refetch_incumbent_after_outer_stagnation: Some(18),
                        refetch_incumbent_after_plan_generation: Some(2_200),
                        reset_builders_on_refetch: false,
                    },
                ))
            })
            // Strategy 7: ILS (Greedy) with related-neighbors heavy ruin emphasis
            .with_strategy_fn(|model| {
                let neigh = neighbors::build_neighbors_from_model(model);
                let operator =
                    operator_library::make_multi_armed_bandit_compound_operator::<
                        i64,
                        DefaultCostEvaluator,
                        ChaCha8Rng,
                    >(&OperatorSelectionConfig::default(), &neigh, 0.85, 1.9);

                let filter_stack = NeighborhoodFilterStack::empty();
                let mh =
                    GreedyDescentMetaheuristic::<i64, DefaultCostEvaluator, ChaCha8Rng>::default();
                let improving_db = Box::new(MetaheuristicLocalSearch::new(
                    Box::new(operator),
                    filter_stack,
                    mh,
                ));

                let ruin_cfg = RuinSelectionConfig {
                    use_random_subset: false,
                    use_time_band: true,
                    use_same_berth_block: true,
                    use_random_walk: true,
                    use_related_neighbors: true,
                    random_subset_k: 0,
                    time_band_len: 180,
                    same_berth_block_len: 50,
                    random_walk_steps: 60,
                    random_walk_same_berth_bias: 0.9,
                    related_k: 25,
                };
                let repair_cfg = RepairSelectionConfig::default();

                let perturb_db = lns_library::make_random_ruin_repair_perturb_pair(
                    &ruin_cfg,
                    &repair_cfg,
                    &neigh,
                );

                let evaluator = DefaultCostEvaluator;
                let rng = ChaCha8Rng::seed_from_u64(202);

                Box::new(strategies::ils::IteratedLocalSearchStrategy::new(
                    strategies::ils::IteratedLocalSearchConfig {
                        max_local_stagnation_steps: 11_000,
                        max_local_steps: None,
                        acceptance_criterion: Box::new(
                            strategies::ils::GreedyDescentAcceptanceCriterion::new(),
                        ),
                        improving_decision_builder: improving_db,
                        perturbing_decision_builder: Box::new(perturb_db),
                        evaluator,
                        rng,
                        refetch_incumbent_after_outer_stagnation: Some(15),
                        refetch_incumbent_after_plan_generation: Some(3_000),
                        reset_builders_on_refetch: false,
                    },
                ))
            })
            // Strategy: ILS BigShake #1 (AlwaysAccept, heavy ruin, quick restarts)
            .with_strategy_fn(|model| {
                use berth_alloc_solver::search::metaheuristic_library::greedy_descent::GreedyDescentMetaheuristic;

                let neigh = neighbors::build_neighbors_from_model(model);

                // All ops enabled, exploratory MAB
                let operator = operator_library::make_multi_armed_bandit_compound_operator::<
                    i64,
                    DefaultCostEvaluator,
                    ChaCha8Rng,
                >(&OperatorSelectionConfig::all_enabled(), &neigh, 0.6, 2.2);

                // Local improving builder with greedy metaheuristic
                let filter_stack = NeighborhoodFilterStack::empty();
                let mh = GreedyDescentMetaheuristic::<i64, DefaultCostEvaluator, ChaCha8Rng>::default();
                let improving_db = Box::new(MetaheuristicLocalSearch::new(
                    Box::new(operator),
                    filter_stack,
                    mh,
                ));

                // BIG ruin: ~35% subset, long bands/walks, related neighbors strong
                let nflex = model.flexible_requests_len();
                let subset_k = std::cmp::max(25, ((nflex as f64) * 0.35).ceil() as usize);

                let ruin_cfg = RuinSelectionConfig {
                    use_random_subset: true,
                    use_time_band: true,
                    use_same_berth_block: true,
                    use_random_walk: true,
                    use_related_neighbors: true,
                    random_subset_k: subset_k,
                    time_band_len: 240,
                    same_berth_block_len: 80,
                    random_walk_steps: 120,
                    random_walk_same_berth_bias: 0.92,
                    related_k: 40,
                };
                let repair_cfg = RepairSelectionConfig {
                    use_cheapest_insertion: true,
                    use_best_fit_by_slack: true,
                    use_regret_insertion: true,
                };

                let perturb_db =
                    lns_library::make_random_ruin_repair_perturb_pair(&ruin_cfg, &repair_cfg, &neigh);

                let evaluator = DefaultCostEvaluator;
                let rng = ChaCha8Rng::seed_from_u64(3101);

                Box::new(strategies::ils::IteratedLocalSearchStrategy::new(
                    strategies::ils::IteratedLocalSearchConfig {
                        max_local_stagnation_steps: 4_000,
                        max_local_steps: None,
                        acceptance_criterion: Box::new(
                            strategies::ils::AlwaysAcceptAcceptanceCriterion::new(),
                        ),
                        improving_decision_builder: improving_db,
                        perturbing_decision_builder: Box::new(perturb_db),
                        evaluator,
                        rng,

                        // Refetch fast when stuck
                        refetch_incumbent_after_outer_stagnation: Some(8),
                        refetch_incumbent_after_plan_generation: Some(1_500),
                        reset_builders_on_refetch: false,
                    },
                ))
            })

            // Strategy: ILS BigShake #2 (Greedy accept, even heavier related+walk)
            .with_strategy_fn(|model| {
                let neigh = neighbors::build_neighbors_from_model(model);

                // Tilt MAB more exploratory
                let operator = operator_library::make_multi_armed_bandit_compound_operator::<
                    i64,
                    DefaultCostEvaluator,
                    ChaCha8Rng,
                >(&OperatorSelectionConfig::all_enabled(), &neigh, 0.55, 2.4);

                let filter_stack = NeighborhoodFilterStack::empty();
                let mh = GreedyDescentMetaheuristic::<i64, DefaultCostEvaluator, ChaCha8Rng>::default();
                let improving_db = Box::new(MetaheuristicLocalSearch::new(
                    Box::new(operator),
                    filter_stack,
                    mh,
                ));

                let nflex = model.flexible_requests_len();
                let subset_k = std::cmp::max(30, ((nflex as f64) * 0.40).ceil() as usize);

                let ruin_cfg = RuinSelectionConfig {
                    use_random_subset: true,
                    use_time_band: true,
                    use_same_berth_block: true,
                    use_random_walk: true,
                    use_related_neighbors: true,
                    random_subset_k: subset_k,
                    time_band_len: 280,
                    same_berth_block_len: 100,
                    random_walk_steps: 150,
                    random_walk_same_berth_bias: 0.94,
                    related_k: 50,
                };
                let repair_cfg = RepairSelectionConfig::default();

                let perturb_db =
                    lns_library::make_random_ruin_repair_perturb_pair(&ruin_cfg, &repair_cfg, &neigh);

                let evaluator = DefaultCostEvaluator;
                let rng = ChaCha8Rng::seed_from_u64(3102);

                Box::new(strategies::ils::IteratedLocalSearchStrategy::new(
                    strategies::ils::IteratedLocalSearchConfig {
                        max_local_stagnation_steps: 5_000,
                        max_local_steps: None,
                        acceptance_criterion: Box::new(
                            strategies::ils::GreedyDescentAcceptanceCriterion::new(),
                        ),
                        improving_decision_builder: improving_db,
                        perturbing_decision_builder: Box::new(perturb_db),
                        evaluator,
                        rng,

                        // Still refetch aggressively to re-center on incumbent
                        refetch_incumbent_after_outer_stagnation: Some(10),
                        refetch_incumbent_after_plan_generation: Some(2_000),
                        reset_builders_on_refetch: false,
                    },
                ))
            })

            // Strategy: ILS BigShake #3 (AlwaysAccept, hyper-destructive)
            .with_strategy_fn(|model| {
                let neigh = neighbors::build_neighbors_from_model(model);

                let operator = operator_library::make_multi_armed_bandit_compound_operator::<
                    i64,
                    DefaultCostEvaluator,
                    ChaCha8Rng,
                >(&OperatorSelectionConfig::all_enabled(), &neigh, 0.5, 2.5);

                let filter_stack = NeighborhoodFilterStack::empty();
                let mh = GreedyDescentMetaheuristic::<i64, DefaultCostEvaluator, ChaCha8Rng>::default();
                let improving_db = Box::new(MetaheuristicLocalSearch::new(
                    Box::new(operator),
                    filter_stack,
                    mh,
                ));

                let nflex = model.flexible_requests_len();
                let subset_k = std::cmp::max(35, ((nflex as f64) * 0.45).ceil() as usize);

                let ruin_cfg = RuinSelectionConfig {
                    use_random_subset: true,
                    use_time_band: true,
                    use_same_berth_block: true,
                    use_random_walk: true,
                    use_related_neighbors: true,
                    random_subset_k: subset_k,
                    time_band_len: 320,
                    same_berth_block_len: 120,
                    random_walk_steps: 180,
                    random_walk_same_berth_bias: 0.95,
                    related_k: 60,
                };
                let repair_cfg = RepairSelectionConfig {
                    use_cheapest_insertion: false, // bias toward structured repairs
                    use_best_fit_by_slack: true,
                    use_regret_insertion: true,
                };

                let perturb_db =
                    lns_library::make_random_ruin_repair_perturb_pair(&ruin_cfg, &repair_cfg, &neigh);

                let evaluator = DefaultCostEvaluator;
                let rng = ChaCha8Rng::seed_from_u64(3103);

                Box::new(strategies::ils::IteratedLocalSearchStrategy::new(
                    strategies::ils::IteratedLocalSearchConfig {
                        max_local_stagnation_steps: 6_000,
                        max_local_steps: None,
                        acceptance_criterion: Box::new(
                            strategies::ils::AlwaysAcceptAcceptanceCriterion::new(),
                        ),
                        improving_decision_builder: improving_db,
                        perturbing_decision_builder: Box::new(perturb_db),
                        evaluator,
                        rng,

                        // Quick restarts to bounce basins
                        refetch_incumbent_after_outer_stagnation: Some(6),
                        refetch_incumbent_after_plan_generation: Some(1_250),
                        reset_builders_on_refetch: false,
                    },
                ))
            })

            // Strategy 8: SA (micro temperature) local micro-tuning; fast restart thresholds
            .with_strategy_fn(|model| {
                let cfg = strategies::sa::SimulatedAnnealingConfig {
                    operator_selection_config: OperatorSelectionConfig {
                        use_relocate: true,
                        use_slide: false,
                        use_swap: true,
                        use_push_insert: false,
                        use_assign: true,
                        use_unassign: true,
                        use_or_opt: true,
                    },
                    initial_temperature: 150.0,
                    step: 30,
                    allow_infeasible_uphill: false,
                    seed: 14,
                    memory_coefficient: 0.55,
                    exploration_coefficient: 1.1,
                    refetch_incumbent_after_stagnation: 2_500,
                    refetch_incumbent_after_plan_generation: Some(20_000),
                };
                strategies::sa::make_simulated_annealing_strategy(cfg, model)
            });

        let outcome = solver.solve(&problem);

        let runtime = t0.elapsed();
        let end_ts = Utc::now();

        let cost_opt = match outcome {
            Ok(Some(solution)) => {
                let c = solution.cost();
                tracing::info!(
                    "Finished [{}] {}: cost={}, runtime={:?}",
                    iteration,
                    file,
                    c,
                    runtime
                );
                Some(c)
            }
            _ => {
                tracing::error!("Failed [{}] {}: runtime={:?}", iteration, file, runtime);
                None
            }
        };

        results.push(RunRecord {
            iteration,
            filename: file,
            start_ts,
            end_ts,
            runtime_ms: runtime.as_millis(),
            cost: cost_opt,
        });
    }

    // Persist results
    let out_path = PathBuf::from("solver_results.json");
    match File::create(&out_path).and_then(|mut f| {
        let json = serde_json::to_string_pretty(&results).expect("serialize results");
        f.write_all(json.as_bytes())
    }) {
        Ok(()) => {
            tracing::info!(
                "Wrote {} run record(s) to {}",
                results.len(),
                out_path.display()
            );
        }
        Err(e) => {
            tracing::error!("Failed to write results to {}: {}", out_path.display(), e);
        }
    }
}
