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
    engine::{
        context::{EngineContext, SearchContext},
        greedy::GreedyOpening,
        search::{SAParams, Search},
        traits::Opening,
    },
    model::{
        neighborhood::{ProximityMap, ProximityMapParameter},
        solver_model::SolverModel,
    },
    scheduling::{
        greedy::GreedyScheduler, pipeline::SchedulingPipeline, tightener::BoundsTightener,
    },
    search::{
        filter::{feasible_berth_filter::FeasibleBerthFilter, filter_stack::FilterStack},
        operator_lib::{
            assign::AssignUnassignedFirstFeasible,
            cross_exchange::CrossExchangeBestImprovement,
            or_opt::OrOptKNeighborsBestImprovement,
            regret::RegretKInsert,
            relocate::RelocateNeighborsBestImprovement,
            ruin::RuinRandomSegment,
            shaw::{ShawDestroyPackInsert, ShawPackParams},
            swap::{RandomSwapNeighborsBlind, SwapNeighborsBestImprovement},
            two_opt_reverse::TwoOptReverseBestImprovement,
        },
        pertubation_lib::{
            greedy::GreedyRuinRepair, insertion::BestInsertionRuinRepair,
            nuke::NukeChainRuinRepair, walk::RandomWalkRuinRepair,
        },
    },
    state::{
        chain_set::index::NodeIndex, err::SolverModelBuildError, search_state::SearchSnapshot,
    },
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::prelude::{Problem, SolutionRef};
use num_traits::{CheckedAdd, CheckedSub, SaturatingSub, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    convert::TryInto,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    time::Duration,
};

/// Runtime-tunable engine knobs.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Target number of worker threads. Clamped to [1, available_parallelism()].
    pub num_workers: usize,
    /// Total wall-clock budget for the solve (shared by workers).
    pub total_time: Duration,
    /// Search objective mixture parameter (passed to SearchContext).
    pub lambda: f64,

    /// Proximity alpha used to build the proximity map.
    pub proximity_alpha: f64,

    /// Operator budgets / limits
    pub relocate_scan_cap: Option<NonZeroUsize>,
    pub cross_exchange_scan_cap: Option<usize>,

    /// Perturbation (ruin/repair) settings
    pub greedy_ruin_remove_k: usize,
    pub greedy_ruin_scan_cap: Option<usize>,

    /// Simulated annealing parameters (per worker)
    pub sa_t_start: f64,
    pub sa_t_end: f64,
    pub sa_randomize_ops: f32,
    /// Trigger a perturbation after this many non-improving iterations (Search reads this).
    /// NOTE: keep here so callers can configure it. Your Search::SAParams should have this field.
    pub sa_stagnation_iters: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            total_time: Duration::from_secs(1),
            lambda: 1.0,
            proximity_alpha: 1.0,
            relocate_scan_cap: Some(NonZeroUsize::new(50_000).unwrap()),
            cross_exchange_scan_cap: Some(50_000),
            greedy_ruin_remove_k: 20,
            greedy_ruin_scan_cap: Some(1_000),
            sa_t_start: 500.0,
            sa_t_end: 10.0,
            sa_randomize_ops: 0.85,
            sa_stagnation_iters: 500,
        }
    }
}

pub struct EngineParams {
    pub proximity_alpha: f64,
}

#[derive(Debug)]
pub struct SolverEngine<'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub,
{
    solver_model: SolverModel<'problem, T>,
    proximity_map: ProximityMap,
    pipeline: SchedulingPipeline<T, GreedyScheduler>,
    filter_stack: FilterStack<T>,
}

impl<'problem, T> SolverEngine<'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost> + 'static,
{
    pub fn new(
        params: EngineParams,
        problem: &'problem Problem<T>,
    ) -> Result<Self, SolverModelBuildError>
    where
        T: std::fmt::Debug + Zero + Send + Sync,
    {
        let solver_model = SolverModel::from_problem(problem)?;
        let proximity_map = ProximityMap::build(
            &solver_model,
            ProximityMapParameter::new(params.proximity_alpha),
        );

        // Default pipeline, filter stack and operators.
        let pipeline = SchedulingPipeline::from_propagators([BoundsTightener], GreedyScheduler);
        let filter_stack = FilterStack::with_filters(vec![Box::new(FeasibleBerthFilter)]);

        Ok(Self {
            solver_model,
            proximity_map,
            pipeline,
            filter_stack,
        })
    }

    /// Back-compat helper: keep your old API but drive it through EngineConfig.
    pub fn solve_with_time_budget(&mut self, total_budget: Duration) -> SolutionRef<'problem, T>
    where
        T: Send + Sync + Zero + SaturatingSub,
    {
        let mut cfg = EngineConfig::default();
        cfg.total_time = total_budget;
        cfg.proximity_alpha = 1.0;
        self.solve_with_config(cfg)
    }

    /// Main entry: run the multi-worker SA+operators+perturbation search with the given config.
    pub fn solve_with_config(&mut self, cfg: EngineConfig) -> SolutionRef<'problem, T>
    where
        T: Send + Sync + Zero + SaturatingSub,
    {
        // Build an initial solution for fallback/export (and for parity).
        let opener = GreedyOpening;
        let initial_state = opener.build(&self.solver_model);

        // Clamp worker count.
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_workers = cfg.num_workers.clamp(1, hw);

        // Even split the time budget per worker (keeps things simple/deterministic).
        let per_worker_time = cfg.total_time;

        let lambda = cfg.lambda;

        // Collect worker snapshots.
        let mut snapshots: Vec<SearchSnapshot<T>> = Vec::with_capacity(num_workers);

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_workers);

            for tid in 0..num_workers {
                // Immutable refs we can share into the thread.
                let model_ref = &self.solver_model;
                let prox_ref = &self.proximity_map;
                let pipe_ref = &self.pipeline;
                let filters_ref = &self.filter_stack;

                // Thread-local copy of config values we need.
                let cfg_t = cfg.clone();

                handles.push(scope.spawn(move || {
                    // Fresh opening per worker to avoid contention.
                    let state = GreedyOpening.build(model_ref);

                    // Contexts
                    let engine_context =
                        EngineContext::new(model_ref, prox_ref, pipe_ref, filters_ref);
                    let mut search_ctx = SearchContext::new(&engine_context, state, lambda);

                    // Neighborhood helper accessors using proximity lists.
                    let get_outgoing = {
                        let lists = prox_ref.any_feasibleish().lists(); // &'engine _
                        Box::new(move |node: NodeIndex, _start: NodeIndex| {
                            lists.outgoing_for(node).unwrap_or(&[])
                        })
                    };
                    let get_incoming = {
                        let lists = prox_ref.any_feasibleish().lists();
                        Box::new(move |node: NodeIndex, _start: NodeIndex| {
                            lists.incoming_for(node).unwrap_or(&[])
                        })
                    };

                    // ----- Operators -----
                    // ----- Operators -----
                    // 1) Relocate neighbors (best improvement)
                    search_ctx.operators_mut().add_operator(Box::new(
                        RelocateNeighborsBestImprovement::new(
                            /*same_chain_only=*/ false,
                            Box::new(move || cfg_t.relocate_scan_cap),
                            Some(get_outgoing.clone()),
                            Some(get_incoming.clone()),
                            None,
                        ),
                    ));

                    // 2) Assign any unassigned node (first feasible)
                    search_ctx
                        .operators_mut()
                        .add_operator(Box::new(AssignUnassignedFirstFeasible::new()));

                    // 3) Swap neighbors (best improvement)
                    search_ctx.operators_mut().add_operator(Box::new(
                        SwapNeighborsBestImprovement::new(
                            /*same_chain_only=*/ false,
                            Box::new(move || cfg_t.relocate_scan_cap),
                            Some(get_outgoing.clone()),
                            Some(get_incoming.clone()),
                            None,
                        ),
                    ));

                    // 4) Random-blind swap (lightweight diversification)
                    search_ctx.operators_mut().add_operator(Box::new(
                        RandomSwapNeighborsBlind::new(
                            /*same_chain_only=*/ false,
                            std::num::NonZeroUsize::new(256).unwrap(),
                            Some(get_outgoing.clone()),
                            Some(get_incoming.clone()),
                            None,
                            Box::new(|n| {
                                if n <= 1 {
                                    0
                                } else {
                                    rand::rng().random_range(0..n)
                                }
                            }),
                        ),
                    ));

                    search_ctx.operators_mut().add_operator(Box::new(
                        RandomSwapNeighborsBlind::new(
                            /*same_chain_only=*/ true,
                            std::num::NonZeroUsize::new(256).unwrap(),
                            Some(get_outgoing.clone()),
                            Some(get_incoming.clone()),
                            None,
                            Box::new(|n| {
                                if n <= 1 {
                                    0
                                } else {
                                    rand::rng().random_range(0..n)
                                }
                            }),
                        ),
                    ));

                    // 5) Cross-exchange (best improvement)
                    search_ctx.operators_mut().add_operator(Box::new(
                        CrossExchangeBestImprovement::new(
                            /*same_chain_only=*/ false,
                            cfg_t.cross_exchange_scan_cap,
                        ),
                    ));

                    search_ctx.operators_mut().add_operator(Box::new(
                        CrossExchangeBestImprovement::new(
                            /*same_chain_only=*/ true,
                            cfg_t.cross_exchange_scan_cap,
                        ),
                    ));

                    // 6) Or-opt (block relocate) with small K
                    search_ctx.operators_mut().add_operator(Box::new(
                        OrOptKNeighborsBestImprovement::new(
                            /*same_chain_only=*/ false,
                            /*max_k=*/ 3,
                            Box::new(move || cfg_t.relocate_scan_cap),
                            Some(get_outgoing.clone()),
                            Some(get_incoming.clone()),
                            None,
                        ),
                    ));

                    search_ctx.operators_mut().add_operator(Box::new(
                        OrOptKNeighborsBestImprovement::new(
                            /*same_chain_only=*/ true,
                            /*max_k=*/ 3,
                            Box::new(move || cfg_t.relocate_scan_cap),
                            Some(get_outgoing.clone()),
                            Some(get_incoming.clone()),
                            None,
                        ),
                    ));

                    // 7) 2-opt reverse (intra-chain)
                    search_ctx.operators_mut().add_operator(Box::new(
                        TwoOptReverseBestImprovement::new(
                            /*same_chain_only=*/ false,
                            cfg_t.cross_exchange_scan_cap,
                        ),
                    ));

                    search_ctx
                        .pertubations_mut()
                        .push(Box::new(BestInsertionRuinRepair::new(
                            40,
                            None,
                            GreedyScheduler,
                        )));

                    search_ctx
                        .pertubations_mut()
                        .push(Box::new(BestInsertionRuinRepair::new(
                            10,
                            Some(10_000),
                            GreedyScheduler,
                        )));

                    // 9) Regret-K insert to reinsert isolated nodes more selectively
                    search_ctx
                        .operators_mut()
                        .add_operator(Box::new(RegretKInsert::new(
                            std::num::NonZeroUsize::new(2).unwrap(),
                            cfg_t.greedy_ruin_scan_cap,
                            GreedyScheduler,
                        )));

                    search_ctx
                        .pertubations_mut()
                        .push(Box::new(RandomWalkRuinRepair::new(
                            0.2,
                            Some(10_000),
                            GreedyScheduler,
                        )));

                    search_ctx
                        .pertubations_mut()
                        .push(Box::new(RandomWalkRuinRepair::new(
                            0.05,
                            Some(10_000),
                            GreedyScheduler,
                        )));

                    // ----- SA params (per worker) -----
                    let sa_params = SAParams {
                        time_limit: per_worker_time,
                        t_start: cfg_t.sa_t_start,
                        t_end: cfg_t.sa_t_end,
                        seed: 0xBEEF_CA_FE ^ (tid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                        randomize_ops: cfg_t.sa_randomize_ops,
                        // This field should exist in your Search::SAParams
                        stagnation_iters: cfg_t.sa_stagnation_iters,
                    };

                    // Run search
                    let search = Search::new(search_ctx);
                    search.run_sa_time_cap(sa_params)
                }));
            }

            // Join
            for h in handles {
                if let Ok(snap) = h.join() {
                    snapshots.push(snap);
                }
            }
        });

        // Pick best snapshot (lowest true cost)
        let best_snapshot_opt = snapshots
            .into_iter()
            .min_by(|a, b| a.true_cost.cmp(&b.true_cost));

        let best_snapshot = match best_snapshot_opt {
            Some(s) => s,
            None => {
                // Fallback to exporting the opening if nothing came back (shouldn’t happen).
                return initial_state
                    .try_into()
                    .expect("initial state export must succeed");
            }
        };

        best_snapshot
            .try_into()
            .expect("snapshot export must succeed")
    }

    #[inline]
    pub fn solver_model(&self) -> &SolverModel<'problem, T> {
        &self.solver_model
    }

    #[inline]
    pub fn proximity_map(&self) -> &ProximityMap {
        &self.proximity_map
    }

    #[inline]
    pub fn pipeline(&self) -> &SchedulingPipeline<T, GreedyScheduler> {
        &self.pipeline
    }

    #[inline]
    pub fn filter_stack(&self) -> &FilterStack<T> {
        &self.filter_stack
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_model::{prelude::SolutionView, problem::loader::ProblemLoader};
    use std::path::{Path, PathBuf};

    #[test]
    fn test_load_all_instances_from_workspace_root_instances_folder_and_create_engine() {
        use std::fs;

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

        let inst_dir = find_instances_dir().expect(
            "Could not find an `instances/` directory in any ancestor of CARGO_MANIFEST_DIR",
        );

        let mut files: Vec<PathBuf> = fs::read_dir(&inst_dir)
            .expect("read_dir(instances) failed")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                    && e.path().extension().map(|x| x == "txt").unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        files.sort();
        assert!(
            !files.is_empty(),
            "No .txt instance files found in {}",
            inst_dir.display()
        );

        let loader = ProblemLoader::default();

        for path in files {
            eprintln!("Loading instance: {}", path.display());
            let problem = loader
                .from_path(&path)
                .unwrap_or_else(|e| panic!("Failed to load {}: {e}", path.display()));

            assert!(
                !problem.berths().is_empty(),
                "No berths parsed in {}",
                path.display()
            );
            assert!(
                !problem.flexible_requests().is_empty(),
                "No flexible requests parsed in {}",
                path.display()
            );

            let mut engine = SolverEngine::new(
                EngineParams {
                    proximity_alpha: 0.5,
                },
                &problem,
            )
            .expect("Failed to create SolverEngine");

            // Use config path (keeps old behavior too)
            let mut cfg = EngineConfig::default();
            cfg.total_time = Duration::from_millis(1000);
            cfg.num_workers = 1;

            let solution = engine.solve_with_config(cfg);
            // Basic sanity: solution flexible assignments count <= requests.
            assert!(
                solution.flexible_assignments().iter().count()
                    <= problem.flexible_requests().iter().count()
            );
        }
    }
}
