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
        operator::traits::NeighborhoodOperator,
        operator_library::swap::SwapSuccessorsFirstImprovement,
    },
    state::{err::SolverModelBuildError, search_state::SearchSnapshot},
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::prelude::{Problem, SolutionRef};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::{convert::TryInto, thread};

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
    operators: Vec<Box<dyn NeighborhoodOperator<T>>>,
}

impl<'problem, T> SolverEngine<'problem, T>
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
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
        let operators: Vec<Box<dyn NeighborhoodOperator<T>>> =
            vec![Box::new(SwapSuccessorsFirstImprovement::default())];

        Ok(Self {
            solver_model,
            proximity_map,
            pipeline,
            filter_stack,
            operators,
        })
    }

    /// Solve:
    /// - Build opening solution
    /// - Run multi-threaded simulated annealing searches
    /// - Pick best snapshot
    /// - Convert to SolutionRef
    pub fn solve(&mut self) -> SolutionRef<'problem, T>
    where
        T: Send + Sync + Zero,
    {
        let opener = GreedyOpening;
        let initial_state = opener.build(&self.solver_model);

        // If no operators, just export opening.
        if self.operators.is_empty() {
            return initial_state
                .try_into()
                .expect("opening state should export to solution");
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        let time_per_thread_ms = 400_u64;
        let lambda = 1.0_f64;

        // Collect snapshots from threads
        let mut snapshots: Vec<SearchSnapshot<T>> = Vec::with_capacity(num_threads);

        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_threads);
            for tid in 0..num_threads {
                let model_ref = &self.solver_model;
                let prox_ref = &self.proximity_map;
                let pipe_ref = &self.pipeline;
                let filters_ref = &self.filter_stack;
                let operators_len = self.operators.len();

                handles.push(scope.spawn(move || {
                    // Fresh initial state per thread
                    let opener = GreedyOpening;
                    let state = opener.build(model_ref);

                    let engine_context =
                        EngineContext::new(model_ref, prox_ref, pipe_ref, filters_ref);
                    let mut search_context = SearchContext::new(&engine_context, state, lambda);

                    // Populate operator pool
                    for _ in 0..operators_len {
                        search_context
                            .operators_mut()
                            .add_operator(Box::new(SwapSuccessorsFirstImprovement::default()));
                    }

                    let sa_params = SAParams {
                        time_limit: std::time::Duration::from_millis(time_per_thread_ms),
                        t_start: 300.0,
                        t_end: 1e-3,
                        seed: 0xABCDEF55_u64 ^ (tid as u64).wrapping_mul(0x9E3779B97F4A7C15),
                        randomize_ops: 0.75,
                    };

                    let search = Search::new(search_context);
                    search.run_sa_time_cap(sa_params) // returns SearchSnapshot<T>
                }));
            }

            for h in handles {
                if let Ok(snap) = h.join() {
                    snapshots.push(snap);
                }
            }
        });

        // Choose best snapshot by true_cost
        let best_snapshot_opt = snapshots
            .into_iter()
            .min_by(|a, b| a.true_cost.cmp(&b.true_cost));

        let best_snapshot = match best_snapshot_opt {
            Some(s) => s,
            None => {
                // Fallback to opening conversion
                return initial_state
                    .try_into()
                    .expect("initial state export must succeed");
            }
        };

        best_snapshot
            .try_into()
            .expect("best snapshot state export must succeed")
    }

    pub fn solve_with_time_budget(
        &mut self,
        total_budget: std::time::Duration,
    ) -> SolutionRef<'problem, T>
    where
        T: Send + Sync + Zero,
    {
        // Scale existing solve(): we reuse logic but override per-thread ms.
        // Simple even split: each thread gets total_budget (we keep structure identical, just bump per-thread time).
        // For a more refined split: divide by number of threads *some factor.
        let opener = GreedyOpening;
        let initial_state = opener.build(&self.solver_model);

        if self.operators.is_empty() {
            return initial_state
                .try_into()
                .expect("opening state should export to solution");
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        let per_thread_time = total_budget; // each thread runs full budget (aggressive). Adjust if needed.
        let lambda = 1.0_f64;

        let mut snapshots: Vec<SearchSnapshot<T>> = Vec::with_capacity(num_threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_threads);
            for tid in 0..num_threads {
                let model_ref = &self.solver_model;
                let prox_ref = &self.proximity_map;
                let pipe_ref = &self.pipeline;
                let filters_ref = &self.filter_stack;
                let operators_len = self.operators.len();

                handles.push(scope.spawn(move || {
                    let opener = GreedyOpening;
                    let state = opener.build(model_ref);

                    let engine_context =
                        EngineContext::new(model_ref, prox_ref, pipe_ref, filters_ref);
                    let mut search_context = SearchContext::new(&engine_context, state, lambda);

                    for _ in 0..operators_len {
                        search_context
                            .operators_mut()
                            .add_operator(Box::new(SwapSuccessorsFirstImprovement::default()));
                    }

                    let sa_params = SAParams {
                        time_limit: per_thread_time,
                        t_start: 500.0,
                        t_end: 1e-3,
                        seed: 0xBEEF_CA_FE ^ (tid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                        randomize_ops: 0.85,
                    };

                    let search = Search::new(search_context);
                    search.run_sa_time_cap(sa_params)
                }));
            }

            for h in handles {
                if let Ok(snap) = h.join() {
                    snapshots.push(snap);
                }
            }
        });

        let best_snapshot_opt = snapshots
            .into_iter()
            .min_by(|a, b| a.true_cost.cmp(&b.true_cost));

        let best_snapshot = match best_snapshot_opt {
            Some(s) => s,
            None => {
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

    #[inline]
    pub fn operators(&self) -> &Vec<Box<dyn NeighborhoodOperator<T>>> {
        &self.operators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_model::{prelude::SolutionView, problem::loader::ProblemLoader};

    #[test]
    fn test_load_all_instances_from_workspace_root_instances_folder_and_create_engine() {
        use std::fs;
        use std::path::{Path, PathBuf};

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

            let solution = engine.solve();
            // Basic sanity: solution flexible assignments count <= requests.
            assert!(
                solution.flexible_assignments().iter().count()
                    <= problem.flexible_requests().iter().count()
            );
        }
    }
}
