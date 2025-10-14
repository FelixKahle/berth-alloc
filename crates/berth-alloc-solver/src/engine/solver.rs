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
    engine::{greedy::GreedyOpening, traits::Opening},
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
    state::err::SolverModelBuildError,
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::prelude::{Problem, SolutionRef};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::vec;

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

    pub fn solve(&mut self) -> SolutionRef<'problem, T>
    where
        T: Send + Sync,
    {
        let opener = GreedyOpening;
        let _initial_state = opener.build(&self.solver_model);

        unimplemented!()
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
    use berth_alloc_model::problem::loader::ProblemLoader;

    // This test might take a bit longer, as it loads and parses all instances
    // in the `instances/` folder at the workspace root, and creates a SolverEngine
    // for each of them to ensure the model builds correctly. As the ProximityMap is build in
    // O(n^2) time, this might take a while for large instances.
    #[test]
    fn test_load_all_instances_from_workspace_root_instances_folder_and_create_engine() {
        use std::fs;
        use std::path::{Path, PathBuf};

        // Find the nearest ancestor that contains an `instances/` directory.
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

        // Gather all .txt files (ignore subdirs/other files).
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

            // Sanity checks: there should be at least one berth and one request in real instances.
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

            // Create the solver engine to ensure the model builds correctly.
            let _ = SolverEngine::new(
                EngineParams {
                    proximity_alpha: 0.5,
                },
                &problem,
            )
            .expect("Failed to create SolverEngine");
        }
    }
}
