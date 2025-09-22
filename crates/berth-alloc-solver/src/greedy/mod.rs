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

pub mod err;

use crate::{
    framework::{
        err::ProposeAssignmentError,
        planning::{BrandedRequest, PlanningContext},
        solver::ConstructionSolver,
        state::{FeasibleSolverState, IncompleteSolverState},
    },
    greedy::err::GreedyError,
    registry::ledger::Ledger,
    terminal::terminalocc::TerminalOccupancy,
};
use berth_alloc_core::prelude::{Cost, TimeDelta};
use num_traits::{CheckedAdd, CheckedSub};
use std::{cmp::Reverse, ops::Mul};

#[derive(Debug, Clone)]
pub struct GreedySolver<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> GreedySolver<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> Default for GreedySolver<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> std::fmt::Display for GreedySolver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GreedySolver")
    }
}

impl<T> ConstructionSolver<T> for GreedySolver<T>
where
    T: Copy + Ord + Mul<Output = Cost> + CheckedAdd + CheckedSub + Into<Cost>,
{
    type Error = GreedyError<T>;

    fn construct<'p>(
        &mut self,
        problem: &'p berth_alloc_model::prelude::Problem<T>,
    ) -> Result<FeasibleSolverState<'p, T>, Self::Error> {
        let terminal = TerminalOccupancy::new(problem.berths().iter());
        let ledger = Ledger::new(problem);
        let mut state = IncompleteSolverState::new(ledger, terminal);
        let planning_ctx = PlanningContext::new(state.ledger(), state.terminal_occupancy());

        let plan_res = planning_ctx.with_builder(|pb| {
            loop {
                let mut reqs: Vec<_> =
                    pb.with_explorer(|ex| ex.iter_unassigned_requests().collect());
                if reqs.is_empty() {
                    break;
                }

                let slack_of = |r: &BrandedRequest<'_, 'p, _, T>| -> Option<TimeDelta<T>> {
                    let window_len = r.req().feasible_window().length();
                    let min_pt = r.req().processing_times().values().copied().min()?;
                    Some(window_len - min_pt)
                };

                reqs.sort_by(|a, b| match (slack_of(a), slack_of(b)) {
                    (Some(sa), Some(sb)) => sa
                        .cmp(&sb)
                        .then_with(|| Reverse(a.req().weight()).cmp(&Reverse(b.req().weight())))
                        .then_with(|| a.req().id().cmp(&b.req().id())),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.req().id().cmp(&b.req().id()),
                });

                let mut placed_in_pass = 0usize;

                for req in reqs {
                    if slack_of(&req).is_none() {
                        continue;
                    }

                    let candidates: Vec<_> =
                        pb.with_explorer(|ex| ex.iter_free_for(req.clone()).collect());
                    if candidates.is_empty() {
                        continue;
                    }

                    'try_free: for free in candidates {
                        let start = free.interval().start();
                        match pb.propose_assignment(req.clone(), start, &free) {
                            Ok(_asg) => {
                                placed_in_pass += 1;
                                break 'try_free;
                            }
                            Err(ProposeAssignmentError::NotFree(_)) => {
                                continue;
                            }
                            Err(_other) => {
                                continue;
                            }
                        }
                    }
                }

                if placed_in_pass == 0 {
                    break;
                }
            }
        });

        let plan = match plan_res {
            Ok(p) => p,
            Err(e) => return Err(GreedyError::from(e)),
        };

        state.apply_plan(plan)?;
        let feasible: FeasibleSolverState<'p, T> = state.try_into()?;
        Ok(feasible)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_model::problem::loader::ProblemLoader;

    #[test]
    fn test_greedy_algorithm_can_solve_sample_instance() {
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

            // Lets solve it with the greedy solver.
            let mut solver = super::GreedySolver::<i64>::new();
            let _ = solver
                .construct(&problem)
                .unwrap_or_else(|e| panic!("Failed to solve instance {}: {e}", path.display()));
        }
    }
}
