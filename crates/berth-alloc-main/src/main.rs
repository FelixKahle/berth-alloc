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

use berth_alloc_model::problem::asg::AssignmentView;
use berth_alloc_model::problem::req::RequestView;
use berth_alloc_model::{prelude::SolutionView, problem::loader::ProblemLoader};
use berth_alloc_solver::engine::solver::{EngineParams, SolverEngine};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Duration;

fn find_instances_dir() -> Option<PathBuf> {
    // Start from crate manifest dir of main crate
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

fn main() {
    // Each instance gets 20 seconds
    let budget_per_instance = Duration::from_secs(60);

    let instances_dir =
        find_instances_dir().expect("Could not locate `instances/` directory in ancestors.");

    let mut instance_files: Vec<PathBuf> = std::fs::read_dir(&instances_dir)
        .expect("read_dir(instances) failed")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && e.path().extension().map(|x| x == "txt").unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    instance_files.sort();

    if instance_files.is_empty() {
        eprintln!(
            "No .txt instance files found in {}",
            instances_dir.display()
        );
        return;
    }

    let loader = ProblemLoader::default();

    println!(
        "Found {} instance(s) in {}. Time budget per instance: {:?}",
        instance_files.len(),
        instances_dir.display(),
        budget_per_instance
    );

    for path in instance_files {
        println!("------------------------------------------------------------");
        println!("Instance: {}", path.display());
        let problem = match loader.from_path(&path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to load {}: {}", path.display(), e);
                continue;
            }
        };

        println!(
            " Berths: {} | Flexible requests: {} | Fixed assignments: {}",
            problem.berths().iter().count(),
            problem.flexible_requests().iter().count(),
            problem.fixed_assignments().iter().count()
        );

        let mut engine = match SolverEngine::new(
            EngineParams {
                proximity_alpha: 0.5,
            },
            &problem,
        ) {
            Ok(e) => e,
            Err(e) => {
                eprintln!(" Engine build failed: {e}");
                continue;
            }
        };

        // Use solve_with_time_budget (20s). If not yet added, call the normal solve().
        let solution = engine.solve_with_time_budget(budget_per_instance);

        let total_cost = solution.cost();
        println!(
            " Solution: flex_assignments={} total_cost={}",
            solution.flexible_assignments().iter().count(),
            total_cost
        );

        for asg in solution.flexible_assignments().iter() {
            let rid = asg.request().id();
            let bid = asg.berth().id();
            let start = asg.start_time();
            let end = asg.end_time();
            let cost = asg.cost();
            println!(
                "  [Flexible]    Request {} -> Berth {} | Time: {}..{} | Cost: {}",
                rid, bid, start, end, cost
            );
        }

        exit(0);
    }
}
