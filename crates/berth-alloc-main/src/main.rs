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
use berth_alloc_solver::{
    framework::{
        solver::{ConstructionSolver, Solver},
        state::SolverStateView,
    },
    greedy::GreedySolver,
    matheuristic::{config::MatheuristicConfig, engine::MatheuristicEngine, oplib},
};
use std::path::{Path, PathBuf};
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
fn instances() -> impl Iterator<Item = Problem<i64>> {
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
        loader.from_path(&f).ok()
    })
}

#[allow(dead_code)]
fn enable_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE)
        .init();
}

fn main() {
    enable_tracing();

    // Solve the f200x15-02.txt problem with both solvers
    let inst_dir = find_instances_dir()
        .expect("Could not find an `instances/` directory in any ancestor of CARGO_MANIFEST_DIR");
    let f20015_02 = inst_dir.join("f200x15-02.txt");
    let loader = ProblemLoader::default();
    let problem = loader.from_path(&f20015_02);
    let problem = match problem {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load problem from {}: {}", f20015_02.display(), e);
            return;
        }
    };
    println!("Loaded problem with {} requests", problem.request_count());
    let solver_state = GreedySolver::<i64>::new()
        .construct(&problem)
        .expect("GreedySolver failed to construct a solution");
    let feasible = solver_state.is_feasible();
    let cost = solver_state.cost();
    println!(
        "GreedySolver produced a {} solution with cost {}",
        if feasible { "feasible" } else { "infeasible" },
        cost
    );

    let construction_solver = GreedySolver::<i64>::new();
    let mut meta_solver = MatheuristicEngine::with_defaults(
        MatheuristicConfig::default(),
        oplib::prelude::op_list::<i64>(&problem),
        construction_solver,
    );
    let solution = meta_solver.solve(&problem).expect("MetaSolver failed");
    // Print None or the cost!
    println!(
        "MetaSolver produced solution: {:?}",
        solution.map(|s| s.cost())
    );
}
