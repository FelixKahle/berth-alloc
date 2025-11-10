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
    search::{
        eval::CostEvaluator,
        metaheuristic::{Metaheuristic, MetaheuristicContext},
    },
    state::plan::Plan,
};

/// Greedy descent metaheuristic:
/// - Accepts only strictly improving candidates (fewer unassigned OR same unassigned + lower cost).
/// - Continues searching while each completed round (a pass over the neighborhood) produced
///   at least one accepted improving plan.
/// - Terminates after a round with no improvement (local optimum reached).
///
/// Internal state tracks whether the current round has accepted an improving plan and whether
/// the metaheuristic is still active (i.e., not at local optimum).
pub struct GreedyDescentMetaheuristic<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    active: bool,
    enumerating_round: bool,
    improved_in_round: bool,
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

impl<T, C, R> GreedyDescentMetaheuristic<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            active: true,
            enumerating_round: false,
            improved_in_round: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, R> Default for GreedyDescentMetaheuristic<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R> Metaheuristic<T, C, R> for GreedyDescentMetaheuristic<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline]
    fn name(&self) -> &str {
        "GreedyDescent"
    }

    #[inline]
    fn reset(&mut self) {
        // Re-enable searching; next call to local_optimum_reached will start a fresh round.
        self.active = true;
        self.enumerating_round = false;
        self.improved_in_round = false;
    }

    /// Called as a gate before starting a round (first call),
    /// and again after finishing a round that yielded no accepted plan
    /// to decide whether to continue.
    ///
    /// Protocol inside MetaheuristicLocalSearch:
    /// 1. First call: if returns true, a round begins (neighbors enumerated).
    /// 2. After a round with zero acceptances, called again to decide continuation.
    /// 3. After a round with ≥1 acceptance, NOT called again until the caller asks for
    ///    the next plan (new search builder invocation), at which point it acts like (1).
    #[inline]
    fn local_optimum_reached<'e, 'r, 's, 'm, 'p>(
        &mut self,
        _context: MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>,
    ) -> bool {
        if !self.active {
            return false;
        }

        if !self.enumerating_round {
            // Starting a new round.
            self.enumerating_round = true;
            self.improved_in_round = false;
            return true;
        }

        // End-of-round evaluation (the search exhausted neighbors with no acceptance
        // OR we are being asked right after finishing a no-improvement round).
        if self.improved_in_round {
            // Allow continuation: start next round immediately.
            self.improved_in_round = false; // reset for the upcoming round
            return true;
        }

        // No improvement in the finished round -> terminate.
        self.active = false;
        self.enumerating_round = false;
        false
    }

    /// Accept only strictly improving plans:
    /// - If delta_unassigned < 0: fewer unassigned -> improvement.
    /// - Else if delta_unassigned == 0 and delta_cost < 0: cost improvement with same feasibility.
    /// - Otherwise reject.
    #[inline]
    fn accept_plan<'e, 'r, 's, 'm, 'p>(
        &mut self,
        _context: MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool {
        let du = plan.fitness_delta.delta_unassigned;
        let dc = plan.fitness_delta.delta_cost;

        let improving = du < 0 || (du == 0 && dc < 0);

        if improving {
            self.improved_in_round = true;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::DefaultCostEvaluator,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::{Fitness, FitnessDelta},
            plan::{DecisionVarPatch, Plan},
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{builder::ProblemBuilder, req::Request},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    type T = i64;

    #[inline]
    fn tp(v: T) -> TimePoint<T> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: T, b: T) -> TimeInterval<T> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: T) -> TimeDelta<T> {
        TimeDelta::new(v)
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn make_problem() -> Problem<T> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, T>::new(rid(1), iv(0, 100), 1, pt1).unwrap();
        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.build().unwrap()
    }

    fn make_state_eval(
        problem: &Problem<T>,
    ) -> (
        SolverModel<'_, T>,
        SolverState<'_, T>,
        DefaultCostEvaluator,
        ChaCha8Rng,
    ) {
        let model = SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fitness = Fitness::new(100, model.flexible_requests_len());
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        let eval = DefaultCostEvaluator;
        let rng = ChaCha8Rng::seed_from_u64(123);
        (model, state, eval, rng)
    }

    fn make_ctx<'e, 'r, 's, 'm, 'p>(
        model: &'m SolverModel<'p, T>,
        state: &'s SolverState<'p, T>,
        eval: &'e DefaultCostEvaluator,
        rng: &'r mut ChaCha8Rng,
    ) -> MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, DefaultCostEvaluator, ChaCha8Rng> {
        MetaheuristicContext::new(model, state, eval, rng)
    }

    #[test]
    fn test_accept_plan_improving_rules() {
        let problem = make_problem();
        let (model, state, eval, mut rng) = make_state_eval(&problem);
        let mut mh = GreedyDescentMetaheuristic::<T, _, _>::new();

        // Improving by fewer unassigned (delta_unassigned < 0)
        let plan_assign = Plan::new_delta(
            vec![DecisionVarPatch::new(
                model.index_manager().request_index(rid(1)).unwrap(),
                DecisionVar::assigned(model.index_manager().berth_index(bid(1)).unwrap(), tp(0)),
            )],
            crate::state::terminal::delta::TerminalDelta::empty(),
            FitnessDelta::new(10, -1), // cost +10, but fewer unassigned => improvement
        );
        assert!(mh.accept_plan(make_ctx(&model, &state, &eval, &mut rng), &plan_assign));

        // Improving by lower cost (delta_unassigned == 0, delta_cost < 0)
        let plan_cost = Plan::new_delta(
            vec![],
            crate::state::terminal::delta::TerminalDelta::empty(),
            FitnessDelta::new(-5, 0),
        );
        assert!(mh.accept_plan(make_ctx(&model, &state, &eval, &mut rng), &plan_cost));

        // Not improving (higher cost, same unassigned)
        let plan_worse = Plan::new_delta(
            vec![],
            crate::state::terminal::delta::TerminalDelta::empty(),
            FitnessDelta::new(5, 0),
        );
        assert!(!mh.accept_plan(make_ctx(&model, &state, &eval, &mut rng), &plan_worse));

        // Not improving (more unassigned)
        let plan_unassign = Plan::new_delta(
            vec![],
            crate::state::terminal::delta::TerminalDelta::empty(),
            FitnessDelta::new(-10, 1),
        );
        assert!(!mh.accept_plan(make_ctx(&model, &state, &eval, &mut rng), &plan_unassign));
    }

    #[test]
    fn test_local_optimum_progression() {
        let problem = make_problem();
        let (model, state, eval, mut rng) = make_state_eval(&problem);
        let mut mh = GreedyDescentMetaheuristic::<T, _, _>::new();

        // Start: should allow first round.
        assert!(mh.local_optimum_reached(make_ctx(&model, &state, &eval, &mut rng)));

        // Simulate a round with improvement (set flag via accept_plan)
        mh.improved_in_round = true;

        // Continuation call after improvement: should allow another round.
        assert!(mh.local_optimum_reached(make_ctx(&model, &state, &eval, &mut rng)));

        // Simulate a round with NO improvement:
        mh.improved_in_round = false;
        assert!(mh.active);
        assert!(mh.enumerating_round);

        // End-of-round check: should terminate.
        assert!(!mh.local_optimum_reached(make_ctx(&model, &state, &eval, &mut rng)));

        // Further calls should remain false until reset.
        assert!(!mh.local_optimum_reached(make_ctx(&model, &state, &eval, &mut rng)));

        // After reset, can start again.
        mh.reset();
        assert!(mh.local_optimum_reached(make_ctx(&model, &state, &eval, &mut rng)));
    }
}
