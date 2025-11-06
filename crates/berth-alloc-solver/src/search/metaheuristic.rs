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
    model::solver_model::SolverModel,
    search::eval::CostEvaluator,
    state::{plan::Plan, solver_state::SolverState},
};

pub struct MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub model: &'m SolverModel<'p, T>,
    pub solver_state: &'s SolverState<'p, T>,
    pub evaluator: &'e C,
    pub rng: &'r mut R,
}

impl<'e, 'r, 's, 'm, 'p, T, C, R> MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        solver_state: &'s SolverState<'p, T>,
        evaluator: &'e C,
        rng: &'r mut R,
    ) -> Self {
        Self {
            model,
            solver_state,
            evaluator,
            rng,
        }
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn solver_state(&self) -> &'s SolverState<'p, T> {
        self.solver_state
    }

    #[inline]
    pub fn evaluator(&self) -> &'e C {
        self.evaluator
    }
}

pub trait Metaheuristic<T, C, R>: Send
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn local_optimum_reached<'e, 'r, 's, 'm, 'p>(
        &mut self,
        context: MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>,
    ) -> bool;

    fn accept_plan<'e, 'r, 's, 'm, 'p>(
        &mut self,
        context: MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool;
}

impl<T, C, R> std::fmt::Debug for dyn Metaheuristic<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic({})", self.name())
    }
}

impl<T, C, R> std::fmt::Display for dyn Metaheuristic<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic({})", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::{CostEvaluator, DefaultCostEvaluator},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::FitnessDelta,
            plan::{DecisionVarPatch, Plan},
            terminal::delta::TerminalDelta,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{builder::ProblemBuilder, req::Request},
    };
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
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

    fn make_basic_problem() -> Problem<i64> {
        // One berth [0,100)
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        // One flexible request window [0,100), weight 1, pt(10) on berth 1
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        let r10 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 100), 1, pt).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.build().unwrap()
    }

    fn make_model_state_eval(
        problem: &Problem<i64>,
    ) -> (
        SolverModel<'_, i64>,
        SolverState<'_, i64>,
        DefaultCostEvaluator,
    ) {
        let model = SolverModel::try_from(problem).expect("model should build");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let eval = DefaultCostEvaluator;
        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    // Dummy metaheuristic that returns preconfigured values and tracks calls.
    struct DummyMetaheuristic {
        name: &'static str,
        ret_local: bool,
        ret_accept: bool,
        calls_local: usize,
        calls_accept: usize,
    }

    impl DummyMetaheuristic {
        fn new(name: &'static str, ret_local: bool, ret_accept: bool) -> Self {
            Self {
                name,
                ret_local,
                ret_accept,
                calls_local: 0,
                calls_accept: 0,
            }
        }
        fn calls(&self) -> (usize, usize) {
            (self.calls_local, self.calls_accept)
        }
    }

    impl Metaheuristic<i64, DefaultCostEvaluator, ChaCha8Rng> for DummyMetaheuristic {
        fn name(&self) -> &str {
            self.name
        }

        fn local_optimum_reached<'e, 'r, 's, 'm, 'p>(
            &mut self,
            _context: MetaheuristicContext<
                'e,
                'r,
                's,
                'm,
                'p,
                i64,
                DefaultCostEvaluator,
                ChaCha8Rng,
            >,
        ) -> bool {
            self.calls_local += 1;
            self.ret_local
        }

        fn accept_plan<'e, 'r, 's, 'm, 'p>(
            &mut self,
            _context: MetaheuristicContext<
                'e,
                'r,
                's,
                'm,
                'p,
                i64,
                DefaultCostEvaluator,
                ChaCha8Rng,
            >,
            _plan: &Plan<'p, i64>,
        ) -> bool {
            self.calls_accept += 1;
            self.ret_accept
        }
    }

    #[test]
    fn test_metaheuristic_context_accessors() {
        let problem = make_basic_problem();
        let (model, state, eval) = make_model_state_eval(&problem);

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let ctx = MetaheuristicContext::new(&model, &state, &eval, &mut rng);

        assert!(std::ptr::eq(ctx.model(), &model));
        assert!(std::ptr::eq(ctx.solver_state(), &state));
        assert!(std::ptr::eq(ctx.evaluator(), &eval));
    }

    #[test]
    fn test_metaheuristic_trait_calls_and_returns() {
        let problem = make_basic_problem();
        let (model, state, eval) = make_model_state_eval(&problem);

        let mut mh = DummyMetaheuristic::new("DummyMH", true, false);

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let ctx = MetaheuristicContext::new(&model, &state, &eval, &mut rng);
        let reached = mh.local_optimum_reached(ctx);
        assert!(reached, "should return configured local optimum value");

        // Build a trivial plan (no patches, empty delta, zero fitness change)
        let plan = Plan::new_delta(
            Vec::<DecisionVarPatch<i64>>::new(),
            TerminalDelta::empty(),
            FitnessDelta::zero(),
        );

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let ctx2 = MetaheuristicContext::new(&model, &state, &eval, &mut rng);
        let accepted = mh.accept_plan(ctx2, &plan);
        assert!(!accepted, "should return configured accept value");

        let (calls_local, calls_accept) = mh.calls();
        assert_eq!(calls_local, 1);
        assert_eq!(calls_accept, 1);
    }

    #[test]
    fn test_context_with_rng_like_operator_environment() {
        // Sanity: show that this context composes alongside typical RNG usage
        // in the broader system (not used directly here, but lifetimes should be fine).
        let problem = make_basic_problem();
        let (model, state, eval) = make_model_state_eval(&problem);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Not used by MetaheuristicContext, but ensure typical rng is available together
        let _sample = rng.next_u32();

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        // Just ensure MetaheuristicContext compiles and works in presence of rng.
        let ctx = MetaheuristicContext::new(&model, &state, &eval, &mut rng);
        let mut mh = DummyMetaheuristic::new("RngCoexist", false, true);
        let _ = mh.local_optimum_reached(ctx);
    }
}
