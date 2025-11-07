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
    model::{index::RequestIndex, solver_model::SolverModel},
    search::{
        eval::CostEvaluator,
        metaheuristic::{Metaheuristic, MetaheuristicContext},
    },
    state::{
        plan::Plan,
        solver_state::{SolverState, SolverStateView},
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};

pub trait CoolingSchedule: Send {
    fn name(&self) -> &str;
    fn reset(&mut self);
    fn on_evaluation(&mut self);
    fn on_accept(&mut self, _improved: bool);
    fn on_reject(&mut self);
    fn reheat(&mut self, factor: f64);
    fn temperature(&self) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct IterReciprocalCooling {
    t0: f64,
    iter: u64,
}

impl IterReciprocalCooling {
    #[inline]
    pub fn new(t0: f64) -> Self {
        Self { t0, iter: 0 }
    }
}

impl CoolingSchedule for IterReciprocalCooling {
    #[inline]
    fn reset(&mut self) {
        self.iter = 0;
    }

    #[inline]
    fn on_evaluation(&mut self) {
        self.iter = self.iter.saturating_add(1);
    }

    #[inline]
    fn on_accept(&mut self, _improved: bool) {
        self.iter = self.iter.saturating_add(1);
    }

    #[inline]
    fn on_reject(&mut self) {
        // No-op in previous design
    }

    #[inline]
    fn reheat(&mut self, factor: f64) {
        // Simple reheating: scale T0.
        if factor.is_finite() && factor > 0.0 {
            self.t0 *= factor;
        }
    }

    #[inline]
    fn temperature(&self) -> f64 {
        if self.iter > 0 {
            self.t0 / (self.iter as f64)
        } else {
            0.0
        }
    }

    #[inline]
    fn name(&self) -> &str {
        "IterReciprocalCooling"
    }
}

/// Computes a default penalty λ for unassigned requests such that
/// the energy function E = λ · (#unassigned) + total_cost strictly prefers
/// any state with fewer unassigned requests over any state with more,
/// by only a slight margin (+1 over the worst-case total cost).
///
/// Derivation:
/// - For a request r with weight w_r and feasible window [a, b), the turnaround
///   time is waiting_time + processing_time. The latest feasible start time
///   ensures turnaround_time ≤ (b - a) = window_length.
/// - Therefore, the per-request cost is bounded by:
///   cost_r ≤ w_r · window_length_r
/// - Summing over all flexible requests yields an upper bound on the total cost:
///   total_cost ≤ Σ_r (w_r · window_length_r)
/// - Setting λ = (Σ_r (w_r · window_length_r)) + 1 makes reducing the number of
///   unassigned requests by 1 always strictly better than any possible increase
///   in total_cost, with only a +1 margin ("slight" preference).
///
/// This establishes a lexicographic preference: fewer unassigned first, then cost.
pub fn default_lambda_unassigned<T>(model: &SolverModel<'_, T>) -> i64
where
    T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
{
    // Max cost per request ≤ weight(req) * feasible_window_length(req)
    // Sum across requests gives an upper bound on total cost. Add 1 to make the
    // penalty just slightly dominate any possible total cost change.
    let mut total_max: Cost = 0;
    for i in 0..model.flexible_requests_len() {
        let ri = RequestIndex::new(i);
        let weight = model.weight(ri);
        let window_len: Cost = model.feasible_interval(ri).length().value().into();
        let per_req_max = weight.saturating_mul(window_len);
        total_max = total_max.saturating_add(per_req_max);
    }
    total_max.saturating_add(1).max(1)
}

#[derive(Clone, Copy, Debug)]
pub struct EnergyParams {
    pub lambda_unassigned: i64,
    pub step: i64,
    pub t0: f64,
    pub allow_infeasible_uphill: bool,
}

impl std::fmt::Display for EnergyParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EnergyParams {{ lambda_unassigned: {}, step: {}, t0: {}, allow_infeasible_uphill: {} }}",
            self.lambda_unassigned, self.step, self.t0, self.allow_infeasible_uphill
        )
    }
}

impl EnergyParams {
    #[inline]
    pub fn with_default_lambda<T>(
        model: &SolverModel<'_, T>,
        step: i64,
        t0: f64,
        allow_infeasible_uphill: bool,
    ) -> Self
    where
        T: Copy + Ord + CheckedAdd + CheckedSub + Into<Cost>,
    {
        let lambda = default_lambda_unassigned(model);
        Self {
            lambda_unassigned: lambda,
            step,
            t0,
            allow_infeasible_uphill,
        }
    }
}

pub struct SimulatedAnnealing<T, C, S, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    S: CoolingSchedule,
    R: rand::Rng,
{
    params: EnergyParams,
    cooling: S,
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

#[derive(Debug, Clone, Copy)]
struct EnergyComponents {
    e: i64,
    e2: i64,
    delta_unassigned: i64,
}

impl<T, C, S, R> SimulatedAnnealing<T, C, S, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    S: CoolingSchedule,
    R: rand::Rng,
{
    #[inline]
    pub fn new(params: EnergyParams, cooling: S) -> Self {
        Self {
            params,
            cooling,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    fn energy_components<'p>(
        &self,
        _model: &SolverModel<'p, T>,
        state: &SolverState<'p, T>,
        plan: &Plan<'p, T>,
    ) -> EnergyComponents {
        let cur_unassigned: i64 = state
            .fitness()
            .unassigned_requests
            .try_into()
            .expect("usize fits i64");
        let cur_cost: Cost = state.fitness().cost;

        let du: i64 = plan.fitness_delta.delta_unassigned as i64;
        let dcost: Cost = plan.fitness_delta.delta_cost;

        let e = self.params.lambda_unassigned * cur_unassigned + cur_cost;
        let e2 = self.params.lambda_unassigned * (cur_unassigned + du) + (cur_cost + dcost);

        EnergyComponents {
            e,
            e2,
            delta_unassigned: du,
        }
    }
}

impl<T, C, S, R> Metaheuristic<T, C, R> for SimulatedAnnealing<T, C, S, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
    S: CoolingSchedule,
{
    #[inline]
    fn name(&self) -> &str {
        "SimulatedAnnealing"
    }

    #[inline]
    fn local_optimum_reached<'e, 'r, 's, 'm, 'p>(
        &mut self,
        _ctx: MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>,
    ) -> bool {
        self.cooling.on_evaluation();
        self.cooling.temperature() > 0.0
    }

    fn accept_plan<'e, 'r, 's, 'm, 'p>(
        &mut self,
        ctx: MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool {
        let components = self.energy_components(ctx.model(), ctx.solver_state(), plan);

        if components.delta_unassigned > 0 && !self.params.allow_infeasible_uphill {
            self.cooling.on_reject();
            return false;
        }

        if components.delta_unassigned < 0 {
            self.cooling.on_accept(true);
            return true;
        }

        if components.e2 <= components.e.saturating_sub(self.params.step) {
            self.cooling.on_accept(true);
            return true;
        }

        let t = self.cooling.temperature();
        if t <= 0.0 {
            self.cooling.on_reject();
            return false;
        }

        let u = ctx.rng.random::<f64>().clamp(1e-12, 1.0 - 1e-12);
        let log2u = u.ln() / std::f64::consts::LN_2;
        let bound = (components.e as f64) - (self.params.step as f64) - log2u * t;
        let accept = (components.e2 as f64) <= bound;

        if accept {
            self.cooling.on_accept(false);
        } else {
            self.cooling.on_reject();
        }
        accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::eval::{CostEvaluator, DefaultCostEvaluator};
    use crate::state::terminal::{delta::TerminalDelta, terminalocc::TerminalOccupancy};
    use crate::state::{
        decisionvar::DecisionVarVec,
        fitness::{Fitness, FitnessDelta},
        plan::Plan,
    };
    use berth_alloc_core::prelude::{TimeInterval, TimePoint};
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem};
    use rand::{SeedableRng, rngs::StdRng};
    use rand_chacha::ChaCha8Rng;

    type T = i64;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }

    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }

    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }

    fn make_minimal_problem() -> Problem<T> {
        use berth_alloc_model::problem::builder::ProblemBuilder;

        let berth = Berth::from_windows(bid(1), vec![iv(0, 100)]);
        let mut builder = ProblemBuilder::new();
        builder.add_berth(berth);
        builder.build().expect("problem build should succeed")
    }

    fn make_minimal_state<'p>(
        model: &'p crate::model::solver_model::SolverModel<'p, T>,
        fitness: Fitness,
    ) -> crate::state::solver_state::SolverState<'p, T> {
        let terminal = TerminalOccupancy::new(model.berths());
        let dv = DecisionVarVec::new(Vec::new());
        crate::state::solver_state::SolverState::new(dv, terminal, fitness)
    }

    fn ctx<'e, 'r, 's, 'm, 'p, C, R>(
        model: &'m crate::model::solver_model::SolverModel<'p, T>,
        state: &'s crate::state::solver_state::SolverState<'p, T>,
        evaluator: &'e C,
        rand: &'r mut R,
    ) -> MetaheuristicContext<'e, 'r, 's, 'm, 'p, T, C, R>
    where
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        MetaheuristicContext::new(model, state, evaluator, rand)
    }

    // Helper to build SA with explicit IterReciprocalCooling and RNG.
    fn sa<R: rand::Rng>(
        lambda_unassigned: i64,
        step: i64,
        t0: f64,
        allow_inf: bool,
    ) -> SimulatedAnnealing<T, DefaultCostEvaluator, IterReciprocalCooling, R> {
        let params = EnergyParams {
            lambda_unassigned,
            step,
            t0,
            allow_infeasible_uphill: allow_inf,
        };
        SimulatedAnnealing::new(params, IterReciprocalCooling::new(t0))
    }

    #[test]
    fn test_metaheuristic_display_and_debug_use_name() {
        let mh = sa(1, 1, 10.0, false);
        let obj: &dyn Metaheuristic<T, DefaultCostEvaluator, ChaCha8Rng> = &mh;

        assert_eq!(format!("{}", obj), "Metaheuristic(SimulatedAnnealing)");
        assert_eq!(format!("{:?}", obj), "Metaheuristic(SimulatedAnnealing)");
    }

    #[test]
    fn test_local_optimum_reached_increments_iter_and_reports_active_when_t_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut mh = sa(1, 1, 8.0, false);

        // Build any valid context; it isn't used in local_optimum_reached
        let problem = make_minimal_problem();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("ok model");
        let state = make_minimal_state(&model, Fitness::new(100, 2));
        let evaluator = DefaultCostEvaluator;

        // First call increments iter from 0 to 1, making temperature t0/1 = 8.0 > 0
        let active = mh.local_optimum_reached(ctx(&model, &state, &evaluator, &mut rng));
        assert!(active, "SA should be active when t>0 after first increment");

        for _ in 0..10 {
            let ok = mh.local_optimum_reached(ctx(&model, &state, &evaluator, &mut rng));
            assert!(ok);
        }
    }

    #[test]
    fn test_accept_plan_rejects_if_unassigned_increases_and_not_allowed() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut mh = sa(10, 1, 0.0, false); // T=0 initially

        let problem = make_minimal_problem();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("ok model");
        let state = make_minimal_state(&model, Fitness::new(100, 5));
        let evaluator = DefaultCostEvaluator;
        let plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            FitnessDelta::new(0, 1), // delta_unassigned > 0
        );

        let accepted = mh.accept_plan(ctx(&model, &state, &evaluator, &mut rng), &plan);
        assert!(
            !accepted,
            "should reject when infeasible uphill is not allowed"
        );
    }

    #[test]
    fn test_accept_plan_accepts_if_unassigned_decreases() {
        let mut rng = StdRng::seed_from_u64(8);
        let mut mh = sa(10, 1, 0.0, false);

        let problem = make_minimal_problem();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("ok model");
        let state = make_minimal_state(&model, Fitness::new(100, 5));
        let evaluator = DefaultCostEvaluator;
        let plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            FitnessDelta::new(0, -1), // strictly reduces unassigned
        );

        let accepted = mh.accept_plan(ctx(&model, &state, &evaluator, &mut rng), &plan);
        assert!(
            accepted,
            "should accept immediately when unassigned decreases"
        );
    }

    #[test]
    fn test_accept_plan_accepts_on_deterministic_step_improvement() {
        let mut rng = StdRng::seed_from_u64(9);
        // step=5 means we need e2 <= e - 5 for deterministic accept
        let mut mh = sa(1, 5, 0.0, true);

        // current: cost=100, unassigned=0 => e=100
        // plan: delta_cost = -10 => e2 = 90, which is <= 95 == 100 - step -> accept
        let problem = make_minimal_problem();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("ok model");
        let state = make_minimal_state(&model, Fitness::new(100, 0));
        let evaluator = DefaultCostEvaluator;
        let plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            FitnessDelta::new(-10, 0),
        );

        let accepted = mh.accept_plan(ctx(&model, &state, &evaluator, &mut rng), &plan);
        assert!(
            accepted,
            "should deterministically accept when e2 <= e - step"
        );
    }

    #[test]
    fn test_accept_plan_rejects_when_t_zero_and_not_step_improvement() {
        let mut rng = StdRng::seed_from_u64(10);
        // t0=0 => temperature() == 0 initially, so no probabilistic acceptance
        let mut mh = sa(1, 2, 0.0, true);

        // current: e = cost=100
        // plan: +1 cost => e2=101, not <= e - step (=98), not delta_unassigned<0 -> reject when T=0
        let problem = make_minimal_problem();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("ok model");
        let state = make_minimal_state(&model, Fitness::new(100, 0));
        let evaluator = DefaultCostEvaluator;
        let plan = Plan::new_delta(Vec::new(), TerminalDelta::empty(), FitnessDelta::new(1, 0));

        let accepted = mh.accept_plan(ctx(&model, &state, &evaluator, &mut rng), &plan);
        assert!(!accepted, "with T=0 and no step improvement, must reject");
    }

    #[test]
    fn test_allow_infeasible_uphill_does_not_force_accept() {
        let mut rng = StdRng::seed_from_u64(11);
        // allow_infeasible_uphill = true, but T=0 and not a deterministic improvement => reject
        let mut mh = sa(10, 100, 0.0, true);

        let problem = make_minimal_problem();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("ok model");
        let state = make_minimal_state(&model, Fitness::new(200, 3));
        let evaluator = DefaultCostEvaluator;
        let plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            FitnessDelta::new(0, 1), // delta_unassigned > 0, uphill infeasible
        );

        let accepted = mh.accept_plan(ctx(&model, &state, &evaluator, &mut rng), &plan);
        assert!(
            !accepted,
            "allow_infeasible_uphill only bypasses the early reject; it doesn't force acceptance"
        );
    }

    #[inline]
    fn td(v: i64) -> berth_alloc_core::prelude::TimeDelta<i64> {
        berth_alloc_core::prelude::TimeDelta::new(v)
    }

    fn make_problem_with_flex_for_lambda() -> Problem<T> {
        use berth_alloc_model::problem::builder::ProblemBuilder;
        use berth_alloc_model::{
            common::FlexibleKind, prelude::RequestIdentifier, problem::req::Request,
        };

        // One berth covering all windows we need
        let berth = Berth::from_windows(bid(1), vec![iv(0, 1000)]);

        // Two flexible requests with weights and windows:
        // r1: weight=3, window=[0,10)  -> length=10
        // r2: weight=2, window=[5,25)  -> length=20
        // Processing time entries must be <= window length; use pt=1 for both.
        let mut builder = ProblemBuilder::new();
        builder.add_berth(berth);

        let mut pt1 = std::collections::BTreeMap::new();
        pt1.insert(bid(1), td(1));
        let r1 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(10), iv(0, 10), 3, pt1)
            .expect("r1 ok");

        let mut pt2 = std::collections::BTreeMap::new();
        pt2.insert(bid(1), td(1));
        let r2 = Request::<FlexibleKind, i64>::new(RequestIdentifier::new(20), iv(5, 25), 2, pt2)
            .expect("r2 ok");

        builder.add_flexible(r1);
        builder.add_flexible(r2);
        builder.build().expect("problem build should succeed")
    }

    #[test]
    fn test_default_lambda_unassigned_no_requests_returns_one() {
        let problem = make_minimal_problem(); // 0 flexible requests
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("model ok");
        let lambda = default_lambda_unassigned(&model);
        assert_eq!(lambda, 1, "with no flexible requests, lambda must be 1");
    }

    #[test]
    fn test_default_lambda_unassigned_matches_sum_plus_one() {
        let problem = make_problem_with_flex_for_lambda();
        let model = crate::model::solver_model::SolverModel::try_from(&problem).expect("model ok");

        // Expected: sum(weight * window_length) + 1
        // r1: w=3, len=10 -> 30
        // r2: w=2, len=20 -> 40
        // total = 70; lambda = 71
        let lambda = default_lambda_unassigned(&model);
        assert_eq!(lambda, 71);
    }
}
