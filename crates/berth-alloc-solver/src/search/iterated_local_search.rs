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
    search::{
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
        lns::{PerturbationProcedure, PerturbationProcedureContext},
    },
    state::{plan::Plan, solver_state::SolverState},
};

#[derive(Debug)]
pub struct IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>
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

impl<'e, 'r, 's, 'm, 'p, T, C, R> IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>
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
}

pub trait IlsAcceptanceCriterion<T, C, R>: Send
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool;
}

impl<'a, T, C, R> std::fmt::Debug for dyn IlsAcceptanceCriterion<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IlsAcceptanceCriterion({})", self.name())
    }
}

impl<'a, T, C, R> std::fmt::Display for dyn IlsAcceptanceCriterion<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IlsAcceptanceCriterion({})", self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GreedyDescentAcceptanceCriterion;

impl<T, C, R> IlsAcceptanceCriterion<T, C, R> for GreedyDescentAcceptanceCriterion
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "GreedyDescentAcceptanceCriterion"
    }

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        _context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool {
        let d = &plan.fitness_delta;

        // Lexicographic improvement: fewer unassigned first, then lower cost.
        if d.delta_unassigned < 0 {
            return true;
        }
        if d.delta_unassigned > 0 {
            return false;
        }

        // Equal unassigned: accept only on strict cost decrease.
        d.delta_cost < 0
    }
}

pub struct PerturbationDecisionBuilder<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + 'n>,
    perturbation_procedure: Box<dyn PerturbationProcedure<T, C, R> + 'n>,
}

impl<'n, T, C, R> PerturbationDecisionBuilder<'n, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + 'n>,
        perturbation_procedure: Box<dyn PerturbationProcedure<T, C, R> + 'n>,
    ) -> Self {
        Self {
            acceptance_criterion,
            perturbation_procedure,
        }
    }

    #[inline]
    pub fn acceptance_criterion(&self) -> &dyn IlsAcceptanceCriterion<T, C, R> {
        self.acceptance_criterion.as_ref()
    }

    #[inline]
    pub fn perturbation_procedure(&self) -> &dyn PerturbationProcedure<T, C, R> {
        self.perturbation_procedure.as_ref()
    }
}

impl<'n, T, C, R> DecisionBuilder<T, C, R> for PerturbationDecisionBuilder<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "PerturbationDecisionBuilder"
    }

    fn next<'b, 'sm, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        // Respect external termination requests from the monitor before doing any work.
        if context.monitor.should_terminate_search() {
            return None;
        }

        // Build a perturbation (ruin + repair or other) using the LNS-style context.
        // We allocate a short-lived PerturbationProcedureContext that borrows the
        // same RNG/work buffer; this keeps lifetimes narrow and avoids aliasing.
        let plan = {
            let mut pp_ctx = PerturbationProcedureContext::new(
                context.model,
                context.state,
                context.evaluator,
                context.rng,
                context.work_buf,
            );
            self.perturbation_procedure.perturb(&mut pp_ctx)
        };

        // Notify monitor that we generatederated a candidate (even if empty).
        context.monitor.on_plan_generated(&plan);

        if context.monitor.should_terminate_search() {
            return None;
        }

        // Empty plans or non-improving plans may be rejected by the acceptance criterion.
        // Construct an acceptance context (re-borrow RNG mutably after perturbation finished).
        let accepted = {
            let mut ac_ctx = IlsAcceptanceCriterionContext::new(
                context.model,
                context.state,
                context.evaluator,
                context.rng,
            );
            // If the plan is empty we can short-circuit; otherwise defer to criterion.
            !plan.is_empty() && self.acceptance_criterion.accept(&mut ac_ctx, &plan)
        };

        if accepted {
            context.monitor.on_plan_accepted(&plan);
            Some(plan)
        } else {
            context.monitor.on_plan_rejected(&plan);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::decisionvar::DecisionVar;
    use crate::{
        model::solver_model::SolverModel,
        search::eval::DefaultCostEvaluator,
        state::{
            decisionvar::DecisionVarVec, fitness::Fitness, plan::Plan, solver_state::SolverState,
            terminal::delta::TerminalDelta,
        },
    };
    use berth_alloc_core::prelude::{TimeInterval, TimePoint};
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use rand::{SeedableRng, rngs::StdRng};

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
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }

    fn make_minimal_problem() -> Problem<T> {
        // One berth, no flexible requests (simplest feasible problem).
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder
            .build()
            .expect("minimal problem with one berth should build")
    }

    fn make_context<'p>(
        problem: &'p Problem<T>,
    ) -> (
        SolverModel<'p, T>,
        SolverState<'p, T>,
        DefaultCostEvaluator,
        StdRng,
    ) {
        let model = SolverModel::try_from(problem).expect("model creation ok");

        let terminal = crate::state::terminal::terminalocc::TerminalOccupancy::new(
            model.problem().berths().iter(),
        );

        let dvars = DecisionVarVec::from(Vec::new()); // no flexible requests
        // Initial fitness (cost=0, unassigned=0).
        let fitness = Fitness::new(0, 0);
        let state = SolverState::new(dvars, terminal, fitness);

        let eval = DefaultCostEvaluator;
        let rng = StdRng::seed_from_u64(12345);

        (model, state, eval, rng)
    }

    #[test]
    fn test_trait_object_debug_and_display() {
        let problem = make_minimal_problem();
        let (model, state, eval, mut rng) = make_context(&problem);
        let crit = GreedyDescentAcceptanceCriterion;
        let mut ctx = IlsAcceptanceCriterionContext::new(&model, &state, &eval, &mut rng);

        let obj: &dyn IlsAcceptanceCriterion<T, DefaultCostEvaluator, StdRng> = &crit;
        assert_eq!(
            format!("{:?}", obj),
            "IlsAcceptanceCriterion(GreedyDescentAcceptanceCriterion)"
        );
        assert_eq!(
            format!("{}", obj),
            "IlsAcceptanceCriterion(GreedyDescentAcceptanceCriterion)"
        );

        // Smoke call to accept with a zero-delta plan (should reject).
        let plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::zero(),
        );
        assert!(!obj.accept(&mut ctx, &plan));
    }

    #[test]
    fn test_greedy_descent_acceptance_logic() {
        let problem = make_minimal_problem();
        let (model, state, eval, mut rng) = make_context(&problem);
        let crit = GreedyDescentAcceptanceCriterion;
        let mut ctx = IlsAcceptanceCriterionContext::new(&model, &state, &eval, &mut rng);

        // 1. Strict improvement in unassigned (delta_unassigned < 0) should accept regardless of cost delta.
        let plan_improve_unassigned = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::new(10, -1), // worse cost but fewer unassigned
        );
        assert!(crit.accept(&mut ctx, &plan_improve_unassigned));

        // 2. Worsening unassigned (delta_unassigned > 0) should reject even if cost decreases.
        let plan_worsen_unassigned = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::new(-50, 1), // better cost but more unassigned
        );
        assert!(!crit.accept(&mut ctx, &plan_worsen_unassigned));

        // 3. Equal unassigned; strictly better cost => accept.
        let plan_better_cost = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::new(-5, 0),
        );
        assert!(crit.accept(&mut ctx, &plan_better_cost));

        // 4. Equal unassigned; cost unchanged => reject.
        let plan_equal_cost = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::new(0, 0),
        );
        assert!(!crit.accept(&mut ctx, &plan_equal_cost));

        // 5. Equal unassigned; cost increases => reject.
        let plan_worse_cost = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::new(7, 0),
        );
        assert!(!crit.accept(&mut ctx, &plan_worse_cost));

        // 6. Larger negative cost (still equal unassigned) accepted.
        let plan_large_cost_improvement = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            crate::state::fitness::FitnessDelta::new(-100, 0),
        );
        assert!(crit.accept(&mut ctx, &plan_large_cost_improvement));
    }

    struct CountingMonitor {
        generated: usize,
        acc: usize,
        rej: usize,
        terminate: bool,
    }

    impl CountingMonitor {
        fn new() -> Self {
            Self {
                generated: 0,
                acc: 0,
                rej: 0,
                terminate: false,
            }
        }
        fn with_terminate() -> Self {
            Self {
                generated: 0,
                acc: 0,
                rej: 0,
                terminate: true,
            }
        }
    }

    impl crate::monitor::search_monitor::TerminationCheck for CountingMonitor {
        fn should_terminate_search(&self) -> bool {
            self.terminate
        }
    }

    impl crate::monitor::search_monitor::PlanEventMonitor<T> for CountingMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, T>) {
            self.generated += 1;
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, T>) {
            self.rej += 1;
        }
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, T>) {
            self.acc += 1;
        }
    }

    // Dummy perturbation that returns a fixed fitness delta plan
    struct DummyPerturb {
        dc: i64,
        du: i32,
    }
    impl DummyPerturb {
        fn new(dc: i64, du: i32) -> Self {
            Self { dc, du }
        }
    }
    impl<T, C, R> PerturbationProcedure<T, C, R> for DummyPerturb
    where
        T: Copy + Ord + std::fmt::Debug,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "DummyPerturb"
        }
        fn perturb<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut crate::search::lns::PerturbationProcedureContext<
                'b,
                'r,
                'c,
                's,
                'm,
                'p,
                T,
                C,
                R,
            >,
        ) -> Plan<'p, T> {
            Plan::new_delta(
                Vec::new(),
                TerminalDelta::empty(),
                crate::state::fitness::FitnessDelta::new(self.dc, self.du),
            )
        }
    }

    #[test]
    fn test_builder_name_and_trait_display_debug() {
        let problem = make_minimal_problem();
        let (model, state, eval, mut rng) = make_context(&problem);
        let mut work_buf: Vec<DecisionVar<T>> = vec![/* empty: no flex */];

        let acceptance = Box::new(GreedyDescentAcceptanceCriterion);
        let perturb = Box::new(DummyPerturb::new(-1, 0)); // accept on lower cost

        let builder: PerturbationDecisionBuilder<T, _, StdRng> =
            PerturbationDecisionBuilder::new(acceptance, perturb);

        assert_eq!(builder.name(), "PerturbationDecisionBuilder");

        // Test Debug/Display through the DecisionBuilder trait object
        let obj: &dyn DecisionBuilder<T, _, _> = &builder;
        assert_eq!(
            format!("{}", obj),
            "DecisionBuilder(PerturbationDecisionBuilder)"
        );
        assert_eq!(
            format!("{:?}", obj),
            "DecisionBuilder(PerturbationDecisionBuilder)"
        );

        // Smoke call to next (should accept and return Some)
        let mut monitor = CountingMonitor::new();
        let mut ctx = crate::search::decision_builder::SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            *state.fitness(),
            &mut monitor,
        );
        let mut builder = builder;
        let plan_opt = builder.next(&mut ctx);
        assert!(plan_opt.is_some());
    }

    #[test]
    fn test_next_accepts_plan_and_emits_events() {
        let problem = make_minimal_problem();
        let (model, state, eval, mut rng) = make_context(&problem);
        let mut work_buf: Vec<DecisionVar<T>> = vec![];

        // Plan will strictly reduce unassigned => must be accepted by greedy
        let acceptance = Box::new(GreedyDescentAcceptanceCriterion);
        let perturb = Box::new(DummyPerturb::new(0, -1));

        let mut builder: PerturbationDecisionBuilder<T, _, StdRng> =
            PerturbationDecisionBuilder::new(acceptance, perturb);

        let mut monitor = CountingMonitor::new();
        let mut ctx = crate::search::decision_builder::SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            *state.fitness(),
            &mut monitor,
        );

        let plan_opt = builder.next(&mut ctx);
        assert!(plan_opt.is_some(), "accepted plan should be returned");
        assert_eq!(monitor.generated, 1, "one plan should be generatederated");
        assert_eq!(monitor.acc, 1, "one plan should be accepted");
        assert_eq!(monitor.rej, 0, "no plan should be rejected");
    }

    #[test]
    fn test_next_rejects_plan_and_emits_events() {
        let problem = make_minimal_problem();
        let (model, state, eval, mut rng) = make_context(&problem);
        let mut work_buf: Vec<DecisionVar<T>> = vec![];

        // Plan has zero delta => greedy should reject
        let acceptance = Box::new(GreedyDescentAcceptanceCriterion);
        let perturb = Box::new(DummyPerturb::new(0, 0));

        let mut builder: PerturbationDecisionBuilder<T, _, StdRng> =
            PerturbationDecisionBuilder::new(acceptance, perturb);

        let mut monitor = CountingMonitor::new();
        let mut ctx = crate::search::decision_builder::SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            *state.fitness(),
            &mut monitor,
        );

        let plan_opt = builder.next(&mut ctx);
        assert!(plan_opt.is_none(), "rejected plan should yield None");
        assert_eq!(monitor.generated, 1, "one plan should be generatederated");
        assert_eq!(monitor.acc, 0, "no plan should be accepted");
        assert_eq!(monitor.rej, 1, "one plan should be rejected");
    }

    #[test]
    fn test_next_respects_termination_early() {
        let problem = make_minimal_problem();
        let (model, state, eval, mut rng) = make_context(&problem);
        let mut work_buf: Vec<DecisionVar<T>> = vec![];

        let acceptance = Box::new(GreedyDescentAcceptanceCriterion);
        let perturb = Box::new(DummyPerturb::new(-1, 0)); // would be accepted if allowed

        let mut builder: PerturbationDecisionBuilder<T, _, StdRng> =
            PerturbationDecisionBuilder::new(acceptance, perturb);

        let mut monitor = CountingMonitor::with_terminate();
        let mut ctx = crate::search::decision_builder::SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            *state.fitness(),
            &mut monitor,
        );

        let plan_opt = builder.next(&mut ctx);
        assert!(plan_opt.is_none(), "early termination yields None");
        assert_eq!(
            (monitor.generated, monitor.acc, monitor.rej),
            (0, 0, 0),
            "no events should be emitted under early termination"
        );
    }
}
