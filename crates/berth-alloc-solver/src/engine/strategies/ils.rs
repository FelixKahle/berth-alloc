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
    core::numeric::SolveNumeric,
    engine::strategy::Strategy,
    model::solver_model::SolverModel,
    monitor::{
        search_monitor::{LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck},
        step::{PlanLimitMonitor, StagnationMonitor},
    },
    search::{
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
    },
    state::{decisionvar::DecisionVar, plan::Plan, solver_state::SolverState},
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
pub struct GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

impl<T, C, R> Default for GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R> GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, R> IlsAcceptanceCriterion<T, C, R> for GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "GreedyDescentAcceptanceCriterion"
    }

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        _context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool {
        let delta = &plan.fitness_delta;

        if delta.delta_unassigned < 0 {
            return true;
        }
        if delta.delta_unassigned > 0 {
            return false;
        }

        delta.delta_cost < 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

impl<T, C, R> Default for AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R> AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, R> IlsAcceptanceCriterion<T, C, R> for AlwaysAcceptAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "AlwaysAcceptAcceptanceCriterion"
    }

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        _context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        _plan: &Plan<'p, T>,
    ) -> bool {
        true
    }
}

pub struct IteratedLocalSearchConfig<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    pub max_local_stagnation_steps: usize, // Number of local steps without improvement to consider stagnated; run pertubation then
    pub max_local_steps: Option<usize>, // Run pertubation after these many local steps even if not stagnated
    pub acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
    pub improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    pub perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    pub evaluator: C,
    pub rng: R,
}

impl<'n, T, C, R> IteratedLocalSearchConfig<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new(
        max_local_stagnation_steps: usize,
        max_local_steps: Option<usize>,
        acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
        improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
        perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
        evaluator: C,
        rng: R,
    ) -> Self {
        Self {
            max_local_stagnation_steps,
            max_local_steps,
            acceptance_criterion,
            improving_decision_builder,
            perturbing_decision_builder,
            evaluator,
            rng,
        }
    }
}

pub struct IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    evaluator: C,
    rng: R,
    max_local_stagnation_steps: usize,
    max_local_steps: Option<usize>,
    acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
    improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
}

impl<'n, T, C, R> IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new(config: IteratedLocalSearchConfig<'n, T, C, R>) -> Self {
        Self {
            max_local_stagnation_steps: config.max_local_stagnation_steps,
            max_local_steps: config.max_local_steps,
            acceptance_criterion: config.acceptance_criterion,
            improving_decision_builder: config.improving_decision_builder,
            perturbing_decision_builder: config.perturbing_decision_builder,
            evaluator: config.evaluator,
            rng: config.rng,
        }
    }

    #[inline(always)]
    pub fn evaluator(&self) -> &C {
        &self.evaluator
    }

    #[inline(always)]
    pub fn rng(&mut self) -> &mut R {
        &mut self.rng
    }

    #[inline(always)]
    pub fn max_local_stagnation_steps(&self) -> usize {
        self.max_local_stagnation_steps
    }

    #[inline(always)]
    pub fn max_local_steps(&self) -> Option<usize> {
        self.max_local_steps
    }

    #[inline(always)]
    pub fn acceptance_criterion(&self) -> &dyn IlsAcceptanceCriterion<T, C, R> {
        self.acceptance_criterion.as_ref()
    }

    #[inline(always)]
    pub fn improving_decision_builder(&self) -> &dyn DecisionBuilder<T, C, R> {
        self.improving_decision_builder.as_ref()
    }

    #[inline(always)]
    pub fn perturbing_decision_builder(&self) -> &dyn DecisionBuilder<T, C, R> {
        self.perturbing_decision_builder.as_ref()
    }

    #[inline]
    fn perturb_once<'p>(
        &mut self,
        model: &SolverModel<'p, T>,
        base_state: &SolverState<'p, T>,
        work_buf: &mut [DecisionVar<T>],
        parent_mon: &mut dyn SearchMonitor<T>,
    ) -> Option<(SolverState<'p, T>, Plan<'p, T>)> {
        let mut working = base_state.clone();
        let mut ctx = SearchContext::new(
            model,
            &working,
            &self.evaluator,
            &mut self.rng,
            work_buf,
            *working.fitness(),
            parent_mon, // parent monitor only sees the final aggregated candidate later
        );
        let plan = self.perturbing_decision_builder.next(&mut ctx)?;
        working.apply_plan(plan.clone());
        Some((working, plan))
    }

    #[inline]
    fn local_improve<'p>(
        &mut self,
        model: &SolverModel<'p, T>,
        mut working: SolverState<'p, T>,
        parent_mon: &mut dyn SearchMonitor<T>,
    ) -> (Plan<'p, T>, SolverState<'p, T>) {
        // Wrap parent for stagnation + plan limit only during this local phase.
        let mut phase_mon = PassThroughWithConstraints::<'_, T>::new(
            parent_mon,
            Some(self.max_local_stagnation_steps),
            self.max_local_steps.map(|v| v as u64),
        );

        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        // We aggregate by collecting step plans, and accumulating fitness delta separately.
        let mut step_plans: Vec<Plan<'p, T>> = Vec::new();
        let mut agg_cost: i64 = 0;
        let mut agg_unassigned: i32 = 0;

        loop {
            if phase_mon.should_terminate_search() {
                break;
            }
            let cur_fit = *working.fitness();
            let mut ctx = SearchContext::new(
                model,
                &working,
                &self.evaluator,
                &mut self.rng,
                &mut work_buf,
                cur_fit,
                &mut phase_mon,
            );
            let step = match self.improving_decision_builder.next(&mut ctx) {
                Some(p) => p,
                None => break,
            };

            // Evaluate improvement based on that step’s delta only (local hill-climb semantics).
            let d = &step.fitness_delta;
            let step_improved =
                (d.delta_unassigned < 0) || (d.delta_unassigned == 0 && d.delta_cost < 0);

            // Apply step to working sandbox.
            working.apply_plan(step.clone());
            agg_cost += d.delta_cost;
            agg_unassigned += d.delta_unassigned;
            step_plans.push(step.clone());

            // Local monitors see each step plan (not the final aggregate yet).
            phase_mon.on_plan_generated(&step);
            if step_improved {
                phase_mon.on_plan_accepted(&step);
            } else {
                phase_mon.on_plan_rejected(&step);
            }
        }

        // Build single aggregated plan once (last-write-wins via concat folding).
        let aggregated = if step_plans.is_empty() {
            Plan::empty()
        } else {
            let mut it = step_plans.into_iter();
            let first = it.next().unwrap();
            it.fold(first, |acc, p| acc.concat(p))
        };

        // Sanity: aggregated.fitness_delta should match sums
        debug_assert_eq!(aggregated.fitness_delta.delta_cost, agg_cost);
        debug_assert_eq!(
            aggregated.fitness_delta.delta_unassigned, agg_unassigned,
            "aggregated delta mismatch"
        );

        (aggregated, working)
    }
}

impl<'n, T, C, R> std::fmt::Display for IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IteratedLocalSearchStrategy(acceptance_criterion={}, improving_decision_builder={}, perturbing_decision_builder={})",
            self.acceptance_criterion(),
            self.improving_decision_builder(),
            self.perturbing_decision_builder()
        )
    }
}

struct PassThroughWithConstraints<'a, T: Copy + Ord> {
    parent: &'a mut dyn SearchMonitor<T>,
    stagnation: Option<StagnationMonitor>,
    plan_limit: Option<PlanLimitMonitor>,
}

impl<'a, T: Copy + Ord> PassThroughWithConstraints<'a, T> {
    pub fn new(
        parent: &'a mut dyn SearchMonitor<T>,
        stagnation_budget: Option<usize>,
        plan_limit: Option<u64>,
    ) -> Self {
        Self {
            parent,
            stagnation: stagnation_budget.map(StagnationMonitor::new),
            plan_limit: plan_limit.map(PlanLimitMonitor::new),
        }
    }

    #[inline]
    fn for_each_local<F: FnMut(&mut dyn SearchMonitor<T>)>(&mut self, mut f: F) {
        if let Some(s) = &mut self.stagnation {
            f(s);
        }
        if let Some(p) = &mut self.plan_limit {
            f(p);
        }
    }

    #[inline]
    fn any_local_terminate(&self) -> bool {
        let stag = self
            .stagnation
            .as_ref()
            .map(|s| s.should_terminate_search())
            .unwrap_or(false);
        let lim = self
            .plan_limit
            .as_ref()
            .map(|p| p.should_terminate_search())
            .unwrap_or(false);
        stag || lim
    }
}

impl<'a, T: Copy + Ord> TerminationCheck for PassThroughWithConstraints<'a, T> {
    fn should_terminate_search(&self) -> bool {
        self.parent.should_terminate_search() || self.any_local_terminate()
    }
}

impl<'a, T: Copy + Ord> PlanEventMonitor<T> for PassThroughWithConstraints<'a, T> {
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>) {
        // Local phase-only: advance local monitors (do not forward to parent)
        self.for_each_local(|m| m.on_plan_generated(plan));
    }

    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>) {
        // Local phase-only: update local monitors
        self.for_each_local(|m| m.on_plan_rejected(plan));
    }

    fn on_plan_accepted<'p>(&mut self, plan: &Plan<'p, T>) {
        // Local phase-only: update local monitors
        self.for_each_local(|m| m.on_plan_accepted(plan));
    }
}

impl<'a, T: Copy + Ord> LifecycleMonitor for PassThroughWithConstraints<'a, T> {
    fn on_search_start(&mut self) {
        self.parent.on_search_start();
        self.for_each_local(|m| m.on_search_start());
    }
    fn on_search_end(&mut self) {
        self.parent.on_search_end();
        self.for_each_local(|m| m.on_search_end());
    }
}

impl<'a, T: Copy + Ord> SearchMonitor<T> for PassThroughWithConstraints<'a, T> {
    fn name(&self) -> &str {
        "PassThroughWithConstraints"
    }
}

impl<'n, T, C, R> Strategy<T> for IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T> + Send + Sync,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "IteratedLocalSearchStrategy"
    }

    fn run<'p, 'e, 'm>(
        &mut self,
        context: &mut crate::engine::strategy::StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>> {
        context.monitor().on_search_start();
        let model = context.model();
        let mut base_state = context.state().clone();
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        loop {
            if context.monitor().should_terminate_search() {
                break;
            }

            // 1. Perturb
            let (working_after_perturb, perturb_plan) =
                match self.perturb_once(model, &base_state, &mut work_buf, context.monitor()) {
                    Some(p) => p,
                    None => break,
                };

            // 2. Local improvement phase (isolated)
            let (local_plan, _improved_state) =
                self.local_improve(model, working_after_perturb, context.monitor());

            // 3. Aggregate total candidate plan (perturb + local)
            let candidate_plan = if local_plan.is_empty() {
                perturb_plan
            } else {
                perturb_plan.concat(local_plan)
            };

            // 4. Emit aggregated plan to parent monitor exactly once
            context.monitor().on_plan_generated(&candidate_plan);

            // 5. Acceptance (relative to base)
            let mut acc_ctx = IlsAcceptanceCriterionContext::new(
                model,
                &base_state,
                &self.evaluator,
                &mut self.rng,
            );
            let accept = self
                .acceptance_criterion
                .accept(&mut acc_ctx, &candidate_plan);

            if accept {
                base_state.apply_plan(candidate_plan.clone());
                context.monitor().on_plan_accepted(&candidate_plan);
                let _ = context.shared_incumbent().try_update(&base_state, model);
            } else {
                context.monitor().on_plan_rejected(&candidate_plan);
                // sandbox improvements are discarded automatically
            }
        }

        context.monitor().on_search_end();
        Some(base_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{shared_incumbent::SharedIncumbent, strategy::StrategyContext},
        model::{
            index::{BerthIndex, RequestIndex},
            solver_model::SolverModel,
        },
        monitor::search_monitor::{
            LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
        },
        search::{
            decision_builder::SearchContext,
            eval::{CostEvaluator, DefaultCostEvaluator},
            lns::{
                RandomRuinRepairPerturbPair, RepairProcedure, RepairProcedureContext, RuinOutcome,
                RuinProcedure, RuinProcedureContext,
            },
        },
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::{Fitness, FitnessDelta},
            plan::{DecisionVarPatch, Plan},
            solver_state::SolverState,
            terminal::{delta::TerminalDelta, terminalocc::TerminalOccupancy},
        },
    };
    use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{builder::ProblemBuilder, req::Request},
    };
    use rand::{SeedableRng, rngs::StdRng};

    // ---------- Problem + model helpers ----------

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

    fn problem_one_berth_two_flex() -> Problem<i64> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);
        let mut pt1 = std::collections::BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, i64>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = std::collections::BTreeMap::new();
        pt2.insert(bid(1), td(5));
        let r2 = Request::<FlexibleKind, i64>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut b = ProblemBuilder::new();
        b.add_berth(b1);
        b.add_flexible(r1);
        b.add_flexible(r2);
        b.build().expect("valid problem")
    }

    fn make_model_state_eval(
        problem: &Problem<i64>,
    ) -> (
        SolverModel<'_, i64>,
        SolverState<'_, i64>,
        DefaultCostEvaluator,
    ) {
        let model = SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let eval = DefaultCostEvaluator;
        // Keep cost > 0 to satisfy debug assertion in apply_plan() when applying neutral perturbations
        let fitness = Fitness::new(100, model.flexible_requests_len());
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    // ---------- Test DecisionBuilders ----------

    // Improving builder: assigns request index 0 once (delta_unassigned = -1, cost 0).
    #[derive(Debug)]
    struct ImproveOne {
        emitted: bool,
    }
    impl ImproveOne {
        fn new() -> Self {
            Self { emitted: false }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for ImproveOne {
        fn name(&self) -> &str {
            "ImproveOne"
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;
            let patch = DecisionVarPatch::new(
                RequestIndex::new(0),
                DecisionVar::assigned(BerthIndex::new(0), tp(0)),
            );
            Some(Plan::new_delta(
                vec![patch],
                TerminalDelta::empty(),
                FitnessDelta::new(0 as Cost, -1),
            ))
        }
    }

    // Non-improving builder: emits one neutral (cost 0, unassigned 0) plan and stops.
    #[derive(Debug)]
    struct NeutralOnce {
        emitted: bool,
    }
    impl NeutralOnce {
        fn new() -> Self {
            Self { emitted: false }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for NeutralOnce {
        fn name(&self) -> &str {
            "NeutralOnce"
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;
            Some(Plan::new_delta(
                Vec::new(),
                TerminalDelta::empty(),
                FitnessDelta::new(0, 0),
            ))
        }
    }

    // Worsening builder: delta_unassigned = 0, delta_cost = +5.
    #[derive(Debug)]
    struct WorseOnce {
        emitted: bool,
    }
    impl WorseOnce {
        fn new() -> Self {
            Self { emitted: false }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for WorseOnce {
        fn name(&self) -> &str {
            "WorseOnce"
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;
            Some(Plan::new_delta(
                Vec::new(),
                TerminalDelta::empty(),
                FitnessDelta::new(5, 0),
            ))
        }
    }

    // Multi improving builder: emits k improving assignments for indices 0..k-1.
    #[derive(Debug)]
    struct ImproveMany {
        next_ix: usize,
        limit: usize,
    }
    impl ImproveMany {
        fn new(limit: usize) -> Self {
            Self { next_ix: 0, limit }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for ImproveMany {
        fn name(&self) -> &str {
            "ImproveMany"
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.next_ix >= self.limit {
                return None;
            }
            let ix = self.next_ix;
            self.next_ix += 1;

            // Build the decision and compute its cost delta against “unassigned” baseline (baseline cost = 0).
            let req = RequestIndex::new(ix);
            let berth = BerthIndex::new(0);
            let start = TimePoint::new(ix as i64);

            let delta_cost = ctx
                .evaluator()
                .eval_request(ctx.model(), req, start, berth)
                .expect("cost eval");

            let patch = DecisionVarPatch::new(req, DecisionVar::assigned(berth, start));
            Some(Plan::new_delta(
                vec![patch],
                TerminalDelta::empty(),
                FitnessDelta::new(delta_cost, -1),
            ))
        }
    }

    // ---------- Monitors ----------

    #[derive(Debug, Default)]
    struct CountingMonitor {
        started: usize,
        ended: usize,
        generated: usize,
        rejected: usize,
        accepted: usize,
        terminate_after: Option<usize>,
        terminated: bool,
    }
    impl CountingMonitor {
        fn with_terminate_after(n: usize) -> Self {
            Self {
                terminate_after: Some(n),
                ..Default::default()
            }
        }
    }
    impl TerminationCheck for CountingMonitor {
        fn should_terminate_search(&self) -> bool {
            self.terminated
        }
    }
    impl PlanEventMonitor<i64> for CountingMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.generated += 1;
            if let Some(limit) = self.terminate_after {
                if self.generated >= limit {
                    self.terminated = true;
                }
            }
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.rejected += 1;
        }
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.accepted += 1;
        }
    }
    impl LifecycleMonitor for CountingMonitor {
        fn on_search_start(&mut self) {
            self.started += 1;
        }
        fn on_search_end(&mut self) {
            self.ended += 1;
        }
    }
    impl SearchMonitor<i64> for CountingMonitor {
        fn name(&self) -> &str {
            "CountingMonitor"
        }
    }

    // ---------- Acceptance helpers ----------

    type GreedyCrit = GreedyDescentAcceptanceCriterion<i64, DefaultCostEvaluator, StdRng>;
    type AlwaysCrit = AlwaysAcceptAcceptanceCriterion<i64, DefaultCostEvaluator, StdRng>;

    // ---------- Tests ----------

    #[test]
    fn test_name_and_display() {
        let problem = problem_one_berth_two_flex();
        let (_m, _s, eval) = make_model_state_eval(&problem);
        let cfg = IteratedLocalSearchConfig::new(
            3,
            Some(5),
            Box::new(GreedyCrit::new()),
            Box::new(NeutralOnce::new()),
            Box::new(NeutralOnce::new()),
            eval,
            StdRng::seed_from_u64(1),
        );
        let strat = IteratedLocalSearchStrategy::new(cfg);
        assert_eq!(strat.name(), "IteratedLocalSearchStrategy");
        assert!(
            format!("{}", strat).contains("acceptance_criterion"),
            "Display should include inner components"
        );
    }

    #[test]
    fn test_no_perturbation_immediate_end() {
        // Perturb builder returns None; loop ends without improvement.
        #[derive(Debug)]
        struct PerturbNone;
        impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for PerturbNone {
            fn name(&self) -> &str {
                "PerturbNone"
            }
            fn next<'b, 'sm, 'c, 's, 'm, 'p>(
                &mut self,
                _ctx: &mut SearchContext<
                    'b,
                    'sm,
                    'c,
                    's,
                    'm,
                    'p,
                    i64,
                    DefaultCostEvaluator,
                    StdRng,
                >,
            ) -> Option<Plan<'p, i64>> {
                None
            }
        }

        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());
        let cfg = IteratedLocalSearchConfig::new(
            2,
            Some(10),
            Box::new(GreedyCrit::new()),
            Box::new(NeutralOnce::new()),
            Box::new(PerturbNone),
            eval,
            StdRng::seed_from_u64(2),
        );
        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let out = strat.run(&mut ctx).unwrap();
        assert_eq!(out, state);
        assert_eq!(mon.generated, 0);
        assert_eq!(mon.accepted + mon.rejected, 0);
    }

    #[test]
    fn test_accept_improvement_updates_incumbent() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());
        let before = shared.peek();
        assert_eq!(before.unassigned_requests, 2);

        let cfg = IteratedLocalSearchConfig::new(
            5,
            Some(10),
            Box::new(GreedyCrit::new()),
            Box::new(ImproveOne::new()),
            Box::new(NeutralOnce::new()),
            eval,
            StdRng::seed_from_u64(3),
        );
        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let final_state = strat.run(&mut ctx).unwrap();
        assert_eq!(mon.generated, 1, "one candidate produced");
        assert_eq!(mon.accepted, 1, "candidate accepted");
        assert_eq!(mon.rejected, 0, "no rejection");
        assert_eq!(final_state.fitness().unassigned_requests, 1);
        assert_eq!(shared.peek().unassigned_requests, 1, "incumbent improved");
    }

    #[test]
    fn test_reject_worse_with_greedy() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let cfg = IteratedLocalSearchConfig::new(
            3,
            Some(3),
            Box::new(GreedyCrit::new()),
            Box::new(WorseOnce::new()),
            Box::new(NeutralOnce::new()),
            eval,
            StdRng::seed_from_u64(4),
        );
        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let final_state = strat.run(&mut ctx).unwrap();
        assert_eq!(mon.generated, 1);
        assert_eq!(mon.accepted, 0);
        assert_eq!(mon.rejected, 1);
        assert_eq!(
            final_state.fitness().unassigned_requests,
            2,
            "state unchanged"
        );
        assert_eq!(shared.peek().unassigned_requests, 2, "incumbent unchanged");
    }

    #[test]
    fn test_stagnation_monitor_triggers_end() {
        // Multiple neutral steps -> stagnation after budget
        #[derive(Debug)]
        struct NeutralRepeat {
            remaining: usize,
        }
        impl NeutralRepeat {
            fn new(n: usize) -> Self {
                Self { remaining: n }
            }
        }
        impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for NeutralRepeat {
            fn name(&self) -> &str {
                "NeutralRepeat"
            }
            fn next<'b, 'sm, 'c, 's, 'm, 'p>(
                &mut self,
                _ctx: &mut SearchContext<
                    'b,
                    'sm,
                    'c,
                    's,
                    'm,
                    'p,
                    i64,
                    DefaultCostEvaluator,
                    StdRng,
                >,
            ) -> Option<Plan<'p, i64>> {
                if self.remaining == 0 {
                    return None;
                }
                self.remaining -= 1;
                Some(Plan::new_delta(
                    Vec::new(),
                    TerminalDelta::empty(),
                    FitnessDelta::new(0, 0),
                ))
            }
        }

        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let cfg = IteratedLocalSearchConfig::new(
            2, // stagnation budget
            Some(50),
            Box::new(GreedyCrit::new()),
            Box::new(NeutralRepeat::new(10)),
            Box::new(NeutralOnce::new()),
            eval,
            StdRng::seed_from_u64(5),
        );
        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let _ = strat.run(&mut ctx);
        assert!(mon.generated >= 1, "at least perturbation candidate");
        assert!(mon.rejected >= 1, "neutral steps treated as non-improving");
    }

    #[test]
    fn test_plan_limit_halts_improvement_phase() {
        // Use ImproveMany that could generate 5 improving moves but set max_local_steps = 1
        // Expect exactly 1 improvement: unassigned 2 -> 1
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let cfg = IteratedLocalSearchConfig::new(
            100,
            Some(1), // large stagnation; limit steps to 1 in local improvement phase
            Box::new(AlwaysCrit::new()),
            Box::new(ImproveMany::new(5)),
            Box::new(NeutralOnce::new()),
            eval,
            StdRng::seed_from_u64(6),
        );
        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let final_state = strat.run(&mut ctx).unwrap();
        // Should have applied exactly 1 improvement: unassigned from 2 -> 1
        assert_eq!(final_state.fitness().unassigned_requests, 1);
        assert_eq!(shared.peek().unassigned_requests, 1);
    }

    #[test]
    fn test_using_random_ruin_repair_pair_as_perturb_builder() {
        struct DummyRuin;
        impl RuinProcedure<i64, DefaultCostEvaluator, StdRng> for DummyRuin {
            fn name(&self) -> &str {
                "DummyRuin"
            }
            fn ruin<'b, 'r, 'c, 's, 'm, 'p>(
                &mut self,
                _ctx: &mut RuinProcedureContext<
                    'b,
                    'r,
                    'c,
                    's,
                    'm,
                    'p,
                    i64,
                    DefaultCostEvaluator,
                    StdRng,
                >,
            ) -> RuinOutcome<'p, i64> {
                RuinOutcome::new(Plan::empty(), Vec::new())
            }
        }
        struct DummyRepair;
        impl RepairProcedure<i64, DefaultCostEvaluator, StdRng> for DummyRepair {
            fn name(&self) -> &str {
                "DummyRepair"
            }
            fn repair<'b, 'r, 'c, 's, 'm, 'p>(
                &mut self,
                _ctx: &mut RepairProcedureContext<
                    'b,
                    'r,
                    'c,
                    's,
                    'm,
                    'p,
                    i64,
                    DefaultCostEvaluator,
                    StdRng,
                >,
                ruined: RuinOutcome<'p, i64>,
            ) -> Plan<'p, i64> {
                ruined.ruined_plan
            }
        }

        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let pair: RandomRuinRepairPerturbPair<i64, DefaultCostEvaluator, StdRng> =
            RandomRuinRepairPerturbPair::new(
                vec![Box::new(DummyRuin)],
                vec![Box::new(DummyRepair)],
            );

        let cfg = IteratedLocalSearchConfig::new(
            3,
            Some(5),
            Box::new(GreedyCrit::new()),
            Box::new(ImproveOne::new()),
            Box::new(pair),
            eval,
            StdRng::seed_from_u64(7),
        );

        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::with_terminate_after(1);
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let _ = strat.run(&mut ctx);
        // Expect at least one candidate generated.
        assert!(mon.generated >= 1);
        assert!(shared.peek().unassigned_requests <= 2);
        assert!(
            mon.terminated,
            "monitor should have requested termination after first candidate"
        );
    }

    #[test]
    fn test_context_invariants_inside_builder() {
        // Builder asserts SearchContext lengths and model pointer identity
        #[derive(Debug)]
        struct AssertingBuilder {
            expected_model_addr: usize,
            called: bool,
        }
        impl AssertingBuilder {
            fn new(model: &SolverModel<'_, i64>) -> Self {
                Self {
                    expected_model_addr: model as *const _ as usize,
                    called: false,
                }
            }
        }
        impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for AssertingBuilder {
            fn name(&self) -> &str {
                "AssertingBuilder"
            }
            fn next<'b, 'sm, 'c, 's, 'm, 'p>(
                &mut self,
                ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
            ) -> Option<Plan<'p, i64>> {
                self.called = true;
                let got = ctx.model() as *const _ as usize;
                assert_eq!(got, self.expected_model_addr, "model pointer mismatch");
                assert_eq!(
                    ctx.work_buf().len(),
                    ctx.model().flexible_requests_len(),
                    "work_buf size mismatch"
                );
                None
            }
        }

        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());
        let cfg = IteratedLocalSearchConfig::new(
            1,
            Some(1),
            Box::new(AlwaysCrit::new()),
            Box::new(AssertingBuilder::new(&model)),
            Box::new(NeutralOnce::new()),
            eval,
            StdRng::seed_from_u64(8),
        );
        let mut strat = IteratedLocalSearchStrategy::new(cfg);
        let mut mon = CountingMonitor::default();
        let mut ctx = StrategyContext::new(&model, &shared, &mut mon, &state);

        let final_state = strat.run(&mut ctx).unwrap();
        assert_eq!(final_state.fitness().unassigned_requests, 2);
    }
}
