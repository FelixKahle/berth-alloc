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
    engine::strategy::{Strategy, StrategyContext},
    model::solver_model::SolverModel,
    monitor::search_monitor::{
        LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
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

/// Trait for acceptance criteria layered on top of a decision builder.
/// These decide whether a candidate local plan is accepted.
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

/// Greedy descent acceptance: accept only strictly improving moves.
/// Improvement ordering: fewer unassigned first, then lower cost.
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

    #[inline]
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

/// Trivial acceptance: accept every candidate.
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

    #[inline]
    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        _context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        _plan: &Plan<'p, T>,
    ) -> bool {
        true
    }
}

/// Local step limit monitor: counts accepted steps inside a phase only.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalStepLimitMonitor {
    accepted: usize,
    limit: usize,
}

impl LocalStepLimitMonitor {
    #[inline]
    pub fn new(limit: usize) -> Self {
        Self { accepted: 0, limit }
    }
    #[inline]
    pub fn accepted(&self) -> usize {
        self.accepted
    }
}

impl TerminationCheck for LocalStepLimitMonitor {
    #[inline]
    fn should_terminate_search(&self) -> bool {
        self.accepted >= self.limit
    }
}

impl<T: Copy + Ord> PlanEventMonitor<T> for LocalStepLimitMonitor {
    fn on_plan_generated<'p>(&mut self, _: &Plan<'p, T>) {}
    fn on_plan_rejected<'p>(&mut self, _: &Plan<'p, T>) {}
    fn on_plan_accepted<'p>(&mut self, _: &Plan<'p, T>) {
        self.accepted = self.accepted.saturating_add(1);
    }
}

impl LifecycleMonitor for LocalStepLimitMonitor {
    fn on_search_start(&mut self) {
        self.accepted = 0;
    }
    fn on_search_end(&mut self) {}
}

impl<T: Copy + Ord> SearchMonitor<T> for LocalStepLimitMonitor {
    fn name(&self) -> &str {
        "LocalStepLimitMonitor"
    }
}

/// Local stagnation monitor: counts consecutive non-improving accepted moves.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalStagnationMonitor {
    consecutive_non_improving: usize,
    budget: usize,
}

impl LocalStagnationMonitor {
    pub fn new(budget: usize) -> Self {
        Self {
            consecutive_non_improving: 0,
            budget,
        }
    }
    #[inline]
    pub fn consecutive_non_improving(&self) -> usize {
        self.consecutive_non_improving
    }
    #[inline]
    pub fn mark_non_improving(&mut self) {
        self.consecutive_non_improving = self.consecutive_non_improving.saturating_add(1);
    }
    #[inline]
    pub fn reset(&mut self) {
        self.consecutive_non_improving = 0;
    }
}

impl TerminationCheck for LocalStagnationMonitor {
    #[inline]
    fn should_terminate_search(&self) -> bool {
        self.consecutive_non_improving >= self.budget
    }
}

impl<T: Copy + Ord> PlanEventMonitor<T> for LocalStagnationMonitor {
    fn on_plan_generated<'p>(&mut self, _: &Plan<'p, T>) {}
    fn on_plan_rejected<'p>(&mut self, _: &Plan<'p, T>) {}
    fn on_plan_accepted<'p>(&mut self, _: &Plan<'p, T>) {
        // Strategy decides when to mark/reset explicitly.
    }
}

impl LifecycleMonitor for LocalStagnationMonitor {
    fn on_search_start(&mut self) {
        self.consecutive_non_improving = 0;
    }
    fn on_search_end(&mut self) {}
}

impl<T: Copy + Ord> SearchMonitor<T> for LocalStagnationMonitor {
    fn name(&self) -> &str {
        "LocalStagnationMonitor"
    }
}

/// Forwards plan events to base + optional local monitors under a unified borrow.
struct ForwardingMonitor<'a, T: Copy + Ord> {
    base: &'a mut dyn SearchMonitor<T>,
    step: Option<&'a mut LocalStepLimitMonitor>,
    stag: Option<&'a mut LocalStagnationMonitor>,
}

impl<'a, T: Copy + Ord> ForwardingMonitor<'a, T> {
    #[inline]
    fn new(
        base: &'a mut dyn SearchMonitor<T>,
        step: Option<&'a mut LocalStepLimitMonitor>,
        stag: Option<&'a mut LocalStagnationMonitor>,
    ) -> Self {
        Self { base, step, stag }
    }
}

impl<'a, T: Copy + Ord> TerminationCheck for ForwardingMonitor<'a, T> {
    fn should_terminate_search(&self) -> bool {
        if self.base.should_terminate_search() {
            return true;
        }
        if let Some(s) = &self.step
            && s.should_terminate_search()
        {
            return true;
        }
        if let Some(s) = &self.stag
            && s.should_terminate_search()
        {
            return true;
        }
        false
    }
}

impl<'a, T: Copy + Ord> PlanEventMonitor<T> for ForwardingMonitor<'a, T> {
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>) {
        self.base.on_plan_generated(plan);
        if let Some(s) = &mut self.step {
            s.on_plan_generated(plan);
        }
        if let Some(s) = &mut self.stag {
            s.on_plan_generated(plan);
        }
    }
    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>) {
        self.base.on_plan_rejected(plan);
        if let Some(s) = &mut self.step {
            s.on_plan_rejected(plan);
        }
        if let Some(s) = &mut self.stag {
            s.on_plan_rejected(plan);
        }
    }
    fn on_plan_accepted<'p>(&mut self, plan: &Plan<'p, T>) {
        self.base.on_plan_accepted(plan);
        if let Some(s) = &mut self.step {
            s.on_plan_accepted(plan);
        }
        if let Some(s) = &mut self.stag {
            s.on_plan_accepted(plan);
        }
    }
}

/// Monitor that defers final accept/reject until after the ILS acceptance criterion.
struct DeferredPlanMonitor<'a, T: Copy + Ord> {
    inner: &'a mut ForwardingMonitor<'a, T>,
    pending_accept: bool,
}

impl<'a, T: Copy + Ord> DeferredPlanMonitor<'a, T> {
    #[inline]
    fn new(inner: &'a mut ForwardingMonitor<'a, T>) -> Self {
        Self {
            inner,
            pending_accept: false,
        }
    }

    #[inline]
    fn finalize_plan<'p>(&mut self, plan: &Plan<'p, T>, accepted: bool) {
        if self.pending_accept {
            if accepted {
                self.inner.on_plan_accepted(plan);
            } else {
                self.inner.on_plan_rejected(plan);
            }
            self.pending_accept = false;
        } else if accepted {
            self.inner.on_plan_accepted(plan);
        } else {
            self.inner.on_plan_rejected(plan);
        }
    }
}

impl<'a, T: Copy + Ord> TerminationCheck for DeferredPlanMonitor<'a, T> {
    fn should_terminate_search(&self) -> bool {
        self.inner.should_terminate_search()
    }
}

impl<'a, T: Copy + Ord> PlanEventMonitor<T> for DeferredPlanMonitor<'a, T> {
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>) {
        self.inner.on_plan_generated(plan);
    }
    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>) {
        self.inner.on_plan_rejected(plan);
    }
    fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, T>) {
        self.pending_accept = true;
    }
}

/// Configuration for IteratedLocalSearchStrategy.
pub struct IteratedLocalSearchConfig<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    pub max_local_stagnation_steps: usize,
    pub max_local_steps: Option<usize>,
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

/// Iterated Local Search strategy: repeatedly performs an improvement phase
/// bounded by local step/stagnation limits, then a perturbation phase to escape
/// local minima. Global termination is governed exclusively by the composite
/// search monitor provided in the strategy context.
pub struct IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    max_local_stagnation_steps: usize,
    max_local_steps: Option<usize>,
    acceptance_criterion: Box<dyn IlsAcceptanceCriterion<T, C, R> + Send>,
    improving_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    perturbing_decision_builder: Box<dyn DecisionBuilder<T, C, R> + Send + 'n>,
    evaluator: C,
    rng: R,
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

    /// Run the improvement (descent) phase.
    /// Returns (stagnated, any_generated).
    fn run_improvement_phase<'e, 'm, 'p>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
        state: &mut SolverState<'p, T>,
        work_buf: &mut [DecisionVar<T>],
    ) -> (bool, bool) {
        let mut step_limit = self.max_local_steps.map(LocalStepLimitMonitor::new);
        let mut stagnation = Some(LocalStagnationMonitor::new(self.max_local_stagnation_steps));
        let mut stagnated = false;
        let mut any_generated = false;

        loop {
            if context.monitor().should_terminate_search() {
                break;
            }
            if let Some(sl) = &step_limit
                && sl.should_terminate_search()
            {
                break;
            }
            if let Some(st) = &stagnation
                && st.should_terminate_search()
            {
                stagnated = true;
                break;
            }

            let current_fitness = *state.fitness();
            let model = context.model();

            let outcome = {
                let base_monitor = context.monitor();
                let mut fwd =
                    ForwardingMonitor::new(base_monitor, step_limit.as_mut(), stagnation.as_mut());
                let mut deferred = DeferredPlanMonitor::new(&mut fwd);

                let next_plan = {
                    let mut sc = SearchContext::new(
                        model,
                        state,
                        &self.evaluator,
                        &mut self.rng,
                        work_buf,
                        current_fitness,
                        &mut deferred,
                    );
                    self.improving_decision_builder.next(&mut sc)
                };

                match next_plan {
                    Some(plan) => {
                        any_generated = true;

                        // Notify: a candidate has been generated.
                        deferred.on_plan_generated(&plan);

                        // Acceptance
                        let mut acc_ctx = IlsAcceptanceCriterionContext::new(
                            model,
                            state,
                            &self.evaluator,
                            &mut self.rng,
                        );
                        let accepted = self.acceptance_criterion.accept(&mut acc_ctx, &plan);

                        let improved = if accepted {
                            let d = &plan.fitness_delta;
                            (d.delta_unassigned < 0)
                                || (d.delta_unassigned == 0 && d.delta_cost < 0)
                        } else {
                            false
                        };

                        deferred.finalize_plan(&plan, accepted);
                        if accepted {
                            state.apply_plan(plan);
                            let _ = context.shared_incumbent().try_update(state, model);
                        }

                        Some((accepted, improved))
                    }
                    None => None,
                }
            };

            let Some((accepted, improved)) = outcome else {
                break;
            };

            // Update stagnation monitor only for accepted moves.
            if let Some(st) = &mut stagnation
                && accepted
            {
                if improved {
                    st.reset();
                } else {
                    st.mark_non_improving();
                }
            }
        }

        (stagnated, any_generated)
    }

    /// Run the perturbation phase. Returns whether any perturbation was generated.
    fn run_perturbation_phase<'e, 'm, 'p>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
        state: &mut SolverState<'p, T>,
        work_buf: &mut [DecisionVar<T>],
    ) -> bool {
        // Single accepted perturbation move budget.
        let mut step_limit = LocalStepLimitMonitor::new(1);
        let mut any_generated = false;

        loop {
            if step_limit.should_terminate_search() {
                break;
            }
            if context.monitor().should_terminate_search() {
                break;
            }

            let current_fitness = *state.fitness();
            let model = context.model();

            let produced = {
                let base_monitor = context.monitor();
                let mut fwd = ForwardingMonitor::new(base_monitor, Some(&mut step_limit), None);
                let mut deferred = DeferredPlanMonitor::new(&mut fwd);

                let next_plan = {
                    let mut sc = SearchContext::new(
                        model,
                        state,
                        &self.evaluator,
                        &mut self.rng,
                        work_buf,
                        current_fitness,
                        &mut deferred,
                    );
                    self.perturbing_decision_builder.next(&mut sc)
                };

                match next_plan {
                    Some(plan) => {
                        any_generated = true;

                        // Notify: a candidate has been generated.
                        deferred.on_plan_generated(&plan);

                        // Unconditional accept for perturbation
                        deferred.finalize_plan(&plan, true);
                        state.apply_plan(plan);
                        let _ = context.shared_incumbent().try_update(state, model);
                        true
                    }
                    None => false,
                }
            };

            if !produced {
                break;
            }
        }

        any_generated
    }
}

impl<'n, T, C, R> From<IteratedLocalSearchConfig<'n, T, C, R>>
    for IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    fn from(config: IteratedLocalSearchConfig<'n, T, C, R>) -> Self {
        IteratedLocalSearchStrategy::new(config)
    }
}

impl<'n, T, C, R> Strategy<T> for IteratedLocalSearchStrategy<'n, T, C, R>
where
    T: SolveNumeric,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "IteratedLocalSearchStrategy"
    }

    fn run<'p, 'e, 'm>(
        &mut self,
        context: &mut StrategyContext<'e, 'm, 'p, T>,
    ) -> Option<SolverState<'p, T>> {
        context.monitor().on_search_start();

        let mut work_buf = vec![DecisionVar::unassigned(); context.model().flexible_requests_len()];
        let mut state = context.state().clone();

        loop {
            if context.monitor().should_terminate_search() {
                break;
            }

            let (_stagnated, improved_generated) =
                self.run_improvement_phase(context, &mut state, &mut work_buf);

            if context.monitor().should_terminate_search() {
                break;
            }

            let perturb_generated = self.run_perturbation_phase(context, &mut state, &mut work_buf);

            // Natural completion: if neither phase produced any candidates, we are done.
            if !improved_generated && !perturb_generated {
                break;
            }
        }

        context.monitor().on_search_end();
        Some(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::shared_incumbent::SharedIncumbent,
        model::{
            index::{BerthIndex, RequestIndex},
            solver_model::SolverModel,
        },
        monitor::search_monitor::{
            LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
        },
        search::eval::DefaultCostEvaluator,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::{Fitness, FitnessDelta},
            plan::{DecisionVarPatch, Plan},
            solver_state::{SolverState, SolverStateView},
            terminal::{delta::TerminalDelta, terminalocc::TerminalOccupancy},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, Request, RequestIdentifier},
        problem::builder::ProblemBuilder,
    };
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::BTreeMap;

    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    fn iv(start: i64, end: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(start), tp(end))
    }
    fn bid(v: u32) -> BerthIdentifier {
        BerthIdentifier::new(v)
    }
    fn rid(v: u32) -> RequestIdentifier {
        RequestIdentifier::new(v)
    }

    fn problem_one_berth_three_flex() -> Problem<i64> {
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);

        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), TimeDelta::new(10));
        let r1 = Request::<FlexibleKind, i64>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), TimeDelta::new(5));
        let r2 = Request::<FlexibleKind, i64>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut pt3 = BTreeMap::new();
        pt3.insert(bid(1), TimeDelta::new(2));
        let r3 = Request::<FlexibleKind, i64>::new(rid(3), iv(0, 200), 1, pt3).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        builder.add_flexible(r3);
        builder.build().expect("valid problem")
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

        let base_fit = eval.eval_fitness(&model, &dvars);
        let start_cost = if base_fit.cost <= 0 { 1 } else { base_fit.cost };
        let fitness = Fitness::new(start_cost, base_fit.unassigned_requests);

        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    #[derive(Debug)]
    struct SequentialAssignBuilder {
        next_request: usize,
        limit: usize,
        name: &'static str,
    }
    impl SequentialAssignBuilder {
        fn new(limit: usize, name: &'static str) -> Self {
            Self {
                next_request: 0,
                limit,
                name,
            }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for SequentialAssignBuilder {
        fn name(&self) -> &str {
            self.name
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.next_request >= self.limit {
                return None;
            }
            let req_ix = self.next_request;
            self.next_request += 1;
            let patch = DecisionVarPatch::new(
                RequestIndex::new(req_ix),
                DecisionVar::assigned(BerthIndex::new(0), tp(0)),
            );
            let fitness_delta = FitnessDelta::new(0, -1);
            Some(Plan::new_delta(
                vec![patch],
                TerminalDelta::empty(),
                fitness_delta,
            ))
        }
    }

    #[derive(Debug)]
    struct NeutralMoveBuilder {
        remaining: usize,
        name: &'static str,
    }
    impl NeutralMoveBuilder {
        fn new(remaining: usize, name: &'static str) -> Self {
            Self { remaining, name }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for NeutralMoveBuilder {
        fn name(&self) -> &str {
            self.name
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            let fitness_delta = FitnessDelta::new(0, 0);
            Some(Plan::new_delta(
                Vec::new(),
                TerminalDelta::empty(),
                fitness_delta,
            ))
        }
    }

    #[derive(Debug)]
    struct WorseningMoveBuilder {
        remaining: usize,
        name: &'static str,
    }
    impl WorseningMoveBuilder {
        fn new(remaining: usize, name: &'static str) -> Self {
            Self { remaining, name }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for WorseningMoveBuilder {
        fn name(&self) -> &str {
            self.name
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            let fitness_delta = FitnessDelta::new(5, 0); // worsening cost
            Some(Plan::new_delta(
                Vec::new(),
                TerminalDelta::empty(),
                fitness_delta,
            ))
        }
    }

    #[derive(Debug)]
    struct SinglePerturbBuilder {
        emitted: bool,
    }
    impl SinglePerturbBuilder {
        fn new() -> Self {
            Self { emitted: false }
        }
    }
    impl DecisionBuilder<i64, DefaultCostEvaluator, StdRng> for SinglePerturbBuilder {
        fn name(&self) -> &str {
            "SinglePerturb"
        }
        fn next<'b, 'sm, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut SearchContext<'b, 'sm, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            if self.emitted {
                return None;
            }
            self.emitted = true;
            let fitness_delta = FitnessDelta::new(0, 0);
            Some(Plan::new_delta(
                Vec::new(),
                TerminalDelta::empty(),
                fitness_delta,
            ))
        }
    }

    #[derive(Default)]
    struct RecordingMonitor {
        started: usize,
        ended: usize,
        generated: usize,
        accepted: usize,
        rejected: usize,
        terminate_after_generated: Option<usize>,
        terminated: bool,
    }
    impl RecordingMonitor {
        fn with_terminate_after(n: usize) -> Self {
            Self {
                terminate_after_generated: Some(n),
                ..Default::default()
            }
        }
    }
    impl TerminationCheck for RecordingMonitor {
        fn should_terminate_search(&self) -> bool {
            self.terminated
        }
    }
    impl LifecycleMonitor for RecordingMonitor {
        fn on_search_start(&mut self) {
            self.started += 1;
        }
        fn on_search_end(&mut self) {
            self.ended += 1;
        }
    }
    impl<T: Copy + Ord> PlanEventMonitor<T> for RecordingMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, T>) {
            self.generated += 1;
            if let Some(limit) = self.terminate_after_generated {
                if self.generated >= limit {
                    self.terminated = true;
                }
            }
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, T>) {
            self.rejected += 1;
        }
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, T>) {
            self.accepted += 1;
        }
    }
    impl<T: Copy + Ord> SearchMonitor<T> for RecordingMonitor {
        fn name(&self) -> &str {
            "RecordingMonitor"
        }
    }

    fn make_shared<'p>(
        model: &SolverModel<'p, i64>,
        state: &SolverState<'p, i64>,
    ) -> SharedIncumbent<'p, i64> {
        SharedIncumbent::new(state.clone()).tap(|inc| {
            let _ = inc.try_update(state, model);
        })
    }

    trait Tap: Sized {
        fn tap<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
            f(&mut self);
            self
        }
    }
    impl<T> Tap for T {}

    #[test]
    fn test_ils_strategy_name() {
        let problem = problem_one_berth_three_flex();
        let (_model, _state, eval) = make_model_state_eval(&problem);
        let improving = Box::new(SequentialAssignBuilder::new(0, "SeqAssign"));
        let perturb = Box::new(SinglePerturbBuilder::new());
        let rng = StdRng::seed_from_u64(7);
        let config = IteratedLocalSearchConfig::new(
            5,
            Some(10),
            Box::new(AlwaysAcceptAcceptanceCriterion::new()),
            improving,
            perturb,
            eval,
            rng,
        );
        let strategy = IteratedLocalSearchStrategy::new(config);
        assert_eq!(strategy.name(), "IteratedLocalSearchStrategy");
    }

    #[test]
    fn test_improvement_phase_respects_step_limit() {
        let problem = problem_one_berth_three_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = make_shared(&model, &state);

        // Improvement builder can produce 3 improving moves.
        let improving = Box::new(SequentialAssignBuilder::new(3, "SeqAssign"));
        // Perturb builder (single move).
        let perturb = Box::new(SinglePerturbBuilder::new());

        let rng = StdRng::seed_from_u64(10);
        let config = IteratedLocalSearchConfig::new(
            10,      // large stagnation limit
            Some(1), // step limit: only 1 improvement accepted per phase
            Box::new(AlwaysAcceptAcceptanceCriterion::new()),
            improving,
            perturb,
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        // Use a monitor to globally stop after we see the two plans this test cares about
        // (one improving candidate + one perturbation candidate).
        let mut monitor = RecordingMonitor::with_terminate_after(2);

        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);
        let final_state = strategy.run(&mut ctx).unwrap();

        assert!(
            final_state.version().value() >= 2,
            "expected at least two applied plans (1 improvement + 1 perturbation)"
        );
        assert_eq!(monitor.started, 1);
        assert_eq!(monitor.ended, 1);
    }

    #[test]
    fn test_stagnation_limit_stops_improvement() {
        let problem = problem_one_berth_three_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let improving = Box::new(NeutralMoveBuilder::new(8, "Neutral"));
        let perturb = Box::new(SinglePerturbBuilder::new());

        let rng = StdRng::seed_from_u64(11);
        let stagnation_budget = 3;
        let config = IteratedLocalSearchConfig::new(
            stagnation_budget,
            None,
            Box::new(AlwaysAcceptAcceptanceCriterion::new()),
            improving,
            perturb,
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let mut monitor = RecordingMonitor::with_terminate_after(10);
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);
        let final_state = strategy.run(&mut ctx).unwrap();

        assert!(
            final_state.version().value() >= stagnation_budget as u64 + 1,
            "expected stagnation budget moves plus perturbation"
        );
        let incumbent = shared.peek();
        assert_eq!(
            incumbent.unassigned_requests,
            state.fitness().unassigned_requests,
            "neutral moves must not change unassigned count"
        );
        assert!(monitor.started == 1 && monitor.ended == 1);
        assert!(monitor.generated <= 10);
    }

    #[test]
    fn test_greedy_acceptance_rejects_worsening_moves() {
        let problem = problem_one_berth_three_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = SharedIncumbent::new(state.clone());

        let improving = Box::new(WorseningMoveBuilder::new(6, "Worse"));
        let perturb = Box::new(SinglePerturbBuilder::new());

        let rng = StdRng::seed_from_u64(12);
        let config = IteratedLocalSearchConfig::new(
            5,
            Some(10),
            Box::new(GreedyDescentAcceptanceCriterion::new()),
            improving,
            perturb,
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let mut monitor = RecordingMonitor::with_terminate_after(12);
        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);
        let final_state = strategy.run(&mut ctx).unwrap();

        assert_eq!(
            final_state.version().value(),
            state.version().value() + 1,
            "expected exactly one applied perturbation"
        );
        let incumbent = shared.peek();
        assert_eq!(
            incumbent.unassigned_requests,
            state.fitness().unassigned_requests,
            "unassigned count should remain unchanged"
        );
        assert!(monitor.rejected > 0);
        assert!(monitor.generated <= 12);
    }

    #[test]
    fn test_monitor_termination_stops_loop() {
        let problem = problem_one_berth_three_flex();
        let (model, state, eval) = make_model_state_eval(&problem);
        let shared = make_shared(&model, &state);

        let improving = Box::new(SequentialAssignBuilder::new(50, "SeqAssign"));
        let perturb = Box::new(SinglePerturbBuilder::new());
        let rng = StdRng::seed_from_u64(13);

        let mut monitor = RecordingMonitor::with_terminate_after(2);

        let config = IteratedLocalSearchConfig::new(
            10,
            Some(100),
            Box::new(AlwaysAcceptAcceptanceCriterion::new()),
            improving,
            perturb,
            eval,
            rng,
        );
        let mut strategy = IteratedLocalSearchStrategy::new(config);

        let mut ctx = StrategyContext::new(&model, &shared, &mut monitor, &state);
        let _ = strategy.run(&mut ctx);

        assert!(monitor.terminated);
        assert_eq!(monitor.started, 1);
        assert_eq!(monitor.ended, 1);
        assert!(monitor.generated >= 2);
    }
}
