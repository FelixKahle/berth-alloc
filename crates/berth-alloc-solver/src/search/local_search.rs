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
        decision_builder::{DecisionBuilder, SearchContext},
        eval::CostEvaluator,
        filter::{FilterContext, NeighborhoodFilterStack},
        metaheuristic::{Metaheuristic, MetaheuristicContext},
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::{decisionvar::DecisionVar, plan::Plan},
};

#[derive(Debug)]
pub struct MetaheuristicLocalSearch<T, C, M, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    M: Metaheuristic<T, C>,
    R: rand::Rng,
{
    local_search_operator: Box<dyn LocalSearchOperator<T, C, R>>,
    filter_stack: NeighborhoodFilterStack<T>,
    metaheuristic: M,
}

impl<T, C, M, R> MetaheuristicLocalSearch<T, C, M, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    M: Metaheuristic<T, C>,
    R: rand::Rng,
{
    pub fn new(
        local_search_operator: Box<dyn LocalSearchOperator<T, C, R>>,
        filter_stack: NeighborhoodFilterStack<T>,
        metaheuristic: M,
    ) -> MetaheuristicLocalSearch<T, C, M, R> {
        MetaheuristicLocalSearch {
            local_search_operator,
            filter_stack,
            metaheuristic,
        }
    }
}

impl<T, C, M, R> DecisionBuilder<T, C, R> for MetaheuristicLocalSearch<T, C, M, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    M: Metaheuristic<T, C>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "MetaheuristicLocalSearch"
    }

    fn next<'b, 't, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut SearchContext<'b, 't, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let model = context.model;
        let state = context.state;
        let evaluator = context.evaluator;
        let rng: &mut R = context.rng;
        let work_buf: &mut [DecisionVar<T>] = context.work_buf;

        let mut op_ctx = OperatorContext::new(model, state, evaluator, rng, work_buf);

        if !self
            .metaheuristic
            .local_optimum_reached(MetaheuristicContext::new(model, state, evaluator))
        {
            return None;
        }

        self.local_search_operator.reset();
        self.local_search_operator.synchronize(&mut op_ctx);

        loop {
            while let Some(plan) = self.local_search_operator.make_next_neighbor(&mut op_ctx) {
                if context.term.tick_neighbor() {
                    return None;
                }

                if !self
                    .filter_stack
                    .accept(FilterContext::new(model, state), &plan)
                {
                    continue;
                }

                let mh_ctx = MetaheuristicContext::new(model, state, evaluator);
                if self.metaheuristic.accept_plan(mh_ctx, &plan) {
                    return Some(plan);
                }
            }

            let cont = self
                .metaheuristic
                .local_optimum_reached(MetaheuristicContext::new(model, state, evaluator));
            if !cont {
                return None;
            }

            self.local_search_operator.reset();
            self.local_search_operator.synchronize(&mut op_ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        monitor::{
            controller::{GlobalController, SearchLimits},
            termination::Termination,
        },
        search::eval::DefaultCostEvaluator,
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            plan::Plan,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{asg::Assignment, req::Request},
    };
    use rand::RngCore;
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::BTreeMap;

    // Helpers to build a tiny problem/state similar to lns.rs tests
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
        // One berth with broad window
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);
        // Two flexible requests, both allowed on berth 1
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, i64>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(5));
        let r2 = Request::<FlexibleKind, i64>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(b1);

        let fixed = berth_alloc_model::problem::asg::AssignmentContainer::<
            _,
            i64,
            Assignment<_, i64>,
        >::new();
        let mut flex = berth_alloc_model::problem::req::RequestContainer::<
            i64,
            Request<FlexibleKind, i64>,
        >::new();
        flex.insert(r1);
        flex.insert(r2);

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn make_state(
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
        let fitness = Fitness::new(0, model.flexible_requests_len());
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    // Dummy Metaheuristics

    // Returns false immediately; next() must return None without touching operator.
    struct AlwaysStopMH;
    impl Metaheuristic<i64, DefaultCostEvaluator> for AlwaysStopMH {
        fn name(&self) -> &str {
            "AlwaysStopMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
        ) -> bool {
            false
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
            _plan: &Plan<'_, i64>,
        ) -> bool {
            false
        }
    }

    // Allows search to proceed and accepts the first plan it sees.
    struct AcceptFirstMH {
        accepted: bool,
    }
    impl AcceptFirstMH {
        fn new() -> Self {
            Self { accepted: false }
        }
    }
    impl Metaheuristic<i64, DefaultCostEvaluator> for AcceptFirstMH {
        fn name(&self) -> &str {
            "AcceptFirstMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
        ) -> bool {
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
            _plan: &Plan<'_, i64>,
        ) -> bool {
            if !self.accepted {
                self.accepted = true;
                true
            } else {
                false
            }
        }
    }

    // Accept the second plan encountered (reject the first).
    struct AcceptSecondMH {
        seen: usize,
    }
    impl AcceptSecondMH {
        fn new() -> Self {
            Self { seen: 0 }
        }
    }
    impl Metaheuristic<i64, DefaultCostEvaluator> for AcceptSecondMH {
        fn name(&self) -> &str {
            "AcceptSecondMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
        ) -> bool {
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
            _plan: &Plan<'_, i64>,
        ) -> bool {
            self.seen += 1;
            self.seen == 2
        }
    }

    // Allows two rounds (initial gate + one continuation); accepts any plan.
    struct ContinueOnceAcceptAlwaysMH {
        budget: usize,
    }
    impl ContinueOnceAcceptAlwaysMH {
        fn new() -> Self {
            Self { budget: 2 } // initial + one continue
        }
    }
    impl Metaheuristic<i64, DefaultCostEvaluator> for ContinueOnceAcceptAlwaysMH {
        fn name(&self) -> &str {
            "ContinueOnceAcceptAlwaysMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
        ) -> bool {
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
            _plan: &Plan<'_, i64>,
        ) -> bool {
            true
        }
    }

    // General continuation budget that never accepts.
    struct ContinueWithBudgetMH {
        budget: usize,
    }
    impl ContinueWithBudgetMH {
        fn new(budget: usize) -> Self {
            Self { budget }
        }
    }
    impl Metaheuristic<i64, DefaultCostEvaluator> for ContinueWithBudgetMH {
        fn name(&self) -> &str {
            "ContinueWithBudgetMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
        ) -> bool {
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
            _plan: &Plan<'_, i64>,
        ) -> bool {
            false
        }
    }

    // A test operator that yields a configured number of neighbors per "round".
    // Round advancement is tied to reset() + synchronize() calls performed by the search.
    struct CountingOperator {
        // For round k, yield schedule[k] neighbors, then None.
        schedule: Vec<usize>,
        round: usize,
        yielded_in_round: usize,
        advance_round_on_sync: bool,
        sync_calls: usize,
        reset_calls: usize,
        make_calls: usize,
    }

    impl CountingOperator {
        fn new(schedule: Vec<usize>) -> Self {
            Self {
                schedule,
                round: usize::MAX, // sentinel so first synchronize sets round = 0
                yielded_in_round: 0,
                advance_round_on_sync: false,
                sync_calls: 0,
                reset_calls: 0,
                make_calls: 0,
            }
        }
        fn current_quota(&self) -> usize {
            self.schedule.get(self.round).copied().unwrap_or(0)
        }
    }

    impl LocalSearchOperator<i64, DefaultCostEvaluator, StdRng> for CountingOperator {
        fn name(&self) -> &str {
            "CountingOperator"
        }

        fn has_fragments(&self) -> bool {
            false
        }

        fn reset(&mut self) {
            self.reset_calls += 1;
            self.advance_round_on_sync = true;
        }

        fn synchronize<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) {
            self.sync_calls += 1;
            // First synchronize establishes round 0; subsequent ones advance after a reset.
            if self.round == usize::MAX {
                self.round = 0;
                self.advance_round_on_sync = false;
            } else if self.advance_round_on_sync {
                self.round = self.round.saturating_add(1);
                self.advance_round_on_sync = false;
            }
            self.yielded_in_round = 0;
        }

        fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, i64>> {
            self.make_calls += 1;
            if self.yielded_in_round < self.current_quota() {
                self.yielded_in_round += 1;
                return Some(Plan::empty());
            }
            None
        }
    }

    // NeighborhoodFilterStack is used only via accept(). Default stack should accept all.
    fn default_filter_stack() -> NeighborhoodFilterStack<i64> {
        NeighborhoodFilterStack::default()
    }

    #[test]
    fn test_next_returns_none_when_metaheuristic_not_ready() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(0);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        let op = Box::new(CountingOperator::new(vec![3])); // would yield neighbors, but shouldn't be called
        let filter_stack = default_filter_stack();
        let mh = AlwaysStopMH;

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);

        let res = search.next(&mut ctx);
        assert!(res.is_none(), "expected None when MH not ready");
    }

    #[test]
    fn test_next_returns_first_accepted_neighbor() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(1);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        // One round with 2 neighbors; MH accepts first.
        let op = Box::new(CountingOperator::new(vec![2]));
        let filter_stack = default_filter_stack();
        let mh = AcceptFirstMH::new();

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);

        let res = search.next(&mut ctx);
        assert!(res.is_some(), "expected Some(plan)");
        // Plan is synthesized by the operator as Plan::empty(); any Some is acceptable.
    }

    #[test]
    fn test_next_accepts_second_neighbor() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(11);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        let op = Box::new(CountingOperator::new(vec![3]));
        let filter_stack = default_filter_stack();
        let mh = AcceptSecondMH::new();

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);

        let res = search.next(&mut ctx);
        assert!(
            res.is_some(),
            "expected Some(plan) when second neighbor is accepted"
        );
    }

    #[test]
    fn test_next_retries_after_reset_when_first_round_has_no_neighbors() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(2);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        // Round 0: 0 neighbors; Round 1: 1 neighbor (accepted).
        let op = Box::new(CountingOperator::new(vec![0, 1]));
        let filter_stack = default_filter_stack();
        let mh = ContinueOnceAcceptAlwaysMH::new();

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);

        let res = search.next(&mut ctx);
        assert!(res.is_some(), "expected Some(plan) on second round");
    }

    // Metaheuristic that allows exactly one round and rejects every plan.
    struct RejectAllMH {
        budget: usize,
    }
    impl RejectAllMH {
        fn new() -> Self {
            Self { budget: 1 } // allow initial gate, then stop
        }
    }
    impl Metaheuristic<i64, DefaultCostEvaluator> for RejectAllMH {
        fn name(&self) -> &str {
            "RejectAllMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
        ) -> bool {
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, i64, DefaultCostEvaluator>,
            _plan: &Plan<'_, i64>,
        ) -> bool {
            false
        }
    }

    #[test]
    fn test_next_rejects_all_neighbors_and_returns_none() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(3);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        // Operator yields two neighbors in the only round; MH rejects all and then stops.
        let op = Box::new(CountingOperator::new(vec![2]));
        let filter_stack = default_filter_stack();
        let mh = RejectAllMH::new();

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);
        let res = search.next(&mut ctx);
        assert!(res.is_none(), "expected None when MH rejects all neighbors");
    }

    #[test]
    fn test_next_multiple_empty_rounds_then_none() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(12);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        // Three rounds, all with zero neighbors; MH allows exactly three rounds.
        let op = Box::new(CountingOperator::new(vec![0, 0, 0]));
        let filter_stack = default_filter_stack();
        let mh = ContinueWithBudgetMH::new(3);

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);
        let res = search.next(&mut ctx);
        assert!(
            res.is_none(),
            "expected None when all rounds yield no neighbors"
        );
    }

    #[test]
    fn test_local_search_context_accessors_and_mutability() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(4);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = Fitness::new(123, 456);

        let ctrl = GlobalController::new(SearchLimits::default_fast());
        let mut term = Termination::from_controller(ctrl);

        let mut ctx = SearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
            &mut term,
        );

        // Accessors return the same references
        assert!(std::ptr::eq(ctx.model(), &model));
        assert!(std::ptr::eq(ctx.state(), &state));
        assert!(std::ptr::eq(ctx.evaluator(), &eval));

        // RNG is mutable and callable
        let _n1 = ctx.rng().next_u32();

        // Work buffer is the same length as decision variables
        let len_dv = model.flexible_requests_len();
        assert_eq!(ctx.work_buf().len(), len_dv);

        // Current fitness is what we passed in
        assert_eq!(ctx.current_fitness(), current_fitness);

        // Termination fast-path should be false initially
        assert!(!ctx.term.should_stop_fast());
    }

    #[test]
    fn test_trait_debug_and_display_for_local_search() {
        let op = Box::new(CountingOperator::new(vec![1]));
        let filter_stack = default_filter_stack();
        let mh = AcceptFirstMH::new();

        let search = MetaheuristicLocalSearch::new(op, filter_stack, mh);
        let ls: &dyn DecisionBuilder<i64, DefaultCostEvaluator, StdRng> = &search;

        assert_eq!(
            format!("{:?}", ls),
            "DecisionBuilder(MetaheuristicLocalSearch)"
        );
        assert_eq!(
            format!("{}", ls),
            "DecisionBuilder(MetaheuristicLocalSearch)"
        );
    }
}
