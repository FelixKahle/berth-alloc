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
        eval::CostEvaluator,
        filter::{FilterContext, NeighborhoodFilterStack},
        metaheuristic::{Metaheuristic, MetaheuristicContext},
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::{decisionvar::DecisionVar, fitness::Fitness, plan::Plan, solver_state::SolverState},
};

#[derive(Debug)]
pub struct LocalSearchContext<'b, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    model: &'m SolverModel<'p, T>,
    state: &'s SolverState<'p, T>,
    evaluator: &'c C,
    rng: &'b mut R,
    work_buf: &'b mut [DecisionVar<T>],
    current_fitness: Fitness,
}

impl<'b, 'c, 's, 'm, 'p, T, C, R> LocalSearchContext<'b, 'c, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        state: &'s SolverState<'p, T>,
        evaluator: &'c C,
        rng: &'b mut R,
        work_buf: &'b mut [DecisionVar<T>],
        current_fitness: Fitness,
    ) -> LocalSearchContext<'b, 'c, 's, 'm, 'p, T, C, R> {
        LocalSearchContext {
            model,
            state,
            evaluator,
            rng,
            work_buf,
            current_fitness,
        }
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn state(&self) -> &'s SolverState<'p, T> {
        self.state
    }

    #[inline]
    pub fn evaluator(&self) -> &'c C {
        self.evaluator
    }

    #[inline]
    pub fn rng(&mut self) -> &mut R {
        self.rng
    }

    #[inline]
    pub fn work_buf(&mut self) -> &mut [DecisionVar<T>] {
        self.work_buf
    }

    #[inline]
    pub fn current_fitness(&self) -> Fitness {
        self.current_fitness
    }
}

pub trait LocalSearch<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn next<'b, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut LocalSearchContext<'b, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>>;
}

impl<T, C, R> std::fmt::Debug for dyn LocalSearch<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearch({})", self.name())
    }
}

impl<T, C, R> std::fmt::Display for dyn LocalSearch<T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearch({})", self.name())
    }
}

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

impl<T, C, M, R> LocalSearch<T, C, R> for MetaheuristicLocalSearch<T, C, M, R>
where
    T: Copy + Ord + std::fmt::Debug,
    C: CostEvaluator<T>,
    M: Metaheuristic<T, C>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "MetaheuristicLocalSearch"
    }

    fn next<'b, 'c, 's, 'm, 'p>(
        &mut self,
        context: &mut LocalSearchContext<'b, 'c, 's, 'm, 'p, T, C, R>,
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
        search::{eval::DefaultCostEvaluator, operator::OperatorStateVersion},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
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
    use rand::{SeedableRng, rngs::StdRng};
    use std::cell::Cell;
    use std::collections::BTreeMap;

    type T = i64;

    // Helpers to build a tiny problem/state similar to lns.rs tests
    #[inline]
    fn tp(v: i64) -> TimePoint<T> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<T> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<T> {
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

    fn problem_one_berth_two_flex() -> Problem<T> {
        // One berth with broad window
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);
        // Two flexible requests, both allowed on berth 1
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, T>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(5));
        let r2 = Request::<FlexibleKind, T>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(b1);

        let fixed =
            berth_alloc_model::problem::asg::AssignmentContainer::<_, T, Assignment<_, T>>::new();
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<T, Request<FlexibleKind, T>>::new();
        flex.insert(r1);
        flex.insert(r2);

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn make_state(
        problem: &Problem<T>,
    ) -> (SolverModel<'_, T>, SolverState<'_, T>, DefaultCostEvaluator) {
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
    impl Metaheuristic<T, DefaultCostEvaluator> for AlwaysStopMH {
        fn name(&self) -> &str {
            "AlwaysStopMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
        ) -> bool {
            false
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
            _plan: &Plan<'_, T>,
        ) -> bool {
            false
        }
    }

    // Allows search to proceed and accepts the first plan it sees.
    struct AcceptFirstMH {
        accepted: Cell<bool>,
    }
    impl AcceptFirstMH {
        fn new() -> Self {
            Self {
                accepted: Cell::new(false),
            }
        }
    }
    impl Metaheuristic<T, DefaultCostEvaluator> for AcceptFirstMH {
        fn name(&self) -> &str {
            "AcceptFirstMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
        ) -> bool {
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
            _plan: &Plan<'_, T>,
        ) -> bool {
            if !self.accepted.get() {
                self.accepted.set(true);
                true
            } else {
                false
            }
        }
    }

    // Allows two rounds (initial gate + one continuation); accepts any plan.
    struct ContinueOnceAcceptAlwaysMH {
        budget: Cell<usize>,
    }
    impl ContinueOnceAcceptAlwaysMH {
        fn new() -> Self {
            Self {
                budget: Cell::new(2),
            }
        } // initial + one continue
    }
    impl Metaheuristic<T, DefaultCostEvaluator> for ContinueOnceAcceptAlwaysMH {
        fn name(&self) -> &str {
            "ContinueOnceAcceptAlwaysMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
        ) -> bool {
            let b = self.budget.get();
            if b == 0 {
                return false;
            }
            self.budget.set(b - 1);
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
            _plan: &Plan<'_, T>,
        ) -> bool {
            true
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
        pub sync_calls: Cell<usize>,
        pub reset_calls: Cell<usize>,
        pub make_calls: Cell<usize>,
    }

    impl CountingOperator {
        fn new(schedule: Vec<usize>) -> Self {
            Self {
                schedule,
                round: usize::MAX, // sentinel so first synchronize sets round = 0
                yielded_in_round: 0,
                advance_round_on_sync: false,
                sync_calls: Cell::new(0),
                reset_calls: Cell::new(0),
                make_calls: Cell::new(0),
            }
        }
        fn current_quota(&self) -> usize {
            self.schedule.get(self.round).copied().unwrap_or(0)
        }
    }

    impl LocalSearchOperator<T, DefaultCostEvaluator, StdRng> for CountingOperator {
        fn name(&self) -> &str {
            "CountingOperator"
        }

        fn state_version(&self) -> OperatorStateVersion {
            OperatorStateVersion::new(self.round as u64)
        }

        fn has_fragments(&self) -> bool {
            false
        }

        fn reset(&mut self) {
            self.reset_calls.set(self.reset_calls.get() + 1);
            self.advance_round_on_sync = true;
        }

        fn synchronize<'b, 'r, 'c, 's, 'm, 'p>(
            &mut self,
            _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, DefaultCostEvaluator, StdRng>,
        ) {
            self.sync_calls.set(self.sync_calls.get() + 1);
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
            _ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, DefaultCostEvaluator, StdRng>,
        ) -> Option<Plan<'p, T>> {
            self.make_calls.set(self.make_calls.get() + 1);
            if self.yielded_in_round < self.current_quota() {
                self.yielded_in_round += 1;
                return Some(Plan::empty());
            }
            None
        }
    }

    // NeighborhoodFilterStack is used only via accept(). Default stack should accept all.
    fn default_filter_stack() -> NeighborhoodFilterStack<T> {
        NeighborhoodFilterStack::default()
    }

    #[test]
    fn test_next_returns_none_when_metaheuristic_not_ready() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(0);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let mut ctx = LocalSearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
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

        let mut ctx = LocalSearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
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
    fn test_next_retries_after_reset_when_first_round_has_no_neighbors() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(2);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let mut ctx = LocalSearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
        );

        // Round 0: 0 neighbors; Round 1: 1 neighbor (accepted).
        let op = Box::new(CountingOperator::new(vec![0, 1]));
        let filter_stack = default_filter_stack();
        let mh = ContinueOnceAcceptAlwaysMH::new();

        let mut search = MetaheuristicLocalSearch::new(op, filter_stack, mh);

        let res = search.next(&mut ctx);
        assert!(res.is_some(), "expected Some(plan) on second round");
    }

    use rand::RngCore;

    // Metaheuristic that allows exactly one round and rejects every plan.
    struct RejectAllMH {
        budget: Cell<usize>,
    }
    impl RejectAllMH {
        fn new() -> Self {
            Self {
                budget: Cell::new(1),
            } // allow initial gate, then stop
        }
    }
    impl Metaheuristic<T, DefaultCostEvaluator> for RejectAllMH {
        fn name(&self) -> &str {
            "RejectAllMH"
        }
        fn local_optimum_reached(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
        ) -> bool {
            let b = self.budget.get();
            if b == 0 {
                return false;
            }
            self.budget.set(b - 1);
            true
        }
        fn accept_plan(
            &mut self,
            _ctx: MetaheuristicContext<'_, '_, '_, '_, T, DefaultCostEvaluator>,
            _plan: &Plan<'_, T>,
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

        let mut ctx = LocalSearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
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
    fn test_local_search_context_accessors_and_mutability() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(4);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];

        let current_fitness = Fitness::new(123, 456);
        let mut ctx = LocalSearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
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
    }

    #[test]
    fn test_trait_debug_and_display_for_local_search() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_state(&problem);
        let mut rng = StdRng::seed_from_u64(5);
        let mut work_buf = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let current_fitness = *state.fitness();

        let mut _ctx = LocalSearchContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut work_buf,
            current_fitness,
        );

        let op = Box::new(CountingOperator::new(vec![1]));
        let filter_stack = default_filter_stack();
        let mh = AcceptFirstMH::new();

        let search = MetaheuristicLocalSearch::new(op, filter_stack, mh);
        let ls: &dyn LocalSearch<T, DefaultCostEvaluator, StdRng> = &search;

        assert_eq!(format!("{:?}", ls), "LocalSearch(MetaheuristicLocalSearch)");
        assert_eq!(format!("{}", ls), "LocalSearch(MetaheuristicLocalSearch)");
    }
}
