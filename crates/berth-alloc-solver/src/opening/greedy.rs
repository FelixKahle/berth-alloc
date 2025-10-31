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
    model::{
        index::{BerthIndex, RequestIndex},
        solver_model::SolverModel,
    },
    opening::opening_strategy::OpeningStrategy,
    state::{
        berth::err::BerthUpdateError,
        decisionvar::{Decision, DecisionVar, DecisionVarVec},
        fitness::Fitness,
        solver_state::SolverState,
        terminal::terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite},
    },
};
use berth_alloc_core::prelude::{Cost, TimeInterval};
use berth_alloc_model::problem::asg::AssignmentView;
use num_traits::{CheckedAdd, CheckedSub, Zero};
use std::{cmp::Reverse, ops::Mul};

#[derive(Debug, Clone)]
pub struct GreedyOpening<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for GreedyOpening<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GreedyOpening<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> OpeningStrategy<T> for GreedyOpening<T>
where
    T: Copy
        + Ord
        + CheckedAdd
        + CheckedSub
        + Zero
        + Into<Cost>
        + Mul<Output = Cost>
        + std::fmt::Debug,
{
    type Error = BerthUpdateError<T>;

    fn build<'m, 'p>(
        &self,
        model: &'m SolverModel<'p, T>,
    ) -> Result<SolverState<'p, T>, Self::Error> {
        let index_manager = model.index_manager();

        let mut terminal: TerminalOccupancy<'p, T> =
            TerminalOccupancy::new(model.problem().berths().iter());
        for a in model.problem().iter_fixed_assignments() {
            let bi: BerthIndex = index_manager
                .berth_index(a.berth_id())
                .expect("fixed assignment berth must be indexable");
            terminal.occupy(bi, a.interval())?;
        }

        let mut dvars = DecisionVarVec::from(vec![
            DecisionVar::unassigned();
            model.flexible_requests_len()
        ]);

        let mut reqs: Vec<RequestIndex> = (0..model.flexible_requests_len())
            .map(RequestIndex::new)
            .collect();
        reqs.sort_by_key(|&ri| {
            let w = model.feasible_interval(ri);
            let window_len = w.end().value() - w.start().value(); // T
            let min_pt = model
                .allowed_berth_indices(ri)
                .iter()
                .filter_map(|&bi| model.processing_time(ri, bi))
                .map(|pt| pt.value())
                .min()
                .unwrap_or_else(T::zero);
            let slack = window_len - min_pt;
            let weight = model.weight(ri);
            (slack, Reverse(weight), ri.get())
        });

        loop {
            let mut placed_in_pass = 0usize;

            for &ri in &reqs {
                if dvars[ri].is_assigned() {
                    continue;
                }
                let window = model.feasible_interval(ri);

                'req_try: for &bi in model.allowed_berth_indices(ri) {
                    let Some(pt) = model.processing_time(ri, bi) else {
                        continue;
                    };

                    let frees: Vec<_> = terminal
                        .iter_free_intervals_for_berths_in([bi], window)
                        .collect();

                    for free in frees {
                        let start = free.interval().start();
                        let end = start + pt;
                        let iv = TimeInterval::new(start, end);
                        if !free.interval().contains_interval(&iv) {
                            continue;
                        }

                        if terminal.occupy(bi, iv).is_ok() {
                            dvars[ri] = DecisionVar::Assigned(Decision {
                                berth_index: bi,
                                start_time: start,
                            });
                            placed_in_pass += 1;
                            break 'req_try;
                        }
                    }
                }
            }

            if placed_in_pass == 0 {
                break;
            }
        }

        let mut total_cost = Cost::zero();
        let mut unassigned = 0usize;
        for (i, dv) in dvars.iter().enumerate() {
            let ri = RequestIndex::new(i);
            match *dv {
                DecisionVar::Unassigned => unassigned += 1,
                DecisionVar::Assigned(dec) => {
                    if let Some(c) = model.cost_of_assignment(ri, dec.berth_index, dec.start_time) {
                        total_cost += c;
                    }
                }
            }
        }
        let fitness = Fitness::new(total_cost, unassigned);
        Ok(SolverState::new(dvars, terminal, fitness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::solver_model::SolverModel;
    use crate::state::{berth::berthocc::BerthRead, solver_state::SolverStateView};
    use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;
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
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn fixed_req(
        id: u32,
        window: (i64, i64),
        pt: &[(u32, i64)],
        weight: i64,
    ) -> Request<FixedKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FixedKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_one_berth_two_flex() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // r1 pt=10 on b1, r2 pt=5 on b1
        flex.insert(flex_req(1, (0, 200), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 200), &[(1, 5)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    #[test]
    fn test_build_assigns_all_when_space() {
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&model).expect("build should succeed");

        // Both requests should be assigned
        assert!(state.is_feasible());
        let assigned = state
            .decision_variables()
            .iter()
            .filter(|dv| dv.is_assigned())
            .count();
        assert_eq!(assigned, 2);
        // Cost > 0
        let mut cost = Cost::zero();
        for (i, dv) in state.decision_variables().iter().enumerate() {
            if let DecisionVar::Assigned(dec) = *dv {
                cost += model
                    .cost_of_assignment(RequestIndex::new(i), dec.berth_index, dec.start_time)
                    .expect("cost defined");
            }
        }
        assert!(cost > Cost::zero(), "cost should be positive");

        // Verify no overlap via terminal occupancy view.
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let b = state
            .terminal_occupancy()
            .berth(b_ix)
            .expect("berth 1 exists");
        // [0,15) cannot be entirely free now
        assert!(b.is_occupied(iv(0, 15)));
    }

    #[test]
    fn test_build_respects_fixed_assignments_blocking() {
        // One berth [0,100]; fixed assignment occupies [10,20).
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        let b1 = berth(1, 0, 100);
        berths.insert(b1.clone());

        let rf = fixed_req(100, (0, 100), &[(1, 10)], 1);
        // Fixed assignment: start at 10, pt=10 => [10,20)
        let af = Assignment::<FixedKind, i64>::new_fixed(rf.clone(), b1.clone(), tp(10)).unwrap();
        let mut fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();
        fixed.insert(af);

        // Two flexible, each pt=10
        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 100), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 100), &[(1, 10)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&model).expect("build should succeed");

        // Both flexible requests should be assigned around the fixed block
        let assigned = state
            .decision_variables()
            .iter()
            .filter(|dv| dv.is_assigned())
            .count();
        assert_eq!(assigned, 2, "both flexible assigned");

        // Terminal occupancy should reflect the fixed block too
        let b_ix = model.index_manager().berth_index(bid(1)).unwrap();
        let occ = state
            .terminal_occupancy()
            .berth(b_ix)
            .expect("berth 1 exists");
        assert!(occ.is_occupied(iv(10, 20)));
    }

    #[test]
    fn test_build_leaves_unassigned_when_insufficient_capacity() {
        // One berth [0,12]; two requests with pt=10 each. Only one fits.
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 12));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        flex.insert(flex_req(1, (0, 12), &[(1, 10)], 1));
        flex.insert(flex_req(2, (0, 12), &[(1, 10)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&model).expect("build should succeed");

        // Exactly one assigned, one unassigned
        let assigned = state
            .decision_variables()
            .iter()
            .filter(|dv| dv.is_assigned())
            .count();
        assert_eq!(assigned, 1);
        assert!(!state.is_feasible());

        // cost > 0 if something was assigned
        let mut cost = Cost::zero();
        for (i, dv) in state.decision_variables().iter().enumerate() {
            if let DecisionVar::Assigned(dec) = *dv {
                cost += model
                    .cost_of_assignment(RequestIndex::new(i), dec.berth_index, dec.start_time)
                    .unwrap();
            }
        }
        assert!(cost > Cost::zero());
    }

    #[test]
    fn test_weight_tiebreaker_when_slack_equal() {
        // Equal slack windows: both [0,20), both pt=5.
        // Higher weight should be scheduled first (earlier start).
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 20));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // r10 heavier weight=5, r20 weight=1
        flex.insert(flex_req(10, (0, 20), &[(1, 5)], 5));
        flex.insert(flex_req(20, (0, 20), &[(1, 5)], 1));

        let prob = Problem::new(berths, fixed, flex).unwrap();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&model).expect("build should succeed");

        // Get start times via decision variables by index
        let im = model.index_manager();
        let r10_ix = im.request_index(rid(10)).unwrap();
        let r20_ix = im.request_index(rid(20)).unwrap();

        let start = |ri: RequestIndex| match state.decision_variables()[ri.get()] {
            DecisionVar::Assigned(d) => d.start_time,
            _ => panic!("request {ri:?} not assigned"),
        };

        let s10 = start(r10_ix).value();
        let s20 = start(r20_ix).value();
        assert!(s10 <= s20, "heavier should not start later");
        assert_eq!(s10, 0, "heavier should start at earliest feasible");
        assert_eq!(s20, 5);
    }

    #[test]
    fn test_resulting_state_integrity() {
        // Sanity: building returns a consistent state
        let prob = problem_one_berth_two_flex();
        let model = SolverModel::try_from(&prob).expect("model ok");

        let solver = GreedyOpening::<i64>::new();
        let state = solver.build(&model).expect("build should succeed");

        // Terminal occupancy has the right number of berths (1)
        assert_eq!(state.terminal_occupancy().berths().len(), 1);

        // Feasibility matches unassigned count == 0,
        // and cost > 0 if feasible.
        assert_eq!(
            state.is_feasible(),
            state.fitness().unassigned_requests == 0
        );

        let mut cost = Cost::zero();
        for (i, dv) in state.decision_variables().iter().enumerate() {
            if let DecisionVar::Assigned(dec) = *dv {
                cost += model
                    .cost_of_assignment(RequestIndex::new(i), dec.berth_index, dec.start_time)
                    .unwrap();
            }
        }
        if state.is_feasible() {
            assert!(cost > Cost::zero());
        }
    }
}
