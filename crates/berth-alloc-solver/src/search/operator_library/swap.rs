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
    model::index::RequestIndex,
    search::{
        eval::CostEvaluator,
        neighboors::NeighborFn,
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::{
        decisionvar::{Decision, DecisionVar},
        solver_state::SolverStateView,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

enum CurrentList<'n> {
    None,
    Slice {
        slice: &'n [RequestIndex],
        pos: usize,
    },
    Vec {
        vec: Vec<RequestIndex>,
        pos: usize,
    },
    Full {
        pos: usize,
        n: usize,
    },
}

impl std::fmt::Display for CurrentList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurrentList::None => write!(f, "CurrentList::None"),
            CurrentList::Slice { slice: _slice, pos } => {
                write!(f, "CurrentList::Slice(pos: {})", pos)
            }
            CurrentList::Vec { vec: _vec, pos } => {
                write!(f, "CurrentList::Vec(pos: {})", pos)
            }
            CurrentList::Full { pos, n } => {
                write!(f, "CurrentList::Full(pos: {}, n: {})", pos, n)
            }
        }
    }
}

impl std::fmt::Debug for CurrentList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurrentList::None => write!(f, "CurrentList::None"),
            CurrentList::Slice { slice: _slice, pos } => {
                write!(f, "CurrentList::Slice(pos: {})", pos)
            }
            CurrentList::Vec { vec: _vec, pos } => {
                write!(f, "CurrentList::Vec(pos: {})", pos)
            }
            CurrentList::Full { pos, n } => {
                write!(f, "CurrentList::Full(pos: {}, n: {})", pos, n)
            }
        }
    }
}

impl<'n> CurrentList<'n> {
    #[inline]
    fn reset(&mut self) {
        *self = CurrentList::None;
    }

    #[inline]
    fn from_optional_neighbors(i: RequestIndex, nf: &Option<NeighborFn<'n>>, n: usize) -> Self {
        match nf {
            Some(NeighborFn::Slice(f)) => CurrentList::Slice {
                slice: f(i),
                pos: 0,
            },
            Some(NeighborFn::Vec(f)) => CurrentList::Vec { vec: f(i), pos: 0 },
            None => CurrentList::Full {
                pos: i.get().saturating_add(1),
                n,
            },
        }
    }
}

impl<'n> Iterator for CurrentList<'n> {
    type Item = RequestIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CurrentList::None => None,
            CurrentList::Slice { slice, pos } => {
                if *pos >= slice.len() {
                    return None;
                }
                let v = slice[*pos];
                *pos += 1;
                Some(v)
            }
            CurrentList::Vec { vec, pos } => {
                if *pos >= vec.len() {
                    return None;
                }
                let v = vec[*pos];
                *pos += 1;
                Some(v)
            }
            CurrentList::Full { pos, n } => {
                if *pos >= *n {
                    return None;
                }
                let v = RequestIndex::new(*pos);
                *pos += 1;
                Some(v)
            }
        }
    }
}

pub struct SwapSlotOp<'n> {
    i: usize,
    current: CurrentList<'n>,
    neighbor_function: Option<NeighborFn<'n>>,
}

impl<'n> std::fmt::Debug for SwapSlotOp<'n> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwapSlotOp")
            .field("i", &self.i)
            .field("current", &self.current)
            .field(
                "neighbor_function",
                &self.neighbor_function.as_ref().map(|_| "Some(...)"),
            )
            .finish()
    }
}

impl<'n> std::fmt::Display for SwapSlotOp<'n> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SwapSlotOp(i: {}, current: {})", self.i, self.current)
    }
}

impl<'n> Default for SwapSlotOp<'n> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'n> SwapSlotOp<'n> {
    #[inline]
    pub fn new() -> Self {
        Self {
            i: 0,
            current: CurrentList::None,
            neighbor_function: None,
        }
    }

    #[inline]
    pub fn with_neighbors(neigh: NeighborFn<'n>) -> Self {
        Self {
            i: 0,
            current: CurrentList::None,
            neighbor_function: Some(neigh),
        }
    }
}

impl<'n, T, C, R> LocalSearchOperator<T, C, R> for SwapSlotOp<'n>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "SwapSlotOp"
    }

    fn reset(&mut self) {
        self.i = 0;
        self.current.reset();
    }

    fn has_fragments(&self) -> bool {
        false
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<crate::state::plan::Plan<'p, T>> {
        let model = ctx.model();
        let dvars = ctx.state().decision_variables();
        let n = dvars.len();

        while self.i < n {
            // Initialize candidate list for r1 if needed
            if matches!(self.current, CurrentList::None) {
                let r1 = RequestIndex::new(self.i);
                self.current = CurrentList::from_optional_neighbors(r1, &self.neighbor_function, n);
            }

            let r1 = RequestIndex::new(self.i);

            // r1 must be assigned to consider a swap
            let (b1, s1) = match dvars.get(r1.get()).copied() {
                Some(DecisionVar::Assigned(Decision {
                    berth_index,
                    start_time,
                })) => (berth_index, start_time),
                _ => {
                    self.i += 1;
                    self.current.reset();
                    continue;
                }
            };

            for r2 in self.current.by_ref() {
                if r2.get() >= n || r2 == r1 {
                    continue;
                }

                // r2 must be assigned
                let (b2, s2) = match dvars.get(r2.get()).copied() {
                    Some(DecisionVar::Assigned(Decision {
                        berth_index,
                        start_time,
                    })) => (berth_index, start_time),
                    _ => continue,
                };

                // identical slots -> skip
                if b1 == b2 && s1 == s2 {
                    continue;
                }

                // allowed-berth pruning
                let a1 = model.allowed_berth_indices(r1);
                let a2 = model.allowed_berth_indices(r2);
                if !a1.contains(&b2) || !a2.contains(&b1) {
                    continue;
                }

                // processing times on destination berths
                let pt1_on_b2 = match model.processing_time(r1, b2) {
                    Some(pt) => pt,
                    None => continue,
                };
                let pt2_on_b1 = match model.processing_time(r2, b1) {
                    Some(pt) => pt,
                    None => continue,
                };

                // window checks for the *exact* swap starts (s2 for r1, s1 for r2)
                let w1 = model.feasible_interval(r1);
                let w2 = model.feasible_interval(r2);
                let end1_needed = s2 + pt1_on_b2;
                let end2_needed = s1 + pt2_on_b1;
                if s2 < w1.start() || end1_needed > w1.end() {
                    continue;
                }
                if s1 < w2.start() || end2_needed > w2.end() {
                    continue;
                }

                // Additional pruning: the swapped job must fit within the other job's current slot length
                // Compute current slot ends based on existing assignments (not after unassignment).
                let pt1_on_b1 = match model.processing_time(r1, b1) {
                    Some(pt) => pt,
                    None => continue,
                };
                let pt2_on_b2 = match model.processing_time(r2, b2) {
                    Some(pt) => pt,
                    None => continue,
                };
                let iv1_curr_end = s1 + pt1_on_b1; // end of r1's current slot on b1
                let iv2_curr_end = s2 + pt2_on_b2; // end of r2's current slot on b2

                if end1_needed > iv2_curr_end {
                    // r1 wouldn't fit in r2's current slot length at s2
                    continue;
                }
                if end2_needed > iv1_curr_end {
                    // r2 wouldn't fit in r1's current slot length at s1
                    continue;
                }

                // Build plan atomically with rollback on failure.
                let mut pb = ctx.builder();
                let sp = pb.savepoint();

                // Make the two holes first; the explorer will then "see" them.
                if pb.propose_unassignment(r1).is_err() {
                    pb.undo_to(sp);
                    continue;
                }
                if pb.propose_unassignment(r2).is_err() {
                    pb.undo_to(sp);
                    continue;
                }

                // Find free-berth intervals that exactly cover the swap placements using the new API
                let g1_opt = pb
                    .iter_free_for_on_berth_in(
                        r1,
                        b2,
                        berth_alloc_core::prelude::TimeInterval::new(s2, end1_needed),
                    )
                    .next();

                let g2_opt = pb
                    .iter_free_for_on_berth_in(
                        r2,
                        b1,
                        berth_alloc_core::prelude::TimeInterval::new(s1, end2_needed),
                    )
                    .next();

                let (g1, g2) = match (g1_opt, g2_opt) {
                    (Some(a), Some(b)) => (a, b),
                    _ => {
                        pb.undo_to(sp);
                        continue;
                    }
                };

                // Commit the two assignments at the exact swapped starts.
                if pb.propose_assignment(r1, s2, &g1).is_err() {
                    pb.undo_to(sp);
                    continue;
                }
                if pb.propose_assignment(r2, s1, &g2).is_err() {
                    pb.undo_to(sp);
                    continue;
                }

                // Yield a single neighbor
                return Some(pb.finalize());
            }

            // exhausted r2 candidates for this r1
            self.i += 1;
            self.current.reset();
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::solver_model::SolverModel,
        search::{
            eval::{CostEvaluator, DefaultCostEvaluator},
            operator::{LocalSearchOperator, OperatorContext},
        },
        state::{
            berth::berthocc::BerthRead,
            decisionvar::{DecisionVar, DecisionVarVec},
            solver_state::SolverState,
            terminal::terminalocc::{TerminalOccupancy, TerminalRead, TerminalWrite},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use rand::{SeedableRng, rngs::StdRng};
    use rand_chacha::ChaCha8Rng;
    use std::{collections::BTreeMap, sync::Arc};

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

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pts: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn problem_two_berths_two_flex(
        pt_r1_b1: i64,
        pt_r1_b2: i64,
        pt_r2_b1: i64,
        pt_r2_b2: i64,
    ) -> Problem<i64> {
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);

        let r1 = flex_req(10, (0, 100), &[(1, pt_r1_b1), (2, pt_r1_b2)], 1);
        let r2 = flex_req(20, (0, 100), &[(1, pt_r2_b1), (2, pt_r2_b2)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        builder.build().expect("valid problem")
    }

    // Build an initial state with two assigned requests (0->b1@t=10, 1->b2@t=20).
    fn build_state_with_assignments<'p>(
        model: &'p SolverModel<'p, T>,
        evaluator: &impl CostEvaluator<T>,
        s1: i64,
        s2: i64,
    ) -> SolverState<'p, T> {
        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();
        let r2 = im.request_index(rid(20)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1.get()] = DecisionVar::assigned(b1, tp(s1));
        dvars[r2.get()] = DecisionVar::assigned(b2, tp(s2));

        // Terminal occupancy reflecting the current assignments
        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv1 = model.interval(r1, b1, tp(s1)).unwrap();
        let iv2 = model.interval(r2, b2, tp(s2)).unwrap();
        terminal.occupy(b1, iv1).unwrap();
        terminal.occupy(b2, iv2).unwrap();

        let fitness = evaluator.eval_fitness(model, &dvars);
        SolverState::new(DecisionVarVec::from(dvars), terminal, fitness)
    }

    fn ctx<'b, 'r, 'c, 's, 'm, 'p, C, R>(
        model: &'m SolverModel<'p, T>,
        state: &'s SolverState<'p, T>,
        evaluator: &'c C,
        rng: &'r mut R,
        buffer: &'b mut [DecisionVar<T>],
    ) -> OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>
    where
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        OperatorContext::new(model, state, evaluator, rng, buffer)
    }

    #[test]
    fn test_swap_yields_plan_and_applies_swap() {
        // Processing times satisfy swap constraints:
        // pt(r1,b2) <= pt(r2,b2) and pt(r2,b1) <= pt(r1,b1)
        let problem = problem_two_berths_two_flex(7, 5, 6, 9);
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let mut state = build_state_with_assignments(&model, &evaluator, 10, 20);

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();

        let plan = op
            .make_next_neighbor(&mut context)
            .expect("expected a swap plan");

        // Delta unassigned should be 0 (unassign both, assign both)
        assert_eq!(plan.fitness_delta.delta_unassigned, 0);
        // Expect 4 patches: unassign r1, unassign r2, assign r1, assign r2
        assert_eq!(plan.decision_var_patches.len(), 4);

        // Apply and verify assignments are swapped
        let prev_f = *state.fitness();
        state.apply_plan(plan);
        let cur_f = *state.fitness();

        // Cost may go up or down depending on evaluator; ensure feasible (no unassigned)
        assert_eq!(cur_f.unassigned_requests, prev_f.unassigned_requests);
        assert!(state.is_feasible());

        // Verify the new DV placement matches swapped berths/times
        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();
        let r2 = im.request_index(rid(20)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        let dv1 = state.decision_variables()[r1.get()];
        let dv2 = state.decision_variables()[r2.get()];
        let assigned1 = dv1.as_assigned().expect("r1 must be assigned");
        let assigned2 = dv2.as_assigned().expect("r2 must be assigned");

        // r1 should now be on b2 at previous r2 start (20)
        // r2 should now be on b1 at previous r1 start (10)
        assert_eq!(assigned1.berth_index, b2);
        assert_eq!(assigned1.start_time, tp(20));
        assert_eq!(assigned2.berth_index, b1);
        assert_eq!(assigned2.start_time, tp(10));

        // Occupancy reflects new intervals
        let iv1_new = model.interval(r1, b2, tp(20)).unwrap();
        let iv2_new = model.interval(r2, b1, tp(10)).unwrap();
        assert!(
            state
                .terminal_occupancy()
                .berth(b2)
                .unwrap()
                .is_occupied(iv1_new)
        );
        assert!(
            state
                .terminal_occupancy()
                .berth(b1)
                .unwrap()
                .is_occupied(iv2_new)
        );
    }

    #[test]
    fn test_with_neighbors_vec_limits_pairs() {
        let problem = problem_two_berths_two_flex(7, 5, 6, 9);
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;
        let state = build_state_with_assignments(&model, &evaluator, 10, 20);

        // Only consider neighbor r2 for r1; for r2, empty vec (no further neighbors)
        let neigh = NeighborFn::Vec(Arc::new(|i| {
            if i.get() == 0 {
                vec![RequestIndex::new(1)]
            } else {
                vec![]
            }
        }));
        let mut op = SwapSlotOp::with_neighbors(neigh);

        let mut rng = StdRng::seed_from_u64(2);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let plan = op.make_next_neighbor(&mut ctx);
        assert!(
            plan.is_some(),
            "restricted neighbor should still yield swap"
        );
        // Next call should exhaust
        assert!(op.make_next_neighbor(&mut ctx).is_none());
    }

    #[test]
    fn test_skips_when_unassigned_present() {
        // Make r1 assigned, r2 unassigned; swap requires both assigned -> should yield None
        let problem = problem_two_berths_two_flex(7, 5, 6, 9);
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Build state with only r1 assigned
        let im = model.index_manager();
        let r1 = im.request_index(rid(10)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r1.get()] = DecisionVar::assigned(b1, tp(10));

        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv1 = model.interval(r1, b1, tp(10)).unwrap();
        terminal.occupy(b1, iv1).unwrap();

        let fitness = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), terminal, fitness);

        let mut rng = StdRng::seed_from_u64(3);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();
        assert!(op.make_next_neighbor(&mut ctx).is_none());
    }

    #[test]
    fn test_prunes_when_not_allowed_on_destination_berth() {
        // r1 allowed only on b1, r2 allowed on b1 and b2, so swapping to r1@b2 is not allowed -> None
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        let r1 = flex_req(10, (0, 100), &[(1, 7)], 1); // only b1
        let r2 = flex_req(20, (0, 100), &[(1, 6), (2, 9)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        let problem = builder.build().expect("valid problem");

        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let state = build_state_with_assignments(&model, &evaluator, 10, 20);

        let mut rng = StdRng::seed_from_u64(4);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();
        assert!(
            op.make_next_neighbor(&mut ctx).is_none(),
            "swap should be pruned by allowed-berth check"
        );
    }

    #[test]
    fn test_window_and_length_checks_prevent_swap() {
        // Choose PT so that end1_on_b2 > iv2_curr.end (pt1_on_b2 > pt2_on_b2)
        let problem = problem_two_berths_two_flex(7, 12, 6, 9); // pt(r1,b2)=12 > pt(r2,b2)=9
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let state = build_state_with_assignments(&model, &evaluator, 10, 20);

        let mut rng = StdRng::seed_from_u64(5);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();
        assert!(
            op.make_next_neighbor(&mut ctx).is_none(),
            "swap should be pruned by end-time length check"
        );
    }

    #[test]
    fn test_neighbor_slice_works_and_ignores_self_and_oob() {
        // Valid configuration where a swap is feasible between r1 and r2
        let problem = problem_two_berths_two_flex(7, 5, 6, 9);
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;
        let state = build_state_with_assignments(&model, &evaluator, 10, 20);

        // Slice for i=0 returns [self(0), valid(1), out-of-bounds(99)]
        // Slice for i=1 returns empty (no neighbors)
        let slice0: &'static [RequestIndex] = Box::leak(
            vec![
                RequestIndex::new(0),
                RequestIndex::new(1),
                RequestIndex::new(99),
            ]
            .into_boxed_slice(),
        );
        let slice1: &'static [RequestIndex] =
            Box::leak(Vec::<RequestIndex>::new().into_boxed_slice());

        let neigh = NeighborFn::Slice(Arc::new(
            move |i| {
                if i.get() == 0 { slice0 } else { slice1 }
            },
        ));

        let mut op = SwapSlotOp::with_neighbors(neigh);

        let mut rng = StdRng::seed_from_u64(12);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        // Should ignore self and oob entries, but still yield a valid plan due to the "1" entry.
        let plan = op.make_next_neighbor(&mut ctx);
        assert!(
            plan.is_some(),
            "expected a swap plan from slice-based neighbors"
        );
        // Exhaust on next call
        assert!(op.make_next_neighbor(&mut ctx).is_none());
    }

    #[test]
    fn test_window_start_pruning_prevents_swap() {
        // r1 window starts at 25; swapping would place r1 at s2=20, which violates w1.start().
        // Make sure the initial assignments are feasible: assign r1 at 30 (>= 25), r2 at 20.
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        // r1 window [25, 100), r2 window [0, 100)
        let r1 = flex_req(10, (25, 100), &[(1, 7), (2, 5)], 1);
        let r2 = flex_req(20, (0, 100), &[(1, 6), (2, 9)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        let problem = builder.build().expect("valid problem");

        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Initial state: r1@b1@30 (valid), r2@b2@20 (valid)
        let state = build_state_with_assignments(&model, &evaluator, 30, 20);

        let mut rng = StdRng::seed_from_u64(13);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();
        assert!(
            op.make_next_neighbor(&mut ctx).is_none(),
            "swap should be pruned because s2 < w1.start()"
        );
    }

    #[test]
    fn test_length_check_other_direction_prevents_swap() {
        // Force end2_on_b1 > iv1_curr.end by making pt(r2,b1) > pt(r1,b1)
        // Use lengths that otherwise allow swaps.
        let problem = problem_two_berths_two_flex(
            7,  // pt(r1,b1)
            5,  // pt(r1,b2)
            10, // pt(r2,b1) larger than pt(r1,b1)
            6,  // pt(r2,b2)
        );
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let state = build_state_with_assignments(&model, &evaluator, 10, 20);

        let mut rng = StdRng::seed_from_u64(14);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();
        assert!(
            op.make_next_neighbor(&mut ctx).is_none(),
            "swap should be pruned because end2_on_b1 > iv1_curr.end"
        );
    }

    #[test]
    fn test_operator_metadata() {
        // Validate name and has_fragments behavior via trait object to nail generics
        let op = SwapSlotOp::new();
        use crate::search::operator::LocalSearchOperator;
        let lso: &dyn LocalSearchOperator<i64, DefaultCostEvaluator, StdRng> = &op;

        assert_eq!(lso.name(), "SwapSlotOp");
        assert!(
            !lso.has_fragments(),
            "swap operator yields independent neighbors"
        );
    }

    #[test]
    fn test_exhaustion_after_single_yield() {
        // In our simple two-request setup, operator typically yields at most one swap
        let problem = problem_two_berths_two_flex(7, 5, 6, 9);
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;
        let state = build_state_with_assignments(&model, &evaluator, 10, 20);

        let mut rng = StdRng::seed_from_u64(15);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SwapSlotOp::new();
        let first = op.make_next_neighbor(&mut ctx);
        assert!(first.is_some(), "expected a first swap plan");

        let second = op.make_next_neighbor(&mut ctx);
        assert!(
            second.is_none(),
            "expected operator to be exhausted after one swap in this setup"
        );
    }
}
