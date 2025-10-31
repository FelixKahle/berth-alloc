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
        operator::{LocalSearchOperator, OperatorContext},
    },
    state::{
        decisionvar::DecisionVar, plan::Plan, solver_state::SolverStateView,
        terminal::terminalocc::FreeBerth,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

/// Implements an "intra-berth slide" neighborhood.
///
/// This operator iterates through each assigned request `r`.
/// It attempts to move `r` to an *earlier* time slot on the *same berth*.
///
/// It specifically looks for the **EARLIEST** available time slot
/// that is still **strictly earlier** than the request's current start time.
#[derive(Debug)]
pub struct SlideOp {
    req_idx: usize,
}

impl Default for SlideOp {
    fn default() -> Self {
        Self::new()
    }
}

impl SlideOp {
    #[inline]
    pub fn new() -> Self {
        Self { req_idx: 0 }
    }
}

impl<T, C, R> LocalSearchOperator<T, C, R> for SlideOp
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "SlideOp"
    }

    fn reset(&mut self) {
        self.req_idx = 0;
    }

    fn has_fragments(&self) -> bool {
        false
    }

    fn make_next_neighbor<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut OperatorContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
    ) -> Option<Plan<'p, T>> {
        let dvars = ctx.state().decision_variables();
        let n = dvars.len();

        while self.req_idx < n {
            let r = RequestIndex::new(self.req_idx);
            self.req_idx += 1;

            let (b, s) = match dvars.get(r.get()).copied() {
                Some(DecisionVar::Assigned(d)) => (d.berth_index, d.start_time),
                _ => continue,
            };

            let mut pb = ctx.builder();
            let sp = pb.savepoint();

            if pb.propose_unassignment(r).is_err() {
                pb.undo_to(sp);
                continue;
            }

            let best = pb.with_explorer(|ex| {
                let w = ex.model().feasible_interval(r);
                let pt = match ex.model().processing_time(r, b) {
                    Some(pt) => pt,
                    None => return None,
                };

                ex.iter_free_for(r)
                    .filter(|fb| fb.berth_index() == b)
                    .filter_map(|fb| {
                        let iv = fb.interval();
                        let candidate_start = iv.start().max(w.start());
                        let candidate_end = candidate_start.checked_add(pt)?;
                        if candidate_start < s && candidate_end <= iv.end() {
                            Some((candidate_start, iv))
                        } else {
                            None
                        }
                    })
                    .min_by_key(|(start, _iv)| *start)
            });

            if let Some((new_start, new_iv)) = best {
                if new_start == s {
                    pb.undo_to(sp);
                    continue;
                }

                let free = FreeBerth::new(new_iv, b);
                if pb.propose_assignment(r, new_start, &free).is_ok() {
                    return Some(pb.finalize());
                }
            }

            pb.undo_to(sp);
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
            decisionvar::{DecisionVar, DecisionVarVec},
            solver_state::SolverState,
            terminal::terminalocc::{TerminalOccupancy, TerminalWrite},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::BTreeMap;

    type T = i64;

    // ---- Helpers ----

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

    // Problem with one berth and two flex requests:
    // - r10 has pt=10 on b1
    // - r20 has pt=5 on b1
    // The test states set assignments to create a gap.
    fn problem_with_gap_r10_window(r10_window: (i64, i64)) -> Problem<i64> {
        let b1 = berth(1, 0, 100);
        let r10 = flex_req(10, r10_window, &[(1, 10)], 1);
        let r20 = flex_req(20, (0, 100), &[(1, 5)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.add_flexible(r20);
        builder.build().expect("valid problem")
    }

    // Build a state with r20 at [0, 5], r10 at [20, 30] (gap [5, 20]).
    fn build_state_with_gap<'p>(
        model: &'p SolverModel<'p, T>,
        evaluator: &impl CostEvaluator<T>,
        r10_start: i64,
        r20_start: i64,
    ) -> SolverState<'p, T> {
        let im = model.index_manager();
        let r10 = im.request_index(rid(10)).unwrap(); // pt=10
        let r20 = im.request_index(rid(20)).unwrap(); // pt=5
        let b1 = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r10.get()] = DecisionVar::assigned(b1, tp(r10_start));
        dvars[r20.get()] = DecisionVar::assigned(b1, tp(r20_start));

        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv10 = model.interval(r10, b1, tp(r10_start)).unwrap();
        let iv20 = model.interval(r20, b1, tp(r20_start)).unwrap();
        terminal.occupy(b1, iv10).unwrap();
        terminal.occupy(b1, iv20).unwrap();

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

    // ---- Tests ----

    #[test]
    fn test_slide_finds_earliest_slot() {
        let problem = problem_with_gap_r10_window((0, 100));
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // r20 @ [0, 5], r10 @ [20, 30], gap [5, 20].
        let mut state = build_state_with_gap(&model, &evaluator, 20, 0);

        let mut rng = StdRng::seed_from_u64(1);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();

        let plan = op
            .make_next_neighbor(&mut context)
            .expect("expected a slide plan");

        // Apply and verify r10 slides to earliest feasible start 5
        state.apply_plan(plan);
        let r10 = model.index_manager().request_index(rid(10)).unwrap();
        let r10_dec = state.decision_variables()[r10.get()].as_assigned().unwrap();
        assert_eq!(r10_dec.start_time, tp(5));
    }

    #[test]
    fn test_slide_finds_nothing_if_packed() {
        let problem = problem_with_gap_r10_window((0, 100));
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Tightly pack: r20 @ [0, 5], r10 @ [5, 15] -> no earlier room for r10, and r20 is at 0.
        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let im = model.index_manager();
        let r10 = im.request_index(rid(10)).unwrap();
        let r20 = im.request_index(rid(20)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();
        dvars[r20.get()] = DecisionVar::assigned(b1, tp(0));
        dvars[r10.get()] = DecisionVar::assigned(b1, tp(5));

        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv20 = model.interval(r20, b1, tp(0)).unwrap(); // [0,5]
        let iv10 = model.interval(r10, b1, tp(5)).unwrap(); // [5,15]
        terminal.occupy(b1, iv20).unwrap();
        terminal.occupy(b1, iv10).unwrap();

        let fitness = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), terminal, fitness);

        let mut rng = StdRng::seed_from_u64(2);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();

        // Checks r10 first -> no slot
        assert!(
            op.make_next_neighbor(&mut context).is_none(),
            "r10 should not find a slot"
        );
        // Checks r20 next -> at t=0, no earlier slot
        assert!(
            op.make_next_neighbor(&mut context).is_none(),
            "r20 should not find a slot"
        );
    }

    #[test]
    fn test_slide_respects_window_and_uses_merged_free_interval() {
        // r10 window [15, 100], gap [5,20], pt=10, current r10@20..30
        // After unassigning r10, the sandbox merges free intervals, yielding [5,30].
        // Earliest feasible start is 15 (15..25 fits within [5,30]).
        let problem = problem_with_gap_r10_window((15, 100));
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let mut state = build_state_with_gap(&model, &evaluator, 20, 0);

        let mut rng = StdRng::seed_from_u64(3);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();
        let plan = op
            .make_next_neighbor(&mut context)
            .expect("should find a move considering merged free interval");

        state.apply_plan(plan);
        let r10_idx = model.index_manager().request_index(rid(10)).unwrap();
        let r10_dec = state.decision_variables()[r10_idx.get()]
            .as_assigned()
            .unwrap();
        assert_eq!(r10_dec.start_time, tp(15));
    }

    #[test]
    fn test_slide_finds_earliest_respecting_window() {
        // r10 window [10, 100], gap [5,20], pt=10 -> earliest start is 10 (fits exactly to 20)
        let problem = problem_with_gap_r10_window((10, 100));
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let mut state = build_state_with_gap(&model, &evaluator, 20, 0);

        let mut rng = StdRng::seed_from_u64(4);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();
        let plan = op
            .make_next_neighbor(&mut context)
            .expect("should find a plan");

        state.apply_plan(plan);
        let r10_idx = model.index_manager().request_index(rid(10)).unwrap();
        let r10_dec = state.decision_variables()[r10_idx.get()]
            .as_assigned()
            .unwrap();
        assert_eq!(r10_dec.start_time, tp(10));
    }

    #[test]
    fn test_operator_metadata() {
        // Validate via trait object
        let op = SlideOp::new();
        let lso: &dyn LocalSearchOperator<i64, DefaultCostEvaluator, StdRng> = &op;

        assert_eq!(lso.name(), "SlideOp");
        assert!(
            !lso.has_fragments(),
            "slide operator yields independent neighbors"
        );
    }

    #[test]
    fn test_exhaustion_and_fresh_instance_yields_again() {
        let problem = problem_with_gap_r10_window((0, 100));
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;
        let state = build_state_with_gap(&model, &evaluator, 20, 0);

        let mut rng = StdRng::seed_from_u64(5);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();
        let first = op.make_next_neighbor(&mut context);
        assert!(first.is_some(), "expected a first slide plan");

        let second = op.make_next_neighbor(&mut context);
        assert!(
            second.is_none(),
            "operator should be exhausted after scanning both requests"
        );

        // Fresh instance behaves like a reset; should yield again
        let mut op2 = SlideOp::new();
        let mut rng2 = StdRng::seed_from_u64(6);
        let mut buffer2 = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context2 = ctx(&model, &state, &evaluator, &mut rng2, &mut buffer2);

        let again = op2.make_next_neighbor(&mut context2);
        assert!(
            again.is_some(),
            "a fresh operator should yield again on same state"
        );
    }

    #[test]
    fn test_skips_unassigned_requests() {
        // Make r10 unassigned and r20 assigned at t=0; there is no earlier slot for r20.
        let problem = problem_with_gap_r10_window((0, 100));
        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        let im = model.index_manager();
        let r20 = im.request_index(rid(20)).unwrap();
        let b1 = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r20.get()] = DecisionVar::assigned(b1, tp(0));

        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv20 = model.interval(r20, b1, tp(0)).unwrap(); // [0,5]
        terminal.occupy(b1, iv20).unwrap();

        let fitness = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), terminal, fitness);

        let mut rng = StdRng::seed_from_u64(7);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();
        // r10 is unassigned -> skipped; r20 at t=0 has no earlier slot -> None
        assert!(op.make_next_neighbor(&mut context).is_none());
        assert!(op.make_next_neighbor(&mut context).is_none());
    }

    #[test]
    fn test_no_cross_berth_slide() {
        // Two berths: earlier free time exists on b2 but not on b1; operator must NOT move across berths.
        // r10 allowed on both berths, assigned on b1 at 20, b1 has no earlier free slot (blocked by r30).
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);
        let r10 = flex_req(10, (0, 100), &[(1, 10), (2, 10)], 1); // pt=10 both
        let r20 = flex_req(20, (0, 100), &[(1, 5)], 1); // on b1 at 0..5
        let r30 = flex_req(30, (0, 100), &[(1, 15)], 1); // on b1 at 5..20 blocks earlier slot

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r10);
        builder.add_flexible(r20);
        builder.add_flexible(r30);
        let problem = builder.build().expect("valid problem");

        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Assign: r20@b1@0..5, r30@b1@5..20, r10@b1@20..30
        let im = model.index_manager();
        let r10i = im.request_index(rid(10)).unwrap();
        let r20i = im.request_index(rid(20)).unwrap();
        let r30i = im.request_index(rid(30)).unwrap();
        let b1i = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r20i.get()] = DecisionVar::assigned(b1i, tp(0));
        dvars[r30i.get()] = DecisionVar::assigned(b1i, tp(5));
        dvars[r10i.get()] = DecisionVar::assigned(b1i, tp(20));

        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv20 = model.interval(r20i, b1i, tp(0)).unwrap(); // [0,5]
        let iv30 = model.interval(r30i, b1i, tp(5)).unwrap(); // [5,20]
        let iv10 = model.interval(r10i, b1i, tp(20)).unwrap(); // [20,30]
        terminal.occupy(b1i, iv20).unwrap();
        terminal.occupy(b1i, iv30).unwrap();
        terminal.occupy(b1i, iv10).unwrap();

        let fitness = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), terminal, fitness);

        // There is free space on b2 at earlier times, but slide must consider same berth only.
        let mut rng = StdRng::seed_from_u64(8);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();
        assert!(
            op.make_next_neighbor(&mut context).is_none(),
            "should not slide across berths; no earlier space on b1"
        );
    }

    #[test]
    fn test_slide_can_shift_to_immediate_gap_before_current_start() {
        // Construct a case where unassigning r10 creates a free interval right before its current
        // start time, allowing a small earlier shift (e.g. from 35 to 34).
        // Setup:
        // - b1 [0,100]
        // - r20 pt=5 @ [0,5]
        // - r30 pt=10 @ [24,34]
        // - r10 pt=10 @ [35,45], window [15,100]
        // After unassigning r10, free intervals earlier than 35 include [34, ...],
        // which allows a valid earlier start at 34 (34..44).
        let b1 = berth(1, 0, 100);
        let r10 = flex_req(10, (15, 100), &[(1, 10)], 1);
        let r20 = flex_req(20, (0, 100), &[(1, 5)], 1);
        let r30 = flex_req(30, (0, 100), &[(1, 10)], 1);

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r10);
        builder.add_flexible(r20);
        builder.add_flexible(r30);
        let problem = builder.build().expect("valid problem");

        let model = SolverModel::try_from(&problem).expect("model ok");
        let evaluator = DefaultCostEvaluator;

        // Assign: r20@0..5, r30@24..34, r10@35..45
        let im = model.index_manager();
        let r10i = im.request_index(rid(10)).unwrap();
        let r20i = im.request_index(rid(20)).unwrap();
        let r30i = im.request_index(rid(30)).unwrap();
        let b1i = im.berth_index(bid(1)).unwrap();

        let mut dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        dvars[r20i.get()] = DecisionVar::assigned(b1i, tp(0));
        dvars[r30i.get()] = DecisionVar::assigned(b1i, tp(24));
        dvars[r10i.get()] = DecisionVar::assigned(b1i, tp(35));

        let mut terminal = TerminalOccupancy::new(model.berths());
        let iv20 = model.interval(r20i, b1i, tp(0)).unwrap(); // [0,5]
        let iv30 = model.interval(r30i, b1i, tp(24)).unwrap(); // [24,34]
        let iv10 = model.interval(r10i, b1i, tp(35)).unwrap(); // [35,45]
        terminal.occupy(b1i, iv20).unwrap();
        terminal.occupy(b1i, iv30).unwrap();
        terminal.occupy(b1i, iv10).unwrap();

        let fitness = evaluator.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), terminal, fitness);

        let mut rng = StdRng::seed_from_u64(9);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut context = ctx(&model, &state, &evaluator, &mut rng, &mut buffer);

        let mut op = SlideOp::new();
        let plan = op
            .make_next_neighbor(&mut context)
            .expect("should find a move to the immediate gap before current start");

        // Apply and validate the new start is 34 (earlier than 35 and within window).
        let mut new_state = state;
        new_state.apply_plan(plan);
        let r10_dec = new_state.decision_variables()[r10i.get()]
            .as_assigned()
            .unwrap();
        assert_eq!(r10_dec.start_time, tp(34));
    }
}
