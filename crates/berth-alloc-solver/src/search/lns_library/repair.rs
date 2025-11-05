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
        lns::{RepairProcedure, RepairProcedureContext, RuinOutcome},
    },
    state::plan::Plan,
    state::terminal::terminalocc::FreeBerth,
};
use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

// Local type aliases to reduce noise.
type TP<T> = TimePoint<T>;
type TD<T> = TimeDelta<T>;
type FB<T> = FreeBerth<T>;
type Cand<T> = (Cost, TP<T>, FB<T>);

/// Iterate earliest-start feasible insertion candidates for a request as:
/// (start_time, free_berth)
///
/// We consider earliest-start placement within each free interval, and only keep
/// intervals whose length is >= processing time for the request on that berth.
fn candidates_for_request<'a, 'b, 'c, 't, 'm, 'p, T, C>(
    pb: &'a crate::search::planner::PlanBuilder<'b, 'c, 't, 'm, 'p, T, C>,
    req: RequestIndex,
) -> impl Iterator<Item = (TP<T>, FB<T>)> + 'a
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub,
    C: CostEvaluator<T>,
{
    pb.iter_free_for(req).filter_map(move |fb| {
        let pt = pb.model().processing_time(req, fb.berth_index())?;
        let len: TD<T> = fb.interval().length();
        if len < pt {
            return None;
        }
        Some((fb.interval().start(), fb))
    })
}

/// A simple insertion to the cheapest available position across berths/windows.
pub struct CheapestInsertionRepair;

impl Default for CheapestInsertionRepair {
    fn default() -> Self {
        Self::new()
    }
}

impl CheapestInsertionRepair {
    pub fn new() -> Self {
        Self
    }
}

impl<T, C, R> RepairProcedure<T, C, R> for CheapestInsertionRepair
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "CheapestInsertionRepair"
    }

    fn repair<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ruined_outcome: RuinOutcome<'p, T>,
    ) -> Plan<'p, T> {
        let mut pb = ctx.builder(ruined_outcome.ruined_plan);

        for &req in &ruined_outcome.ruined {
            let mut best: Option<Cand<T>> = None;

            for (start, fb) in candidates_for_request(&pb, req) {
                if let Some(cost) = pb.peek_cost(req, start, &fb) {
                    match best {
                        None => best = Some((cost, start, fb)),
                        Some((bc, _, _)) if cost < bc => best = Some((cost, start, fb)),
                        _ => {}
                    }
                }
            }

            if let Some((_cost, start, fb)) = best {
                let _ = pb.propose_assignment(req, start, &fb);
            }
        }

        pb.finalize()
    }
}

/// Randomly inserts each ruined request into one feasible free interval (if exists).
///
/// We compute candidate (start, berth_index) pairs first (dropping the temp builder),
/// randomly pick one, then revalidate/apply it with a fresh builder using
/// `iter_free_for_on_berth_in` to avoid borrow issues.
pub struct RandomFeasibleInsertionRepair;

impl Default for RandomFeasibleInsertionRepair {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomFeasibleInsertionRepair {
    pub fn new() -> Self {
        Self
    }
}

impl<T, C, R> RepairProcedure<T, C, R> for RandomFeasibleInsertionRepair
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomFeasibleInsertionRepair"
    }

    fn repair<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ruined_outcome: RuinOutcome<'p, T>,
    ) -> Plan<'p, T> {
        let mut acc = ruined_outcome.ruined_plan;

        for &req in &ruined_outcome.ruined {
            // 1) Compute candidates as (start, berth_index) from a temporary builder
            let choices: Vec<_> = {
                let pb = ctx.builder(acc.clone());
                candidates_for_request(&pb, req)
                    .map(|(start, fb)| (start, fb.berth_index()))
                    .collect()
            };

            if choices.is_empty() {
                continue;
            }

            // 2) Randomly pick after builder is dropped
            let pick_idx = ctx.rng().random_range(0..choices.len());
            let (start, berth) = choices[pick_idx];

            // 3) Apply choice with a new builder; revalidate against current free intervals
            let step = {
                let mut pb = ctx.builder(acc.clone());

                if let Some(pt) = pb.model().processing_time(req, berth) {
                    let asg_iv = TimeInterval::new(start, start + pt);

                    // Revalidate: is there a free berth interval on this berth covering [start, start+pt)?
                    if let Some(fb2) = { pb.iter_free_for_on_berth_in(req, berth, asg_iv).next() } {
                        let _ = pb.propose_assignment(req, start, &fb2);
                        pb.finalize()
                    } else {
                        Plan::empty()
                    }
                } else {
                    // Not allowed or no processing time on this berth
                    Plan::empty()
                }
            };

            acc = acc.concat(step);
        }

        acc
    }
}

/// Regret-2 insertion: iteratively choose the ruined request with largest (c2 - c1),
/// where c1 and c2 are the first and second best incremental costs. If only one position
/// exists, treat c2 as +infinity to bias picking constrained requests first.
pub struct RegretInsertionRepair;

impl Default for RegretInsertionRepair {
    fn default() -> Self {
        Self::new()
    }
}

impl RegretInsertionRepair {
    pub fn new() -> Self {
        Self
    }
}

impl<T, C, R> RepairProcedure<T, C, R> for RegretInsertionRepair
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RegretInsertionRepair"
    }

    fn repair<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ruined_outcome: RuinOutcome<'p, T>,
    ) -> Plan<'p, T> {
        let mut pb = ctx.builder(ruined_outcome.ruined_plan);
        let mut remaining: Vec<RequestIndex> = ruined_outcome.ruined.clone();

        while !remaining.is_empty() {
            #[allow(clippy::type_complexity)]
            let mut best_choice: Option<(usize, Cost, TP<T>, FB<T>, Cost)> = None;

            for (i, &req) in remaining.iter().enumerate() {
                let mut best1: Option<Cand<T>> = None;
                let mut best2: Option<Cand<T>> = None;

                for (start, fb) in candidates_for_request(&pb, req) {
                    if let Some(cost) = pb.peek_cost(req, start, &fb) {
                        match best1 {
                            None => best1 = Some((cost, start, fb)),
                            Some((c1, _, _)) if cost < c1 => {
                                best2 = best1.take();
                                best1 = Some((cost, start, fb));
                            }
                            _ => match best2 {
                                None => best2 = Some((cost, start, fb)),
                                Some((c2, _, _)) if cost < c2 => best2 = Some((cost, start, fb)),
                                _ => {}
                            },
                        }
                    }
                }

                let Some((c1, s1, fb1)) = best1 else {
                    continue;
                };

                let c2 = best2.map(|t| t.0).unwrap_or(i64::MAX);
                let regret: Cost = c2.saturating_sub(c1);

                match best_choice {
                    None => best_choice = Some((i, c1, s1, fb1, regret)),
                    Some((_, bc1, _, _, br)) => {
                        if regret > br || (regret == br && c1 < bc1) {
                            best_choice = Some((i, c1, s1, fb1, regret));
                        }
                    }
                }
            }

            let Some((i_rem, _best_cost, start, fb, _reg)) = best_choice else {
                break;
            };

            let req = remaining.remove(i_rem);
            let _ = pb.propose_assignment(req, start, &fb);
        }

        pb.finalize()
    }
}

/// Best-fit by slack: choose the free interval with minimal slack (interval.length - pt),
/// tie-breaking by lower cost then earlier start.
pub struct BestFitBySlackInsertionRepair;

impl Default for BestFitBySlackInsertionRepair {
    fn default() -> Self {
        Self::new()
    }
}

impl BestFitBySlackInsertionRepair {
    pub fn new() -> Self {
        Self
    }
}

impl<T, C, R> RepairProcedure<T, C, R> for BestFitBySlackInsertionRepair
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "BestFitBySlackInsertionRepair"
    }

    fn repair<'b, 'r, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut RepairProcedureContext<'b, 'r, 'c, 's, 'm, 'p, T, C, R>,
        ruined_outcome: RuinOutcome<'p, T>,
    ) -> Plan<'p, T> {
        let mut pb = ctx.builder(ruined_outcome.ruined_plan);

        for &req in &ruined_outcome.ruined {
            #[allow(clippy::type_complexity)]
            let mut best: Option<(Cost, Cost, TP<T>, FB<T>)> = None; // (slack, cost, start, fb)

            for (start, fb) in candidates_for_request(&pb, req) {
                let pt = match pb.model().processing_time(req, fb.berth_index()) {
                    Some(pt) => pt,
                    None => continue,
                };
                let len: TD<T> = fb.interval().length();
                let slack_cost: Cost = (len - pt).value().into();

                let cost = match pb.peek_cost(req, start, &fb) {
                    Some(c) => c,
                    None => continue,
                };

                match &best {
                    None => best = Some((slack_cost, cost, start, fb)),
                    Some((bsl, bc, bs, _)) => {
                        if slack_cost < *bsl
                            || (slack_cost == *bsl && cost < *bc)
                            || (slack_cost == *bsl && cost == *bc && start < *bs)
                        {
                            best = Some((slack_cost, cost, start, fb));
                        }
                    }
                }
            }

            if let Some((_sl, _c, start, fb)) = best {
                let _ = pb.propose_assignment(req, start, &fb);
            }
        }

        pb.finalize()
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
            solver_state::SolverState,
            terminal::terminalocc::{TerminalOccupancy, TerminalWrite},
        },
    };
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{
        common::FlexibleKind,
        prelude::{AssignmentContainer, Berth, BerthIdentifier, Problem, RequestIdentifier},
        problem::{asg::Assignment, builder::ProblemBuilder, req::Request},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng as StdRng;
    use std::collections::BTreeMap;

    type T = i64;

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
        let b1 = Berth::from_windows(bid(1), [iv(0, 1000)]);
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        let r1 = Request::<FlexibleKind, T>::new(rid(1), iv(0, 200), 1, pt1).unwrap();

        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(5));
        let r2 = Request::<FlexibleKind, T>::new(rid(2), iv(0, 200), 1, pt2).unwrap();

        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(b1);

        let fixed = AssignmentContainer::<_, T, Assignment<_, T>>::new();
        let mut flex =
            berth_alloc_model::problem::req::RequestContainer::<T, Request<FlexibleKind, T>>::new();
        flex.insert(r1);
        flex.insert(r2);

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn make_unassigned_state(
        problem: &Problem<T>,
    ) -> (SolverModel<'_, T>, SolverState<'_, T>, DefaultCostEvaluator) {
        let model = SolverModel::try_from(problem).expect("model ok");
        let term = TerminalOccupancy::new(problem.berths().iter());
        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let eval = DefaultCostEvaluator;
        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);
        (model, state, eval)
    }

    #[test]
    fn cheapest_insertion_assigns_all_on_single_berth() {
        let problem = problem_one_berth_two_flex();
        let (model, state, eval) = make_unassigned_state(&problem);

        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let r2 = model.index_manager().request_index(rid(2)).unwrap();

        let ruined = vec![r1, r2];
        let baseline = Plan::empty();
        let outcome = crate::search::lns::RuinOutcome::new(baseline, ruined);

        let mut rng = StdRng::seed_from_u64(42);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = crate::search::lns::RepairProcedureContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut buffer,
        );

        let mut repair = CheapestInsertionRepair::new();
        let plan = repair.repair(&mut ctx, outcome);

        assert_eq!(plan.decision_var_patches.len(), 2);
        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -2);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn random_feasible_insertion_assigns_one_when_possible() {
        // One berth, one request
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        let r = Request::<FlexibleKind, T>::new(rid(10), iv(0, 100), 2, pt).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_flexible(r);
        let problem = builder.build().unwrap();

        let (model, state, eval) = make_unassigned_state(&problem);
        let rix = model.index_manager().request_index(rid(10)).unwrap();

        let ruined = vec![rix];
        let baseline = Plan::empty();
        let outcome = crate::search::lns::RuinOutcome::new(baseline, ruined);

        let mut rng = StdRng::seed_from_u64(7);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = crate::search::lns::RepairProcedureContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut buffer,
        );

        let mut repair = RandomFeasibleInsertionRepair::new();
        let plan = repair.repair(&mut ctx, outcome);

        assert_eq!(plan.decision_var_patches.len(), 1);
        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -1);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn regret_insertion_assigns_all_when_space_exists() {
        // Two berths, three flex requests
        let b1 = Berth::from_windows(bid(1), [iv(0, 100)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 100)]);
        let mut reqs = Vec::new();

        // R1 allowed on both, PT 10/10
        let mut pt1 = BTreeMap::new();
        pt1.insert(bid(1), td(10));
        pt1.insert(bid(2), td(10));
        reqs.push(Request::<FlexibleKind, T>::new(rid(1), iv(0, 100), 1, pt1).unwrap());
        // R2 allowed on both, PT 15/15
        let mut pt2 = BTreeMap::new();
        pt2.insert(bid(1), td(15));
        pt2.insert(bid(2), td(15));
        reqs.push(Request::<FlexibleKind, T>::new(rid(2), iv(0, 100), 2, pt2).unwrap());
        // R3 only on b2, PT 5
        let mut pt3 = BTreeMap::new();
        pt3.insert(bid(2), td(5));
        reqs.push(Request::<FlexibleKind, T>::new(rid(3), iv(0, 100), 3, pt3).unwrap());

        let mut pb = ProblemBuilder::new();
        pb.add_berth(b1).add_berth(b2);
        for r in reqs {
            pb.add_flexible(r);
        }
        let problem = pb.build().unwrap();

        let (model, state, eval) = make_unassigned_state(&problem);
        let r1 = model.index_manager().request_index(rid(1)).unwrap();
        let r2 = model.index_manager().request_index(rid(2)).unwrap();
        let r3 = model.index_manager().request_index(rid(3)).unwrap();

        let ruined = vec![r1, r2, r3];
        let outcome = crate::search::lns::RuinOutcome::new(Plan::empty(), ruined);

        let mut rng = StdRng::seed_from_u64(1234);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = crate::search::lns::RepairProcedureContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut buffer,
        );

        let mut repair = RegretInsertionRepair::new();
        let plan = repair.repair(&mut ctx, outcome);

        assert_eq!(plan.decision_var_patches.len(), 3);
        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -3);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn best_fit_by_slack_prefers_tight_gap() {
        // Two berths: b1 has a tight free gap [40,50), b2 is fully free [0,50)
        // One request PT=10 allowed on both => best-fit by slack should choose b1@[40,50)
        let b1 = Berth::from_windows(bid(1), [iv(0, 50)]);
        let b2 = Berth::from_windows(bid(2), [iv(0, 50)]);

        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(10));
        pt.insert(bid(2), td(10));
        let r = Request::<FlexibleKind, T>::new(rid(10), iv(0, 50), 1, pt).unwrap();

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1.clone()).add_berth(b2.clone());
        builder.add_flexible(r);
        let problem = builder.build().unwrap();

        let model = SolverModel::try_from(&problem).unwrap();
        let eval = DefaultCostEvaluator;

        // Terminal occupancy with b1 occupied [0,40), free [40,50); b2 free [0,50)
        let mut term = TerminalOccupancy::new(problem.berths().iter());
        term.occupy(
            model.index_manager().berth_index(bid(1)).unwrap(),
            iv(0, 40),
        )
        .unwrap();

        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);

        let rix = model.index_manager().request_index(rid(10)).unwrap();

        let ruined = vec![rix];
        let outcome = crate::search::lns::RuinOutcome::new(Plan::empty(), ruined);

        let mut rng = StdRng::seed_from_u64(99);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = crate::search::lns::RepairProcedureContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut buffer,
        );

        let mut repair = BestFitBySlackInsertionRepair::new();
        let plan = repair.repair(&mut ctx, outcome);

        assert_eq!(plan.decision_var_patches.len(), 1);
        let patch = &plan.decision_var_patches[0];
        let bi1 = model.index_manager().berth_index(bid(1)).unwrap();
        if let crate::state::decisionvar::DecisionVar::Assigned(d) = patch.patch {
            assert_eq!(d.berth_index, bi1, "should choose b1");
            assert_eq!(d.start_time, tp(40), "should start at tight gap start");
        } else {
            panic!("expected assigned DV");
        }

        assert!(plan.fitness_delta.delta_cost > 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, -1);
        assert!(!plan.terminal_delta.is_empty());
    }

    #[test]
    fn candidates_respect_processing_time() {
        // One berth [0, 20). Free intervals [0,5), [10,20)
        // One request with PT=12 should NOT fit anywhere (no candidate)
        let b1 = Berth::from_windows(bid(1), [iv(0, 20)]);
        let mut pt = BTreeMap::new();
        pt.insert(bid(1), td(12));
        let r = Request::<FlexibleKind, T>::new(rid(1), iv(0, 20), 1, pt).unwrap();

        let mut pbuilder = ProblemBuilder::new();
        pbuilder.add_berth(b1);
        pbuilder.add_flexible(r);
        let problem = pbuilder.build().unwrap();

        let model = SolverModel::try_from(&problem).unwrap();
        let eval = DefaultCostEvaluator;

        // Build terminal occupancy with an occupied middle gap to create [0,5), [10,20)
        let mut term = TerminalOccupancy::new(problem.berths().iter());
        let bix = model.index_manager().berth_index(bid(1)).unwrap();
        term.occupy(bix, iv(5, 10)).unwrap();

        let dvars = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let fitness = eval.eval_fitness(&model, &dvars);
        let state = SolverState::new(DecisionVarVec::from(dvars), term, fitness);

        let rix = model.index_manager().request_index(rid(1)).unwrap();

        let ruined = vec![rix];
        let outcome = crate::search::lns::RuinOutcome::new(Plan::empty(), ruined);

        let mut rng = StdRng::seed_from_u64(55);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = crate::search::lns::RepairProcedureContext::new(
            &model,
            &state,
            &eval,
            &mut rng,
            &mut buffer,
        );

        let mut cheapest = CheapestInsertionRepair::new();
        let plan = cheapest.repair(&mut ctx, outcome);

        // No feasible free interval long enough (12) exists -> 0 candidates -> 0 patches
        assert_eq!(plan.decision_var_patches.len(), 0);
        assert_eq!(plan.fitness_delta.delta_cost, 0);
        assert_eq!(plan.fitness_delta.delta_unassigned, 0);
        assert!(plan.terminal_delta.is_empty());
    }
}
