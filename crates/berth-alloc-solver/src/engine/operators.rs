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

use crate::search::operator::{DestroyOperator, LocalMoveOperator, RepairOperator};
use crate::search::planner::{CostEvaluator, PlanningContext};
use crate::state::plan::Plan;
use berth_alloc_core::math::emwa::Ewma;
use berth_alloc_core::prelude::Cost;
use rand::Rng;
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;

type Ew64 = Ewma<f64, f64>;

#[derive(Clone, Debug)]
pub struct OperatorStats {
    pub attempts: u64,
    pub proposals: u64,
    pub accepts: u64,
    pub fails: u64,
    pub runtime_ns: u128,   // accumulated wall time
    pub total_delta: i64,   // sum of positive deltas (improvements)
    pub last_delta: Cost,   // last observed delta
    pub reward_ewma: Ew64,  // smoothed positive delta / sqrt(time)
    pub runtime_ewma: Ew64, // smoothed runtime (ns)
    pub accept_ewma: Ew64,  // smoothed acceptance prob (0/1)
}

impl Default for OperatorStats {
    fn default() -> Self {
        Self {
            attempts: 0,
            proposals: 0,
            accepts: 0,
            fails: 0,
            runtime_ns: 0,
            total_delta: 0,
            last_delta: 0,
            // Tuned to be responsive but stable
            reward_ewma: Ew64::new(0.30).expect("alpha ok"),
            runtime_ewma: Ew64::new(0.15).expect("alpha ok"),
            accept_ewma: Ew64::new(0.15).expect("alpha ok"),
        }
    }
}

impl fmt::Display for OperatorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "att={},prop={},acc={},fail={},rw={:.3},rt={:.3},acp={:.3}",
            self.attempts,
            self.proposals,
            self.accepts,
            self.fails,
            self.reward_ewma.value().unwrap_or(0.0),
            self.runtime_ewma.value().unwrap_or(0.0),
            self.accept_ewma.value().unwrap_or(0.0)
        )
    }
}

impl OperatorStats {
    #[inline]
    fn score(&self) -> f64 {
        // E3: reward × speed × acceptance smoothing
        // reward := pos_delta / sqrt(ns+1)
        // speed  := 1/sqrt(rt_ewma + 1)   (favor faster ops)
        // acceptance factor := 0.25 + accept_ewma  (keeps exploration alive)
        let reward = self.reward_ewma.value().unwrap_or(0.0).max(0.0);
        let rt = self.runtime_ewma.value().unwrap_or(0.0).max(0.0);
        let speed = 1.0 / (rt.sqrt() + 1.0);
        let acc = 0.25 + self.accept_ewma.value().unwrap_or(0.0).clamp(0.0, 1.0);
        reward * speed * acc
    }
}

#[derive(Clone)]
pub struct SoftmaxSelector {
    /// Used when caller does not provide temperature.
    pub base_temp: f64,
    /// Floor for each exp weight (pre-normalization). Prevents starvation.
    pub min_p: f64,
    /// Optional nonlinearity on scores before softmax (s' = clamp(s,lo,hi)^power).
    pub power: f64,
    /// Optional clipping of raw scores before power/shaping.
    pub clip_lo: f64,
    pub clip_hi: f64,
}

impl Default for SoftmaxSelector {
    fn default() -> Self {
        Self {
            base_temp: 1.0,
            min_p: 1e-6,
            power: 1.0,
            clip_lo: -1.0e12, // effectively no clip by default
            clip_hi: 1.0e12,
        }
    }
}

impl SoftmaxSelector {
    pub fn with_base_temp(mut self, t: f64) -> Self {
        self.base_temp = t.max(1e-9);
        self
    }
    pub fn with_min_p(mut self, p: f64) -> Self {
        self.min_p = p.max(0.0);
        self
    }
    pub fn with_power(mut self, p: f64) -> Self {
        self.power = p.max(1e-9);
        self
    }
    pub fn with_clip(mut self, lo: f64, hi: f64) -> Self {
        self.clip_lo = lo.min(hi);
        self.clip_hi = hi;
        self
    }

    /// Pick index using softmax over `scores`, with optional:
    /// - `temp`: strategy-supplied temperature (SA); else `base_temp`.
    /// - `map`: optional score shaping callback: `map(raw_score, temp) -> shaped_score`.
    ///
    /// Numerically stable; applies: clip -> map (optional) -> power -> softmax(temp).
    pub fn pick_with<R: rand::Rng>(
        &self,
        rng: &mut R,
        scores: &[f64],
        temp: Option<f64>,
        map: Option<fn(f64, f64) -> f64>,
    ) -> usize {
        debug_assert!(!scores.is_empty());
        let t = temp.unwrap_or(self.base_temp).max(1e-9);

        // Transform scores
        let transformed: Vec<f64> = scores
            .iter()
            .map(|&s| {
                let mut v = s.clamp(self.clip_lo, self.clip_hi);
                if let Some(f) = map {
                    v = f(v, t);
                }
                v = v.powf(self.power);
                v
            })
            .collect();

        // Stabilize
        let m = transformed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Softmax with floor
        let expw: Vec<f64> = transformed
            .iter()
            .map(|&v| ((v - m) / t).exp().max(self.min_p))
            .collect();

        let sum: f64 = expw.iter().sum();
        let mut r = rng.random::<f64>() * sum;
        for (i, w) in expw.iter().enumerate() {
            r -= *w;
            if r <= 0.0 {
                return i;
            }
        }
        expw.len() - 1
    }

    /// Convenience when you don’t need shaping or an explicit temp.
    #[inline]
    pub fn pick<R: rand::Rng>(&self, rng: &mut R, scores: &[f64], temp: Option<f64>) -> usize {
        self.pick_with(rng, scores, temp, None)
    }
}

// ======================= Family Policy =======================

pub trait FamilyApi<T: Copy + Ord, C: CostEvaluator<T>, R: Rng> {
    type DynOp: ?Sized + Send + Sync;

    fn name(op: &Self::DynOp) -> &str;

    fn call<'b, 'c, 's, 'm, 'p>(
        op: &Self::DynOp,
        ctx: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>>;
}

pub struct LocalFamily;
impl<T: Copy + Ord, C: CostEvaluator<T>, R: Rng> FamilyApi<T, C, R> for LocalFamily {
    type DynOp = dyn LocalMoveOperator<T, C, R>;
    #[inline]
    fn name(op: &Self::DynOp) -> &str {
        op.name()
    }
    #[inline]
    fn call<'b, 'c, 's, 'm, 'p>(
        op: &Self::DynOp,
        ctx: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        op.propose(ctx, rng)
    }
}

pub struct DestroyFamily;
impl<T: Copy + Ord, C: CostEvaluator<T>, R: Rng> FamilyApi<T, C, R> for DestroyFamily {
    type DynOp = dyn DestroyOperator<T, C, R>;
    #[inline]
    fn name(op: &Self::DynOp) -> &str {
        op.name()
    }
    #[inline]
    fn call<'b, 'c, 's, 'm, 'p>(
        op: &Self::DynOp,
        ctx: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        op.propose(ctx, rng)
    }
}

pub struct RepairFamily;
impl<T: Copy + Ord, C: CostEvaluator<T>, R: Rng> FamilyApi<T, C, R> for RepairFamily {
    type DynOp = dyn RepairOperator<T, C, R>;
    #[inline]
    fn name(op: &Self::DynOp) -> &str {
        op.name()
    }
    #[inline]
    fn call<'b, 'c, 's, 'm, 'p>(
        op: &Self::DynOp,
        ctx: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
    ) -> Option<Plan<'p, T>> {
        op.repair(ctx, rng)
    }
}

// ======================= Pool =======================

pub struct OperatorPool<T, C, R, F>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: Rng,
    F: FamilyApi<T, C, R>,
{
    ops: Vec<Box<F::DynOp>>,
    stats: Vec<OperatorStats>,
    selector: SoftmaxSelector,
    _phantom: PhantomData<(T, C, R, F)>,
}

impl<T, C, R, F> Default for OperatorPool<T, C, R, F>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: Rng,
    F: FamilyApi<T, C, R>,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R, F> OperatorPool<T, C, R, F>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: Rng,
    F: FamilyApi<T, C, R>,
{
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            stats: Vec::new(),
            selector: SoftmaxSelector::default(),
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn with_selector(mut self, s: SoftmaxSelector) -> Self {
        self.selector = s;
        self
    }

    pub fn push(&mut self, op: Box<F::DynOp>) {
        self.ops.push(op);
        self.stats.push(OperatorStats::default());
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    #[inline]
    pub fn stats(&self) -> &[OperatorStats] {
        &self.stats
    }
    #[inline]
    pub fn stats_mut(&mut self) -> &mut [OperatorStats] {
        &mut self.stats
    }
    #[inline]
    pub fn name(&self, idx: usize) -> &str {
        F::name(&*self.ops[idx])
    }

    fn scores(&self) -> Vec<f64> {
        self.stats.iter().map(|s| s.score()).collect()
    }

    /// Select, call, and time an operator. Returns a handle that the caller
    /// must consume via `.accept(delta)` or `.reject()`. No other bookkeeping needed.
    pub fn apply<'b, 'c, 's, 'm, 'p>(
        &mut self,
        ctx: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
        rng: &mut R,
        temp: Option<f64>,
    ) -> ProposedOp<'_, 'p, T, C, R, F> {
        debug_assert!(!self.ops.is_empty(), "OperatorPool is empty");

        let idx = {
            // If you want strictly uniform during early warmup, you could gate this by attempts.
            let sc = self.scores();
            self.selector.pick(rng, &sc, temp)
        };

        self.stats[idx].attempts += 1;

        let t0 = Instant::now();
        let plan = F::call(&*self.ops[idx], ctx, rng);
        let dt = t0.elapsed().as_nanos();

        self.stats[idx].runtime_ns += dt;
        self.stats[idx].runtime_ewma.observe(dt as f64);

        let proposed = plan.is_some();
        if proposed {
            self.stats[idx].proposals += 1;
        } else {
            self.stats[idx].fails += 1;
        }

        ProposedOp {
            pool: self,
            idx,
            plan,
            runtime_ns: dt,
            _phantom: PhantomData,
        }
    }
}

// Proposed operation handle — owns a &mut borrow to the pool until accepted/rejected.
pub struct ProposedOp<'pool, 'p, T, C, R, F>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: Rng,
    F: FamilyApi<T, C, R>,
{
    pool: &'pool mut OperatorPool<T, C, R, F>,
    idx: usize,
    plan: Option<Plan<'p, T>>,
    runtime_ns: u128,
    _phantom: PhantomData<(T, C, R, F)>,
}

impl<'pool, 'p, T, C, R, F> ProposedOp<'pool, 'p, T, C, R, F>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: Rng,
    F: FamilyApi<T, C, R>,
{
    #[inline]
    pub fn idx(&self) -> usize {
        self.idx
    }
    #[inline]
    pub fn name(&self) -> &str {
        self.pool.name(self.idx)
    }
    pub fn plan(&self) -> Option<&Plan<'p, T>> {
        self.plan.as_ref()
    }

    pub fn take_plan(&mut self) -> Option<Plan<'p, T>> {
        self.plan.take()
    }

    pub fn accept(&mut self, delta: i64) {
        let s = &mut self.pool.stats[self.idx];
        s.accepts += 1;
        s.last_delta = delta;
        if delta > 0 {
            s.total_delta += delta;
        }
        let pos = (delta as f64).max(0.0);
        let denom = (self.runtime_ns as f64).sqrt() + 1.0;
        s.reward_ewma.observe(pos / denom);
        s.accept_ewma.observe(1.0);
    }

    pub fn reject(&mut self) {
        let s = &mut self.pool.stats[self.idx];
        s.last_delta = 0;
        s.reward_ewma.observe(0.0);
        s.accept_ewma.observe(0.0);
    }
}

// Friendly aliases
pub type LocalPool<T, C, R> = OperatorPool<T, C, R, LocalFamily>;
pub type DestroyPool<T, C, R> = OperatorPool<T, C, R, DestroyFamily>;
pub type RepairPool<T, C, R> = OperatorPool<T, C, R, RepairFamily>;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // Bring in minimal solver plumbing to build a PlanningContext
    use crate::{
        model::solver_model::SolverModel,
        search::planner::{DefaultCostEvaluator, PlanningContext},
        state::{
            decisionvar::{DecisionVar, DecisionVarVec},
            fitness::Fitness,
            solver_state::SolverState,
            terminal::terminalocc::TerminalOccupancy,
        },
    };
    use berth_alloc_core::prelude::{Cost, TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::prelude::*;

    // ---------- Tiny problem/model/context helpers ----------

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
        let mut m = std::collections::BTreeMap::new();
        for (b, d) in pt {
            m.insert(bid(*b), TimeDelta::new(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    fn make_problem() -> Problem<i64> {
        let mut berths = berth_alloc_model::problem::berth::BerthContainer::new();
        berths.insert(berth(1, 0, 1000));

        let fixed = AssignmentContainer::<FixedKind, i64, Assignment<FixedKind, i64>>::new();

        let mut flex = RequestContainer::<i64, Request<FlexibleKind, i64>>::new();
        // one tiny request to instantiate state sizing
        flex.insert(flex_req(1, (0, 100), &[(1, 10)], 1));

        Problem::new(berths, fixed, flex).unwrap()
    }

    fn empty_state<'p>(model: &SolverModel<'p, i64>) -> SolverState<'p, i64> {
        let dv_len = model.flexible_requests_len();
        let dv = DecisionVarVec::from(vec![DecisionVar::unassigned(); dv_len]);
        let term = TerminalOccupancy::new(model.problem().berths().iter());

        // no cost, all flex requests unassigned initially
        let fitness = Fitness::new(Cost::from(0), dv_len);
        SolverState::new(dv, term, fitness)
    }

    fn make_ctx<'b, 'c, 's, 'm, 'p>(
        model: &'m SolverModel<'p, i64>,
        state: &'s SolverState<'p, i64>,
        buffer: &'b mut [DecisionVar<i64>],
    ) -> PlanningContext<'b, 'c, 's, 'm, 'p, i64, DefaultCostEvaluator> {
        PlanningContext::new(model, state, &DefaultCostEvaluator, buffer)
    }

    // ---------- Dummy local operators for testing ----------

    // Always returns Some(plan) with zero deltas.
    struct DummySomeOp;
    impl<T, C, R> crate::search::operator::LocalMoveOperator<T, C, R> for DummySomeOp
    where
        T: Copy + Ord,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "DummySomeOp"
        }
        fn propose<'b, 'c, 's, 'm, 'p>(
            &self,
            _context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
            _rng: &mut R,
        ) -> Option<crate::state::plan::Plan<'p, T>> {
            Some(crate::state::plan::Plan::new_delta(
                Vec::new(),
                crate::state::terminal::delta::TerminalDelta::empty(),
                0.into(),
                0,
            ))
        }
    }

    // Always returns None.
    struct DummyNoneOp;
    impl<T, C, R> crate::search::operator::LocalMoveOperator<T, C, R> for DummyNoneOp
    where
        T: Copy + Ord,
        C: CostEvaluator<T>,
        R: rand::Rng,
    {
        fn name(&self) -> &str {
            "DummyNoneOp"
        }
        fn propose<'b, 'c, 's, 'm, 'p>(
            &self,
            _context: &mut PlanningContext<'b, 'c, 's, 'm, 'p, T, C>,
            _rng: &mut R,
        ) -> Option<crate::state::plan::Plan<'p, T>> {
            None
        }
    }

    // ---------- Tests ----------

    #[test]
    fn test_softmax_selector_argmax_at_low_temperature() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sel = SoftmaxSelector::default()
            .with_base_temp(1e-9)
            .with_min_p(0.0)
            .with_power(1.0);

        let scores = [1.0, 2.0, 10.0];
        for _ in 0..50 {
            let i = sel.pick(&mut rng, &scores, None);
            assert_eq!(i, 2, "low temp should pick argmax deterministically");
        }
    }

    #[test]
    fn test_operator_pool_apply_accept_updates_stats() {
        // Build minimal model/context
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let state = empty_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let mut pool: super::LocalPool<i64, DefaultCostEvaluator, rand::rngs::StdRng> =
            OperatorPool::new();

        pool.push(Box::new(DummySomeOp));
        pool.push(Box::new(DummyNoneOp));

        // Sanity before apply
        assert_eq!(pool.len(), 2);
        let stats_before_all = pool.stats().to_vec(); // snapshot all indices

        // Deterministic RNG for selection
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        // Apply and get the proposed handle
        let mut prop = pool.apply(&mut ctx, &mut rng, Some(1.0));

        // Capture which op was chosen and whether it proposed a plan
        let idx = prop.idx();
        let had_plan = prop.plan().is_some();

        // Consume the handle to release the mutable borrow to the pool
        if had_plan {
            prop.accept(5);
        } else {
            prop.reject();
        }

        // Now we can inspect stats
        let stats_after = &pool.stats()[idx];
        let stats_before = &stats_before_all[idx];

        // attempts must increase by 1 on the chosen op
        assert_eq!(stats_after.attempts, stats_before.attempts + 1);

        if had_plan {
            // proposed + accepted
            assert_eq!(stats_after.proposals, stats_before.proposals + 1);
            assert_eq!(stats_after.accepts, stats_before.accepts + 1);
            assert_eq!(stats_after.fails, stats_before.fails);
            assert_eq!(stats_after.last_delta, 5);
            // EWMA values should be defined and non-negative
            assert!(stats_after.reward_ewma.value().unwrap_or(0.0) >= 0.0);
            assert!(stats_after.accept_ewma.value().unwrap_or(0.0) > 0.0);
        } else {
            // rejected path — no proposal/accept increments, fails may increment depending on None/Some return
            assert_eq!(stats_after.proposals, stats_before.proposals);
            assert_eq!(stats_after.accepts, stats_before.accepts);
            assert_eq!(stats_after.last_delta, 0);
            assert!(stats_after.accept_ewma.value().unwrap_or(0.0) >= 0.0);
        }
    }

    #[test]
    fn test_operator_pool_apply_reject_updates_stats() {
        // Force a None-proposing operator to be in the pool.
        let prob = make_problem();
        let model = SolverModel::try_from(&prob).expect("model ok");
        let state = empty_state(&model);
        let mut buffer = vec![DecisionVar::unassigned(); model.flexible_requests_len()];
        let mut ctx = make_ctx(&model, &state, &mut buffer);

        let mut pool: super::LocalPool<i64, DefaultCostEvaluator, rand::rngs::StdRng> =
            OperatorPool::new();
        pool.push(Box::new(DummyNoneOp));

        let stats_before_all = pool.stats().to_vec();

        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let mut prop = pool.apply(&mut ctx, &mut rng, Some(1.0));

        let idx = prop.idx();
        // Since plan is None, record reject to release borrow
        prop.reject();

        let stats_after = &pool.stats()[idx];
        let stats_before = &stats_before_all[idx];

        assert_eq!(stats_after.attempts, stats_before.attempts + 1);
        // If proposal was None, the pool increments fails during apply() and reject() records zero reward/acceptance
        assert_eq!(stats_after.fails, stats_before.fails + 1);
        assert_eq!(stats_after.proposals, stats_before.proposals);
        assert_eq!(stats_after.accepts, stats_before.accepts);
        assert_eq!(stats_after.last_delta, 0);
    }
}
