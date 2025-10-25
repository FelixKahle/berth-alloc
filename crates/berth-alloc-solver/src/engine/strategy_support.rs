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

use crate::state::fitness::Fitness;
use core::cmp::Ordering;

// ---------------------------------------------------------------------------
// Rolling improvement history & epsilon (data-driven, no magic constants)
// ---------------------------------------------------------------------------

/// Ring buffer for recent strict-improvement magnitudes (positive cost deltas).
#[derive(Clone, Debug)]
pub struct RollingImprovements {
    data: Vec<i64>,
    head: usize,
    len: usize,
    cap: usize,
}

impl RollingImprovements {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            data: vec![0; cap],
            head: 0,
            len: 0,
            cap,
        }
    }
    #[inline]
    pub fn push(&mut self, v: i64) {
        let v = v.max(0);
        self.data[self.head] = v;
        self.head = (self.head + 1) % self.cap;
        self.len = (self.len + 1).min(self.cap);
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Snapshot of current values (unsorted).
    #[inline]
    pub fn values(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.len);
        if self.len == 0 {
            return out;
        }
        let start = (self.head + self.cap - self.len) % self.cap;
        for i in 0..self.len {
            out.push(self.data[(start + i) % self.cap]);
        }
        out
    }
}

/// Median over `Vec<i64>` using partial selection (no full sort).
#[inline]
pub fn median_i64(mut xs: Vec<i64>) -> i64 {
    if xs.is_empty() {
        return 0;
    }
    let mid = xs.len() / 2;
    *xs.select_nth_unstable(mid).1
}

/// Data-driven epsilon = median of last improvements, with a minimum floor.
#[derive(Clone, Debug)]
pub struct MedianHistoryEpsilon {
    history: RollingImprovements,
    min_eps: i64,
}
impl MedianHistoryEpsilon {
    pub fn new(history_capacity: usize, min_eps: i64) -> Self {
        Self {
            history: RollingImprovements::new(history_capacity),
            min_eps: min_eps.max(1),
        }
    }
    #[inline]
    pub fn record(&mut self, improvement_cost_delta: i64) {
        self.history.push(improvement_cost_delta.max(0));
    }
    /// Compute epsilon (≥1). Uses median if enough samples exist.
    #[inline]
    pub fn epsilon(&self) -> i64 {
        if self.history.len() < 3 {
            return self.min_eps.max(1);
        }
        median_i64(self.history.values()).max(self.min_eps).max(1)
    }
    #[inline]
    pub fn history(&self) -> &RollingImprovements {
        &self.history
    }
    #[inline]
    pub fn history_mut(&mut self) -> &mut RollingImprovements {
        &mut self.history
    }
}

// ---------------------------------------------------------------------------
// Stale progress tracking over Fitness
// ---------------------------------------------------------------------------

/// Lexicographic comparison on your concrete `Fitness` (unassigned, then cost).
#[inline]
pub fn lex_cmp(a: &Fitness, b: &Fitness) -> Ordering {
    match a.unassigned_requests.cmp(&b.unassigned_requests) {
        Ordering::Less => Ordering::Less,
        Ordering::Greater => Ordering::Greater,
        Ordering::Equal => a.cost.cmp(&b.cost),
    }
}

/// True if `b` is strictly better than `a` under lexicographic order.
#[inline]
pub fn is_strict_better(a: &Fitness, b: &Fitness) -> bool {
    lex_cmp(a, b) == Ordering::Greater
}

/// Track stale progress (rounds since last strict improvement) and improvement sizes.
#[derive(Clone, Debug)]
pub struct StaleTracker {
    local_best: Fitness,
    rounds_since_improve: usize,
    improvements: RollingImprovements,
    /// After a refetch/reset, wait for the next strict improvement before refetching again.
    waiting_for_next_improvement: bool,
}

impl StaleTracker {
    pub fn new(initial_best: Fitness, history_cap: usize) -> Self {
        Self {
            local_best: initial_best,
            rounds_since_improve: 0,
            improvements: RollingImprovements::new(history_cap),
            waiting_for_next_improvement: false,
        }
    }

    #[inline]
    pub fn local_best(&self) -> &Fitness {
        &self.local_best
    }
    #[inline]
    pub fn rounds_since_improve(&self) -> usize {
        self.rounds_since_improve
    }
    #[inline]
    pub fn improvements(&self) -> &RollingImprovements {
        &self.improvements
    }
    #[inline]
    pub fn improvements_mut(&mut self) -> &mut RollingImprovements {
        &mut self.improvements
    }

    /// Call once per outer round/epoch with the current **true** fitness.
    /// Returns `Some(delta_cost)` if a strict improvement occurred; else `None`.
    pub fn on_round_end(&mut self, current: Fitness) -> Option<i64> {
        if is_strict_better(&self.local_best, &current) {
            let delta = self.local_best.cost.saturating_sub(current.cost).max(0);
            self.local_best = current;
            self.rounds_since_improve = 0;
            self.improvements.push(delta);
            self.waiting_for_next_improvement = false;
            Some(delta)
        } else {
            self.rounds_since_improve = self.rounds_since_improve.saturating_add(1);
            None
        }
    }

    /// True if you’ve been stale for at least `s` rounds and not blocked by cooldown.
    #[inline]
    pub fn is_stale(&self, s: usize) -> bool {
        !self.waiting_for_next_improvement && self.rounds_since_improve >= s.max(1)
    }

    /// Call after refetch/reset to require one strict improvement before refetching again.
    #[inline]
    pub fn arm_cooldown_until_next_improvement(&mut self) {
        self.waiting_for_next_improvement = true;
        self.rounds_since_improve = 0;
    }
}

// ---------------------------------------------------------------------------
// Materiality and DV distance (strategy-agnostic)
// ---------------------------------------------------------------------------

/// Data-driven materiality: incumbent is "worth it" if lex-better, or at equal
/// unassigned improves cost by at least `epsilon` (≥1).
#[inline]
pub fn materially_better(cur: &Fitness, inc: &Fitness, epsilon: i64) -> bool {
    if inc.unassigned_requests < cur.unassigned_requests {
        return true;
    }
    if inc.unassigned_requests > cur.unassigned_requests {
        return false;
    }
    inc.cost + epsilon < cur.cost
}

/// Abstract access to decision variables without binding this module to any state type.
pub trait HasDecisionVars<T> {
    /// Return a read-only slice of decision variables for distance comparison.
    fn decision_vars(&self) -> &[T];
}

/// Fraction of decision variables that differ between two states (∈ [0,1]).
/// Works for any type that implements `HasDecisionVars<T>` where `T: PartialEq`.
#[inline]
pub fn dv_distance<S, T>(a: &S, b: &S) -> f64
where
    S: HasDecisionVars<T>,
    T: PartialEq,
{
    let a_vars = a.decision_vars();
    let b_vars = b.decision_vars();
    let n = core::cmp::min(a_vars.len(), b_vars.len()).max(1);
    let mut diff = 0usize;
    for i in 0..n {
        if a_vars[i] != b_vars[i] {
            diff += 1;
        }
    }
    diff as f64 / n as f64
}

// ---------------------------------------------------------------------------
// Deterministic "kick" (strategy-agnostic)
// ---------------------------------------------------------------------------

/// Minimal interface for applying one attempt of an operator by index.
pub trait ApplyOnceByIndex {
    /// Attempt to apply operator `index` once; return true if it made a change.
    fn apply_once(&mut self, index: usize) -> bool;
}

/// Deterministic kick: apply the first `k` operators once each (indices 0..k).
/// Returns how many operators actually produced a change.
#[inline]
pub fn deterministic_kick<A: ApplyOnceByIndex>(ops: &mut A, k: usize) -> usize {
    let mut applied = 0usize;
    for i in 0..k {
        if ops.apply_once(i) {
            applied += 1;
        }
    }
    applied
}

#[inline]
pub fn patience_from_exploration_budget(
    exploration_batches_per_round: usize,
    inner_steps_mean: usize,
    reshuffle_each_step: bool,
) -> usize {
    let batches = exploration_batches_per_round.max(1);
    let k = if reshuffle_each_step { 8usize } else { 4usize };
    let base = k.saturating_mul(batches);

    // Normalize by how much work is already done inside a round.
    let scale_div = ((inner_steps_mean as f64) / 256.0).clamp(1.0, 4.0);
    ((base as f64) / scale_div).ceil() as usize
}

/// Epochs needed to shrink temperature by factor `beta` (0<beta<1) with per-epoch `alpha` (0<alpha<1).
#[inline]
pub fn epochs_to_shrink(alpha: f64, beta: f64) -> usize {
    debug_assert!(alpha > 0.0 && alpha < 1.0);
    debug_assert!(beta > 0.0 && beta < 1.0);

    let e = (beta.ln() / alpha.ln()).ceil();
    if e.is_finite() && e > 0.0 {
        e as usize
    } else {
        1
    }
}

/// Patience from a cooling schedule: number of epochs to halve temperature.
#[inline]
pub fn patience_from_cooling_halving(alpha_per_epoch: f64) -> usize {
    epochs_to_shrink(alpha_per_epoch, 0.5).max(1)
}

/// Patience from a stagnation pulse threshold: allow one pulse span to fire and one to verify.
#[inline]
pub fn patience_from_pulse_threshold(pulse_stagnation_rounds: usize) -> usize {
    (2 * pulse_stagnation_rounds).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq)]
    struct DV(i32);
    #[derive(Clone)]
    struct DummyState {
        dv: Vec<DV>,
    }
    impl HasDecisionVars<DV> for DummyState {
        fn decision_vars(&self) -> &[DV] {
            &self.dv
        }
    }

    #[test]
    fn test_median() {
        assert_eq!(median_i64(vec![]), 0);
        assert_eq!(median_i64(vec![5]), 5);
        let m = median_i64(vec![3, 9, 1, 7, 5]);
        assert!(m == 5 || m == 3 || m == 7); // any valid median by partial select
    }

    #[test]
    fn test_dv_distance() {
        let a = DummyState {
            dv: vec![DV(1), DV(2), DV(3)],
        };
        let b = DummyState {
            dv: vec![DV(1), DV(9), DV(3)],
        };
        let d = dv_distance(&a, &b);
        assert!((d - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_patience_exploration_budget() {
        let s1 = patience_from_exploration_budget(1, 256, true);
        let s2 = patience_from_exploration_budget(8, 64, false);
        assert!(s1 >= 1 && s2 >= 1);
    }
}
