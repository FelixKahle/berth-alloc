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
//
use berth_alloc_core::{math::emwa::Ewma, prelude::Cost};
use std::time::{Duration, Instant};

#[inline]
fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

/// Unified per-operator stats (strategy-local).
#[derive(Debug, Clone)]
pub struct OperatorStats {
    /// attempted `propose()` calls (including None)
    pub calls: u64,
    /// `propose()` produced a plan (`Some(plan)`)
    pub successes: u64,
    /// produced plan was accepted/applied
    pub accepts: u64,

    /// Sum of TRUE/base deltas over accepted steps (new_cost - old_cost).
    /// Negative is improvement.
    pub total_true_delta: Cost,

    /// Sum of an optional "total" delta (may be the same as true delta by default).
    /// Negative is improvement.
    pub total_delta: Cost,

    // -------- Smoothed metrics (EWMA) --------
    /// wall time per `propose()` in milliseconds
    pub ew_runtime_ms: Ewma<f64, f64>,
    /// TRUE/base delta per accepted step (signed; negative is good)
    pub ew_delta_true: Ewma<f64, f64>,
    /// acceptance probability over produced plans (0/1 per produced outcome)
    pub ew_accept_prob: Ewma<f64, f64>,

    // -------- Last outcome (telemetry) --------
    pub last_runtime_ms: f64,
    pub last_accepted: bool,
}

impl Default for OperatorStats {
    fn default() -> Self {
        Self {
            calls: 0,
            successes: 0,
            accepts: 0,
            total_true_delta: 0,
            total_delta: 0,
            ew_runtime_ms: Ewma::new(0.20).expect("EWMA alpha (runtime)"),
            ew_delta_true: Ewma::new(0.20).expect("EWMA alpha (delta_true)"),
            ew_accept_prob: Ewma::new(0.25).expect("EWMA alpha (accept_prob)"), // ↑ from 0.10
            last_runtime_ms: 0.0,
            last_accepted: false,
        }
    }
}

impl OperatorStats {
    pub fn new(
        runtime_smooth_alpha: f64,
        delta_true_smooth_alpha: f64,
        accept_prob_smooth_alpha: f64,
    ) -> Self {
        Self {
            calls: 0,
            successes: 0,
            accepts: 0,
            total_true_delta: 0,
            total_delta: 0,
            ew_runtime_ms: Ewma::new(runtime_smooth_alpha).expect("EWMA alpha (runtime)"),
            ew_delta_true: Ewma::new(delta_true_smooth_alpha).expect("EWMA alpha (delta_true)"),
            ew_accept_prob: Ewma::new(accept_prob_smooth_alpha).expect("EWMA alpha (accept_prob)"),
            last_runtime_ms: 0.0,
            last_accepted: false,
        }
    }

    /// Start a timing span around `propose()`.
    #[inline]
    pub fn propose_started() -> Instant {
        Instant::now()
    }

    /// Record the outcome of `propose()` (whether it produced a plan).
    pub fn record_propose(&mut self, start: Instant, produced: bool) {
        self.calls = self.calls.saturating_add(1);

        let dt_ms = duration_ms(start.elapsed());
        self.last_runtime_ms = dt_ms;
        self.ew_runtime_ms.observe(dt_ms);

        if produced {
            self.successes = self.successes.saturating_add(1);
        }
        // acceptance recorded later via `record_outcome*()`
    }

    /// Record whether a produced plan was accepted and its TRUE/base delta.
    /// `delta_true` = new_cost - old_cost (negative improvement is good).
    /// Also accumulates `total_delta` with the same value by default.
    pub fn record_outcome(&mut self, accepted: bool, delta_true: f64) {
        self.record_outcome_dual(accepted, delta_true, delta_true);
    }

    /// Record outcome, separating true/base delta from a second total delta metric.
    /// Engines that track an alternate "total" delta (e.g., augmented cost) can use this.
    pub fn record_outcome_dual(&mut self, accepted: bool, delta_true: f64, delta_total: f64) {
        self.last_accepted = accepted;

        if accepted {
            self.accepts = self.accepts.saturating_add(1);
            self.ew_delta_true.observe(delta_true);
            self.ew_accept_prob.observe(1.0);

            // Accumulate only accepted deltas
            self.total_true_delta = self.total_true_delta.saturating_add(delta_true as Cost);
            self.total_delta = self.total_delta.saturating_add(delta_total as Cost);
        } else {
            self.ew_accept_prob.observe(0.0);
        }
    }

    // ---------------- Convenience ratios ----------------

    /// Acceptance ratio across all calls (includes cases where no plan was produced).
    #[inline]
    pub fn acceptance_ratio_all(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.accepts as f64) / (self.calls as f64)
        }
    }

    /// Acceptance ratio conditional on a plan being produced.
    #[inline]
    pub fn acceptance_ratio_when_produced(&self) -> f64 {
        if self.successes == 0 {
            0.0
        } else {
            (self.accepts as f64) / (self.successes as f64)
        }
    }

    /// Production ratio: how often this operator yields a candidate plan.
    #[inline]
    pub fn production_ratio(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.successes as f64) / (self.calls as f64)
        }
    }

    // ---------------- Resets ----------------

    /// Reset only EWMAs (keep counters and totals).
    pub fn reset_ewmas(&mut self) {
        let a_r = self.ew_runtime_ms.alpha();
        let a_d = self.ew_delta_true.alpha();
        let a_p = self.ew_accept_prob.alpha();
        self.ew_runtime_ms = Ewma::new(a_r).expect("EWMA alpha (runtime)");
        self.ew_delta_true = Ewma::new(a_d).expect("EWMA alpha (delta_true)");
        self.ew_accept_prob = Ewma::new(a_p).expect("EWMA alpha (accept_prob)");
    }

    /// Hard reset everything.
    pub fn reset_all(&mut self) {
        *self = Self::new(
            self.ew_runtime_ms.alpha(),
            self.ew_delta_true.alpha(),
            self.ew_accept_prob.alpha(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_updates_counters_and_ewma() {
        let mut s = OperatorStats::default();

        // call with no plan
        let t0 = OperatorStats::propose_started();
        s.record_propose(t0, false);
        assert_eq!(s.calls, 1);
        assert_eq!(s.successes, 0);
        assert!(s.ew_runtime_ms.value().unwrap() >= 0.0);

        // produced but rejected
        let t1 = OperatorStats::propose_started();
        s.record_propose(t1, true);
        s.record_outcome(false, 0.0);
        assert_eq!(s.calls, 2);
        assert_eq!(s.successes, 1);
        assert_eq!(s.accepts, 0);

        // produced and accepted (improvement)
        let t2 = OperatorStats::propose_started();
        s.record_propose(t2, true);
        s.record_outcome(true, -5.0);
        assert_eq!(s.accepts, 1);
        assert_eq!(s.total_true_delta, -5);
        assert_eq!(s.total_delta, -5);
        assert!(s.ew_delta_true.value().unwrap() <= 0.0);
        assert!(s.acceptance_ratio_when_produced() > 0.0);
    }

    #[test]
    fn test_dual_outcome_accumulates_separately() {
        let mut s = OperatorStats::default();
        let t = OperatorStats::propose_started();
        s.record_propose(t, true);
        s.record_outcome_dual(true, -3.0, -7.0);
        assert_eq!(s.total_true_delta, -3);
        assert_eq!(s.total_delta, -7);
    }
}
