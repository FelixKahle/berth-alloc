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

use crate::monitor::{controller::GlobalController, lease::Lease, stop::StopToken};
use std::fmt;
use std::sync::Arc;

/// Thread-local termination helper that:
/// - consumes optional iteration/neighbor leases without atomics on the hot path
/// - samples global/slow checks (time budget, no-improvement) at a configurable frequency
#[derive(Debug)]
pub struct Termination {
    stop: StopToken,
    sample_every: u32,
    sample_cnt: u32,

    // local leases (optional, zero-atomic hot path)
    iter_lease: Option<Lease>,
    neigh_lease: Option<Lease>,

    // shared controller for no-improve checks
    ctrl: Arc<GlobalController>,
}

impl Termination {
    /// Construct from a shared GlobalController.
    pub fn from_controller(ctrl: Arc<GlobalController>) -> Self {
        Self {
            stop: ctrl.stop.clone(),
            sample_every: ctrl.limits.sample_every.max(1),
            sample_cnt: 0,
            iter_lease: ctrl.make_iteration_lease(),
            neigh_lease: ctrl.make_neighbor_lease(),
            ctrl,
        }
    }

    /// Returns true if termination should occur according to sampled checks.
    /// Sampling reduces overhead on the hot path.
    #[inline]
    fn sampled_stop_check(&mut self) -> bool {
        self.sample_cnt = self.sample_cnt.wrapping_add(1);
        if !self.sample_cnt.is_multiple_of(self.sample_every) {
            return false;
        }

        // Time limit (absolute)
        if self.ctrl.time_exceeded() {
            self.stop.request_stop();
            return true;
        }

        // No-improvement (time)
        if let Some(w) = self.ctrl.limits.no_improve_time {
            let ns = self.ctrl.improve.since_last_improve_ns();
            if ns >= w.as_nanos() as u64 {
                self.stop.request_stop();
                return true;
            }
        }

        // No-improvement (accepted iterations)
        if let Some(k) = self.ctrl.limits.no_improve_iters
            && self.ctrl.improve.accepted_since_last_improve() >= k
        {
            self.stop.request_stop();
            return true;
        }

        // External stop
        self.stop.is_set()
    }

    /// Call at start of each DB iteration (outer loop).
    #[inline]
    pub fn tick_iteration(&mut self) -> bool {
        if let Some(l) = &mut self.iter_lease
            && !l.take_one()
        {
            self.stop.request_stop();
            return true;
        }
        self.sampled_stop_check()
    }

    /// Call once per generated neighbor in the operator loop.
    #[inline]
    pub fn tick_neighbor(&mut self) -> bool {
        if let Some(l) = &mut self.neigh_lease
            && !l.take_one()
        {
            self.stop.request_stop();
            return true;
        }
        self.sampled_stop_check()
    }

    /// Fast path for outer loop: just check global stop without sampling.
    #[inline]
    pub fn should_stop_fast(&self) -> bool {
        self.stop.is_set()
    }

    /// Returns whether an iteration lease is configured.
    #[inline]
    pub fn has_iteration_budget(&self) -> bool {
        self.iter_lease.is_some()
    }

    /// Returns whether a neighbor lease is configured.
    #[inline]
    pub fn has_neighbor_budget(&self) -> bool {
        self.neigh_lease.is_some()
    }

    /// Returns the sampling period.
    #[inline]
    pub fn sample_every(&self) -> u32 {
        self.sample_every
    }
}

impl fmt::Display for Termination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Termination{{ stop_set: {}, sample_every: {}, sample_cnt: {}, iter_budget: {}, neigh_budget: {} }}",
            self.stop.is_set(),
            self.sample_every,
            self.sample_cnt,
            self.iter_lease.is_some(),
            self.neigh_lease.is_some()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::controller::{GlobalController, SearchLimits};
    use std::time::Duration;

    #[test]
    fn fast_stop_reflects_external_stop() {
        let ctrl = GlobalController::new(SearchLimits::default());
        let term = Termination::from_controller(ctrl.clone());
        assert!(!term.should_stop_fast());
        ctrl.stop.request_stop();
        assert!(term.should_stop_fast());
    }

    #[test]
    fn iteration_lease_exhausts_after_limit() {
        let limits = SearchLimits::default()
            .with_max_iterations(Some(3))
            .with_iteration_chunk(2)
            .with_sample_every(8);
        let ctrl = GlobalController::new(limits);
        let mut term = Termination::from_controller(ctrl);

        assert!(term.has_iteration_budget());
        // Consume 3 tokens -> first three calls are false, fourth returns true
        assert!(!term.tick_iteration());
        assert!(!term.tick_iteration());
        assert!(!term.tick_iteration());
        assert!(term.tick_iteration());
    }

    #[test]
    fn neighbor_lease_exhausts_after_limit() {
        let limits = SearchLimits::default()
            .with_max_neighbors(Some(2))
            .with_neighbor_chunk(1)
            .with_sample_every(8);
        let ctrl = GlobalController::new(limits);
        let mut term = Termination::from_controller(ctrl);

        assert!(term.has_neighbor_budget());
        assert!(!term.tick_neighbor());
        assert!(!term.tick_neighbor());
        assert!(term.tick_neighbor());
    }

    #[test]
    fn zero_duration_times_out_immediately_on_sample() {
        let limits = SearchLimits::default()
            .with_max_duration(Some(Duration::from_secs(0)))
            .with_sample_every(1);
        let ctrl = GlobalController::new(limits);
        let mut term = Termination::from_controller(ctrl);

        // With sample_every=1, we check each call.
        assert!(term.tick_iteration());
    }

    #[test]
    fn no_improve_time_zero_triggers_stop() {
        let limits = SearchLimits::default()
            .with_no_improve_time(Some(Duration::from_secs(0)))
            .with_sample_every(1);
        let ctrl = GlobalController::new(limits);
        let mut term = Termination::from_controller(ctrl);

        assert!(term.tick_neighbor());
    }

    #[test]
    fn no_improve_iters_triggers_after_threshold() {
        let limits = SearchLimits::default()
            .with_no_improve_iters(Some(2))
            .with_sample_every(1);
        let ctrl = GlobalController::new(limits);
        let mut term = Termination::from_controller(ctrl.clone());

        // Two accepted events recorded globally
        ctrl.on_accepted();
        ctrl.on_accepted();

        // Should stop on the next tick due to accepted-since-improve >= 2
        assert!(term.tick_iteration());
    }

    #[test]
    fn display_contains_key_fields() {
        let ctrl = GlobalController::new(
            SearchLimits::default()
                .with_max_neighbors(Some(1))
                .with_sample_every(4),
        );
        let term = Termination::from_controller(ctrl);
        let s = term.to_string();
        assert!(s.contains("Termination{"));
        assert!(s.contains("sample_every: 4"));
        assert!(s.contains("neigh_budget: true"));
    }
}
