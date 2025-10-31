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

use crate::monitor::lease::{ChunkDispenser, Lease};
use crate::monitor::stop::{ImprovementState, StopToken};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Limits and sampling parameters that govern when a search should stop.
/// - The “global” limits (neighbors/iterations) are enforced via leases to reduce atomic pressure.
/// - `sample_every` controls how often to check slower termination conditions on hot paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchLimits {
    pub max_duration: Option<Duration>,
    pub max_neighbors: Option<u64>,        // global
    pub max_iterations: Option<u64>,       // global DB iterations
    pub no_improve_time: Option<Duration>, // since last incumbent improvement
    pub no_improve_iters: Option<u64>,     // accepted since last incumbent improvement
    pub neighbor_chunk: u64,               // e.g., 50_000
    pub iteration_chunk: u64,              // e.g., 4_096
    pub sample_every: u32,                 // e.g., 64 (hot-path sampling)
}

impl Default for SearchLimits {
    /// Sensible defaults for production usage: no explicit hard limits,
    /// but tuned chunk and sampling sizes for good performance.
    fn default() -> Self {
        Self {
            max_duration: None,
            max_neighbors: None,
            max_iterations: None,
            no_improve_time: None,
            no_improve_iters: None,
            neighbor_chunk: 50_000,
            iteration_chunk: 4_096,
            sample_every: 64,
        }
    }
}

impl SearchLimits {
    /// Alias that is kept for compatibility with earlier code.
    #[inline]
    pub fn default_fast() -> Self {
        Self::default()
    }

    /// Returns true when there are no hard or no-improvement limits configured.
    #[inline]
    pub fn is_unbounded(&self) -> bool {
        self.max_duration.is_none()
            && self.max_neighbors.is_none()
            && self.max_iterations.is_none()
            && self.no_improve_time.is_none()
            && self.no_improve_iters.is_none()
    }

    // Builder-style setters

    #[inline]
    pub fn with_max_duration(mut self, d: Option<Duration>) -> Self {
        self.max_duration = d;
        self
    }
    #[inline]
    pub fn with_max_neighbors(mut self, n: Option<u64>) -> Self {
        self.max_neighbors = n;
        self
    }
    #[inline]
    pub fn with_max_iterations(mut self, n: Option<u64>) -> Self {
        self.max_iterations = n;
        self
    }
    #[inline]
    pub fn with_no_improve_time(mut self, d: Option<Duration>) -> Self {
        self.no_improve_time = d;
        self
    }
    #[inline]
    pub fn with_no_improve_iters(mut self, n: Option<u64>) -> Self {
        self.no_improve_iters = n;
        self
    }
    #[inline]
    pub fn with_neighbor_chunk(mut self, c: u64) -> Self {
        self.neighbor_chunk = c.max(1);
        self
    }
    #[inline]
    pub fn with_iteration_chunk(mut self, c: u64) -> Self {
        self.iteration_chunk = c.max(1);
        self
    }
    #[inline]
    pub fn with_sample_every(mut self, k: u32) -> Self {
        self.sample_every = k.max(1);
        self
    }
}

impl fmt::Display for SearchLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Render only what matters; useful in logs.
        write!(
            f,
            "SearchLimits{{ max_duration: {:?}, max_neighbors: {:?}, max_iterations: {:?}, \
             no_improve_time: {:?}, no_improve_iters: {:?}, neighbor_chunk: {}, \
             iteration_chunk: {}, sample_every: {} }}",
            self.max_duration,
            self.max_neighbors,
            self.max_iterations,
            self.no_improve_time,
            self.no_improve_iters,
            self.neighbor_chunk,
            self.iteration_chunk,
            self.sample_every
        )
    }
}

/// Global controller coordinating termination across threads.
/// Creates chunk dispensers and exposes event hooks for accepted/improved solutions.
#[derive(Debug)]
pub struct GlobalController {
    pub stop: StopToken,
    pub limits: SearchLimits,
    pub start: Instant,
    pub improve: Arc<ImprovementState>,
    neigh_disp: Option<Arc<ChunkDispenser>>,
    iter_disp: Option<Arc<ChunkDispenser>>,
}

impl GlobalController {
    /// Construct a new controller. Prefer wrapping in `Arc` for multi-threaded usage.
    pub fn new(limits: SearchLimits) -> Arc<Self> {
        let stop = StopToken::new();
        let improve = Arc::new(ImprovementState::new());
        let neigh_disp = limits
            .max_neighbors
            .map(|m| ChunkDispenser::new(m, limits.neighbor_chunk.max(1), stop.clone()));
        let iter_disp = limits
            .max_iterations
            .map(|m| ChunkDispenser::new(m, limits.iteration_chunk.max(1), stop.clone()));
        Arc::new(Self {
            stop,
            limits,
            start: Instant::now(),
            improve,
            neigh_disp,
            iter_disp,
        })
    }

    /// Returns a thread-local lease for the neighbor budget, when configured.
    #[inline]
    pub fn make_neighbor_lease(&self) -> Option<Lease> {
        self.neigh_disp.as_ref().map(|d| Lease::new(d.clone()))
    }

    /// Returns a thread-local lease for the iteration budget, when configured.
    #[inline]
    pub fn make_iteration_lease(&self) -> Option<Lease> {
        self.iter_disp.as_ref().map(|d| Lease::new(d.clone()))
    }

    /// Event: a plan was accepted (not necessarily improving the incumbent).
    #[inline]
    pub fn on_accepted(&self) {
        self.improve.on_accepted();
    }

    /// Event: the incumbent improved.
    #[inline]
    pub fn on_incumbent_improvement(&self) {
        self.improve.on_incumbent_improved();
    }

    /// Returns whether the absolute time budget (if any) has been exceeded.
    #[inline]
    pub fn time_exceeded(&self) -> bool {
        self.limits
            .max_duration
            .is_some_and(|d| self.start.elapsed() >= d)
    }

    /// Returns elapsed time since this controller was created.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Accessors for structured logging or inspection.
    #[inline]
    pub fn limits(&self) -> &SearchLimits {
        &self.limits
    }
    #[inline]
    pub fn stop_token(&self) -> &StopToken {
        &self.stop
    }
    #[inline]
    pub fn improvement(&self) -> &Arc<ImprovementState> {
        &self.improve
    }
}

impl fmt::Display for GlobalController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GlobalController{{ elapsed: {:?}, limits: {} }}",
            self.elapsed(),
            self.limits
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_limits_default_and_display() {
        let d = SearchLimits::default();
        assert_eq!(d.max_duration, None);
        assert!(d.is_unbounded());
        let s = d.to_string();
        assert!(s.contains("SearchLimits"));
        assert!(s.contains("neighbor_chunk"));
        assert!(s.contains("iteration_chunk"));
    }

    #[test]
    fn search_limits_builder_roundtrip_eq() {
        let a = SearchLimits::default()
            .with_max_duration(Some(Duration::from_millis(10)))
            .with_max_neighbors(Some(123))
            .with_max_iterations(Some(456))
            .with_no_improve_time(Some(Duration::from_secs(2)))
            .with_no_improve_iters(Some(3))
            .with_neighbor_chunk(7)
            .with_iteration_chunk(11)
            .with_sample_every(5);

        let b = SearchLimits {
            max_duration: Some(Duration::from_millis(10)),
            max_neighbors: Some(123),
            max_iterations: Some(456),
            no_improve_time: Some(Duration::from_secs(2)),
            no_improve_iters: Some(3),
            neighbor_chunk: 7,
            iteration_chunk: 11,
            sample_every: 5,
        };

        assert_eq!(a, b);
        assert!(!a.is_unbounded());
        assert!(a.to_string().contains("max_neighbors: Some(123)"));
    }

    #[test]
    fn controller_leases_and_stop_signal() {
        // Configure small neighbor budget split into chunks.
        let limits = SearchLimits::default()
            .with_max_neighbors(Some(5))
            .with_neighbor_chunk(3); // 3 + 2 on second lease

        let ctrl = GlobalController::new(limits);
        // Neighbor lease present; iteration lease absent.
        let mut lease = ctrl
            .make_neighbor_lease()
            .expect("neighbor lease should be present");
        assert!(ctrl.make_iteration_lease().is_none());

        // Consume exactly 5 tokens; next call should fail.
        let mut taken = 0u64;
        while lease.take_one() {
            taken += 1;
        }
        assert_eq!(taken, 5, "should consume the exact neighbor budget");

        // Stop token should be set by the dispenser upon exhausting limit.
        assert!(ctrl.stop_token().is_set());
    }

    #[test]
    fn controller_time_exceeded_zero_duration_is_immediate() {
        let limits = SearchLimits::default().with_max_duration(Some(Duration::from_secs(0)));
        let ctrl = GlobalController::new(limits);
        assert!(
            ctrl.time_exceeded(),
            "zero budget should time out immediately"
        );
    }

    #[test]
    fn improvement_events_accounting() {
        let ctrl = GlobalController::new(SearchLimits::default());

        // Initially zero since last improvement.
        assert_eq!(
            ctrl.improvement().accepted_since_last_improve(),
            0,
            "initial accepted since last improve"
        );

        // Two accepted events increment global accepted counter.
        ctrl.on_accepted();
        ctrl.on_accepted();
        assert_eq!(
            ctrl.improvement().accepted_since_last_improve(),
            2,
            "accepted counter should reflect two accepts"
        );

        // Mark an incumbent improvement; resets the baseline.
        ctrl.on_incumbent_improvement();
        assert_eq!(
            ctrl.improvement().accepted_since_last_improve(),
            0,
            "counter resets after improvement"
        );
    }

    #[test]
    fn controller_display_contains_limits() {
        let ctrl = GlobalController::new(
            SearchLimits::default()
                .with_max_neighbors(Some(42))
                .with_iteration_chunk(99),
        );
        let s = ctrl.to_string();
        assert!(s.contains("GlobalController"));
        assert!(s.contains("limits:"));
        assert!(s.contains("max_neighbors: Some(42)"));
        assert!(s.contains("iteration_chunk: 99"));
    }
}
