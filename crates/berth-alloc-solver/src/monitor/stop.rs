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

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// Cheap cooperative stop token shared across threads.
#[derive(Clone, Default, Debug)]
pub struct StopToken(Arc<AtomicBool>);

impl StopToken {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn is_set(&self) -> bool {
        self.0.load(Relaxed)
    }
    #[inline]
    pub fn request_stop(&self) {
        self.0.store(true, Relaxed)
    }
}

impl PartialEq for StopToken {
    /// Equality is based on identity (same underlying Arc), not current value.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for StopToken {}

impl fmt::Display for StopToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.is_set();
        write!(f, "StopToken(set: {})", set)
    }
}

impl AsRef<AtomicBool> for StopToken {
    #[inline]
    fn as_ref(&self) -> &AtomicBool {
        &self.0
    }
}

/// Global improvement state (incumbent-based).
/// Timestamps are monotonic nanoseconds since `start`.
#[derive(Debug)]
pub struct ImprovementState {
    start: Instant,
    accepted_global: AtomicU64,          // total accepted plans
    last_improve_ns: AtomicU64,          // ns since start of last incumbent improvement
    last_improve_accepted_at: AtomicU64, // accepted count at last improvement
}

impl Default for ImprovementState {
    fn default() -> Self {
        Self::new()
    }
}

impl ImprovementState {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            accepted_global: AtomicU64::new(0),
            last_improve_ns: AtomicU64::new(0),
            last_improve_accepted_at: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn now_ns(&self) -> u64 {
        // Safe for centuries at nanosecond resolution in u64.
        self.start.elapsed().as_nanos() as u64
    }

    /// Returns the new total after incrementing accepted count.
    #[inline]
    pub fn on_accepted(&self) -> u64 {
        self.accepted_global.fetch_add(1, Relaxed) + 1
    }

    /// Update last-improvement timestamp and baseline accepted count.
    #[inline]
    pub fn on_incumbent_improved(&self) {
        let now = self.now_ns();
        let acc = self.accepted_global.load(Relaxed);
        self.last_improve_ns.store(now, Relaxed);
        self.last_improve_accepted_at.store(acc, Relaxed);
    }

    /// Nanoseconds elapsed since the last incumbent improvement was recorded.
    #[inline]
    pub fn since_last_improve_ns(&self) -> u64 {
        let now = self.now_ns();
        let last = self.last_improve_ns.load(Relaxed);
        now.saturating_sub(last)
    }

    /// Number of accepted plans since the last incumbent improvement.
    #[inline]
    pub fn accepted_since_last_improve(&self) -> u64 {
        let acc = self.accepted_global.load(Relaxed);
        let at = self.last_improve_accepted_at.load(Relaxed);
        acc.saturating_sub(at)
    }

    /// Total number of accepted plans so far.
    #[inline]
    pub fn accepted_total(&self) -> u64 {
        self.accepted_global.load(Relaxed)
    }
}

impl fmt::Display for ImprovementState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Snapshots: may be slightly racy, but fine for diagnostics.
        write!(
            f,
            "ImprovementState{{ accepted_total: {}, since_last_improve_ns: {} }}",
            self.accepted_total(),
            self.since_last_improve_ns()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_token_identity_and_behavior() {
        let a = StopToken::new();
        let b = a.clone();
        let c = StopToken::new();

        // Identity equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Initially unset
        assert!(!a.is_set());
        assert!(!b.is_set());

        // Request stop via one handle reflects in the clone
        b.request_stop();
        assert!(a.is_set());
        assert!(b.is_set());
        assert!(!c.is_set());

        // Display formatting
        let s = a.to_string();
        assert!(s.contains("StopToken("));
        assert!(s.contains("set: true"));
    }

    #[test]
    fn improvement_state_accept_and_improve_flow() {
        let imp = ImprovementState::new();
        // Start with zero
        assert_eq!(imp.accepted_total(), 0);
        assert_eq!(imp.accepted_since_last_improve(), 0);
        let t0 = imp.now_ns();

        // Two accepts
        assert_eq!(imp.on_accepted(), 1);
        assert_eq!(imp.on_accepted(), 2);
        assert_eq!(imp.accepted_total(), 2);
        assert_eq!(imp.accepted_since_last_improve(), 2);

        // Mark improvement -> since_last resets
        imp.on_incumbent_improved();
        assert_eq!(imp.accepted_since_last_improve(), 0);

        // Further accept increments since_last
        imp.on_accepted();
        assert_eq!(imp.accepted_since_last_improve(), 1);

        // Monotonic-ish time checks (non-strict due to coarse timers on some platforms)
        let t1 = imp.now_ns();
        assert!(t1 >= t0);

        // Display contains fields
        let d = format!("{}", imp);
        assert!(d.contains("ImprovementState"));
        assert!(d.contains("accepted_total"));
    }
}
