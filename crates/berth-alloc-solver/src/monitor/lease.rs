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
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::monitor::stop::StopToken;

/// Global dispenser: carve the global limit into large chunks.
/// Threads lease chunks; hot path only decrements a local counter.
#[derive(Debug)]
pub struct ChunkDispenser {
    issued: AtomicU64,
    limit: u64,
    chunk: u64,
    stop: StopToken,
}

impl ChunkDispenser {
    /// Create a new chunk dispenser.
    /// - `limit`: total number of tokens available globally.
    /// - `chunk`: per-lease issuance size (will be clamped to at least 1).
    /// - `stop`: shared stop token that will be set when the limit is exhausted.
    pub fn new(limit: u64, chunk: u64, stop: StopToken) -> Arc<Self> {
        Arc::new(Self {
            issued: AtomicU64::new(0),
            limit,
            chunk: chunk.max(1),
            stop,
        })
    }

    /// Lease a new batch of tokens from the global budget.
    /// Returns (granted, exhausted_flag).
    /// When exhausted_flag is true (or granted == 0), no more tokens remain globally.
    #[inline]
    fn lease(&self) -> (u64, bool) {
        let start = self.issued.fetch_add(self.chunk, Relaxed);
        if start >= self.limit {
            self.stop.request_stop();
            return (0, true);
        }
        let end = (start + self.chunk).min(self.limit);
        let got = end - start;
        if end == self.limit {
            self.stop.request_stop();
        }
        (got, false)
    }

    /// Snapshot of currently issued tokens (monotonic up to `limit`), for diagnostics.
    #[inline]
    pub fn issued(&self) -> u64 {
        self.issued.load(Relaxed)
    }

    /// Total global limit configured at construction.
    #[inline]
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// Chunk size for leases.
    #[inline]
    pub fn chunk(&self) -> u64 {
        self.chunk
    }
}

impl fmt::Display for ChunkDispenser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Snapshot state; not guaranteed to be perfectly consistent due to concurrency,
        // but sufficient for logs/diagnostics.
        write!(
            f,
            "ChunkDispenser{{ issued: {}, limit: {}, chunk: {} }}",
            self.issued(),
            self.limit(),
            self.chunk()
        )
    }
}

/// Thread-local lease with zero atomics while tokens remain.
#[derive(Debug, Clone)]
pub struct Lease {
    left: u64,
    disp: Arc<ChunkDispenser>,
}

impl Lease {
    /// Create a new lease bound to the given dispenser.
    /// Initially empty; the first `take_one()` call will fetch a batch.
    pub fn new(disp: Arc<ChunkDispenser>) -> Self {
        Self { left: 0, disp }
    }

    /// Consume one token; returns false if global budget is exhausted.
    #[inline]
    pub fn take_one(&mut self) -> bool {
        if self.left == 0 {
            let (got, ex) = self.disp.lease();
            if ex || got == 0 {
                return false;
            }
            self.left = got;
        }
        self.left -= 1;
        true
    }

    /// Remaining locally-buffered tokens in this lease (does not reflect global pool).
    #[inline]
    pub fn remaining_local(&self) -> u64 {
        self.left
    }

    /// Returns true if no locally-buffered tokens remain.
    #[inline]
    pub fn is_empty_local(&self) -> bool {
        self.left == 0
    }

    /// Returns the underlying dispenser (Arc clone), useful for inspection in tests/logs.
    #[inline]
    pub fn dispenser(&self) -> Arc<ChunkDispenser> {
        self.disp.clone()
    }
}

impl PartialEq for Lease {
    /// Equality uses pointer identity for the dispenser and the local remaining count.
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && Arc::ptr_eq(&self.disp, &other.disp)
    }
}
impl Eq for Lease {}

impl fmt::Display for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = &self.disp;
        write!(
            f,
            "Lease{{ remaining_local: {}, dispenser: (issued: {}, limit: {}, chunk: {}) }}",
            self.left,
            d.issued(),
            d.limit(),
            d.chunk()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::stop::StopToken;

    #[test]
    fn chunk_dispenser_enforces_limit_and_sets_stop() {
        let stop = StopToken::new();
        let disp = ChunkDispenser::new(5, 3, stop.clone());
        let mut lease = Lease::new(disp.clone());

        let mut taken = 0u64;
        while lease.take_one() {
            taken += 1;
        }
        assert_eq!(taken, 5, "should consume exactly the global budget");
        assert!(stop.is_set(), "stop should be set after exhaustion");

        // Display/Debug should contain informative fields.
        let d = format!("{}", disp);
        assert!(d.contains("ChunkDispenser"));
        assert!(d.contains("limit: 5"));
        assert!(disp.issued() >= 5);
        let l = format!("{}", lease);
        assert!(l.contains("Lease"));
    }

    #[test]
    fn zero_limit_exhausts_immediately() {
        let stop = StopToken::new();
        let disp = ChunkDispenser::new(0, 10, stop.clone());
        let mut lease = Lease::new(disp);

        assert!(!lease.take_one(), "no tokens should be available");
        assert!(stop.is_set(), "stop should be set with zero limit");
    }

    #[test]
    fn chunk_size_is_clamped_to_at_least_one() {
        let stop = StopToken::new();
        let disp = ChunkDispenser::new(2, 0, stop);
        assert_eq!(disp.chunk(), 1, "chunk should be clamped to >= 1");
    }

    #[test]
    fn multiple_leases_share_budget_without_overconsumption() {
        let stop = StopToken::new();
        let disp = ChunkDispenser::new(7, 3, stop.clone());
        let mut a = Lease::new(disp.clone());
        let mut b = Lease::new(disp);

        let mut taken = 0u64;
        loop {
            let mut progressed = false;
            if a.take_one() {
                taken += 1;
                progressed = true;
            }
            if b.take_one() {
                taken += 1;
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        assert_eq!(taken, 7, "total taken should match the limit");
        assert!(stop.is_set(), "stop should be set after exhaustion");
    }

    #[test]
    fn lease_equality_and_local_remaining() {
        let stop = StopToken::new();
        let disp = ChunkDispenser::new(4, 2, stop);
        let mut a = Lease::new(disp.clone());
        let mut b = Lease::new(disp);

        // Both point to same dispenser and have the same local state: equal.
        assert_eq!(a, b);

        // Take one from A; not equal anymore (local remaining differs).
        assert!(a.take_one());
        assert_ne!(a, b);

        // B also takes one (will fetch its own chunk), but equality may still differ
        // depending on remaining_local values.
        assert!(b.take_one());
        // Not asserting equality here since they could have different local counts.
        assert!(a.remaining_local() <= 1 && b.remaining_local() <= 1);
    }
}
