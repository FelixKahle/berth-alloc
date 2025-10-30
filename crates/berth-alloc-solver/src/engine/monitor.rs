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

use std::time::{Duration, Instant};

use berth_alloc_core::prelude::Cost;

pub trait SearchMonitor: Send {
    fn name(&self) -> &str;

    fn on_search_start(&mut self) {}
    fn on_search_end(&mut self) {}

    fn on_iteration_start(&mut self, _iter: u64) {}
    fn on_iteration_end(&mut self, _iter: u64) {}

    fn on_begin_neighbor(&mut self, _iter: u64, _neighbor_ix: u64) {}
    fn on_end_neighbor(
        &mut self,
        _iter: u64,
        _neighbor_ix: u64,
        _accepted: bool,
        _delta_cost: Cost,
    ) {
    }

    fn on_accepted(&mut self, _iter: u64, _new_cost: Cost, _old_cost: Cost) {}
    fn on_new_incumbent(&mut self, _iter: u64, _cost: Cost) {}

    fn on_restart(&mut self, _iter: u64) {}

    fn should_terminate(&self) -> bool;
}

impl std::fmt::Debug for dyn SearchMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

impl std::fmt::Display for dyn SearchMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

#[derive(Default)]
pub struct CompositeSearchMonitor {
    monitors: Vec<Box<dyn SearchMonitor>>,
}

impl CompositeSearchMonitor {
    #[inline]
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    #[inline]
    pub fn with<M: SearchMonitor + 'static>(mut self, m: M) -> Self {
        self.monitors.push(Box::new(m));
        self
    }

    #[inline]
    pub fn push<M: SearchMonitor + 'static>(&mut self, m: M) {
        self.monitors.push(Box::new(m));
    }
}

impl SearchMonitor for CompositeSearchMonitor {
    #[inline]
    fn name(&self) -> &str {
        "CompositeSearchMonitor"
    }

    #[inline]
    fn on_search_start(&mut self) {
        for m in &mut self.monitors {
            m.on_search_start();
        }
    }

    #[inline]
    fn on_search_end(&mut self) {
        for m in &mut self.monitors {
            m.on_search_end();
        }
    }

    #[inline]
    fn on_iteration_start(&mut self, it: u64) {
        for m in &mut self.monitors {
            m.on_iteration_start(it);
        }
    }

    #[inline]
    fn on_iteration_end(&mut self, it: u64) {
        for m in &mut self.monitors {
            m.on_iteration_end(it);
        }
    }

    #[inline]
    fn on_begin_neighbor(&mut self, it: u64, nx: u64) {
        for m in &mut self.monitors {
            m.on_begin_neighbor(it, nx);
        }
    }

    #[inline]
    fn on_end_neighbor(&mut self, it: u64, nx: u64, acc: bool, dc: Cost) {
        for m in &mut self.monitors {
            m.on_end_neighbor(it, nx, acc, dc);
        }
    }

    #[inline]
    fn on_accepted(&mut self, it: u64, new_c: Cost, old_c: Cost) {
        for m in &mut self.monitors {
            m.on_accepted(it, new_c, old_c);
        }
    }

    #[inline]
    fn on_new_incumbent(&mut self, it: u64, c: Cost) {
        for m in &mut self.monitors {
            m.on_new_incumbent(it, c);
        }
    }

    #[inline]
    fn on_restart(&mut self, it: u64) {
        for m in &mut self.monitors {
            m.on_restart(it);
        }
    }

    #[inline]
    fn should_terminate(&self) -> bool {
        self.monitors.iter().any(|m| m.should_terminate())
    }
}

#[derive(Debug)]
pub struct TimeLimitMonitor {
    limit: Duration,
    started_at: Option<Instant>,
}

impl TimeLimitMonitor {
    #[inline]
    pub fn new(limit: Duration) -> Self {
        Self {
            limit,
            started_at: None,
        }
    }

    #[inline]
    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|t0| t0.elapsed())
    }

    #[inline]
    pub fn remaining(&self) -> Option<Duration> {
        self.elapsed().map(|e| self.limit.saturating_sub(e))
    }

    #[inline]
    fn timed_out(&self) -> bool {
        match self.started_at {
            None => false,
            Some(t0) => t0.elapsed() >= self.limit,
        }
    }
}

impl SearchMonitor for TimeLimitMonitor {
    #[inline]
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    #[inline]
    fn on_search_start(&mut self) {
        self.started_at = Some(Instant::now());
    }

    #[inline]
    fn should_terminate(&self) -> bool {
        self.timed_out()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_name() {
        let m = TimeLimitMonitor::new(Duration::from_secs(1));
        assert_eq!(m.name(), "TimeLimitMonitor");
    }

    #[test]
    fn test_zero_duration_times_out_immediately_after_start() {
        let mut m = TimeLimitMonitor::new(Duration::from_secs(0));
        assert_eq!(m.should_terminate(), false, "not started yet");
        m.on_search_start();
        assert!(
            m.should_terminate(),
            "zero budget should time out immediately"
        );
        assert_eq!(m.remaining(), Some(Duration::from_secs(0)));
    }

    #[test]
    fn test_positive_duration_not_timed_out_right_after_start() {
        let mut m = TimeLimitMonitor::new(Duration::from_millis(50));
        m.on_search_start();
        assert!(
            !m.should_terminate(),
            "immediately after start should have time left"
        );
        let rem = m.remaining().unwrap();
        assert!(rem <= Duration::from_millis(50) && rem >= Duration::from_millis(0));
    }
}
