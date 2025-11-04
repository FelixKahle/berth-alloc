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
    monitor::search_monitor::{
        LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
    },
    state::plan::Plan,
};

pub struct TimeLimitMonitor {
    start_time: std::time::Instant,
    time_limit: std::time::Duration,
}

impl TimeLimitMonitor {
    #[inline]
    pub fn new(time_limit: std::time::Duration) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            time_limit,
        }
    }
}

impl TerminationCheck for TimeLimitMonitor {
    #[inline]
    fn should_terminate_search(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }
}

impl<T> PlanEventMonitor<T> for TimeLimitMonitor
where
    T: Copy + Ord,
{
    #[inline]
    fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, T>) {}

    #[inline]
    fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, T>) {}

    #[inline]
    fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, T>) {}
}

impl LifecycleMonitor for TimeLimitMonitor {
    #[inline]
    fn on_search_start(&mut self) {
        self.start_time = std::time::Instant::now();
    }

    #[inline]
    fn on_search_end(&mut self) {}
}

impl<T> SearchMonitor<T> for TimeLimitMonitor
where
    T: Copy + Ord,
{
    #[inline]
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::search_monitor::{
        LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
    };
    use crate::state::plan::Plan;
    use std::thread;
    use std::time::Duration;

    // Helper: small sleep to reduce flakiness while keeping tests fast.
    fn short_sleep_ms(ms: u64) {
        thread::sleep(Duration::from_millis(ms));
    }

    #[test]
    fn test_time_limit_monitor_initial_state_not_terminated() {
        // Large enough limit so this cannot spuriously fail on slow CI.
        let m = TimeLimitMonitor::new(Duration::from_millis(250));
        assert!(
            !m.should_terminate_search(),
            "monitor should not terminate immediately after construction"
        );
    }

    #[test]
    fn test_time_limit_monitor_terminates_after_elapsed() {
        // Small limit; wait longer than the limit.
        let m = TimeLimitMonitor::new(Duration::from_millis(10));
        short_sleep_ms(25);
        assert!(
            m.should_terminate_search(),
            "monitor should terminate after time limit elapses"
        );
    }

    #[test]
    fn test_time_limit_monitor_on_search_start_resets_timer() {
        let mut m = TimeLimitMonitor::new(Duration::from_millis(5));
        short_sleep_ms(15);
        assert!(
            m.should_terminate_search(),
            "should be terminated after initial limit elapsed"
        );

        // Reset the timer via lifecycle start; should become non-terminated immediately after.
        m.on_search_start();
        assert!(
            !m.should_terminate_search(),
            "on_search_start should reset the timer and clear termination state"
        );

        // After waiting beyond the limit again, it should terminate once more.
        short_sleep_ms(15);
        assert!(
            m.should_terminate_search(),
            "should terminate again after the limit elapses post-reset"
        );
    }

    #[test]
    fn test_time_limit_monitor_plan_events_are_noops() {
        let mut m = TimeLimitMonitor::new(Duration::from_secs(1)); // plenty of time to avoid accidental timeouts
        let plan: Plan<'static, i64> = Plan::empty();

        m.on_plan_generated(&plan);
        m.on_plan_rejected(&plan);
        m.on_plan_accepted(&plan);

        assert!(
            !m.should_terminate_search(),
            "plan events must not affect termination state"
        );
    }

    #[test]
    fn test_time_limit_monitor_zero_duration_immediate_terminate() {
        let m = TimeLimitMonitor::new(Duration::from_millis(0));
        assert!(
            m.should_terminate_search(),
            "zero-duration monitor should be immediately terminated"
        );
    }

    #[test]
    fn test_time_limit_monitor_trait_object_usage() {
        let mut m = TimeLimitMonitor::new(Duration::from_millis(50));

        // Use via SearchMonitor trait object to ensure trait-object behavior works end-to-end.
        fn drive(mon: &mut dyn SearchMonitor<i64>) -> bool {
            mon.on_search_start();
            mon.should_terminate_search()
        }

        let terminated_immediately = drive(&mut m);
        assert!(
            !terminated_immediately,
            "just after on_search_start, the monitor should not be terminated"
        );

        // Wait and re-check via TerminationCheck
        short_sleep_ms(60);
        assert!(
            m.should_terminate_search(),
            "monitor should become terminated after elapsed limit when used via trait object"
        );

        // End should be a no-op and not panic
        m.on_search_end();
    }

    #[test]
    fn test_time_limit_monitor_is_send_and_usable_in_thread() {
        let handle = thread::spawn(|| {
            let mut m = TimeLimitMonitor::new(Duration::from_millis(5));
            m.on_search_start();
            short_sleep_ms(15);
            assert!(
                m.should_terminate_search(),
                "monitor should terminate inside spawned thread after limit"
            );
        });

        handle.join().expect("thread should complete successfully");
    }

    #[test]
    fn test_time_limit_monitor_multiple_starts_reset_timer_each_time() {
        let mut m = TimeLimitMonitor::new(Duration::from_millis(5));

        m.on_search_start();
        short_sleep_ms(10);
        assert!(
            m.should_terminate_search(),
            "should terminate after first interval"
        );

        m.on_search_start();
        assert!(
            !m.should_terminate_search(),
            "restart should clear previous termination"
        );
        short_sleep_ms(10);
        assert!(
            m.should_terminate_search(),
            "should terminate again after second interval"
        );
    }
}
