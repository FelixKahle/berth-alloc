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

#[derive(Debug)]
pub struct PlanLimitMonitor {
    plan_limit: u64,
    plans_generated: u64,
}

impl PlanLimitMonitor {
    #[inline]
    pub fn new(plan_limit: u64) -> Self {
        Self {
            plan_limit,
            plans_generated: 0,
        }
    }

    #[inline]
    pub fn plan_limit(&self) -> u64 {
        self.plan_limit
    }

    #[inline]
    pub fn plans_generated(&self) -> u64 {
        self.plans_generated
    }
}

impl TerminationCheck for PlanLimitMonitor {
    #[inline]
    fn should_terminate_search(&self) -> bool {
        self.plans_generated >= self.plan_limit
    }
}

impl<T> PlanEventMonitor<T> for PlanLimitMonitor
where
    T: Copy + Ord,
{
    #[inline]
    fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, T>) {
        // Count every generated candidate
        self.plans_generated = self.plans_generated.saturating_add(1);
    }

    #[inline]
    fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, T>) {
        // no-op
    }

    #[inline]
    fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, T>) {
        // no-op; acceptance does not change the generation count
    }
}

impl LifecycleMonitor for PlanLimitMonitor {
    #[inline]
    fn on_search_start(&mut self) {
        // Reset counter at the start of each search run
        self.plans_generated = 0;
    }

    #[inline]
    fn on_search_end(&mut self) {
        // no-op
    }
}

impl<T> SearchMonitor<T> for PlanLimitMonitor
where
    T: Copy + Ord,
{
    #[inline]
    fn name(&self) -> &str {
        "PlanLimitMonitor"
    }
}

/// Creates a boxed PlanLimitMonitor with a limit of 1 plan.
#[inline(always)]
pub fn make_single_plan_limit_monitor<T>() -> Box<dyn SearchMonitor<T>>
where
    T: Copy + Ord,
{
    Box::new(PlanLimitMonitor::new(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::search_monitor::{
        LifecycleMonitor, PlanEventMonitor, SearchMonitor, TerminationCheck,
    };
    use crate::state::plan::Plan;

    #[inline]
    fn make_plan() -> Plan<'static, i64> {
        Plan::empty()
    }

    #[test]
    fn test_plan_limit_monitor_initial_state_not_terminated_when_limit_positive() {
        let m = PlanLimitMonitor::new(10);
        assert!(
            !m.should_terminate_search(),
            "not terminated at construction when limit > 0"
        );
        assert_eq!(m.plan_limit(), 10);
        assert_eq!(m.plans_generated(), 0);
    }

    #[test]
    fn test_plan_limit_monitor_zero_limit_immediate_terminate() {
        let m = PlanLimitMonitor::new(0);
        assert!(
            m.should_terminate_search(),
            "should be immediately terminated when plan_limit == 0"
        );
        assert_eq!(m.plan_limit(), 0);
        assert_eq!(m.plans_generated(), 0);
    }

    #[test]
    fn test_plan_limit_monitor_counts_generated_candidates_and_terminates_at_limit() {
        let mut m = PlanLimitMonitor::new(3);
        let plan = make_plan();

        assert!(!m.should_terminate_search());
        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 1);
        assert!(!m.should_terminate_search());

        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 2);
        assert!(!m.should_terminate_search());

        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 3);
        assert!(
            m.should_terminate_search(),
            "should terminate once generated count reaches limit"
        );
    }

    #[test]
    fn test_plan_limit_monitor_reject_and_accept_do_not_change_generated_count() {
        let mut m = PlanLimitMonitor::new(5);
        let plan = make_plan();

        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 1);

        m.on_plan_rejected(&plan);
        assert_eq!(m.plans_generated(), 1, "reject should not change count");

        m.on_plan_accepted(&plan);
        assert_eq!(m.plans_generated(), 1, "accepted should not change count");

        assert!(!m.should_terminate_search());
    }

    #[test]
    fn test_plan_limit_monitor_lifecycle_start_resets_counter() {
        let mut m = PlanLimitMonitor::new(2);
        let plan = make_plan();

        // First run: generate up to limit
        m.on_search_start();
        m.on_plan_generated(&plan);
        m.on_plan_generated(&plan);
        assert!(m.should_terminate_search());
        assert_eq!(m.plans_generated(), 2);

        // Second run: reset and ensure we start fresh
        m.on_search_start();
        assert_eq!(
            m.plans_generated(),
            0,
            "on_search_start should reset the counter"
        );
        assert!(!m.should_terminate_search());

        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 1);
        assert!(!m.should_terminate_search());
    }

    #[test]
    fn test_plan_limit_monitor_trait_object_usage_for_plan_events_and_lifecycle() {
        let mut m = PlanLimitMonitor::new(2);
        let plan = make_plan();

        // Use via SearchMonitor (lifecycle)
        {
            let mon: &mut dyn SearchMonitor<i64> = &mut m;
            mon.on_search_start();
        }

        // Use via PlanEventMonitor (plan events)
        {
            let mon: &mut dyn PlanEventMonitor<i64> = &mut m;
            mon.on_plan_generated(&plan);
            mon.on_plan_generated(&plan);
        }

        // Check termination via TerminationCheck
        {
            let tc: &dyn TerminationCheck = &m;
            assert!(
                tc.should_terminate_search(),
                "should terminate after 2 generated plans"
            );
        }

        // End-of-search is a no-op
        {
            let mon: &mut dyn SearchMonitor<i64> = &mut m;
            mon.on_search_end();
        }
    }

    #[test]
    fn test_plan_limit_monitor_name_through_trait_object() {
        let mut m = PlanLimitMonitor::new(1);
        let mon: &mut dyn SearchMonitor<i64> = &mut m;
        assert_eq!(mon.name(), "PlanLimitMonitor");
    }

    #[test]
    fn test_plan_limit_monitor_accessors_report_values() {
        let mut m = PlanLimitMonitor::new(4);
        let plan = make_plan();

        assert_eq!(m.plan_limit(), 4);
        assert_eq!(m.plans_generated(), 0);

        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 1);

        m.on_plan_generated(&plan);
        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 3);
        assert!(!m.should_terminate_search());

        m.on_plan_generated(&plan);
        assert_eq!(m.plans_generated(), 4);
        assert!(m.should_terminate_search());
    }
}
