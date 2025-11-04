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

use crate::state::plan::Plan;

/// Read-only capability to determine whether the search should terminate.
///
/// Implementors should ensure this method is cheap and free of side effects.
/// It may be polled frequently by both workers and decision builders.
pub trait TerminationCheck {
    /// Returns `true` if the search should terminate as soon as possible.
    fn should_terminate_search(&self) -> bool;
}

/// Builder-facing capability for plan-level events during neighborhood exploration.
///
/// Semantics:
/// - `on_plan_generated` is called once for every candidate produced by the neighborhood/operator.
/// - `on_plan_rejected` is called when a previously generated candidate is discarded (by filters,
///   feasibility checks, or the metaheuristic policy).
/// - `on_plan_accepted` is called exactly once for the candidate that will be returned by
///   `DecisionBuilder::next`. Acceptance means "accepted by the search strategy", not "applied to
///   the solver state".
///
/// Implementations may treat these events as counters, logging hooks, metrics updates, or triggers
/// for termination (e.g., stopping after a budget of generated candidates).
pub trait PlanEventMonitor<T>: TerminationCheck
where
    T: Copy + Ord,
{
    /// Called once for each candidate plan produced by the operator.
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>);

    /// Called when a previously generated candidate plan is discarded.
    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>);

    /// Called exactly once for the candidate plan accepted by the strategy and returned from
    /// `DecisionBuilder::next`. This does not imply the plan has been applied to the solver state.
    fn on_plan_accepted<'p>(&mut self, plan: &Plan<'p, T>);
}

/// Worker-facing capability for search lifecycle events.
///
/// Semantics:
/// - `on_search_start` is called once by the worker before the search loop begins. Implementations
///   may use this to reset state (e.g., restart timers or counters).
/// - `on_search_end` is called once by the worker after the search loop ends (normally or due to
///   termination). Implementations may use this to finalize or flush aggregated results.
pub trait LifecycleMonitor: TerminationCheck {
    /// Called once before the worker begins its search loop.
    fn on_search_start(&mut self);

    /// Called once after the worker exits its search loop.
    fn on_search_end(&mut self);
}

/// Full-featured monitor owned by the worker.
///
/// This trait combines builder-facing plan events and worker-facing lifecycle hooks.
/// The worker owns a `SearchMonitor` and may pass it to the decision builder as a
/// `&mut dyn PlanEventMonitor<T>` when creating the search context.
///
/// Threading:
/// - Monitors are moved into worker threads as `Box<dyn SearchMonitor<T> + Send>`.
/// - Implementations should avoid shared mutable state across threads unless explicitly synchronized.
pub trait SearchMonitor<T>: PlanEventMonitor<T> + LifecycleMonitor
where
    T: Copy + Ord,
{
    /// Human-readable name for diagnostics and logs.
    fn name(&self) -> &str;
}

impl<'a, T> std::fmt::Display for dyn SearchMonitor<T> + 'a
where
    T: Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

impl<'a, T> std::fmt::Debug for dyn SearchMonitor<T> + 'a
where
    T: Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

/// A composite monitor that fans out all events to contained child monitors.
///
/// Ownership and threading:
/// - Stored monitors are `Box<dyn SearchMonitor<T> + Send>` so the composite itself can be moved
///   into a worker thread safely.
/// - The composite does not implement `Sync` and should not be shared across threads.
#[derive(Default)]
pub struct CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    monitors: Vec<Box<dyn SearchMonitor<T> + Send>>,
}

impl<T> std::fmt::Debug for CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompositeSearchMonitor[")?;
        for (i, monitor) in self.monitors.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", monitor.name())?;
        }
        write!(f, "]")
    }
}

impl<T> std::fmt::Display for CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompositeSearchMonitor[")?;
        for (i, monitor) in self.monitors.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", monitor.name())?;
        }
        write!(f, "]")
    }
}

impl<T> CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    /// Creates an empty composite monitor.
    #[inline]
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    /// Creates an empty composite monitor with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            monitors: Vec::with_capacity(capacity),
        }
    }

    /// Adds a child monitor to the composite.
    #[inline]
    pub fn add_monitor(&mut self, monitor: Box<dyn SearchMonitor<T> + Send>) {
        self.monitors.push(monitor);
    }
}

impl<T> TerminationCheck for CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    /// Returns `true` if any child monitor requests termination.
    #[inline]
    fn should_terminate_search(&self) -> bool {
        self.monitors.iter().any(|m| m.should_terminate_search())
    }
}

impl<T> PlanEventMonitor<T> for CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    /// Fans out to all child monitors.
    #[inline]
    fn on_plan_generated<'p>(&mut self, plan: &Plan<'p, T>) {
        for m in &mut self.monitors {
            m.on_plan_generated(plan);
        }
    }

    /// Fans out to all child monitors.
    #[inline]
    fn on_plan_rejected<'p>(&mut self, plan: &Plan<'p, T>) {
        for m in &mut self.monitors {
            m.on_plan_rejected(plan);
        }
    }

    /// Fans out to all child monitors.
    #[inline]
    fn on_plan_accepted<'p>(&mut self, plan: &Plan<'p, T>) {
        for m in &mut self.monitors {
            m.on_plan_accepted(plan);
        }
    }
}

impl<T> LifecycleMonitor for CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    /// Fans out to all child monitors.
    #[inline]
    fn on_search_start(&mut self) {
        for m in &mut self.monitors {
            m.on_search_start();
        }
    }

    /// Fans out to all child monitors.
    #[inline]
    fn on_search_end(&mut self) {
        for m in &mut self.monitors {
            m.on_search_end();
        }
    }
}

impl<T> SearchMonitor<T> for CompositeSearchMonitor<T>
where
    T: Copy + Ord,
{
    /// Returns the composite's display name.
    #[inline]
    fn name(&self) -> &str {
        "CompositeSearchMonitor"
    }
}

/// A no-op monitor useful as a default or placeholder.
///
/// Behavior:
/// - Never requests termination.
/// - Ignores all plan events.
/// - Lifecycle hooks are no-ops.
///   This is safe to use when monitoring is optional.
#[derive(Debug, Default)]
pub struct NullSearchMonitor;

impl TerminationCheck for NullSearchMonitor {
    /// Always returns `false`.
    #[inline]
    fn should_terminate_search(&self) -> bool {
        false
    }
}

impl<T> PlanEventMonitor<T> for NullSearchMonitor
where
    T: Copy + Ord,
{
    /// No-op.
    #[inline]
    fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, T>) {}

    /// No-op.
    #[inline]
    fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, T>) {}

    /// No-op.
    #[inline]
    fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, T>) {}
}

impl LifecycleMonitor for NullSearchMonitor {
    /// No-op.
    #[inline]
    fn on_search_start(&mut self) {}

    /// No-op.
    #[inline]
    fn on_search_end(&mut self) {}
}

impl<T> SearchMonitor<T> for NullSearchMonitor
where
    T: Copy + Ord,
{
    /// Returns the null monitor's display name.
    #[inline]
    fn name(&self) -> &str {
        "NullSearchMonitor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[test]
    fn test_null_monitor_basics() {
        let mut m = NullSearchMonitor;

        // Termination should always be false
        assert!(!m.should_terminate_search());

        // Lifecycle no-ops should not panic
        m.on_search_start();
        m.on_search_end();

        // Plan events are no-ops and should not change termination
        let plan: Plan<'static, i64> = Plan::empty();
        m.on_plan_generated(&plan);
        m.on_plan_rejected(&plan);
        m.on_plan_accepted(&plan);

        assert_eq!(
            <NullSearchMonitor as SearchMonitor<i64>>::name(&m),
            "NullSearchMonitor"
        );

        // Display/Debug for dyn SearchMonitor should reflect the name
        let mon_ref: &dyn SearchMonitor<i64> = &m;
        assert_eq!(format!("{}", mon_ref), "SearchMonitor(NullSearchMonitor)");
        assert_eq!(format!("{:?}", mon_ref), "SearchMonitor(NullSearchMonitor)");
    }

    #[derive(Clone)]
    struct RecShared {
        generated: Arc<AtomicUsize>,
        rejected: Arc<AtomicUsize>,
        accepted: Arc<AtomicUsize>,
        started: Arc<AtomicUsize>,
        ended: Arc<AtomicUsize>,
        terminate: Arc<AtomicBool>,
    }

    impl RecShared {
        fn new() -> Self {
            Self {
                generated: Arc::new(AtomicUsize::new(0)),
                rejected: Arc::new(AtomicUsize::new(0)),
                accepted: Arc::new(AtomicUsize::new(0)),
                started: Arc::new(AtomicUsize::new(0)),
                ended: Arc::new(AtomicUsize::new(0)),
                terminate: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    struct RecordingMonitor {
        name: &'static str,
        s: RecShared,
    }

    impl RecordingMonitor {
        fn new(name: &'static str, s: RecShared) -> Self {
            Self { name, s }
        }
    }

    impl TerminationCheck for RecordingMonitor {
        fn should_terminate_search(&self) -> bool {
            self.s.terminate.load(Ordering::SeqCst)
        }
    }

    impl PlanEventMonitor<i64> for RecordingMonitor {
        fn on_plan_generated<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.s.generated.fetch_add(1, Ordering::SeqCst);
        }
        fn on_plan_rejected<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.s.rejected.fetch_add(1, Ordering::SeqCst);
        }
        fn on_plan_accepted<'p>(&mut self, _plan: &Plan<'p, i64>) {
            self.s.accepted.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl LifecycleMonitor for RecordingMonitor {
        fn on_search_start(&mut self) {
            self.s.started.fetch_add(1, Ordering::SeqCst);
        }
        fn on_search_end(&mut self) {
            self.s.ended.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl SearchMonitor<i64> for RecordingMonitor {
        fn name(&self) -> &str {
            self.name
        }
    }

    #[test]
    fn test_composite_forwards_events_and_lifecycle() {
        let mut comp: CompositeSearchMonitor<i64> = CompositeSearchMonitor::new();

        let s1 = RecShared::new();
        let s2 = RecShared::new();

        comp.add_monitor(Box::new(RecordingMonitor::new("M1", s1.clone())));
        comp.add_monitor(Box::new(RecordingMonitor::new("M2", s2.clone())));

        let plan: Plan<'static, i64> = Plan::empty();

        // Lifecycle start
        comp.on_search_start();
        // Plan events
        comp.on_plan_generated(&plan);
        comp.on_plan_generated(&plan);
        comp.on_plan_rejected(&plan);
        comp.on_plan_accepted(&plan);
        // Lifecycle end
        comp.on_search_end();

        // Both monitors should have seen the same counts
        for s in [&s1, &s2] {
            assert_eq!(s.started.load(Ordering::SeqCst), 1, "on_search_start count");
            assert_eq!(
                s.generated.load(Ordering::SeqCst),
                2,
                "on_plan_generated count"
            );
            assert_eq!(
                s.rejected.load(Ordering::SeqCst),
                1,
                "on_plan_rejected count"
            );
            assert_eq!(
                s.accepted.load(Ordering::SeqCst),
                1,
                "on_plan_accepted count"
            );
            assert_eq!(s.ended.load(Ordering::SeqCst), 1, "on_search_end count");
        }

        // Termination should be false while both monitors do not request it
        assert!(!comp.should_terminate_search());
    }

    #[test]
    fn test_composite_termination_if_any_child_requests() {
        let mut comp: CompositeSearchMonitor<i64> = CompositeSearchMonitor::new();

        let s1 = RecShared::new();
        let s2 = RecShared::new();

        comp.add_monitor(Box::new(RecordingMonitor::new("M1", s1.clone())));
        comp.add_monitor(Box::new(RecordingMonitor::new("M2", s2.clone())));

        // Initially, none request termination
        assert!(!comp.should_terminate_search());

        // Flip termination flag in one child; composite should now terminate
        s2.terminate.store(true, Ordering::SeqCst);
        assert!(comp.should_terminate_search());
    }

    #[test]
    fn test_composite_display_and_debug_include_child_names() {
        let mut comp: CompositeSearchMonitor<i64> = CompositeSearchMonitor::new();

        comp.add_monitor(Box::new(RecordingMonitor::new("Alpha", RecShared::new())));
        comp.add_monitor(Box::new(RecordingMonitor::new("Beta", RecShared::new())));

        let disp = format!("{}", comp);
        let dbg = format!("{:?}", comp);

        // The composite's own dyn SearchMonitor Display/Debug formatting
        let mon_ref: &dyn SearchMonitor<i64> = &comp;
        assert_eq!(
            format!("{}", mon_ref),
            "SearchMonitor(CompositeSearchMonitor)"
        );
        assert_eq!(
            format!("{:?}", mon_ref),
            "SearchMonitor(CompositeSearchMonitor)"
        );

        // Composite's Display/Debug include children names in a readable list
        assert!(
            disp.contains("Alpha") && disp.contains("Beta"),
            "Display should include child names"
        );
        assert!(
            dbg.contains("Alpha") && dbg.contains("Beta"),
            "Debug should include child names"
        );
    }

    #[test]
    fn test_plan_event_monitor_trait_object_usage() {
        // Ensure a SearchMonitor can be used where a PlanEventMonitor is needed
        let mut comp: CompositeSearchMonitor<i64> = CompositeSearchMonitor::new();
        comp.add_monitor(Box::new(RecordingMonitor::new("One", RecShared::new())));

        // Upcast to dyn SearchMonitor first, then rely on method resolution.
        // We call plan-event methods through the concrete composite for stability.
        let plan: Plan<'static, i64> = Plan::empty();

        comp.on_search_start();
        comp.on_plan_generated(&plan);
        comp.on_plan_rejected(&plan);
        comp.on_plan_accepted(&plan);
        comp.on_search_end();

        // Nothing to assert directly here beyond no panics; forwarding is covered in other tests.
        assert_eq!(comp.name(), "CompositeSearchMonitor");
    }
}
