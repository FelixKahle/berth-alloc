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
    model::solver_model::SolverModel,
    state::{plan::Plan, solver_state::SolverState},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterContext<'s, 'm, 'p, T>
where
    T: Copy + Ord,
{
    pub model: &'m SolverModel<'p, T>,
    pub solver_state: &'s SolverState<'p, T>,
}

impl<'s, 'm, 'p, T> FilterContext<'s, 'm, 'p, T>
where
    T: Copy + Ord,
{
    #[inline]
    pub fn new(model: &'m SolverModel<'p, T>, solver_state: &'s SolverState<'p, T>) -> Self {
        Self {
            model,
            solver_state,
        }
    }

    #[inline]
    pub fn model(&self) -> &'m SolverModel<'p, T> {
        self.model
    }

    #[inline]
    pub fn solver_state(&self) -> &'s SolverState<'p, T> {
        self.solver_state
    }
}

pub trait NeighborhoodFilter<T>: Send {
    fn name(&self) -> &str;

    fn accept<'s, 'm, 'p>(
        &mut self,
        context: FilterContext<'s, 'm, 'p, T>,
        plan: &Plan<'p, T>,
    ) -> bool
    where
        T: Copy + Ord;
}

impl<T> std::fmt::Display for dyn NeighborhoodFilter<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NeighborhoodFilter({})", self.name())
    }
}

impl<T> std::fmt::Debug for dyn NeighborhoodFilter<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NeighborhoodFilter({})", self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborhoodFilterStatistic {
    called: u64,
    accepted: u64,
    total_runtime_ns: u64,
}

impl std::fmt::Display for NeighborhoodFilterStatistic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NeighborhoodFilterStatistic{{ called: {}, accepted: {}, rejected: {}, total_runtime_ns: {} }}",
            self.called,
            self.accepted,
            self.rejected(),
            self.total_runtime_ns
        )
    }
}

impl Default for NeighborhoodFilterStatistic {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborhoodFilterStatistic {
    #[inline]
    pub fn new() -> Self {
        Self {
            called: 0,
            accepted: 0,
            total_runtime_ns: 0,
        }
    }

    #[inline]
    pub fn called(&self) -> u64 {
        self.called
    }

    #[inline]
    pub fn accepted(&self) -> u64 {
        self.accepted
    }

    #[inline]
    pub fn rejected(&self) -> u64 {
        self.called.saturating_sub(self.accepted)
    }

    #[inline]
    pub fn total_runtime_ns(&self) -> u64 {
        self.total_runtime_ns
    }

    #[inline]
    fn record_call(&mut self, accepted: bool, runtime_ns: u64) {
        self.called += 1;
        if accepted {
            self.accepted += 1;
        }
        self.total_runtime_ns = self.total_runtime_ns.saturating_add(runtime_ns);
    }

    #[inline]
    fn average_runtime_ns(&self) -> u64 {
        if self.called == 0 {
            0
        } else {
            self.total_runtime_ns / self.called
        }
    }
}

pub struct NeighborhoodFilterStack<T> {
    filters: Vec<Box<dyn NeighborhoodFilter<T>>>,
    statistics: Vec<NeighborhoodFilterStatistic>,
}

impl<T> Default for NeighborhoodFilterStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> NeighborhoodFilterStack<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            statistics: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            filters: Vec::with_capacity(capacity),
            statistics: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn with_filter_box(mut self, filter: Box<dyn NeighborhoodFilter<T>>) -> Self {
        self.filters.push(filter);
        self.statistics.push(NeighborhoodFilterStatistic::new());
        self
    }

    #[inline]
    pub fn add_filter_box(mut self, filter: Box<dyn NeighborhoodFilter<T>>) -> Self {
        self.filters.push(filter);
        self.statistics.push(NeighborhoodFilterStatistic::new());
        self
    }

    #[inline]
    pub fn add_filter<F>(&mut self, filter: F)
    where
        F: NeighborhoodFilter<T> + 'static,
    {
        self.filters.push(Box::new(filter));
        self.statistics.push(NeighborhoodFilterStatistic::new());
    }

    #[inline]
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: NeighborhoodFilter<T> + 'static,
    {
        self.add_filter(filter);
        self
    }

    #[inline]
    pub fn filters(&self) -> &[Box<dyn NeighborhoodFilter<T>>] {
        &self.filters
    }

    #[inline]
    pub fn statistics(&self) -> &[NeighborhoodFilterStatistic] {
        &self.statistics
    }

    #[inline]
    pub fn clear(&mut self) {
        self.filters.clear();
        self.statistics.clear();
    }

    #[inline]
    pub fn remove_filter_at(&mut self, index: usize) -> Option<Box<dyn NeighborhoodFilter<T>>> {
        if index < self.filters.len() {
            self.statistics.remove(index);
            Some(self.filters.remove(index))
        } else {
            None
        }
    }

    #[inline]
    pub fn accept<'s, 'm, 'p>(
        &mut self,
        context: FilterContext<'s, 'm, 'p, T>,
        plan: &Plan<'p, T>,
    ) -> bool
    where
        T: Copy + Ord,
    {
        if self.filters.is_empty() {
            return true;
        }

        let mut order: Vec<usize> = (0..self.filters.len()).collect();
        order.sort_by(|&a, &b| {
            let aa = self
                .statistics
                .get(a)
                .map(|s| s.average_runtime_ns())
                .unwrap_or(0);
            let bb = self
                .statistics
                .get(b)
                .map(|s| s.average_runtime_ns())
                .unwrap_or(0);
            aa.cmp(&bb).then(a.cmp(&b))
        });

        for idx in order {
            let start = std::time::Instant::now();
            let accepted = {
                let filter = &mut self.filters[idx];
                filter.accept(context, plan)
            };
            let elapsed_ns = start.elapsed().as_nanos() as u64;

            if let Some(stat) = self.statistics.get_mut(idx) {
                stat.record_call(accepted, elapsed_ns);
            }

            if !accepted {
                return false;
            }
        }
        true
    }
}

impl<T> std::fmt::Debug for NeighborhoodFilterStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.filters.iter().map(|flt| flt.name()).collect();
        f.debug_struct("NeighborhoodFilterStack")
            .field("filters", &names)
            .finish()
    }
}

impl<T> std::fmt::Display for NeighborhoodFilterStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NeighborhoodFilterStack[")?;
        for (i, filter) in self.filters.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", filter.name())?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod static_assert {
    use super::*;
    use static_assertions::const_assert_eq;

    const_assert_eq!(
        core::mem::size_of::<FilterContext<'static, 'static, 'static, i64>>(),
        16
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyFilter(&'static str);

    impl NeighborhoodFilter<i64> for DummyFilter {
        fn name(&self) -> &str {
            self.0
        }

        fn accept<'s, 'm, 'p>(
            &mut self,
            _context: FilterContext<'s, 'm, 'p, i64>,
            _plan: &Plan<'p, i64>,
        ) -> bool
        where
            i64: Copy + Ord,
        {
            true
        }
    }

    // Mirrors the production ordering logic, but only used here in tests.
    fn compute_expected_order(stack: &NeighborhoodFilterStack<i64>) -> Vec<usize> {
        let mut order: Vec<usize> = (0..stack.filters.len()).collect();
        order.sort_by(|&a, &b| {
            let aa = stack.statistics[a].average_runtime_ns();
            let bb = stack.statistics[b].average_runtime_ns();
            aa.cmp(&bb).then(a.cmp(&b))
        });
        order
    }

    #[test]
    fn test_dyn_filter_display_and_debug_use_name() {
        let f = DummyFilter("Alpha");
        let obj: &dyn NeighborhoodFilter<i64> = &f;

        assert_eq!(format!("{}", obj), "NeighborhoodFilter(Alpha)");
        assert_eq!(format!("{:?}", obj), "NeighborhoodFilter(Alpha)");
    }

    #[test]
    fn test_stack_display_and_debug_show_filter_names() {
        let mut stack = NeighborhoodFilterStack::<i64>::new();

        // Empty
        assert_eq!(format!("{}", stack), "NeighborhoodFilterStack[]");
        assert_eq!(
            format!("{:?}", stack),
            "NeighborhoodFilterStack { filters: [] }"
        );

        // With elements
        stack.add_filter(DummyFilter("A"));
        stack.add_filter(DummyFilter("B"));

        assert_eq!(format!("{}", stack), "NeighborhoodFilterStack[A, B]");
        assert_eq!(
            format!("{:?}", stack),
            "NeighborhoodFilterStack { filters: [\"A\", \"B\"] }"
        );
    }

    #[test]
    fn test_statistic_defaults() {
        let s = NeighborhoodFilterStatistic::new();
        assert_eq!(s.called(), 0);
        assert_eq!(s.accepted(), 0);
        assert_eq!(s.rejected(), 0);
        assert_eq!(s.total_runtime_ns(), 0);
        assert_eq!(s.average_runtime_ns(), 0);
    }

    #[test]
    fn test_statistic_record_call_math() {
        let mut s = NeighborhoodFilterStatistic::new();
        s.record_call(false, 5); // rejected
        s.record_call(true, 10); // accepted

        assert_eq!(s.called(), 2);
        assert_eq!(s.accepted(), 1);
        assert_eq!(s.rejected(), 1);
        assert_eq!(s.total_runtime_ns(), 15);
        assert_eq!(s.average_runtime_ns(), 7); // integer division
    }

    #[test]
    fn test_statistics_stay_in_sync_with_filters() {
        let mut stack = NeighborhoodFilterStack::<i64>::with_capacity(2);
        assert_eq!(stack.filters().len(), 0);
        assert_eq!(stack.statistics().len(), 0);

        stack.add_filter(DummyFilter("A"));
        stack.add_filter(DummyFilter("B"));
        assert_eq!(stack.filters().len(), 2);
        assert_eq!(stack.statistics().len(), 2);

        // Remove first
        let removed = stack.remove_filter_at(0);
        assert!(removed.is_some());
        assert_eq!(stack.filters().len(), 1);
        assert_eq!(stack.statistics().len(), 1);

        // Clear
        stack.clear();
        assert_eq!(stack.filters().len(), 0);
        assert_eq!(stack.statistics().len(), 0);
    }

    #[test]
    fn test_remove_out_of_range_is_none_and_safe() {
        let mut stack = NeighborhoodFilterStack::<i64>::new();
        stack.add_filter(DummyFilter("A"));
        assert!(stack.remove_filter_at(1).is_none()); // index 1 out of range
        assert_eq!(stack.filters().len(), 1);
        assert_eq!(stack.statistics().len(), 1);
    }

    #[test]
    fn test_builder_add_filter_box_chaining() {
        // Ensure the builder-style add_filter_box returns Self and keeps stats aligned
        let stack = NeighborhoodFilterStack::<i64>::new()
            .add_filter_box(Box::new(DummyFilter("A")))
            .add_filter_box(Box::new(DummyFilter("B")));

        assert_eq!(stack.filters().len(), 2);
        assert_eq!(stack.statistics().len(), 2);
        assert_eq!(format!("{}", stack), "NeighborhoodFilterStack[A, B]");
    }

    #[test]
    fn test_order_prefers_shorter_average_first_and_unknowns_first() {
        let mut stack = NeighborhoodFilterStack::<i64>::new();
        stack.add_filter(DummyFilter("Slow")); // idx 0
        stack.add_filter(DummyFilter("Fast")); // idx 1
        stack.add_filter(DummyFilter("New")); // idx 2 (unknown avg -> 0)

        // Simulate stats:
        // Slow: avg = 100 ns (1000 / 10)
        // Fast: avg =   2 ns (20 / 10)
        // New:  avg =   0 ns (no calls yet)
        {
            let s0 = &mut stack.statistics[0];
            s0.record_call(true, 1000);
            for _ in 0..9 {
                s0.record_call(true, 0);
            } // 10 calls total

            let s1 = &mut stack.statistics[1];
            s1.record_call(true, 20);
            for _ in 0..9 {
                s1.record_call(true, 0);
            } // 10 calls total
        }

        let order = compute_expected_order(&stack);
        // Unknown (0 ns) first -> idx 2, then Fast (2 ns) -> idx 1, then Slow (100 ns) -> idx 0
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn test_order_is_stable_when_averages_equal() {
        let mut stack = NeighborhoodFilterStack::<i64>::new();
        stack.add_filter(DummyFilter("X")); // idx 0
        stack.add_filter(DummyFilter("Y")); // idx 1

        // X: avg = 50 (100/2), Y: avg = 50 (200/4) -> same average
        {
            let sx = &mut stack.statistics[0];
            sx.record_call(true, 60);
            sx.record_call(true, 40);

            let sy = &mut stack.statistics[1];
            for _ in 0..4 {
                sy.record_call(true, 50);
            }
        }

        let order = compute_expected_order(&stack);
        // Stable by original index when equal averages
        assert_eq!(order, vec![0, 1]);
    }
}
