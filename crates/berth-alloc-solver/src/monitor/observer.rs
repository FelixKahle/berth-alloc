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

/// Observer for high-level search lifecycle events.
/// All methods have default no-op implementations.
pub trait SearchObserver: Send {
    /// A short identifier for logging and diagnostics.
    fn name(&self) -> &str {
        "SearchObserver"
    }

    fn on_search_start(&mut self) {}
    fn on_search_end(&mut self) {}

    fn on_iteration_accepted(&mut self, _iter_id: u64, _new_cost: i64, _old_cost: i64) {}
    fn on_new_incumbent(&mut self, _iter_id: u64, _cost: i64) {}
}

/// A no-op observer useful as default.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullObserver;

impl SearchObserver for NullObserver {
    fn name(&self) -> &str {
        "NullObserver"
    }
}

/// An observer that forwards events to a list of boxed observers.
/// Useful to combine independent observers.
#[derive(Default)]
pub struct CompositeSearchObserver {
    observers: Vec<Box<dyn SearchObserver + Send>>,
}

impl CompositeSearchObserver {
    #[inline]
    pub fn new() -> Self {
        Self {
            observers: Vec::new(),
        }
    }

    /// Chain-builder: add an observer and return self.
    #[inline]
    pub fn with<O: SearchObserver + Send + 'static>(mut self, o: O) -> Self {
        self.observers.push(Box::new(o));
        self
    }

    /// Push an observer into the composite.
    #[inline]
    pub fn push<O: SearchObserver + Send + 'static>(&mut self, o: O) {
        self.observers.push(Box::new(o));
    }

    /// Returns the number of inner observers.
    #[inline]
    pub fn len(&self) -> usize {
        self.observers.len()
    }

    /// Returns true when there are no inner observers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }
}

impl SearchObserver for CompositeSearchObserver {
    fn name(&self) -> &str {
        "CompositeSearchObserver"
    }

    fn on_search_start(&mut self) {
        for o in &mut self.observers {
            o.on_search_start();
        }
    }
    fn on_search_end(&mut self) {
        for o in &mut self.observers {
            o.on_search_end();
        }
    }
    fn on_iteration_accepted(&mut self, iter_id: u64, new_cost: i64, old_cost: i64) {
        for o in &mut self.observers {
            o.on_iteration_accepted(iter_id, new_cost, old_cost);
        }
    }
    fn on_new_incumbent(&mut self, iter_id: u64, cost: i64) {
        for o in &mut self.observers {
            o.on_new_incumbent(iter_id, cost);
        }
    }
}

impl std::fmt::Debug for CompositeSearchObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositeSearchObserver")
            .field("len", &self.observers.len())
            .finish()
    }
}

/// Implement pretty printing for trait objects with any lifetime.
/// This avoids requiring concrete types at call sites for logging/diagnostics.
impl<'a> fmt::Debug for dyn SearchObserver + 'a {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SearchObserver({})", self.name())
    }
}

impl<'a> fmt::Display for dyn SearchObserver + 'a {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SearchObserver({})", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingObserver {
        calls: Arc<Mutex<Calls>>,
        label: &'static str,
    }

    #[derive(Default, Debug, PartialEq, Eq, Clone)]
    struct Calls {
        start: u32,
        end: u32,
        accepted: u32,
        new_inc: u32,
        // last payloads
        last_iter: Option<u64>,
        last_new_cost: Option<i64>,
        last_old_cost: Option<i64>,
        last_inc_cost: Option<i64>,
    }

    impl RecordingObserver {
        fn new(label: &'static str) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Calls::default())),
                label,
            }
        }

        #[allow(dead_code)]
        fn snapshot(&self) -> Calls {
            self.calls.lock().unwrap().clone()
        }
    }

    impl SearchObserver for RecordingObserver {
        fn name(&self) -> &str {
            self.label
        }
        fn on_search_start(&mut self) {
            self.calls.lock().unwrap().start += 1;
        }
        fn on_search_end(&mut self) {
            self.calls.lock().unwrap().end += 1;
        }
        fn on_iteration_accepted(&mut self, iter_id: u64, new_cost: i64, old_cost: i64) {
            let mut c = self.calls.lock().unwrap();
            c.accepted += 1;
            c.last_iter = Some(iter_id);
            c.last_new_cost = Some(new_cost);
            c.last_old_cost = Some(old_cost);
        }
        fn on_new_incumbent(&mut self, iter_id: u64, cost: i64) {
            let mut c = self.calls.lock().unwrap();
            c.new_inc += 1;
            c.last_iter = Some(iter_id);
            c.last_inc_cost = Some(cost);
        }
    }

    #[test]
    fn null_observer_is_noop_and_prints() {
        let mut o = NullObserver::default();
        o.on_search_start();
        o.on_iteration_accepted(1, 10, 20);
        o.on_new_incumbent(2, 5);
        o.on_search_end();

        let d = format!("{:?}", o);
        assert!(d.contains("NullObserver"));
        let t: &dyn SearchObserver = &o;
        assert_eq!(format!("{:?}", t), "SearchObserver(NullObserver)");
        assert_eq!(format!("{}", t), "SearchObserver(NullObserver)");
    }

    #[test]
    fn composite_forwards_calls_to_children() {
        let mut c = CompositeSearchObserver::new();
        assert!(c.is_empty());
        c.push(NullObserver::default()); // a noop child
        c = c
            .with(RecordingObserver::new("rec1"))
            .with(RecordingObserver::new("rec2"));
        assert_eq!(c.len(), 3);

        // Build a composite we can inspect post-calls.
        let r1 = RecordingObserver::new("rec1");
        let r1_arc = r1.calls.clone();
        let r2 = RecordingObserver::new("rec2");
        let r2_arc = r2.calls.clone();

        let mut cc = CompositeSearchObserver::new().with(r1).with(r2);

        cc.on_search_start();
        cc.on_iteration_accepted(7, 123, 200);
        cc.on_new_incumbent(8, 111);
        cc.on_search_end();

        let s1 = r1_arc.lock().unwrap().clone();
        let s2 = r2_arc.lock().unwrap().clone();

        for s in &[s1, s2] {
            assert_eq!(s.start, 1);
            assert_eq!(s.end, 1);
            assert_eq!(s.accepted, 1);
            assert_eq!(s.new_inc, 1);
            assert_eq!(s.last_iter, Some(8)); // last event was new_incumbent at iter 8
            assert_eq!(s.last_new_cost, Some(123));
            assert_eq!(s.last_old_cost, Some(200));
            assert_eq!(s.last_inc_cost, Some(111));
        }

        // Formatting for dyn object
        let t: &dyn SearchObserver = &cc;
        assert_eq!(
            format!("{:?}", t),
            "SearchObserver(CompositeSearchObserver)"
        );
        assert_eq!(format!("{}", t), "SearchObserver(CompositeSearchObserver)");
    }

    // Compile-time assertion: CompositeSearchObserver is Send.
    fn assert_send<T: Send>() {}
    #[test]
    fn composite_is_send() {
        assert_send::<CompositeSearchObserver>();
    }
}
