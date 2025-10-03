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

use crate::state::chain_set::{base::ChainSet, delta::ChainSetDelta, view::ChainSetView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainSetOverlay<'base, 'delta> {
    base: &'base ChainSet,
    delta: &'delta ChainSetDelta,
}

impl<'base, 'delta> ChainSetOverlay<'base, 'delta> {
    #[inline]
    pub fn new(base: &'base ChainSet, delta: &'delta ChainSetDelta) -> Self {
        Self { base, delta }
    }

    #[inline]
    fn total_nodes(&self) -> usize {
        self.base.next_slice().len()
    }

    #[inline]
    fn next_of(&self, node: usize) -> usize {
        if let Some(nxt) = self.delta.next_override_for_tail(node) {
            nxt
        } else {
            self.base.next_slice()[node]
        }
    }

    #[inline]
    fn prev_of(&self, node: usize) -> usize {
        if let Some(prev) = self.delta.prev_override_for_head(node) {
            prev
        } else {
            self.base.previous_slice()[node]
        }
    }
}

impl<'base, 'delta> ChainSetView for ChainSetOverlay<'base, 'delta> {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.base.num_nodes()
    }

    #[inline]
    fn num_chains(&self) -> usize {
        self.base.num_chains()
    }

    #[inline]
    fn start_of_chain(&self, chain: usize) -> usize {
        self.base.start_of_chain(chain)
    }

    #[inline]
    fn end_of_chain(&self, chain: usize) -> usize {
        self.base.end_of_chain(chain)
    }

    #[inline]
    fn next_node(&self, node: usize) -> Option<usize> {
        if node >= self.total_nodes() {
            return None;
        }
        Some(self.next_of(node))
    }

    #[inline]
    fn prev_node(&self, node: usize) -> Option<usize> {
        if node >= self.total_nodes() {
            return None;
        }
        Some(self.prev_of(node))
    }

    #[inline]
    fn is_sentinel_node(&self, node: usize) -> bool {
        node >= self.num_nodes()
    }

    #[inline]
    fn is_head_node(&self, node: usize) -> bool {
        node >= self.num_nodes() && ((node - self.num_nodes()) & 1) == 0
    }

    #[inline]
    fn is_tail_node(&self, node: usize) -> bool {
        node >= self.num_nodes() && ((node - self.num_nodes()) & 1) == 1
    }

    #[inline]
    fn is_node_unperformed(&self, node: usize) -> bool {
        debug_assert!(node < self.num_nodes());
        self.next_of(node) == node && self.prev_of(node) == node
    }

    #[inline]
    fn is_chain_empty(&self, chain: usize) -> bool {
        debug_assert!(chain < self.num_chains());
        let s = self.start_of_chain(chain);
        let e = self.end_of_chain(chain);
        self.next_of(s) == e && self.prev_of(e) == s
    }

    #[inline]
    fn iter_chain(&self, chain: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(chain < self.num_chains());
        let s = self.start_of_chain(chain);
        let e = self.end_of_chain(chain);

        let mut cur = self.next_of(s);

        #[cfg(debug_assertions)]
        let mut steps = 0usize;
        #[cfg(debug_assertions)]
        let cap = self.total_nodes();

        std::iter::from_fn(move || {
            if cur == e {
                return None;
            }

            #[cfg(debug_assertions)]
            {
                steps += 1;
                debug_assert!(
                    steps <= cap,
                    "cycle detected during overlay iteration (exceeded {})",
                    cap
                );
            }

            let out = cur;
            cur = self.next_of(cur);
            Some(out)
        })
    }
}

#[cfg(test)]
mod apply_tests {
    use super::ChainSet;
    use crate::state::chain_set::{
        delta::{ChainNextRewire, ChainSetDelta},
        view::ChainSetView,
    };

    // Helper that collects a chain into a Vec for easy assertions.
    fn collect_chain(cs: &ChainSet, chain: usize) -> Vec<usize> {
        cs.iter_chain(chain).collect::<Vec<_>>()
    }

    #[test]
    fn test_apply_rewire_builds_chain_and_updates_prev_next() {
        let mut cs = ChainSet::new(6, 1);
        let s = cs.start_of_chain(0);
        let e = cs.end_of_chain(0);

        // Initially empty
        assert!(cs.is_chain_empty(0));
        assert_eq!(collect_chain(&cs, 0), vec![]);

        // start -> 2
        cs.apply_rewire(ChainNextRewire::new(s, 2));
        // 2 -> end
        cs.apply_rewire(ChainNextRewire::new(2, e));

        assert!(!cs.is_chain_empty(0));
        assert_eq!(collect_chain(&cs, 0), vec![2]);

        // Pointers are consistent
        assert_eq!(cs.next_node(s), Some(2));
        assert_eq!(cs.prev_node(2), Some(s));
        assert_eq!(cs.next_node(2), Some(e));
        assert_eq!(cs.prev_node(e), Some(2));

        // Node 2 is no longer "unperformed"
        assert!(!cs.is_node_unperformed(2));

        // Add another: 2 -> 4, then 4 -> end
        cs.apply_rewire(ChainNextRewire::new(2, 4));
        cs.apply_rewire(ChainNextRewire::new(4, e));

        assert_eq!(collect_chain(&cs, 0), vec![2, 4]);

        // Old head 'end' was detached when 2->4 was applied; re-attached by 4->end
        assert_eq!(cs.prev_node(e), Some(4));
    }

    #[test]
    fn test_apply_delta_applies_all_rewires_single_chain() {
        let mut cs = ChainSet::new(8, 1);
        let s = cs.start_of_chain(0);
        let e = cs.end_of_chain(0);

        let mut delta = ChainSetDelta::new();
        // Build: start -> 3 -> 5 -> end
        delta.push_rewire(ChainNextRewire::new(s, 3));
        delta.push_rewire(ChainNextRewire::new(3, 5));
        delta.push_rewire(ChainNextRewire::new(5, e));

        cs.apply_delta(&delta);

        assert_eq!(collect_chain(&cs, 0), vec![3, 5]);

        // Pointers consistent
        assert_eq!(cs.prev_node(3), Some(s));
        assert_eq!(cs.next_node(3), Some(5));
        assert_eq!(cs.prev_node(5), Some(3));
        assert_eq!(cs.next_node(5), Some(e));

        // Unperformed flags
        assert!(!cs.is_node_unperformed(3));
        assert!(!cs.is_node_unperformed(5));
        // Others remain unperformed
        for &n in &[0, 1, 2, 4, 6, 7] {
            assert!(cs.is_node_unperformed(n));
        }
    }

    #[test]
    fn test_apply_delta_multiple_chains() {
        let mut cs = ChainSet::new(10, 2);
        let s0 = cs.start_of_chain(0);
        let e0 = cs.end_of_chain(0);
        let s1 = cs.start_of_chain(1);
        let e1 = cs.end_of_chain(1);

        let mut delta = ChainSetDelta::with_capacity(6);
        // Chain 0: start0 -> 2 -> 9 -> end0
        delta.push_rewire(ChainNextRewire::new(s0, 2));
        delta.push_rewire(ChainNextRewire::new(2, 9));
        delta.push_rewire(ChainNextRewire::new(9, e0));

        // Chain 1: start1 -> 0 -> end1
        delta.push_rewire(ChainNextRewire::new(s1, 0));
        delta.push_rewire(ChainNextRewire::new(0, e1));

        cs.apply_delta(&delta);

        assert_eq!(collect_chain(&cs, 0), vec![2, 9]);
        assert_eq!(collect_chain(&cs, 1), vec![0]);

        // Independence checks
        assert_eq!(cs.prev_node(e0), Some(9));
        assert_eq!(cs.prev_node(e1), Some(0));
    }

    #[test]
    fn test_apply_delta_is_noop_for_empty_delta() {
        let mut cs = ChainSet::new(4, 1);
        let before: Vec<usize> = collect_chain(&cs, 0);
        let delta = ChainSetDelta::new();

        cs.apply_delta(&delta);

        let after: Vec<usize> = collect_chain(&cs, 0);
        assert_eq!(before, after);
        assert!(cs.is_chain_empty(0));
    }
}
