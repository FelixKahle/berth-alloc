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

use crate::state::chain::delta::ChainDelta;

#[derive(Clone, Debug)]
pub struct VirtualDoubleChain<'a> {
    next: &'a [usize],
    delta: &'a ChainDelta,
    start: &'a [usize],
    end: &'a [usize],
}

impl<'a> VirtualDoubleChain<'a> {
    #[inline]
    pub fn new(
        next: &'a [usize],
        start: &'a [usize],
        end: &'a [usize],
        delta: &'a ChainDelta,
    ) -> Self {
        Self {
            next,
            delta,
            start,
            end,
        }
    }

    #[inline]
    pub fn iter_berth(&self, berth: usize) -> impl Iterator<Item = usize> + '_ {
        let s = self.start[berth];
        let e = self.end[berth];
        let mut cur = self.delta.next_after(self.next, s);
        std::iter::from_fn(move || {
            if cur == e {
                return None;
            }
            let out = cur;
            cur = self.delta.next_after(self.next, cur);
            Some(out)
        })
    }

    #[inline]
    pub fn next(&self, i: usize) -> usize {
        self.delta.next_after(self.next, i)
    }

    #[inline]
    pub fn changed(&self, i: usize) -> bool {
        self.delta.changed(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::static_assertions::assert_impl_all;

    // Build a simple base layout compatible with DoubleChain's sentinel scheme, but only using next/start/end.
    // - There are `n_nodes` regular nodes: 0..n_nodes-1
    // - For each berth b, we create start = base + 2*b, end = start + 1, where base = n_nodes
    // - `sequences` lists the node sequence for each berth; nodes are linked: start -> first -> ... -> last -> end
    fn make_base(n_nodes: usize, sequences: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let b = sequences.len();
        let base = n_nodes;
        let len = n_nodes + 2 * b;

        let mut next = vec![0; len];
        // initialize nodes to self-loops
        for i in 0..n_nodes {
            next[i] = i;
        }

        // build start/end arrays and sentinel linkages
        let mut start = Vec::with_capacity(b);
        let mut end = Vec::with_capacity(b);

        for berth in 0..b {
            let s = base + 2 * berth;
            let e = s + 1;
            start.push(s);
            end.push(e);
            // default ring: start -> end, end -> end
            next[s] = e;
            next[e] = e;

            if let Some(first) = sequences[berth].first().copied() {
                // start -> first
                next[s] = first;
                // chain nodes
                for w in sequences[berth].windows(2) {
                    let (a, b) = (w[0], w[1]);
                    next[a] = b;
                }
                // last -> end
                let last = *sequences[berth].last().unwrap();
                next[last] = e;
            }
        }

        (next, start, end)
    }

    fn collect(vdc: &VirtualDoubleChain, berth: usize) -> Vec<usize> {
        vdc.iter_berth(berth).collect()
    }

    #[test]
    fn trait_impls() {
        assert_impl_all!(VirtualDoubleChain<'static>: Send, Sync, Clone, std::fmt::Debug);
    }

    #[test]
    fn empty_berths_iterate_to_nothing() {
        // Two empty berths; five dangling nodes (unused) are fine
        let (next, start, end) = make_base(5, &[vec![], vec![]]);
        let cd = ChainDelta::new();
        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);

        assert!(collect(&vdc, 0).is_empty());
        assert!(collect(&vdc, 1).is_empty());

        // Passthrough next/changed behavior
        // For in-range indices (nodes, sentinels): changed should be false
        for i in 0..next.len() {
            assert!(!vdc.changed(i));
            assert_eq!(vdc.next(i), next[i]);
        }
    }

    #[test]
    fn single_berth_sequence_and_passthrough() {
        // One berth with [0,1,2]
        let (next, start, end) = make_base(4, &[vec![0, 1, 2]]);
        let cd = ChainDelta::new();
        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);

        assert_eq!(collect(&vdc, 0), vec![0, 1, 2]);

        // next() should passthrough without overrides
        for i in 0..next.len() {
            assert_eq!(vdc.next(i), next[i]);
            assert!(!vdc.changed(i));
        }
    }

    #[test]
    fn override_tail_in_middle_truncates_sequence() {
        // Berth 0: [0,1,2,3]
        let (next, start, end) = make_base(6, &[vec![0, 1, 2, 3]]);
        let mut cd = ChainDelta::new();

        // Override tail=1 -> end (truncate after 1)
        let s0 = start[0];
        let e0 = end[0];
        // Guard expected_head is base next[1] which is 2
        cd.push(1, next[1], e0);

        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
        assert!(vdc.changed(1));
        assert_eq!(vdc.next(1), e0);
        assert_eq!(collect(&vdc, 0), vec![0, 1]);

        // Unaffected tails pass through
        assert!(!vdc.changed(0));
        assert_eq!(vdc.next(0), next[0]);
        assert!(!vdc.changed(s0));
        assert_eq!(vdc.next(s0), next[s0]);
    }

    #[test]
    fn last_wins_for_same_tail() {
        // Berth 0: [0,1,2,3]
        let (next, start, end) = make_base(6, &[vec![0, 1, 2, 3]]);
        let mut cd = ChainDelta::new();
        let e0 = end[0];

        // First override 2 -> 3 (no change in effect since base is 2->3)
        cd.push(2, next[2], 3);
        {
            let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
            assert!(vdc.changed(2));
            assert_eq!(vdc.next(2), 3);
            assert_eq!(collect(&vdc, 0), vec![0, 1, 2, 3]);
        }

        // Second override 2 -> end (truncate after 2). Latest should win.
        cd.push(2, 3, e0);
        {
            let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
            assert!(vdc.changed(2));
            assert_eq!(vdc.next(2), e0);
            assert_eq!(collect(&vdc, 0), vec![0, 1, 2]);
        }
    }

    #[test]
    fn override_start_sentinel_next_skips_first_node() {
        // Berth 0: [0,1,2]
        let (next, start, end) = make_base(5, &[vec![0, 1, 2]]);
        let mut cd = ChainDelta::new();

        let s0 = start[0];
        // Change start's next from 0 to 1, skipping node 0
        cd.push(s0, next[s0], 1);

        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
        assert!(vdc.changed(s0));
        assert_eq!(vdc.next(s0), 1);
        assert_eq!(collect(&vdc, 0), vec![1, 2]);
    }

    #[test]
    fn touch_many_does_not_affect_virtual_view() {
        // Berth 0: [0,1]
        let (next, start, end) = make_base(3, &[vec![0, 1]]);
        let mut cd = ChainDelta::new();

        // Mark as touched but no overrides
        cd.touch_many(&[0, 1, start[0], end[0]]);

        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
        assert_eq!(collect(&vdc, 0), vec![0, 1]);

        // No tail is changed
        for i in 0..next.len() {
            assert!(!vdc.changed(i));
            assert_eq!(vdc.next(i), next[i]);
        }
    }

    #[test]
    fn clear_resets_overrides_and_iteration_reverts() {
        // Berth 0: [0,1,2]
        let (next, start, end) = make_base(5, &[vec![0, 1, 2]]);
        let mut cd = ChainDelta::new();

        // Set an override: 1 -> end (so iteration would be [0,1])
        let e0 = end[0];
        cd.push(1, next[1], e0);

        {
            let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
            assert_eq!(collect(&vdc, 0), vec![0, 1]);
            assert!(vdc.changed(1));
            assert_eq!(vdc.next(1), e0);
        }

        // Clear overrides, then re-evaluate
        cd.clear();

        {
            let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
            assert_eq!(collect(&vdc, 0), vec![0, 1, 2]);
            assert!(!vdc.changed(1));
            assert_eq!(vdc.next(1), next[1]);
        }
    }

    #[test]
    fn multiple_berths_with_independent_overrides() {
        // Berth 0: [0,1], Berth 1: [2,3]
        let (next, start, end) = make_base(6, &[vec![0, 1], vec![2, 3]]);
        let mut cd = ChainDelta::new();

        let e0 = end[0];
        let e1 = end[1];

        // Trim both lists by redirecting last tails to end sentinels
        cd.push(0, next[0], e0);
        cd.push(2, next[2], e1);

        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
        assert_eq!(collect(&vdc, 0), vec![0]); // 1 removed
        assert_eq!(collect(&vdc, 1), vec![2]); // 3 removed

        assert!(vdc.changed(0) && vdc.changed(2));
        assert_eq!(vdc.next(0), e0);
        assert_eq!(vdc.next(2), e1);
    }

    #[test]
    fn next_for_large_index_with_override_works_without_panics() {
        // Minimal base with one empty berth
        let (next, start, end) = make_base(1, &[vec![]]);
        let mut cd = ChainDelta::new();

        // Set an override for an index far beyond base len.
        // This is safe because next() will return the override and not index into base.
        cd.push(1_000, 0, 42);

        let vdc = VirtualDoubleChain::new(&next, &start, &end, &cd);
        assert!(vdc.changed(1_000));
        assert_eq!(vdc.next(1_000), 42);
    }

    #[test]
    fn clone_observes_same_view() {
        // Berth 0: [0,1,2]
        let (next, start, end) = make_base(4, &[vec![0, 1, 2]]);
        let mut cd = ChainDelta::new();

        // Override 1 -> end (truncate)
        let e0 = end[0];
        cd.push(1, next[1], e0);

        let v1 = VirtualDoubleChain::new(&next, &start, &end, &cd);
        let v2 = v1.clone();

        assert_eq!(collect(&v1, 0), vec![0, 1]);
        assert_eq!(collect(&v2, 0), vec![0, 1]);
        assert!(v1.changed(1) && v2.changed(1));
        assert_eq!(v1.next(1), e0);
        assert_eq!(v2.next(1), e0);
    }
}
