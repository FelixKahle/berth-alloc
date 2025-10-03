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

use crate::state::chain_set::{
    delta::{ChainNextRewire, ChainSetDelta},
    view::ChainSetView,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSet {
    next: Vec<usize>,
    prev: Vec<usize>,
    start: Vec<usize>,
    end: Vec<usize>,
    num_nodes: usize,
}

impl ChainSet {
    #[inline]
    pub fn new(num_nodes: usize, num_chains: usize) -> Self {
        let total = num_nodes + 2 * num_chains;

        let mut next = vec![0; total];
        let mut prev = vec![0; total];

        for i in 0..num_nodes {
            next[i] = i;
            prev[i] = i;
        }

        let mut start = Vec::with_capacity(num_chains);
        let mut end = Vec::with_capacity(num_chains);

        for b in 0..num_chains {
            let s = num_nodes + 2 * b;
            let e = s + 1;
            start.push(s);
            end.push(e);

            next[s] = e;
            prev[s] = s;
            next[e] = e;
            prev[e] = s;
        }

        Self {
            next,
            prev,
            start,
            end,
            num_nodes,
        }
    }

    #[inline]
    fn num_total_nodes(&self) -> usize {
        self.next.len()
    }

    #[inline]
    pub fn next_slice(&self) -> &[usize] {
        &self.next
    }

    #[inline]
    pub fn previous_slice(&self) -> &[usize] {
        &self.prev
    }

    #[inline]
    pub fn set_next(&mut self, tail: usize, new_head: usize) {
        debug_assert!(tail < self.next.len());
        debug_assert!(new_head < self.next.len());
        debug_assert!(
            !(self.is_head_node(tail) && new_head == tail),
            "set_next would create start->start self-loop"
        );
        debug_assert!(
            !self.is_head_node(new_head),
            "new_head must not be a head sentinel"
        );

        let old_head = self.next[tail];
        if old_head != new_head && self.prev[old_head] == tail {
            self.prev[old_head] = old_head;
        }
        self.next[tail] = new_head;
        self.prev[new_head] = tail;
    }

    #[inline]
    pub fn apply_rewire(&mut self, rewire: ChainNextRewire) {
        self.set_next(rewire.tail(), rewire.successor());
    }

    #[inline]
    pub fn apply_delta(&mut self, delta: &ChainSetDelta) {
        for &r in delta.rewires() {
            self.apply_rewire(r);
        }
    }
}

impl ChainSetView for ChainSet {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    fn num_chains(&self) -> usize {
        debug_assert!(self.start.len() == self.end.len());

        self.start.len()
    }

    #[inline]
    fn start_of_chain(&self, chain: usize) -> usize {
        debug_assert!(chain < self.num_chains());

        self.start[chain]
    }

    #[inline]
    fn end_of_chain(&self, chain: usize) -> usize {
        debug_assert!(chain < self.num_chains());

        self.end[chain]
    }

    #[inline]
    fn next_node(&self, node: usize) -> Option<usize> {
        let num_total_nodes = self.num_total_nodes();

        if node >= num_total_nodes {
            return None;
        }

        Some(self.next[node])
    }

    #[inline]
    fn prev_node(&self, node: usize) -> Option<usize> {
        let num_total_nodes = self.num_total_nodes();

        if node >= num_total_nodes {
            return None;
        }

        Some(self.prev[node])
    }

    #[inline]
    fn is_sentinel_node(&self, node: usize) -> bool {
        node >= self.num_nodes
    }

    #[inline]
    fn is_head_node(&self, node: usize) -> bool {
        node >= self.num_nodes && ((node - self.num_nodes) & 1) == 0
    }

    #[inline]
    fn is_tail_node(&self, node: usize) -> bool {
        node >= self.num_nodes && ((node - self.num_nodes) & 1) == 1
    }

    #[inline]
    fn is_node_unperformed(&self, node: usize) -> bool {
        debug_assert!(node < self.num_nodes());
        self.next[node] == node && self.prev[node] == node
    }

    #[inline]
    fn is_chain_empty(&self, chain: usize) -> bool {
        debug_assert!(chain < self.num_chains());

        let num_total_nodes = self.num_total_nodes();
        let start = self.start[chain];
        let end = self.end[chain];

        debug_assert!(start < num_total_nodes);
        debug_assert!(end < num_total_nodes);

        self.next[start] == end && self.prev[end] == start
    }

    #[inline]
    fn iter_chain(&self, chain: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(chain < self.num_chains());

        let num_total_nodes = self.num_total_nodes();
        let start = self.start[chain];
        let end = self.end[chain];

        debug_assert!(start < num_total_nodes);
        debug_assert!(end < num_total_nodes);

        ChainIter::new(&self.next, self.next[start], end)
    }
}

#[derive(Debug, Clone)]
pub struct ChainIter<'slice> {
    next: &'slice [usize],
    current: usize,
    end: usize,
}

impl<'slice> ChainIter<'slice> {
    #[inline]
    fn new(next: &'slice [usize], start: usize, end: usize) -> Self {
        Self {
            next,
            current: start,
            end,
        }
    }
}

impl<'slice> Iterator for ChainIter<'slice> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.end {
            return None;
        }
        let out = self.current;
        self.current = self.next[self.current];
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link_sequence(cs: &mut ChainSet, chain: usize, nodes: &[usize]) {
        // Helper that links a sequence of nodes into the given chain:
        // start -> n0 -> n1 -> ... -> nk -> end
        let s = cs.start_of_chain(chain);
        let e = cs.end_of_chain(chain);

        let mut tail = s;
        for &n in nodes {
            cs.set_next(tail, n);
            tail = n;
        }
        cs.set_next(tail, e);
    }

    // Helper that collects a chain into a Vec for easy assertions.
    fn collect_chain(cs: &ChainSet, chain: usize) -> Vec<usize> {
        cs.iter_chain(chain).collect::<Vec<_>>()
    }

    #[test]
    fn test_initial_layout_and_sentinels() {
        let num_nodes = 5;
        let num_chains = 2;
        let cs = ChainSet::new(num_nodes, num_chains);

        // Counts
        assert_eq!(cs.num_nodes(), num_nodes);
        assert_eq!(cs.num_chains(), num_chains);

        // Chain 0 sentinels
        let s0 = cs.start_of_chain(0);
        let e0 = cs.end_of_chain(0);
        assert!(cs.is_sentinel_node(s0));
        assert!(cs.is_sentinel_node(e0));
        assert!(cs.is_head_node(s0));
        assert!(cs.is_tail_node(e0));
        assert!(cs.is_chain_empty(0));
        assert_eq!(cs.next_node(s0), Some(e0));
        assert_eq!(cs.prev_node(e0), Some(s0));

        // Chain 1 sentinels
        let s1 = cs.start_of_chain(1);
        let e1 = cs.end_of_chain(1);
        assert!(cs.is_sentinel_node(s1));
        assert!(cs.is_sentinel_node(e1));
        assert!(cs.is_head_node(s1));
        assert!(cs.is_tail_node(e1));
        assert!(cs.is_chain_empty(1));
        assert_eq!(cs.next_node(s1), Some(e1));
        assert_eq!(cs.prev_node(e1), Some(s1));

        // Regular nodes are not sentinels, initially "unperformed"
        for n in 0..num_nodes {
            assert!(!cs.is_sentinel_node(n));
            assert!(cs.is_node_unperformed(n));
        }

        // Iteration over empty chains yields nothing
        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(cs.iter_chain(1).collect::<Vec<_>>(), Vec::<usize>::new());
    }

    #[test]
    fn test_link_single_node_into_chain() {
        let num_nodes = 3;
        let num_chains = 1;
        let mut cs = ChainSet::new(num_nodes, num_chains);

        let s = cs.start_of_chain(0);
        let e = cs.end_of_chain(0);

        // Link node 0 into chain 0: start -> 0 -> end
        cs.set_next(s, 0);
        cs.set_next(0, e);

        assert!(!cs.is_chain_empty(0));
        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![0]);

        // Node 0 should no longer be "unperformed"
        assert!(!cs.is_node_unperformed(0));
        // Other nodes remain unperformed
        assert!(cs.is_node_unperformed(1));
        assert!(cs.is_node_unperformed(2));

        // Structural checks
        assert_eq!(cs.next_node(s), Some(0));
        assert_eq!(cs.prev_node(0), Some(s));
        assert_eq!(cs.next_node(0), Some(e));
        assert_eq!(cs.prev_node(e), Some(0));
    }

    #[test]
    fn test_link_multiple_nodes_ordering() {
        let num_nodes = 6;
        let num_chains = 1;
        let mut cs = ChainSet::new(num_nodes, num_chains);

        // Build: start -> 2 -> 4 -> 1 -> end
        link_sequence(&mut cs, 0, &[2, 4, 1]);

        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![2, 4, 1]);

        let s = cs.start_of_chain(0);
        let e = cs.end_of_chain(0);

        // Check local links
        assert_eq!(cs.next_node(s), Some(2));
        assert_eq!(cs.prev_node(2), Some(s));
        assert_eq!(cs.next_node(2), Some(4));
        assert_eq!(cs.prev_node(4), Some(2));
        assert_eq!(cs.next_node(4), Some(1));
        assert_eq!(cs.prev_node(1), Some(4));
        assert_eq!(cs.next_node(1), Some(e));
        assert_eq!(cs.prev_node(e), Some(1));

        // Nodes on the chain are not "unperformed"
        for &n in &[2, 4, 1] {
            assert!(!cs.is_node_unperformed(n));
        }
        // Nodes not on the chain remain unperformed
        for &n in &[0, 3, 5] {
            assert!(cs.is_node_unperformed(n));
        }
    }

    #[test]
    fn test_multiple_chains_are_independent() {
        let num_nodes = 10;
        let num_chains = 3;
        let mut cs = ChainSet::new(num_nodes, num_chains);

        // Chain 0: [0, 1, 2]
        link_sequence(&mut cs, 0, &[0, 1, 2]);

        // Chain 1: [7]
        link_sequence(&mut cs, 1, &[7]);

        // Chain 2: []
        // leave empty

        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(cs.iter_chain(1).collect::<Vec<_>>(), vec![7]);
        assert!(cs.is_chain_empty(2));
        assert_eq!(cs.iter_chain(2).collect::<Vec<_>>(), Vec::<usize>::new());

        // Ensure modifying one chain didn't stomp another
        let s0 = cs.start_of_chain(0);
        let e0 = cs.end_of_chain(0);
        let s1 = cs.start_of_chain(1);
        let e1 = cs.end_of_chain(1);
        let s2 = cs.start_of_chain(2);
        let e2 = cs.end_of_chain(2);

        assert_eq!(cs.prev_node(e0), Some(2));
        assert_eq!(cs.prev_node(e1), Some(7));
        assert_eq!(cs.prev_node(e2), Some(s2)); // still empty

        // Spot-check sentinel typing
        for &x in &[s0, e0, s1, e1, s2, e2] {
            assert!(cs.is_sentinel_node(x));
        }
    }

    #[test]
    fn test_iter_chain_never_yields_sentinels() {
        let mut cs = ChainSet::new(4, 2);

        // Chain 0: [3]
        link_sequence(&mut cs, 0, &[3]);

        // Chain 1: [0, 2]
        link_sequence(&mut cs, 1, &[0, 2]);

        for chain in 0..cs.num_chains() {
            let s = cs.start_of_chain(chain);
            let e = cs.end_of_chain(chain);
            for n in cs.iter_chain(chain) {
                assert_ne!(n, s);
                assert_ne!(n, e);
                assert!(!cs.is_sentinel_node(n));
            }
        }
    }

    #[test]
    fn test_out_of_bounds_node_indices_return_none() {
        let cs = ChainSet::new(3, 2);
        let last_end = cs.end_of_chain(cs.num_chains() - 1);
        let oob = last_end + 1; // total size of the internal arrays

        assert_eq!(cs.next_node(oob), None);
        assert_eq!(cs.prev_node(oob), None);
    }

    #[test]
    fn test_set_next_steps_keep_chain_iter_valid() {
        let mut cs = ChainSet::new(5, 1);
        let s = cs.start_of_chain(0);
        let e = cs.end_of_chain(0);

        // Start empty: []
        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![]);

        // Insert first node: [2]
        cs.set_next(s, 2); // start -> 2 (isolates previous head 'end')
        cs.set_next(2, e); // 2 -> end
        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![2]);

        // Insert second after 2: [2, 4]
        // rewire 2 -> 4 (isolates head 'end' again), then 4 -> end
        cs.set_next(2, 4);
        cs.set_next(4, e);
        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![2, 4]);
        // Insert at head: set start -> 1, then 1 -> 2
        cs.set_next(s, 1);
        cs.set_next(1, 2);
        assert_eq!(cs.iter_chain(0).collect::<Vec<_>>(), vec![1, 2, 4]);
    }

    #[test]
    fn test_chain_ref_display() {
        use crate::state::chain_set::view::ChainRef;

        let mut cs = ChainSet::new(5, 1);
        link_sequence(&mut cs, 0, &[0, 3, 4]);
        let c0 = ChainRef::new(&cs, 0);
        assert_eq!(format!("{}", c0), "0->3->4");
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
    fn test_apply_rewire_auto_isolates_new_head() {
        let mut cs = ChainSet::new(5, 1);
        let s = cs.start_of_chain(0);
        let e = cs.end_of_chain(0);

        // Build start -> 0 -> 1 -> end
        cs.apply_rewire(ChainNextRewire::new(s, 0));
        cs.apply_rewire(ChainNextRewire::new(0, 1));
        cs.apply_rewire(ChainNextRewire::new(1, e));

        // Previously this would panic; now it auto-isolates `1` by cutting 0->1.
        cs.apply_rewire(ChainNextRewire::new(s, 1));

        // Now chain is start -> 1 -> end
        assert_eq!(collect_chain(&cs, 0), vec![1]);

        // Finish by ensuring back pointer consistency with an explicit confirm:
        cs.apply_rewire(ChainNextRewire::new(1, e));
        assert_eq!(cs.prev_node(e), Some(1));

        // And node 0 no longer has `s` as predecessor.
        assert_eq!(cs.prev_node(0), Some(0));
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
