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
    index::{ChainIndex, NodeIndex},
    view::ChainSetView,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSet {
    next: Vec<NodeIndex>,
    prev: Vec<NodeIndex>,
    start: Vec<NodeIndex>,
    end: Vec<NodeIndex>,
    num_nodes: usize,
}

impl ChainSet {
    #[inline]
    pub fn new(num_nodes: usize, num_chains: usize) -> Self {
        let total = num_nodes + 2 * num_chains;

        let mut next = vec![NodeIndex(0); total];
        let mut prev = vec![NodeIndex(0); total];

        for i in 0..num_nodes {
            next[i] = i.into();
            prev[i] = i.into();
        }

        let mut start = Vec::with_capacity(num_chains);
        let mut end = Vec::with_capacity(num_chains);

        for b in 0..num_chains {
            let s = num_nodes + 2 * b;
            let e = s + 1;
            start.push(s.into());
            end.push(e.into());

            next[s] = e.into();
            prev[s] = s.into();
            next[e] = e.into();
            prev[e] = s.into();
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
    pub fn num_total_nodes(&self) -> usize {
        self.next.len()
    }

    #[inline]
    pub fn next_slice(&self) -> &[NodeIndex] {
        &self.next
    }

    #[inline]
    pub fn previous_slice(&self) -> &[NodeIndex] {
        &self.prev
    }

    /// See rustdoc in your previous snippet — semantics unchanged; now uses `NodeIndex`.
    #[inline]
    pub(crate) fn set_next(&mut self, tail: NodeIndex, new_head: NodeIndex) {
        let tail_index = tail.get();
        let new_head_index = new_head.get();

        let num_total_nodes = self.num_total_nodes();

        debug_assert!(tail_index < num_total_nodes, "tail out of bounds");
        debug_assert!(new_head_index < num_total_nodes, "new_head out of bounds");

        assert!(
            !(self.is_head_node(tail) && new_head_index == tail_index),
            "set_next would create start->start self-loop (tail = {})",
            tail_index
        );
        assert!(
            !self.is_head_node(new_head),
            "new_head must not be a head sentinel (new_head = {})",
            new_head_index
        );
        assert!(
            !self.is_tail_node(tail),
            "tail must not be a tail sentinel (tail = {})",
            tail_index
        );

        let old_head = self.next[tail_index];

        if old_head == new_head {
            return;
        }

        if self.prev[old_head.get()] == tail {
            self.prev[old_head.get()] = old_head;
        }

        self.next[tail_index] = new_head;
        self.prev[new_head_index] = tail;

        debug_assert!(
            self.next[tail_index] == new_head,
            "post: next[tail] != new_head (tail={}, next[tail]={:?}, new_head={:?})",
            tail_index,
            self.next[tail_index],
            new_head
        );
        debug_assert!(
            self.prev[new_head_index] == tail,
            "post: prev[new_head] != tail (new_head={}, prev[new_head]={:?}, tail={:?})",
            new_head_index,
            self.prev[new_head_index],
            tail
        );

        debug_assert!(
            !self.is_head_node(self.next[tail_index]),
            "no edges may point to a head sentinel (tail={})",
            tail_index
        );
    }

    #[inline]
    pub(crate) fn apply_rewire(&mut self, rewire: ChainNextRewire) {
        self.set_next(rewire.tail(), rewire.successor());
    }

    #[inline]
    pub fn apply_delta(&mut self, delta: ChainSetDelta) {
        for &r in delta.rewires() {
            self.apply_rewire(r);
        }
    }

    #[inline(always)]
    fn is_head_node(&self, node: NodeIndex) -> bool {
        let u = node.get();
        u >= self.num_nodes && ((u - self.num_nodes) & 1) == 0
    }

    #[inline(always)]
    fn is_tail_node(&self, node: NodeIndex) -> bool {
        let u = node.get();
        u >= self.num_nodes && ((u - self.num_nodes) & 1) == 1
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
    fn start_of_chain(&self, chain: ChainIndex) -> NodeIndex {
        debug_assert!(chain.get() < self.num_chains());
        self.start[chain.get()]
    }

    #[inline]
    fn end_of_chain(&self, chain: ChainIndex) -> NodeIndex {
        debug_assert!(chain.get() < self.num_chains());
        self.end[chain.get()]
    }

    #[inline]
    fn next_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        let u = node.get();
        let num_total_nodes = self.num_total_nodes();

        if u >= num_total_nodes {
            return None;
        }

        Some(self.next[u])
    }

    #[inline]
    fn prev_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        let u = node.get();
        let num_total_nodes = self.num_total_nodes();

        if u >= num_total_nodes {
            return None;
        }

        Some(self.prev[u])
    }

    #[inline]
    fn is_sentinel_node(&self, node: NodeIndex) -> bool {
        node.get() >= self.num_nodes
    }

    #[inline]
    fn is_head_node(&self, node: NodeIndex) -> bool {
        self.is_head_node(node)
    }

    #[inline]
    fn is_tail_node(&self, node: NodeIndex) -> bool {
        self.is_tail_node(node)
    }

    #[inline]
    fn is_node_unperformed(&self, node: NodeIndex) -> bool {
        debug_assert!(node.get() < self.num_nodes());
        self.next[node.get()] == node && self.prev[node.get()] == node
    }

    #[inline]
    fn is_chain_empty(&self, chain: ChainIndex) -> bool {
        debug_assert!(chain.get() < self.num_chains());

        let num_total_nodes = self.num_total_nodes();
        let start = self.start[chain.get()];
        let end = self.end[chain.get()];

        debug_assert!(start.get() < num_total_nodes);
        debug_assert!(end.get() < num_total_nodes);

        self.next[start.get()] == end && self.prev[end.get()] == start
    }

    #[inline]
    fn chain_of_node(&self, node: NodeIndex) -> Option<ChainIndex> {
        if self.is_sentinel_node(node) || node.get() >= self.num_nodes {
            return None;
        }
        let mut cur = node;
        let mut steps_left = self.num_total_nodes();
        while steps_left > 0 {
            if self.is_head_node(cur) {
                let cid = (cur.get() - self.num_nodes) >> 1;
                return Some(ChainIndex(cid));
            }
            cur = self.prev[cur.get()];
            steps_left -= 1;
        }
        None
    }

    #[inline]
    fn position_in_chain(&self, node: NodeIndex) -> Option<usize> {
        if self.is_sentinel_node(node) || node.get() >= self.num_nodes {
            return None;
        }
        let mut cur = node;
        let mut pos = 0usize;
        let mut steps_left = self.num_total_nodes();
        while steps_left > 0 {
            if self.is_head_node(cur) {
                return Some(pos);
            }
            cur = self.prev[cur.get()];
            pos += 1;
            steps_left -= 1;
        }
        None
    }

    #[inline]
    fn iter_chain(&self, chain: ChainIndex) -> Self::NodeIter<'_> {
        debug_assert!(chain.get() < self.num_chains());

        let num_total_nodes = self.num_total_nodes();
        let start = self.start[chain.get()];
        let end = self.end[chain.get()];

        debug_assert!(start.get() < num_total_nodes);
        debug_assert!(end.get() < num_total_nodes);

        ChainIter::new(&self.next, self.next[start.get()], end)
    }

    type NodeIter<'a> = ChainIter<'a>;
}

#[derive(Debug, Clone)]
pub struct ChainIter<'slice> {
    next: &'slice [NodeIndex],
    current: NodeIndex,
    end: NodeIndex,
    steps_left: usize,
}

impl<'slice> ChainIter<'slice> {
    #[inline]
    fn new(next: &'slice [NodeIndex], start: NodeIndex, end: NodeIndex) -> Self {
        Self {
            next,
            current: start,
            end,
            steps_left: next.len(),
        }
    }
}

impl<'slice> Iterator for ChainIter<'slice> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.end || self.steps_left == 0 {
            return None;
        }
        self.steps_left -= 1;
        let out = self.current;
        self.current = self.next[self.current.get()];
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain_set::index::{ChainIndex, NodeIndex};
    use crate::state::chain_set::view::ChainRef;

    fn link_sequence(cs: &mut ChainSet, chain: ChainIndex, nodes: &[NodeIndex]) {
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
    fn collect_chain(cs: &ChainSet, chain: ChainIndex) -> Vec<NodeIndex> {
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
        let s0 = cs.start_of_chain(ChainIndex(0));
        let e0 = cs.end_of_chain(ChainIndex(0));
        assert!(cs.is_sentinel_node(s0));
        assert!(cs.is_sentinel_node(e0));
        assert!(cs.is_head_node(s0));
        assert!(cs.is_tail_node(e0));
        assert!(cs.is_chain_empty(ChainIndex(0)));
        assert_eq!(cs.next_node(s0), Some(e0));
        assert_eq!(cs.prev_node(e0), Some(s0));

        // Chain 1 sentinels
        let s1 = cs.start_of_chain(ChainIndex(1));
        let e1 = cs.end_of_chain(ChainIndex(1));
        assert!(cs.is_sentinel_node(s1));
        assert!(cs.is_sentinel_node(e1));
        assert!(cs.is_head_node(s1));
        assert!(cs.is_tail_node(e1));
        assert!(cs.is_chain_empty(ChainIndex(1)));
        assert_eq!(cs.next_node(s1), Some(e1));
        assert_eq!(cs.prev_node(e1), Some(s1));

        // Regular nodes are not sentinels, initially "unperformed"
        for n in 0..num_nodes {
            let n = NodeIndex(n);
            assert!(!cs.is_sentinel_node(n));
            assert!(cs.is_node_unperformed(n));
        }

        // Iteration over empty chains yields nothing
        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            Vec::<NodeIndex>::new()
        );
        assert_eq!(
            cs.iter_chain(ChainIndex(1)).collect::<Vec<_>>(),
            Vec::<NodeIndex>::new()
        );
    }

    #[test]
    fn test_link_single_node_into_chain() {
        let num_nodes = 3;
        let num_chains = 1;
        let mut cs = ChainSet::new(num_nodes, num_chains);

        let s = cs.start_of_chain(ChainIndex(0));
        let e = cs.end_of_chain(ChainIndex(0));

        // Link node 0 into chain 0: start -> 0 -> end
        cs.set_next(s, NodeIndex(0));
        cs.set_next(NodeIndex(0), e);

        assert!(!cs.is_chain_empty(ChainIndex(0)));
        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            vec![NodeIndex(0)]
        );

        // Node 0 should no longer be "unperformed"
        assert!(!cs.is_node_unperformed(NodeIndex(0)));
        // Other nodes remain unperformed
        assert!(cs.is_node_unperformed(NodeIndex(1)));
        assert!(cs.is_node_unperformed(NodeIndex(2)));

        // Structural checks
        assert_eq!(cs.next_node(s), Some(NodeIndex(0)));
        assert_eq!(cs.prev_node(NodeIndex(0)), Some(s));
        assert_eq!(cs.next_node(NodeIndex(0)), Some(e));
        assert_eq!(cs.prev_node(e), Some(NodeIndex(0)));
    }

    #[test]
    fn test_link_multiple_nodes_ordering() {
        let num_nodes = 6;
        let num_chains = 1;
        let mut cs = ChainSet::new(num_nodes, num_chains);

        // Build: start -> 2 -> 4 -> 1 -> end
        link_sequence(
            &mut cs,
            ChainIndex(0),
            &[NodeIndex(2), NodeIndex(4), NodeIndex(1)],
        );

        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            vec![NodeIndex(2), NodeIndex(4), NodeIndex(1)]
        );

        let s = cs.start_of_chain(ChainIndex(0));
        let e = cs.end_of_chain(ChainIndex(0));

        // Check local links
        assert_eq!(cs.next_node(s), Some(NodeIndex(2)));
        assert_eq!(cs.prev_node(NodeIndex(2)), Some(s));
        assert_eq!(cs.next_node(NodeIndex(2)), Some(NodeIndex(4)));
        assert_eq!(cs.prev_node(NodeIndex(4)), Some(NodeIndex(2)));
        assert_eq!(cs.next_node(NodeIndex(4)), Some(NodeIndex(1)));
        assert_eq!(cs.prev_node(NodeIndex(1)), Some(NodeIndex(4)));
        assert_eq!(cs.next_node(NodeIndex(1)), Some(e));
        assert_eq!(cs.prev_node(e), Some(NodeIndex(1)));

        // Nodes on the chain are not "unperformed"
        for &n in &[NodeIndex(2), NodeIndex(4), NodeIndex(1)] {
            assert!(!cs.is_node_unperformed(n));
        }
        // Nodes not on the chain remain unperformed
        for &n in &[NodeIndex(0), NodeIndex(3), NodeIndex(5)] {
            assert!(cs.is_node_unperformed(n));
        }
    }

    #[test]
    fn test_multiple_chains_are_independent() {
        let num_nodes = 10;
        let num_chains = 3;
        let mut cs = ChainSet::new(num_nodes, num_chains);

        // Chain 0: [0, 1, 2]
        link_sequence(
            &mut cs,
            ChainIndex(0),
            &[NodeIndex(0), NodeIndex(1), NodeIndex(2)],
        );

        // Chain 1: [7]
        link_sequence(&mut cs, ChainIndex(1), &[NodeIndex(7)]);

        // Chain 2: []
        // leave empty

        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            vec![NodeIndex(0), NodeIndex(1), NodeIndex(2)]
        );
        assert_eq!(
            cs.iter_chain(ChainIndex(1)).collect::<Vec<_>>(),
            vec![NodeIndex(7)]
        );
        assert!(cs.is_chain_empty(ChainIndex(2)));
        assert_eq!(
            cs.iter_chain(ChainIndex(2)).collect::<Vec<_>>(),
            Vec::<NodeIndex>::new()
        );

        // Ensure modifying one chain didn't stomp another
        let s0 = cs.start_of_chain(ChainIndex(0));
        let e0 = cs.end_of_chain(ChainIndex(0));
        let s1 = cs.start_of_chain(ChainIndex(1));
        let e1 = cs.end_of_chain(ChainIndex(1));
        let s2 = cs.start_of_chain(ChainIndex(2));
        let e2 = cs.end_of_chain(ChainIndex(2));

        assert_eq!(cs.prev_node(e0), Some(NodeIndex(2)));
        assert_eq!(cs.prev_node(e1), Some(NodeIndex(7)));
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
        link_sequence(&mut cs, ChainIndex(0), &[NodeIndex(3)]);

        // Chain 1: [0, 2]
        link_sequence(&mut cs, ChainIndex(1), &[NodeIndex(0), NodeIndex(2)]);

        for chain in 0..cs.num_chains() {
            let ci = ChainIndex(chain);
            let s = cs.start_of_chain(ci);
            let e = cs.end_of_chain(ci);
            for n in cs.iter_chain(ci) {
                assert_ne!(n, s);
                assert_ne!(n, e);
                assert!(!cs.is_sentinel_node(n));
            }
        }
    }

    #[test]
    fn test_out_of_bounds_node_indices_return_none() {
        let cs = ChainSet::new(3, 2);
        let last_end = cs.end_of_chain(ChainIndex(cs.num_chains() - 1));
        let oob = NodeIndex(last_end.get() + 1); // total size of the internal arrays

        assert_eq!(cs.next_node(oob), None);
        assert_eq!(cs.prev_node(oob), None);
    }

    #[test]
    fn test_set_next_steps_keep_chain_iter_valid() {
        let mut cs = ChainSet::new(5, 1);
        let s = cs.start_of_chain(ChainIndex(0));
        let e = cs.end_of_chain(ChainIndex(0));

        // Start empty: []
        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            Vec::<NodeIndex>::new()
        );

        // Insert first node: [2]
        cs.set_next(s, NodeIndex(2)); // start -> 2 (isolates previous head 'end')
        cs.set_next(NodeIndex(2), e); // 2 -> end
        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            vec![NodeIndex(2)]
        );

        // Insert second after 2: [2, 4]
        // rewire 2 -> 4 (isolates head 'end' again), then 4 -> end
        cs.set_next(NodeIndex(2), NodeIndex(4));
        cs.set_next(NodeIndex(4), e);
        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            vec![NodeIndex(2), NodeIndex(4)]
        );
        // Insert at head: set start -> 1, then 1 -> 2
        cs.set_next(s, NodeIndex(1));
        cs.set_next(NodeIndex(1), NodeIndex(2));
        assert_eq!(
            cs.iter_chain(ChainIndex(0)).collect::<Vec<_>>(),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(4)]
        );
    }

    #[test]
    fn test_chain_ref_display() {
        let mut cs = ChainSet::new(5, 1);
        link_sequence(
            &mut cs,
            ChainIndex(0),
            &[NodeIndex(0), NodeIndex(3), NodeIndex(4)],
        );
        let c0 = ChainRef::new(&cs, ChainIndex(0));

        // NodeIndex implements Display as "NodeIndex{}" in your code,
        // so the rendered chain reflects that.
        assert_eq!(format!("{}", c0), "0->3->4");
    }

    #[test]
    fn test_apply_rewire_builds_chain_and_updates_prev_next() {
        let mut cs = ChainSet::new(6, 1);
        let s = cs.start_of_chain(ChainIndex(0));
        let e = cs.end_of_chain(ChainIndex(0));

        // Initially empty
        assert!(cs.is_chain_empty(ChainIndex(0)));
        assert_eq!(collect_chain(&cs, ChainIndex(0)), vec![]);

        // start -> 2
        // ChainNextRewire still takes usize; pass `.get()`
        cs.apply_rewire(ChainNextRewire::new(s, NodeIndex(2)));
        // 2 -> end
        cs.apply_rewire(ChainNextRewire::new(NodeIndex(2), e));

        assert!(!cs.is_chain_empty(ChainIndex(0)));
        assert_eq!(collect_chain(&cs, ChainIndex(0)), vec![NodeIndex(2)]);

        // Pointers are consistent
        assert_eq!(cs.next_node(s), Some(NodeIndex(2)));
        assert_eq!(cs.prev_node(NodeIndex(2)), Some(s));
        assert_eq!(cs.next_node(NodeIndex(2)), Some(e));
        assert_eq!(cs.prev_node(e), Some(NodeIndex(2)));

        // Node 2 is no longer "unperformed"
        assert!(!cs.is_node_unperformed(NodeIndex(2)));

        // Add another: 2 -> 4, then 4 -> end
        cs.apply_rewire(ChainNextRewire::new(NodeIndex(2), NodeIndex(4)));
        cs.apply_rewire(ChainNextRewire::new(NodeIndex(4), e));

        assert_eq!(
            collect_chain(&cs, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(4)]
        );

        // Old head 'end' was detached when 2->4 was applied; re-attached by 4->end
        assert_eq!(cs.prev_node(e), Some(NodeIndex(4)));
    }

    #[test]
    fn test_apply_rewire_auto_isolates_new_head() {
        let mut cs = ChainSet::new(5, 1);
        let s = cs.start_of_chain(ChainIndex(0));
        let e = cs.end_of_chain(ChainIndex(0));

        // Build start -> 0 -> 1 -> end
        cs.apply_rewire(ChainNextRewire::new(s, NodeIndex(0)));
        cs.apply_rewire(ChainNextRewire::new(NodeIndex(0), NodeIndex(1)));
        cs.apply_rewire(ChainNextRewire::new(NodeIndex(1), e));

        // Previously this would panic; now it auto-isolates `1` by cutting 0->1.
        cs.apply_rewire(ChainNextRewire::new(s, NodeIndex(1)));

        // Now chain is start -> 1 -> end
        assert_eq!(collect_chain(&cs, ChainIndex(0)), vec![NodeIndex(1)]);

        // Finish by ensuring back pointer consistency with an explicit confirm:
        cs.apply_rewire(ChainNextRewire::new(NodeIndex(1), e));
        assert_eq!(cs.prev_node(e), Some(NodeIndex(1)));

        // And node 0 no longer has `s` as predecessor.
        assert_eq!(cs.prev_node(NodeIndex(0)), Some(NodeIndex(0)));
    }

    #[test]
    fn test_apply_delta_applies_all_rewires_single_chain() {
        let mut cs = ChainSet::new(8, 1);
        let s = cs.start_of_chain(ChainIndex(0));
        let e = cs.end_of_chain(ChainIndex(0));

        let mut delta = ChainSetDelta::new();
        // Build: start -> 3 -> 5 -> end
        delta.push_rewire(ChainNextRewire::new(s, NodeIndex(3)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(3), NodeIndex(5)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(5), e));

        cs.apply_delta(delta);

        assert_eq!(
            collect_chain(&cs, ChainIndex(0)),
            vec![NodeIndex(3), NodeIndex(5)]
        );

        // Pointers consistent
        assert_eq!(cs.prev_node(NodeIndex(3)), Some(s));
        assert_eq!(cs.next_node(NodeIndex(3)), Some(NodeIndex(5)));
        assert_eq!(cs.prev_node(NodeIndex(5)), Some(NodeIndex(3)));
        assert_eq!(cs.next_node(NodeIndex(5)), Some(e));

        // Unperformed flags
        assert!(!cs.is_node_unperformed(NodeIndex(3)));
        assert!(!cs.is_node_unperformed(NodeIndex(5)));
        // Others remain unperformed
        for &n in &[
            NodeIndex(0),
            NodeIndex(1),
            NodeIndex(2),
            NodeIndex(4),
            NodeIndex(6),
            NodeIndex(7),
        ] {
            assert!(cs.is_node_unperformed(n));
        }
    }

    #[test]
    fn test_apply_delta_multiple_chains() {
        let mut cs = ChainSet::new(10, 2);
        let s0 = cs.start_of_chain(ChainIndex(0));
        let e0 = cs.end_of_chain(ChainIndex(0));
        let s1 = cs.start_of_chain(ChainIndex(1));
        let e1 = cs.end_of_chain(ChainIndex(1));

        let mut delta = ChainSetDelta::with_capacity(6);
        // Chain 0: start0 -> 2 -> 9 -> end0
        delta.push_rewire(ChainNextRewire::new(s0, NodeIndex(2)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(2), NodeIndex(9)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(9), e0));

        // Chain 1: start1 -> 0 -> end1
        delta.push_rewire(ChainNextRewire::new(s1, NodeIndex(0)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(0), e1));

        cs.apply_delta(delta);

        assert_eq!(
            collect_chain(&cs, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(9)]
        );
        // was ChainIndex(2) — out of range for 2 chains
        assert_eq!(collect_chain(&cs, ChainIndex(1)), vec![NodeIndex(0)]);

        // Independence checks
        assert_eq!(cs.prev_node(e0), Some(NodeIndex(9)));
        assert_eq!(cs.prev_node(e1), Some(NodeIndex(0)));
    }

    #[test]
    fn test_apply_delta_is_noop_for_empty_delta() {
        let mut cs = ChainSet::new(4, 1);
        let before: Vec<NodeIndex> = collect_chain(&cs, ChainIndex(0));
        let delta = ChainSetDelta::new();

        cs.apply_delta(delta);

        let after: Vec<NodeIndex> = collect_chain(&cs, ChainIndex(0));
        assert_eq!(before, after);
        assert!(cs.is_chain_empty(ChainIndex(0)));
    }

    #[test]
    fn test_chain_of_node_and_position_in_chain_basic() {
        let mut cs = ChainSet::new(8, 2);

        // Chain 0: [2, 4, 1]
        link_sequence(
            &mut cs,
            ChainIndex(0),
            &[NodeIndex(2), NodeIndex(4), NodeIndex(1)],
        );

        // Chain 1: [5]
        link_sequence(&mut cs, ChainIndex(1), &[NodeIndex(5)]);

        // On-chain nodes report correct chain and position (position is 1-based from head)
        assert_eq!(cs.chain_of_node(NodeIndex(2)), Some(ChainIndex(0)));
        assert_eq!(cs.position_in_chain(NodeIndex(2)), Some(1));

        assert_eq!(cs.chain_of_node(NodeIndex(4)), Some(ChainIndex(0)));
        assert_eq!(cs.position_in_chain(NodeIndex(4)), Some(2));

        assert_eq!(cs.chain_of_node(NodeIndex(1)), Some(ChainIndex(0)));
        assert_eq!(cs.position_in_chain(NodeIndex(1)), Some(3));

        assert_eq!(cs.chain_of_node(NodeIndex(5)), Some(ChainIndex(1)));
        assert_eq!(cs.position_in_chain(NodeIndex(5)), Some(1));

        // Unperformed nodes: no chain, no position
        for &n in &[NodeIndex(0), NodeIndex(3), NodeIndex(6), NodeIndex(7)] {
            assert_eq!(
                cs.chain_of_node(n),
                None,
                "node {:?} should have no chain",
                n
            );
            assert_eq!(
                cs.position_in_chain(n),
                None,
                "node {:?} should have no position",
                n
            );
        }

        // Sentinel nodes must return None
        let s0 = cs.start_of_chain(ChainIndex(0));
        let e0 = cs.end_of_chain(ChainIndex(0));
        let s1 = cs.start_of_chain(ChainIndex(1));
        let e1 = cs.end_of_chain(ChainIndex(1));
        for &x in &[s0, e0, s1, e1] {
            assert!(cs.is_sentinel_node(x));
            assert_eq!(cs.chain_of_node(x), None);
            assert_eq!(cs.position_in_chain(x), None);
        }

        // Out-of-bounds nodes must return None as well
        let last_end = cs.end_of_chain(ChainIndex(cs.num_chains() - 1));
        let oob = NodeIndex(last_end.get() + 1);
        assert_eq!(cs.chain_of_node(oob), None);
        assert_eq!(cs.position_in_chain(oob), None);
    }

    #[test]
    fn test_chain_of_node_and_position_in_chain_multiple_nodes() {
        let mut cs = ChainSet::new(10, 1);

        // Chain 0: [0, 3, 9, 4]
        link_sequence(
            &mut cs,
            ChainIndex(0),
            &[NodeIndex(0), NodeIndex(3), NodeIndex(9), NodeIndex(4)],
        );

        // Positions should be 1, 2, 3, 4 respectively
        let expected = &[
            (NodeIndex(0), 1usize),
            (NodeIndex(3), 2usize),
            (NodeIndex(9), 3usize),
            (NodeIndex(4), 4usize),
        ];
        for &(n, pos) in expected {
            assert_eq!(cs.chain_of_node(n), Some(ChainIndex(0)));
            assert_eq!(
                cs.position_in_chain(n),
                Some(pos),
                "node {:?} should be at position {}",
                n,
                pos
            );
        }

        // Nodes not present remain None
        for &n in &[
            NodeIndex(1),
            NodeIndex(2),
            NodeIndex(5),
            NodeIndex(6),
            NodeIndex(7),
            NodeIndex(8),
        ] {
            assert_eq!(cs.chain_of_node(n), None);
            assert_eq!(cs.position_in_chain(n), None);
        }
    }
}
