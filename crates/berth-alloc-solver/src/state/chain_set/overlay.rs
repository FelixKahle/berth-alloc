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
    base::ChainSet,
    delta::ChainSetDelta,
    index::{ChainIndex, NodeIndex},
    view::ChainSetView,
};

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
    fn next_of(&self, node: NodeIndex) -> NodeIndex {
        if let Some(nxt) = self.delta.next_override_for_tail(node) {
            nxt
        } else {
            self.base.next_slice()[node.get()]
        }
    }

    #[inline]
    fn prev_of(&self, node: NodeIndex) -> NodeIndex {
        if let Some(prev) = self.delta.prev_override_for_head(node) {
            prev
        } else {
            self.base.previous_slice()[node.get()]
        }
    }

    #[inline]
    pub fn earliest_impacted_on_chain(&self, chain: ChainIndex) -> Option<NodeIndex> {
        let head = self.start_of_chain(chain);
        let first = {
            let n = self.next_of(head);
            if self.is_sentinel_node(n) {
                return None;
            } else {
                n
            }
        };

        if self.delta.is_node_touched(first) {
            return Some(first);
        }
        if self.delta.is_tail_overridden(first) {
            let succ = self.next_of(first);
            return if self.is_sentinel_node(succ) {
                Some(first)
            } else {
                Some(succ)
            };
        }

        let mut prev = first;
        loop {
            let succ = self.next_of(prev);
            if self.is_sentinel_node(succ) {
                break;
            }
            let cur = succ;

            if self.delta.is_tail_overridden(prev) || self.delta.is_node_touched(cur) {
                return Some(cur);
            }
            prev = cur;
        }

        Some(first)
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
    fn start_of_chain(&self, chain: ChainIndex) -> NodeIndex {
        self.base.start_of_chain(chain)
    }

    #[inline]
    fn end_of_chain(&self, chain: ChainIndex) -> NodeIndex {
        self.base.end_of_chain(chain)
    }

    #[inline]
    fn next_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        if node.get() >= self.total_nodes() {
            return None;
        }
        Some(self.next_of(node))
    }

    #[inline]
    fn prev_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        if node.get() >= self.total_nodes() {
            return None;
        }
        Some(self.prev_of(node))
    }

    #[inline]
    fn is_sentinel_node(&self, node: NodeIndex) -> bool {
        node.get() >= self.num_nodes()
    }

    #[inline]
    fn is_head_node(&self, node: NodeIndex) -> bool {
        let n = node.get();
        n >= self.num_nodes() && ((n - self.num_nodes()) & 1) == 0
    }

    #[inline]
    fn is_tail_node(&self, node: NodeIndex) -> bool {
        let n = node.get();
        n >= self.num_nodes() && ((n - self.num_nodes()) & 1) == 1
    }

    #[inline]
    fn is_node_unperformed(&self, node: NodeIndex) -> bool {
        debug_assert!(node.get() < self.num_nodes());
        self.next_of(node) == node && self.prev_of(node) == node
    }

    #[inline]
    fn is_chain_empty(&self, chain: ChainIndex) -> bool {
        debug_assert!(chain.get() < self.num_chains());
        let s = self.start_of_chain(chain);
        let e = self.end_of_chain(chain);
        self.next_of(s) == e && self.prev_of(e) == s
    }

    #[inline]
    fn iter_chain(&self, chain: ChainIndex) -> Self::NodeIter<'_> {
        debug_assert!(chain.get() < self.num_chains());

        let start = self.start_of_chain(chain);
        let end = self.end_of_chain(chain);
        ChainSetOverlayIter {
            overlay: self,
            current: self.next_of(start),
            end,
            steps_left: self.total_nodes(),
        }
    }

    #[inline]
    fn chain_of_node(&self, node: NodeIndex) -> Option<ChainIndex> {
        if self.is_sentinel_node(node) || node.get() >= self.num_nodes() {
            return None;
        }

        let mut cur = node;
        let mut steps_left = self.total_nodes();
        while steps_left > 0 {
            if self.is_head_node(cur) {
                let cid = (cur.get() - self.num_nodes()) >> 1;
                if cid < self.num_chains() {
                    return Some(ChainIndex(cid));
                } else {
                    return None;
                }
            }
            cur = self.prev_of(cur);
            steps_left -= 1;
        }
        None
    }

    #[inline]
    fn position_in_chain(&self, node: NodeIndex) -> Option<usize> {
        if self.is_sentinel_node(node) || node.get() >= self.num_nodes() {
            return None;
        }

        let mut cur = node;
        let mut pos = 0usize;
        let mut steps_left = self.total_nodes();
        while steps_left > 0 {
            if self.is_head_node(cur) {
                return Some(pos);
            }
            cur = self.prev_of(cur);
            pos += 1;
            steps_left -= 1;
        }
        None
    }

    type NodeIter<'a>
        = ChainSetOverlayIter<'a>
    where
        Self: 'a;
}

pub struct ChainSetOverlayIter<'a> {
    overlay: &'a ChainSetOverlay<'a, 'a>,
    current: NodeIndex,
    end: NodeIndex,
    steps_left: usize,
}

impl<'a> Iterator for ChainSetOverlayIter<'a> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.end || self.steps_left == 0 {
            return None;
        }
        self.steps_left -= 1;
        let out = self.current;
        self.current = self.overlay.next_of(self.current);
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain_set::{
        delta::{ChainNextRewire, ChainSetDelta},
        index::{ChainIndex, NodeIndex},
        view::ChainSetView,
    };

    // Helper that collects a chain into a Vec for easy assertions.
    fn collect_chain(cs: &ChainSet, chain: ChainIndex) -> Vec<NodeIndex> {
        cs.iter_chain(chain).collect::<Vec<_>>()
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
        for &n in &[0, 1, 2, 4, 6, 7] {
            assert!(cs.is_node_unperformed(NodeIndex(n)));
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
    fn test_overlay_chain_of_node_and_position_basic() {
        use super::ChainSetOverlay;

        let base = ChainSet::new(8, 2);
        let s0 = base.start_of_chain(ChainIndex(0));
        let e0 = base.end_of_chain(ChainIndex(0));
        let s1 = base.start_of_chain(ChainIndex(1));
        let e1 = base.end_of_chain(ChainIndex(1));

        // Compose chains entirely via delta overrides:
        // Chain 0: [2, 4, 1]
        // Chain 1: [5]
        let mut delta = ChainSetDelta::new();
        delta.push_rewire(ChainNextRewire::new(s0, NodeIndex(2)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(2), NodeIndex(4)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(4), NodeIndex(1)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(1), e0));

        delta.push_rewire(ChainNextRewire::new(s1, NodeIndex(5)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(5), e1));

        let overlay = ChainSetOverlay::new(&base, &delta);

        // On-chain nodes report correct chain and position (1-based from head)
        assert_eq!(overlay.chain_of_node(NodeIndex(2)), Some(ChainIndex(0)));
        assert_eq!(overlay.position_in_chain(NodeIndex(2)), Some(1));

        assert_eq!(overlay.chain_of_node(NodeIndex(4)), Some(ChainIndex(0)));
        assert_eq!(overlay.position_in_chain(NodeIndex(4)), Some(2));

        assert_eq!(overlay.chain_of_node(NodeIndex(1)), Some(ChainIndex(0)));
        assert_eq!(overlay.position_in_chain(NodeIndex(1)), Some(3));

        assert_eq!(overlay.chain_of_node(NodeIndex(5)), Some(ChainIndex(1)));
        assert_eq!(overlay.position_in_chain(NodeIndex(5)), Some(1));

        // Unperformed nodes remain None
        for &n in &[NodeIndex(0), NodeIndex(3), NodeIndex(6), NodeIndex(7)] {
            assert_eq!(
                overlay.chain_of_node(n),
                None,
                "node {:?} should have no chain",
                n
            );
            assert_eq!(
                overlay.position_in_chain(n),
                None,
                "node {:?} should have no position",
                n
            );
        }

        // Sentinel nodes must return None
        for &x in &[s0, e0, s1, e1] {
            assert!(overlay.is_sentinel_node(x));
            assert_eq!(overlay.chain_of_node(x), None);
            assert_eq!(overlay.position_in_chain(x), None);
        }

        // Out-of-bounds index also yields None
        let last_end = base.end_of_chain(ChainIndex(base.num_chains() - 1));
        let oob = NodeIndex(last_end.get() + 1);
        assert_eq!(overlay.chain_of_node(oob), None);
        assert_eq!(overlay.position_in_chain(oob), None);
    }

    #[test]
    fn test_overlay_chain_of_node_and_position_multiple_nodes() {
        use super::ChainSetOverlay;

        let base = ChainSet::new(10, 1);
        let s = base.start_of_chain(ChainIndex(0));
        let e = base.end_of_chain(ChainIndex(0));

        // Chain 0 via delta: [0, 3, 9, 4]
        let mut delta = ChainSetDelta::new();
        delta.push_rewire(ChainNextRewire::new(s, NodeIndex(0)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(0), NodeIndex(3)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(3), NodeIndex(9)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(9), NodeIndex(4)));
        delta.push_rewire(ChainNextRewire::new(NodeIndex(4), e));

        let overlay = ChainSetOverlay::new(&base, &delta);

        // Positions should be 1..=4 respectively
        let expected = &[
            (NodeIndex(0), 1usize),
            (NodeIndex(3), 2usize),
            (NodeIndex(9), 3usize),
            (NodeIndex(4), 4usize),
        ];
        for &(n, pos) in expected {
            assert_eq!(overlay.chain_of_node(n), Some(ChainIndex(0)));
            assert_eq!(overlay.position_in_chain(n), Some(pos));
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
            assert_eq!(overlay.chain_of_node(n), None);
            assert_eq!(overlay.position_in_chain(n), None);
        }
    }

    #[test]
    fn test_earliest_none_on_empty_chain() {
        let base = ChainSet::new(8, 1); // empty chain 0
        let delta = ChainSetDelta::new(); // no overrides
        let ov = ChainSetOverlay::new(&base, &delta);

        assert_eq!(ov.earliest_impacted_on_chain(ChainIndex(0)), None);
    }

    /// Build a concrete chain in `base` by applying a delta (so the overlay can be clean/empty).
    fn build_chain(base: &mut ChainSet, chain: ChainIndex, nodes: &[NodeIndex]) {
        let s = base.start_of_chain(chain);
        let e = base.end_of_chain(chain);
        let mut d = ChainSetDelta::new();
        let mut prev = s;
        for &n in nodes {
            d.push_rewire(ChainNextRewire::new(prev, n));
            prev = n;
        }
        d.push_rewire(ChainNextRewire::new(prev, e));
        base.apply_delta(d);
    }

    #[test]
    fn test_earliest_falls_back_to_first_when_no_impact() {
        let mut base = ChainSet::new(16, 1);
        build_chain(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        let delta = ChainSetDelta::new(); // no changes
        let ov = ChainSetOverlay::new(&base, &delta);

        assert_eq!(
            ov.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(1))
        );
    }

    #[test]
    fn test_earliest_is_first_when_first_touched() {
        let mut base = ChainSet::new(16, 1);
        build_chain(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        let mut delta = ChainSetDelta::new();
        delta.touch_node(NodeIndex(1)); // touch first real node
        let ov = ChainSetOverlay::new(&base, &delta);

        assert_eq!(
            ov.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(1))
        );
    }

    #[test]
    fn earliest_successor_when_first_tail_changed_but_successor_unchanged() {
        // Build: s -> 1 -> 2 -> 3 -> e
        let mut base = ChainSet::new(16, 1);
        build_chain(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        // Mark the *edge* at 1 as changed but keep the same successor 2.
        // (This is how to flag "changed tail" without altering topology.)
        let mut delta = ChainSetDelta::new();
        delta.set_next(NodeIndex(1), NodeIndex(2));
        let ov = ChainSetOverlay::new(&base, &delta);

        assert_eq!(
            ov.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(2))
        );
    }

    #[test]
    fn test_earliest_middle_successor_when_middle_tail_changed() {
        // Build: s -> 1 -> 2 -> 3 -> 4 -> e
        let mut base = ChainSet::new(20, 1);
        build_chain(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        // Flag the edge at node 2 as changed but keep successor (3) the same.
        let mut delta = ChainSetDelta::new();
        delta.set_next(NodeIndex(2), NodeIndex(3));
        let ov = ChainSetOverlay::new(&base, &delta);

        assert_eq!(
            ov.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(3))
        );
    }

    #[test]
    fn test_earliest_respects_overlay_rewiring_of_first_edge() {
        // Build: s -> 1 -> 2 -> 3 -> e
        let mut base = ChainSet::new(16, 1);
        build_chain(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        // Rewire 1 -> 3 in the delta (edge changed and successor different).
        let mut delta = ChainSetDelta::new();
        delta.set_next(NodeIndex(1), NodeIndex(3));
        let ov = ChainSetOverlay::new(&base, &delta);

        // Now overlay traversal sees s -> 1 -> 3 ..., so earliest impacted is 3.
        assert_eq!(
            ov.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(3))
        );
    }

    #[test]
    fn test_earliest_detects_touched_non_first() {
        // Build: s -> 1 -> 2 -> 3 -> e
        let mut base = ChainSet::new(16, 1);
        build_chain(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        // Touch node 3 only
        let mut delta = ChainSetDelta::new();
        delta.touch_node(NodeIndex(3));
        let ov = ChainSetOverlay::new(&base, &delta);

        assert_eq!(
            ov.earliest_impacted_on_chain(ChainIndex(0)),
            Some(NodeIndex(3))
        );
    }
}
