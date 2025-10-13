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
    delta::{ChainNextRewire, ChainSetDelta},
    index::NodeIndex,
    overlay::ChainSetOverlay,
    view::ChainSetView,
};

#[derive(Debug)]
pub struct ChainSetDeltaBuilder<'a> {
    base: &'a ChainSet,
    delta: ChainSetDelta,
}

impl<'a> ChainSetDeltaBuilder<'a> {
    #[inline]
    pub fn new(base: &'a ChainSet) -> Self {
        Self {
            base,
            delta: ChainSetDelta::new(),
        }
    }

    #[inline]
    pub fn build(self) -> ChainSetDelta {
        self.delta
    }

    #[inline]
    pub fn delta(&self) -> &ChainSetDelta {
        &self.delta
    }

    #[inline]
    fn overlay(&self) -> ChainSetOverlay<'a, '_> {
        ChainSetOverlay::new(self.base, &self.delta)
    }

    #[inline]
    fn set_next(&mut self, tail: NodeIndex, succ: NodeIndex) {
        let num_total_nodes = self.base.num_total_nodes();

        debug_assert!(tail.get() < num_total_nodes, "tail oob");
        debug_assert!(succ.get() < num_total_nodes, "succ oob");

        assert!(
            !self.base.is_head_node(succ),
            "builder must never set a head sentinel as successor (succ={})",
            succ
        );

        assert!(
            !self.base.is_tail_node(tail),
            "builder must never modify a tail sentinel (tail={})",
            tail
        );

        if tail == succ {
            assert!(
                !self.base.is_sentinel_node(tail),
                "cannot isolate a sentinel (tail==succ=={})",
                tail
            );
        }
        if let Some(cur) = self.overlay().next_node(tail)
            && cur == succ
        {
            return;
        }

        self.delta.push_rewire(ChainNextRewire::new(tail, succ));
    }

    #[inline]
    fn succ(&self, node: NodeIndex) -> NodeIndex {
        self.overlay()
            .next_node(node)
            .expect("node out of bounds (succ)")
    }

    #[allow(dead_code)]
    #[inline]
    fn pred(&self, node: NodeIndex) -> NodeIndex {
        self.overlay()
            .prev_node(node)
            .expect("node out of bounds (pred)")
    }

    #[inline]
    fn is_sentinel(&self, x: NodeIndex) -> bool {
        self.base.is_sentinel_node(x)
    }

    #[inline(always)]
    fn detach_node(&mut self, node: NodeIndex) {
        if self.is_sentinel(node) {
            return;
        }
        let ov = self.overlay();
        let (p, s) = (
            ov.prev_node(node).expect("detach: pred oob"),
            ov.next_node(node).expect("detach: succ oob"),
        );
        if p == node && s == node {
            return;
        }
        self.set_next(p, s);
        self.set_next(node, node);
    }

    #[inline]
    pub fn insert_after(&mut self, prev: NodeIndex, node: NodeIndex) -> bool {
        if self.is_sentinel(node) {
            return false;
        }
        let old = self.succ(prev);
        self.detach_node(node);
        self.set_next(prev, node);
        self.set_next(node, old);
        true
    }

    #[inline]
    pub fn remove_after(&mut self, prev: NodeIndex) -> Option<NodeIndex> {
        let x = {
            let ov = self.overlay();
            match ov.next_node(prev) {
                Some(n) if !self.is_sentinel(n) => n,
                _ => return None,
            }
        };

        let succ_x = self.succ(x);
        self.set_next(prev, succ_x);
        self.set_next(x, x);
        Some(x)
    }

    #[inline]
    pub fn move_after(&mut self, dst_prev: NodeIndex, src_prev: NodeIndex) -> &mut Self {
        if dst_prev == src_prev {
            return self;
        }

        let dst_insert_prev = {
            let ov = self.overlay();
            if self.base.is_head_node(dst_prev) {
                // paired tail sentinel is +1 by layout; then find last actual
                let end = NodeIndex(dst_prev.get() + 1);
                ov.prev_node(end).expect("move_after: end oob")
            } else {
                dst_prev
            }
        };

        let (x, succ_x, old_dst) = {
            let ov = self.overlay();
            let x = match ov.next_node(src_prev) {
                Some(n) if !self.is_sentinel(n) => n,
                _ => return self,
            };
            let succ_x = ov.next_node(x).expect("move_after: x succ oob");
            let old_dst = ov
                .next_node(dst_insert_prev)
                .expect("move_after: dst_insert_prev oob");
            (x, succ_x, old_dst)
        };

        self.set_next(src_prev, succ_x);
        self.set_next(dst_insert_prev, x);
        self.set_next(x, old_dst);
        self
    }

    #[inline]
    pub fn move_block_after(
        &mut self,
        dst_prev: NodeIndex,
        src_prev: NodeIndex,
        last: NodeIndex,
    ) -> &mut Self {
        let (first, after_last, old_dst) = {
            let ov = self.overlay();
            let first = ov
                .next_node(src_prev)
                .expect("move_block_after: src_prev oob");
            if self.is_sentinel(first) {
                return self;
            }
            let after_last = ov.next_node(last).expect("move_block_after: last succ oob");
            let old_dst = ov
                .next_node(dst_prev)
                .expect("move_block_after: dst_prev oob");
            (first, after_last, old_dst)
        };

        if old_dst == first || dst_prev == last {
            return self;
        }

        self.set_next(src_prev, after_last);
        self.set_next(dst_prev, first);
        self.set_next(last, old_dst);
        self
    }

    #[inline]
    pub fn swap_adjacent_after(&mut self, prev: NodeIndex) -> &mut Self {
        let (a, b, tail) = {
            let ov = self.overlay();
            let a = match ov.next_node(prev) {
                Some(n) if !self.is_sentinel(n) => n,
                _ => return self,
            };
            let b = match ov.next_node(a) {
                Some(n) if !self.is_sentinel(n) => n,
                _ => return self,
            };
            let tail = ov.next_node(b).expect("swap_adjacent_after: b succ oob");
            (a, b, tail)
        };

        self.set_next(a, tail);
        self.set_next(prev, b);
        self.set_next(b, a);
        self
    }

    #[inline]
    pub fn swap_after(&mut self, p: NodeIndex, q: NodeIndex) -> &mut Self {
        if p == q {
            return self;
        }

        let (a, b) = {
            let ov = self.overlay();
            let a = ov.next_node(p).expect("swap_after: p oob");
            let b = ov.next_node(q).expect("swap_after: q oob");
            if self.is_sentinel(a) || self.is_sentinel(b) || a == b {
                return self;
            }
            (a, b)
        };

        if q == a {
            return self.swap_adjacent_after(p);
        }
        if p == b {
            return self.swap_adjacent_after(q);
        }

        let (a_next, b_next) = {
            let ov = self.overlay();
            (
                ov.next_node(a).expect("a_next oob"),
                ov.next_node(b).expect("b_next oob"),
            )
        };

        self.set_next(p, b);
        self.set_next(q, a);
        self.set_next(a, b_next);
        self.set_next(b, a_next);
        self
    }

    #[inline]
    pub fn two_opt(&mut self, p: NodeIndex, q: NodeIndex) -> &mut Self {
        let (a, b) = {
            let ov = self.overlay();
            let a = ov.next_node(p).expect("2-opt: p has no successor");
            let b = ov.next_node(q).expect("2-opt: q has no successor");
            if self.is_sentinel(a) || self.is_sentinel(b) || a == q {
                return self;
            }
            (a, b)
        };

        let mut path_to_reverse = Vec::new();
        let mut current = a;
        loop {
            path_to_reverse.push(current);
            if current == q {
                break;
            }
            current = self.succ(current);
            if path_to_reverse.len() > self.base.num_nodes() {
                return self;
            }
        }

        self.set_next(p, q);
        let mut last_node_in_reversed_segment = q;
        while let Some(node) = path_to_reverse.pop() {
            if node != q {
                self.set_next(last_node_in_reversed_segment, node);
                last_node_in_reversed_segment = node;
            }
        }
        self.set_next(a, b);
        self
    }

    #[inline]
    pub fn two_opt_star_intra(&mut self, head: NodeIndex, p: NodeIndex, q: NodeIndex) -> &mut Self {
        debug_assert!(self.base.is_head_node(head));

        if p == q {
            return self;
        }

        let (a, b, a_next, b_next) = {
            let ov = self.overlay();
            let a = ov.next_node(p).expect("p oob");
            let b = ov.next_node(q).expect("q oob");
            let a_next = ov.next_node(a).expect("a_next oob");
            let b_next = ov.next_node(b).expect("b_next oob");
            (a, b, a_next, b_next)
        };

        self.set_next(p, b);
        self.set_next(q, a);
        self.set_next(a, b_next);
        self.set_next(b, a_next);
        self
    }

    #[inline]
    pub fn splice_run_after(
        &mut self,
        dst_prev: NodeIndex,
        src_prev: NodeIndex,
        last: NodeIndex,
    ) -> &mut Self {
        self.move_block_after(dst_prev, src_prev, last)
    }

    #[inline]
    pub fn detach(&mut self, node: NodeIndex) -> &mut Self {
        self.detach_node(node);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain_set::{base::ChainSet, index::ChainIndex};

    fn link_sequence(cs: &mut ChainSet, chain: ChainIndex, nodes: &[NodeIndex]) {
        let s = cs.start_of_chain(chain);
        let e = cs.end_of_chain(chain);
        let mut tail = s;
        for &n in nodes {
            cs.set_next(tail, n);
            tail = n;
        }
        cs.set_next(tail, e);
    }

    fn collect(view: &impl ChainSetView, chain: ChainIndex) -> Vec<NodeIndex> {
        view.iter_chain(chain).collect()
    }

    #[test]
    fn test_insert_and_remove() {
        let mut base = ChainSet::new(8, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(3), NodeIndex(5)],
        ); // s -> 1 -> 3 -> 5 -> e
        let e = base.end_of_chain(ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        assert!(b.insert_after(NodeIndex(1), NodeIndex(2))); // s -> 1 -> 2 -> 3 -> 5 -> e
        let removed = b.remove_after(NodeIndex(3)); // removes 5, parks as 5->5
        assert_eq!(removed, Some(NodeIndex(5)));

        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );
        assert_eq!(base.prev_node(e), Some(NodeIndex(3)));
        assert_eq!(base.prev_node(NodeIndex(5)), Some(NodeIndex(5)));
        assert_eq!(base.next_node(NodeIndex(5)), Some(NodeIndex(5)));
    }

    #[test]
    fn test_move_between_chains() {
        let mut base = ChainSet::new(10, 2);
        // c0: [1,4,7] ; c1: [2]
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(4), NodeIndex(7)],
        );
        link_sequence(&mut base, ChainIndex(1), &[NodeIndex(2)]);

        let s0 = base.start_of_chain(ChainIndex(0));
        let s1 = base.start_of_chain(ChainIndex(1));

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_after(s1, s0); // move node after s0 (1) to after s1

        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(4), NodeIndex(7)]
        );
        assert_eq!(
            collect(&base, ChainIndex(1)),
            vec![NodeIndex(2), NodeIndex(1)]
        );
    }

    #[test]
    fn test_move_block_oropt_k2() {
        let mut base = ChainSet::new(12, 1);
        // s -> 1,2,3,4,5,6 -> e
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[
                NodeIndex(1),
                NodeIndex(2),
                NodeIndex(3),
                NodeIndex(4),
                NodeIndex(5),
                NodeIndex(6),
            ],
        );

        // Move block [2,3] (after prev=1, last=3) to after 5
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_block_after(NodeIndex(5), NodeIndex(1), NodeIndex(3));

        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![
                NodeIndex(1),
                NodeIndex(4),
                NodeIndex(5),
                NodeIndex(2),
                NodeIndex(3),
                NodeIndex(6)
            ]
        );
    }

    #[test]
    fn test_test_swap_adjacent_and_nonadjacent() {
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[
                NodeIndex(1),
                NodeIndex(3),
                NodeIndex(5),
                NodeIndex(7),
                NodeIndex(9),
            ],
        ); // s->1->3->5->7->9->e
        let s = base.start_of_chain(ChainIndex(0));

        // Adjacent swap 1 and 3
        let mut b1 = ChainSetDeltaBuilder::new(&base);
        b1.swap_adjacent_after(s);
        base.apply_delta(b1.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![
                NodeIndex(3),
                NodeIndex(1),
                NodeIndex(5),
                NodeIndex(7),
                NodeIndex(9)
            ]
        );

        // Non-adjacent swap successors of (s) and (1): currently succ(s)=3, succ(1)=5
        let mut b2 = ChainSetDeltaBuilder::new(&base);
        b2.swap_after(s, NodeIndex(1));
        base.apply_delta(b2.build());
        // Expected: s->5->1->3->7->9
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![
                NodeIndex(5),
                NodeIndex(1),
                NodeIndex(3),
                NodeIndex(7),
                NodeIndex(9)
            ]
        );
    }

    #[test]
    fn test_insert_after_in_same_chain_moves_existing_node() {
        // s->1->2->3->4->e, insert node 3 after 1 => s->1->3->2->4->e
        let mut base = ChainSet::new(8, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        assert!(b.insert_after(NodeIndex(1), NodeIndex(3)));
        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(3), NodeIndex(2), NodeIndex(4)]
        );
    }

    #[test]
    fn test_insert_after_into_other_chain() {
        // c0: [1,2], c1: [3]; insert 2 after 3 => c0:[1], c1:[3,2]
        let mut base = ChainSet::new(8, 2);
        link_sequence(&mut base, ChainIndex(0), &[NodeIndex(1), NodeIndex(2)]);
        link_sequence(&mut base, ChainIndex(1), &[NodeIndex(3)]);

        let s1 = base.start_of_chain(ChainIndex(1));

        let mut b = ChainSetDeltaBuilder::new(&base);
        assert!(b.insert_after(NodeIndex(3), NodeIndex(2)));
        base.apply_delta(b.build());

        assert_eq!(collect(&base, ChainIndex(0)), vec![NodeIndex(1)]);
        assert_eq!(
            collect(&base, ChainIndex(1)),
            vec![NodeIndex(3), NodeIndex(2)]
        );

        // Now insert 1 right after chain head of c1 (i.e., before 3)
        let mut b2 = ChainSetDeltaBuilder::new(&base);
        assert!(b2.insert_after(s1, NodeIndex(1)));
        base.apply_delta(b2.build());
        assert_eq!(
            collect(&base, ChainIndex(1)),
            vec![NodeIndex(1), NodeIndex(3), NodeIndex(2)]
        );
    }

    #[test]
    fn test_insert_after_rejects_sentinel() {
        let mut base = ChainSet::new(6, 1);
        link_sequence(&mut base, ChainIndex(0), &[NodeIndex(1), NodeIndex(2)]);
        let e = base.end_of_chain(ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        // inserting the tail sentinel is not allowed
        assert!(!b.insert_after(NodeIndex(1), e));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2)]
        );
    }

    #[test]
    fn test_remove_after_head_of_non_empty_chain() {
        let mut base = ChainSet::new(6, 1);
        link_sequence(&mut base, ChainIndex(0), &[NodeIndex(1), NodeIndex(2)]);
        let s = base.start_of_chain(ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        let removed = b.remove_after(s);
        assert_eq!(removed, Some(NodeIndex(1)));
        base.apply_delta(b.build());

        assert_eq!(collect(&base, ChainIndex(0)), vec![NodeIndex(2)]);
        // 1 must be isolated
        assert_eq!(base.next_node(NodeIndex(1)), Some(NodeIndex(1)));
        assert_eq!(base.prev_node(NodeIndex(1)), Some(NodeIndex(1)));
    }

    #[test]
    fn test_remove_after_returns_none_at_end() {
        let mut base = ChainSet::new(6, 1);
        link_sequence(&mut base, ChainIndex(0), &[NodeIndex(1), NodeIndex(2)]);
        // prev is '2' whose successor is tail sentinel
        let mut b = ChainSetDeltaBuilder::new(&base);
        let removed = b.remove_after(NodeIndex(2));
        assert_eq!(removed, None);
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2)]
        );
    }

    #[test]
    fn test_move_after_noop_when_src_prev_is_at_end() {
        let mut base = ChainSet::new(8, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        // src_prev=3 is followed by end => noop
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_after(NodeIndex(1), NodeIndex(3));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );
    }

    #[test]
    fn test_move_after_appends_when_dst_prev_is_head_sentinel_same_chain() {
        // Move the first node to the end by passing dst_prev=head sentinel
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        let s = base.start_of_chain(ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_after(s, s); // equal => noop by contract
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );

        // Now move node after s (1) to after current last (3)
        let mut b4 = ChainSetDeltaBuilder::new(&base);
        b4.move_after(NodeIndex(3), s);
        base.apply_delta(b4.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(3), NodeIndex(1)]
        );
    }

    #[test]
    fn test_move_after_between_chains_when_dst_is_empty_chain() {
        let mut base = ChainSet::new(10, 2);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        // chain 1 is empty
        let s0 = base.start_of_chain(ChainIndex(0));
        let s1 = base.start_of_chain(ChainIndex(1));

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_after(s1, s0); // move node 1 after head of empty chain => becomes first
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(3)]
        );
        assert_eq!(collect(&base, ChainIndex(1)), vec![NodeIndex(1)]);
    }

    #[test]
    fn test_move_block_after_singleton_is_move_after() {
        // s->1->2->3->4->e, move block [2] to after 4 => s->1->3->4->2->e
        let mut base = ChainSet::new(12, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_block_after(NodeIndex(4), NodeIndex(1), NodeIndex(2)); // block [2]
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(3), NodeIndex(4), NodeIndex(2)]
        );
    }

    #[test]
    fn test_move_block_after_middle_segment() {
        // s->1->2->3->4->5->e, move [2,3,4] to after 5 => s->1->5->2->3->4->e
        let mut base = ChainSet::new(16, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[
                NodeIndex(1),
                NodeIndex(2),
                NodeIndex(3),
                NodeIndex(4),
                NodeIndex(5),
            ],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_block_after(NodeIndex(5), NodeIndex(1), NodeIndex(4));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![
                NodeIndex(1),
                NodeIndex(5),
                NodeIndex(2),
                NodeIndex(3),
                NodeIndex(4)
            ]
        );
    }

    #[test]
    fn test_move_block_after_noop_when_src_prev_points_to_tail() {
        // src_prev is last node => first is sentinel => noop
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.move_block_after(NodeIndex(1), NodeIndex(3), NodeIndex(3)); // first = succ(3) = tail => noop
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );
    }

    #[test]
    fn test_swap_adjacent_after_on_last_pair() {
        // s->1->2->3->e, prev=2 => swap 3 with tail? No, successor is tail => noop
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.swap_adjacent_after(NodeIndex(2));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );

        // Now swap the first two: prev=s
        let s = base.start_of_chain(ChainIndex(0));
        let mut b2 = ChainSetDeltaBuilder::new(&base);
        b2.swap_adjacent_after(s);
        base.apply_delta(b2.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(1), NodeIndex(3)]
        );
    }

    #[test]
    fn test_swap_adjacent_after_noop_when_not_enough_nodes() {
        // Empty chain
        let mut base = ChainSet::new(6, 1);
        let s = base.start_of_chain(ChainIndex(0));
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.swap_adjacent_after(s);
        base.apply_delta(b.build());
        assert_eq!(collect(&base, ChainIndex(0)), Vec::<NodeIndex>::new());

        // Single node
        link_sequence(&mut base, ChainIndex(0), &[NodeIndex(1)]);
        let mut b2 = ChainSetDeltaBuilder::new(&base);
        b2.swap_adjacent_after(s);
        base.apply_delta(b2.build());
        assert_eq!(collect(&base, ChainIndex(0)), vec![NodeIndex(1)]);
    }

    #[test]
    fn test_swap_after_noop_when_p_eq_q_or_same_successor_or_sentinel() {
        let mut base = ChainSet::new(10, 1);
        let s = base.start_of_chain(ChainIndex(0));
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        // p==q
        let mut b1 = ChainSetDeltaBuilder::new(&base);
        b1.swap_after(NodeIndex(1), NodeIndex(1));
        base.apply_delta(b1.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );

        // identical successors
        let mut b2 = ChainSetDeltaBuilder::new(&base);
        b2.swap_after(s, s);
        base.apply_delta(b2.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );

        // a sentinel successor path (choose q=3 so succ(3)=tail)
        let mut b3 = ChainSetDeltaBuilder::new(&base);
        b3.swap_after(s, NodeIndex(3));
        base.apply_delta(b3.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3)]
        );
    }

    #[test]
    fn test_swap_after_adjacent_both_directions() {
        // p->a->b and q==a
        let mut base = ChainSet::new(12, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        ); // s->1->2->3->4->e
        let s = base.start_of_chain(ChainIndex(0));

        let mut b1 = ChainSetDeltaBuilder::new(&base);
        b1.swap_after(s, NodeIndex(1)); // q==a (1->2)
        base.apply_delta(b1.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(1), NodeIndex(3), NodeIndex(4)]
        );

        // Now the other adjacency direction: q->b->a and p==b
        // Build a fresh layout for clarity: s->5->6->7->e, swap_after with p=6 and q=5 gives q->b->a with p==b
        let mut base2 = ChainSet::new(12, 1);
        link_sequence(
            &mut base2,
            ChainIndex(0),
            &[NodeIndex(5), NodeIndex(6), NodeIndex(7)],
        );
        let mut b2 = ChainSetDeltaBuilder::new(&base2);
        b2.swap_after(NodeIndex(6), NodeIndex(5)); // p==b (6), q->b(6)->a(7)
        base2.apply_delta(b2.build());
        assert_eq!(
            collect(&base2, ChainIndex(0)),
            vec![NodeIndex(5), NodeIndex(7), NodeIndex(6)]
        );
    }

    #[test]
    fn test_swap_after_non_adjacent_general_case() {
        // s->1->3->5->7->e, swap successors of p=s (1) and q=3 (5)
        let mut base = ChainSet::new(14, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(3), NodeIndex(5), NodeIndex(7)],
        );
        let s = base.start_of_chain(ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.swap_after(s, NodeIndex(3));
        base.apply_delta(b.build());
        // Expected: s->5->3->1->7
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(5), NodeIndex(3), NodeIndex(1), NodeIndex(7)]
        );
    }

    #[test]
    fn test_two_opt_star_intra_basic() {
        // s->1->2->3->4->e, with p=1 (a=2), q=3 (b=4) => s->1->4->3->2->e
        let mut base = ChainSet::new(16, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        let s = base.start_of_chain(ChainIndex(0));
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.two_opt_star_intra(s, NodeIndex(1), NodeIndex(3));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(4), NodeIndex(3), NodeIndex(2)]
        );
    }

    #[test]
    fn test_splice_run_after_moves_suffix() {
        // s->1->2->3->4->e, splice run from src_prev=1..last=4 after dst_prev=1 (self)
        // This will effectively keep the same order (moving the whole tail after 1 right after 1)
        let mut base = ChainSet::new(16, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.splice_run_after(NodeIndex(1), NodeIndex(1), NodeIndex(4));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)]
        );
    }

    #[test]
    fn test_detach_isolate_middle_node() {
        // s->1->2->3->e, detach 2 => s->1->3->e, and 2 isolated
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.detach(NodeIndex(2));
        base.apply_delta(b.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(3)]
        );
        assert_eq!(base.next_node(NodeIndex(2)), Some(NodeIndex(2)));
        assert_eq!(base.prev_node(NodeIndex(2)), Some(NodeIndex(2)));
    }

    #[test]
    fn test_detach_head_and_tail_nodes() {
        // Detach head actual node
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );
        let mut b1 = ChainSetDeltaBuilder::new(&base);
        b1.detach(NodeIndex(1));
        base.apply_delta(b1.build());
        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(2), NodeIndex(3)]
        );

        // Detach tail actual node
        let mut b2 = ChainSetDeltaBuilder::new(&base);
        b2.detach(NodeIndex(3));
        base.apply_delta(b2.build());
        assert_eq!(collect(&base, ChainIndex(0)), vec![NodeIndex(2)]);
        assert_eq!(base.next_node(NodeIndex(3)), Some(NodeIndex(3)));
        assert_eq!(base.prev_node(NodeIndex(3)), Some(NodeIndex(3)));
    }

    #[test]
    fn test_detach_sentinel_is_noop() {
        let mut base = ChainSet::new(8, 1);
        let s = base.start_of_chain(ChainIndex(0));
        let e = base.end_of_chain(ChainIndex(0));
        link_sequence(&mut base, ChainIndex(0), &[NodeIndex(1), NodeIndex(2)]);

        let snapshot = collect(&base, ChainIndex(0));
        let mut b = ChainSetDeltaBuilder::new(&base);
        b.detach(s).detach(e);
        base.apply_delta(b.build());

        assert_eq!(collect(&base, ChainIndex(0)), snapshot);
    }

    #[test]
    fn test_composition_of_multiple_ops_respects_overlay() {
        // Start: s->1->2->3->4->e
        let mut base = ChainSet::new(16, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        let s = base.start_of_chain(ChainIndex(0));
        let mut b = ChainSetDeltaBuilder::new(&base);

        // 1. swap_adjacent_after(s): s->2->1->3->4->e
        // 2. move_after(4, 1): moves node after 1 (3) to after 4 => s->2->1->4->3->e
        // 3. insert_after(1, 2): moves 2 after 1 => s->1->2->4->3->e
        b.swap_adjacent_after(s)
            .move_after(NodeIndex(4), NodeIndex(1))
            .insert_after(NodeIndex(1), NodeIndex(2));

        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(2), NodeIndex(4), NodeIndex(3)]
        );
    }

    #[test]
    fn test_splice_run_after_between_chains() {
        // c0: [1,2,3,4], c1: [5], move block [2,3] after 5
        let mut base = ChainSet::new(16, 2);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );
        link_sequence(&mut base, ChainIndex(1), &[NodeIndex(5)]);

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.splice_run_after(NodeIndex(5), NodeIndex(1), NodeIndex(3));
        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![NodeIndex(1), NodeIndex(4)]
        );
        assert_eq!(
            collect(&base, ChainIndex(1)),
            vec![NodeIndex(5), NodeIndex(2), NodeIndex(3)]
        );
    }

    #[test]
    fn test_two_opt_basic_reversal() {
        // s->1->2->3->4->5->e, with p=1 (a=2), q=4 (b=5) => s->1->4->3->2->5->e
        let mut base = ChainSet::new(16, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[
                NodeIndex(1),
                NodeIndex(2),
                NodeIndex(3),
                NodeIndex(4),
                NodeIndex(5),
            ],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.two_opt(NodeIndex(1), NodeIndex(4));
        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![
                NodeIndex(1),
                NodeIndex(4),
                NodeIndex(3),
                NodeIndex(2),
                NodeIndex(5)
            ]
        );
    }

    #[test]
    fn test_two_opt_noop_when_adjacent() {
        // s->1->2->3->4->e, with p=1 (a=2), q=2 (b=3) => a==q => no-op
        let mut base = ChainSet::new(12, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3), NodeIndex(4)],
        );

        let snapshot = collect(&base, ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.two_opt(NodeIndex(1), NodeIndex(2));
        base.apply_delta(b.build());

        assert_eq!(collect(&base, ChainIndex(0)), snapshot);
    }

    #[test]
    fn test_two_opt_noop_when_q_successor_is_tail() {
        // s->1->2->3->e, with p=1 (a=2), q=3 (b=tail) => b is sentinel => no-op
        let mut base = ChainSet::new(10, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[NodeIndex(1), NodeIndex(2), NodeIndex(3)],
        );

        let snapshot = collect(&base, ChainIndex(0));

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.two_opt(NodeIndex(1), NodeIndex(3));
        base.apply_delta(b.build());

        assert_eq!(collect(&base, ChainIndex(0)), snapshot);
    }

    #[test]
    fn test_two_opt_reverse_long_segment() {
        // s->1->2->3->4->5->6->7->e, with p=1 (a=2), q=6 (b=7) => s->1->6->5->4->3->2->7->e
        let mut base = ChainSet::new(20, 1);
        link_sequence(
            &mut base,
            ChainIndex(0),
            &[
                NodeIndex(1),
                NodeIndex(2),
                NodeIndex(3),
                NodeIndex(4),
                NodeIndex(5),
                NodeIndex(6),
                NodeIndex(7),
            ],
        );

        let mut b = ChainSetDeltaBuilder::new(&base);
        b.two_opt(NodeIndex(1), NodeIndex(6));
        base.apply_delta(b.build());

        assert_eq!(
            collect(&base, ChainIndex(0)),
            vec![
                NodeIndex(1),
                NodeIndex(6),
                NodeIndex(5),
                NodeIndex(4),
                NodeIndex(3),
                NodeIndex(2),
                NodeIndex(7),
            ]
        );
    }
}
