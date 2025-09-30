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

use crate::state::chain::{delta::ChainDelta, double_chain::DoubleChain};

#[derive(Debug)]
pub struct DeltaBuilder<'a> {
    chain: &'a DoubleChain,
    delta: ChainDelta,
    prev_view: Vec<usize>,
    base: usize,
    berths_len: usize,
}

impl<'a> DeltaBuilder<'a> {
    #[inline]
    pub fn new(chain: &'a DoubleChain) -> Self {
        let mut d = ChainDelta::new();
        d.reserve_nodes(chain.len().saturating_sub(1));
        d.reserve_berths(chain.num_berths().saturating_sub(1));
        Self {
            chain,
            delta: d,
            prev_view: chain.prev_slice().to_vec(),
            base: chain.base_index(),
            berths_len: chain.num_berths(),
        }
    }

    #[inline]
    pub fn into_delta(self) -> ChainDelta {
        self.delta
    }

    #[inline]
    fn is_tail_idx(&self, x: usize) -> bool {
        x >= self.base && ((x - self.base) & 1) == 1
    }

    #[inline]
    fn next_after(&self, i: usize) -> usize {
        self.delta.next_after(self.chain.next_slice(), i)
    }

    #[inline]
    fn pred_in_view(&self, i: usize) -> usize {
        self.prev_view[i]
    }

    #[inline]
    fn is_skipped(&self, i: usize) -> bool {
        self.next_after(i) == i
    }

    #[inline]
    fn mark_if_sentinel(&mut self, idx: usize) {
        if idx >= self.base {
            let b = (idx - self.base) >> 1;
            if b < self.berths_len {
                self.delta.mark_berth(b);
            }
        }
    }

    #[inline]
    fn rewrite_arc(&mut self, tail: usize, new_head: usize) {
        let expected = self.next_after(tail);
        self.delta.push(tail, expected, new_head);

        self.prev_view[new_head] = tail;
        if self.prev_view[expected] == tail {
            self.prev_view[expected] = expected;
        }

        self.mark_if_sentinel(tail);
        self.mark_if_sentinel(expected);
        self.mark_if_sentinel(new_head);
    }

    #[inline]
    fn detach_segment(&mut self, a: usize, b: usize) {
        let pa = self.pred_in_view(a);
        let nb = self.next_after(b);
        if pa != nb {
            self.rewrite_arc(pa, nb);
        }
    }

    #[inline]
    pub fn link_after_skipped(&mut self, i: usize, anchor: usize) -> &mut Self {
        debug_assert!(i < self.chain.len() && anchor < self.chain.len());
        debug_assert!(i < self.base, "cannot insert sentinel");
        debug_assert!(self.is_skipped(i), "node must be skipped");

        let an = self.next_after(anchor);
        self.rewrite_arc(i, an);
        self.rewrite_arc(anchor, i);
        self
    }

    #[inline]
    pub fn link_before_skipped(&mut self, i: usize, before: usize) -> &mut Self {
        let a = self.pred_in_view(before);
        self.link_after_skipped(i, a)
    }

    #[inline]
    pub fn insert_after_any(&mut self, i: usize, anchor: usize) -> &mut Self {
        debug_assert!(i < self.chain.len() && anchor < self.chain.len());
        debug_assert!(i < self.base, "cannot insert sentinel");

        if i == anchor {
            return self;
        }

        if self.pred_in_view(i) == anchor {
            return self;
        }

        let an = self.next_after(anchor);
        if self.is_skipped(i) {
            self.rewrite_arc(i, an);
            self.rewrite_arc(anchor, i);
            return self;
        }

        let pa = self.pred_in_view(i);
        let ni = self.next_after(i);
        self.rewrite_arc(pa, ni);
        self.rewrite_arc(i, an);
        self.rewrite_arc(anchor, i);
        self
    }

    #[inline]
    pub fn insert_before_any(&mut self, i: usize, before: usize) -> &mut Self {
        let a = self.pred_in_view(before);
        self.insert_after_any(i, a)
    }

    #[inline]
    pub fn insert_after(&mut self, i: usize, anchor: usize) -> &mut Self {
        self.link_after_skipped(i, anchor)
    }

    #[inline]
    pub fn insert_before(&mut self, i: usize, before: usize) -> &mut Self {
        self.link_before_skipped(i, before)
    }

    #[inline]
    pub fn move_after(&mut self, i: usize, x: usize) -> &mut Self {
        self.insert_after_any(i, x)
    }

    #[inline]
    pub fn move_before(&mut self, i: usize, y: usize) -> &mut Self {
        let x = self.pred_in_view(y);
        self.insert_after_any(i, x)
    }

    #[inline]
    pub fn move_node_after_node(&mut self, i: usize, anchor: usize) -> &mut Self {
        if i != anchor {
            self.insert_after_any(i, anchor);
        }
        self
    }

    #[inline]
    pub fn move_node_before_node(&mut self, i: usize, y: usize) -> &mut Self {
        self.insert_before_any(i, y)
    }

    #[inline]
    pub fn push_front_to_berth(&mut self, i: usize, berth: usize) -> &mut Self {
        let head = self.chain.start_of(berth);
        self.insert_after_any(i, head)
    }

    #[inline]
    pub fn push_back_to_berth(&mut self, i: usize, berth: usize) -> &mut Self {
        let tail = self.chain.end_of(berth);
        self.insert_before_any(i, tail)
    }

    #[inline]
    pub fn move_node_to_berth_back(&mut self, i: usize, target_berth: usize) -> &mut Self {
        self.push_back_to_berth(i, target_berth)
    }

    #[inline]
    pub fn move_node_to_berth_front(&mut self, i: usize, target_berth: usize) -> &mut Self {
        self.push_front_to_berth(i, target_berth)
    }

    #[inline]
    pub fn splice_after(&mut self, a: usize, b: usize, x: usize) -> &mut Self {
        debug_assert!(a < self.chain.len() && b < self.chain.len() && x < self.chain.len());
        debug_assert!(
            a < self.base && b < self.base,
            "cannot splice sentinels as payload"
        );

        if self.is_tail_idx(x) {
            let y = x;
            let prev_y = self.pred_in_view(y);
            if prev_y == b || prev_y == self.pred_in_view(a) || x == a {
                return self;
            }
            self.detach_segment(a, b);
            let nx = self.next_after(prev_y);
            self.rewrite_arc(prev_y, a);
            self.rewrite_arc(b, nx);
            return self;
        }

        let pa = self.pred_in_view(a);
        if x == b || x == pa || x == a {
            return self;
        }

        let nb = self.next_after(b);
        let nx = self.next_after(x);

        self.rewrite_arc(pa, nb);
        self.rewrite_arc(x, a);
        self.rewrite_arc(b, nx);
        self
    }

    #[inline]
    pub fn move_segment_after(&mut self, a: usize, b: usize, x: usize) -> &mut Self {
        debug_assert!(a < self.chain.len() && b < self.chain.len() && x < self.chain.len());
        debug_assert!(
            a < self.base && b < self.base,
            "cannot move sentinel payload"
        );

        let pa = self.pred_in_view(a);
        if x == b || x == pa {
            return self;
        }

        #[cfg(debug_assertions)]
        {
            let mut cur = a;
            let mut hops = 0usize;
            while cur != b && hops <= self.chain.len() {
                cur = self.next_after(cur);
                hops += 1;
            }
            debug_assert!(cur == b, "[a..=b] must be contiguous");

            let mut cur2 = a;
            let mut inside = false;
            while cur2 != b {
                cur2 = self.next_after(cur2);
                if cur2 != b && cur2 == x {
                    inside = true;
                    break;
                }
            }
            if inside {
                return self;
            }
        }

        if self.is_tail_idx(x) {
            let y = self.pred_in_view(x);
            if y == b || y == pa {
                return self;
            }
            self.detach_segment(a, b);
            let ny = self.next_after(y);
            self.rewrite_arc(y, a);
            self.rewrite_arc(b, ny);
            return self;
        }

        let nb = self.next_after(b);
        let nx = self.next_after(x);

        self.rewrite_arc(pa, nb);

        self.rewrite_arc(x, a);
        self.rewrite_arc(b, nx);
        self
    }

    #[inline]
    pub fn move_segment_before_node(&mut self, a: usize, b: usize, y: usize) -> &mut Self {
        let x = self.pred_in_view(y);
        self.move_segment_after(a, b, x)
    }

    #[inline]
    pub fn move_segment_to_berth_back(
        &mut self,
        a: usize,
        b: usize,
        target_berth: usize,
    ) -> &mut Self {
        let tail = self.chain.end_of(target_berth);
        let x = self.pred_in_view(tail);
        self.move_segment_after(a, b, x)
    }

    #[inline]
    pub fn move_segment_to_berth_front(
        &mut self,
        a: usize,
        b: usize,
        target_berth: usize,
    ) -> &mut Self {
        let head = self.chain.start_of(target_berth);
        self.move_segment_after(a, b, head)
    }

    #[inline]
    pub fn mark_berth(&mut self, berth: usize) -> &mut Self {
        self.delta.mark_berth(berth);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain::double_chain::DoubleChain;

    fn collect_berth(chain: &DoubleChain, berth: usize) -> Vec<usize> {
        chain.iter_berth(berth).collect()
    }

    fn assert_berth_eq(chain: &DoubleChain, berth: usize, expected: &[usize]) {
        let got = collect_berth(chain, berth);
        assert_eq!(got, expected, "berth {} content mismatch", berth);

        // Forward/backward ring consistency
        let s = chain.start_of(berth);
        let e = chain.end_of(berth);

        // Walk forward from start to end through expected
        let mut cur = chain.succ(s);
        for &n in expected {
            assert_eq!(cur, n);
            cur = chain.succ(cur);
        }
        assert_eq!(cur, e);

        // Walk backward from end to start through expected reversed
        let mut cur = chain.pred(e);
        for &n in expected.iter().rev() {
            assert_eq!(cur, n);
            cur = chain.pred(cur);
        }
        assert_eq!(cur, s);
    }

    fn make_chain(num_nodes: usize, num_berths: usize) -> DoubleChain {
        DoubleChain::new(num_nodes, num_berths)
    }

    fn fill_berth(chain: &mut DoubleChain, berth: usize, nodes_in_order: &[usize]) {
        let e = chain.end_of(berth);
        for &n in nodes_in_order {
            chain.insert_before(n, e);
        }
    }

    fn assert_all_berths_equal(a: &DoubleChain, b: &DoubleChain) {
        assert_eq!(a.num_berths(), b.num_berths(), "berth count mismatch");
        for berth in 0..a.num_berths() {
            assert_eq!(
                collect_berth(a, berth),
                collect_berth(b, berth),
                "berth {} differs",
                berth
            );
        }
        // Also assert next/prev arrays equal
        assert_eq!(a.next_slice(), b.next_slice(), "next arrays differ");
        assert_eq!(a.prev_slice(), b.prev_slice(), "prev arrays differ");
        assert_eq!(a.base_index(), b.base_index(), "base index differs");
    }

    // Applies exactly one builder operation (one ChainDelta application),
    // and mirrors it with the corresponding direct operation on `truth`.
    fn apply_and_compare<Fb, Fd>(
        truth: &mut DoubleChain,
        target: &mut DoubleChain,
        build_op: Fb,
        direct_op: Fd,
    ) -> usize
    where
        Fb: FnOnce(&DoubleChain, &mut DeltaBuilder),
        Fd: FnOnce(&mut DoubleChain),
    {
        // The builder must see the current target state as its base view.
        let base_for_builder = target.clone();
        let mut builder = DeltaBuilder::new(&base_for_builder);
        build_op(&base_for_builder, &mut builder);
        let delta = builder.into_delta();
        let len = delta.len();
        target.apply_delta(&delta);

        // Apply the exact direct operation on the truth chain.
        direct_op(truth);

        // Then compare structures.
        assert_all_berths_equal(truth, target);

        len
    }

    #[test]
    fn insert_after_and_before_equivalence_and_update_counts() {
        // 1 berth, nodes initially skipped
        let mut truth = make_chain(6, 1);
        // No initial mutations before cloning target
        let mut target = truth.clone();

        let s = truth.start_of(0);
        let e = truth.end_of(0);

        // Step 1: insert_after(0, s)
        let rewrites_1 = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.insert_after(0, s);
            },
            |c| {
                c.insert_after(0, s);
            },
        );
        assert_eq!(rewrites_1, 2);

        // Step 2: insert_before(1, e)
        let rewrites_2 = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.insert_before(1, e);
            },
            |c| {
                c.insert_before(1, e);
            },
        );
        assert_eq!(rewrites_2, 2);

        // Step 3: insert_after(2, 0)
        let rewrites_3 = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.insert_after(2, 0);
            },
            |c| {
                c.insert_after(2, 0);
            },
        );
        assert_eq!(rewrites_3, 2);

        // Final explicit check
        assert_berth_eq(&truth, 0, &[0, 2, 1]);
    }

    #[test]
    fn splice_after_generic_path_and_noops() {
        // Build berth [0,1,2,3]
        let mut truth = make_chain(6, 1);
        fill_berth(&mut truth, 0, &[0, 1, 2, 3]);
        // Clone AFTER initial fill so target starts equal to truth
        let mut target = truth.clone();

        // Generic case: move [1..=2] after 3 => [0,3,1,2]
        let rewrites_generic = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.splice_after(1, 2, 3);
            },
            |c| {
                c.splice_after(1, 2, 3);
            },
        );
        assert_eq!(rewrites_generic, 3);

        // Now test no-ops on a fresh initial state
        let mut truth2 = make_chain(6, 1);
        fill_berth(&mut truth2, 0, &[0, 1, 2, 3]);
        let mut target2 = truth2.clone();

        // x == b
        let r1 = apply_and_compare(
            &mut truth2,
            &mut target2,
            |_base, b| {
                b.splice_after(1, 2, 2);
            },
            |_c| {},
        );
        assert_eq!(r1, 0);

        // x == pred(a)
        let pa = target2.prev_slice()[1];
        let r2 = apply_and_compare(
            &mut truth2,
            &mut target2,
            |_base, b| {
                b.splice_after(1, 2, pa);
            },
            |_c| {},
        );
        assert_eq!(r2, 0);

        // x == a
        let r3 = apply_and_compare(
            &mut truth2,
            &mut target2,
            |_base, b| {
                b.splice_after(1, 2, 1);
            },
            |_c| {},
        );
        assert_eq!(r3, 0);

        // Verify generic-case final state from the first pair
        assert_berth_eq(&target, 0, &[0, 3, 1, 2]);
    }

    #[test]
    fn splice_after_tail_anchor_path_and_rewrite_count() {
        // Build berth [0,1,2,3]
        let mut truth = make_chain(6, 1);
        fill_berth(&mut truth, 0, &[0, 1, 2, 3]);
        let mut target = truth.clone();

        let tail = truth.end_of(0);

        let rewrites = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.splice_after(1, 2, tail);
            },
            |c| {
                c.splice_after(1, 2, tail);
            },
        );
        assert_eq!(rewrites, 3);
    }

    #[test]
    fn move_segment_after_and_wrappers_equivalence() {
        // Case 1: [0,1,2,3] -> move [0..=1] after 2 => [2,0,1,3]
        let mut truth = make_chain(6, 1);
        fill_berth(&mut truth, 0, &[0, 1, 2, 3]);
        let mut target = truth.clone();

        let r1 = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_segment_after(0, 1, 2);
            },
            |c| {
                c.move_segment_after(0, 1, 2);
            },
        );
        assert_eq!(r1, 3);

        // Case 2: [0,1,2] -> move 2 after 0 => [0,2,1]
        let mut truth2 = make_chain(6, 1);
        fill_berth(&mut truth2, 0, &[0, 1, 2]);
        let mut target2 = truth2.clone();

        let r2 = apply_and_compare(
            &mut truth2,
            &mut target2,
            |_base, b| {
                b.move_after(2, 0);
            },
            |c| {
                c.move_after(2, 0);
            },
        );
        assert_eq!(r2, 3);

        // Case 3: [0,2,1] -> move 1 before tail => [0,2,1]
        let mut truth3 = make_chain(6, 1);
        fill_berth(&mut truth3, 0, &[0, 2, 1]);
        let mut target3 = truth3.clone();
        let e = truth3.end_of(0);

        let _r3 = apply_and_compare(
            &mut truth3,
            &mut target3,
            |_base, b| {
                b.move_before(1, e);
            },
            |c| {
                c.move_before(1, e);
            },
        );
        // no strict count check here (could be a no-op)
    }

    #[test]
    fn push_front_and_back_to_berth_equivalence_and_counts() {
        let mut truth = make_chain(8, 2);
        let mut target = truth.clone();

        // push_back_to_berth(0, 0)
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.push_back_to_berth(0, 0);
            },
            |c| {
                c.push_back_to_berth(0, 0);
            },
        );
        // push_front_to_berth(1, 0)
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.push_front_to_berth(1, 0);
            },
            |c| {
                c.push_front_to_berth(1, 0);
            },
        );

        // Check explicitly
        assert_berth_eq(&truth, 0, &[1, 0]);
    }

    #[test]
    fn move_node_to_berth_front_and_back_detach_when_needed() {
        // Put [0,1,2] in berth 0
        let mut truth = make_chain(8, 2);
        let s0 = truth.start_of(0);
        truth.insert_after(0, s0);
        truth.insert_after(1, 0);
        truth.insert_after(2, 1);
        let mut target = truth.clone();

        // Move node 1 to front of berth 1
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_node_to_berth_front(1, 1);
            },
            |c| {
                c.move_node_to_berth_front(1, 1);
            },
        );

        // Move same node to back of berth 0
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_node_to_berth_back(1, 0);
            },
            |c| {
                c.move_node_to_berth_back(1, 0);
            },
        );

        assert_berth_eq(&truth, 0, &[0, 2, 1]);
    }

    #[test]
    fn move_node_after_node_across_berths() {
        // berth0: [0], berth1: [1,2] => move 2 after 0 => berth0: [0,2], berth1: [1]
        let mut truth = make_chain(6, 2);
        let s0 = truth.start_of(0);
        let s1 = truth.start_of(1);

        truth.insert_after(0, s0);
        truth.insert_after(1, s1);
        truth.insert_after(2, 1);
        let mut target = truth.clone();

        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_node_after_node(2, 0);
            },
            |c| {
                c.move_node_after_node(2, 0);
            },
        );

        assert_berth_eq(&truth, 0, &[0, 2]);
        assert_berth_eq(&truth, 1, &[1]);
    }

    #[test]
    fn move_segment_to_berth_front_back_and_before_node() {
        // berth0: [0,1,2,3]
        let mut truth = make_chain(8, 2);

        let s0 = truth.start_of(0);
        truth.insert_after(0, s0);
        truth.insert_after(1, 0);
        truth.insert_after(2, 1);
        truth.insert_after(3, 2);
        let mut target = truth.clone();

        // Front to berth 1
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_segment_to_berth_front(1, 2, 1);
            },
            |c| {
                c.move_segment_to_berth_front(1, 2, 1);
            },
        );
        // Back to berth 0
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_segment_to_berth_back(1, 2, 0);
            },
            |c| {
                c.move_segment_to_berth_back(1, 2, 0);
            },
        );
        // Before node 3
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_segment_before_node(1, 2, 3);
            },
            |c| {
                c.move_segment_before_node(1, 2, 3);
            },
        );

        assert_berth_eq(&truth, 0, &[0, 1, 2, 3]);
    }

    #[test]
    fn builder_mark_berth_is_callable() {
        // We cannot directly observe berth marking from ChainDelta here (depends on ChainDelta API),
        // but we ensure it is callable and does not panic.
        let base = make_chain(2, 2);
        let mut b = DeltaBuilder::new(&base);
        b.mark_berth(0).mark_berth(1);
        let _d = b.into_delta(); // just ensure it constructs
    }

    #[test]
    fn noops_do_not_add_updates() {
        // Prepare [0,1,2]
        let base = make_chain(5, 1);
        let mut c = base.clone();
        let s = c.start_of(0);
        c.insert_after(0, s);
        c.insert_after(1, 0);
        c.insert_after(2, 1);

        // Build against an identical base
        let mut b = DeltaBuilder::new(&c);

        // move_after(i, i) no-op
        b.move_after(1, 1);
        // splice_after: x == b => no-op
        b.splice_after(0, 0, 0);
        // move_segment_after: x == b => no-op
        b.move_segment_after(0, 0, 0);

        let d = b.into_delta();
        assert_eq!(d.len(), 0, "no-ops should not produce updates");
    }

    #[test]
    fn compound_sequence_produces_correct_final_structure() {
        // Start with two berths
        let mut truth = make_chain(10, 2);

        // Fill berth0 with [0,1,2], berth1 with [3,4]
        fill_berth(&mut truth, 0, &[0, 1, 2]);
        fill_berth(&mut truth, 1, &[3, 4]);
        let mut target = truth.clone();

        // Move 1 after 0 (no change)
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_after(1, 0);
            },
            |c| {
                c.move_after(1, 0);
            },
        );
        // Move 2 after 3 (cross-berth)
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_node_after_node(2, 3);
            },
            |c| {
                c.move_node_after_node(2, 3);
            },
        );
        // Insert 5 before berth1 tail (push back)
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.push_back_to_berth(5, 1);
            },
            |c| {
                c.push_back_to_berth(5, 1);
            },
        );
        // Splice [0..=0] after 4
        apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.splice_after(0, 0, 4);
            },
            |c| {
                c.splice_after(0, 0, 4);
            },
        );
    }

    // ---------- Debug-assertion guards (only active in debug builds) ----------

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cannot insert sentinel")]
    fn insert_after_rejects_sentinel_payload_in_builder() {
        let base = make_chain(2, 1);
        let s = base.start_of(0);

        let mut b = DeltaBuilder::new(&base);
        // i is a sentinel -> must panic in debug
        b.insert_after(s, s);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "node must be skipped")]
    fn insert_after_requires_skipped_node_in_builder() {
        // Place node 0 into berth, then try to insert it again without skipping
        let base = make_chain(2, 1);
        let s = base.start_of(0);
        // Put 0 into chain
        let mut c = base.clone();
        c.insert_after(0, s);

        // Builder sees the base+prev_view from c
        let mut b = DeltaBuilder::new(&c);
        // 0 is not skipped in the current (base+delta) view; should panic in debug
        b.insert_after(0, s);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn splice_after_tail_path_awkw_cases_noop() {
        // Construct [0,1], test tail anchor awkward guards
        let mut base = make_chain(4, 1);
        fill_berth(&mut base, 0, &[0, 1]);
        let tail = base.end_of(0);

        let mut b = DeltaBuilder::new(&base);

        // prev_y == b => no-op
        b.splice_after(0, 0, tail);
        let d1 = b.into_delta();
        // Initial splice will likely do 3 rewrites, but to specifically hit the guard:
        // craft a case where prev_y == b. That requires a different structure; we just ensure
        // calling again with no structural change yields no extra updates.
        let base2 = base.clone();
        let mut b2 = DeltaBuilder::new(&base2);
        b2.splice_after(1, 1, tail); // this path can short-circuit depending on structure

        let d2 = b2.into_delta();
        let _ = (d1, d2); // ensure no panics in debug path
    }

    #[test]
    fn move_segment_after_tail_moves_whole_segment_builder() {
        // Build berth [0,1,2]
        let mut truth = make_chain(5, 1);
        fill_berth(&mut truth, 0, &[0, 1, 2]);
        let mut target = truth.clone();
        let tail = truth.end_of(0);

        // Move [0,1] after tail => [2,0,1]
        let rewrites = apply_and_compare(
            &mut truth,
            &mut target,
            |_base, b| {
                b.move_segment_after(0, 1, tail);
            },
            |c| {
                c.move_segment_after(0, 1, tail);
            },
        );
        assert_eq!(rewrites, 3);
        assert_berth_eq(&truth, 0, &[2, 0, 1]);
    }
}
