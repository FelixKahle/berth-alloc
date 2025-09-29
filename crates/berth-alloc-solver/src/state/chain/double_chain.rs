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

use crate::state::chain::delta::{ArcRewrite, ChainDelta};

#[derive(Debug, Clone)]
pub struct DoubleChain {
    next: Vec<usize>,
    prev: Vec<usize>,
    start: Vec<usize>,
    end: Vec<usize>,
    base: usize,
}

impl DoubleChain {
    #[inline]
    pub fn new(num_nodes: usize, num_berths: usize) -> Self {
        let total = num_nodes + 2 * num_berths;
        let mut next = vec![0; total];
        let mut prev = vec![0; total];

        for i in 0..num_nodes {
            next[i] = i;
            prev[i] = i;
        }

        let mut start = Vec::with_capacity(num_berths);
        let mut end = Vec::with_capacity(num_berths);

        for b in 0..num_berths {
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
            base: num_nodes,
        }
    }

    #[inline(always)]
    fn is_sentinel(&self, x: usize) -> bool {
        x >= self.base
    }

    #[inline(always)]
    fn is_tail_idx(&self, x: usize) -> bool {
        x >= self.base && ((x - self.base) & 1) == 1
    }

    #[inline(always)]
    fn is_head_idx(&self, x: usize) -> bool {
        x >= self.base && ((x - self.base) & 1) == 0
    }

    #[inline]
    pub fn next_slice(&self) -> &[usize] {
        &self.next
    }

    #[inline]
    pub fn prev_slice(&self) -> &[usize] {
        &self.prev
    }

    #[inline]
    pub fn base_index(&self) -> usize {
        self.base
    }

    #[inline]
    pub fn start_slice(&self) -> &[usize] {
        &self.start
    }

    #[inline]
    pub fn end_slice(&self) -> &[usize] {
        &self.end
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.next.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn num_berths(&self) -> usize {
        self.start.len()
    }

    #[inline]
    pub fn start_of(&self, berth: usize) -> usize {
        debug_assert!(berth < self.num_berths());
        self.start[berth]
    }

    #[inline]
    pub fn end_of(&self, berth: usize) -> usize {
        debug_assert!(berth < self.num_berths());
        self.end[berth]
    }

    #[inline]
    pub fn is_skipped(&self, i: usize) -> bool {
        debug_assert!(i < self.len());
        self.next[i] == i
    }

    #[inline(always)]
    pub fn is_tail(&self, x: usize) -> bool {
        self.is_tail_idx(x)
    }

    #[inline(always)]
    pub fn is_head(&self, x: usize) -> bool {
        self.is_head_idx(x)
    }

    #[inline]
    pub fn succ(&self, i: usize) -> usize {
        debug_assert!(i < self.len());
        self.next[i]
    }

    #[inline]
    pub fn pred(&self, i: usize) -> usize {
        debug_assert!(i < self.len());
        self.prev[i]
    }

    #[inline(always)]
    pub fn skip(&mut self, i: usize) {
        debug_assert!(i < self.len());
        if self.is_sentinel(i) || self.next[i] == i {
            return;
        }

        let p = self.prev[i];
        let n = self.next[i];
        debug_assert!(p < self.len() && n < self.len());

        self.next[p] = n;
        self.prev[n] = p;
        self.next[i] = i;
        self.prev[i] = i;
    }

    #[inline(always)]
    pub fn insert_after(&mut self, i: usize, anchor: usize) {
        debug_assert!(i < self.len() && anchor < self.len());
        debug_assert!(!self.is_sentinel(i), "cannot insert sentinel");
        debug_assert!(self.is_skipped(i), "node must be skipped before insert");

        if self.is_tail_idx(anchor) {
            let a = self.prev[anchor];
            let an = self.next[a];
            debug_assert!(a < self.len() && an < self.len());

            self.prev[i] = a;
            self.next[i] = an;
            self.next[a] = i;
            self.prev[an] = i;
            return;
        }

        let an = self.next[anchor];
        debug_assert!(an < self.len());
        self.prev[i] = anchor;
        self.next[i] = an;
        self.next[anchor] = i;
        self.prev[an] = i;
    }

    #[inline]
    pub fn insert_before(&mut self, i: usize, before: usize) {
        debug_assert!(before < self.len());
        let a = self.prev[before];
        self.insert_after(i, a);
    }

    #[inline]
    pub fn detach_segment(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len() && b < self.len());
        if a == b && self.is_skipped(a) {
            return;
        }
        debug_assert!(!self.is_sentinel(a) && !self.is_sentinel(b));

        let pa = self.prev[a];
        let nb = self.next[b];
        debug_assert!(pa < self.len() && nb < self.len());

        self.next[pa] = nb;
        self.prev[nb] = pa;
    }

    #[inline(always)]
    pub fn splice_after(&mut self, a: usize, b: usize, x: usize) {
        debug_assert!(a < self.len() && b < self.len() && x < self.len());
        debug_assert!(!self.is_sentinel(a) && !self.is_sentinel(b));

        let pa = self.prev[a];
        if x == b || x == pa || x == a {
            return;
        }

        #[cfg(debug_assertions)]
        {
            let mut cur = a;
            let mut hops = 0usize;
            while cur != b && hops <= self.len() {
                cur = self.next[cur];
                hops += 1;
            }
            debug_assert!(cur == b, "[a..=b] must be contiguous");

            let mut inside = false;
            cur = a;
            while cur != b {
                cur = self.next[cur];
                if cur != b && cur == x {
                    inside = true;
                    break;
                }
            }
            if inside {
                return;
            }
        }

        if self.is_tail_idx(x) {
            let y = self.prev[x];
            if y == b || y == pa {
                return;
            }
            let nb = self.next[b];
            let ny = self.next[y];

            self.next[pa] = nb;
            self.prev[nb] = pa;

            self.next[y] = a;
            self.prev[a] = y;
            self.next[b] = ny;
            self.prev[ny] = b;
            return;
        }

        let nb = self.next[b];
        let nx = self.next[x];

        self.next[pa] = nb;
        self.prev[nb] = pa;

        self.next[x] = a;
        self.prev[a] = x;
        self.next[b] = nx;
        self.prev[nx] = b;
    }

    #[inline(always)]
    pub fn move_segment_after(&mut self, a: usize, b: usize, x: usize) {
        debug_assert!(a < self.len() && b < self.len() && x < self.len());
        debug_assert!(!self.is_sentinel(a) && !self.is_sentinel(b));

        if x == b || self.prev[a] == x {
            return;
        }
        if self.is_tail_idx(x) {
            self.move_before(a, x);
        } else {
            let pa = self.prev[a];
            let nb = self.next[b];
            let nx = self.next[x];

            self.next[pa] = nb;
            self.prev[nb] = pa;

            self.next[x] = a;
            self.prev[a] = x;
            self.next[b] = nx;
            self.prev[nx] = b;
        }
    }

    #[inline]
    pub fn move_after(&mut self, i: usize, x: usize) {
        if i == x {
            return;
        }
        self.move_segment_after(i, i, x)
    }

    #[inline]
    pub fn move_before(&mut self, i: usize, y: usize) {
        debug_assert!(y < self.len());
        let x = self.prev[y];
        self.move_after(i, x)
    }

    #[inline]
    pub fn push_back_to_berth(&mut self, i: usize, berth: usize) {
        debug_assert!(!self.is_sentinel(i));
        let tail = self.end_of(berth);
        self.insert_before(i, tail)
    }

    #[inline]
    pub fn push_front_to_berth(&mut self, i: usize, berth: usize) {
        debug_assert!(!self.is_sentinel(i));
        let head = self.start_of(berth);
        self.insert_after(i, head)
    }

    #[inline]
    pub fn move_node_to_berth_back(&mut self, i: usize, target_berth: usize) {
        debug_assert!(!self.is_sentinel(i));
        if !self.is_skipped(i) {
            self.skip(i);
        }
        self.push_back_to_berth(i, target_berth)
    }

    #[inline]
    pub fn move_node_to_berth_front(&mut self, i: usize, target_berth: usize) {
        debug_assert!(!self.is_sentinel(i));
        if !self.is_skipped(i) {
            self.skip(i);
        }
        self.push_front_to_berth(i, target_berth)
    }

    #[inline]
    pub fn move_node_after_node(&mut self, i: usize, anchor: usize) {
        debug_assert!(!self.is_sentinel(i));
        if i == anchor {
            return;
        }
        if !self.is_skipped(i) {
            self.skip(i);
        }
        self.insert_after(i, anchor)
    }

    #[inline]
    pub fn move_segment_to_berth_back(&mut self, a: usize, b: usize, target_berth: usize) {
        debug_assert!(!self.is_sentinel(a) && !self.is_sentinel(b));
        let tail = self.end_of(target_berth);
        let x = self.prev[tail];
        self.move_segment_after(a, b, x)
    }

    #[inline]
    pub fn move_segment_to_berth_front(&mut self, a: usize, b: usize, target_berth: usize) {
        debug_assert!(!self.is_sentinel(a) && !self.is_sentinel(b));
        let head = self.start_of(target_berth);
        self.move_segment_after(a, b, head)
    }

    #[inline]
    pub fn move_segment_before_node(&mut self, a: usize, b: usize, y: usize) {
        let x = self.pred(y);
        self.move_segment_after(a, b, x)
    }

    #[inline]
    pub fn move_node_before_node(&mut self, i: usize, y: usize) {
        debug_assert!(!self.is_sentinel(i));
        if !self.is_skipped(i) {
            self.skip(i);
        }
        self.insert_before(i, y)
    }

    #[inline]
    pub fn apply_arcrewrite(&mut self, r: ArcRewrite) {
        let n = self.len();
        debug_assert!(r.tail() < n && r.expected_head() < n && r.new_head() < n);
        debug_assert_eq!(self.next[r.tail()], r.expected_head(), "guard mismatch");

        self.next[r.tail()] = r.new_head();

        if self.prev[r.expected_head()] == r.tail() {
            self.prev[r.expected_head()] = r.expected_head();
        }
        self.prev[r.new_head()] = r.tail();

        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(self.next[r.tail()], r.new_head());
            debug_assert_eq!(self.prev[r.new_head()], r.tail());
        }
    }

    #[inline]
    pub fn apply_delta(&mut self, delta: &ChainDelta) {
        #[cfg(debug_assertions)]
        {
            use std::collections::HashSet;
            let mut seen = HashSet::with_capacity(delta.len());
            for u in delta.updates() {
                let inserted = seen.insert(u.tail());
                debug_assert!(inserted, "duplicate tail {} in ChainDelta", u.tail());
            }
        }
        for &r in delta.updates() {
            self.apply_arcrewrite(r);
        }
    }

    #[inline]
    pub fn iter_berth<'a>(&'a self, berth: usize) -> impl Iterator<Item = usize> + 'a {
        debug_assert!(berth < self.num_berths());
        let s = self.start[berth];
        let e = self.end[berth];
        ChainIter::new(&self.next, self.next[s], e)
    }
}

#[derive(Debug, Clone)]
pub struct ChainIter<'a> {
    next: &'a [usize],
    cur: usize,
    end: usize,
}

impl<'a> ChainIter<'a> {
    #[inline]
    fn new(next: &'a [usize], start: usize, end: usize) -> Self {
        Self {
            next,
            cur: start,
            end,
        }
    }
}

impl<'a> Iterator for ChainIter<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.end {
            return None;
        }
        let out = self.cur;
        self.cur = self.next[self.cur];
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::chain::delta::{ArcRewrite, ChainDelta};

    // Helpers
    fn collect_berth(chain: &DoubleChain, berth: usize) -> Vec<usize> {
        chain.iter_berth(berth).collect()
    }

    fn assert_berth_eq(chain: &DoubleChain, berth: usize, expected: &[usize]) {
        let got = collect_berth(chain, berth);
        assert_eq!(got, expected, "berth {} content mismatch", berth);

        // Also verify forward/backward ring consistency via succ/pred around sentinels.
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

    #[test]
    fn test_new_initial_state_and_invariants() {
        let n = 5;
        let b = 3;
        let c = DoubleChain::new(n, b);

        assert_eq!(c.len(), n + 2 * b);
        assert_eq!(c.num_berths(), b);

        for i in 0..n {
            assert!(c.is_skipped(i));
            assert!(!c.is_tail(i));
            assert!(!c.is_head(i));
            assert_eq!(c.succ(i), i);
            assert_eq!(c.pred(i), i);
        }

        for berth in 0..b {
            let s = c.start_of(berth);
            let e = c.end_of(berth);
            assert!(c.is_head(s));
            assert!(c.is_tail(e));
            assert_eq!(c.succ(s), e);
            assert_eq!(c.pred(s), s);
            assert_eq!(c.succ(e), e);
            assert_eq!(c.pred(e), s);
            assert_berth_eq(&c, berth, &[]);
        }
    }

    #[test]
    fn test_is_sentinel_head_tail_checks() {
        let n = 4;
        let b = 2;
        let c = DoubleChain::new(n, b);

        for i in 0..n {
            assert!(!c.is_head(i));
            assert!(!c.is_tail(i));
        }
        for berth in 0..b {
            let s = c.start_of(berth);
            let e = c.end_of(berth);
            assert!(c.is_head(s) && !c.is_tail(s));
            assert!(!c.is_head(e) && c.is_tail(e));
        }
    }

    #[test]
    fn test_skip_behavior() {
        let mut c = DoubleChain::new(3, 1);

        // Skipping already-skipped nodes is a no-op
        for i in 0..3 {
            c.skip(i);
            assert!(c.is_skipped(i));
        }

        // Skipping sentinels is a no-op
        let s = c.start_of(0);
        let e = c.end_of(0);
        c.skip(s);
        c.skip(e);
        assert_eq!(c.succ(s), e);
        assert_eq!(c.pred(e), s);

        // Insert nodes then skip to detach
        c.insert_after(0, s);
        c.insert_after(1, 0);
        assert_berth_eq(&c, 0, &[0, 1]);
        c.skip(1);
        assert!(c.is_skipped(1));
        assert_berth_eq(&c, 0, &[0]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cannot insert sentinel")]
    fn test_insert_after_rejects_sentinel_payload() {
        let mut c = DoubleChain::new(1, 1);
        let s = c.start_of(0);
        // i is sentinel -> debug assert should fire
        c.insert_after(s, s);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "node must be skipped before insert")]
    fn test_insert_after_requires_skipped_node() {
        let mut c = DoubleChain::new(2, 1);
        let s = c.start_of(0);
        c.insert_after(0, s);
        // Insert the same node again without skipping -> debug assert
        c.insert_after(0, s);
    }

    #[test]
    fn test_insert_after_and_before_normal_and_tail_anchor() {
        let mut c = DoubleChain::new(4, 1);
        let s = c.start_of(0);
        let e = c.end_of(0);

        // Insert 0 after head
        c.insert_after(0, s);
        assert_berth_eq(&c, 0, &[0]);

        // Tail-anchor path: inserting after tail places before tail
        c.insert_after(1, e);
        assert_berth_eq(&c, 0, &[0, 1]);

        // Insert in the middle: after 0
        c.insert_after(2, 0);
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // insert_before before tail should put node just before end
        c.insert_before(3, e);
        assert_berth_eq(&c, 0, &[0, 2, 1, 3]);
    }

    #[test]
    fn test_detach_segment_basic() {
        let mut c = DoubleChain::new(6, 1);
        let s = c.start_of(0);

        // No-op for (a==b) and skipped
        c.detach_segment(0, 0);
        assert!(c.is_skipped(0));

        // Setup: 0,1,2,3 in berth
        c.insert_after(0, s);
        c.insert_after(1, 0);
        c.insert_after(2, 1);
        c.insert_after(3, 2);
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // Detach [1..=2]
        c.detach_segment(1, 2);
        assert_berth_eq(&c, 0, &[0, 3]);

        // Detached segment [1..=2] still linked
        assert_eq!(c.succ(1), 2);
        assert_eq!(c.pred(2), 1);
    }

    #[test]
    fn test_splice_after_moves_segment_and_noops() {
        let mut c = DoubleChain::new(6, 1);
        let s = c.start_of(0);

        // Build: 0,1,2,3
        for i in 0..4 {
            c.insert_after(i, if i == 0 { s } else { i - 1 });
        }
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // Move [1..=2] after 3 -> becomes 0,3,1,2
        c.splice_after(1, 2, 3);
        assert_berth_eq(&c, 0, &[0, 3, 1, 2]);

        // No-op when anchor equals b or prev[a]
        c.splice_after(1, 2, 2);
        assert_berth_eq(&c, 0, &[0, 3, 1, 2]);

        // No-op when x is strictly inside (a,b) (guarded in debug)
        #[cfg(debug_assertions)]
        {
            c.splice_after(1, 2, 1);
            assert_berth_eq(&c, 0, &[0, 3, 1, 2]);
        }
    }

    #[test]
    fn test_move_segment_after_and_tail_cases() {
        let mut c = DoubleChain::new(6, 2);

        // Place 0,1,2 in berth 0; 3 in berth 1
        let s0 = c.start_of(0);
        let s1 = c.start_of(1);
        let t1 = c.end_of(1);

        c.insert_after(0, s0);
        c.insert_after(1, 0);
        c.insert_after(2, 1);
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        c.insert_after(3, s1);
        assert_berth_eq(&c, 1, &[3]);

        // Move [1..=2] after 3 -> berth1 becomes [3,1,2], berth0 becomes [0]
        c.move_segment_after(1, 2, 3);
        assert_berth_eq(&c, 0, &[0]);
        assert_berth_eq(&c, 1, &[3, 1, 2]);

        // Tail case: moving [0..=0] "after tail" means before tail sentinel
        c.move_segment_after(0, 0, t1);
        assert_berth_eq(&c, 1, &[3, 1, 2, 0]);

        // No-ops
        c.move_segment_after(1, 2, 2); // x == b
        c.move_segment_after(1, 2, c.pred(1)); // prev[a] == x
        assert_berth_eq(&c, 1, &[3, 1, 2, 0]);
    }

    #[test]
    fn test_move_after_and_move_before_wrappers() {
        let mut c = DoubleChain::new(5, 1);
        let s = c.start_of(0);
        let e = c.end_of(0);

        c.insert_after(0, s);
        c.insert_after(1, 0);
        c.insert_after(2, 1);
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        // move_after: move 2 after 0 -> [0,2,1]
        c.move_after(2, 0);
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // move_after no-op when i == x
        c.move_after(0, 0);
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // move_before: move 1 before end (i.e., to back) -> [0,2,1]
        c.move_before(1, e);
        assert_berth_eq(&c, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_push_front_and_back_to_berth() {
        let mut c = DoubleChain::new(6, 2);

        // Push into empty berth 0
        c.push_back_to_berth(0, 0);
        assert_berth_eq(&c, 0, &[0]);

        // Push front into same berth
        c.push_front_to_berth(1, 0);
        assert_berth_eq(&c, 0, &[1, 0]);

        // Add to other berth
        c.push_back_to_berth(2, 1);
        c.push_front_to_berth(3, 1);
        assert_berth_eq(&c, 1, &[3, 2]);
    }

    #[test]
    fn test_move_node_to_berth_front_and_back() {
        let mut c = DoubleChain::new(6, 2);

        // Put nodes into berth 0
        let s0 = c.start_of(0);
        c.insert_after(0, s0);
        c.insert_after(1, 0);
        c.insert_after(2, 1);
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        // Move node 1 to front of berth 1
        c.move_node_to_berth_front(1, 1);
        assert_berth_eq(&c, 0, &[0, 2]);
        assert_berth_eq(&c, 1, &[1]);

        // Move same node to back of berth 0
        c.move_node_to_berth_back(1, 0);
        assert_berth_eq(&c, 0, &[0, 2, 1]);
        assert_berth_eq(&c, 1, &[]);
    }

    #[test]
    fn test_move_node_after_node_across_berths() {
        let mut c = DoubleChain::new(6, 2);
        let s0 = c.start_of(0);
        let s1 = c.start_of(1);

        c.insert_after(0, s0);
        c.insert_after(1, 0);
        assert_berth_eq(&c, 0, &[0, 1]);

        c.insert_after(2, s1);
        assert_berth_eq(&c, 1, &[2]);

        // Move node 2 after node 0 (cross-berth)
        c.move_node_after_node(2, 0);
        assert_berth_eq(&c, 0, &[0, 2, 1]);
        assert_berth_eq(&c, 1, &[]);

        // No-op when i == anchor
        c.move_node_after_node(0, 0);
        assert_berth_eq(&c, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_move_segment_to_berth_front_and_back_and_before_node() {
        let mut c = DoubleChain::new(8, 2);

        let s0 = c.start_of(0);
        c.insert_after(0, s0);
        c.insert_after(1, 0);
        c.insert_after(2, 1);
        c.insert_after(3, 2);
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // Front to berth 1
        c.move_segment_to_berth_front(1, 2, 1);
        assert_berth_eq(&c, 0, &[0, 3]);
        assert_berth_eq(&c, 1, &[1, 2]);

        // Back to berth 0
        c.move_segment_to_berth_back(1, 2, 0);
        assert_berth_eq(&c, 0, &[0, 3, 1, 2]);
        assert_berth_eq(&c, 1, &[]);

        // Before a node: move [1,2] before 3 => [0,1,2,3]
        c.move_segment_before_node(1, 2, 3);
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_iter_berth_empty() {
        let c = DoubleChain::new(5, 2);
        assert!(collect_berth(&c, 0).is_empty());
        assert!(collect_berth(&c, 1).is_empty());
    }

    #[test]
    fn test_apply_arcrewrite_success_and_prev_maintenance() {
        // Build one berth with nodes: [0,1,2]
        let mut c = DoubleChain::new(3, 1);
        let s = c.start_of(0);
        let e = c.end_of(0);

        c.insert_after(0, s);
        c.insert_after(1, 0);
        c.insert_after(2, 1);
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        // Rewrite arc at tail=1 from expected_head=2 to new_head=e
        c.apply_arcrewrite(ArcRewrite::new(1, 2, e));
        assert_berth_eq(&c, 0, &[0, 1]);

        // Local prev maintenance checks
        assert_eq!(c.pred(e), 1); // new head's prev now tail
        assert_eq!(c.pred(2), 2); // old head's prev self-point if it pointed back to tail
        assert_eq!(c.succ(1), e);

        // Rewriting from start sentinel to end empties the berth
        c.apply_arcrewrite(ArcRewrite::new(s, 0, e));
        assert_berth_eq(&c, 0, &[]);
        assert_eq!(c.pred(e), s);
        assert_eq!(c.pred(0), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_apply_arcrewrite_guard_mismatch_panics_in_debug() {
        let mut c = DoubleChain::new(2, 1);
        let s = c.start_of(0);
        let e = c.end_of(0);

        // s -> 0 -> 1 -> e
        c.insert_after(0, s);
        c.insert_after(1, 0);

        // Mismatch: tail=0 has next=1, but we claim expected_head=e
        c.apply_arcrewrite(ArcRewrite::new(0, e, 1));
    }

    #[test]
    fn test_apply_delta_multiple_independent_rewrites_across_berths() {
        // Two berths:
        // berth 0: s0 -> 0 -> 1 -> e0
        // berth 1: s1 -> 2 -> 3 -> e1
        let mut c = DoubleChain::new(4, 2);

        let s0 = c.start_of(0);
        let e0 = c.end_of(0);
        let s1 = c.start_of(1);
        let e1 = c.end_of(1);

        c.insert_after(0, s0);
        c.insert_after(1, 0);
        c.insert_after(2, s1);
        c.insert_after(3, 2);

        assert_berth_eq(&c, 0, &[0, 1]);
        assert_berth_eq(&c, 1, &[2, 3]);

        // Delta trims the last arc on each berth
        let mut d = ChainDelta::new();
        d.push(0, 1, e0);
        d.push(2, 3, e1);

        c.apply_delta(&d);
        assert_berth_eq(&c, 0, &[0]);
        assert_berth_eq(&c, 1, &[2]);

        // Prev heads updated accordingly
        assert_eq!(c.pred(e0), 0);
        assert_eq!(c.pred(e1), 2);
        // Old heads' prevs self-point
        assert_eq!(c.pred(1), 1);
        assert_eq!(c.pred(3), 3);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_apply_delta_guard_mismatch_panics_in_debug() {
        // Build: s -> 0 -> 1 -> 2 -> e
        let mut c = DoubleChain::new(3, 1);
        let s = c.start_of(0);
        let e = c.end_of(0);
        c.insert_after(0, s);
        c.insert_after(1, 0);
        c.insert_after(2, 1);

        // First update valid; second has a guard mismatch
        let mut d = ChainDelta::new();
        d.push(0, 1, 2); // valid
        d.push(2, 0, e); // invalid: next[2]==e, not 0

        // Should panic due to guard mismatch assertion in debug
        c.apply_delta(&d);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "duplicate tail")]
    fn test_apply_delta_duplicate_tail_panics_in_debug() {
        let mut c = DoubleChain::new(2, 1);
        let s = c.start_of(0);
        let e = c.end_of(0);

        c.insert_after(0, s); // s -> 0 -> e
        c.insert_after(1, 0); // s -> 0 -> 1 -> e

        // delta contains two rewrites for the same tail (0)
        let mut d = ChainDelta::new();
        d.push(0, 1, e);
        d.push(0, e, 1); // duplicate tail

        // Should panic due to debug-only duplicate-tail guard
        c.apply_delta(&d);
    }
}
