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

pub mod err;

use crate::metaheuristic::chain::err::{
    BerthOutOfBoundsError, ChainError, NodeInsertionError, NodeIsSentinelError,
    NodeNotSkippedError, NodeOutOfBoundsError, SegmentError, SentinelInSegmentError,
};

#[derive(Debug, Clone)]
pub struct Chain {
    next: Vec<usize>,
    prev: Vec<usize>,
    start: Vec<usize>,
    end: Vec<usize>,
    base: usize,
}

impl Chain {
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
    pub fn start_of(&self, berth: usize) -> Result<usize, BerthOutOfBoundsError> {
        self.start
            .get(berth)
            .copied()
            .ok_or(BerthOutOfBoundsError::new(berth, self.num_berths()))
    }

    #[inline]
    pub fn end_of(&self, berth: usize) -> Result<usize, BerthOutOfBoundsError> {
        self.end
            .get(berth)
            .copied()
            .ok_or(BerthOutOfBoundsError::new(berth, self.num_berths()))
    }

    #[inline]
    pub fn is_skipped(&self, i: usize) -> Result<bool, NodeOutOfBoundsError> {
        self.next
            .get(i)
            .ok_or_else(|| NodeOutOfBoundsError::new(i, self.len()))
            .map(|&n| n == i)
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
    pub fn succ(&self, i: usize) -> Result<usize, NodeOutOfBoundsError> {
        self.next
            .get(i)
            .copied()
            .ok_or(NodeOutOfBoundsError::new(i, self.len()))
    }

    #[inline]
    pub fn pred(&self, i: usize) -> Result<usize, NodeOutOfBoundsError> {
        self.prev
            .get(i)
            .copied()
            .ok_or(NodeOutOfBoundsError::new(i, self.len()))
    }

    #[inline(always)]
    pub fn skip(&mut self, i: usize) -> Result<(), NodeOutOfBoundsError> {
        if i >= self.len() {
            return Err(NodeOutOfBoundsError::new(i, self.len()));
        }
        if self.is_sentinel(i) {
            return Ok(());
        }
        if self.next[i] == i {
            return Ok(());
        }

        let p = self.prev[i];
        let n = self.next[i];

        debug_assert!(p < self.len() && n < self.len());

        self.next[p] = n;
        self.prev[n] = p;
        self.next[i] = i;
        self.prev[i] = i;
        Ok(())
    }

    #[inline(always)]
    pub fn insert_after(&mut self, i: usize, anchor: usize) -> Result<(), NodeInsertionError> {
        if i >= self.len() {
            return Err(NodeInsertionError::NodeOutOfBounds(
                NodeOutOfBoundsError::new(i, self.len()),
            ));
        }

        if self.is_sentinel(i) {
            return Err(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            ));
        }

        if !self.is_skipped(i)? {
            return Err(NodeInsertionError::NodeNotSkipped(
                NodeNotSkippedError::new(i),
            ));
        }
        if anchor >= self.len() {
            return Err(NodeInsertionError::NodeOutOfBounds(
                NodeOutOfBoundsError::new(anchor, self.len()),
            ));
        }

        if self.is_tail_idx(anchor) {
            let a = self.prev[anchor];
            let an = self.next[a];

            debug_assert!(a < self.len() && an < self.len());

            self.prev[i] = a;
            self.next[i] = an;
            self.next[a] = i;
            self.prev[an] = i;
            return Ok(());
        }

        let an = self.next[anchor];
        debug_assert!(an < self.len());

        self.prev[i] = anchor;
        self.next[i] = an;
        self.next[anchor] = i;
        self.prev[an] = i;
        Ok(())
    }

    #[inline]
    pub fn insert_before(&mut self, i: usize, before: usize) -> Result<(), NodeInsertionError> {
        let a = self.prev.get(before).copied().ok_or_else(|| {
            NodeInsertionError::NodeOutOfBounds(NodeOutOfBoundsError::new(before, self.len()))
        })?;
        self.insert_after(i, a)
    }

    #[inline]
    pub fn detach_segment(&mut self, a: usize, b: usize) -> Result<(), SegmentError> {
        if a == b && self.is_skipped(a).map_err(SegmentError::from)? {
            return Ok(());
        }
        if self.is_sentinel(a) || self.is_sentinel(b) {
            return Err(SegmentError::SentinelInSegment(
                SentinelInSegmentError::new(a, b),
            ));
        }

        let pa = *self
            .prev
            .get(a)
            .ok_or_else(|| SegmentError::from(NodeOutOfBoundsError::new(a, self.len())))?;
        let nb = *self
            .next
            .get(b)
            .ok_or_else(|| SegmentError::from(NodeOutOfBoundsError::new(b, self.len())))?;

        debug_assert!(pa < self.len() && nb < self.len());

        self.next[pa] = nb;
        self.prev[nb] = pa;
        Ok(())
    }

    #[inline(always)]
    pub fn splice_after(&mut self, a: usize, b: usize, x: usize) -> Result<(), SegmentError> {
        if a >= self.len() || b >= self.len() || x >= self.len() {
            return Err(SegmentError::from(NodeOutOfBoundsError::new(
                a.max(b).max(x),
                self.len(),
            )));
        }
        if self.is_sentinel(a) || self.is_sentinel(b) {
            return Err(SegmentError::SentinelInSegment(
                SentinelInSegmentError::new(a, b),
            ));
        }

        let pa = self.prev[a];

        if x == b || x == pa || x == a {
            return Ok(());
        }

        #[cfg(debug_assertions)]
        {
            let mut cur = a;
            let mut hops = 0usize;
            while cur != b && hops <= self.len() {
                cur = self.next[cur];
                hops += 1;
            }

            debug_assert!(cur == b, "splice_after: [a..=b] is not contiguous");

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
                return Ok(());
            }
        }

        if self.is_tail_idx(x) {
            let y = self.prev[x];
            if y == b || y == pa {
                return Ok(());
            }
            let nb = self.next[b];
            let ny = self.next[y];

            debug_assert!(pa < self.len() && nb < self.len() && ny < self.len());

            self.next[pa] = nb;
            self.prev[nb] = pa;

            self.next[y] = a;
            self.prev[a] = y;
            self.next[b] = ny;
            self.prev[ny] = b;
            return Ok(());
        }

        let nb = self.next[b];
        let nx = self.next[x];

        debug_assert!(pa < self.len() && nb < self.len() && nx < self.len());

        self.next[pa] = nb;
        self.prev[nb] = pa;

        self.next[x] = a;
        self.prev[a] = x;
        self.next[b] = nx;
        self.prev[nx] = b;

        Ok(())
    }

    #[inline(always)]
    pub fn move_segment_after(&mut self, a: usize, b: usize, x: usize) -> Result<(), SegmentError> {
        if a >= self.len() || b >= self.len() || x >= self.len() {
            return Err(SegmentError::from(NodeOutOfBoundsError::new(
                a.max(b).max(x),
                self.len(),
            )));
        }
        if self.is_sentinel(a) || self.is_sentinel(b) {
            return Err(SegmentError::SentinelInSegment(
                SentinelInSegmentError::new(a, b),
            ));
        }
        if x == b || self.prev[a] == x {
            return Ok(());
        }
        if self.is_tail_idx(x) {
            self.move_before(a, x)
        } else {
            let pa = self.prev[a];
            let nb = self.next[b];
            let nx = self.next[x];

            debug_assert!(pa < self.len() && nb < self.len() && nx < self.len());

            self.next[pa] = nb;
            self.prev[nb] = pa;

            self.next[x] = a;
            self.prev[a] = x;
            self.next[b] = nx;
            self.prev[nx] = b;
            Ok(())
        }
    }

    #[inline]
    pub fn move_after(&mut self, i: usize, x: usize) -> Result<(), SegmentError> {
        if i == x {
            return Ok(());
        }

        self.move_segment_after(i, i, x)
    }

    #[inline]
    pub fn move_before(&mut self, i: usize, y: usize) -> Result<(), SegmentError> {
        let x = self
            .prev
            .get(y)
            .copied()
            .ok_or_else(|| NodeOutOfBoundsError::new(y, self.len()))?;

        debug_assert!(x < self.len());

        self.move_after(i, x)
    }

    #[inline]
    pub fn push_back_to_berth(&mut self, i: usize, berth: usize) -> Result<(), ChainError> {
        if self.is_sentinel(i) {
            return Err(ChainError::from(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            )));
        }
        let tail = self
            .end
            .get(berth)
            .copied()
            .ok_or(ChainError::BerthOutOfBounds(BerthOutOfBoundsError::new(
                berth,
                self.num_berths(),
            )))?;
        debug_assert!(tail < self.len());
        self.insert_before(i, tail).map_err(ChainError::from)
    }

    #[inline]
    pub fn push_front_to_berth(&mut self, i: usize, berth: usize) -> Result<(), ChainError> {
        if self.is_sentinel(i) {
            return Err(ChainError::from(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            )));
        }
        let head = self
            .start
            .get(berth)
            .copied()
            .ok_or(BerthOutOfBoundsError::new(berth, self.num_berths()))?;
        debug_assert!(head < self.len());
        self.insert_after(i, head).map_err(ChainError::from)
    }

    #[inline]
    pub fn move_node_to_berth_back(
        &mut self,
        i: usize,
        target_berth: usize,
    ) -> Result<(), ChainError> {
        if self.is_sentinel(i) {
            return Err(ChainError::from(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            )));
        }
        if !self.is_skipped(i).map_err(ChainError::from)? {
            self.skip(i).map_err(ChainError::from)?;
        }
        self.push_back_to_berth(i, target_berth)
    }

    #[inline]
    pub fn move_node_to_berth_front(
        &mut self,
        i: usize,
        target_berth: usize,
    ) -> Result<(), ChainError> {
        if self.is_sentinel(i) {
            return Err(ChainError::from(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            )));
        }
        if !self.is_skipped(i).map_err(ChainError::from)? {
            self.skip(i).map_err(ChainError::from)?;
        }
        self.push_front_to_berth(i, target_berth)
    }

    #[inline]
    pub fn move_node_after_node(&mut self, i: usize, anchor: usize) -> Result<(), ChainError> {
        if self.is_sentinel(i) {
            return Err(ChainError::from(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            )));
        }
        if i == anchor {
            return Ok(());
        }
        if !self.is_skipped(i).map_err(ChainError::from)? {
            self.skip(i).map_err(ChainError::from)?;
        }
        self.insert_after(i, anchor).map_err(ChainError::from)
    }

    #[inline]
    pub fn move_segment_to_berth_back(
        &mut self,
        a: usize,
        b: usize,
        target_berth: usize,
    ) -> Result<(), ChainError> {
        if self.is_sentinel(a) || self.is_sentinel(b) {
            return Err(ChainError::from(SegmentError::SentinelInSegment(
                SentinelInSegmentError::new(a, b),
            )));
        }
        let tail = self.end_of(target_berth).map_err(ChainError::from)?;
        let x = self.prev[tail];
        self.move_segment_after(a, b, x).map_err(ChainError::from)
    }

    #[inline]
    pub fn move_segment_to_berth_front(
        &mut self,
        a: usize,
        b: usize,
        target_berth: usize,
    ) -> Result<(), ChainError> {
        if self.is_sentinel(a) || self.is_sentinel(b) {
            return Err(ChainError::from(SegmentError::SentinelInSegment(
                SentinelInSegmentError::new(a, b),
            )));
        }
        let head = self.start_of(target_berth).map_err(ChainError::from)?;
        self.move_segment_after(a, b, head)
            .map_err(ChainError::from)
    }

    #[inline]
    pub fn move_segment_before_node(
        &mut self,
        a: usize,
        b: usize,
        y: usize,
    ) -> Result<(), ChainError> {
        let x = self.pred(y).map_err(ChainError::from)?; // x = y.prev
        self.move_segment_after(a, b, x).map_err(ChainError::from)
    }

    #[inline]
    pub fn move_node_before_node(&mut self, i: usize, y: usize) -> Result<(), ChainError> {
        if self.is_sentinel(i) {
            return Err(ChainError::from(NodeInsertionError::NodeIsSentinel(
                NodeIsSentinelError::new(i),
            )));
        }
        if !self.is_skipped(i).map_err(ChainError::from)? {
            self.skip(i).map_err(ChainError::from)?;
        }
        self.insert_before(i, y).map_err(ChainError::from)
    }

    #[inline]
    pub fn iter_berth<'a>(
        &'a self,
        berth: usize,
    ) -> Result<impl Iterator<Item = usize> + 'a, BerthOutOfBoundsError> {
        if berth >= self.num_berths() {
            return Err(BerthOutOfBoundsError::new(berth, self.num_berths()));
        }

        let s = self.start[berth];
        let e = self.end[berth];

        debug_assert!(s < self.len());
        debug_assert!(e < self.len());

        Ok(ChainIter::new(&self.next, self.next[s], e))
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
    use super::err::*;
    use super::*;

    fn collect_berth(chain: &Chain, berth: usize) -> Vec<usize> {
        chain.iter_berth(berth).unwrap().collect()
    }

    fn assert_berth_eq(chain: &Chain, berth: usize, expected: &[usize]) {
        let got = collect_berth(chain, berth);
        assert_eq!(got, expected, "berth {} content mismatch", berth);

        // Also verify ring around sentinels
        let s = chain.start_of(berth).unwrap();
        let e = chain.end_of(berth).unwrap();

        // Walk forward from start to end through expected
        let mut cur = chain.succ(s).unwrap();
        for &n in expected {
            assert_eq!(cur, n);
            cur = chain.succ(cur).unwrap();
        }
        assert_eq!(cur, e);

        // Walk backward from end to start through expected reversed
        let mut cur = chain.pred(e).unwrap();
        for &n in expected.iter().rev() {
            assert_eq!(cur, n);
            cur = chain.pred(cur).unwrap();
        }
        assert_eq!(cur, s);
    }

    #[test]
    fn test_new_initial_state_and_invariants() {
        let n = 5;
        let b = 3;
        let c = Chain::new(n, b);

        assert_eq!(c.len(), n + 2 * b);
        assert_eq!(c.num_berths(), b);

        for i in 0..n {
            assert!(c.is_skipped(i).unwrap());
            assert!(!c.is_sentinel(i));
            assert!(!c.is_head(i));
            assert!(!c.is_tail(i));
            assert_eq!(c.succ(i).unwrap(), i);
            assert_eq!(c.pred(i).unwrap(), i);
        }

        for berth in 0..b {
            let s = c.start_of(berth).unwrap();
            let e = c.end_of(berth).unwrap();
            assert!(c.is_sentinel(s) && c.is_head(s) && !c.is_tail(s));
            assert!(c.is_sentinel(e) && !c.is_head(e) && c.is_tail(e));

            assert_eq!(c.succ(s).unwrap(), e);
            assert_eq!(c.pred(s).unwrap(), s); // by construction
            assert_eq!(c.succ(e).unwrap(), e);
            assert_eq!(c.pred(e).unwrap(), s);

            assert_berth_eq(&c, berth, &[]);
        }

        // OOB checks
        assert!(matches!(
            c.start_of(b).unwrap_err(),
            BerthOutOfBoundsError { .. }
        ));
        assert!(matches!(
            c.end_of(b).unwrap_err(),
            BerthOutOfBoundsError { .. }
        ));
        assert!(matches!(
            c.succ(c.len()).unwrap_err(),
            NodeOutOfBoundsError { .. }
        ));
        assert!(matches!(
            c.pred(c.len()).unwrap_err(),
            NodeOutOfBoundsError { .. }
        ));
    }

    #[test]
    fn test_is_sentinel_head_tail_checks() {
        let n = 4;
        let b = 2;
        let c = Chain::new(n, b);
        for i in 0..n {
            assert!(!c.is_sentinel(i));
            assert!(!c.is_head(i));
            assert!(!c.is_tail(i));
        }
        for berth in 0..b {
            let s = c.start_of(berth).unwrap();
            let e = c.end_of(berth).unwrap();
            assert!(c.is_sentinel(s) && c.is_head(s) && !c.is_tail(s));
            assert!(c.is_sentinel(e) && !c.is_head(e) && c.is_tail(e));
        }
    }

    #[test]
    fn test_skip_behavior() {
        let mut c = Chain::new(3, 1);

        // Skipping already-skipped is no-op
        for i in 0..3 {
            c.skip(i).unwrap();
            assert!(c.is_skipped(i).unwrap());
        }

        // Skipping a sentinel is no-op
        let s = c.start_of(0).unwrap();
        let e = c.end_of(0).unwrap();
        c.skip(s).unwrap();
        c.skip(e).unwrap();
        assert_eq!(c.succ(s).unwrap(), e);
        assert_eq!(c.pred(e).unwrap(), s);

        // Insert node then skip to detach
        c.insert_after(0, s).unwrap();
        c.insert_after(1, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 1]);
        c.skip(1).unwrap();
        assert!(c.is_skipped(1).unwrap());
        assert_berth_eq(&c, 0, &[0]);
    }

    #[test]
    fn test_insert_after_normal_and_tail_anchor() {
        let mut c = Chain::new(4, 1);
        let s = c.start_of(0).unwrap();
        let e = c.end_of(0).unwrap();

        // Errors: i OOB, anchor OOB, i is sentinel, i not skipped
        assert!(matches!(
            c.insert_after(999, s).unwrap_err(),
            NodeInsertionError::NodeOutOfBounds(_)
        ));
        assert!(matches!(
            c.insert_after(s, s).unwrap_err(),
            NodeInsertionError::NodeIsSentinel(_)
        ));
        assert!(matches!(
            c.insert_after(0, 999).unwrap_err(),
            NodeInsertionError::NodeOutOfBounds(_)
        ));

        // Normal: insert 0 after head
        c.insert_after(0, s).unwrap();
        assert_berth_eq(&c, 0, &[0]);

        // Not skipped error: try inserting 0 again
        assert!(matches!(
            c.insert_after(0, s).unwrap_err(),
            NodeInsertionError::NodeNotSkipped(_)
        ));

        // Tail-anchor path: inserting after tail should place before tail
        c.insert_after(1, e).unwrap();
        assert_berth_eq(&c, 0, &[0, 1]);

        // Insert in the middle
        c.insert_after(2, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_insert_before_variants() {
        let mut c = Chain::new(4, 1);
        let s = c.start_of(0).unwrap();
        let e = c.end_of(0).unwrap();

        c.insert_after(0, s).unwrap();
        c.insert_after(2, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2]);

        // insert_before before tail should put node just before end
        c.insert_before(1, e).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // insert_before before first node -> becomes new first
        c.insert_before(3, 0).unwrap();
        assert_berth_eq(&c, 0, &[3, 0, 2, 1]);
    }

    #[test]
    fn test_detach_segment_basic_and_errors() {
        let mut c = Chain::new(6, 1);
        let s = c.start_of(0).unwrap();

        // No-op for skipped a==b
        assert!(c.detach_segment(0, 0).is_ok());

        // Setup: 0,1,2,3 in berth
        c.insert_after(0, s).unwrap();
        c.insert_after(1, 0).unwrap();
        c.insert_after(2, 1).unwrap();
        c.insert_after(3, 2).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // Error if sentinel in segment
        let e = c.end_of(0).unwrap();
        assert!(matches!(
            c.detach_segment(0, e).unwrap_err(),
            SegmentError::SentinelInSegment(_)
        ));

        // Detach [1..=2]
        c.detach_segment(1, 2).unwrap();
        assert_berth_eq(&c, 0, &[0, 3]);

        // The detached segment [1..=2] should still be linked together
        assert_eq!(c.succ(1).unwrap(), 2);
        assert_eq!(c.pred(2).unwrap(), 1);
    }

    #[test]
    fn test_splice_after_moves_segment() {
        let mut c = Chain::new(6, 1);
        let s = c.start_of(0).unwrap();

        // Build: 0,1,2,3
        for i in 0..4 {
            c.insert_after(i, if i == 0 { s } else { i - 1 }).unwrap();
        }
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // Move [1..=2] after 3 -> becomes 0,3,1,2
        c.splice_after(1, 2, 3).unwrap();
        assert_berth_eq(&c, 0, &[0, 3, 1, 2]);

        // No-op when x == b
        c.splice_after(1, 2, 2).unwrap();
        assert_berth_eq(&c, 0, &[0, 3, 1, 2]);
    }

    #[test]
    fn test_move_segment_after_normal_and_tail() {
        let mut c = Chain::new(6, 2);

        // Place 0,1,2 in berth 0; 3 in berth 1
        let s0 = c.start_of(0).unwrap();
        let s1 = c.start_of(1).unwrap();
        let t1 = c.end_of(1).unwrap();

        c.insert_after(0, s0).unwrap();
        c.insert_after(1, 0).unwrap();
        c.insert_after(2, 1).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        c.insert_after(3, s1).unwrap();
        assert_berth_eq(&c, 1, &[3]);

        // Move [1..=2] after 3 -> berth1 becomes [3,1,2], berth0 becomes [0]
        c.move_segment_after(1, 2, 3).unwrap();
        assert_berth_eq(&c, 0, &[0]);
        assert_berth_eq(&c, 1, &[3, 1, 2]);

        // Tail case: moving [0..=0] "after tail" means before tail sentinel
        let tail1 = t1; // == end_of(1)
        c.move_segment_after(0, 0, tail1).unwrap();
        assert_berth_eq(&c, 1, &[3, 1, 2, 0]);

        // No-op cases
        c.move_segment_after(1, 2, 2).unwrap(); // x == b
        c.move_segment_after(1, 2, c.pred(1).unwrap()).unwrap(); // prev[a] == x
        assert_berth_eq(&c, 1, &[3, 1, 2, 0]);

        // OOB and sentinel-in-segment errors
        assert!(matches!(
            c.move_segment_after(999, 2, 3).unwrap_err(),
            SegmentError::NodeOutOfBounds(_)
        ));
        assert!(matches!(
            c.move_segment_after(c.start_of(1).unwrap(), 2, 3)
                .unwrap_err(),
            SegmentError::SentinelInSegment(_)
        ));
    }

    #[test]
    fn test_move_after_and_move_before_wrappers() {
        let mut c = Chain::new(5, 1);
        let s = c.start_of(0).unwrap();
        let e = c.end_of(0).unwrap();

        c.insert_after(0, s).unwrap();
        c.insert_after(1, 0).unwrap();
        c.insert_after(2, 1).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        // move_after: move 2 after 0 -> [0,2,1]
        c.move_after(2, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // move_after no-op when i == x
        c.move_after(0, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // move_before: move 1 before end (i.e., to back) -> [0,2,1]
        c.move_before(1, e).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);

        // OOB y in move_before
        assert!(matches!(
            c.move_before(1, 999).unwrap_err(),
            SegmentError::NodeOutOfBounds(_)
        ));
    }

    #[test]
    fn test_push_front_and_back_to_berth_and_errors() {
        let mut c = Chain::new(6, 2);

        // Reject sentinels as payload
        let s0 = c.start_of(0).unwrap();
        assert!(matches!(
            c.push_back_to_berth(s0, 0).unwrap_err(),
            ChainError::NodeInsertion(NodeInsertionError::NodeIsSentinel(_))
        ));

        // Push into empty berth 0
        c.push_back_to_berth(0, 0).unwrap();
        assert_berth_eq(&c, 0, &[0]);

        // Push front into same berth
        c.push_front_to_berth(1, 0).unwrap();
        assert_berth_eq(&c, 0, &[1, 0]);

        // Add to other berth
        c.push_back_to_berth(2, 1).unwrap();
        c.push_front_to_berth(3, 1).unwrap();
        assert_berth_eq(&c, 1, &[3, 2]);
    }

    #[test]
    fn test_move_node_to_berth_front_and_back() {
        let mut c = Chain::new(6, 2);

        // Put nodes into berth 0
        let s0 = c.start_of(0).unwrap();
        c.insert_after(0, s0).unwrap();
        c.insert_after(1, 0).unwrap();
        c.insert_after(2, 1).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2]);

        // Move node 1 to front of berth 1
        c.move_node_to_berth_front(1, 1).unwrap();
        assert_berth_eq(&c, 0, &[0, 2]);
        assert_berth_eq(&c, 1, &[1]);

        // Move same node to back of berth 0
        c.move_node_to_berth_back(1, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);
        assert_berth_eq(&c, 1, &[]);
    }

    #[test]
    fn test_move_node_after_node_across_berths() {
        let mut c = Chain::new(6, 2);

        let s0 = c.start_of(0).unwrap();
        let s1 = c.start_of(1).unwrap();

        c.insert_after(0, s0).unwrap();
        c.insert_after(1, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 1]);

        c.insert_after(2, s1).unwrap();
        assert_berth_eq(&c, 1, &[2]);

        // Move node 2 after node 0 (cross-berth)
        c.move_node_after_node(2, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);
        assert_berth_eq(&c, 1, &[]);

        // No-op when i == anchor
        c.move_node_after_node(0, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_move_segment_to_berth_front_and_back_and_before_node() {
        let mut c = Chain::new(8, 2);

        let s0 = c.start_of(0).unwrap();
        c.insert_after(0, s0).unwrap();
        c.insert_after(1, 0).unwrap();
        c.insert_after(2, 1).unwrap();
        c.insert_after(3, 2).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // Front to berth 1
        c.move_segment_to_berth_front(1, 2, 1).unwrap();
        assert_berth_eq(&c, 0, &[0, 3]);
        assert_berth_eq(&c, 1, &[1, 2]);

        // Back to berth 0
        c.move_segment_to_berth_back(1, 2, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 3, 1, 2]);
        assert_berth_eq(&c, 1, &[]);

        // Before a node: move [1,2] before 3 => [0,1,2,3]
        c.move_segment_before_node(1, 2, 3).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_iter_berth_empty_and_oob() {
        let c = Chain::new(5, 2);
        assert!(collect_berth(&c, 0).is_empty());
        assert!(collect_berth(&c, 1).is_empty());

        assert!(c.iter_berth(999).is_err());
    }

    #[test]
    fn test_splice_after_anchor_inside_segment_is_noop() {
        let mut c = Chain::new(6, 1);
        let s = c.start_of(0).unwrap();
        // build 0,1,2,3
        c.insert_after(0, s).unwrap();
        c.insert_after(1, 0).unwrap();
        c.insert_after(2, 1).unwrap();
        c.insert_after(3, 2).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);

        // x=1 (inside [1..=2] but not endpoints) => no-op
        c.splice_after(1, 2, 1).unwrap();
        assert_berth_eq(&c, 0, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_splice_after_tail_anchors_before_tail() {
        let mut c = Chain::new(6, 1);
        let s = c.start_of(0).unwrap();
        let t = c.end_of(0).unwrap();
        c.insert_after(0, s).unwrap();
        c.insert_after(1, 0).unwrap();
        assert_berth_eq(&c, 0, &[0, 1]);

        // move [0..=0] after tail => before tail
        c.splice_after(0, 0, t).unwrap();
        assert_berth_eq(&c, 0, &[1, 0]);
    }
}
