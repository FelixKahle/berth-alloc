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

use crate::metaheuristic::chainox::err::{
    DestInsideRangeError, DifferentPathsError, NodeAlreadyActiveError, NodeInactiveError,
    NodeNotInactiveError, SpliceRangeError,
};
use std::{iter::FusedIterator, num::NonZeroUsize};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeKey(NonZeroUsize);

impl NodeKey {
    #[inline]
    pub fn get(self) -> usize {
        self.0.get()
    }

    #[inline]
    fn from_index(i1: usize) -> Self {
        Self(NonZeroUsize::new(i1).expect("1-based"))
    }
}

impl std::fmt::Display for NodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeKey({})", self.0)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PathKey(usize);

impl PathKey {
    #[inline]
    pub const fn to_raw(self) -> usize {
        self.0
    }

    #[inline]
    pub const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
}

impl From<PathKey> for usize {
    #[inline]
    fn from(val: PathKey) -> Self {
        val.0
    }
}

impl std::fmt::Display for PathKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PathKey({})", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct PathIterator<'a> {
    next_raw: &'a [usize],
    cur: usize,
}

impl<'a> PathIterator<'a> {
    fn new(top: &'a PathArena, p: PathKey) -> Self {
        Self {
            next_raw: &top.next_raw,
            cur: top.head_raw_of(p),
        }
    }
}

impl<'a> Iterator for PathIterator<'a> {
    type Item = NodeKey;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == 0 {
            return None;
        }
        let cur = self.cur;
        self.cur = self.next_raw[cur];
        Some(NodeKey::from_index(cur))
    }
}

impl<'a> FusedIterator for PathIterator<'a> {}

#[derive(Debug, Clone)]
pub struct PathArena {
    prev_raw: Vec<usize>,    // 0 = null
    next_raw: Vec<usize>,    // 0 = null
    path_of_raw: Vec<usize>, // path+1; 0 = inactive
    free_list: Vec<usize>,   // stack of reusable node indices (1-based)
    head_raw: Vec<usize>,    // 0 = empty
    tail_raw: Vec<usize>,    // 0 = empty
    len: Vec<usize>,
}

#[inline(always)]
fn enc(opt: Option<NodeKey>) -> usize {
    opt.map_or(0, |k| k.get())
}

#[inline(always)]
fn dec(raw: usize) -> Option<NodeKey> {
    NonZeroUsize::new(raw).map(|nz| NodeKey::from_index(nz.get()))
}

impl PathArena {
    #[inline(always)]
    fn raw(&self, u: NodeKey) -> usize {
        u.get()
    }

    #[inline(always)]
    fn pi(&self, p: PathKey) -> usize {
        p.to_raw()
    }

    #[inline(always)]
    fn head_raw_of(&self, p: PathKey) -> usize {
        self.head_raw[self.pi(p)]
    }

    #[inline(always)]
    fn tail_raw_of(&self, p: PathKey) -> usize {
        self.tail_raw[self.pi(p)]
    }

    #[inline(always)]
    fn next_raw_of(&self, u: NodeKey) -> usize {
        self.next_raw[self.raw(u)]
    }

    #[inline(always)]
    fn prev_raw_of(&self, u: NodeKey) -> usize {
        self.prev_raw[self.raw(u)]
    }

    #[inline(always)]
    fn set_head_raw(&mut self, p: PathKey, raw: usize) {
        let i = self.pi(p);
        self.head_raw[i] = raw;
    }

    #[inline(always)]
    fn set_tail_raw(&mut self, p: PathKey, raw: usize) {
        let i = self.pi(p);
        self.tail_raw[i] = raw;
    }

    #[inline(always)]
    fn set_next_raw(&mut self, u: NodeKey, raw: usize) {
        let i = self.raw(u);
        self.next_raw[i] = raw;
    }

    #[inline(always)]
    fn set_prev_raw(&mut self, u: NodeKey, raw: usize) {
        let i = self.raw(u);
        self.prev_raw[i] = raw;
    }

    #[inline]
    pub fn with_capacities(node_cap: usize, path_cap: usize) -> Self {
        Self {
            prev_raw: vec![0; node_cap + 1],
            next_raw: vec![0; node_cap + 1],
            path_of_raw: vec![0; node_cap + 1],
            free_list: Vec::new(),
            head_raw: Vec::with_capacity(path_cap),
            tail_raw: Vec::with_capacity(path_cap),
            len: Vec::with_capacity(path_cap),
        }
    }

    #[inline]
    pub fn new(initial_nodes: usize, initial_paths: usize) -> Self {
        let mut s = Self::with_capacities(initial_nodes, initial_paths);
        for _ in 0..initial_paths {
            s.create_path();
        }
        s
    }

    #[inline]
    pub fn reserve_nodes(&mut self, additional: usize) {
        self.prev_raw.reserve(additional);
        self.next_raw.reserve(additional);
        self.path_of_raw.reserve(additional);
    }

    #[inline]
    pub fn reserve_paths(&mut self, additional: usize) {
        self.head_raw.reserve(additional);
        self.tail_raw.reserve(additional);
        self.len.reserve(additional);
    }

    #[inline]
    pub fn create_path(&mut self) -> PathKey {
        self.head_raw.push(0);
        self.tail_raw.push(0);
        self.len.push(0);
        PathKey::from_raw(self.head_raw.len() - 1)
    }

    #[inline]
    pub fn alloc_node(&mut self) -> NodeKey {
        if let Some(i1) = self.free_list.pop() {
            debug_assert_eq!(self.prev_raw[i1], 0);
            debug_assert_eq!(self.next_raw[i1], 0);
            debug_assert_eq!(self.path_of_raw[i1], 0);
            return NodeKey::from_index(i1);
        }
        let i1 = self.prev_raw.len();
        self.prev_raw.push(0);
        self.next_raw.push(0);
        self.path_of_raw.push(0);
        NodeKey::from_index(i1)
    }

    #[inline]
    pub fn free_node(&mut self, u: NodeKey) -> Result<(), NodeNotInactiveError<NodeKey>> {
        let i = u.get();
        if self.path_of_raw[i] != 0 {
            return Err(NodeNotInactiveError::new(u));
        }
        debug_assert_eq!(self.prev_raw[i], 0);
        debug_assert_eq!(self.next_raw[i], 0);
        self.free_list.push(i);
        Ok(())
    }

    #[inline]
    pub fn num_paths(&self) -> usize {
        self.head_raw.len()
    }

    #[inline]
    pub fn is_active(&self, u: NodeKey) -> bool {
        self.path_of_raw[u.get()] != 0
    }

    #[inline]
    pub fn path_of(&self, u: NodeKey) -> Option<PathKey> {
        let raw = self.path_of_raw[u.get()];
        (raw != 0).then(|| PathKey::from_raw(raw - 1))
    }

    #[inline]
    pub fn path_len(&self, p: PathKey) -> usize {
        self.len[self.pi(p)]
    }

    #[inline]
    pub fn head(&self, p: PathKey) -> Option<NodeKey> {
        dec(self.head_raw_of(p))
    }

    #[inline]
    pub fn tail(&self, p: PathKey) -> Option<NodeKey> {
        dec(self.tail_raw_of(p))
    }

    #[inline]
    pub fn next(&self, u: NodeKey) -> Option<NodeKey> {
        dec(self.next_raw_of(u))
    }

    #[inline]
    pub fn prev(&self, u: NodeKey) -> Option<NodeKey> {
        dec(self.prev_raw_of(u))
    }

    #[inline]
    fn ensure_active(&self, u: NodeKey) -> Result<PathKey, NodeInactiveError<NodeKey>> {
        self.path_of(u).ok_or_else(|| NodeInactiveError::new(u))
    }

    #[inline]
    pub fn is_head(&self, p: PathKey, u: NodeKey) -> bool {
        self.head(p) == Some(u)
    }

    #[inline]
    pub fn is_tail(&self, p: PathKey, u: NodeKey) -> bool {
        self.tail(p) == Some(u)
    }

    #[inline]
    pub fn neighbors(&self, u: NodeKey) -> (Option<NodeKey>, Option<NodeKey>) {
        (self.prev(u), self.next(u))
    }

    #[inline]
    pub fn num_active_nodes(&self) -> usize {
        self.path_of_raw.iter().skip(1).filter(|&&r| r != 0).count()
    }

    #[inline]
    fn ensure_same_path(
        &self,
        a: NodeKey,
        b: NodeKey,
    ) -> Result<PathKey, SpliceRangeError<NodeKey, PathKey>> {
        let pa = self.ensure_active(a)?;
        let pb = self.ensure_active(b)?;
        if pa == pb {
            Ok(pa)
        } else {
            Err(DifferentPathsError::new(pa, pb).into())
        }
    }

    #[inline]
    fn ensure_dest_on_path(
        &self,
        p: PathKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        if let Some(d) = dest {
            let pd = self.ensure_active(d)?;
            if pd != p {
                return Err(DifferentPathsError::new(p, pd).into());
            }
        }
        Ok(())
    }

    #[inline]
    fn detach_range(
        &mut self,
        p: PathKey,
        first: NodeKey,
        last: NodeKey,
    ) -> (Option<NodeKey>, Option<NodeKey>) {
        let before = dec(self.prev_raw_of(first));
        let after = dec(self.next_raw_of(last));

        match before {
            Some(b) => self.set_next_raw(b, enc(after)),
            None => self.set_head_raw(p, enc(after)),
        }
        match after {
            Some(a) => self.set_prev_raw(a, enc(before)),
            None => self.set_tail_raw(p, enc(before)),
        }
        (before, after)
    }

    #[inline]
    fn splice_after(&mut self, p: PathKey, first: NodeKey, last: NodeKey, dest: Option<NodeKey>) {
        let (prev, next) = match dest {
            Some(d) => (Some(d), dec(self.next_raw_of(d))),
            None => (None, dec(self.head_raw_of(p))),
        };

        self.set_prev_raw(first, enc(prev));
        self.set_next_raw(last, enc(next));

        match prev {
            Some(pr) => self.set_next_raw(pr, self.raw(first)),
            None => self.set_head_raw(p, self.raw(first)),
        }
        match next {
            Some(nx) => self.set_prev_raw(nx, self.raw(last)),
            None => self.set_tail_raw(p, self.raw(last)),
        }
    }

    #[inline]
    fn link_between(
        &mut self,
        p: PathKey,
        u: NodeKey,
        prev: Option<NodeKey>,
        next: Option<NodeKey>,
    ) {
        self.set_prev_raw(u, enc(prev));
        self.set_next_raw(u, enc(next));

        match prev {
            Some(pr) => self.set_next_raw(pr, self.raw(u)),
            None => self.set_head_raw(p, self.raw(u)),
        }
        match next {
            Some(nx) => self.set_prev_raw(nx, self.raw(u)),
            None => self.set_tail_raw(p, self.raw(u)),
        }
    }

    #[inline]
    pub fn activate_after(
        &mut self,
        p: PathKey,
        u: NodeKey,
        anchor: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        if self.is_active(u) {
            return Err(NodeAlreadyActiveError::new(u));
        }
        let next = anchor
            .map(|a| dec(self.next_raw_of(a)))
            .unwrap_or_else(|| dec(self.head_raw_of(p)));
        self.link_between(p, u, anchor, next);
        let ui = self.raw(u);
        let pi = self.pi(p);
        self.path_of_raw[ui] = pi + 1;
        self.len[pi] += 1;
        Ok(())
    }

    #[inline]
    pub fn push_front(
        &mut self,
        p: PathKey,
        u: NodeKey,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        self.activate_after(p, u, None)
    }

    #[inline]
    pub fn push_back(
        &mut self,
        p: PathKey,
        u: NodeKey,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        let anchor = self.tail(p);
        self.activate_after(p, u, anchor)
    }

    #[inline]
    pub fn deactivate(
        &mut self,
        u: NodeKey,
    ) -> Result<(PathKey, Option<NodeKey>, Option<NodeKey>), NodeInactiveError<NodeKey>> {
        let p = self.ensure_active(u)?;
        let pi = self.pi(p);
        let prev = dec(self.prev_raw_of(u));
        let next = dec(self.next_raw_of(u));

        match prev {
            Some(pr) => self.set_next_raw(pr, enc(next)),
            None => self.set_head_raw(p, enc(next)),
        }
        match next {
            Some(nx) => self.set_prev_raw(nx, enc(prev)),
            None => self.set_tail_raw(p, enc(prev)),
        }

        let ui = self.raw(u);
        self.set_prev_raw(u, 0);
        self.set_next_raw(u, 0);
        self.path_of_raw[ui] = 0;
        self.len[pi] -= 1;
        Ok((p, prev, next))
    }

    #[inline]
    pub fn remove(
        &mut self,
        u: NodeKey,
    ) -> Result<(PathKey, Option<NodeKey>, Option<NodeKey>), NodeInactiveError<NodeKey>> {
        self.deactivate(u)
    }

    pub fn move_range_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let p = self.ensure_same_path(first, last)?;
        self.ensure_dest_on_path(p, dest)?;

        // Fast no-ops:
        if let Some(d) = dest {
            if d == last || self.prev(first) == Some(d) {
                return Ok(());
            }
            // forbid dest inside range
            if self.is_in_range(p, first, last, d) {
                return Err(DestInsideRangeError::new(first, last, d).into());
            }
        } else if self.head(p) == Some(first) {
            return Ok(());
        }

        let _ = self.detach_range(p, first, last);
        self.splice_after(p, first, last, dest);
        Ok(())
    }

    #[inline]
    pub fn splice_range_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        self.move_range_after(first, last, dest)
    }

    #[inline]
    pub fn insert_after(
        &mut self,
        p: PathKey,
        u: NodeKey,
        anchor: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        self.activate_after(p, u, anchor)
    }

    #[inline]
    pub fn insert_before(
        &mut self,
        p: PathKey,
        u: NodeKey,
        before: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        match before {
            Some(b) => self.activate_after(p, u, self.prev(b)),
            None => self.push_back(p, u),
        }
    }

    #[inline]
    pub fn move_node_to_path_back(
        &mut self,
        u: NodeKey,
        dst: PathKey,
    ) -> Result<(), NodeInactiveError<NodeKey>> {
        let _ = self.deactivate(u)?;
        let _ = self.push_back(dst, u);
        Ok(())
    }

    pub fn reverse_range(
        &mut self,
        first: NodeKey,
        last: NodeKey,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let p = self.ensure_same_path(first, last)?;
        if first == last {
            return Ok(());
        }

        let before = self.prev(first);
        let after = self.next(last);

        let mut cur = first;
        loop {
            let idx = self.raw(cur);
            let n_raw = self.next_raw[idx];
            std::mem::swap(&mut self.prev_raw[idx], &mut self.next_raw[idx]);
            if cur == last {
                break;
            }
            cur = NodeKey::from_index(n_raw);
        }

        self.set_prev_raw(last, enc(before));
        self.set_next_raw(first, enc(after));

        match before {
            Some(b) => self.set_next_raw(b, self.raw(last)),
            None => self.set_head_raw(p, self.raw(last)),
        }
        match after {
            Some(a) => self.set_prev_raw(a, self.raw(first)),
            None => self.set_tail_raw(p, self.raw(first)),
        }

        Ok(())
    }

    #[inline]
    pub fn move_after(
        &mut self,
        u: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        self.move_range_after(u, u, dest)
    }

    /// Move `u` to be **before** `dest` on the same path (`None` -> back).
    #[inline]
    pub fn move_before(
        &mut self,
        u: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        match dest {
            Some(d) => self.move_range_after(u, u, self.prev(d)),
            None => {
                // move to back: after current tail (unless already tail)
                let p = self.ensure_active(u)?;
                let tail = self.tail(p);
                if tail == Some(u) {
                    return Ok(());
                }
                self.move_range_after(u, u, tail)
            }
        }
    }

    #[inline]
    pub fn move_to_front(&mut self, u: NodeKey) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let p = self.ensure_active(u)?;
        if self.is_head(p, u) {
            return Ok(());
        }
        self.move_range_after(u, u, None)
    }

    #[inline]
    pub fn move_to_back(&mut self, u: NodeKey) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let p = self.ensure_active(u)?;
        let tail = self.tail(p);
        if tail == Some(u) {
            return Ok(());
        }
        self.move_range_after(u, u, tail)
    }

    #[inline]
    pub fn reverse_path(&mut self, p: PathKey) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        match (self.head(p), self.tail(p)) {
            (Some(h), Some(t)) if h != t => self.reverse_range(h, t),
            _ => Ok(()),
        }
    }

    #[inline]
    pub fn kth(&self, p: PathKey, k: usize) -> Option<NodeKey> {
        let mut cur = self.head_raw_of(p);
        let mut i = 0usize;
        while cur != 0 && i < k {
            cur = self.next_raw[cur];
            i += 1;
        }
        dec(cur)
    }

    #[inline]
    pub fn distance_forward(&self, a: NodeKey, b: NodeKey) -> Option<usize> {
        let pa = self.path_of(a)?;
        if self.path_of(b)? != pa {
            return None;
        }
        let mut cur = a;
        let mut d = 0usize;
        loop {
            if cur == b {
                return Some(d);
            }
            cur = self.next(cur)?;
            d += 1;
        }
    }

    #[inline]
    pub fn clear_path(&mut self, p: PathKey) {
        let pi = self.pi(p);

        let mut cur = self.head_raw[pi];
        while cur != 0 {
            let next = self.next_raw[cur];
            self.prev_raw[cur] = 0;
            self.next_raw[cur] = 0;
            self.path_of_raw[cur] = 0;
            cur = next;
        }

        self.head_raw[pi] = 0;
        self.tail_raw[pi] = 0;
        self.len[pi] = 0;
    }

    pub fn iter_path<'a>(&'a self, p: PathKey) -> PathIterator<'a> {
        PathIterator::new(self, p)
    }

    #[cfg(debug_assertions)]
    pub fn check_path(&self, p: PathKey) {
        let mut count = 0;
        let mut cur = self.head_raw_of(p);
        let mut prev = 0usize;
        while cur != 0 {
            assert!(cur < self.prev_raw.len());
            assert_eq!(self.path_of_raw[cur], self.pi(p) + 1);
            assert_eq!(self.prev_raw[cur], prev);
            prev = cur;
            cur = self.next_raw[cur];
            count += 1;
        }
        assert_eq!(self.tail_raw_of(p), prev);
        assert_eq!(self.len[self.pi(p)], count);
        if count == 0 {
            assert_eq!(self.head_raw_of(p), 0);
            assert_eq!(self.tail_raw_of(p), 0);
        }
    }

    #[inline]
    fn is_in_range(&self, p: PathKey, first: NodeKey, last: NodeKey, x: NodeKey) -> bool {
        debug_assert_eq!(self.path_of(first), Some(p));
        debug_assert_eq!(self.path_of(last), Some(p));
        debug_assert_eq!(self.path_of(x), Some(p));

        let mut cur = first;
        loop {
            if cur == x {
                return true;
            }
            if cur == last {
                return false;
            }
            cur = dec(self.next_raw_of(cur)).expect("range must be contiguous");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_nodes(top: &mut PathArena, n: usize) -> Vec<NodeKey> {
        (0..n).map(|_| top.alloc_node()).collect()
    }

    fn to_vec(top: &PathArena, p: PathKey) -> Vec<NodeKey> {
        top.iter_path(p).collect()
    }

    fn back_to_front(top: &PathArena, p: PathKey) -> Vec<NodeKey> {
        let mut v = Vec::new();
        let mut cur = top.tail(p);
        while let Some(u) = cur {
            v.push(u);
            cur = top.prev(u);
        }
        v
    }

    fn assert_path_eq(top: &PathArena, p: PathKey, expected: &[NodeKey]) {
        assert_eq!(
            top.path_len(p),
            expected.len(),
            "path_len mismatch for p={}",
            p
        );
        let head = top.head(p);
        let tail = top.tail(p);
        match expected.first() {
            Some(&e) => assert_eq!(head, Some(e), "head mismatch for p={}", p),
            None => assert_eq!(head, None, "head should be None on empty path for p={}", p),
        }
        match expected.last() {
            Some(&e) => assert_eq!(tail, Some(e), "tail mismatch for p={}", p),
            None => assert_eq!(tail, None, "tail should be None on empty path for p={}", p),
        }
        let actual = to_vec(top, p);
        assert_eq!(actual, expected, "forward order mismatch for p={}", p);
        let mut expected_rev = expected.to_vec();
        expected_rev.reverse();
        let actual_rev = back_to_front(top, p);
        assert_eq!(
            actual_rev, expected_rev,
            "backward order mismatch for p={}",
            p
        );

        #[cfg(debug_assertions)]
        {
            top.check_path(p);
        }
    }

    #[test]
    fn test_basic_path_creation_and_lengths() {
        let mut top = PathArena::with_capacities(0, 0);
        assert_eq!(top.num_paths(), 0);
        let p0 = top.create_path();
        let p1 = top.create_path();
        let p2 = top.create_path();
        assert_eq!(top.num_paths(), 3);
        assert_eq!(top.path_len(p0), 0);
        assert_eq!(top.path_len(p1), 0);
        assert_eq!(top.path_len(p2), 0);
        assert_eq!(top.head(p0), None);
        assert_eq!(top.tail(p0), None);
        assert_eq!(to_vec(&top, p0), vec![]);
    }

    #[test]
    fn test_alloc_and_free_nodes_reuse_stack_lifo() {
        let mut top = PathArena::with_capacities(0, 1);
        let p = top.create_path();

        let a = top.alloc_node();
        let b = top.alloc_node();
        let c = top.alloc_node();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert!(!top.is_active(a));
        assert!(!top.is_active(b));

        assert!(top.free_node(b).is_ok());
        assert!(top.free_node(a).is_ok());

        let a2 = top.alloc_node();
        assert_eq!(a2, a);
        let b2 = top.alloc_node();
        assert_eq!(b2, b);

        assert!(top.push_back(p, c).is_ok());
        match top.free_node(c) {
            Err(e) => assert_eq!(e.node(), c),
            Ok(_) => panic!("expected NodeNotInactiveError, got Ok"),
        }

        assert!(top.deactivate(c).is_ok());
        assert!(top.free_node(c).is_ok());
    }

    #[test]
    fn test_push_front_back_and_links_are_correct() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);

        let nodes = mk_nodes(&mut top, 4);
        let (a, b, c, d) = (nodes[0], nodes[1], nodes[2], nodes[3]);

        assert!(top.push_back(p, a).is_ok());
        assert!(top.push_back(p, b).is_ok());
        assert_path_eq(&top, p, &[a, b]);

        assert!(top.push_front(p, c).is_ok());
        assert_path_eq(&top, p, &[c, a, b]);

        assert!(top.insert_after(p, d, Some(a)).is_ok());
        assert_path_eq(&top, p, &[c, a, d, b]);

        assert_eq!(top.next(c), Some(a));
        assert_eq!(top.prev(c), None);
        assert_eq!(top.next(a), Some(d));
        assert_eq!(top.prev(a), Some(c));
        assert_eq!(top.next(d), Some(b));
        assert_eq!(top.prev(d), Some(a));
        assert_eq!(top.next(b), None);
        assert_eq!(top.prev(b), Some(d));
    }

    #[test]
    fn test_activation_errors_when_already_active() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let u = top.alloc_node();
        assert!(top.push_back(p, u).is_ok());

        match top.push_back(p, u) {
            Err(e) => assert_eq!(e.node(), u),
            Ok(_) => panic!("expected NodeAlreadyActiveError, got Ok"),
        }

        let res1 = top.insert_before(p, u, None);
        assert!(res1.is_err());
        if let Err(e) = res1 {
            assert_eq!(e.node(), u);
        }

        let res2 = top.insert_after(p, u, None);
        assert!(res2.is_err());
        if let Err(e) = res2 {
            assert_eq!(e.node(), u);
        }
    }

    #[test]
    fn test_deactivate_and_remove_update_links_and_lengths() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let (a, b, c) = {
            let ns = mk_nodes(&mut top, 3);
            (ns[0], ns[1], ns[2])
        };

        assert!(top.push_back(p, a).is_ok());
        assert!(top.push_back(p, b).is_ok());
        assert!(top.push_back(p, c).is_ok());
        assert_path_eq(&top, p, &[a, b, c]);

        let r = top.deactivate(b);
        match r {
            Ok((pp, prev, next)) => {
                assert_eq!(pp, p);
                assert_eq!(prev, Some(a));
                assert_eq!(next, Some(c));
            }
            other => panic!("unexpected deactivate result: {:?}", other),
        }
        assert!(!top.is_active(b));
        assert_path_eq(&top, p, &[a, c]);

        let r2 = top.remove(c);
        let (pp2, prev2, next2) = r2.unwrap();
        assert_eq!(pp2, p);
        assert_eq!(prev2, Some(a));
        assert_eq!(next2, None);
        assert_path_eq(&top, p, &[a]);

        assert!(top.remove(a).is_ok());
        assert_path_eq(&top, p, &[]);
    }

    #[test]
    fn test_deactivate_errors_on_inactive() {
        let mut top = PathArena::new(0, 1);
        let u = top.alloc_node();
        match top.deactivate(u) {
            Err(e) => assert_eq!(e.node(), u),
            Ok(_) => panic!("expected NodeInactiveError, got Ok"),
        }
        match top.remove(u) {
            Err(e) => assert_eq!(e.node(), u),
            Ok(_) => panic!("expected NodeInactiveError, got Ok"),
        }
    }

    #[test]
    fn test_move_range_after_noop_cases() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b, c, d, e] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert_path_eq(&top, p, &[a, b, c, d, e]);

        assert!(top.move_range_after(b, d, Some(a)).is_ok());
        assert_path_eq(&top, p, &[a, b, c, d, e]);

        assert!(top.move_range_after(a, c, None).is_ok());
        assert_path_eq(&top, p, &[a, b, c, d, e]);
    }

    #[test]
    fn test_move_range_after_regular_cases() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 6);
        let (a, b, c, d, e, f) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]);

        for u in [a, b, c, d, e, f] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert_path_eq(&top, p, &[a, b, c, d, e, f]);

        assert!(top.move_range_after(b, d, Some(e)).is_ok());
        assert_path_eq(&top, p, &[a, e, b, c, d, f]);

        assert!(top.move_range_after(b, d, None).is_ok());
        assert_path_eq(&top, p, &[b, c, d, a, e, f]);

        assert!(top.move_range_after(e, e, Some(a)).is_ok());
        assert_path_eq(&top, p, &[b, c, d, a, e, f]);

        assert!(top.move_range_after(f, f, None).is_ok());
        assert_path_eq(&top, p, &[f, b, c, d, a, e]);
    }

    #[test]
    fn test_move_range_after_errors() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        let ns = mk_nodes(&mut top, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b, c] {
            assert!(top.push_back(p0, u).is_ok());
        }
        for u in [d, e] {
            assert!(top.push_back(p1, u).is_ok());
        }

        let err1 = top.move_range_after(a, d, None).unwrap_err();
        assert!(matches!(err1, SpliceRangeError::DifferentPaths(_)));

        let err2 = top.move_range_after(a, b, Some(d)).unwrap_err();
        assert!(matches!(err2, SpliceRangeError::DifferentPaths(_)));

        let err3 = top.move_range_after(a, c, Some(b)).unwrap_err();
        assert!(matches!(err3, SpliceRangeError::DestInsideRange(_)));
    }

    #[test]
    fn test_splice_range_after_alias_works() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);
        for u in [a, b, c, d] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert!(top.splice_range_after(b, c, Some(a)).is_ok());
        assert_path_eq(&top, p, &[a, b, c, d]);

        assert!(top.splice_range_after(b, c, None).is_ok());
        assert_path_eq(&top, p, &[b, c, a, d]);
    }

    #[test]
    fn test_insert_before_variants() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        assert!(top.insert_before(p, a, None).is_ok());
        assert_path_eq(&top, p, &[a]);

        assert!(top.insert_before(p, b, Some(a)).is_ok());
        assert_path_eq(&top, p, &[b, a]);

        assert!(top.insert_before(p, c, None).is_ok());
        assert_path_eq(&top, p, &[b, a, c]);

        assert!(top.insert_before(p, d, Some(a)).is_ok());
        assert_path_eq(&top, p, &[b, d, a, c]);
    }

    #[test]
    fn test_move_node_to_path_back_and_lengths() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        let ns = mk_nodes(&mut top, 3);
        let (a, b, c) = (ns[0], ns[1], ns[2]);

        for u in [a, b, c] {
            assert!(top.push_back(p0, u).is_ok());
        }
        assert_eq!(top.path_len(p0), 3);
        assert_eq!(top.path_len(p1), 0);

        assert!(top.move_node_to_path_back(b, p1).is_ok());
        assert_path_eq(&top, p0, &[a, c]);
        assert_path_eq(&top, p1, &[b]);
        assert_eq!(top.path_of(b), Some(p1));
        assert!(top.is_active(b));
    }

    #[test]
    fn test_reverse_range_various_cases() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 6);
        let (a, b, c, d, e, f) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]);

        for u in [a, b, c, d, e, f] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert_path_eq(&top, p, &[a, b, c, d, e, f]);

        assert!(top.reverse_range(c, c).is_ok());
        assert_path_eq(&top, p, &[a, b, c, d, e, f]);

        assert!(top.reverse_range(b, c).is_ok());
        assert_path_eq(&top, p, &[a, c, b, d, e, f]);

        assert!(top.reverse_range(b, e).is_ok());
        assert_path_eq(&top, p, &[a, c, e, d, b, f]);

        assert!(top.reverse_range(a, f).is_ok());
        assert_path_eq(&top, p, &[f, b, d, e, c, a]);
    }

    #[test]
    fn test_iter_path_and_empty_cases() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        assert_eq!(to_vec(&top, p0), vec![]);
        let ns = mk_nodes(&mut top, 2);
        let (a, b) = (ns[0], ns[1]);
        assert!(top.push_back(p1, a).is_ok());
        assert!(top.push_back(p1, b).is_ok());
        let v = to_vec(&top, p1);
        assert_eq!(v, vec![a, b]);
    }

    #[test]
    fn test_path_of_and_is_active_toggling() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let u = top.alloc_node();
        assert_eq!(top.path_of(u), None);
        assert!(!top.is_active(u));
        assert!(top.push_front(p, u).is_ok());
        assert_eq!(top.path_of(u), Some(p));
        assert!(top.is_active(u));
        assert!(top.deactivate(u).is_ok());
        assert_eq!(top.path_of(u), None);
        assert!(!top.is_active(u));
    }

    #[test]
    fn test_reserve_capacity_and_growth() {
        let mut top = PathArena::with_capacities(1, 1);
        top.reserve_nodes(10);
        top.reserve_paths(5);

        let p0 = top.create_path();
        let p1 = top.create_path();
        let p2 = top.create_path();
        let ns = mk_nodes(&mut top, 8);

        for (i, u) in ns.iter().copied().enumerate() {
            let p = match i % 3 {
                0 => p0,
                1 => p1,
                _ => p2,
            };
            assert!(top.push_back(p, u).is_ok());
        }

        assert_eq!(top.path_len(p0) + top.path_len(p1) + top.path_len(p2), 8);
    }

    #[test]
    fn test_display_impls() {
        let mut top = PathArena::new(0, 1);
        let p = top.create_path();
        let u = top.alloc_node();

        let s1 = format!("{}", p);
        assert!(
            s1.starts_with("PathKey(") && s1.ends_with(")"),
            "bad PathKey fmt: {}",
            s1
        );

        let s2 = format!("{}", u);
        assert!(
            s2.starts_with("NodeKey(") && s2.ends_with(")"),
            "bad NodeKey fmt: {}",
            s2
        );
    }

    #[test]
    fn test_create_multiple_paths_and_head_tail_consistency() {
        let mut top = PathArena::new(0, 3);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        let p2 = PathKey::from_raw(2);

        let ns0 = mk_nodes(&mut top, 3);
        for u in &ns0 {
            assert!(top.push_back(p0, *u).is_ok());
        }
        assert_path_eq(&top, p0, &ns0);

        let ns1 = mk_nodes(&mut top, 2);
        for u in &ns1 {
            assert!(top.push_front(p1, *u).is_ok());
        }
        let mut expected1 = ns1.clone();
        expected1.reverse();
        assert_path_eq(&top, p1, &expected1);

        assert_path_eq(&top, p2, &[]);
    }

    #[test]
    fn test_move_and_reverse_edge_cross_checks() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 7);
        let (a, b, c, d, e, f, g) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], ns[6]);
        for u in [a, b, c, d, e, f, g] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert_path_eq(&top, p, &[a, b, c, d, e, f, g]);

        // Move [c..e] after g -> a, b, f, g, c, d, e
        assert!(top.move_range_after(c, e, Some(g)).is_ok());
        assert_path_eq(&top, p, &[a, b, f, g, c, d, e]);

        // Reverse [b..f] (which is [b, f]) -> a, f, b, g, c, d, e
        assert!(top.reverse_range(b, f).is_ok());
        assert_path_eq(&top, p, &[a, f, b, g, c, d, e]);

        // Move [b..d] (now [b, g, c, d]) to front -> b, g, c, d, a, f, e
        assert!(top.move_range_after(b, d, None).is_ok());
        assert_path_eq(&top, p, &[b, g, c, d, a, f, e]);

        // Reverse [b..e] (entire list currently) -> e, f, a, d, c, g, b
        assert!(top.reverse_range(b, e).is_ok());
        assert_path_eq(&top, p, &[e, f, a, d, c, g, b]);

        // Move [e..e] after a -> f, a, e, d, c, g, b
        assert!(top.move_range_after(e, e, Some(a)).is_ok());
        assert_path_eq(&top, p, &[f, a, e, d, c, g, b]);
    }

    // New tests for recently added helpers and move variants:

    #[test]
    fn test_is_head_is_tail_and_neighbors() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]); // d stays inactive initially

        for u in [a, b, c] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert!(top.is_head(p, a));
        assert!(!top.is_head(p, b));
        assert!(!top.is_head(p, c));

        assert!(top.is_tail(p, c));
        assert!(!top.is_tail(p, b));
        assert!(!top.is_tail(p, a));

        assert_eq!(top.neighbors(a), (None, Some(b)));
        assert_eq!(top.neighbors(b), (Some(a), Some(c)));
        assert_eq!(top.neighbors(c), (Some(b), None));

        // Inactive node neighbors are None/None
        assert_eq!(top.neighbors(d), (None, None));
    }

    #[test]
    fn test_num_active_nodes_across_paths_and_after_clear() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut top, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b] {
            assert!(top.push_back(p0, u).is_ok());
        }
        for u in [c, d, e] {
            assert!(top.push_back(p1, u).is_ok());
        }
        assert_eq!(top.num_active_nodes(), 5);

        // Deactivate one on p1
        assert!(top.deactivate(d).is_ok());
        assert_eq!(top.num_active_nodes(), 4);

        // Clear p0 entirely -> only c and e remain active
        top.clear_path(p0);
        assert_eq!(top.num_active_nodes(), 2);
        assert!(!top.is_active(a));
        assert!(!top.is_active(b));
        assert!(top.is_active(c));
        assert!(!top.is_active(d));
        assert!(top.is_active(e));
    }

    #[test]
    fn test_move_after_and_move_before_front_back() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        for u in [a, b, c, d] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert_path_eq(&top, p, &[a, b, c, d]);

        // move_after to front (None) — moving a is a no-op (already head)
        assert!(top.move_after(a, None).is_ok());
        assert_path_eq(&top, p, &[a, b, c, d]);

        // move_after(c) after a -> a, c, b, d
        assert!(top.move_after(c, Some(a)).is_ok());
        assert_path_eq(&top, p, &[a, c, b, d]);

        // move_after(b) with None -> move b to front
        assert!(top.move_after(b, None).is_ok());
        assert_path_eq(&top, p, &[b, a, c, d]);

        // move_before(d) before a -> b, d, a, c
        assert!(top.move_before(d, Some(a)).is_ok());
        assert_path_eq(&top, p, &[b, d, a, c]);

        // move_before(b, None) -> move b to back
        assert!(top.move_before(b, None).is_ok());
        assert_path_eq(&top, p, &[d, a, c, b]);

        // move_before(b, None) when already tail -> no-op
        assert!(top.move_before(b, None).is_ok());
        assert_path_eq(&top, p, &[d, a, c, b]);
    }

    #[test]
    fn test_move_to_front_and_back_noop_cases() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 3);
        let (a, b, c) = (ns[0], ns[1], ns[2]);
        for u in [a, b, c] {
            assert!(top.push_back(p, u).is_ok());
        }
        assert_path_eq(&top, p, &[a, b, c]);

        // No-ops
        assert!(top.move_to_front(a).is_ok());
        assert!(top.move_to_back(c).is_ok());
        assert_path_eq(&top, p, &[a, b, c]);

        // Move c to front -> c, a, b
        assert!(top.move_to_front(c).is_ok());
        assert_path_eq(&top, p, &[c, a, b]);

        // Move c to back -> a, b, c
        assert!(top.move_to_back(c).is_ok());
        assert_path_eq(&top, p, &[a, b, c]);
    }

    #[test]
    fn test_reverse_path_empty_single_multi() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        // Empty path: no-op
        assert!(top.reverse_path(p0).is_ok());
        assert_path_eq(&top, p0, &[]);

        // Single node: no-op
        let a = top.alloc_node();
        assert!(top.push_back(p0, a).is_ok());
        assert!(top.reverse_path(p0).is_ok());
        assert_path_eq(&top, p0, &[a]);

        // Multi
        let ns = mk_nodes(&mut top, 3);
        let (b, c, d) = (ns[0], ns[1], ns[2]);
        for u in [b, c, d] {
            assert!(top.push_back(p1, u).is_ok());
        }
        assert!(top.reverse_path(p1).is_ok());
        assert_path_eq(&top, p1, &[d, c, b]);
    }

    #[test]
    fn test_kth_and_distance_forward() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut top, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);
        for u in [a, b, c, d, e] {
            assert!(top.push_back(p0, u).is_ok());
        }

        // kth
        assert_eq!(top.kth(p0, 0), Some(a));
        assert_eq!(top.kth(p0, 1), Some(b));
        assert_eq!(top.kth(p0, 2), Some(c));
        assert_eq!(top.kth(p0, 3), Some(d));
        assert_eq!(top.kth(p0, 4), Some(e));
        assert_eq!(top.kth(p0, 5), None);

        // distance_forward on same path
        assert_eq!(top.distance_forward(a, a), Some(0));
        assert_eq!(top.distance_forward(a, d), Some(3));
        assert_eq!(top.distance_forward(b, e), Some(3));
        // Not reachable forward (b before a)
        assert_eq!(top.distance_forward(d, a), None);

        // Cross-path distance -> None
        let x = top.alloc_node();
        assert!(top.push_back(p1, x).is_ok());
        assert_eq!(top.distance_forward(a, x), None);
    }

    #[test]
    fn test_clear_path_deactivates_all_nodes() {
        let mut top = PathArena::new(0, 2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut top, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        assert!(top.push_back(p0, a).is_ok());
        assert!(top.push_back(p0, b).is_ok());
        assert!(top.push_back(p0, c).is_ok());
        assert!(top.push_back(p1, d).is_ok());
        assert_eq!(top.num_active_nodes(), 4);

        top.clear_path(p0);

        assert_path_eq(&top, p0, &[]);
        assert!(top.head(p0).is_none());
        assert!(top.tail(p0).is_none());
        assert_eq!(top.path_len(p0), 0);

        // All nodes formerly in p0 are inactive now
        for u in [a, b, c] {
            assert!(!top.is_active(u));
            assert_eq!(top.neighbors(u), (None, None));
            assert_eq!(top.path_of(u), None);
        }

        // Nodes in other paths remain untouched
        assert_path_eq(&top, p1, &[d]);
        assert_eq!(top.num_active_nodes(), 1);
    }

    #[test]
    fn test_iter_path_is_fused() {
        let mut top = PathArena::new(0, 1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut top, 2);
        let (a, b) = (ns[0], ns[1]);
        assert!(top.push_back(p, a).is_ok());
        assert!(top.push_back(p, b).is_ok());

        let mut it = top.iter_path(p);
        assert_eq!(it.next(), Some(a));
        assert_eq!(it.next(), Some(b));
        assert_eq!(it.next(), None);
        // FusedIterator: subsequent next() calls must keep returning None
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }
}
