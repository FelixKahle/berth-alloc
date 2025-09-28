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

use crate::metaheuristic::chainox::{
    err::{
        DestInsideRangeError, DifferentPathsError, NodeAlreadyActiveError, NodeInactiveError,
        NodeNotInactiveError, SpliceRangeError,
    },
    key::{NodeKey, PathKey},
    traits::{PathAlloc, PathEdit, PathView},
};
use std::{iter::FusedIterator, num::NonZeroUsize};

#[derive(Debug, Clone)]
pub struct PathIterator<'a> {
    next_raw: &'a [usize],
    prev_raw: &'a [usize],
    front: usize, // head
    back: usize,  // tail
    done: bool,
}

impl<'a> PathIterator<'a> {
    fn new(top: &'a PathArena, p: PathKey) -> Self {
        Self {
            next_raw: &top.next_raw,
            prev_raw: &top.prev_raw,
            front: top.head_raw_of(p),
            back: top.tail_raw_of(p),
            done: top.head_raw_of(p) == 0,
        }
    }
}

impl<'a> Iterator for PathIterator<'a> {
    type Item = NodeKey;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let cur = self.front;
        if cur == 0 {
            self.done = true;
            return None;
        }
        if cur == self.back {
            self.done = true;
        }
        self.front = self.next_raw[cur];
        Some(NodeKey::from_index(cur))
    }
}

impl<'a> DoubleEndedIterator for PathIterator<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let cur = self.back;
        if cur == 0 {
            self.done = true;
            return None;
        }
        if cur == self.front {
            self.done = true;
        }
        self.back = self.prev_raw[cur];
        Some(NodeKey::from_index(cur))
    }
}

impl<'a> FusedIterator for PathIterator<'a> {}

#[derive(Debug, Clone)]
pub struct RangeIterator<'a> {
    next_raw: &'a [usize],
    prev_raw: &'a [usize],
    front: usize, // starts at first
    back: usize,  // starts at last
    done: bool,
}

impl<'a> RangeIterator<'a> {
    fn new(top: &'a PathArena, first: NodeKey, last: NodeKey) -> Self {
        Self {
            next_raw: &top.next_raw,
            prev_raw: &top.prev_raw,
            front: first.get(),
            back: last.get(),
            done: false,
        }
    }
}

impl<'a> Iterator for RangeIterator<'a> {
    type Item = NodeKey;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let cur = self.front;
        if cur == 0 {
            self.done = true;
            return None;
        }
        if cur == self.back {
            self.done = true;
        }
        self.front = self.next_raw[cur];
        Some(NodeKey::from_index(cur))
    }
}

impl<'a> DoubleEndedIterator for RangeIterator<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let cur = self.back;
        if cur == 0 {
            self.done = true;
            return None;
        }
        if cur == self.front {
            self.done = true;
        }
        self.back = self.prev_raw[cur];
        Some(NodeKey::from_index(cur))
    }
}

impl<'a> std::iter::FusedIterator for RangeIterator<'a> {}

#[derive(Debug, Clone)]
pub struct PathArena {
    /// Stores the index of the previous node for each node. `0` represents `None`.
    prev_raw: Vec<usize>,
    /// Stores the index of the next node for each node. `0` represents `None`.
    next_raw: Vec<usize>,
    /// Stores `path_index + 1` for each node. `0` indicates the node is inactive.
    path_of_raw: Vec<usize>,
    /// A stack of reusable (freed) node indices (1-based).
    free_list: Vec<usize>,
    /// Stores the index of the head node for each path. `0` means the path is empty.
    head_raw: Vec<usize>,
    /// Stores the index of the tail node for each path. `0` means the path is empty.
    tail_raw: Vec<usize>,
    /// Stores the number of nodes (length) for each path.
    len: Vec<usize>,
}

/// Encodes an `Option<NodeKey>` into a raw `usize`. `None` becomes `0`.
#[inline(always)]
fn enc(node_key_opt: Option<NodeKey>) -> usize {
    node_key_opt.map_or(0, |key| key.get())
}

/// Decodes a raw `usize` into an `Option<NodeKey>`. `0` becomes `None`.
#[inline(always)]
fn dec(raw_index: usize) -> Option<NodeKey> {
    NonZeroUsize::new(raw_index).map(|non_zero| NodeKey::from_index(non_zero.get()))
}

impl PathArena {
    /// Gets the raw `usize` index from a `NodeKey`.
    #[inline(always)]
    fn raw(&self, node_key: NodeKey) -> usize {
        node_key.get()
    }

    /// Gets the raw `usize` index from a `PathKey`.
    #[inline(always)]
    fn path_index(&self, path_key: PathKey) -> usize {
        path_key.to_raw()
    }

    /// Gets the raw head index of a path.
    #[inline(always)]
    fn head_raw_of(&self, path_key: PathKey) -> usize {
        self.head_raw[self.path_index(path_key)]
    }

    /// Gets the raw tail index of a path.
    #[inline(always)]
    fn tail_raw_of(&self, path_key: PathKey) -> usize {
        self.tail_raw[self.path_index(path_key)]
    }

    /// Gets the raw next index of a node.
    #[inline(always)]
    fn next_raw_of(&self, node_key: NodeKey) -> usize {
        self.next_raw[self.raw(node_key)]
    }

    /// Gets the raw previous index of a node.
    #[inline(always)]
    fn prev_raw_of(&self, node_key: NodeKey) -> usize {
        self.prev_raw[self.raw(node_key)]
    }

    /// Sets the raw head index for a path.
    #[inline(always)]
    fn set_head_raw(&mut self, path_key: PathKey, raw_node_index: usize) {
        let index = self.path_index(path_key);
        self.head_raw[index] = raw_node_index;
    }

    /// Sets the raw tail index for a path.
    #[inline(always)]
    fn set_tail_raw(&mut self, path_key: PathKey, raw_node_index: usize) {
        let index = self.path_index(path_key);
        self.tail_raw[index] = raw_node_index;
    }

    /// Sets the raw next index for a node.
    #[inline(always)]
    fn set_next_raw(&mut self, node_key: NodeKey, raw_next_index: usize) {
        let index = self.raw(node_key);
        self.next_raw[index] = raw_next_index;
    }

    /// Sets the raw previous index for a node.
    #[inline(always)]
    fn set_prev_raw(&mut self, node_key: NodeKey, raw_prev_index: usize) {
        let index = self.raw(node_key);
        self.prev_raw[index] = raw_prev_index;
    }

    /// Internal: link a single node between prev/next in path.
    #[inline]
    fn link_between(
        &mut self,
        path_key: PathKey,
        node_to_link: NodeKey,
        prev_node: Option<NodeKey>,
        next_node: Option<NodeKey>,
    ) {
        self.set_prev_raw(node_to_link, enc(prev_node));
        self.set_next_raw(node_to_link, enc(next_node));

        match prev_node {
            Some(prev) => self.set_next_raw(prev, self.raw(node_to_link)),
            None => self.set_head_raw(path_key, self.raw(node_to_link)),
        }
        match next_node {
            Some(next) => self.set_prev_raw(next, self.raw(node_to_link)),
            None => self.set_tail_raw(path_key, self.raw(node_to_link)),
        }
    }

    #[inline]
    fn detach_range(
        &mut self,
        path_key: PathKey,
        first: NodeKey,
        last: NodeKey,
    ) -> (Option<NodeKey>, Option<NodeKey>) {
        let before = dec(self.prev_raw_of(first));
        let after = dec(self.next_raw_of(last));

        match before {
            Some(node) => self.set_next_raw(node, enc(after)),
            None => self.set_head_raw(path_key, enc(after)),
        }
        match after {
            Some(node) => self.set_prev_raw(node, enc(before)),
            None => self.set_tail_raw(path_key, enc(before)),
        }
        (before, after)
    }

    #[inline]
    fn splice_after(
        &mut self,
        path_key: PathKey,
        first: NodeKey,
        last: NodeKey,
        dest: Option<NodeKey>,
    ) {
        let (prev, next) = match dest {
            Some(destination_node) => (
                Some(destination_node),
                dec(self.next_raw_of(destination_node)),
            ),
            None => (None, dec(self.head_raw_of(path_key))),
        };

        self.set_prev_raw(first, enc(prev));
        self.set_next_raw(last, enc(next));

        match prev {
            Some(prev_node) => self.set_next_raw(prev_node, self.raw(first)),
            None => self.set_head_raw(path_key, self.raw(first)),
        }
        match next {
            Some(next_node) => self.set_prev_raw(next_node, self.raw(last)),
            None => self.set_tail_raw(path_key, self.raw(last)),
        }
    }

    #[inline]
    fn ensure_same_path(
        &self,
        node_a: NodeKey,
        node_b: NodeKey,
    ) -> Result<PathKey, SpliceRangeError<NodeKey, PathKey>> {
        let path_a = self.ensure_active(node_a)?;
        let path_b = self.ensure_active(node_b)?;
        if path_a == path_b {
            Ok(path_a)
        } else {
            Err(DifferentPathsError::new(path_a, path_b).into())
        }
    }

    #[inline]
    fn ensure_active(&self, node_key: NodeKey) -> Result<PathKey, NodeInactiveError<NodeKey>> {
        self.path_of(node_key)
            .ok_or_else(|| NodeInactiveError::new(node_key))
    }

    #[inline]
    fn ensure_dest_on_path(
        &self,
        path_key: PathKey,
        dest_node: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        if let Some(dest) = dest_node {
            let path_dest = self.ensure_active(dest)?;
            if path_dest != path_key {
                return Err(DifferentPathsError::new(path_key, path_dest).into());
            }
        }
        Ok(())
    }

    #[inline]
    fn is_in_range(&self, path_key: PathKey, first: NodeKey, last: NodeKey, x: NodeKey) -> bool {
        debug_assert_eq!(self.path_of(first), Some(path_key));
        debug_assert_eq!(self.path_of(last), Some(path_key));
        debug_assert_eq!(self.path_of(x), Some(path_key));

        let mut current_node = first;
        loop {
            if current_node == x {
                return true;
            }
            if current_node == last {
                return false;
            }
            current_node = dec(self.next_raw_of(current_node)).expect("range must be contiguous");
        }
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
    pub fn new(initial_paths: usize) -> Self {
        let mut arena = Self::with_capacities(0, initial_paths);
        for _ in 0..initial_paths {
            arena.create_path_raw();
        }
        arena
    }

    /// Returns the total number of active nodes across all paths.
    #[inline]
    pub fn num_active_nodes(&self) -> usize {
        self.path_of_raw
            .iter()
            .skip(1)
            .filter(|&&raw| raw != 0)
            .count()
    }

    /// Forward distance from `a` to `b` following `next`. O(distance).
    #[inline]
    pub fn distance_forward(&self, a: NodeKey, b: NodeKey) -> Option<usize> {
        let path_a = self.path_of(a)?;
        if self.path_of(b)? != path_a {
            return None;
        }
        let mut current_node = a;
        let mut distance = 0usize;
        loop {
            if current_node == b {
                return Some(distance);
            }
            current_node = self.next(current_node)?;
            distance += 1;
        }
    }

    #[inline]
    fn create_path_raw(&mut self) -> PathKey {
        self.head_raw.push(0);
        self.tail_raw.push(0);
        self.len.push(0);
        PathKey::from_raw(self.head_raw.len() - 1)
    }

    #[inline]
    fn alloc_node_raw(&mut self) -> NodeKey {
        if let Some(one_based_index) = self.free_list.pop() {
            debug_assert_eq!(self.prev_raw[one_based_index], 0);
            debug_assert_eq!(self.next_raw[one_based_index], 0);
            debug_assert_eq!(self.path_of_raw[one_based_index], 0);
            return NodeKey::from_index(one_based_index);
        }
        let one_based_index = self.prev_raw.len();
        self.prev_raw.push(0);
        self.next_raw.push(0);
        self.path_of_raw.push(0);
        NodeKey::from_index(one_based_index)
    }
}

impl PathView for PathArena {
    #[inline]
    fn head(&self, path_key: PathKey) -> Option<NodeKey> {
        dec(self.head_raw_of(path_key))
    }

    #[inline]
    fn tail(&self, path_key: PathKey) -> Option<NodeKey> {
        dec(self.tail_raw_of(path_key))
    }

    #[inline]
    fn next(&self, node_key: NodeKey) -> Option<NodeKey> {
        dec(self.next_raw_of(node_key))
    }

    #[inline]
    fn prev(&self, node_key: NodeKey) -> Option<NodeKey> {
        dec(self.prev_raw_of(node_key))
    }

    #[inline]
    fn path_of(&self, node_key: NodeKey) -> Option<PathKey> {
        let raw_path_index = self.path_of_raw[node_key.get()];
        (raw_path_index != 0).then(|| PathKey::from_raw(raw_path_index - 1))
    }

    #[inline]
    fn path_len(&self, path_key: PathKey) -> usize {
        self.len[self.path_index(path_key)]
    }

    #[inline]
    fn kth(&self, path_key: PathKey, k: usize) -> Option<NodeKey> {
        let mut current_raw_index = self.head_raw_of(path_key);
        let mut index = 0usize;
        while current_raw_index != 0 && index < k {
            current_raw_index = self.next_raw[current_raw_index];
            index += 1;
        }
        dec(current_raw_index)
    }

    /// Returns a forward iterator over the nodes in a path.
    #[inline]
    fn iter_path(&self, path_key: PathKey) -> impl DoubleEndedIterator<Item = NodeKey> + '_ {
        PathIterator::new(self, path_key)
    }

    /// Returns an iterator over a contiguous [first..=last] range.
    #[inline]
    fn iter_range(
        &self,
        first: NodeKey,
        last: NodeKey,
    ) -> Result<impl DoubleEndedIterator<Item = NodeKey> + '_, SpliceRangeError<NodeKey, PathKey>>
    {
        self.ensure_same_path(first, last)?;
        Ok(RangeIterator::new(self, first, last))
    }
}

impl PathEdit for PathArena {
    #[inline]
    fn insert_after(
        &mut self,
        path_key: PathKey,
        node_to_insert: NodeKey,
        anchor: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        if self.is_active(node_to_insert) {
            return Err(NodeAlreadyActiveError::new(node_to_insert));
        }
        let next_node = anchor
            .map(|anchor_node| dec(self.next_raw_of(anchor_node)))
            .unwrap_or_else(|| dec(self.head_raw_of(path_key)));

        self.link_between(path_key, node_to_insert, anchor, next_node);

        let node_index = self.raw(node_to_insert);
        let path_index = self.path_index(path_key);
        self.path_of_raw[node_index] = path_index + 1;
        self.len[path_index] += 1;
        Ok(())
    }

    #[inline]
    fn insert_before(
        &mut self,
        path_key: PathKey,
        node_to_insert: NodeKey,
        before: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        match before {
            Some(b) => self.insert_after(path_key, node_to_insert, self.prev(b)),
            None => self.push_back(path_key, node_to_insert),
        }
    }

    #[inline]
    fn deactivate(
        &mut self,
        node_key: NodeKey,
    ) -> Result<(PathKey, Option<NodeKey>, Option<NodeKey>), NodeInactiveError<NodeKey>> {
        let path_key = self.ensure_active(node_key)?;
        let path_index = self.path_index(path_key);
        let prev_node = dec(self.prev_raw_of(node_key));
        let next_node = dec(self.next_raw_of(node_key));

        match prev_node {
            Some(prev) => self.set_next_raw(prev, enc(next_node)),
            None => self.set_head_raw(path_key, enc(next_node)),
        }
        match next_node {
            Some(next) => self.set_prev_raw(next, enc(prev_node)),
            None => self.set_tail_raw(path_key, enc(prev_node)),
        }

        let node_index = self.raw(node_key);
        self.set_prev_raw(node_key, 0);
        self.set_next_raw(node_key, 0);
        self.path_of_raw[node_index] = 0;
        self.len[path_index] -= 1;
        Ok((path_key, prev_node, next_node))
    }

    fn move_range_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        destination_node: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let path_key = self.ensure_same_path(first, last)?;
        self.ensure_dest_on_path(path_key, destination_node)?;

        // No-ops
        if let Some(dest) = destination_node {
            if dest == last || self.prev(first) == Some(dest) {
                return Ok(());
            }
            if self.is_in_range(path_key, first, last, dest) {
                return Err(DestInsideRangeError::new(first, last, dest).into());
            }
        } else if self.head(path_key) == Some(first) {
            return Ok(());
        }

        let _ = self.detach_range(path_key, first, last);
        self.splice_after(path_key, first, last, destination_node);
        Ok(())
    }

    fn reverse_range(
        &mut self,
        first: NodeKey,
        last: NodeKey,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let path_key = self.ensure_same_path(first, last)?;
        if first == last {
            return Ok(());
        }

        let before_range = self.prev(first);
        let after_range = self.next(last);

        // Swap prev/next pointers for all nodes in the range.
        let mut current_node_key = first;
        loop {
            let node_index = self.raw(current_node_key);
            let next_raw_index = self.next_raw[node_index];
            std::mem::swap(
                &mut self.prev_raw[node_index],
                &mut self.next_raw[node_index],
            );
            if current_node_key == last {
                break;
            }
            current_node_key = NodeKey::from_index(next_raw_index);
        }

        // Re-link the reversed range with the rest of the path.
        self.set_prev_raw(last, enc(before_range));
        self.set_next_raw(first, enc(after_range));

        match before_range {
            Some(before) => self.set_next_raw(before, self.raw(last)),
            None => self.set_head_raw(path_key, self.raw(last)),
        }
        match after_range {
            Some(after) => self.set_prev_raw(after, self.raw(first)),
            None => self.set_tail_raw(path_key, self.raw(first)),
        }

        Ok(())
    }

    fn move_range_to_path_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest_path: PathKey,
        dest_after: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let src_path = self.ensure_same_path(first, last)?;

        if src_path == dest_path {
            return self.move_range_after(first, last, dest_after);
        }

        if let Some(anchor_node) = dest_after {
            let anchor_path = self.ensure_active(anchor_node)?;
            if anchor_path != dest_path {
                return Err(DifferentPathsError::new(dest_path, anchor_path).into());
            }
        }

        let _ = self.detach_range(src_path, first, last);

        let mut range_len = 0usize;
        let dest_path_raw_index = self.path_index(dest_path) + 1;

        let mut current_node_index = first.get();
        let last_node_index = last.get();
        loop {
            let next_node_index = self.next_raw[current_node_index];
            self.path_of_raw[current_node_index] = dest_path_raw_index;
            range_len += 1;
            if current_node_index == last_node_index {
                break;
            }
            current_node_index = next_node_index;
        }

        let src_path_index = self.path_index(src_path);
        let dest_path_index = self.path_index(dest_path);
        self.len[src_path_index] -= range_len;
        self.len[dest_path_index] += range_len;

        self.splice_after(dest_path, first, last, dest_after);

        Ok(())
    }

    #[inline]
    fn clear_path(&mut self, p: PathKey) {
        let path_index = self.path_index(p);
        let mut current_node_index = self.head_raw[path_index];

        while current_node_index != 0 {
            let next_node_index = self.next_raw[current_node_index];
            // Deactivate node
            self.prev_raw[current_node_index] = 0;
            self.next_raw[current_node_index] = 0;
            self.path_of_raw[current_node_index] = 0;
            current_node_index = next_node_index;
        }

        self.head_raw[path_index] = 0;
        self.tail_raw[path_index] = 0;
        self.len[path_index] = 0;
    }
}

impl PathAlloc for PathArena {
    #[inline]
    fn create_path(&mut self) -> PathKey {
        self.create_path_raw()
    }

    #[inline]
    fn alloc_node(&mut self) -> NodeKey {
        self.alloc_node_raw()
    }

    #[inline]
    fn free_node(&mut self, node_key: NodeKey) -> Result<(), NodeNotInactiveError<NodeKey>> {
        let node_index = node_key.get();
        if self.path_of_raw[node_index] != 0 {
            return Err(NodeNotInactiveError::new(node_key));
        }
        debug_assert_eq!(self.prev_raw[node_index], 0);
        debug_assert_eq!(self.next_raw[node_index], 0);
        self.free_list.push(node_index);
        Ok(())
    }

    #[inline]
    fn reserve_nodes(&mut self, additional: usize) {
        self.prev_raw.reserve(additional);
        self.next_raw.reserve(additional);
        self.path_of_raw.reserve(additional);
    }

    #[inline]
    fn reserve_paths(&mut self, additional: usize) {
        self.head_raw.reserve(additional);
        self.tail_raw.reserve(additional);
        self.len.reserve(additional);
    }

    #[inline]
    fn num_paths(&self) -> usize {
        self.head_raw.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_nodes(arena: &mut PathArena, n: usize) -> Vec<NodeKey> {
        (0..n).map(|_| arena.alloc_node()).collect()
    }

    fn to_vec(arena: &PathArena, p: PathKey) -> Vec<NodeKey> {
        arena.iter_path(p).collect()
    }

    fn back_to_front(arena: &PathArena, p: PathKey) -> Vec<NodeKey> {
        let mut v = Vec::new();
        let mut cur = arena.tail(p);
        while let Some(u) = cur {
            v.push(u);
            cur = arena.prev(u);
        }
        v
    }

    fn assert_path_eq(arena: &PathArena, p: PathKey, expected: &[NodeKey]) {
        assert_eq!(
            arena.path_len(p),
            expected.len(),
            "path_len mismatch for p={}",
            p
        );
        let head = arena.head(p);
        let tail = arena.tail(p);
        match expected.first() {
            Some(&e) => assert_eq!(head, Some(e), "head mismatch for p={}", p),
            None => assert_eq!(head, None, "head should be None on empty path for p={}", p),
        }
        match expected.last() {
            Some(&e) => assert_eq!(tail, Some(e), "tail mismatch for p={}", p),
            None => assert_eq!(tail, None, "tail should be None on empty path for p={}", p),
        }
        let actual = to_vec(arena, p);
        assert_eq!(actual, expected, "forward order mismatch for p={}", p);
        let mut expected_rev = expected.to_vec();
        expected_rev.reverse();
        let actual_rev = back_to_front(arena, p);
        assert_eq!(
            actual_rev, expected_rev,
            "backward order mismatch for p={}",
            p
        );
    }

    #[test]
    fn test_basic_path_creation_and_lengths() {
        let mut arena = PathArena::with_capacities(0, 0);
        assert_eq!(arena.num_paths(), 0);
        let p0 = arena.create_path();
        let p1 = arena.create_path();
        let p2 = arena.create_path();
        assert_eq!(arena.num_paths(), 3);
        assert_eq!(arena.path_len(p0), 0);
        assert_eq!(arena.path_len(p1), 0);
        assert_eq!(arena.path_len(p2), 0);
        assert_eq!(arena.head(p0), None);
        assert_eq!(arena.tail(p0), None);
        assert_eq!(to_vec(&arena, p0), vec![]);
    }

    #[test]
    fn test_alloc_and_free_nodes_reuse_stack_lifo() {
        let mut arena = PathArena::with_capacities(0, 1);
        let p = arena.create_path();

        let a = arena.alloc_node();
        let b = arena.alloc_node();
        let c = arena.alloc_node();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert!(!arena.is_active(a));
        assert!(!arena.is_active(b));

        assert!(arena.free_node(b).is_ok());
        assert!(arena.free_node(a).is_ok());

        let a2 = arena.alloc_node();
        assert_eq!(a2, a);
        let b2 = arena.alloc_node();
        assert_eq!(b2, b);

        assert!(arena.push_back(p, c).is_ok());
        match arena.free_node(c) {
            Err(e) => assert_eq!(e.node(), c),
            Ok(_) => panic!("expected NodeNotInactiveError, got Ok"),
        }

        assert!(arena.deactivate(c).is_ok());
        assert!(arena.free_node(c).is_ok());
    }

    #[test]
    fn test_push_front_back_and_links_are_correct() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);

        let nodes = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (nodes[0], nodes[1], nodes[2], nodes[3]);

        assert!(arena.push_back(p, a).is_ok());
        assert!(arena.push_back(p, b).is_ok());
        assert_path_eq(&arena, p, &[a, b]);

        assert!(arena.push_front(p, c).is_ok());
        assert_path_eq(&arena, p, &[c, a, b]);

        assert!(arena.insert_after(p, d, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[c, a, d, b]);

        assert_eq!(arena.next(c), Some(a));
        assert_eq!(arena.prev(c), None);
        assert_eq!(arena.next(a), Some(d));
        assert_eq!(arena.prev(a), Some(c));
        assert_eq!(arena.next(d), Some(b));
        assert_eq!(arena.prev(d), Some(a));
        assert_eq!(arena.next(b), None);
        assert_eq!(arena.prev(b), Some(d));
    }

    #[test]
    fn test_activation_errors_when_already_active() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let u = arena.alloc_node();
        assert!(arena.push_back(p, u).is_ok());

        match arena.push_back(p, u) {
            Err(e) => assert_eq!(e.node(), u),
            Ok(_) => panic!("expected NodeAlreadyActiveError, got Ok"),
        }

        let res1 = arena.insert_before(p, u, None);
        assert!(res1.is_err());
        if let Err(e) = res1 {
            assert_eq!(e.node(), u);
        }

        let res2 = arena.insert_after(p, u, None);
        assert!(res2.is_err());
        if let Err(e) = res2 {
            assert_eq!(e.node(), u);
        }
    }

    #[test]
    fn test_deactivate_and_remove_update_links_and_lengths() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let (a, b, c) = {
            let ns = mk_nodes(&mut arena, 3);
            (ns[0], ns[1], ns[2])
        };

        assert!(arena.push_back(p, a).is_ok());
        assert!(arena.push_back(p, b).is_ok());
        assert!(arena.push_back(p, c).is_ok());
        assert_path_eq(&arena, p, &[a, b, c]);

        let r = arena.deactivate(b);
        match r {
            Ok((pp, prev, next)) => {
                assert_eq!(pp, p);
                assert_eq!(prev, Some(a));
                assert_eq!(next, Some(c));
            }
            other => panic!("unexpected deactivate result: {:?}", other),
        }
        assert!(!arena.is_active(b));
        assert_path_eq(&arena, p, &[a, c]);

        let r2 = arena.remove(c);
        let (pp2, prev2, next2) = r2.unwrap();
        assert_eq!(pp2, p);
        assert_eq!(prev2, Some(a));
        assert_eq!(next2, None);
        assert_path_eq(&arena, p, &[a]);

        assert!(arena.remove(a).is_ok());
        assert_path_eq(&arena, p, &[]);
    }

    #[test]
    fn test_deactivate_errors_on_inactive() {
        let mut arena = PathArena::new(1);
        let u = arena.alloc_node();
        match arena.deactivate(u) {
            Err(e) => assert_eq!(e.node(), u),
            Ok(_) => panic!("expected NodeInactiveError, got Ok"),
        }
        match arena.remove(u) {
            Err(e) => assert_eq!(e.node(), u),
            Ok(_) => panic!("expected NodeInactiveError, got Ok"),
        }
    }

    #[test]
    fn test_move_range_after_noop_cases() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b, c, d, e] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c, d, e]);

        assert!(arena.move_range_after(b, d, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[a, b, c, d, e]);

        assert!(arena.move_range_after(a, c, None).is_ok());
        assert_path_eq(&arena, p, &[a, b, c, d, e]);
    }

    #[test]
    fn test_move_range_after_regular_cases() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 6);
        let (a, b, c, d, e, f) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]);

        for u in [a, b, c, d, e, f] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c, d, e, f]);

        assert!(arena.move_range_after(b, d, Some(e)).is_ok());
        assert_path_eq(&arena, p, &[a, e, b, c, d, f]);

        assert!(arena.move_range_after(b, d, None).is_ok());
        assert_path_eq(&arena, p, &[b, c, d, a, e, f]);

        assert!(arena.move_range_after(e, e, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[b, c, d, a, e, f]);

        assert!(arena.move_range_after(f, f, None).is_ok());
        assert_path_eq(&arena, p, &[f, b, c, d, a, e]);
    }

    #[test]
    fn test_move_range_after_errors() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        let ns = mk_nodes(&mut arena, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b, c] {
            assert!(arena.push_back(p0, u).is_ok());
        }
        for u in [d, e] {
            assert!(arena.push_back(p1, u).is_ok());
        }

        let err1 = arena.move_range_after(a, d, None).unwrap_err();
        assert!(matches!(err1, SpliceRangeError::DifferentPaths(_)));

        let err2 = arena.move_range_after(a, b, Some(d)).unwrap_err();
        assert!(matches!(err2, SpliceRangeError::DifferentPaths(_)));

        let err3 = arena.move_range_after(a, c, Some(b)).unwrap_err();
        assert!(matches!(err3, SpliceRangeError::DestInsideRange(_)));
    }

    #[test]
    fn test_splice_range_after_alias_works() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);
        for u in [a, b, c, d] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert!(arena.splice_range_after(b, c, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[a, b, c, d]);

        assert!(arena.splice_range_after(b, c, None).is_ok());
        assert_path_eq(&arena, p, &[b, c, a, d]);
    }

    #[test]
    fn test_insert_before_variants() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        assert!(arena.insert_before(p, a, None).is_ok());
        assert_path_eq(&arena, p, &[a]);

        assert!(arena.insert_before(p, b, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[b, a]);

        assert!(arena.insert_before(p, c, None).is_ok());
        assert_path_eq(&arena, p, &[b, a, c]);

        assert!(arena.insert_before(p, d, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[b, d, a, c]);
    }

    #[test]
    fn test_move_node_to_path_back_and_lengths() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        let ns = mk_nodes(&mut arena, 3);
        let (a, b, c) = (ns[0], ns[1], ns[2]);

        for u in [a, b, c] {
            assert!(arena.push_back(p0, u).is_ok());
        }
        assert_eq!(arena.path_len(p0), 3);
        assert_eq!(arena.path_len(p1), 0);

        assert!(arena.move_node_to_path_back(b, p1).is_ok());
        assert_path_eq(&arena, p0, &[a, c]);
        assert_path_eq(&arena, p1, &[b]);
        assert_eq!(arena.path_of(b), Some(p1));
        assert!(arena.is_active(b));
    }

    #[test]
    fn test_reverse_range_various_cases() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 6);
        let (a, b, c, d, e, f) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]);

        for u in [a, b, c, d, e, f] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c, d, e, f]);

        assert!(arena.reverse_range(c, c).is_ok());
        assert_path_eq(&arena, p, &[a, b, c, d, e, f]);

        assert!(arena.reverse_range(b, c).is_ok());
        assert_path_eq(&arena, p, &[a, c, b, d, e, f]);

        assert!(arena.reverse_range(b, e).is_ok());
        assert_path_eq(&arena, p, &[a, c, e, d, b, f]);

        assert!(arena.reverse_range(a, f).is_ok());
        assert_path_eq(&arena, p, &[f, b, d, e, c, a]);
    }

    #[test]
    fn test_iter_path_and_empty_cases() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        assert_eq!(to_vec(&arena, p0), vec![]);
        let ns = mk_nodes(&mut arena, 2);
        let (a, b) = (ns[0], ns[1]);
        assert!(arena.push_back(p1, a).is_ok());
        assert!(arena.push_back(p1, b).is_ok());
        let v = to_vec(&arena, p1);
        assert_eq!(v, vec![a, b]);
    }

    #[test]
    fn test_path_of_and_is_active_toggling() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let u = arena.alloc_node();
        assert_eq!(arena.path_of(u), None);
        assert!(!arena.is_active(u));
        assert!(arena.push_front(p, u).is_ok());
        assert_eq!(arena.path_of(u), Some(p));
        assert!(arena.is_active(u));
        assert!(arena.deactivate(u).is_ok());
        assert_eq!(arena.path_of(u), None);
        assert!(!arena.is_active(u));
    }

    #[test]
    fn test_reserve_capacity_and_growth() {
        let mut arena = PathArena::with_capacities(1, 1);
        arena.reserve_nodes(10);
        arena.reserve_paths(5);

        let p0 = arena.create_path();
        let p1 = arena.create_path();
        let p2 = arena.create_path();
        let ns = mk_nodes(&mut arena, 8);

        for (i, u) in ns.iter().copied().enumerate() {
            let p = match i % 3 {
                0 => p0,
                1 => p1,
                _ => p2,
            };
            assert!(arena.push_back(p, u).is_ok());
        }

        assert_eq!(
            arena.path_len(p0) + arena.path_len(p1) + arena.path_len(p2),
            8
        );
    }

    #[test]
    fn test_display_impls() {
        let mut arena = PathArena::new(1);
        let p = arena.create_path();
        let u = arena.alloc_node();

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
        let mut arena = PathArena::new(3);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);
        let p2 = PathKey::from_raw(2);

        let ns0 = mk_nodes(&mut arena, 3);
        for u in &ns0 {
            assert!(arena.push_back(p0, *u).is_ok());
        }
        assert_path_eq(&arena, p0, &ns0);

        let ns1 = mk_nodes(&mut arena, 2);
        for u in &ns1 {
            assert!(arena.push_front(p1, *u).is_ok());
        }
        let mut expected1 = ns1.clone();
        expected1.reverse();
        assert_path_eq(&arena, p1, &expected1);

        assert_path_eq(&arena, p2, &[]);
    }

    #[test]
    fn test_move_and_reverse_edge_cross_checks() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 7);
        let (a, b, c, d, e, f, g) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], ns[6]);
        for u in [a, b, c, d, e, f, g] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c, d, e, f, g]);

        // Move [c..e] after g -> a, b, f, g, c, d, e
        assert!(arena.move_range_after(c, e, Some(g)).is_ok());
        assert_path_eq(&arena, p, &[a, b, f, g, c, d, e]);

        // Reverse [b..f] (which is [b, f]) -> a, f, b, g, c, d, e
        assert!(arena.reverse_range(b, f).is_ok());
        assert_path_eq(&arena, p, &[a, f, b, g, c, d, e]);

        // Move [b..d] (now [b, g, c, d]) to front -> b, g, c, d, a, f, e
        assert!(arena.move_range_after(b, d, None).is_ok());
        assert_path_eq(&arena, p, &[b, g, c, d, a, f, e]);

        // Reverse [b..e] (entire list currently) -> e, f, a, d, c, g, b
        assert!(arena.reverse_range(b, e).is_ok());
        assert_path_eq(&arena, p, &[e, f, a, d, c, g, b]);

        // Move [e..e] after a -> f, a, e, d, c, g, b
        assert!(arena.move_range_after(e, e, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[f, a, e, d, c, g, b]);
    }

    #[test]
    fn test_is_head_is_tail_and_neighbors() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]); // d stays inactive initially

        for u in [a, b, c] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert!(arena.is_head(p, a));
        assert!(!arena.is_head(p, b));
        assert!(!arena.is_head(p, c));

        assert!(arena.is_tail(p, c));
        assert!(!arena.is_tail(p, b));
        assert!(!arena.is_tail(p, a));

        assert_eq!(arena.neighbors(a), (None, Some(b)));
        assert_eq!(arena.neighbors(b), (Some(a), Some(c)));
        assert_eq!(arena.neighbors(c), (Some(b), None));

        // Inactive node neighbors are None/None
        assert_eq!(arena.neighbors(d), (None, None));
    }

    #[test]
    fn test_num_active_nodes_across_paths_and_after_clear() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b] {
            assert!(arena.push_back(p0, u).is_ok());
        }
        for u in [c, d, e] {
            assert!(arena.push_back(p1, u).is_ok());
        }
        assert_eq!(arena.num_active_nodes(), 5);

        // Deactivate one on p1
        assert!(arena.deactivate(d).is_ok());
        assert_eq!(arena.num_active_nodes(), 4);

        // Clear p0 entirely -> only c and e remain active
        arena.clear_path(p0);
        assert_eq!(arena.num_active_nodes(), 2);
        assert!(!arena.is_active(a));
        assert!(!arena.is_active(b));
        assert!(arena.is_active(c));
        assert!(!arena.is_active(d));
        assert!(arena.is_active(e));
    }

    #[test]
    fn test_move_after_and_move_before_front_back() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        for u in [a, b, c, d] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c, d]);

        // move_after to front (None) — moving a is a no-op (already head)
        assert!(arena.move_after(a, None).is_ok());
        assert_path_eq(&arena, p, &[a, b, c, d]);

        // move_after(c) after a -> a, c, b, d
        assert!(arena.move_after(c, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[a, c, b, d]);

        // move_after(b) with None -> move b to front
        assert!(arena.move_after(b, None).is_ok());
        assert_path_eq(&arena, p, &[b, a, c, d]);

        // move_before(d) before a -> b, d, a, c
        assert!(arena.move_before(d, Some(a)).is_ok());
        assert_path_eq(&arena, p, &[b, d, a, c]);

        // move_before(b, None) -> move b to back
        assert!(arena.move_before(b, None).is_ok());
        assert_path_eq(&arena, p, &[d, a, c, b]);

        // move_before(b, None) when already tail -> no-op
        assert!(arena.move_before(b, None).is_ok());
        assert_path_eq(&arena, p, &[d, a, c, b]);
    }

    #[test]
    fn test_move_to_front_and_back_noop_cases() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 3);
        let (a, b, c) = (ns[0], ns[1], ns[2]);
        for u in [a, b, c] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c]);

        // No-ops
        assert!(arena.move_to_front(a).is_ok());
        assert!(arena.move_to_back(c).is_ok());
        assert_path_eq(&arena, p, &[a, b, c]);

        // Move c to front -> c, a, b
        assert!(arena.move_to_front(c).is_ok());
        assert_path_eq(&arena, p, &[c, a, b]);

        // Move c to back -> a, b, c
        assert!(arena.move_to_back(c).is_ok());
        assert_path_eq(&arena, p, &[a, b, c]);
    }

    #[test]
    fn test_reverse_path_empty_single_multi() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        // Empty path: no-op
        assert!(arena.reverse_path(p0).is_ok());
        assert_path_eq(&arena, p0, &[]);

        // Single node: no-op
        let a = arena.alloc_node();
        assert!(arena.push_back(p0, a).is_ok());
        assert!(arena.reverse_path(p0).is_ok());
        assert_path_eq(&arena, p0, &[a]);

        // Multi
        let ns = mk_nodes(&mut arena, 3);
        let (b, c, d) = (ns[0], ns[1], ns[2]);
        for u in [b, c, d] {
            assert!(arena.push_back(p1, u).is_ok());
        }
        assert!(arena.reverse_path(p1).is_ok());
        assert_path_eq(&arena, p1, &[d, c, b]);
    }

    #[test]
    fn test_kth_and_distance_forward() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);
        for u in [a, b, c, d, e] {
            assert!(arena.push_back(p0, u).is_ok());
        }

        // kth
        assert_eq!(arena.kth(p0, 0), Some(a));
        assert_eq!(arena.kth(p0, 1), Some(b));
        assert_eq!(arena.kth(p0, 2), Some(c));
        assert_eq!(arena.kth(p0, 3), Some(d));
        assert_eq!(arena.kth(p0, 4), Some(e));
        assert_eq!(arena.kth(p0, 5), None);

        // distance_forward on same path
        assert_eq!(arena.distance_forward(a, a), Some(0));
        assert_eq!(arena.distance_forward(a, d), Some(3));
        assert_eq!(arena.distance_forward(b, e), Some(3));
        // Not reachable forward (b before a)
        assert_eq!(arena.distance_forward(d, a), None);

        // Cross-path distance -> None
        let x = arena.alloc_node();
        assert!(arena.push_back(p1, x).is_ok());
        assert_eq!(arena.distance_forward(a, x), None);
    }

    #[test]
    fn test_clear_path_deactivates_all_nodes() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        assert!(arena.push_back(p0, a).is_ok());
        assert!(arena.push_back(p0, b).is_ok());
        assert!(arena.push_back(p0, c).is_ok());
        assert!(arena.push_back(p1, d).is_ok());
        assert_eq!(arena.num_active_nodes(), 4);

        arena.clear_path(p0);

        assert_path_eq(&arena, p0, &[]);
        assert!(arena.head(p0).is_none());
        assert!(arena.tail(p0).is_none());
        assert_eq!(arena.path_len(p0), 0);

        // All nodes formerly in p0 are inactive now
        for u in [a, b, c] {
            assert!(!arena.is_active(u));
            assert_eq!(arena.neighbors(u), (None, None));
            assert_eq!(arena.path_of(u), None);
        }

        // Nodes in other paths remain untouched
        assert_path_eq(&arena, p1, &[d]);
        assert_eq!(arena.num_active_nodes(), 1);
    }

    #[test]
    fn test_iter_path_is_fused() {
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 2);
        let (a, b) = (ns[0], ns[1]);
        assert!(arena.push_back(p, a).is_ok());
        assert!(arena.push_back(p, b).is_ok());

        let mut it = arena.iter_path(p);
        assert_eq!(it.next(), Some(a));
        assert_eq!(it.next(), Some(b));
        assert_eq!(it.next(), None);
        // FusedIterator: subsequent next() calls must keep returning None
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    // Additional tests for newly added/untested functions in PathArena.

    #[test]
    fn test_is_path_start_end_and_is_on_path() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 4);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);

        // Build p0 = [a, b, c]
        for u in [a, b, c] {
            assert!(arena.push_back(p0, u).is_ok());
        }
        assert!(arena.is_path_start_node(a));
        assert!(!arena.is_path_start_node(b));
        assert!(!arena.is_path_start_node(c));

        assert!(arena.is_path_end_node(c));
        assert!(!arena.is_path_end_node(b));
        assert!(!arena.is_path_end_node(a));

        assert!(arena.is_on_path(a, p0));
        assert!(arena.is_on_path(b, p0));
        assert!(arena.is_on_path(c, p0));
        assert!(!arena.is_on_path(a, p1));

        // Single-node path: node is both start and end
        assert!(arena.push_back(p1, d).is_ok());
        assert!(arena.is_path_start_node(d));
        assert!(arena.is_path_end_node(d));

        // Inactive node -> neither start nor end
        assert!(arena.deactivate(d).is_ok());
        assert!(!arena.is_path_start_node(d));
        assert!(!arena.is_path_end_node(d));
        assert!(!arena.is_on_path(d, p1));
    }

    #[test]
    fn test_iter_range_basics_and_fused_and_error() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);
        for u in [a, b, c, d, e] {
            assert!(arena.push_back(p0, u).is_ok());
        }

        // Limit the iterator’s lifetime to this block so it drops before we mutate `arena`.
        {
            let mut it = arena.iter_range(b, d).expect("same path");
            assert_eq!(it.next(), Some(b));
            assert_eq!(it.next(), Some(c));
            assert_eq!(it.next(), Some(d));
            assert_eq!(it.next(), None);
            // FusedIterator: keep returning None
            assert_eq!(it.next(), None);
        } // `it` dropped here

        // Different paths -> error
        let x = arena.alloc_node();
        assert!(arena.push_back(p1, x).is_ok());
        let err = arena.iter_range(a, x);
        assert!(err.is_err());
    }

    #[test]
    fn test_move_range_to_path_after_basic_and_front() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 6);
        let (a, b, c, d, x, y) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]);

        // p0: a, b, c, d
        for u in [a, b, c, d] {
            assert!(arena.push_back(p0, u).is_ok());
        }
        // p1: x, y
        for u in [x, y] {
            assert!(arena.push_back(p1, u).is_ok());
        }
        assert_path_eq(&arena, p0, &[a, b, c, d]);
        assert_path_eq(&arena, p1, &[x, y]);

        // Move [b..c] after x -> p0: [a, d], p1: [x, b, c, y]
        assert!(arena.move_range_to_path_after(b, c, p1, Some(x)).is_ok());
        assert_path_eq(&arena, p0, &[a, d]);
        assert_path_eq(&arena, p1, &[x, b, c, y]);
        assert!(arena.is_on_path(b, p1) && arena.is_on_path(c, p1));
        assert_eq!(arena.path_len(p0), 2);
        assert_eq!(arena.path_len(p1), 4);

        // Move [a..a] to front of p1 (dest None) -> p1: [a, x, b, c, y], p0: [d]
        assert!(arena.move_range_to_path_after(a, a, p1, None).is_ok());
        assert_path_eq(&arena, p0, &[d]);
        assert_path_eq(&arena, p1, &[a, x, b, c, y]);

        // If dest is on wrong path -> error
        let z = arena.alloc_node();
        assert!(arena.push_back(p0, z).is_ok());
        let wrong = arena.move_range_to_path_after(z, z, p1, Some(d)); // d is on p0
        assert!(wrong.is_err());
    }

    #[test]
    fn test_move_range_before_to_path_variants() {
        let mut arena = PathArena::new(2);
        let p0 = PathKey::from_raw(0);
        let p1 = PathKey::from_raw(1);

        let ns = mk_nodes(&mut arena, 6);
        let (a, b, c, x, y, z) = (ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]);

        // p0: a, b, c
        for u in [a, b, c] {
            assert!(arena.push_back(p0, u).is_ok());
        }
        // p1: x, y, z
        for u in [x, y, z] {
            assert!(arena.push_back(p1, u).is_ok());
        }
        assert_path_eq(&arena, p0, &[a, b, c]);
        assert_path_eq(&arena, p1, &[x, y, z]);

        // Move [b..b] before y -> p0: [a, c], p1: [x, b, y, z]
        assert!(arena.move_range_to_path_before(b, b, p1, Some(y)).is_ok());
        assert_path_eq(&arena, p0, &[a, c]);
        assert_path_eq(&arena, p1, &[x, b, y, z]);

        // Move [a..c] (now [a,c] remaining) before None => to back of p1
        assert!(arena.move_range_to_path_before(a, c, p1, None).is_ok());
        assert_path_eq(&arena, p0, &[]);
        assert_path_eq(&arena, p1, &[x, b, y, z, a, c]);
    }

    #[test]
    fn test_move_range_to_path_intra_path_delegation() {
        // Same-path operations should delegate to O(1) intra-path move.
        let mut arena = PathArena::new(1);
        let p = PathKey::from_raw(0);
        let ns = mk_nodes(&mut arena, 5);
        let (a, b, c, d, e) = (ns[0], ns[1], ns[2], ns[3], ns[4]);

        for u in [a, b, c, d, e] {
            assert!(arena.push_back(p, u).is_ok());
        }
        assert_path_eq(&arena, p, &[a, b, c, d, e]);

        // move_range_to_path_after with same dest_path should behave like move_range_after
        assert!(arena.move_range_to_path_after(b, d, p, Some(a)).is_ok()); // move [b..d] after a -> no-op (already after a's next?), but legal
        // Force a visible change:
        assert!(arena.move_range_to_path_after(d, e, p, Some(a)).is_ok()); // move [d..e] after a => a, d, e, b, c
        assert_path_eq(&arena, p, &[a, d, e, b, c]);

        // move_range_before_to_path with same path: before b -> a, d, e, b, c (already correct)
        assert!(arena.move_range_to_path_before(c, c, p, Some(b)).is_ok()); // move c before b -> a, d, e, c, b
        assert_path_eq(&arena, p, &[a, d, e, c, b]);
    }
}
