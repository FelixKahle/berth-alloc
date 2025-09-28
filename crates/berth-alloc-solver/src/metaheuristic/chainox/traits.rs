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
    err::{NodeAlreadyActiveError, NodeInactiveError, NodeNotInactiveError, SpliceRangeError},
    key::{NodeKey, PathKey},
};

pub trait PathView {
    fn head(&self, path_key: PathKey) -> Option<NodeKey>;
    fn tail(&self, path_key: PathKey) -> Option<NodeKey>;
    fn next(&self, node_key: NodeKey) -> Option<NodeKey>;
    fn prev(&self, node_key: NodeKey) -> Option<NodeKey>;
    fn path_of(&self, node_key: NodeKey) -> Option<PathKey>;
    fn path_len(&self, path_key: PathKey) -> usize;

    #[inline]
    fn is_active(&self, u: NodeKey) -> bool {
        self.path_of(u).is_some()
    }

    #[inline]
    fn is_head(&self, p: PathKey, u: NodeKey) -> bool {
        self.head(p) == Some(u)
    }

    #[inline]
    fn is_tail(&self, p: PathKey, u: NodeKey) -> bool {
        self.tail(p) == Some(u)
    }

    #[inline]
    fn neighbors(&self, u: NodeKey) -> (Option<NodeKey>, Option<NodeKey>) {
        (self.prev(u), self.next(u))
    }

    #[inline]
    fn is_path_start_node(&self, u: NodeKey) -> bool {
        self.is_active(u) && self.prev(u).is_none()
    }

    #[inline]
    fn is_path_end_node(&self, u: NodeKey) -> bool {
        self.is_active(u) && self.next(u).is_none()
    }

    #[inline]
    fn is_on_path(&self, u: NodeKey, p: PathKey) -> bool {
        self.path_of(u) == Some(p)
    }

    #[inline]
    fn distance_forward(&self, a: NodeKey, b: NodeKey) -> Option<usize> {
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

    fn kth(&self, path_key: PathKey, k: usize) -> Option<NodeKey>;

    fn iter_path(&self, path_key: PathKey) -> impl DoubleEndedIterator<Item = NodeKey> + '_;

    fn iter_range(
        &self,
        first: NodeKey,
        last: NodeKey,
    ) -> Result<impl DoubleEndedIterator<Item = NodeKey> + '_, SpliceRangeError<NodeKey, PathKey>>;
}

pub trait PathEdit: PathView {
    fn insert_after(
        &mut self,
        path_key: PathKey,
        node_to_insert: NodeKey,
        anchor: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>>;

    fn insert_before(
        &mut self,
        path_key: PathKey,
        node_to_insert: NodeKey,
        before: Option<NodeKey>,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>>;

    fn deactivate(
        &mut self,
        node_key: NodeKey,
    ) -> Result<(PathKey, Option<NodeKey>, Option<NodeKey>), NodeInactiveError<NodeKey>>;

    fn move_range_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        destination_node: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>>;

    fn reverse_range(
        &mut self,
        first: NodeKey,
        last: NodeKey,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>>;

    fn move_range_to_path_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest_path: PathKey,
        dest_after: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>>;

    #[inline]
    fn push_front(
        &mut self,
        p: PathKey,
        u: NodeKey,
    ) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        self.insert_after(p, u, None)
    }

    #[inline]
    fn push_back(&mut self, p: PathKey, u: NodeKey) -> Result<(), NodeAlreadyActiveError<NodeKey>> {
        let tail = self.tail(p);
        self.insert_after(p, u, tail)
    }

    #[inline]
    fn remove(
        &mut self,
        u: NodeKey,
    ) -> Result<(PathKey, Option<NodeKey>, Option<NodeKey>), NodeInactiveError<NodeKey>> {
        self.deactivate(u)
    }

    #[inline]
    fn splice_range_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        self.move_range_after(first, last, dest)
    }

    #[inline]
    fn move_after(
        &mut self,
        u: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        self.move_range_after(u, u, dest)
    }

    #[inline]
    fn move_before(
        &mut self,
        u: NodeKey,
        dest: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        match dest {
            Some(d) => self.move_range_after(u, u, self.prev(d)),
            None => {
                let p = self
                    .path_of(u)
                    .ok_or_else(|| SpliceRangeError::InactiveNode(NodeInactiveError::new(u)))?;
                if self.tail(p) == Some(u) {
                    return Ok(());
                }
                self.move_range_after(u, u, self.tail(p))
            }
        }
    }

    #[inline]
    fn move_to_front(&mut self, u: NodeKey) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let p = self
            .path_of(u)
            .ok_or_else(|| SpliceRangeError::InactiveNode(NodeInactiveError::new(u)))?;
        if self.is_head(p, u) {
            return Ok(());
        }
        self.move_range_after(u, u, None)
    }

    #[inline]
    fn move_to_back(&mut self, u: NodeKey) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let p = self
            .path_of(u)
            .ok_or_else(|| SpliceRangeError::InactiveNode(NodeInactiveError::new(u)))?;
        let t = self.tail(p);
        if t == Some(u) {
            return Ok(());
        }
        self.move_range_after(u, u, t)
    }

    #[inline]
    fn move_range_to_path_before(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest_path: PathKey,
        dest_before: Option<NodeKey>,
    ) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        let after = match dest_before {
            Some(db) => self.prev(db),
            None => self.tail(dest_path),
        };
        self.move_range_to_path_after(first, last, dest_path, after)
    }

    #[inline]
    fn reverse_path(&mut self, p: PathKey) -> Result<(), SpliceRangeError<NodeKey, PathKey>> {
        match (self.head(p), self.tail(p)) {
            (Some(h), Some(t)) if h != t => self.reverse_range(h, t),
            _ => Ok(()),
        }
    }

    fn clear_path(&mut self, _p: PathKey);

    #[inline]
    fn move_node_to_path_back(
        &mut self,
        u: NodeKey,
        dest: PathKey,
    ) -> Result<(), NodeInactiveError<NodeKey>> {
        let (src, prev, next) = self.deactivate(u)?;
        let _ = (src, prev, next);
        self.push_back(dest, u)
            .map_err(|_| NodeInactiveError::new(u))
    }
}

pub trait PathAlloc {
    fn create_path(&mut self) -> PathKey;
    fn alloc_node(&mut self) -> NodeKey;
    fn free_node(&mut self, node_key: NodeKey) -> Result<(), NodeNotInactiveError<NodeKey>>;
    fn reserve_nodes(&mut self, additional: usize);
    fn reserve_paths(&mut self, additional: usize);
    fn num_paths(&self) -> usize;
}
