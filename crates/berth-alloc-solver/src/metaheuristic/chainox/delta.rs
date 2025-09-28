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

use crate::metaheuristic::chainox::arena::{NodeKey, PathArena, PathKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArcRewire {
    /// Tail node whose successor we want to change.
    pub tail: NodeKey,
    /// Optional CAS guard: only apply if the current successor equals `old_head`.
    pub old_head: Option<NodeKey>,
    /// New successor (None means "no successor" / end of path).
    pub new_head: Option<NodeKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentRehome {
    /// First node (inclusive) of the contiguous segment to move.
    pub first: NodeKey,
    /// Last node (inclusive) of the contiguous segment to move.
    pub last: NodeKey,
    /// Destination path for the segment.
    pub dest_path: PathKey,
    /// Insert segment after this node. None => to the *front* of `dest_path`.
    pub insert_after: Option<NodeKey>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PathDelta {
    /// Optional successor rewires (coalesced last-writer-wins).
    pub arc_rewires: Vec<ArcRewire>,
    /// High-level segment moves; to be applied with arena splice ops.
    pub rehomes: Vec<SegmentRehome>,
}

impl PathDelta {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.arc_rewires.is_empty() && self.rehomes.is_empty()
    }
    #[inline]
    pub fn push_rewire(&mut self, r: ArcRewire) {
        self.arc_rewires.push(r);
    }
    #[inline]
    pub fn push_rehome(&mut self, r: SegmentRehome) {
        self.rehomes.push(r);
    }
}

/// A very permissive recorder of edits against a snapshot `&PathArena`.
/// - **No validation**. No errors.
/// - Best-effort inference of `dest_path` and anchors from `view`.
/// - Coalesces arc rewires (last-writer-wins per `tail`) on `finish()`.
#[derive(Debug)]
pub struct PathDeltaBuilder<'a> {
    view: &'a PathArena,
    arc_rewires: Vec<ArcRewire>,
    rehomes: Vec<SegmentRehome>,
}

impl<'a> PathDeltaBuilder<'a> {
    #[inline]
    pub fn new(view: &'a PathArena) -> Self {
        Self {
            view,
            arc_rewires: Vec::new(),
            rehomes: Vec::new(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.arc_rewires.is_empty() && self.rehomes.is_empty()
    }

    /// Clear recorded edits (keeps borrowing the same view).
    #[inline]
    pub fn clear(&mut self) {
        self.arc_rewires.clear();
        self.rehomes.clear();
    }

    /// Finish and coalesce arc rewires (last-writer-wins per `tail`) in O(K).
    #[inline]
    pub fn finish(self) -> PathDelta {
        let arc_rewires = coalesce_arc_rewires_dense(self.arc_rewires);
        PathDelta {
            arc_rewires,
            rehomes: self.rehomes,
        }
    }

    /// Record an arc rewire `tail -> new_head` (optionally guarded by `old_head`).
    #[inline]
    pub fn rewire_arc(
        &mut self,
        tail: NodeKey,
        old_head: Option<NodeKey>,
        new_head: Option<NodeKey>,
    ) -> &mut Self {
        self.arc_rewires.push(ArcRewire {
            tail,
            old_head,
            new_head,
        });
        self
    }

    /// Record a segment rehome `[first..=last]` → `dest_path` after `insert_after` (None => front).
    #[inline]
    pub fn rehome(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest_path: PathKey,
        insert_after: Option<NodeKey>,
    ) -> &mut Self {
        self.rehomes.push(SegmentRehome {
            first,
            last,
            dest_path,
            insert_after,
        });
        self
    }

    /// Move a single node within *some* path after `after` (or to front if `None`).
    /// Best-effort `dest_path`: prefer `after`'s path, else `u`'s path, else PathKey(0).
    #[inline]
    pub fn move_after(&mut self, u: NodeKey, after: Option<NodeKey>) -> &mut Self {
        let dest_path = after
            .and_then(|a| self.view.path_of(a))
            .or_else(|| self.view.path_of(u))
            .unwrap_or(PathKey::from_raw(0));
        self.rehome(u, u, dest_path, after)
    }

    /// Move a single node before `before`. If `before` is `None`, to the back of its path.
    #[inline]
    pub fn move_before(&mut self, u: NodeKey, before: Option<NodeKey>) -> &mut Self {
        match before {
            Some(b) => {
                let after = self.view.prev(b);
                // dest_path best-effort from `b`, otherwise `u`, otherwise 0
                let dest_path = self
                    .view
                    .path_of(b)
                    .or_else(|| self.view.path_of(u))
                    .unwrap_or(PathKey::from_raw(0));
                self.rehome(u, u, dest_path, after)
            }
            None => {
                // to back: after current tail of u's path (best-effort)
                let p = self.view.path_of(u).unwrap_or(PathKey::from_raw(0));
                let after = self.view.tail(p);
                self.rehome(u, u, p, after)
            }
        }
    }

    /// Move contiguous segment `[first..=last]` after `after` (or to front if `None`) within *some* path.
    /// Best-effort `dest_path`: prefer `after`'s path, else `first`'s path, else PathKey(0).
    #[inline]
    pub fn move_range_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        after: Option<NodeKey>,
    ) -> &mut Self {
        let dest_path = after
            .and_then(|a| self.view.path_of(a))
            .or_else(|| self.view.path_of(first))
            .unwrap_or(PathKey::from_raw(0));
        self.rehome(first, last, dest_path, after)
    }

    /// Cross-/same-path move: `[first..=last]` to `dest_path` after `after` (front if `None`).
    #[inline]
    pub fn move_range_to_path_after(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest_path: PathKey,
        after: Option<NodeKey>,
    ) -> &mut Self {
        self.rehome(first, last, dest_path, after)
    }

    /// Cross-/same-path move: `[first..=last]` to `dest_path` *before* `before` (back if `None`).
    #[inline]
    pub fn move_range_to_path_before(
        &mut self,
        first: NodeKey,
        last: NodeKey,
        dest_path: PathKey,
        before: Option<NodeKey>,
    ) -> &mut Self {
        let after = match before {
            Some(b) => self.view.prev(b),
            None => self.view.tail(dest_path),
        };
        self.rehome(first, last, dest_path, after)
    }

    /// Move a whole path `src` to the *front* of `dst` (keeps node order).
    #[inline]
    pub fn prepend_path(&mut self, src: PathKey, dst: PathKey) -> &mut Self {
        if src == dst {
            return self;
        }
        if let (Some(f), Some(l)) = (self.view.head(src), self.view.tail(src)) {
            self.rehome(f, l, dst, None);
        }
        self
    }

    /// Move a whole path `src` to the *back* of `dst` (keeps node order).
    #[inline]
    pub fn append_path(&mut self, src: PathKey, dst: PathKey) -> &mut Self {
        if src == dst {
            return self;
        }
        if let (Some(f), Some(l)) = (self.view.head(src), self.view.tail(src)) {
            let after = self.view.tail(dst);
            self.rehome(f, l, dst, after);
        }
        self
    }

    /// Move a single node `u` to another path `dest_path` after `after` (or to front).
    #[inline]
    pub fn move_node_to_path_after(
        &mut self,
        u: NodeKey,
        dest_path: PathKey,
        after: Option<NodeKey>,
    ) -> &mut Self {
        self.move_range_to_path_after(u, u, dest_path, after)
    }

    /// Move a node `u` before `before` in `dest_path` (or to back).
    #[inline]
    pub fn move_node_to_path_before(
        &mut self,
        u: NodeKey,
        dest_path: PathKey,
        before: Option<NodeKey>,
    ) -> &mut Self {
        self.move_range_to_path_before(u, u, dest_path, before)
    }

    /// Move a node to the front of its current path (best-effort).
    #[inline]
    pub fn move_to_front(&mut self, u: NodeKey) -> &mut Self {
        let p = self.view.path_of(u).unwrap_or(PathKey::from_raw(0));
        self.rehome(u, u, p, None)
    }

    /// Move a node to the back of its current path (best-effort).
    #[inline]
    pub fn move_to_back(&mut self, u: NodeKey) -> &mut Self {
        let p = self.view.path_of(u).unwrap_or(PathKey::from_raw(0));
        let after = self.view.tail(p);
        self.rehome(u, u, p, after)
    }

    /// Bulk: move many single nodes to the back of `dest_path` preserving order (best-effort).
    #[inline]
    pub fn move_many_to_path_back<I>(&mut self, nodes: I, dest_path: PathKey) -> &mut Self
    where
        I: IntoIterator<Item = NodeKey>,
    {
        let mut after = self.view.tail(dest_path);
        for u in nodes {
            self.move_node_to_path_after(u, dest_path, after);
            after = Some(u); // chain
        }
        self
    }
}

#[inline(always)]
fn coalesce_arc_rewires_dense(v: Vec<ArcRewire>) -> Vec<ArcRewire> {
    use std::collections::BTreeMap;

    // Insert in input order so later entries overwrite earlier ones (last-writer-wins).
    let mut by_tail: BTreeMap<usize, ArcRewire> = BTreeMap::new();
    for r in v {
        by_tail.insert(r.tail.get(), r);
    }
    // BTreeMap iterates in key order => ascending by tail.get().
    by_tail.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metaheuristic::chainox::arena::{NodeKey, PathArena, PathKey};

    fn mk_path(arena: &mut PathArena, len: usize) -> (PathKey, Vec<NodeKey>) {
        let p = arena.create_path();
        let mut nodes = Vec::with_capacity(len);
        for _ in 0..len {
            let u = arena.alloc_node();
            arena.push_back(p, u).unwrap();
            nodes.push(u);
        }
        (p, nodes)
    }

    #[test]
    fn pathdelta_api_basics() {
        let mut d = PathDelta::new();
        assert!(d.is_empty());
        let mut arena = PathArena::new();
        let p = arena.create_path();
        let u = arena.alloc_node();
        arena.push_back(p, u).unwrap();

        d.push_rehome(SegmentRehome {
            first: u,
            last: u,
            dest_path: p,
            insert_after: None,
        });
        assert!(!d.is_empty());
        assert_eq!(d.rehomes.len(), 1);
        assert!(d.arc_rewires.is_empty());

        // Arc rewires collected into PathDelta
        d.push_rewire(ArcRewire {
            tail: u,
            old_head: None,
            new_head: None,
        });
        assert_eq!(d.arc_rewires.len(), 1);
    }

    #[test]
    fn builder_clear_and_is_empty() {
        let mut arena = PathArena::new();
        let (p, nodes) = mk_path(&mut arena, 2);
        let (a, b) = (nodes[0], nodes[1]);

        let mut bld = PathDeltaBuilder::new(&arena);
        assert!(bld.is_empty());

        bld.rehome(a, a, p, Some(b));
        assert!(!bld.is_empty());

        bld.clear();
        assert!(bld.is_empty());

        let d = bld.finish();
        assert!(d.rehomes.is_empty() && d.arc_rewires.is_empty());
    }

    #[test]
    fn rewire_arc_finish_coalesces_last_writer_wins_and_orders_by_tail_index() {
        let mut arena = PathArena::new();
        // Create a bunch of nodes to act as tails/heads
        let (_p1, n1) = mk_path(&mut arena, 4);
        let (_p2, n2) = mk_path(&mut arena, 3);

        // Tails: t1 < t2 < t3 < t4 by allocation order
        let (t1, _, t3, t4) = (n1[0], n1[1], n1[2], n1[3]);
        // Heads (arbitrary)
        let (h1, h2, h3) = (n2[0], n2[1], n2[2]);

        let mut bld = PathDeltaBuilder::new(&arena);
        // Interleave rewires across tails with some overwrites
        bld.rewire_arc(t3, Some(h1), Some(h2)) // 1st for t3
            .rewire_arc(t1, None, Some(h1)) // 1st for t1
            .rewire_arc(t3, None, Some(h3)) // overwrite t3 (last wins)
            .rewire_arc(t4, Some(h2), None) // only for t4
            .rewire_arc(t1, Some(h2), Some(h2)); // overwrite t1 (last wins)

        let dlt = bld.finish();

        // Expect exactly one per tail: t1, t3, t4
        // And in ascending order by tail.get() (dense coalescer produces this order)
        assert_eq!(dlt.arc_rewires.len(), 3);

        let tails: Vec<usize> = dlt.arc_rewires.iter().map(|r| r.tail.get()).collect();
        let mut sorted = tails.clone();
        sorted.sort_unstable();
        assert_eq!(tails, sorted);

        // Validate last-writer for each tail
        let find = |tail: NodeKey| dlt.arc_rewires.iter().find(|r| r.tail == tail).cloned();
        let r_t1 = find(t1).expect("t1 rewire");
        let r_t3 = find(t3).expect("t3 rewire");
        let r_t4 = find(t4).expect("t4 rewire");

        assert_eq!(
            r_t1,
            ArcRewire {
                tail: t1,
                old_head: Some(h2),
                new_head: Some(h2)
            }
        );
        assert_eq!(
            r_t3,
            ArcRewire {
                tail: t3,
                old_head: None,
                new_head: Some(h3)
            }
        );
        assert_eq!(
            r_t4,
            ArcRewire {
                tail: t4,
                old_head: Some(h2),
                new_head: None
            }
        );
    }

    #[test]
    fn rehome_records_exactly_as_requested() {
        let mut arena = PathArena::new();
        let (p1, n1) = mk_path(&mut arena, 3);
        let (p2, n2) = mk_path(&mut arena, 2);
        let (a, b, c) = (n1[0], n1[1], n1[2]);
        let (x, y) = (n2[0], n2[1]);

        let mut bld = PathDeltaBuilder::new(&arena);
        bld.rehome(b, c, p2, Some(x))
            .rehome(a, a, p1, None)
            .rehome(x, y, p1, Some(c));
        let d = bld.finish();

        assert_eq!(
            d.rehomes,
            vec![
                SegmentRehome {
                    first: b,
                    last: c,
                    dest_path: p2,
                    insert_after: Some(x)
                },
                SegmentRehome {
                    first: a,
                    last: a,
                    dest_path: p1,
                    insert_after: None
                },
                SegmentRehome {
                    first: x,
                    last: y,
                    dest_path: p1,
                    insert_after: Some(c)
                },
            ]
        );
    }

    #[test]
    fn move_after_prefers_after_path_then_u_path_then_zero() {
        let mut arena = PathArena::new();
        let (p1, n1) = mk_path(&mut arena, 2);
        let (p2, n2) = mk_path(&mut arena, 1);
        let (a, b) = (n1[0], n1[1]);
        let x = n2[0];

        // Inactive node (never pushed)
        let inactive = arena.alloc_node();

        // 1) after Some(x) -> dest p2
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_after(a, Some(x));
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: a,
                dest_path: p2,
                insert_after: Some(x)
            }]
        );

        // 2) after None -> use u's path p1, insert front (None)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_after(b, None);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: b,
                last: b,
                dest_path: p1,
                insert_after: None
            }]
        );

        // 3) both unknown (inactive u, after None) -> dest PathKey(0)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_after(inactive, None);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: inactive,
                last: inactive,
                dest_path: PathKey::from_raw(0),
                insert_after: None
            }]
        );
    }

    #[test]
    fn move_before_variants() {
        let mut arena = PathArena::new();
        let (p1, n1) = mk_path(&mut arena, 3);
        let (a, _, c) = (n1[0], n1[1], n1[2]);

        // before Some(a) -> after prev(a)=None, dest from `a` -> p1
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_before(c, Some(a));
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: c,
                last: c,
                dest_path: p1,
                insert_after: None
            }]
        );

        // before None -> to back of its own path (after tail)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_before(a, None);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: a,
                dest_path: p1,
                insert_after: arena.tail(p1)
            }]
        );
    }

    #[test]
    fn move_range_after_best_effort() {
        let mut arena = PathArena::new();
        let (p1, n1) = mk_path(&mut arena, 4);
        let (p2, n2) = mk_path(&mut arena, 2);
        let (a, b, c, d) = (n1[0], n1[1], n1[2], n1[3]);
        let (x, _y) = (n2[0], n2[1]);

        // after Some(x) -> dest_path from x (p2)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_range_after(b, c, Some(x));
        let dlt = bld.finish();
        assert_eq!(
            dlt.rehomes,
            vec![SegmentRehome {
                first: b,
                last: c,
                dest_path: p2,
                insert_after: Some(x)
            }]
        );

        // after None -> dest_path from first node's path (p1), to front
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_range_after(a, d, None);
        let dlt = bld.finish();
        assert_eq!(
            dlt.rehomes,
            vec![SegmentRehome {
                first: a,
                last: d,
                dest_path: p1,
                insert_after: None
            }]
        );
    }

    #[test]
    fn move_range_to_path_after_and_before() {
        let mut arena = PathArena::new();
        let (_, n1) = mk_path(&mut arena, 3);
        let (p2, n2) = mk_path(&mut arena, 2);
        let (a, b, c) = (n1[0], n1[1], n1[2]);
        let (x, y) = (n2[0], n2[1]);

        // Explicit to path after Some(y)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_range_to_path_after(a, b, p2, Some(y));
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: b,
                dest_path: p2,
                insert_after: Some(y)
            }]
        );

        // Explicit to path after None (front)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_range_to_path_after(b, c, p2, None);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: b,
                last: c,
                dest_path: p2,
                insert_after: None
            }]
        );

        // Before Some(x) -> after prev(x)=None (since x is head), so front
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_range_to_path_before(a, a, p2, Some(x));
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: a,
                dest_path: p2,
                insert_after: None
            }]
        );

        // Before None -> to back (after tail)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_range_to_path_before(a, b, p2, None);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: b,
                dest_path: p2,
                insert_after: arena.tail(p2)
            }]
        );
    }

    #[test]
    fn prepend_and_append_whole_path_cases() {
        let mut arena = PathArena::new();
        let (p1, n1) = mk_path(&mut arena, 3);
        let (p2, n2) = mk_path(&mut arena, 2);
        let (a, _b, c) = (n1[0], n1[1], n1[2]);
        let (_x, y) = (n2[0], n2[1]);

        // Same src/dst -> no-op (no rehomes)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.prepend_path(p1, p1).append_path(p2, p2);
        let d = bld.finish();
        assert!(d.rehomes.is_empty());

        // Empty src path -> no-op
        let p_empty = arena.create_path();
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.prepend_path(p_empty, p1).append_path(p_empty, p2);
        let d = bld.finish();
        assert!(d.rehomes.is_empty());

        // Prepend p1 to p2 -> [a..c] to front of p2
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.prepend_path(p1, p2);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: c,
                dest_path: p2,
                insert_after: None
            }]
        );

        // Append p1 to p2 -> [a..c] after tail(y)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.append_path(p1, p2);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: c,
                dest_path: p2,
                insert_after: Some(y)
            }]
        );
    }

    #[test]
    fn move_node_to_path_variants() {
        let mut arena = PathArena::new();
        let (_, n1) = mk_path(&mut arena, 2);
        let (p2, n2) = mk_path(&mut arena, 2);
        let (a, b) = (n1[0], n1[1]);
        let (x, y) = (n2[0], n2[1]);

        // After Some(y)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_node_to_path_after(a, p2, Some(y));
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: a,
                dest_path: p2,
                insert_after: Some(y)
            }]
        );

        // Before Some(x) (x is head) -> front
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_node_to_path_before(b, p2, Some(x));
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: b,
                last: b,
                dest_path: p2,
                insert_after: None
            }]
        );

        // Before None -> to back after tail(y)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_node_to_path_before(a, p2, None);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: a,
                dest_path: p2,
                insert_after: Some(y)
            }]
        );
    }

    #[test]
    fn move_to_front_and_back() {
        let mut arena = PathArena::new();
        let (p, n) = mk_path(&mut arena, 3);
        let (a, b, _) = (n[0], n[1], n[2]);

        // Front
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_to_front(b);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: b,
                last: b,
                dest_path: p,
                insert_after: None
            }]
        );

        // Back (after tail)
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_to_back(a);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![SegmentRehome {
                first: a,
                last: a,
                dest_path: p,
                insert_after: arena.tail(p)
            }]
        );
    }

    #[test]
    fn move_many_to_path_back_chains_after_correctly_for_empty_and_nonempty_dest() {
        let mut arena = PathArena::new();
        let (src, n1) = mk_path(&mut arena, 3);
        let (dst_nonempty, n2) = mk_path(&mut arena, 2);
        let dst_empty = arena.create_path();

        let (a, b, c) = (n1[0], n1[1], n1[2]);
        let (_x, y) = (n2[0], n2[1]);

        // Non-empty dest: first after tail(y), then after a
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_many_to_path_back([a, b, c], dst_nonempty);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![
                SegmentRehome {
                    first: a,
                    last: a,
                    dest_path: dst_nonempty,
                    insert_after: Some(y)
                },
                SegmentRehome {
                    first: b,
                    last: b,
                    dest_path: dst_nonempty,
                    insert_after: Some(a)
                },
                SegmentRehome {
                    first: c,
                    last: c,
                    dest_path: dst_nonempty,
                    insert_after: Some(b)
                },
            ]
        );

        // Empty dest: first after None, then after first, etc.
        let mut bld = PathDeltaBuilder::new(&arena);
        bld.move_many_to_path_back([a, c], dst_empty);
        let d = bld.finish();
        assert_eq!(
            d.rehomes,
            vec![
                SegmentRehome {
                    first: a,
                    last: a,
                    dest_path: dst_empty,
                    insert_after: None
                },
                SegmentRehome {
                    first: c,
                    last: c,
                    dest_path: dst_empty,
                    insert_after: Some(a)
                },
            ]
        );

        let _ = src; // silence if unused
    }

    #[test]
    fn chaining_builder_methods_returns_self() {
        let mut arena = PathArena::new();
        let (p1, n1) = mk_path(&mut arena, 2);
        let (p2, n2) = mk_path(&mut arena, 1);
        let (a, b) = (n1[0], n1[1]);
        let x = n2[0];

        let mut bld = PathDeltaBuilder::new(&arena);
        bld.rewire_arc(a, None, Some(b))
            .move_after(a, Some(b))
            .move_before(b, None)
            .move_range_after(a, b, Some(x))
            .move_range_to_path_after(a, b, p2, Some(x))
            .move_range_to_path_before(a, b, p2, None)
            .move_node_to_path_after(a, p2, None)
            .move_node_to_path_before(b, p2, Some(x))
            .prepend_path(p1, p2)
            .append_path(p1, p2)
            .move_to_front(a)
            .move_to_back(b)
            .move_many_to_path_back([a, b], p2);

        let d = bld.finish();
        // We only assert that something was recorded and the one arc rewire exists post-coalesce.
        assert!(!d.rehomes.is_empty());
        assert_eq!(d.arc_rewires.len(), 1);
    }

    #[test]
    fn coalesce_function_handles_empty_and_trivial() {
        // Directly exercise the helper
        let v: Vec<ArcRewire> = vec![];
        assert!(super::coalesce_arc_rewires_dense(v).is_empty());

        let mut arena = PathArena::new();
        let (_p, nodes) = mk_path(&mut arena, 1);
        let t = nodes[0];
        let out = super::coalesce_arc_rewires_dense(vec![ArcRewire {
            tail: t,
            old_head: None,
            new_head: None,
        }]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tail, t);
    }
}
