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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArcRewrite {
    tail: usize,
    expected_head: usize,
    new_head: usize,
}

impl ArcRewrite {
    #[inline]
    pub fn new(tail: usize, expected_head: usize, new_head: usize) -> Self {
        Self {
            tail,
            expected_head,
            new_head,
        }
    }

    #[inline]
    pub fn tail(&self) -> usize {
        self.tail
    }

    #[inline]
    pub fn expected_head(&self) -> usize {
        self.expected_head
    }

    #[inline]
    pub fn new_head(&self) -> usize {
        self.new_head
    }
}

impl std::fmt::Display for ArcRewrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArcRewrite(tail: {}, expected_head: {}, new_head: {})",
            self.tail, self.expected_head, self.new_head
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainDelta {
    updates: Vec<ArcRewrite>,
    touched: Vec<usize>,
    marks: Vec<u32>,
    tail_vals: Vec<usize>,
    tail_marks: Vec<u32>,
    tail_generation: u32,
    generation: u32,
    changed_tails: Vec<usize>,
    changed_tail_marks: Vec<u32>,
}

impl Default for ChainDelta {
    fn default() -> Self {
        Self {
            updates: Vec::new(),
            touched: Vec::new(),
            marks: Vec::new(),
            generation: 1,
            tail_vals: Vec::new(),
            tail_marks: Vec::new(),
            tail_generation: 1,
            changed_tails: Vec::new(),
            changed_tail_marks: Vec::new(),
        }
    }
}

impl ChainDelta {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            updates: Vec::with_capacity(cap),
            touched: Vec::with_capacity(cap * 2),
            marks: Vec::new(),
            generation: 1,
            tail_vals: Vec::with_capacity(cap),
            tail_marks: Vec::new(),
            tail_generation: 1,
            changed_tails: Vec::with_capacity(cap),
            changed_tail_marks: Vec::new(),
        }
    }

    #[inline]
    pub fn is_touched(&self, i: usize) -> bool {
        i < self.marks.len() && self.marks[i] == self.generation
    }

    #[inline]
    pub fn touch_many(&mut self, ids: &[usize]) {
        if let Some(&mx) = ids.iter().max() {
            let need = mx + 1;
            if self.marks.len() < need {
                self.marks.resize(need, 0);
            }
        }
        for &i in ids {
            if self.marks[i] != self.generation {
                self.marks[i] = self.generation;
                self.touched.push(i);
            }
        }
    }

    #[inline]
    pub fn changed_tails(&self) -> &[usize] {
        &self.changed_tails
    }

    #[inline]
    pub fn updates(&self) -> &[ArcRewrite] {
        &self.updates
    }

    #[inline]
    pub fn touched(&self) -> &[usize] {
        &self.touched
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.updates.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    #[inline]
    pub fn next_after<'a>(&'a self, base_next: &'a [usize], i: usize) -> usize {
        if i < self.tail_marks.len() && self.tail_marks[i] == self.tail_generation {
            self.tail_vals[i]
        } else {
            base_next[i]
        }
    }

    #[inline]
    pub fn changed(&self, i: usize) -> bool {
        i < self.tail_marks.len() && self.tail_marks[i] == self.tail_generation
    }

    #[inline]
    pub fn push(&mut self, tail: usize, expected_head: usize, new_head: usize) {
        self.updates
            .push(ArcRewrite::new(tail, expected_head, new_head));
        self.touch(tail);
        self.touch(expected_head);
        self.touch(new_head);
        self.set_override(tail, new_head);
    }

    #[inline]
    pub fn push_update(&mut self, u: ArcRewrite) {
        self.updates.push(u);
        self.touch(u.tail());
        self.touch(u.expected_head());
        self.touch(u.new_head());
        self.set_override(u.tail(), u.new_head());
    }

    #[inline]
    pub fn reserve_nodes(&mut self, max_index_inclusive: usize) {
        let need = max_index_inclusive + 1;
        if self.marks.len() < need {
            self.marks.resize(need, 0);
        }
        if self.tail_marks.len() < need {
            self.tail_marks.resize(need, 0);
            self.tail_vals.resize(need, 0);
        }
        if self.changed_tail_marks.len() < need {
            self.changed_tail_marks.resize(need, 0);
        }
    }

    #[inline]
    pub fn override_of(&self, tail: usize) -> Option<usize> {
        if tail < self.tail_marks.len() && self.tail_marks[tail] == self.tail_generation {
            Some(self.tail_vals[tail])
        } else {
            None
        }
    }

    #[inline]
    pub fn iter_changed_tails(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.changed_tails.iter().copied()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.updates.clear();
        self.touched.clear();
        self.changed_tails.clear();

        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }

        self.tail_generation = self.tail_generation.wrapping_add(1);
        if self.tail_generation == 0 {
            self.tail_marks.fill(0);
            self.changed_tail_marks.fill(0);
            self.tail_generation = 1;
        }
    }

    #[inline]
    fn touch(&mut self, i: usize) {
        if i >= self.marks.len() {
            self.marks.resize(i + 1, 0);
        }
        if self.marks[i] != self.generation {
            self.marks[i] = self.generation;
            self.touched.push(i);
        }
    }

    #[inline]
    fn set_override(&mut self, tail: usize, new_head: usize) {
        if tail >= self.tail_marks.len() {
            let need = tail + 1;
            self.tail_marks.resize(need, 0);
            self.tail_vals.resize(need, 0);
        }
        self.tail_vals[tail] = new_head;
        self.tail_marks[tail] = self.tail_generation;
        self.note_changed_tail(tail);
    }

    #[inline]
    fn note_changed_tail(&mut self, t: usize) {
        if t >= self.changed_tail_marks.len() {
            self.changed_tail_marks.resize(t + 1, 0);
        }
        if self.changed_tail_marks[t] != self.tail_generation {
            self.changed_tail_marks[t] = self.tail_generation;
            self.changed_tails.push(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn arcrewrite_new_accessors_and_display() {
        let rw = ArcRewrite::new(1, 2, 3);
        assert_eq!(rw.tail(), 1);
        assert_eq!(rw.expected_head(), 2);
        assert_eq!(rw.new_head(), 3);

        // Display format
        let s = format!("{}", rw);
        assert_eq!(s, "ArcRewrite(tail: 1, expected_head: 2, new_head: 3)");

        // Eq semantics for identical values
        let rw2 = ArcRewrite::new(1, 2, 3);
        assert_eq!(rw, rw2);
    }

    #[test]
    fn chain_delta_new_is_empty() {
        let cd = ChainDelta::new();
        assert!(cd.is_empty());
        assert_eq!(cd.len(), 0);
        assert!(cd.updates().is_empty());
        assert!(cd.touched().is_empty());
    }

    #[test]
    fn chain_delta_with_capacity_and_reserve_nodes() {
        let mut cd = ChainDelta::with_capacity(4);
        assert!(cd.is_empty());

        // Reserve a larger node index and then touch it; should not panic and should record touches.
        cd.reserve_nodes(10);
        cd.push(10, 11, 10);
        // dedup of the same index across the three touches in push
        assert_eq!(cd.touched(), &[10, 11]);
        assert_eq!(cd.len(), 1);
        assert_eq!(cd.updates(), &[ArcRewrite::new(10, 11, 10)]);

        // Reserving a smaller number than current capacity should be a no-op
        cd.reserve_nodes(5);
        // Add another update to ensure behavior continues as expected
        cd.push(12, 13, 14);
        assert_eq!(
            cd.touched(),
            &[10, 11, 12, 13, 14],
            "touches should deduplicate within a generation and preserve order of first-touch"
        );
        assert_eq!(
            cd.updates(),
            &[ArcRewrite::new(10, 11, 10), ArcRewrite::new(12, 13, 14)]
        );
    }

    #[test]
    fn push_records_update_and_touched_dedup() {
        let mut cd = ChainDelta::new();
        // Note that 5 is repeated; dedup should keep only one instance in touched for this generation.
        cd.push(5, 7, 5);

        assert_eq!(cd.len(), 1);
        assert_eq!(cd.updates(), &[ArcRewrite::new(5, 7, 5)]);
        assert_eq!(
            cd.touched(),
            &[5, 7],
            "duplicate node indices in a single push should be deduplicated"
        );
    }

    #[test]
    fn push_update_behaves_like_push() {
        let mut cd = ChainDelta::new();
        let u = ArcRewrite::new(2, 3, 4);
        cd.push_update(u);

        assert_eq!(cd.len(), 1);
        assert_eq!(cd.updates(), &[u]);
        assert_eq!(cd.touched(), &[2, 3, 4]);
    }

    #[test]
    fn touched_dedup_across_multiple_pushes_in_same_generation() {
        let mut cd = ChainDelta::new();

        // 3 is touched in both pushes; it should appear only once in touched.
        cd.push(1, 2, 3);
        cd.push(3, 4, 5);

        assert_eq!(cd.len(), 2);
        assert_eq!(
            cd.touched(),
            &[1, 2, 3, 4, 5],
            "node 3 should only appear once despite being touched in both pushes"
        );
    }

    #[test]
    fn clear_resets_updates_and_touched_and_allows_retouch() {
        let mut cd = ChainDelta::new();

        cd.push(1, 2, 3);
        assert_eq!(cd.updates(), &[ArcRewrite::new(1, 2, 3)]);
        assert_eq!(cd.touched(), &[1, 2, 3]);

        cd.clear();
        assert!(cd.is_empty());
        assert!(cd.updates().is_empty());
        assert!(cd.touched().is_empty());

        // Touching the same nodes after clear should add them back to touched for the new generation
        cd.push(3, 4, 5);
        assert_eq!(cd.len(), 1);
        assert_eq!(cd.updates(), &[ArcRewrite::new(3, 4, 5)]);
        assert_eq!(
            cd.touched(),
            &[3, 4, 5],
            "after clear, nodes should be touchable again in the next generation"
        );
    }

    #[test]
    fn updates_order_is_preserved() {
        let mut cd = ChainDelta::new();
        cd.push(1, 2, 3);
        cd.push(4, 5, 6);
        cd.push_update(ArcRewrite::new(7, 8, 9));

        assert_eq!(
            cd.updates(),
            &[
                ArcRewrite::new(1, 2, 3),
                ArcRewrite::new(4, 5, 6),
                ArcRewrite::new(7, 8, 9),
            ]
        );
    }

    #[test]
    fn chain_delta_clone_and_eq() {
        let mut a = ChainDelta::new();
        a.push(1, 2, 3);
        a.push(4, 5, 6);

        let b = a.clone();
        assert_eq!(a, b);

        let mut c = ChainDelta::new();
        c.push(1, 2, 3);
        c.push(4, 5, 6);
        assert_eq!(
            a, c,
            "two independently built, identical deltas should be equal"
        );

        // After clear, identical states should still compare equal
        let mut x = a.clone();
        let mut y = c.clone();
        x.clear();
        y.clear();
        assert_eq!(x, y);
        assert!(x.is_empty() && y.is_empty());
    }

    #[test]
    fn complex_mixture_of_push_and_push_update_with_dedup() {
        let mut cd = ChainDelta::new();

        // Mix push and push_update and ensure touched remains a set (per generation) in order of first appearance.
        cd.push(10, 20, 30); // touches: 10,20,30
        cd.push_update(ArcRewrite::new(30, 40, 50)); // touches new: 40,50 (30 already touched)
        cd.push(50, 60, 10); // touches new: 60 (50,10 already touched)
        cd.push_update(ArcRewrite::new(70, 80, 20)); // touches new: 70,80 (20 already touched)

        assert_eq!(cd.touched(), &[10, 20, 30, 40, 50, 60, 70, 80]);

        assert_eq!(cd.len(), 4);
        assert_eq!(
            cd.updates(),
            &[
                ArcRewrite::new(10, 20, 30),
                ArcRewrite::new(30, 40, 50),
                ArcRewrite::new(50, 60, 10),
                ArcRewrite::new(70, 80, 20),
            ]
        );
    }

    #[test]
    fn touched_multiple_duplicates_collapse_to_one() {
        let mut cd = ChainDelta::new();

        // Repeatedly touch the same node (42) in various combinations
        cd.push(42, 42, 42); // first touch of 42
        cd.push_update(ArcRewrite::new(42, 42, 42)); // all duplicates
        cd.push(42, 43, 42); // introduces 43
        cd.push(44, 45, 42); // introduces 44, 45
        cd.push(42, 46, 47); // introduces 46, 47

        let touched = cd.touched();

        // 42 should appear only once
        let cnt_42 = touched.iter().filter(|&&x| x == 42).count();
        assert_eq!(cnt_42, 1, "node 42 should only appear once in touched");

        // Order should reflect first-touch order: 42 first, followed by newly-seen nodes
        assert_eq!(touched, &[42, 43, 44, 45, 46, 47]);
    }

    #[test]
    fn is_touched_basic_and_out_of_bounds() {
        let mut cd = ChainDelta::new();

        // Initially nothing is touched; out-of-bounds should return false too
        assert!(!cd.is_touched(0));
        assert!(!cd.is_touched(999_999));

        // After a push, those nodes are touched
        cd.push(10, 20, 10); // 10 and 20 should be touched (10 deduped)
        assert!(cd.is_touched(10));
        assert!(cd.is_touched(20));

        // Unseen in-bounds and out-of-bounds should be false
        assert!(!cd.is_touched(21));
        assert!(!cd.is_touched(999_999));
    }

    #[test]
    fn touch_many_basic_dedup_and_order() {
        let mut cd = ChainDelta::new();

        // touch_many dedups within the same call and preserves order of first appearance
        cd.touch_many(&[5, 3, 5, 7, 3]);
        assert_eq!(cd.touched(), &[5, 3, 7]);
        assert!(cd.is_touched(5) && cd.is_touched(3) && cd.is_touched(7));
        assert!(!cd.is_touched(6));

        // Calling again in the same generation should only append new ids in order
        cd.touch_many(&[3, 8, 5, 9, 8]);
        assert_eq!(cd.touched(), &[5, 3, 7, 8, 9]);
        assert!(cd.is_touched(8) && cd.is_touched(9));
    }

    #[test]
    fn touch_many_empty_is_noop() {
        let mut cd = ChainDelta::new();
        cd.touch_many(&[]);
        assert!(cd.touched().is_empty());
        assert!(cd.updates().is_empty());

        // After something is touched, an empty call is still a no-op
        cd.touch_many(&[1, 2]);
        let before = cd.touched().to_vec();
        cd.touch_many(&[]);
        assert_eq!(cd.touched(), &before[..]);
    }

    #[test]
    fn touch_many_resizes_and_interacts_with_clear() {
        let mut cd = ChainDelta::new();

        // Large index should cause internal resize without panic
        cd.touch_many(&[1_000]);
        assert!(cd.is_touched(1_000));
        assert_eq!(cd.touched(), &[1_000]);

        // clear should reset the generation and touched state
        cd.clear();
        assert!(!cd.is_touched(1_000));
        assert!(cd.touched().is_empty());

        // Touch again after clear; nodes should be recordable again
        cd.touch_many(&[1_000, 500]);
        assert_eq!(cd.touched(), &[1_000, 500]);
        assert!(cd.is_touched(500));
    }

    #[test]
    fn touch_many_with_prior_push_preserves_global_order() {
        let mut cd = ChainDelta::new();

        // First, some touches come via push
        cd.push(1, 2, 3);
        assert_eq!(cd.touched(), &[1, 2, 3]);

        // Now use touch_many containing a mix of previously touched and new ids
        cd.touch_many(&[3, 4, 1, 5, 2, 5]);

        // Only new ids (4, 5) should be appended in first-seen order
        assert_eq!(cd.touched(), &[1, 2, 3, 4, 5]);

        // is_touched reflects the same generation state
        for i in [1, 2, 3, 4, 5] {
            assert!(cd.is_touched(i));
        }
        assert!(!cd.is_touched(6));
    }

    #[test]
    fn is_touched_and_reserve_nodes_integration() {
        let mut cd = ChainDelta::new();

        // Reserve should not mark things as touched
        cd.reserve_nodes(50);
        assert!(!cd.is_touched(50));
        assert!(cd.touched().is_empty());

        // touch_many should activate the mark
        cd.touch_many(&[50]);
        assert!(cd.is_touched(50));
        assert_eq!(cd.touched(), &[50]);
    }

    #[test]
    fn arcrewrite_is_hashable_and_dedups_in_set() {
        let a = ArcRewrite::new(1, 2, 3);
        let b = ArcRewrite::new(1, 2, 3);
        let c = ArcRewrite::new(1, 2, 4);

        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(
            !set.insert(b),
            "identical ArcRewrite should dedup in HashSet"
        );
        assert!(set.insert(c), "different new_head makes it distinct");
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn next_after_uses_base_when_unmodified_and_changed_false() {
        let cd = ChainDelta::new();
        let base_next = vec![10, 11, 12, 13, 14];

        for i in 0..base_next.len() {
            assert_eq!(
                cd.next_after(&base_next, i),
                base_next[i],
                "without overrides, next_after should use base_next"
            );
            assert!(
                !cd.changed(i),
                "changed() should be false for all indices when no overrides exist"
            );
        }

        // Out-of-range for changed is false
        assert!(!cd.changed(1000));
    }

    #[test]
    fn next_after_reflects_single_override_and_changed_true() {
        let mut cd = ChainDelta::new();
        let base_next = vec![0, 1, 2, 3, 4, 5, 6];

        // Override tail=2 -> new_head=9 (base had 2->2)
        cd.push(2, 99, 9);

        // Unchanged tails use base
        assert_eq!(cd.next_after(&base_next, 1), 1);
        assert!(!cd.changed(1));

        // Changed tail uses override
        assert_eq!(cd.next_after(&base_next, 2), 9);
        assert!(cd.changed(2));

        // A different index is not changed
        assert!(!cd.changed(5));
        assert_eq!(cd.next_after(&base_next, 5), 5);
    }

    #[test]
    fn multiple_overrides_last_wins_for_same_tail() {
        let mut cd = ChainDelta::new();
        let base_next = vec![0, 1, 2, 3, 4, 5];

        cd.push(3, 100, 7);
        assert_eq!(cd.next_after(&base_next, 3), 7);
        assert!(cd.changed(3));

        // New override for the same tail should take effect
        cd.push(3, 101, 8);
        assert_eq!(
            cd.next_after(&base_next, 3),
            8,
            "latest override for a tail must win"
        );
        assert!(cd.changed(3));

        // Non-overridden tails unchanged
        assert_eq!(cd.next_after(&base_next, 4), 4);
        assert!(!cd.changed(4));
    }

    #[test]
    fn push_update_sets_override_and_changed() {
        let mut cd = ChainDelta::new();
        let base_next = vec![0, 1, 2, 3, 4];

        let u = ArcRewrite::new(1, 2, 42);
        cd.push_update(u);

        assert!(cd.changed(1));
        assert_eq!(cd.next_after(&base_next, 1), 42);

        // Others use base
        assert!(!cd.changed(0));
        assert_eq!(cd.next_after(&base_next, 0), 0);
    }

    #[test]
    fn clear_resets_overrides_and_changed() {
        let mut cd = ChainDelta::new();
        let base_next = vec![0, 1, 2, 3, 4];

        cd.push(2, 9, 99);
        assert!(cd.changed(2));
        assert_eq!(cd.next_after(&base_next, 2), 99);

        cd.clear();

        // After clear, no overrides should be active
        assert!(!cd.changed(2));
        assert_eq!(cd.next_after(&base_next, 2), 2);

        // Touched and updates are cleared too (covered elsewhere but kept here for coherence)
        assert!(cd.touched().is_empty());
        assert!(cd.updates().is_empty());
    }

    #[test]
    fn reserve_nodes_resizes_tail_buffers_and_leaves_state_unmodified() {
        let mut cd = ChainDelta::new();
        let base_next = (0..=20).collect::<Vec<usize>>();

        // Reserve should allow safe queries for indices without implying a change
        cd.reserve_nodes(20);

        // No changes yet
        for i in 0..=20 {
            assert_eq!(cd.next_after(&base_next, i), base_next[i]);
            assert!(!cd.changed(i));
        }

        // Now perform an override at a high index and ensure it applies
        cd.push(20, 0, 777);
        assert!(cd.changed(20));
        assert_eq!(cd.next_after(&base_next, 20), 777);

        // Neighbor indices remain unchanged
        assert!(!cd.changed(19));
        assert_eq!(cd.next_after(&base_next, 19), 19);
    }

    #[test]
    fn changed_tails_lists_unique_tails_last_wins() {
        let mut cd = ChainDelta::new();
        cd.push(3, 0, 10);
        cd.push(5, 0, 11);
        cd.push(3, 10, 12); // same tail again

        let mut v = cd.changed_tails().to_vec();
        v.sort_unstable();
        assert_eq!(v, vec![3, 5]); // dedup per generation

        assert_eq!(cd.override_of(3), Some(12)); // last wins
        assert_eq!(cd.override_of(5), Some(11));
    }

    #[test]
    fn changed_tails_clears_on_clear() {
        let mut cd = ChainDelta::new();
        cd.push(1, 0, 2);
        assert_eq!(cd.changed_tails(), &[1]);
        cd.clear();
        assert!(cd.changed_tails().is_empty());
    }
}
