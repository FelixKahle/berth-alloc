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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainNextRewire {
    tail: usize,
    successor: usize,
}

impl ChainNextRewire {
    #[inline]
    pub fn new(tail: usize, successor: usize) -> Self {
        Self { tail, successor }
    }

    #[inline]
    pub fn tail(&self) -> usize {
        self.tail
    }

    #[inline]
    pub fn successor(&self) -> usize {
        self.successor
    }
}

impl std::fmt::Display for ChainNextRewire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChainNextRewire(tail: {}, successor: {})",
            self.tail, self.successor
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSetDelta {
    rewires: Vec<ChainNextRewire>,
    touched_nodes: Vec<usize>,
    touched_marks: Vec<u32>,
    tail_next_overrides: Vec<usize>,
    head_prev_overrides: Vec<usize>,
    changed_tails: Vec<usize>,
    affected_chains: Vec<usize>,
    tail_next_marks: Vec<u32>,
    head_prev_marks: Vec<u32>,
    changed_tail_marks: Vec<u32>,
    chain_marks: Vec<u32>,
    chain_epoch: u32,
    override_epoch: u32,
    touched_epoch: u32,
}

impl Default for ChainSetDelta {
    #[inline]
    fn default() -> Self {
        Self {
            rewires: Vec::new(),
            touched_nodes: Vec::new(),
            touched_marks: Vec::new(),
            touched_epoch: 1,
            tail_next_overrides: Vec::new(),
            tail_next_marks: Vec::new(),
            head_prev_overrides: Vec::new(),
            head_prev_marks: Vec::new(),
            changed_tails: Vec::new(),
            changed_tail_marks: Vec::new(),
            affected_chains: Vec::new(),
            chain_marks: Vec::new(),
            chain_epoch: 1,
            override_epoch: 1,
        }
    }
}

impl ChainSetDelta {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            rewires: Vec::with_capacity(capacity),

            touched_nodes: Vec::with_capacity(capacity.saturating_mul(2)),
            touched_marks: Vec::new(),
            touched_epoch: 1,

            tail_next_overrides: Vec::new(),
            tail_next_marks: Vec::new(),

            head_prev_overrides: Vec::new(),
            head_prev_marks: Vec::new(),

            changed_tails: Vec::with_capacity(capacity),
            changed_tail_marks: Vec::new(),

            affected_chains: Vec::with_capacity(capacity),
            chain_marks: Vec::new(),
            chain_epoch: 1,

            override_epoch: 1,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rewires.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rewires.is_empty()
    }

    #[inline]
    pub fn rewires(&self) -> &[ChainNextRewire] {
        &self.rewires
    }

    #[inline]
    pub fn touched_nodes(&self) -> &[usize] {
        &self.touched_nodes
    }

    #[inline]
    pub fn changed_tails(&self) -> &[usize] {
        &self.changed_tails
    }

    #[inline]
    pub fn affected_chains(&self) -> &[usize] {
        &self.affected_chains
    }

    #[inline]
    pub fn touch_node(&mut self, node: usize) {
        if node >= self.touched_marks.len() {
            self.touched_marks.resize(node + 1, 0);
        }
        if self.touched_marks[node] != self.touched_epoch {
            self.touched_marks[node] = self.touched_epoch;
            self.touched_nodes.push(node);
        }
    }

    #[inline]
    pub fn is_node_touched(&self, node: usize) -> bool {
        node < self.touched_marks.len() && self.touched_marks[node] == self.touched_epoch
    }

    #[inline]
    pub fn mark_chain(&mut self, chain: usize) {
        if chain >= self.chain_marks.len() {
            self.chain_marks.resize(chain + 1, 0);
        }
        if self.chain_marks[chain] != self.chain_epoch {
            self.chain_marks[chain] = self.chain_epoch;
            self.affected_chains.push(chain);
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.rewires.clear();

        self.touched_nodes.clear();
        self.touched_epoch = self.touched_epoch.wrapping_add(1);
        if self.touched_epoch == 0 {
            self.touched_marks.fill(0);
            self.touched_epoch = 1;
        }

        self.changed_tails.clear();
        self.override_epoch = self.override_epoch.wrapping_add(1);
        if self.override_epoch == 0 {
            self.tail_next_marks.fill(0);
            self.head_prev_marks.fill(0);
            self.changed_tail_marks.fill(0);
            self.override_epoch = 1;
        }

        self.affected_chains.clear();
        self.chain_epoch = self.chain_epoch.wrapping_add(1);
        if self.chain_epoch == 0 {
            self.chain_marks.fill(0);
            self.chain_epoch = 1;
        }
    }

    #[inline]
    pub fn is_tail_overridden(&self, tail: usize) -> bool {
        tail < self.changed_tail_marks.len() && self.changed_tail_marks[tail] == self.override_epoch
    }

    #[inline]
    pub fn is_head_overridden(&self, head: usize) -> bool {
        head < self.head_prev_marks.len() && self.head_prev_marks[head] == self.override_epoch
    }

    #[inline]
    pub fn next_override_for_tail(&self, tail: usize) -> Option<usize> {
        if tail < self.tail_next_marks.len() && self.tail_next_marks[tail] == self.override_epoch {
            Some(self.tail_next_overrides[tail])
        } else {
            None
        }
    }

    #[inline]
    pub fn prev_override_for_head(&self, head: usize) -> Option<usize> {
        if self.is_head_overridden(head) {
            Some(self.head_prev_overrides[head])
        } else {
            None
        }
    }

    #[inline(always)]
    fn ensure_tail_slot(&mut self, tail: usize) {
        if tail >= self.tail_next_marks.len() {
            let need = tail + 1;
            self.tail_next_marks.resize(need, 0);
            self.tail_next_overrides.resize(need, 0);
        }
        if tail >= self.changed_tail_marks.len() {
            self.changed_tail_marks.resize(tail + 1, 0);
        }
    }

    #[inline(always)]
    fn ensure_head_slot(&mut self, head: usize) {
        if head >= self.head_prev_marks.len() {
            let need = head + 1;
            self.head_prev_marks.resize(need, 0);
            self.head_prev_overrides.resize(need, 0);
        }
    }

    #[inline]
    pub fn set_next(&mut self, tail: usize, successor: usize) {
        let prior_successor = self.next_override_for_tail(tail);
        self.ensure_tail_slot(tail);
        self.tail_next_overrides[tail] = successor;
        self.tail_next_marks[tail] = self.override_epoch;

        if self.changed_tail_marks[tail] != self.override_epoch {
            self.changed_tail_marks[tail] = self.override_epoch;
            self.changed_tails.push(tail);
        }

        self.ensure_head_slot(successor);
        self.head_prev_overrides[successor] = tail;
        self.head_prev_marks[successor] = self.override_epoch;

        if let Some(old_succ) = prior_successor
            && old_succ != successor
        {
            self.ensure_head_slot(old_succ);
            self.head_prev_overrides[old_succ] = old_succ;
            self.head_prev_marks[old_succ] = self.override_epoch;

            self.ensure_tail_slot(old_succ);
            self.tail_next_overrides[old_succ] = old_succ;
            self.tail_next_marks[old_succ] = self.override_epoch;
        }
    }

    #[inline]
    pub fn push_rewire(&mut self, r: ChainNextRewire) {
        self.set_next(r.tail(), r.successor());
        self.touch_node(r.tail());
        self.touch_node(r.successor());
        self.rewires.push(r);
    }
}

impl std::fmt::Display for ChainSetDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in &self.rewires {
            writeln!(f, "  {}", r)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ChainNextRewire, ChainSetDelta};

    #[test]
    fn test_default_state_is_empty_and_unset() {
        let d = ChainSetDelta::new();

        // Introspection
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert!(d.rewires().is_empty());
        assert!(d.touched_nodes().is_empty());
        assert!(d.changed_tails().is_empty());
        assert!(d.affected_chains().is_empty());

        // No overrides
        assert!(!d.is_tail_overridden(0));
        assert!(!d.is_head_overridden(0));
        assert_eq!(d.next_override_for_tail(0), None);
        assert_eq!(d.prev_override_for_head(0), None);

        // No touches
        assert!(!d.is_node_touched(0));
        assert!(!d.is_node_touched(10));
    }

    #[test]
    fn test_with_capacity_allocates_append_only_vectors() {
        let d = ChainSetDelta::with_capacity(8);

        // Still empty
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert!(d.rewires().is_empty());

        // No marks/overrides pre-set
        assert_eq!(d.next_override_for_tail(0), None);
        assert_eq!(d.prev_override_for_head(0), None);
        assert!(!d.is_tail_overridden(0));
        assert!(!d.is_head_overridden(0));
    }

    #[test]
    fn test_set_next_basic_overrides_both_directions() {
        let mut d = ChainSetDelta::new();

        // Set next[2] = 5
        d.set_next(2, 5);

        // Forward/backward overlay queries
        assert!(d.is_tail_overridden(2));
        assert!(d.is_head_overridden(5));
        assert_eq!(d.next_override_for_tail(2), Some(5));
        assert_eq!(d.prev_override_for_head(5), Some(2));

        // Non-touched unless we explicitly touch or push_rewire
        assert!(!d.is_node_touched(2));
        assert!(!d.is_node_touched(5));

        // changed_tails records 2 exactly once
        assert_eq!(d.changed_tails(), &[2]);
    }

    #[test]
    fn test_set_next_replaces_old_successor_and_detaches_old() {
        let mut d = ChainSetDelta::new();

        // First: 7 -> 3
        d.set_next(7, 3);
        assert_eq!(d.next_override_for_tail(7), Some(3));
        assert_eq!(d.prev_override_for_head(3), Some(7));
        assert!(d.is_head_overridden(3));
        assert!(d.is_tail_overridden(7));

        // Replace: 7 -> 9 (same epoch)
        d.set_next(7, 9);

        // New mapping
        assert_eq!(d.next_override_for_tail(7), Some(9));
        assert_eq!(d.prev_override_for_head(9), Some(7));
        assert!(d.is_head_overridden(9));

        // Old successor 3 must be detached in overlay (self loops)
        assert!(d.is_head_overridden(3));
        assert_eq!(d.prev_override_for_head(3), Some(3));
        assert_eq!(d.next_override_for_tail(3), Some(3));

        // changed_tails still has 7 exactly once
        assert_eq!(d.changed_tails(), &[7]);
    }

    #[test]
    fn test_set_next_idempotent_no_duplicate_changed_tails() {
        let mut d = ChainSetDelta::new();

        d.set_next(4, 4); // self-loop allowed
        d.set_next(4, 4);
        d.set_next(4, 4);

        assert!(d.is_tail_overridden(4));
        assert!(d.is_head_overridden(4));
        assert_eq!(d.changed_tails(), &[4]);
    }

    #[test]
    fn test_push_rewire_records_and_touches_nodes() {
        let mut d = ChainSetDelta::new();

        d.push_rewire(ChainNextRewire::new(1, 2));

        // Rewires recorded
        assert_eq!(d.len(), 1);
        assert_eq!(d.rewires(), &[ChainNextRewire::new(1, 2)]);

        // Touches applied for both tail and successor
        assert!(d.is_node_touched(1));
        assert!(d.is_node_touched(2));

        // Overlay set as well
        assert_eq!(d.next_override_for_tail(1), Some(2));
        assert_eq!(d.prev_override_for_head(2), Some(1));

        // changed_tails includes 1
        assert_eq!(d.changed_tails(), &[1]);
    }

    #[test]
    fn test_touch_node_is_deduplicated_per_epoch() {
        let mut d = ChainSetDelta::new();

        // Touch same node many times in same epoch
        d.touch_node(10);
        d.touch_node(10);
        d.touch_node(10);

        assert!(d.is_node_touched(10));
        assert_eq!(d.touched_nodes(), &[10]);

        // Touch another node
        d.touch_node(3);
        assert!(d.is_node_touched(3));
        assert_eq!(d.touched_nodes(), &[10, 3]);

        // Clear -> epoch advances, dedup resets
        d.clear();

        // No nodes are considered touched in the new epoch
        assert!(!d.is_node_touched(10));
        assert!(!d.is_node_touched(3));
        assert!(d.touched_nodes().is_empty());

        // Touch again -> collected anew
        d.touch_node(10);
        assert_eq!(d.touched_nodes(), &[10]);
    }

    #[test]
    fn test_mark_chain_dedup_and_epoch() {
        let mut d = ChainSetDelta::new();

        // Mark same chain multiple times in same epoch
        d.mark_chain(2);
        d.mark_chain(2);
        assert_eq!(d.affected_chains(), &[2]);

        // Mark another chain
        d.mark_chain(0);
        assert_eq!(d.affected_chains(), &[2, 0]);

        // Clear -> epoch bumps, dedup resets
        d.clear();
        assert!(d.affected_chains().is_empty());

        // Mark again in new epoch; should reappear
        d.mark_chain(2);
        assert_eq!(d.affected_chains(), &[2]);
    }

    #[test]
    fn test_clear_resets_overlay_visibility_and_lists() {
        let mut d = ChainSetDelta::new();

        d.set_next(5, 6);
        d.touch_node(1);
        d.mark_chain(3);

        // Sanity before clear
        assert!(d.is_tail_overridden(5));
        assert!(d.is_head_overridden(6));
        assert!(!d.changed_tails().is_empty());
        assert!(!d.touched_nodes().is_empty());
        assert!(!d.affected_chains().is_empty());

        d.clear();

        // After clear, nothing should be visible in current epoch
        assert!(!d.is_tail_overridden(5));
        assert!(!d.is_head_overridden(6));
        assert_eq!(d.changed_tails(), &[]);
        assert_eq!(d.touched_nodes(), &[]);
        assert_eq!(d.affected_chains(), &[]);

        // Override queries should return None in new epoch
        assert_eq!(d.next_override_for_tail(5), None);
        assert_eq!(d.prev_override_for_head(6), None);
    }

    #[test]
    fn test_multiple_tails_and_last_writer_wins_for_head() {
        let mut d = ChainSetDelta::new();

        // Two different tails pointing to the same head
        d.set_next(1, 9);
        assert_eq!(d.prev_override_for_head(9), Some(1));
        assert!(d.is_head_overridden(9));
        assert!(d.is_tail_overridden(1));

        d.set_next(2, 9);
        // Head 9 now points back to tail 2 (last-writer-wins behavior)
        assert_eq!(d.prev_override_for_head(9), Some(2));
        assert!(d.is_tail_overridden(2));

        // changed_tails contains both 1 and 2 (order preserved by first-seen)
        let tails = d.changed_tails().to_vec();
        assert!(tails.contains(&1));
        assert!(tails.contains(&2));
        // 1 must come before 2 as 1 was first seen
        let pos1 = tails.iter().position(|&x| x == 1).unwrap();
        let pos2 = tails.iter().position(|&x| x == 2).unwrap();
        assert!(pos1 < pos2);
    }

    #[test]
    fn test_display_lists_rewires_one_per_line() {
        let mut d = ChainSetDelta::new();
        d.push_rewire(ChainNextRewire::new(0, 2));
        d.push_rewire(ChainNextRewire::new(3, 4));

        let s = format!("{}", d);
        let lines: Vec<_> = s.lines().collect();

        // Expect two lines, each prefixed by two spaces and formatted by ChainNextRewire::Display
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "  ChainNextRewire(tail: 0, successor: 2)");
        assert_eq!(lines[1], "  ChainNextRewire(tail: 3, successor: 4)");
    }
}
