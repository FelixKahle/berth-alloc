// Copyright (c) 2026 Felix Kahle.
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

use crate::state::chain_set::index::{ChainIndex, NodeIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainNextRewire {
    tail: NodeIndex,
    successor: NodeIndex,
}

impl ChainNextRewire {
    #[inline]
    pub fn new(tail: NodeIndex, successor: NodeIndex) -> Self {
        Self { tail, successor }
    }

    #[inline]
    pub fn tail(&self) -> NodeIndex {
        self.tail
    }

    #[inline]
    pub fn successor(&self) -> NodeIndex {
        self.successor
    }
}

impl std::fmt::Display for ChainNextRewire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChainNextRewire(tail: {}, successor: {})",
            self.tail.get(),
            self.successor.get()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSetDelta {
    rewires: Vec<ChainNextRewire>,
    touched_nodes: Vec<NodeIndex>,
    tail_next_overrides: Vec<NodeIndex>,
    head_prev_overrides: Vec<NodeIndex>,
    changed_tails: Vec<NodeIndex>,
    affected_chains: Vec<ChainIndex>,
    touched_marks: Vec<u32>,
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
    pub fn touched_nodes(&self) -> &[NodeIndex] {
        &self.touched_nodes
    }

    #[inline]
    pub fn changed_tails(&self) -> &[NodeIndex] {
        &self.changed_tails
    }

    #[inline]
    pub fn affected_chains(&self) -> &[ChainIndex] {
        &self.affected_chains
    }

    #[inline]
    pub fn touch_node(&mut self, node: NodeIndex) {
        let node = node.get();

        if node >= self.touched_marks.len() {
            self.touched_marks.resize(node + 1, 0);
        }
        if self.touched_marks[node] != self.touched_epoch {
            self.touched_marks[node] = self.touched_epoch;
            self.touched_nodes.push(node.into());
        }
    }

    #[inline]
    pub fn is_node_touched(&self, node: NodeIndex) -> bool {
        let node = node.get();
        node < self.touched_marks.len() && self.touched_marks[node] == self.touched_epoch
    }

    #[inline]
    pub fn mark_chain(&mut self, chain: ChainIndex) {
        let chain = chain.get();

        if chain >= self.chain_marks.len() {
            self.chain_marks.resize(chain + 1, 0);
        }
        if self.chain_marks[chain] != self.chain_epoch {
            self.chain_marks[chain] = self.chain_epoch;
            self.affected_chains.push(chain.into());
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
    pub fn is_tail_overridden(&self, tail: NodeIndex) -> bool {
        let tail = tail.get();

        tail < self.changed_tail_marks.len() && self.changed_tail_marks[tail] == self.override_epoch
    }

    #[inline]
    pub fn is_head_overridden(&self, head: NodeIndex) -> bool {
        let head = head.get();

        head < self.head_prev_marks.len() && self.head_prev_marks[head] == self.override_epoch
    }

    #[inline]
    pub fn next_override_for_tail(&self, tail: NodeIndex) -> Option<NodeIndex> {
        let tail = tail.get();

        if tail < self.tail_next_marks.len() && self.tail_next_marks[tail] == self.override_epoch {
            Some(self.tail_next_overrides[tail])
        } else {
            None
        }
    }

    #[inline]
    pub fn prev_override_for_head(&self, head: NodeIndex) -> Option<NodeIndex> {
        if self.is_head_overridden(head) {
            let head = head.get();
            Some(self.head_prev_overrides[head])
        } else {
            None
        }
    }

    #[inline(always)]
    fn ensure_tail_slot(&mut self, tail: NodeIndex) {
        let tail = tail.get();

        if tail >= self.tail_next_marks.len() {
            let need = tail + 1;
            self.tail_next_marks.resize(need, 0);
            self.tail_next_overrides.resize(need, NodeIndex(0));
        }
        if tail >= self.changed_tail_marks.len() {
            self.changed_tail_marks.resize(tail + 1, 0);
        }
    }

    #[inline(always)]
    fn ensure_head_slot(&mut self, head: NodeIndex) {
        let head = head.get();

        if head >= self.head_prev_marks.len() {
            let need = head + 1;
            self.head_prev_marks.resize(need, 0);
            self.head_prev_overrides.resize(need, NodeIndex(0));
        }
    }

    #[inline]
    pub fn set_next(&mut self, tail: NodeIndex, successor: NodeIndex) {
        let prior_successor = self.next_override_for_tail(tail);

        self.ensure_tail_slot(tail);
        self.ensure_head_slot(successor);

        let tail_index = tail.get();
        let successor_index = successor.get();
        let epoch = self.override_epoch;

        self.tail_next_overrides[tail_index] = successor;
        self.tail_next_marks[tail_index] = epoch;

        if self.changed_tail_marks[tail_index] != epoch {
            self.changed_tail_marks[tail_index] = epoch;
            self.changed_tails.push(tail);
        }

        self.head_prev_overrides[successor_index] = tail;
        self.head_prev_marks[successor_index] = epoch;

        if let Some(os) = prior_successor
            && os != successor
        {
            self.ensure_head_slot(os);
            self.ensure_tail_slot(os);

            let os_idx = os.get();

            // Self-links denote detachment
            self.head_prev_overrides[os_idx] = os;
            self.head_prev_marks[os_idx] = epoch;

            self.tail_next_overrides[os_idx] = os;
            self.tail_next_marks[os_idx] = epoch;
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
    use crate::state::chain_set::index::{ChainIndex, NodeIndex};

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
        assert!(!d.is_tail_overridden(NodeIndex(0)));
        assert!(!d.is_head_overridden(NodeIndex(0)));
        assert_eq!(d.next_override_for_tail(NodeIndex(0)), None);
        assert_eq!(d.prev_override_for_head(NodeIndex(0)), None);

        // No touches
        assert!(!d.is_node_touched(NodeIndex(0)));
        assert!(!d.is_node_touched(NodeIndex(10)));
    }

    #[test]
    fn test_with_capacity_allocates_append_only_vectors() {
        let d = ChainSetDelta::with_capacity(8);

        // Still empty
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert!(d.rewires().is_empty());

        // No marks/overrides pre-set
        assert_eq!(d.next_override_for_tail(NodeIndex(0)), None);
        assert_eq!(d.prev_override_for_head(NodeIndex(0)), None);
        assert!(!d.is_tail_overridden(NodeIndex(0)));
        assert!(!d.is_head_overridden(NodeIndex(0)));
    }

    #[test]
    fn test_set_next_basic_overrides_both_directions() {
        let mut d = ChainSetDelta::new();

        // Set next[2] = 5
        d.set_next(NodeIndex(2), NodeIndex(5));

        // Forward/backward overlay queries
        assert!(d.is_tail_overridden(NodeIndex(2)));
        assert!(d.is_head_overridden(NodeIndex(5)));
        assert_eq!(d.next_override_for_tail(NodeIndex(2)), Some(NodeIndex(5)));
        assert_eq!(d.prev_override_for_head(NodeIndex(5)), Some(NodeIndex(2)));

        // Non-touched unless we explicitly touch or push_rewire
        assert!(!d.is_node_touched(NodeIndex(2)));
        assert!(!d.is_node_touched(NodeIndex(5)));

        // changed_tails records 2 exactly once
        assert_eq!(d.changed_tails(), &[NodeIndex(2)]);
    }

    #[test]
    fn test_set_next_replaces_old_successor_and_detaches_old() {
        let mut d = ChainSetDelta::new();

        // First: 7 -> 3
        d.set_next(NodeIndex(7), NodeIndex(3));
        assert_eq!(d.next_override_for_tail(NodeIndex(7)), Some(NodeIndex(3)));
        assert_eq!(d.prev_override_for_head(NodeIndex(3)), Some(NodeIndex(7)));
        assert!(d.is_head_overridden(NodeIndex(3)));
        assert!(d.is_tail_overridden(NodeIndex(7)));

        // Replace: 7 -> 9 (same epoch)
        d.set_next(NodeIndex(7), NodeIndex(9));

        // New mapping
        assert_eq!(d.next_override_for_tail(NodeIndex(7)), Some(NodeIndex(9)));
        assert_eq!(d.prev_override_for_head(NodeIndex(9)), Some(NodeIndex(7)));
        assert!(d.is_head_overridden(NodeIndex(9)));

        // Old successor 3 must be detached in overlay (self loops)
        assert!(d.is_head_overridden(NodeIndex(3)));
        assert_eq!(d.prev_override_for_head(NodeIndex(3)), Some(NodeIndex(3)));
        assert_eq!(d.next_override_for_tail(NodeIndex(3)), Some(NodeIndex(3)));

        // changed_tails still has 7 exactly once
        assert_eq!(d.changed_tails(), &[NodeIndex(7)]);
    }

    #[test]
    fn test_set_next_idempotent_no_duplicate_changed_tails() {
        let mut d = ChainSetDelta::new();

        d.set_next(NodeIndex(4), NodeIndex(4)); // self-loop allowed
        d.set_next(NodeIndex(4), NodeIndex(4));
        d.set_next(NodeIndex(4), NodeIndex(4));

        assert!(d.is_tail_overridden(NodeIndex(4)));
        assert!(d.is_head_overridden(NodeIndex(4)));
        assert_eq!(d.changed_tails(), &[NodeIndex(4)]);
    }

    #[test]
    fn test_push_rewire_records_and_touches_nodes() {
        let mut d = ChainSetDelta::new();

        d.push_rewire(ChainNextRewire::new(NodeIndex(1), NodeIndex(2)));

        // Rewires recorded
        assert_eq!(d.len(), 1);
        assert_eq!(
            d.rewires(),
            &[ChainNextRewire::new(NodeIndex(1), NodeIndex(2))]
        );

        // Touches applied for both tail and successor
        assert!(d.is_node_touched(NodeIndex(1)));
        assert!(d.is_node_touched(NodeIndex(2)));

        // Overlay set as well
        assert_eq!(d.next_override_for_tail(NodeIndex(1)), Some(NodeIndex(2)));
        assert_eq!(d.prev_override_for_head(NodeIndex(2)), Some(NodeIndex(1)));

        // changed_tails includes 1
        assert_eq!(d.changed_tails(), &[NodeIndex(1)]);
    }

    #[test]
    fn test_touch_node_is_deduplicated_per_epoch() {
        let mut d = ChainSetDelta::new();

        // Touch same node many times in same epoch
        d.touch_node(NodeIndex(10));
        d.touch_node(NodeIndex(10));
        d.touch_node(NodeIndex(10));

        assert!(d.is_node_touched(NodeIndex(10)));
        assert_eq!(d.touched_nodes(), &[NodeIndex(10)]);

        // Touch another node
        d.touch_node(NodeIndex(3));
        assert!(d.is_node_touched(NodeIndex(3)));
        assert_eq!(d.touched_nodes(), &[NodeIndex(10), NodeIndex(3)]);

        // Clear -> epoch advances, dedup resets
        d.clear();

        // No nodes are considered touched in the new epoch
        assert!(!d.is_node_touched(NodeIndex(10)));
        assert!(!d.is_node_touched(NodeIndex(3)));
        assert!(d.touched_nodes().is_empty());

        // Touch again -> collected anew
        d.touch_node(NodeIndex(10));
        assert_eq!(d.touched_nodes(), &[NodeIndex(10)]);
    }

    #[test]
    fn test_mark_chain_dedup_and_epoch() {
        let mut d = ChainSetDelta::new();

        // Mark same chain multiple times in same epoch
        d.mark_chain(ChainIndex(2));
        d.mark_chain(ChainIndex(2));
        assert_eq!(d.affected_chains(), &[ChainIndex(2)]);

        // Mark another chain
        d.mark_chain(ChainIndex(0));
        assert_eq!(d.affected_chains(), &[ChainIndex(2), ChainIndex(0)]);

        // Clear -> epoch bumps, dedup resets
        d.clear();
        assert!(d.affected_chains().is_empty());

        // Mark again in new epoch; should reappear
        d.mark_chain(ChainIndex(2));
        assert_eq!(d.affected_chains(), &[ChainIndex(2)]);
    }

    #[test]
    fn test_clear_resets_overlay_visibility_and_lists() {
        let mut d = ChainSetDelta::new();

        d.set_next(NodeIndex(5), NodeIndex(6));
        d.touch_node(NodeIndex(1));
        d.mark_chain(ChainIndex(3));

        // Sanity before clear
        assert!(d.is_tail_overridden(NodeIndex(5)));
        assert!(d.is_head_overridden(NodeIndex(6)));
        assert!(!d.changed_tails().is_empty());
        assert!(!d.touched_nodes().is_empty());
        assert!(!d.affected_chains().is_empty());

        d.clear();

        // After clear, nothing should be visible in current epoch
        assert!(!d.is_tail_overridden(NodeIndex(5)));
        assert!(!d.is_head_overridden(NodeIndex(6)));
        assert_eq!(d.changed_tails(), &[]);
        assert_eq!(d.touched_nodes(), &[]);
        assert_eq!(d.affected_chains(), &[]);

        // Override queries should return None in new epoch
        assert_eq!(d.next_override_for_tail(NodeIndex(5)), None);
        assert_eq!(d.prev_override_for_head(NodeIndex(6)), None);
    }

    #[test]
    fn test_multiple_tails_and_last_writer_wins_for_head() {
        let mut d = ChainSetDelta::new();

        // Two different tails pointing to the same head
        d.set_next(NodeIndex(1), NodeIndex(9));
        assert_eq!(d.prev_override_for_head(NodeIndex(9)), Some(NodeIndex(1)));
        assert!(d.is_head_overridden(NodeIndex(9)));
        assert!(d.is_tail_overridden(NodeIndex(1)));

        d.set_next(NodeIndex(2), NodeIndex(9));
        // Head 9 now points back to tail 2 (last-writer-wins behavior)
        assert_eq!(d.prev_override_for_head(NodeIndex(9)), Some(NodeIndex(2)));
        assert!(d.is_tail_overridden(NodeIndex(2)));

        // changed_tails contains both 1 and 2 (order preserved by first-seen)
        let tails = d.changed_tails().to_vec();
        assert!(tails.contains(&NodeIndex(1)));
        assert!(tails.contains(&NodeIndex(2)));
        // 1 must come before 2 as 1 was first seen
        let pos1 = tails.iter().position(|&x| x == NodeIndex(1)).unwrap();
        let pos2 = tails.iter().position(|&x| x == NodeIndex(2)).unwrap();
        assert!(pos1 < pos2);
    }

    #[test]
    fn test_display_lists_rewires_one_per_line() {
        let mut d = ChainSetDelta::new();
        d.push_rewire(ChainNextRewire::new(NodeIndex(0), NodeIndex(2)));
        d.push_rewire(ChainNextRewire::new(NodeIndex(3), NodeIndex(4)));

        let s = format!("{}", d);
        let lines: Vec<_> = s.lines().collect();

        // Expect two lines, each prefixed by two spaces and formatted by ChainNextRewire::Display
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "  ChainNextRewire(tail: 0, successor: 2)");
        assert_eq!(lines[1], "  ChainNextRewire(tail: 3, successor: 4)");
    }
}
