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

use crate::state::chain_set::index::{ChainIndex, NodeIndex};

pub trait ChainSetView {
    type NodeIter<'a>: Iterator<Item = NodeIndex> + 'a
    where
        Self: 'a;

    fn num_nodes(&self) -> usize;
    fn num_chains(&self) -> usize;

    fn start_of_chain(&self, chain: ChainIndex) -> NodeIndex;
    fn real_start_of_chain(&self, chain: ChainIndex) -> Option<NodeIndex> {
        let start = self.start_of_chain(chain);
        self.next_node(start)
    }
    fn end_of_chain(&self, chain: ChainIndex) -> NodeIndex;
    fn real_end_of_chain(&self, chain: ChainIndex) -> Option<NodeIndex> {
        let end = self.end_of_chain(chain);
        self.prev_node(end)
    }
    fn chain(&self, chain: ChainIndex) -> ChainRef<'_, Self>
    where
        Self: Sized,
    {
        ChainRef::new(self, chain)
    }

    #[inline]
    fn first_real_node(&self, start_node: NodeIndex) -> Option<NodeIndex> {
        let mut current = start_node;
        let mut steps_left = self.num_nodes() + 2 * self.num_chains();
        while steps_left > 0 {
            let next = self.next_node(current)?;
            if !self.is_sentinel_node(next) {
                return Some(next);
            }
            if next == current {
                return None;
            }
            current = next;
            steps_left -= 1;
        }
        None
    }

    fn next_node(&self, node: NodeIndex) -> Option<NodeIndex>;

    #[inline]
    fn next_real_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        let mut current = node;
        let mut steps_left = self.num_nodes() + 2 * self.num_chains();
        while steps_left > 0 {
            let next = self.next_node(current)?;
            if !self.is_sentinel_node(next) {
                return Some(next);
            }
            if next == current {
                return None;
            }
            current = next;
            steps_left -= 1;
        }
        None
    }

    fn prev_node(&self, node: NodeIndex) -> Option<NodeIndex>;

    #[inline]
    fn prev_real_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        let mut current = node;
        let mut steps_left = self.num_nodes() + 2 * self.num_chains();
        while steps_left > 0 {
            let prev = self.prev_node(current)?;
            if !self.is_sentinel_node(prev) {
                return Some(prev);
            }
            if prev == current {
                return None;
            }
            current = prev;
            steps_left -= 1;
        }
        None
    }

    fn is_sentinel_node(&self, node: NodeIndex) -> bool;
    fn is_head_node(&self, node: NodeIndex) -> bool;
    fn is_tail_node(&self, node: NodeIndex) -> bool;

    fn is_node_unperformed(&self, node: NodeIndex) -> bool;

    fn is_chain_empty(&self, chain: ChainIndex) -> bool;

    fn chain_of_node(&self, node: NodeIndex) -> Option<ChainIndex>;
    fn position_in_chain(&self, node: NodeIndex) -> Option<usize>;

    fn iter_chain(&self, chain: ChainIndex) -> Self::NodeIter<'_>;

    #[inline]
    fn resolve_slice_bounds(
        &self,
        chain: ChainIndex,
        start_node: NodeIndex,
        end_node_exclusive: Option<NodeIndex>,
    ) -> (Option<NodeIndex>, NodeIndex) {
        let end_exclusive = end_node_exclusive.unwrap_or_else(|| self.end_of_chain(chain));
        if start_node == end_exclusive {
            return (None, end_exclusive);
        }
        if self.is_sentinel_node(start_node) {
            let first = self
                .next_real_node(start_node)
                .filter(|&n| n != end_exclusive);
            (first, end_exclusive)
        } else {
            (Some(start_node), end_exclusive)
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainRef<'chain, C: ChainSetView> {
    chain_view: &'chain C,
    chain: ChainIndex,
}

impl<'chain, C: ChainSetView> Clone for ChainRef<'chain, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'chain, C: ChainSetView> Copy for ChainRef<'chain, C> {}

impl<'chain, C: ChainSetView> ChainRef<'chain, C> {
    #[inline]
    pub fn new(chain_view: &'chain C, chain: ChainIndex) -> Self {
        assert!(chain.get() < chain_view.num_chains());
        Self { chain_view, chain }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chain_view.is_chain_empty(self.chain)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.chain_view.iter_chain(self.chain)
    }

    #[inline]
    pub fn start(&self) -> NodeIndex {
        self.chain_view.start_of_chain(self.chain)
    }

    #[inline]
    pub fn real_start(&self) -> Option<NodeIndex> {
        self.chain_view.real_start_of_chain(self.chain)
    }

    #[inline]
    pub fn end(&self) -> NodeIndex {
        self.chain_view.end_of_chain(self.chain)
    }

    #[inline]
    pub fn real_end(&self) -> Option<NodeIndex> {
        self.chain_view.real_end_of_chain(self.chain)
    }

    #[inline]
    pub fn chain_index(&self) -> ChainIndex {
        self.chain
    }

    #[inline]
    pub fn next(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.chain_view.next_node(node)
    }

    #[inline]
    pub fn next_real(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.chain_view.next_real_node(node)
    }

    #[inline]
    pub fn prev(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.chain_view.prev_node(node)
    }

    #[inline]
    pub fn prev_real(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.chain_view.prev_real_node(node)
    }

    #[inline]
    pub fn first_real_node(&self, start_node: NodeIndex) -> Option<NodeIndex> {
        self.chain_view.first_real_node(start_node)
    }

    #[inline]
    pub fn is_sentinel_node(&self, node: NodeIndex) -> bool {
        self.chain_view.is_sentinel_node(node)
    }

    #[inline]
    pub fn is_head_node(&self, node: NodeIndex) -> bool {
        self.chain_view.is_head_node(node)
    }

    #[inline]
    pub fn is_tail_node(&self, node: NodeIndex) -> bool {
        self.chain_view.is_tail_node(node)
    }

    #[inline]
    pub fn resolve_slice(
        &self,
        start_node: NodeIndex,
        end_node_exclusive: Option<NodeIndex>,
    ) -> (Option<NodeIndex>, NodeIndex) {
        self.chain_view
            .resolve_slice_bounds(self.chain, start_node, end_node_exclusive)
    }

    #[inline]
    pub fn chain_view(&self) -> &'chain C {
        self.chain_view
    }
}

impl<'chain, C: ChainSetView> std::fmt::Display for ChainRef<'chain, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut it = self
            .iter()
            .filter(|&n| !self.chain_view.is_sentinel_node(n));

        if let Some(first) = it.next() {
            write!(f, "{}", first.get())?;
            for node in it {
                write!(f, "->{}", node.get())?;
            }
        }
        Ok(())
    }
}

pub trait ChainSetViewDyn {
    fn num_nodes(&self) -> usize;
    fn num_chains(&self) -> usize;
    fn start_of_chain(&self, chain: ChainIndex) -> NodeIndex;
    fn end_of_chain(&self, chain: ChainIndex) -> NodeIndex;
    fn next_node(&self, node: NodeIndex) -> Option<NodeIndex>;
    fn prev_node(&self, node: NodeIndex) -> Option<NodeIndex>;
    fn is_sentinel_node(&self, node: NodeIndex) -> bool;
    fn is_head_node(&self, node: NodeIndex) -> bool;
    fn is_tail_node(&self, node: NodeIndex) -> bool;
    fn is_node_unperformed(&self, node: NodeIndex) -> bool;
    fn is_chain_empty(&self, chain: ChainIndex) -> bool;
    fn chain_of_node(&self, node: NodeIndex) -> Option<ChainIndex>;
    fn position_in_chain(&self, node: NodeIndex) -> Option<usize>;
}

pub struct ChainSetViewDynAdapter<'a, C: ChainSetView>(pub &'a C);

impl<'a, C: ChainSetView> ChainSetViewDyn for ChainSetViewDynAdapter<'a, C> {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.0.num_nodes()
    }
    #[inline]
    fn num_chains(&self) -> usize {
        self.0.num_chains()
    }
    #[inline]
    fn start_of_chain(&self, chain: ChainIndex) -> NodeIndex {
        self.0.start_of_chain(chain)
    }
    #[inline]
    fn end_of_chain(&self, chain: ChainIndex) -> NodeIndex {
        self.0.end_of_chain(chain)
    }
    #[inline]
    fn next_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.0.next_node(node)
    }
    #[inline]
    fn prev_node(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.0.prev_node(node)
    }
    #[inline]
    fn is_sentinel_node(&self, node: NodeIndex) -> bool {
        self.0.is_sentinel_node(node)
    }
    #[inline]
    fn is_head_node(&self, node: NodeIndex) -> bool {
        self.0.is_head_node(node)
    }
    #[inline]
    fn is_tail_node(&self, node: NodeIndex) -> bool {
        self.0.is_tail_node(node)
    }
    #[inline]
    fn is_node_unperformed(&self, node: NodeIndex) -> bool {
        self.0.is_node_unperformed(node)
    }
    #[inline]
    fn is_chain_empty(&self, chain: ChainIndex) -> bool {
        self.0.is_chain_empty(chain)
    }
    #[inline]
    fn chain_of_node(&self, node: NodeIndex) -> Option<ChainIndex> {
        self.0.chain_of_node(node)
    }
    #[inline]
    fn position_in_chain(&self, node: NodeIndex) -> Option<usize> {
        self.0.position_in_chain(node)
    }
}

pub trait ChainViewDyn {
    fn chain_index(&self) -> ChainIndex;
    fn start(&self) -> NodeIndex;
    fn end(&self) -> NodeIndex;
    fn real_start(&self) -> Option<NodeIndex>;
    fn real_end(&self) -> Option<NodeIndex>;
    fn next_real(&self, n: NodeIndex) -> Option<NodeIndex>;
    fn prev_real(&self, n: NodeIndex) -> Option<NodeIndex>;
    fn first_real_node(&self, after: NodeIndex) -> Option<NodeIndex>;
    fn is_sentinel_node(&self, n: NodeIndex) -> bool;
    fn resolve_slice(
        &self,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
    ) -> (Option<NodeIndex>, NodeIndex);
}

pub struct ChainViewDynAdapter<'a, C: ChainSetView>(pub ChainRef<'a, C>);

impl<'a, C: ChainSetView> ChainViewDyn for ChainViewDynAdapter<'a, C> {
    #[inline]
    fn chain_index(&self) -> ChainIndex {
        self.0.chain_index()
    }
    #[inline]
    fn start(&self) -> NodeIndex {
        self.0.start()
    }
    #[inline]
    fn end(&self) -> NodeIndex {
        self.0.end()
    }
    #[inline]
    fn real_start(&self) -> Option<NodeIndex> {
        self.0.real_start()
    }
    #[inline]
    fn real_end(&self) -> Option<NodeIndex> {
        self.0.real_end()
    }
    #[inline]
    fn next_real(&self, n: NodeIndex) -> Option<NodeIndex> {
        self.0.next_real(n)
    }
    #[inline]
    fn prev_real(&self, n: NodeIndex) -> Option<NodeIndex> {
        self.0.prev_real(n)
    }
    #[inline]
    fn first_real_node(&self, after: NodeIndex) -> Option<NodeIndex> {
        self.0.first_real_node(after)
    }
    #[inline]
    fn is_sentinel_node(&self, n: NodeIndex) -> bool {
        self.0.is_sentinel_node(n)
    }
    #[inline]
    fn resolve_slice(
        &self,
        start_inclusive: NodeIndex,
        end_exclusive: Option<NodeIndex>,
    ) -> (Option<NodeIndex>, NodeIndex) {
        self.0.resolve_slice(start_inclusive, end_exclusive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[derive(Default)]
    struct MockChainSet {
        chains: Vec<Vec<NodeIndex>>,
        unperformed: HashSet<NodeIndex>,
    }

    impl MockChainSet {
        fn new(chains: Vec<Vec<NodeIndex>>) -> Self {
            Self {
                chains,
                unperformed: HashSet::new(),
            }
        }

        #[allow(dead_code)]
        fn with_unperformed(mut self, nodes: impl IntoIterator<Item = NodeIndex>) -> Self {
            self.unperformed.extend(nodes);
            self
        }

        fn find_node(&self, node: NodeIndex) -> Option<(NodeIndex, NodeIndex)> {
            for (ci, c) in self.chains.iter().enumerate() {
                if let Some(pos) = c.iter().position(|&n| n == node) {
                    return Some((ci.into(), pos.into()));
                }
            }
            None
        }
    }

    impl ChainSetView for MockChainSet {
        fn num_nodes(&self) -> usize {
            self.chains.iter().map(|c| c.len()).sum()
        }

        fn num_chains(&self) -> usize {
            self.chains.len()
        }

        fn start_of_chain(&self, chain: ChainIndex) -> NodeIndex {
            // For this mock: first node if non-empty, otherwise 0 (unused in display tests)
            let chain = chain.get();
            self.chains[chain].first().copied().unwrap_or(NodeIndex(0))
        }

        fn end_of_chain(&self, chain: ChainIndex) -> NodeIndex {
            // For this mock: last node if non-empty, otherwise 0 (unused in display tests)
            let chain = chain.get();
            self.chains[chain].last().copied().unwrap_or(NodeIndex(0))
        }

        fn next_node(&self, node: NodeIndex) -> Option<NodeIndex> {
            let (ci, pos) = self.find_node(node)?;
            self.chains[ci.get()].get(pos.get() + 1).copied()
        }

        fn prev_node(&self, node: NodeIndex) -> Option<NodeIndex> {
            let (ci, pos) = self.find_node(node)?;
            if pos.get() > 0 {
                Some(self.chains[ci.get()][pos.get() - 1].into())
            } else {
                None
            }
        }

        fn is_sentinel_node(&self, _node: NodeIndex) -> bool {
            // This mock has no sentinels in iteration
            false
        }

        fn is_head_node(&self, node: NodeIndex) -> bool {
            if let Some((ci, pos)) = self.find_node(node) {
                pos.get() == 0 && !self.chains[ci.get()].is_empty()
            } else {
                false
            }
        }

        fn is_tail_node(&self, node: NodeIndex) -> bool {
            if let Some((ci, pos)) = self.find_node(node) {
                pos.get() + 1 == self.chains[ci.get()].len() && !self.chains[ci.get()].is_empty()
            } else {
                false
            }
        }

        fn is_node_unperformed(&self, node: NodeIndex) -> bool {
            self.unperformed.contains(&node)
        }

        fn is_chain_empty(&self, chain: ChainIndex) -> bool {
            let chain = chain.get();
            self.chains[chain].is_empty()
        }

        fn iter_chain(&self, chain: ChainIndex) -> Self::NodeIter<'_> {
            let chain = chain.get();
            self.chains[chain].iter().copied()
        }

        type NodeIter<'a>
            = std::iter::Copied<std::slice::Iter<'a, NodeIndex>>
        where
            Self: 'a;

        fn chain_of_node(&self, node: NodeIndex) -> Option<ChainIndex> {
            for (ci, chain) in self.chains.iter().enumerate() {
                if chain.contains(&node) {
                    return Some(ChainIndex(ci));
                }
            }
            None
        }

        fn position_in_chain(&self, node: NodeIndex) -> Option<usize> {
            for (_, chain) in self.chains.iter().enumerate() {
                if let Some(pos) = chain.iter().position(|&n| n == node) {
                    return Some(pos);
                }
            }
            None
        }
    }

    #[test]
    fn test_display_formats_non_empty_chain() {
        let mock = MockChainSet::new(vec![vec![NodeIndex(2), NodeIndex(3), NodeIndex(5)]]);
        let c0 = ChainRef::new(&mock, ChainIndex(0));

        let s = format!("{}", c0);
        assert_eq!(s, "2->3->5");

        // Sanity checks
        assert!(!c0.is_empty());
        assert_eq!(c0.start(), NodeIndex(2));
        assert_eq!(c0.end(), NodeIndex(5));
        assert_eq!(
            c0.iter().collect::<Vec<_>>(),
            vec![NodeIndex(2), NodeIndex(3), NodeIndex(5)]
        );

        // next/prev behavior
        assert_eq!(mock.next_node(NodeIndex(2)), Some(NodeIndex(3)));
        assert_eq!(mock.next_node(NodeIndex(3)), Some(NodeIndex(5)));
        assert_eq!(mock.next_node(NodeIndex(5)), None);
        assert_eq!(mock.prev_node(NodeIndex(5)), Some(NodeIndex(3)));
        assert_eq!(mock.prev_node(NodeIndex(3)), Some(NodeIndex(2)));
        assert_eq!(mock.prev_node(NodeIndex(2)), None);

        // head/tail flags
        assert!(mock.is_head_node(NodeIndex(2)));
        assert!(mock.is_tail_node(NodeIndex(5)));
        assert!(!mock.is_sentinel_node(NodeIndex(2)));
        assert!(!mock.is_sentinel_node(NodeIndex(5)));
    }

    #[test]
    fn test_display_empty_chain_is_empty_string() {
        let mock = MockChainSet::new(vec![vec![], vec![NodeIndex(10)]]);
        let empty = ChainRef::new(&mock, ChainIndex(0));
        let non_empty = ChainRef::new(&mock, ChainIndex(1));

        assert!(empty.is_empty());
        assert_eq!(format!("{}", empty), "");

        assert!(!non_empty.is_empty());
        assert_eq!(format!("{}", non_empty), "10");
    }

    #[test]
    fn test_display_ignores_unperformed_in_chainref() {
        // ChainRef::Display doesn't mark unperformed nodes specially,
        // but ensure the presence of such nodes doesn't break formatting.
        let mock = MockChainSet::new(vec![vec![NodeIndex(7), NodeIndex(8), NodeIndex(9)]])
            .with_unperformed([NodeIndex(8)]);
        let c0 = ChainRef::new(&mock, ChainIndex(0));
        assert_eq!(format!("{}", c0), "7->8->9");
        assert!(mock.is_node_unperformed(NodeIndex(8)));
    }
}
