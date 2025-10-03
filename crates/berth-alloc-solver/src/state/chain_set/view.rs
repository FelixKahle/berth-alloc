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

pub trait ChainSetView {
    fn num_nodes(&self) -> usize;
    fn num_chains(&self) -> usize;

    fn start_of_chain(&self, chain: usize) -> usize;
    fn end_of_chain(&self, chain: usize) -> usize;
    fn chain(&self, chain: usize) -> ChainRef<'_, Self>
    where
        Self: Sized,
    {
        ChainRef::new(self, chain)
    }

    fn next_node(&self, node: usize) -> Option<usize>;
    fn prev_node(&self, node: usize) -> Option<usize>;

    fn is_sentinel_node(&self, node: usize) -> bool;
    fn is_head_node(&self, node: usize) -> bool;
    fn is_tail_node(&self, node: usize) -> bool;

    fn is_node_unperformed(&self, node: usize) -> bool;

    fn is_chain_empty(&self, chain: usize) -> bool;

    fn iter_chain(&self, chain: usize) -> impl Iterator<Item = usize> + '_;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainRef<'chain, C: ChainSetView> {
    chain_view: &'chain C,
    chain: usize,
}

impl<'chain, C: ChainSetView> ChainRef<'chain, C> {
    #[inline]
    pub fn new(chain_view: &'chain C, chain: usize) -> Self {
        assert!(chain < chain_view.num_chains());

        Self { chain_view, chain }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chain_view.is_chain_empty(self.chain)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.chain_view.iter_chain(self.chain)
    }

    #[inline]
    pub fn start(&self) -> usize {
        self.chain_view.start_of_chain(self.chain)
    }

    #[inline]
    pub fn end(&self) -> usize {
        self.chain_view.end_of_chain(self.chain)
    }
}

impl<'chain, C: ChainSetView> std::fmt::Display for ChainRef<'chain, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut it = self
            .iter()
            .filter(|&n| !self.chain_view.is_sentinel_node(n));

        if let Some(first) = it.next() {
            write!(f, "{}", first)?;
            for node in it {
                write!(f, "->{}", node)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[derive(Default)]
    struct MockChainSet {
        chains: Vec<Vec<usize>>,
        unperformed: HashSet<usize>,
    }

    impl MockChainSet {
        fn new(chains: Vec<Vec<usize>>) -> Self {
            Self {
                chains,
                unperformed: HashSet::new(),
            }
        }

        #[allow(dead_code)]
        fn with_unperformed(mut self, nodes: impl IntoIterator<Item = usize>) -> Self {
            self.unperformed.extend(nodes);
            self
        }

        fn find_node(&self, node: usize) -> Option<(usize, usize)> {
            for (ci, c) in self.chains.iter().enumerate() {
                if let Some(pos) = c.iter().position(|&n| n == node) {
                    return Some((ci, pos));
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

        fn start_of_chain(&self, chain: usize) -> usize {
            // For this mock: first node if non-empty, otherwise 0 (unused in display tests)
            self.chains[chain].first().copied().unwrap_or(0)
        }

        fn end_of_chain(&self, chain: usize) -> usize {
            // For this mock: last node if non-empty, otherwise 0 (unused in display tests)
            self.chains[chain].last().copied().unwrap_or(0)
        }

        fn next_node(&self, node: usize) -> Option<usize> {
            let (ci, pos) = self.find_node(node)?;
            self.chains[ci].get(pos + 1).copied()
        }

        fn prev_node(&self, node: usize) -> Option<usize> {
            let (ci, pos) = self.find_node(node)?;
            if pos > 0 {
                Some(self.chains[ci][pos - 1])
            } else {
                None
            }
        }

        fn is_sentinel_node(&self, _node: usize) -> bool {
            // This mock has no sentinels in iteration
            false
        }

        fn is_head_node(&self, node: usize) -> bool {
            if let Some((ci, pos)) = self.find_node(node) {
                pos == 0 && !self.chains[ci].is_empty()
            } else {
                false
            }
        }

        fn is_tail_node(&self, node: usize) -> bool {
            if let Some((ci, pos)) = self.find_node(node) {
                pos + 1 == self.chains[ci].len() && !self.chains[ci].is_empty()
            } else {
                false
            }
        }

        fn is_node_unperformed(&self, node: usize) -> bool {
            self.unperformed.contains(&node)
        }

        fn is_chain_empty(&self, chain: usize) -> bool {
            self.chains[chain].is_empty()
        }

        fn iter_chain(&self, chain: usize) -> impl Iterator<Item = usize> + '_ {
            self.chains[chain].iter().copied()
        }
    }

    #[test]
    fn test_display_formats_non_empty_chain() {
        let mock = MockChainSet::new(vec![vec![2, 3, 5]]);
        let c0 = ChainRef::new(&mock, 0);

        let s = format!("{}", c0);
        assert_eq!(s, "2->3->5");

        // Sanity checks
        assert!(!c0.is_empty());
        assert_eq!(c0.start(), 2);
        assert_eq!(c0.end(), 5);
        assert_eq!(c0.iter().collect::<Vec<_>>(), vec![2, 3, 5]);

        // next/prev behavior
        assert_eq!(mock.next_node(2), Some(3));
        assert_eq!(mock.next_node(3), Some(5));
        assert_eq!(mock.next_node(5), None);
        assert_eq!(mock.prev_node(5), Some(3));
        assert_eq!(mock.prev_node(3), Some(2));
        assert_eq!(mock.prev_node(2), None);

        // head/tail flags
        assert!(mock.is_head_node(2));
        assert!(mock.is_tail_node(5));
        assert!(!mock.is_sentinel_node(2));
        assert!(!mock.is_sentinel_node(5));
    }

    #[test]
    fn test_display_empty_chain_is_empty_string() {
        let mock = MockChainSet::new(vec![vec![], vec![10]]);
        let empty = ChainRef::new(&mock, 0);
        let non_empty = ChainRef::new(&mock, 1);

        assert!(empty.is_empty());
        assert_eq!(format!("{}", empty), "");

        assert!(!non_empty.is_empty());
        assert_eq!(format!("{}", non_empty), "10");
    }

    #[test]
    fn test_display_ignores_unperformed_in_chainref() {
        // ChainRef::Display doesn't mark unperformed nodes specially,
        // but ensure the presence of such nodes doesn't break formatting.
        let mock = MockChainSet::new(vec![vec![7, 8, 9]]).with_unperformed([8]);
        let c0 = ChainRef::new(&mock, 0);
        assert_eq!(format!("{}", c0), "7->8->9");
        assert!(mock.is_node_unperformed(8));
    }
}
