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

use crate::model::index::RequestIndex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborLists {
    incoming: Vec<Vec<RequestIndex>>, // [v] contains neighbors u with u→v
    outgoing: Vec<Vec<RequestIndex>>, // [u] contains neighbors v with u→v
}

impl NeighborLists {
    #[inline]
    pub fn new(incoming: Vec<Vec<RequestIndex>>, outgoing: Vec<Vec<RequestIndex>>) -> Self {
        debug_assert_eq!(
            incoming.len(),
            outgoing.len(),
            "incoming/outgoing size mismatch"
        );
        Self { incoming, outgoing }
    }

    #[inline]
    pub fn incoming(&self) -> &[Vec<RequestIndex>] {
        &self.incoming
    }

    #[inline]
    pub fn outgoing(&self) -> &[Vec<RequestIndex>] {
        &self.outgoing
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        // A graph is only truly empty if it has no nodes,
        // so checking either list's top-level emptiness is sufficient.
        self.outgoing.is_empty()
    }

    #[inline]
    pub fn incoming_for(&self, node: RequestIndex) -> Option<&[RequestIndex]> {
        self.incoming.get(node.get()).map(|v| v.as_slice())
    }

    #[inline]
    pub fn outgoing_for(&self, node: RequestIndex) -> Option<&[RequestIndex]> {
        self.outgoing.get(node.get()).map(|v| v.as_slice())
    }
}

// --- Test Module ---
#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a sample graph for tests.
    // Graph:
    // 0 -> 1
    // 1 -> 2
    // 2 -> 0
    fn create_sample_lists() -> NeighborLists {
        let outgoing = vec![
            vec![RequestIndex::new(1)], // 0 -> {1}
            vec![RequestIndex::new(2)], // 1 -> {2}
            vec![RequestIndex::new(0)], // 2 -> {0}
        ];
        let incoming = vec![
            vec![RequestIndex::new(2)], // 0 <- {2}
            vec![RequestIndex::new(0)], // 1 <- {0}
            vec![RequestIndex::new(1)], // 2 <- {1}
        ];
        NeighborLists::new(incoming, outgoing)
    }

    #[test]
    fn test_new_and_getters() {
        let lists = create_sample_lists();
        assert_eq!(lists.outgoing().len(), 3);
        assert_eq!(lists.incoming().len(), 3);
        assert_eq!(lists.outgoing()[0], vec![RequestIndex::new(1)]);
    }

    #[test]
    fn test_is_empty() {
        let non_empty_lists = create_sample_lists();
        assert!(!non_empty_lists.is_empty());

        let empty_lists = NeighborLists::new(vec![], vec![]);
        assert!(empty_lists.is_empty());
    }

    #[test]
    fn test_outgoing_for() {
        let lists = create_sample_lists();

        // Node 0 should have one outgoing neighbor: 1
        assert_eq!(
            lists.outgoing_for(RequestIndex::new(0)),
            Some([RequestIndex::new(1)].as_slice())
        );

        // Test an out-of-bounds node
        assert_eq!(lists.outgoing_for(RequestIndex::new(99)), None);
    }

    #[test]
    fn test_incoming_for() {
        let lists = create_sample_lists();

        // Node 2 should have one incoming neighbor: 1
        assert_eq!(
            lists.incoming_for(RequestIndex::new(2)),
            Some([RequestIndex::new(1)].as_slice())
        );

        // Test an out-of-bounds node
        assert_eq!(lists.incoming_for(RequestIndex::new(99)), None);
    }

    #[test]
    fn test_node_with_no_neighbors() {
        let outgoing = vec![vec![], vec![RequestIndex::new(0)]]; // 1 -> 0
        let incoming = vec![vec![RequestIndex::new(1)], vec![]]; // 0 <- 1
        let lists = NeighborLists::new(incoming, outgoing);

        // Node 0 has no outgoing neighbors
        assert_eq!(
            lists.outgoing_for(RequestIndex::new(0)),
            Some([].as_slice())
        );

        // Node 1 has no incoming neighbors
        assert_eq!(
            lists.incoming_for(RequestIndex::new(1)),
            Some([].as_slice())
        );
    }
}
