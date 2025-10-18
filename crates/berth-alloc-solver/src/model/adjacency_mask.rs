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

use std::cmp::Ordering;

use fixedbitset::FixedBitSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjacencyMask {
    successor_bits: Vec<FixedBitSet>,
    predecessor_bits: Vec<FixedBitSet>,
}

impl AdjacencyMask {
    #[inline]
    pub fn new(num_flexible_requests: usize) -> Self {
        let mut outgoing = Vec::with_capacity(num_flexible_requests);
        let mut incoming = Vec::with_capacity(num_flexible_requests);
        for _ in 0..num_flexible_requests {
            outgoing.push(FixedBitSet::with_capacity(num_flexible_requests));
            incoming.push(FixedBitSet::with_capacity(num_flexible_requests));
        }
        Self {
            successor_bits: outgoing,
            predecessor_bits: incoming,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.successor_bits.len() == self.predecessor_bits.len());

        self.successor_bits.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        debug_assert!(self.successor_bits.len() == self.predecessor_bits.len());

        self.successor_bits.is_empty()
    }

    #[inline]
    pub fn set(&mut self, source_node_index: usize, target_node_index: usize, has_arc: bool) {
        self.successor_bits[source_node_index].set(target_node_index, has_arc);
        self.predecessor_bits[target_node_index].set(source_node_index, has_arc);
    }

    #[inline]
    pub fn contains(&self, source_node_index: usize, target_node_index: usize) -> bool {
        self.successor_bits[source_node_index].contains(target_node_index)
    }

    #[inline]
    pub fn outgoing_arcs(&self, source_node_index: usize) -> &FixedBitSet {
        &self.successor_bits[source_node_index]
    }

    #[inline]
    pub fn incoming_arcs(&self, target_node_index: usize) -> &FixedBitSet {
        &self.predecessor_bits[target_node_index]
    }

    pub fn from_lists(
        num_flexible_requests: usize,
        outgoing_neighbor_lists: &[Vec<usize>],
    ) -> Self {
        let mut indicator = Self::new(num_flexible_requests);
        for (source_node_index, neighbor_list) in outgoing_neighbor_lists.iter().enumerate() {
            for &target_node_index in neighbor_list {
                indicator.set(source_node_index, target_node_index, true);
            }
        }
        indicator
    }

    pub fn from_lists_topk_ratio(
        num_flexible_requests: usize,
        sorted_outgoing_lists: &[Vec<usize>],
        ratio: f64,
    ) -> Self {
        debug_assert!(ratio > 0.0 && ratio <= 1.0, "Ratio must be in (0, 1]");
        let mut indicator = Self::new(num_flexible_requests);
        for (source_node_index, neighbor_list) in sorted_outgoing_lists.iter().enumerate() {
            let num_to_keep = ((neighbor_list.len() as f64) * ratio).ceil() as usize;
            for &target_node_index in neighbor_list.iter().take(num_to_keep) {
                indicator.set(source_node_index, target_node_index, true);
            }
        }
        indicator
    }

    pub fn from_scores_topk_ratio<F>(
        num_flexible_requests: usize,
        mut score_fn: F,
        ratio: f64,
    ) -> Self
    where
        F: FnMut(usize, usize) -> Option<f64>,
    {
        debug_assert!(ratio > 0.0 && ratio <= 1.0, "Ratio must be in (0, 1]");
        let mut indicator = Self::new(num_flexible_requests);
        let mut scored_neighbors: Vec<(usize, f64)> = Vec::with_capacity(num_flexible_requests);

        for source_node_index in 0..num_flexible_requests {
            scored_neighbors.clear();
            for target_node_index in 0..num_flexible_requests {
                if source_node_index == target_node_index {
                    continue;
                }
                if let Some(score) = score_fn(source_node_index, target_node_index) {
                    scored_neighbors.push((target_node_index, score));
                }
            }

            scored_neighbors.sort_unstable_by(|a, b| {
                match a.1.partial_cmp(&b.1) {
                    Some(ord) => ord,
                    None => Ordering::Greater, // treat NaN as worst
                }
            });

            let num_to_keep = ((scored_neighbors.len() as f64) * ratio).ceil() as usize;
            for &(target_node_index, _) in scored_neighbors.iter().take(num_to_keep) {
                indicator.set(source_node_index, target_node_index, true);
            }
        }
        indicator
    }

    pub fn and(&self, other_indicator: &Self) -> Self {
        let num_flexible_requests = self.len();
        assert_eq!(
            num_flexible_requests,
            other_indicator.len(),
            "Indicators must have the same number of nodes."
        );

        let mut result = Self::new(num_flexible_requests);
        for index in 0..num_flexible_requests {
            // Intersect outgoing arcs
            let mut outgoing_row = self.successor_bits[index].clone();
            outgoing_row.intersect_with(&other_indicator.successor_bits[index]);
            result.successor_bits[index] = outgoing_row;

            // Intersect incoming arcs
            let mut incoming_row = self.predecessor_bits[index].clone();
            incoming_row.intersect_with(&other_indicator.predecessor_bits[index]);
            result.predecessor_bits[index] = incoming_row;
        }
        result
    }

    pub fn or(&self, other_indicator: &Self) -> Self {
        let num_flexible_requests = self.len();
        assert_eq!(
            num_flexible_requests,
            other_indicator.len(),
            "Indicators must have the same number of nodes."
        );

        let mut result = Self::new(num_flexible_requests);
        for index in 0..num_flexible_requests {
            // Union outgoing arcs
            let mut outgoing_row = self.successor_bits[index].clone();
            outgoing_row.union_with(&other_indicator.successor_bits[index]);
            result.successor_bits[index] = outgoing_row;

            // Union incoming arcs
            let mut incoming_row = self.predecessor_bits[index].clone();
            incoming_row.union_with(&other_indicator.predecessor_bits[index]);
            result.predecessor_bits[index] = incoming_row;
        }
        result
    }

    #[inline]
    pub fn neighbors_out_unordered(&self, source_node_index: usize) -> fixedbitset::Ones<'_> {
        self.successor_bits[source_node_index].ones()
    }

    #[inline]
    pub fn neighbors_out_ordered<'a>(
        &'a self,
        source_node_index: usize,
        order: &'a [usize],
    ) -> impl Iterator<Item = usize> + 'a {
        order
            .iter()
            .copied()
            .filter(move |&target_node_index| self.contains(source_node_index, target_node_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_len() {
        let indicator = AdjacencyMask::new(10);
        assert_eq!(indicator.len(), 10);
        assert!(!indicator.is_empty());
        assert_eq!(indicator.successor_bits.len(), 10);
        assert_eq!(indicator.predecessor_bits.len(), 10);

        let empty_indicator = AdjacencyMask::new(0);
        assert!(empty_indicator.is_empty());
    }

    #[test]
    fn test_set_and_contains_updates_both_matrices() {
        let mut indicator = AdjacencyMask::new(5);

        // Set an arc and verify outgoing and incoming state
        indicator.set(0, 1, true);
        assert!(indicator.contains(0, 1));
        assert!(!indicator.contains(1, 0));
        assert!(indicator.outgoing_arcs(0).contains(1));
        assert!(indicator.incoming_arcs(1).contains(0));
        assert!(!indicator.incoming_arcs(0).contains(1));

        // Unset the arc and verify again
        indicator.set(0, 1, false);
        assert!(!indicator.contains(0, 1));
        assert!(!indicator.outgoing_arcs(0).contains(1));
        assert!(!indicator.incoming_arcs(1).contains(0));
    }

    #[test]
    fn test_from_lists_builds_correctly() {
        let lists = vec![vec![1, 2], vec![2], vec![]]; // Arcs: 0->1, 0->2, 1->2
        let indicator = AdjacencyMask::from_lists(3, &lists);

        assert!(indicator.contains(0, 1) && indicator.contains(0, 2) && indicator.contains(1, 2));
        assert!(!indicator.contains(2, 0));
        assert_eq!(
            // Changed .iter() to .ones()
            indicator.incoming_arcs(2).ones().collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(indicator.incoming_arcs(0).count_ones(..), 0);
    }

    #[test]
    fn test_from_lists_topk_ratio_truncates() {
        let sorted_lists = vec![vec![3, 1, 2, 0], vec![0, 2, 1]];
        // Ratio 0.5: keep ceil(4 * 0.5)=2 for list 0, and ceil(3 * 0.5)=2 for list 1.
        let indicator = AdjacencyMask::from_lists_topk_ratio(4, &sorted_lists, 0.5);

        assert!(indicator.contains(0, 3) && indicator.contains(0, 1));
        assert!(!indicator.contains(0, 2)); // Truncated
        assert!(indicator.contains(1, 0) && indicator.contains(1, 2));
        assert!(!indicator.contains(1, 1)); // Truncated
    }

    #[test]
    fn test_from_scores_topk_ratio_selects_best() {
        // Score is just the target index. Lower is better.
        let score_function = |_, target| Some(target as f64);
        // num_flexible_requests=4. Neighbors for node 0 are 1,2,3. Scores are 1.0, 2.0, 3.0.
        // Ratio 0.5 keeps ceil(3 * 0.5)=2 best scores: 1.0 and 2.0.
        // So arcs 0->1 and 0->2 should exist.
        let indicator = AdjacencyMask::from_scores_topk_ratio(4, score_function, 0.5);

        assert!(indicator.contains(0, 1));
        assert!(indicator.contains(0, 2));
        assert!(!indicator.contains(0, 3)); // Worse score, truncated
    }

    #[test]
    fn test_logical_and_intersection() {
        let indicator1 = AdjacencyMask::from_lists(3, &[vec![0, 1], vec![1, 2], vec![0, 2]]);
        let indicator2 = AdjacencyMask::from_lists(3, &[vec![1, 2], vec![2], vec![0, 1]]);

        let result = indicator1.and(&indicator2);

        // Arcs that exist in both: 0->1, 1->2, 2->0
        assert!(result.contains(0, 1));
        assert!(!result.contains(0, 0)); // Only in indicator1
        assert!(!result.contains(0, 2)); // Only in indicator1
        assert!(result.contains(1, 2));
        assert!(result.contains(2, 0));
    }

    #[test]
    fn test_logical_or_union() {
        let indicator1 = AdjacencyMask::from_lists(3, &[vec![0], vec![1]]);
        let indicator2 = AdjacencyMask::from_lists(3, &[vec![1], vec![2]]);

        let result = indicator1.or(&indicator2);

        // Arcs that exist in either: 0->0, 0->1, 1->1, 1->2
        assert!(result.contains(0, 0));
        assert!(result.contains(0, 1));
        assert!(result.contains(1, 1));
        assert!(result.contains(1, 2));
        assert!(!result.contains(2, 0));
    }

    #[test]
    fn test_neighbor_iterators() {
        let indicator =
            AdjacencyMask::from_lists(4, &[vec![3, 1], vec![], vec![0, 1, 2, 3], vec![0]]);

        // Unordered
        let mut neighbors_of_0: Vec<_> = indicator.neighbors_out_unordered(0).collect();
        neighbors_of_0.sort();
        assert_eq!(neighbors_of_0, vec![1, 3]);

        // Ordered
        let custom_order = vec![0, 1, 2, 3];
        let ordered_neighbors: Vec<_> = indicator.neighbors_out_ordered(2, &custom_order).collect();
        assert_eq!(ordered_neighbors, vec![0, 1, 2, 3]);

        let reverse_order = vec![3, 2, 1, 0];
        let reverse_ordered_neighbors: Vec<_> =
            indicator.neighbors_out_ordered(2, &reverse_order).collect();
        assert_eq!(reverse_ordered_neighbors, vec![3, 2, 1, 0]);
    }
}
