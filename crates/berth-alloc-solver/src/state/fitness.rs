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

use berth_alloc_core::prelude::Cost;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fitness {
    pub cost: Cost,
    pub unassigned_requests: usize,
}

impl Fitness {
    #[inline]
    pub fn new(cost: Cost, unassigned_requests: usize) -> Self {
        Self {
            cost,
            unassigned_requests,
        }
    }
}

impl PartialOrd for Fitness {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fitness {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by unassigned requests, then by cost
        match self.unassigned_requests.cmp(&other.unassigned_requests) {
            std::cmp::Ordering::Equal => self.cost.cmp(&other.cost),
            ord => ord,
        }
    }
}

impl std::fmt::Display for Fitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fitness(cost: {}, unassigned_requests: {})",
            self.cost, self.unassigned_requests
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unassigned_primary_ordering() {
        let a = Fitness::new(100, 0);
        let b = Fitness::new(10, 1);
        // a has fewer unassigned (0 < 1) so a < b
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_cost_secondary_ordering_when_unassigned_equal() {
        let a = Fitness::new(10, 1);
        let b = Fitness::new(20, 1);
        // Same unassigned; lower cost is less
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_equality_when_both_equal() {
        let a = Fitness::new(10, 1);
        let b = Fitness::new(10, 1);
        assert_eq!(a, b);
        assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn test_sorting_mixed_values() {
        let mut v = vec![
            Fitness::new(50, 2),
            Fitness::new(10, 1),
            Fitness::new(5, 3),
            Fitness::new(20, 1),
            Fitness::new(0, 0),
            Fitness::new(0, 2),
        ];
        v.sort();
        let by_tuple: Vec<_> = v.iter().map(|f| (f.unassigned_requests, f.cost)).collect();
        assert_eq!(
            by_tuple,
            vec![(0, 0), (1, 10), (1, 20), (2, 0), (2, 50), (3, 5)]
        );
    }

    #[test]
    fn test_partial_cmp_matches_cmp() {
        use std::cmp::Ordering::*;
        let cases = vec![
            (Fitness::new(1, 0), Fitness::new(2, 0), Less),
            (Fitness::new(1, 1), Fitness::new(1, 0), Greater),
            (Fitness::new(10, 2), Fitness::new(5, 2), Greater),
            (Fitness::new(7, 3), Fitness::new(7, 3), Equal),
        ];
        for (a, b, expected) in cases {
            assert_eq!(a.cmp(&b), expected);
            assert_eq!(a.partial_cmp(&b), Some(expected));
        }
    }
}
