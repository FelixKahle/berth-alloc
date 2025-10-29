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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FitnessDelta {
    pub delta_cost: Cost,
    pub delta_unassigned: i32,
}

impl std::fmt::Display for FitnessDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FitnessDelta(delta_cost: {}, delta_unassigned: {})",
            self.delta_cost, self.delta_unassigned
        )
    }
}

impl std::ops::Add for FitnessDelta {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            delta_cost: self
                .delta_cost
                .checked_add(rhs.delta_cost)
                .expect("Cost addition overflowed"),
            delta_unassigned: self
                .delta_unassigned
                .checked_add(rhs.delta_unassigned)
                .expect("Unassigned requests addition overflowed"),
        }
    }
}

impl std::ops::AddAssign for FitnessDelta {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.delta_cost = self
            .delta_cost
            .checked_add(rhs.delta_cost)
            .expect("Cost addition overflowed");
        self.delta_unassigned = self
            .delta_unassigned
            .checked_add(rhs.delta_unassigned)
            .expect("Unassigned requests addition overflowed");
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Fitness {
    pub cost: Cost,
    pub unassigned_requests: usize,
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

impl Fitness {
    #[inline]
    pub const fn new(cost: Cost, unassigned_requests: usize) -> Self {
        Self {
            cost,
            unassigned_requests,
        }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self {
            cost: 0,
            unassigned_requests: 0,
        }
    }

    #[inline]
    pub fn is_feasible(&self) -> bool {
        self.unassigned_requests == 0
    }

    /// Returns a new `Fitness` equal to `self` with `delta` applied.
    ///
    /// Panics
    ///
    /// - Panics with “attempt to add with overflow” if adding `delta.delta_cost`
    ///   to `self.cost` overflows the underlying `i64`.
    /// - Panics with “delta_unassigned does not fit in isize” if
    ///   `delta.delta_unassigned` cannot be represented as an `isize`.
    /// - Panics with “attempt to add/subtract with overflow” if applying
    ///   `delta.delta_unassigned` would overflow or underflow the underlying
    ///   `usize` of `self.unassigned_requests`.
    #[inline]
    pub fn apply_delta(&self, delta: &FitnessDelta) -> Self {
        let new_cost = self
            .cost
            .checked_add(delta.delta_cost)
            .expect("attempt to add with overflow");

        let delta_isize = isize::try_from(delta.delta_unassigned)
            .expect("delta_unassigned does not fit in isize");

        let new_unassigned = self
            .unassigned_requests
            .checked_add_signed(delta_isize)
            .expect("attempt to add/subtract with overflow");

        Self {
            cost: new_cost,
            unassigned_requests: new_unassigned,
        }
    }
}

impl Ord for Fitness {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.unassigned_requests.cmp(&other.unassigned_requests) {
            std::cmp::Ordering::Equal => self.cost.cmp(&other.cost),
            ord => ord,
        }
    }
}

impl PartialOrd for Fitness {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FitnessDelta {
    #[inline]
    pub const fn new(delta_cost: Cost, delta_unassigned: i32) -> Self {
        Self {
            delta_cost,
            delta_unassigned,
        }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self {
            delta_cost: 0,
            delta_unassigned: 0,
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.delta_cost == 0 && self.delta_unassigned == 0
    }
}

/// Adds a `FitnessDelta` to a `Fitness`, producing a new `Fitness`.
///
/// Panics
///
/// - Panics with “attempt to add with overflow” if adding `delta_cost` overflows.
/// - Panics with “delta_unassigned does not fit in isize” if the delta cannot be
///   represented as an `isize`.
/// - Panics with “attempt to add/subtract with overflow” if applying `delta_unassigned`
///   overflows/underflows the `usize` count.
impl std::ops::Add<FitnessDelta> for Fitness {
    type Output = Fitness;

    #[inline]
    fn add(self, rhs: FitnessDelta) -> Fitness {
        self.apply_delta(&rhs)
    }
}

/// Adds a `FitnessDelta` to a `Fitness` in-place.
///
/// Panics
///
/// - Panics with “attempt to add with overflow” if adding `delta_cost` overflows.
/// - Panics with “delta_unassigned does not fit in isize” if the delta cannot be
///   represented as an `isize`.
/// - Panics with “attempt to add/subtract with overflow” if applying `delta_unassigned`
///   overflows/underflows the `usize` count.
impl std::ops::AddAssign<FitnessDelta> for Fitness {
    #[inline]
    fn add_assign(&mut self, rhs: FitnessDelta) {
        *self = self.apply_delta(&rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_fitness_new_zero_is_feasible_and_display() {
        let f = Fitness::new(456, 3);
        assert_eq!(f.cost, 456);
        assert_eq!(f.unassigned_requests, 3);
        assert!(!f.is_feasible());
        assert_eq!(f.to_string(), "Fitness(cost: 456, unassigned_requests: 3)");

        let z = Fitness::zero();
        assert_eq!(z.cost, 0);
        assert_eq!(z.unassigned_requests, 0);
        assert!(z.is_feasible());
        assert_eq!(z.to_string(), "Fitness(cost: 0, unassigned_requests: 0)");
    }

    #[test]
    fn test_fitness_ord_and_partial_ord_lex_ordering() {
        // Ordering: fewer unassigned first; tie-break on lower cost
        let a = Fitness::new(100, 0);
        let b = Fitness::new(150, 0);
        let c = Fitness::new(1, 1);
        let d = Fitness::new(0, 2);

        assert!(a < b, "lower cost among equal unassigned should be better");
        assert!(a < c, "feasible dominates infeasible");
        assert!(c < d, "fewer unassigned dominates");

        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
        assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));

        let mut v = vec![d, b, c, a];
        v.sort();
        assert_eq!(v, vec![a, b, c, d]);
    }

    #[test]
    fn test_fitness_delta_new_zero_is_zero_and_display() {
        let d = FitnessDelta::new(12, -2);
        assert_eq!(d.delta_cost, 12);
        assert_eq!(d.delta_unassigned, -2);
        assert_eq!(
            d.to_string(),
            "FitnessDelta(delta_cost: 12, delta_unassigned: -2)"
        );

        let z = FitnessDelta::zero();
        assert_eq!(z.delta_cost, 0);
        assert_eq!(z.delta_unassigned, 0);
        assert!(z.is_zero());
        assert_eq!(
            z.to_string(),
            "FitnessDelta(delta_cost: 0, delta_unassigned: 0)"
        );
    }

    #[test]
    fn test_fitness_delta_add_and_add_assign_ok() {
        let a = FitnessDelta::new(10, 1);
        let b = FitnessDelta::new(-3, -2);
        let c = a + b;
        assert_eq!(c.delta_cost, 7);
        assert_eq!(c.delta_unassigned, -1);

        let mut acc = FitnessDelta::new(5, 0);
        acc += FitnessDelta::new(5, 3);
        assert_eq!(acc.delta_cost, 10);
        assert_eq!(acc.delta_unassigned, 3);
    }

    #[test]
    fn test_apply_delta_ok_paths() {
        let f = Fitness::new(100, 2);

        // +50 cost, assign one (delta_unassigned = -1)
        let f2 = f.apply_delta(&FitnessDelta::new(50, -1));
        assert_eq!(f2.cost, 150);
        assert_eq!(f2.unassigned_requests, 1);

        // -25 cost, unassign two (delta_unassigned = +2)
        let f3 = f2.apply_delta(&FitnessDelta::new(-25, 2));
        assert_eq!(f3.cost, 125);
        assert_eq!(f3.unassigned_requests, 3);
    }

    #[test]
    fn test_add_and_add_assign_fitness_with_delta_ok() {
        let base = Fitness::new(10, 1);
        let delta = FitnessDelta::new(5, -1);

        // Add
        let res = base + delta;
        assert_eq!(res.cost, 15);
        assert_eq!(res.unassigned_requests, 0);

        // AddAssign
        let mut f = Fitness::new(10, 1);
        f += FitnessDelta::new(-3, 2);
        assert_eq!(f.cost, 7);
        assert_eq!(f.unassigned_requests, 3);
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn test_apply_delta_cost_overflow_panics() {
        // i64::MAX + 1 overflows
        let f = Fitness::new(i64::MAX, 0);
        let _ = f.apply_delta(&FitnessDelta::new(1, 0));
    }

    #[test]
    #[should_panic(expected = "attempt to add/subtract with overflow")]
    fn test_apply_delta_unassigned_underflow_panics() {
        // 0 - 1 underflows
        let f = Fitness::new(0, 0);
        let _ = f.apply_delta(&FitnessDelta::new(0, -1));
    }

    #[test]
    #[should_panic(expected = "attempt to add/subtract with overflow")]
    fn test_apply_delta_unassigned_overflow_panics() {
        // usize::MAX + 1 overflows
        let f = Fitness::new(0, usize::MAX);
        let _ = f.apply_delta(&FitnessDelta::new(0, 1));
    }

    #[test]
    #[should_panic(expected = "Cost addition overflowed")]
    fn test_fitness_delta_add_cost_overflow_panics() {
        // FitnessDelta + FitnessDelta where delta_cost overflows
        let a = FitnessDelta::new(i64::MAX, 0);
        let b = FitnessDelta::new(1, 0);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "Unassigned requests addition overflowed")]
    fn test_fitness_delta_add_unassigned_overflow_panics() {
        // FitnessDelta + FitnessDelta where delta_unassigned overflows i32
        let a = FitnessDelta::new(0, i32::MAX);
        let b = FitnessDelta::new(0, 1);
        let _ = a + b;
    }
}
