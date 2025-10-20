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

use super::features::Feature;
use std::collections::HashMap;

/// Decay varieties for the penalty memory.
#[derive(Clone, Copy, Debug)]
pub enum DecayMode {
    Multiplicative { num: u32, den: u32 }, // v := floor(v * num / den)
    Subtractive { step: i64 },             // v := max(0, v - step)
}

/// Penalty map with optional decay and a saturation cap.
#[derive(Clone)]
pub struct PenaltyStore {
    pub(crate) map: HashMap<Feature, i64>,
    decay: Option<DecayMode>,
    max_penalty: i64,
}

impl Default for PenaltyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PenaltyStore {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            decay: None,
            max_penalty: i64::MAX / 4,
        }
    }
    #[inline]
    pub fn with_decay(mut self, d: DecayMode) -> Self {
        self.decay = Some(d);
        self
    }
    #[inline]
    pub fn with_max_penalty(mut self, cap: i64) -> Self {
        self.max_penalty = cap.max(1);
        self
    }

    #[inline]
    pub fn add_one(&mut self, f: Feature, step: i64) {
        let e = self.map.entry(f).or_insert(0);
        *e = (*e + step).min(self.max_penalty);
    }

    #[inline]
    pub fn get(&self, f: &Feature) -> i64 {
        *self.map.get(f).unwrap_or(&0)
    }

    #[inline]
    pub fn sum<'a>(&self, feats: impl IntoIterator<Item = &'a Feature>) -> i64 {
        let mut s = 0i64;
        for f in feats {
            s = s.saturating_add(self.get(f));
        }
        s
    }

    pub fn decay_once(&mut self) {
        if let Some(d) = self.decay {
            match d {
                DecayMode::Multiplicative { num, den } => {
                    if den == 0 || num >= den {
                        return;
                    }
                    for v in self.map.values_mut() {
                        if *v > 0 {
                            *v = ((*v as i128 * num as i128) / den as i128) as i64;
                        }
                    }
                }
                DecayMode::Subtractive { step } => {
                    for v in self.map.values_mut() {
                        *v = (*v - step).max(0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::feature_signal::features::Feature;

    #[test]
    fn test_add_one_get_and_sum_basics() {
        let mut store = PenaltyStore::new();
        let f1 = Feature::Request { req: 1 };
        let f2 = Feature::BerthTime { berth: 2, tb: 7 };

        // Missing entries default to 0
        assert_eq!(store.get(&f1), 0);
        assert_eq!(store.get(&f2), 0);

        store.add_one(f1.clone(), 3);
        assert_eq!(store.get(&f1), 3);

        store.add_one(f1.clone(), 5);
        assert_eq!(store.get(&f1), 8);

        // sum over [f1, f2] where f2 still zero
        let s1 = store.sum([&f1, &f2]);
        assert_eq!(s1, 8);

        store.add_one(f2.clone(), 4);
        let s2 = store.sum([&f1, &f2]);
        assert_eq!(s2, 12);
    }

    #[test]
    fn test_max_penalty_cap_and_floor_to_one() {
        // Cap at 5
        let mut store = PenaltyStore::new().with_max_penalty(5);
        let f = Feature::RequestBerth { req: 0, berth: 0 };

        store.add_one(f.clone(), 10);
        assert_eq!(store.get(&f), 5, "should clamp to cap 5");

        store.add_one(f.clone(), 3);
        assert_eq!(store.get(&f), 5, "stays at cap");

        // Cap floor to 1 when given 0
        let mut store2 = PenaltyStore::new().with_max_penalty(0);
        let g = Feature::TimeBucket { tb: 42 };
        store2.add_one(g.clone(), 10);
        assert_eq!(store2.get(&g), 1, "cap=0 should floor to 1");
    }

    #[test]
    fn test_decay_multiplicative_basic_and_guards() {
        // Basic decay: v := floor(v * num / den), only for v > 0
        let mut store =
            PenaltyStore::new().with_decay(DecayMode::Multiplicative { num: 1, den: 2 });
        let f1 = Feature::Request { req: 7 };
        let f2 = Feature::Berth { berth: 3 };
        let f3 = Feature::RequestTime { req: 1, tb: 9 };

        store.add_one(f1.clone(), 7); // >0
        store.add_one(f2.clone(), 0); // exactly 0
        store.add_one(f3.clone(), 1); // >0

        store.decay_once();
        assert_eq!(store.get(&f1), 3, "floor(7*1/2) = 3");
        assert_eq!(store.get(&f2), 0, "zero remains zero");
        assert_eq!(store.get(&f3), 0, "floor(1*1/2) = 0");

        // Guard: den == 0 => no-op
        let mut store2 =
            PenaltyStore::new().with_decay(DecayMode::Multiplicative { num: 1, den: 0 });
        let h = Feature::Request { req: 2 };
        store2.add_one(h.clone(), 10);
        store2.decay_once();
        assert_eq!(store2.get(&h), 10, "den=0 must be a no-op");

        // Guard: num >= den => no-op (e.g., 2/2)
        let mut store3 =
            PenaltyStore::new().with_decay(DecayMode::Multiplicative { num: 2, den: 2 });
        let k = Feature::Request { req: 3 };
        store3.add_one(k.clone(), 10);
        store3.decay_once();
        assert_eq!(store3.get(&k), 10, "num >= den must be a no-op");
    }

    #[test]
    fn test_decay_subtractive_basic() {
        // v := max(0, v - step)
        let mut store = PenaltyStore::new().with_decay(DecayMode::Subtractive { step: 3 });
        let f1 = Feature::TimeBucket { tb: 1 };
        let f2 = Feature::TimeBucket { tb: 2 };
        let f3 = Feature::TimeBucket { tb: 3 };

        store.add_one(f1.clone(), 10);
        store.add_one(f2.clone(), 2);
        store.add_one(f3.clone(), 0);

        store.decay_once();
        assert_eq!(store.get(&f1), 7);
        assert_eq!(store.get(&f2), 0, "floors at 0");
        assert_eq!(store.get(&f3), 0, "remains 0");
    }

    #[test]
    fn test_sum_saturating_add_on_overflow() {
        let mut store = PenaltyStore::new().with_max_penalty(i64::MAX);
        let f1 = Feature::Request { req: 10 };
        let f2 = Feature::Request { req: 11 };

        // Set very large penalties so their sum would overflow without saturation.
        store.add_one(f1.clone(), i64::MAX - 10);
        store.add_one(f2.clone(), i64::MAX - 20);

        let s = store.sum([&f1, &f2]);
        assert_eq!(s, i64::MAX, "sum should saturate to i64::MAX");
    }

    #[test]
    fn test_clone_produces_independent_copy() {
        let mut store = PenaltyStore::new();
        let f = Feature::Request { req: 1 };
        store.add_one(f.clone(), 5);

        let cloned = store.clone();
        assert_eq!(cloned.get(&f), 5);

        // Mutate original; clone must not change
        store.add_one(f.clone(), 2);
        assert_eq!(store.get(&f), 7);
        assert_eq!(cloned.get(&f), 5, "cloned instance is independent");
    }
}
