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

use crate::{
    model::index::RequestIndex,
    state::{decisionvar::DecisionVar, fitness::FitnessDelta, terminal::delta::TerminalDelta},
};
use fixedbitset::FixedBitSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionVarPatch<T> {
    pub index: RequestIndex,
    pub patch: DecisionVar<T>,
}

impl<T: Copy + Ord> DecisionVarPatch<T> {
    #[inline]
    pub const fn new(index: RequestIndex, patch: DecisionVar<T>) -> Self {
        Self { index, patch }
    }
}

impl<T: Copy + Ord + std::fmt::Display> std::fmt::Display for DecisionVarPatch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DecisionVarPatch(index: {}, patch: {})",
            self.index, self.patch
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan<'p, T: Copy + Ord> {
    pub decision_var_patches: Vec<DecisionVarPatch<T>>,
    pub terminal_delta: TerminalDelta<'p, T>,
    pub fitness_delta: FitnessDelta,
}

impl<'p, T: Copy + Ord> Plan<'p, T> {
    #[inline]
    pub fn new_delta(
        decision_var_patches: Vec<DecisionVarPatch<T>>,
        terminal_delta: TerminalDelta<'p, T>,
        fitness_delta: FitnessDelta,
    ) -> Self {
        Self {
            decision_var_patches,
            terminal_delta,
            fitness_delta,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.decision_var_patches.is_empty()
            && self.terminal_delta.is_empty()
            && self.fitness_delta.delta_cost == 0
            && self.fitness_delta.delta_unassigned == 0
    }

    #[inline]
    pub fn concat(self, other: Plan<'p, T>) -> Plan<'p, T> {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }

        let terminal_delta = {
            let mut updates: Vec<_> = self.terminal_delta.into_iter().collect();
            updates.extend(other.terminal_delta);

            if updates.is_empty() {
                TerminalDelta::empty()
            } else {
                let max_key = updates.iter().map(|(ix, _)| ix.get()).max().unwrap_or(0);
                let mut seen = FixedBitSet::with_capacity(max_key + 1);
                seen.grow(max_key + 1);

                let mut dedup = Vec::with_capacity(updates.len());
                for (ix, occ) in updates.into_iter().rev() {
                    let key = ix.get();
                    if !seen.contains(key) {
                        seen.set(key, true);
                        dedup.push((ix, occ));
                    }
                }
                dedup.reverse();
                TerminalDelta::from_updates(dedup)
            }
        };

        let mut patches = self.decision_var_patches;
        patches.extend(other.decision_var_patches);

        let dedup_patches = if patches.is_empty() {
            Vec::new()
        } else {
            let max_key = patches.iter().map(|p| p.index.get()).max().unwrap_or(0);
            let mut seen = FixedBitSet::with_capacity(max_key + 1);
            seen.grow(max_key + 1);

            let mut dedup = Vec::with_capacity(patches.len());
            for p in patches.into_iter().rev() {
                let key = p.index.get();
                if !seen.contains(key) {
                    seen.set(key, true);
                    dedup.push(p);
                }
            }
            dedup.reverse();
            dedup
        };

        let fitness_delta = FitnessDelta {
            delta_cost: self.fitness_delta.delta_cost + other.fitness_delta.delta_cost,
            delta_unassigned: self.fitness_delta.delta_unassigned
                + other.fitness_delta.delta_unassigned,
        };

        Plan::new_delta(dedup_patches, terminal_delta, fitness_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::index::{BerthIndex, RequestIndex};
    use crate::state::berth::berthocc::{BerthOccupancy, BerthWrite};
    use berth_alloc_core::prelude::{TimeInterval, TimePoint};
    use berth_alloc_model::prelude::{Berth, BerthIdentifier};

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex::new(n)
    }

    fn mk_berths() -> Vec<Berth<i64>> {
        vec![
            Berth::from_windows(bid(1), [iv(0, 100)]),
            Berth::from_windows(bid(2), [iv(0, 100)]),
        ]
    }

    #[test]
    fn test_concat_fast_paths_empty_plans() {
        let empty = Plan::new_delta(
            Vec::<DecisionVarPatch<i64>>::new(),
            TerminalDelta::empty(),
            FitnessDelta::zero(),
        );

        let non_empty_patch = Plan::new_delta(
            vec![DecisionVarPatch::new(
                RequestIndex::new(1),
                DecisionVar::unassigned(),
            )],
            TerminalDelta::empty(),
            FitnessDelta::zero(),
        );

        // Make base live for the duration of the test so BerthOccupancy can borrow it
        let base = mk_berths();
        let occ0 = BerthOccupancy::new(&base[0]);
        let delta = TerminalDelta::from_updates(vec![(bi(0), occ0)]);
        let non_empty_term = Plan::new_delta(Vec::new(), delta, FitnessDelta::zero());

        // empty + X = X
        assert_eq!(
            empty.clone().concat(non_empty_patch.clone()),
            non_empty_patch
        );
        assert_eq!(empty.clone().concat(non_empty_term.clone()), non_empty_term);

        // X + empty = X
        assert_eq!(
            non_empty_patch.clone().concat(empty.clone()),
            non_empty_patch
        );
        assert_eq!(non_empty_term.clone().concat(empty.clone()), non_empty_term);
    }

    #[test]
    fn test_concat_terminal_delta_last_write_wins_and_order() {
        // Build distinct occupancies for the same berth indices to detect last-write-wins.
        // self: (b0 -> occ0a), (b1 -> occ1a)
        // other: (b1 -> occ1b), (b0 -> occ0b)
        // Combined last writes: for b0 -> occ0b (from other), for b1 -> occ1b (from other)
        // Order of kept updates: [ (b1 -> occ1b), (b0 -> occ0b) ] because last occurrences at positions 3 and 4.
        let base = mk_berths();

        let mut occ0a = BerthOccupancy::new(&base[0]);
        occ0a.occupy(iv(0, 10)).expect("occupy ok");
        let mut occ1a = BerthOccupancy::new(&base[1]);
        occ1a.occupy(iv(10, 20)).expect("occupy ok");

        let mut occ1b = BerthOccupancy::new(&base[1]);
        occ1b.occupy(iv(30, 40)).expect("occupy ok");
        let mut occ0b = BerthOccupancy::new(&base[0]);
        occ0b.occupy(iv(50, 60)).expect("occupy ok");

        // Clone for equality checks later (TerminalDelta takes ownership)
        let occ1b_clone = occ1b.clone();
        let occ0b_clone = occ0b.clone();

        let self_plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::from_updates(vec![(bi(0), occ0a), (bi(1), occ1a)]),
            FitnessDelta::zero(),
        );
        let other_plan = Plan::new_delta(
            Vec::new(),
            TerminalDelta::from_updates(vec![(bi(1), occ1b), (bi(0), occ0b)]),
            FitnessDelta::zero(),
        );

        let out = self_plan.concat(other_plan);
        let updates = out.terminal_delta.updates();
        assert_eq!(
            updates.len(),
            2,
            "one update per berth with last-write-wins"
        );

        // Expected order: b1 then b0
        assert_eq!(updates[0].0, bi(1));
        assert_eq!(updates[0].1, occ1b_clone);
        assert_eq!(updates[1].0, bi(0));
        assert_eq!(updates[1].1, occ0b_clone);
    }

    #[test]
    fn test_concat_sums_fitness_deltas() {
        let p1: Plan<'_, i64> =
            Plan::new_delta(Vec::new(), TerminalDelta::empty(), FitnessDelta::new(5, 1));
        let p2 = Plan::new_delta(
            Vec::new(),
            TerminalDelta::empty(),
            FitnessDelta::new(-2, -3),
        );

        let out = p1.concat(p2);
        assert_eq!(out.fitness_delta.delta_cost, 3);
        assert_eq!(out.fitness_delta.delta_unassigned, -2);
    }

    #[test]
    fn test_concat_patches_last_write_wins_and_order() {
        // self patches: [ (0 -> U), (1 -> A@10) ]
        // other patches: [ (1 -> U), (0 -> A@20), (2 -> U) ]
        // Combined last-writes: (1 -> U at other[0]), (0 -> A@20 at other[1]), (2 -> U at other[2])
        // Expected order of kept patches follows chronological order of those last occurrences:
        // positions: 3,4,5 -> order [ (1 -> U), (0 -> A@20), (2 -> U) ]

        let self_plan = Plan::new_delta(
            vec![
                DecisionVarPatch::new(RequestIndex::new(0), DecisionVar::unassigned()),
                DecisionVarPatch::new(
                    RequestIndex::new(1),
                    DecisionVar::assigned(BerthIndex::new(0), TimePoint::new(10)),
                ),
            ],
            TerminalDelta::empty(),
            FitnessDelta::zero(),
        );
        let other_plan = Plan::new_delta(
            vec![
                DecisionVarPatch::new(RequestIndex::new(1), DecisionVar::unassigned()),
                DecisionVarPatch::new(
                    RequestIndex::new(0),
                    DecisionVar::assigned(BerthIndex::new(0), TimePoint::new(20)),
                ),
                DecisionVarPatch::new(RequestIndex::new(2), DecisionVar::unassigned()),
            ],
            TerminalDelta::empty(),
            FitnessDelta::zero(),
        );

        let out = self_plan.concat(other_plan);
        let patches = &out.decision_var_patches;
        assert_eq!(
            patches.len(),
            3,
            "must keep one per index with last-write-wins"
        );

        // Expected indices in order: 1, 0, 2
        assert_eq!(patches[0].index, RequestIndex::new(1));
        match patches[0].patch {
            DecisionVar::Unassigned => {}
            _ => panic!("expected index 1 to be Unassigned"),
        }

        assert_eq!(patches[1].index, RequestIndex::new(0));
        match patches[1].patch {
            DecisionVar::Assigned(dec) => {
                assert_eq!(dec.berth_index, BerthIndex::new(0));
                assert_eq!(dec.start_time, TimePoint::new(20));
            }
            _ => panic!("expected index 0 to be Assigned@20 on b0"),
        }

        assert_eq!(patches[2].index, RequestIndex::new(2));
        match patches[2].patch {
            DecisionVar::Unassigned => {}
            _ => panic!("expected index 2 to be Unassigned"),
        }

        // Terminal delta is empty here.
        assert!(out.terminal_delta.is_empty());
    }
}
