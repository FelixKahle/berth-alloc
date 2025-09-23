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

use crate::matheuristic::support::acceptance::acceptance_prob;
use berth_alloc_core::prelude::Cost;
use rand::RngCore;

// Local alias to avoid repeating the long path in the default type parameter.
type DefaultPlan<'p, T> = crate::framework::planning::Plan<'p, T>;

#[derive(Debug, Clone)]
pub struct Candidate<'p, T: Ord + Copy, P = DefaultPlan<'p, T>> {
    pub op_idx: usize,
    pub plan: P,
    pub delta: Cost,
    pub unassigned: usize,
    pub feasible: bool,
    pub energy: f64,
    _phantom: std::marker::PhantomData<&'p T>,
}

impl<'p, T: Ord + Copy, P> Candidate<'p, T, P> {
    pub fn new(
        op_idx: usize,
        plan: P,
        delta: Cost,
        unassigned: usize,
        feasible: bool,
        energy: f64,
    ) -> Self {
        Self {
            op_idx,
            plan,
            delta,
            unassigned,
            feasible,
            energy,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait CandidateReducer<T: Ord + Copy>: Send + Sync {
    fn pick<'p>(
        &self,
        a: Option<Candidate<'p, T>>,
        b: Option<Candidate<'p, T>>,
        temp: f64,
        rng: &mut dyn RngCore,
    ) -> Option<Candidate<'p, T>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimulatedAnnealingReducer;

impl SimulatedAnnealingReducer {
    /// Generic helper the trait impl delegates to. Tests can call this with a
    /// dummy plan payload type (e.g., `()`), while production uses the default.
    pub fn pick_generic<'p, T: Copy + Ord, P>(
        &self,
        a: Option<Candidate<'p, T, P>>,
        b: Option<Candidate<'p, T, P>>,
        temp: f64,
        rng: &mut dyn RngCore,
    ) -> Option<Candidate<'p, T, P>> {
        match (a, b) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x),
            (Some(x), Some(y)) => {
                // 1) Feasibility strictly dominates
                if x.feasible != y.feasible {
                    return Some(if y.feasible { y } else { x });
                }
                // 2) Fewer unassigned strictly dominates
                if x.unassigned != y.unassigned {
                    return Some(if y.unassigned < x.unassigned { y } else { x });
                }
                // 3) SA on energy (lower is better), with tie via coin flip
                let d = y.energy - x.energy;
                if d.abs() < 1e-15 {
                    if (rng.next_u64() & 1) == 0 {
                        Some(x)
                    } else {
                        Some(y)
                    }
                } else {
                    let p = acceptance_prob(d, temp);
                    let toss = (rng.next_u64() as f64) / (u64::MAX as f64);
                    if p > 0.0 && toss < p {
                        Some(y)
                    } else {
                        Some(x)
                    }
                }
            }
        }
    }
}

impl<T: Copy + Ord> CandidateReducer<T> for SimulatedAnnealingReducer {
    fn pick<'p>(
        &self,
        a: Option<Candidate<'p, T>>,
        b: Option<Candidate<'p, T>>,
        temp: f64,
        rng: &mut dyn RngCore,
    ) -> Option<Candidate<'p, T>> {
        // Delegate to the generic helper using the default `Plan` payload.
        self.pick_generic::<T, DefaultPlan<'p, T>>(a, b, temp, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    /// Deterministic RNG implementing only the required `RngCore` members.
    struct FixedRng {
        v: u64,
    }
    impl FixedRng {
        fn new(v: u64) -> Self {
            Self { v }
        }
    }
    impl rand::RngCore for FixedRng {
        fn next_u32(&mut self) -> u32 {
            self.v as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.v
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let bytes = self.v.to_le_bytes();
            for (i, b) in dest.iter_mut().enumerate() {
                *b = bytes[i % 8];
            }
        }
    }

    // Candidate builder using unit plan payload `()`, so we don't need a real Plan.
    fn cand<'p, T: Copy + Ord>(
        op_idx: usize,
        energy: f64,
        unassigned: usize,
        feasible: bool,
        delta: Cost,
    ) -> Candidate<'p, T, ()> {
        Candidate {
            op_idx,
            plan: (),
            delta,
            unassigned,
            feasible,
            energy,
            _phantom: std::marker::PhantomData,
        }
    }

    #[test]
    fn feasibility_strictly_dominates() {
        type TT = i64;

        let x = cand::<TT>(
            0,
            /*energy*/ 10.0,
            /*u*/ 5,
            /*feasible*/ false,
            Cost::zero(),
        );
        let y = cand::<TT>(
            1,
            /*energy*/ 99.0,
            /*u*/ 5,
            /*feasible*/ true,
            Cost::zero(),
        );

        let mut rng = FixedRng::new(0);
        let red = SimulatedAnnealingReducer::default();

        let got = red
            .pick_generic::<TT, ()>(Some(x), Some(y), /*temp*/ 1.0, &mut rng)
            .unwrap();
        assert!(got.feasible);
        assert_eq!(got.op_idx, 1);
    }

    #[test]
    fn fewer_unassigned_dominates_when_feasibility_equal() {
        type TT = i64;

        // both infeasible; y has fewer unassigned
        let x = cand::<TT>(0, 1.0, 7, false, Cost::zero());
        let y = cand::<TT>(1, 2.0, 5, false, Cost::zero());

        let mut rng = FixedRng::new(0);
        let red = SimulatedAnnealingReducer::default();

        let got = red
            .pick_generic::<TT, ()>(Some(x), Some(y), 1.0, &mut rng)
            .unwrap();
        assert_eq!(got.unassigned, 5);
        assert_eq!(got.op_idx, 1);

        // both feasible; y has fewer unassigned (0 vs 1)
        let x2 = cand::<TT>(2, 1.0, 1, true, Cost::zero());
        let y2 = cand::<TT>(3, 2.0, 0, true, Cost::zero());

        let got2 = red
            .pick_generic::<TT, ()>(Some(x2), Some(y2), 1.0, &mut rng)
            .unwrap();
        assert_eq!(got2.unassigned, 0);
        assert_eq!(got2.op_idx, 3);
    }

    #[test]
    fn energy_tie_uses_rng_low_bit_coinflip() {
        type TT = i64;

        // equal feasibility and unassigned, equal energy => coin flip via next_u64() & 1
        let x = cand::<TT>(10, 42.0, 5, false, Cost::zero());
        let y = cand::<TT>(11, 42.0, 5, false, Cost::zero());
        let red = SimulatedAnnealingReducer::default();

        // next_u64() even -> choose x
        let mut rng_even = FixedRng::new(0);
        let got_even = red
            .pick_generic::<TT, ()>(Some(x.clone()), Some(y.clone()), 1.0, &mut rng_even)
            .unwrap();
        assert_eq!(got_even.op_idx, 10);

        // next_u64() odd -> choose y
        let mut rng_odd = FixedRng::new(1);
        let got_odd = red
            .pick_generic::<TT, ()>(Some(x), Some(y), 1.0, &mut rng_odd)
            .unwrap();
        assert_eq!(got_odd.op_idx, 11);
    }

    #[test]
    fn strictly_lower_energy_is_always_accepted_when_delta_e_negative() {
        type TT = i64;

        // y has lower energy than x => d = y.energy - x.energy < 0 ⇒ p = 1
        let x = cand::<TT>(0, 10.0, 5, false, Cost::zero());
        let y = cand::<TT>(1, 9.0, 5, false, Cost::zero());

        let mut rng = FixedRng::new(123);
        let red = SimulatedAnnealingReducer::default();

        let got = red
            .pick_generic::<TT, ()>(Some(x), Some(y), /*temp*/ 0.5, &mut rng)
            .unwrap();
        assert_eq!(got.op_idx, 1);
    }

    #[test]
    fn overwhelmingly_positive_delta_energy_with_tiny_temp_is_rejected() {
        type TT = i64;

        // d is huge positive; temp extremely small ⇒ exp(-d/temp) ~ 0 ⇒ keep x
        let x = cand::<TT>(0, 0.0, 5, false, Cost::zero());
        let y = cand::<TT>(1, 1.0e9, 5, false, Cost::zero());

        let mut rng = FixedRng::new(u64::MAX); // toss irrelevant when p == 0
        let red = SimulatedAnnealingReducer::default();

        let got = red
            .pick_generic::<TT, ()>(Some(x), Some(y), /*temp*/ 1e-12, &mut rng)
            .unwrap();
        assert_eq!(got.op_idx, 0);
    }

    #[test]
    fn none_and_singleton_cases_pass_through() {
        type TT = i64;

        let red = SimulatedAnnealingReducer::default();
        let mut rng = FixedRng::new(0);

        // (None, None)
        assert!(
            red.pick_generic::<TT, ()>(None, None, 1.0, &mut rng)
                .is_none()
        );

        // (Some, None)
        let a = cand::<TT>(7, 0.0, 0, true, Cost::zero());
        let got = red
            .pick_generic::<TT, ()>(Some(a), None, 1.0, &mut rng)
            .unwrap();
        assert_eq!(got.op_idx, 7);

        // (None, Some)
        let b = cand::<TT>(8, 0.0, 0, true, Cost::zero());
        let got2 = red
            .pick_generic::<TT, ()>(None, Some(b), 1.0, &mut rng)
            .unwrap();
        assert_eq!(got2.op_idx, 8);
    }
}
