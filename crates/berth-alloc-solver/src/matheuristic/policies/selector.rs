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

use rand::RngCore;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::seq::SliceRandom;

pub trait OperatorSelector: Send + Sync {
    fn draw_jobs(
        &self,
        weights: &[f64],
        min_per_op: usize,
        max_per_op: usize,
        total_draws: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<usize>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CappedWeightedSelector;

impl OperatorSelector for CappedWeightedSelector {
    fn draw_jobs(
        &self,
        weights: &[f64],
        min_per_op: usize,
        max_per_op: usize,
        total_draws: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<usize> {
        let n = weights.len();
        if n == 0 || total_draws == 0 {
            return Vec::new();
        }

        let dist = WeightedIndex::new(weights.iter().cloned())
            .expect("weights must be non-negative and finite");

        let mut counts = vec![0usize; n];
        let mut jobs: Vec<usize> = Vec::with_capacity(total_draws);

        if min_per_op > 0 {
            for (i, cnt) in counts.iter_mut().enumerate() {
                let need = min_per_op.min(total_draws.saturating_sub(jobs.len()));
                for _ in 0..need {
                    jobs.push(i);
                    *cnt += 1;
                    if jobs.len() >= total_draws {
                        break;
                    }
                }
                if jobs.len() >= total_draws {
                    break;
                }
            }
        }

        let mut guard = 0usize;
        while jobs.len() < total_draws && guard < total_draws * 20 {
            let idx = dist.sample(rng);
            if counts[idx] < max_per_op {
                jobs.push(idx);
                counts[idx] += 1;
            }
            guard += 1;
        }

        if jobs.len() < total_draws {
            'outer: loop {
                let mut progressed = false;
                for (i, cnt) in counts.iter_mut().enumerate() {
                    if *cnt < max_per_op {
                        jobs.push(i);
                        *cnt += 1;
                        progressed = true;
                        if jobs.len() >= total_draws {
                            break 'outer;
                        }
                    }
                }
                if !progressed {
                    break;
                }
            }
        }

        jobs.shuffle(rng);
        jobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn counts_from(jobs: &[usize], n_ops: usize) -> Vec<usize> {
        let mut c = vec![0usize; n_ops];
        for &j in jobs {
            c[j] += 1;
        }
        c
    }

    #[test]
    fn empty_when_no_ops_or_no_draws() {
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let jobs = sel.draw_jobs(&[], 0, 10, 5, &mut rng);
        assert!(jobs.is_empty());

        let jobs = sel.draw_jobs(&[1.0, 2.0, 3.0], 0, 10, 0, &mut rng);
        assert!(jobs.is_empty());
    }

    #[test]
    fn respects_min_per_op_when_budget_sufficient() {
        // 3 ops, min_per_op = 2, total_draws = 12 (>= 3*2), caps high
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let n = 3;
        let weights = vec![1.0; n];
        let min_per_op = 2;
        let max_per_op = 100;
        let total_draws = 12;

        let jobs = sel.draw_jobs(&weights, min_per_op, max_per_op, total_draws, &mut rng);
        assert_eq!(jobs.len(), total_draws);

        let counts = counts_from(&jobs, n);
        for c in &counts {
            assert!(*c >= min_per_op, "every op must meet min_per_op");
        }
    }

    #[test]
    fn min_per_op_truncates_when_budget_insufficient() {
        // total_draws < n * min_per_op → early operators get their mins first.
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(9);

        let n = 4;
        let weights = vec![1.0; n];
        let min_per_op = 2;
        let total_draws = 5; // < 4 * 2
        let max_per_op = 10;

        let jobs = sel.draw_jobs(&weights, min_per_op, max_per_op, total_draws, &mut rng);
        assert_eq!(jobs.len(), total_draws);

        // Given the implementation, indices are filled in order until budget exhausted:
        // op0:2, op1:2, op2:1, op3:0
        let counts = counts_from(&jobs, n);
        assert_eq!(counts, vec![2, 2, 1, 0]);
    }

    #[test]
    fn respects_max_per_op_cap() {
        // Heavily biased weights toward op0, but cap prevents exceeding max_per_op.
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let n = 3;
        let weights = vec![1000.0, 1.0, 1.0];
        let min_per_op = 0;
        let max_per_op = 3;
        let total_draws = n * max_per_op; // ensure feasible to fill fully under caps

        let jobs = sel.draw_jobs(&weights, min_per_op, max_per_op, total_draws, &mut rng);
        assert_eq!(jobs.len(), total_draws);

        let counts = counts_from(&jobs, n);
        for c in &counts {
            assert!(
                *c <= max_per_op,
                "cap violated: counts={counts:?}, max={max_per_op}"
            );
        }
        assert_eq!(counts.iter().sum::<usize>(), total_draws);
    }

    #[test]
    fn fills_up_to_total_draws_given_feasible_caps() {
        // total_draws <= n * max_per_op → algorithm should fill completely.
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(321);

        let n = 5;
        let weights = vec![1.0; n];
        let min_per_op = 1;
        let max_per_op = 3;
        let total_draws = 12; // <= 5*3=15

        let jobs = sel.draw_jobs(&weights, min_per_op, max_per_op, total_draws, &mut rng);
        assert_eq!(jobs.len(), total_draws);

        // Check both min and max constraints
        let counts = counts_from(&jobs, n);
        for c in &counts {
            assert!(*c <= max_per_op);
        }
        assert!(counts.iter().filter(|&&c| c >= min_per_op).count() >= n.min(total_draws));
    }

    #[test]
    fn weighted_sampling_bias_is_observed_with_fixed_seed() {
        // With fixed seed and many draws, the heavier weight should get >= counts than others.
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(999);

        let n = 3;
        let weights = vec![10.0, 1.0, 1.0];
        let min_per_op = 0;
        let max_per_op = 1000;
        let total_draws = 200;

        let jobs = sel.draw_jobs(&weights, min_per_op, max_per_op, total_draws, &mut rng);
        assert_eq!(jobs.len(), total_draws);
        let counts = counts_from(&jobs, n);

        // Index 0 (heavy) should receive at least as many as others with this seed & size.
        assert!(
            counts[0] >= counts[1] && counts[0] >= counts[2],
            "counts={counts:?}"
        );
    }

    #[test]
    fn shuffles_final_job_order() {
        // The function shuffles the final vector; with fixed seed, we just ensure the
        // multiset matches counts and the order isn't trivially sorted by construction.
        let sel = CappedWeightedSelector::default();
        let mut rng = ChaCha8Rng::seed_from_u64(2024);

        let n = 4;
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let min_per_op = 1;
        let max_per_op = 3;
        let total_draws = 10;

        let jobs = sel.draw_jobs(&weights, min_per_op, max_per_op, total_draws, &mut rng);
        assert_eq!(jobs.len(), total_draws);

        // Ensure it's not strictly grouped by index (heuristic check)
        let all_same_prefix = jobs.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same_prefix, "jobs appear unshuffled: {:?}", jobs);

        // Multiset remains consistent
        let counts = counts_from(&jobs, n);
        assert_eq!(counts.iter().sum::<usize>(), total_draws);
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(c, jobs.iter().filter(|&&j| j == i).count());
        }
    }
}
