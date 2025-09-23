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

use crate::matheuristic::operatorpool::OperatorStats;

pub trait OperatorScorer: Send + Sync {
    fn raw_score(&self, s: &OperatorStats) -> f64;
    fn to_weights(&self, raw: &[f64], tau: f64, explore_frac: f64) -> Vec<f64>;
}

#[derive(Clone, Copy, Debug)]
pub struct LinearScorer {
    pub speed_weight: f64,
    pub success_weight: f64,
    pub min_ns_per_proposal: f64,
    pub bootstrap_success_rate: f64,
}

impl OperatorScorer for LinearScorer {
    fn raw_score(&self, s: &OperatorStats) -> f64 {
        let ns = (s.emwa_gen_ns_per_proposal + s.emwa_eval_ns_per_proposal)
            .max(self.min_ns_per_proposal);
        let speed = 1.0 / ns;
        let succ = if s.attempts > 0 {
            s.accepted as f64 / s.attempts as f64
        } else {
            self.bootstrap_success_rate
        };
        self.speed_weight * speed + self.success_weight * succ
    }

    fn to_weights(&self, scores: &[f64], tau: f64, explore_frac: f64) -> Vec<f64> {
        let t = tau.max(1e-6);
        let maxv = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut w: Vec<f64> = scores.iter().map(|s| ((*s - maxv) / t).exp()).collect();
        if explore_frac > 0.0 {
            let sum: f64 = w.iter().sum();
            let avg = if sum > 0.0 {
                sum / w.len() as f64
            } else {
                1.0 / w.len() as f64
            };
            for wi in &mut w {
                *wi = (1.0 - explore_frac) * *wi + explore_frac * avg;
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn mk_stats(gen_ns: f64, eval_ns: f64, attempts: u64, accepted: u64) -> OperatorStats {
        let mut s = OperatorStats::default();
        s.emwa_gen_ns_per_proposal = gen_ns;
        s.emwa_eval_ns_per_proposal = eval_ns;
        s.attempts = attempts;
        s.accepted = accepted;
        s
    }

    #[test]
    fn raw_score_uses_bootstrap_when_no_attempts() {
        let scorer = LinearScorer {
            speed_weight: 2.0,
            success_weight: 3.0,
            min_ns_per_proposal: 10.0,
            bootstrap_success_rate: 0.4,
        };
        // ns = max(gen+eval, min_ns) = max(3+4, 10) = 10 → speed=0.1
        let s = mk_stats(3.0, 4.0, 0, 0);
        let got = scorer.raw_score(&s);
        let expected = 2.0 * 0.1 + 3.0 * 0.4; // = 0.2 + 1.2 = 1.4
        assert!(
            approx_eq(got, expected, 1e-12),
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn raw_score_uses_observed_success_when_attempts_present() {
        let scorer = LinearScorer {
            speed_weight: 1.5,
            success_weight: 2.5,
            min_ns_per_proposal: 1.0,
            bootstrap_success_rate: 0.0, // should be ignored
        };
        // ns = max(2 + 3, 1) = 5 → speed = 0.2
        // success = 3 / 10 = 0.3
        let s = mk_stats(2.0, 3.0, 10, 3);
        let got = scorer.raw_score(&s);
        let expected = 1.5 * 0.2 + 2.5 * 0.3; // 0.3 + 0.75 = 1.05
        assert!(
            approx_eq(got, expected, 1e-12),
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn raw_score_respects_min_ns_floor() {
        let scorer = LinearScorer {
            speed_weight: 1.0,
            success_weight: 0.0,
            min_ns_per_proposal: 100.0,
            bootstrap_success_rate: 0.0,
        };
        // gen+eval = 1.0, but min_ns=100 → speed=1/100
        let s = mk_stats(0.4, 0.6, 0, 0);
        let got = scorer.raw_score(&s);
        assert!(
            approx_eq(got, 0.01, 1e-12),
            "expected speed=0.01, got {got}"
        );
    }

    #[test]
    fn to_weights_softmax_basic_and_ordering() {
        let scorer = LinearScorer {
            speed_weight: 0.0,
            success_weight: 1.0,
            min_ns_per_proposal: 1.0,
            bootstrap_success_rate: 0.0,
        };
        // scores chosen directly
        let scores = vec![2.0, 1.0, -1.0];
        let w = scorer.to_weights(&scores, /*tau*/ 1.0, /*explore_frac*/ 0.0);
        assert_eq!(w.len(), 3);
        // preserve ordering: higher score → larger weight
        assert!(
            w[0] > w[1] && w[1] > w[2],
            "weights not ordered by scores: {:?}",
            w
        );
        // positive weights
        assert!(w.iter().all(|x| *x > 0.0 && x.is_finite()));
    }

    #[test]
    fn to_weights_is_shift_invariant() {
        let sc = LinearScorer {
            speed_weight: 0.0,
            success_weight: 1.0,
            min_ns_per_proposal: 1.0,
            bootstrap_success_rate: 0.0,
        };
        let scores = vec![1.0, 0.0, -3.0, 2.0];
        let w1 = sc.to_weights(&scores, 0.7, 0.0);

        // Add a constant to all scores; softmax should be identical
        let c = 5.1234;
        let shifted: Vec<f64> = scores.iter().map(|x| x + c).collect();
        let w2 = sc.to_weights(&shifted, 0.7, 0.0);

        assert_eq!(w1.len(), w2.len());
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-12),
                "shift invariance broken: {a} vs {b}"
            );
        }
    }

    #[test]
    fn to_weights_tau_is_clamped_and_not_nan() {
        let sc = LinearScorer {
            speed_weight: 0.0,
            success_weight: 1.0,
            min_ns_per_proposal: 1.0,
            bootstrap_success_rate: 0.0,
        };
        let scores = vec![3.0, 2.0, 1.0];

        // tau <= 0 should be clamped to 1e-6 internally
        for tau in [0.0, -1.0] {
            let w = sc.to_weights(&scores, tau, 0.0);

            // Under extreme peaking, non-max weights may underflow to 0. That's OK.
            assert!(
                w.iter().all(|x| x.is_finite() && *x >= 0.0),
                "weights must be finite & non-negative: {:?}",
                w
            );

            // At least one weight (the max) must be strictly positive.
            assert!(
                w.iter().any(|x| *x > 0.0),
                "at least one weight should be > 0: {:?}",
                w
            );

            // With tiny tau, the largest score dominates.
            assert!(
                w[0] >= w[1] && w[0] >= w[2],
                "dominance expected at tiny tau: {:?}",
                w
            );
            assert!(w[0] > 0.0, "top weight should be positive");
        }
    }

    #[test]
    fn to_weights_explore_frac_blends_toward_uniform() {
        let sc = LinearScorer {
            speed_weight: 0.0,
            success_weight: 1.0,
            min_ns_per_proposal: 1.0,
            bootstrap_success_rate: 0.0,
        };
        let scores = vec![5.0, 0.0, 0.0, 0.0];

        // No exploration: peaked at index 0
        let w0 = sc.to_weights(&scores, 0.5, 0.0);
        let s0: f64 = w0.iter().sum();
        let p0: Vec<f64> = w0.iter().map(|x| x / s0).collect();
        assert!(
            p0[0] > 0.5,
            "should be peaked at index 0 without exploration"
        );

        // Full exploration: uniform distribution, regardless of scores
        let w1 = sc.to_weights(&scores, 0.5, 1.0);
        let s1: f64 = w1.iter().sum();
        let p1: Vec<f64> = w1.iter().map(|x| x / s1).collect();
        for pi in p1 {
            assert!(approx_eq(pi, 1.0 / 4.0, 1e-12));
        }

        // Partial exploration should move the distribution toward uniform
        let w_half = sc.to_weights(&scores, 0.5, 0.5);
        let s_half: f64 = w_half.iter().sum();
        let p_half: Vec<f64> = w_half.iter().map(|x| x / s_half).collect();
        assert!(
            p_half[0] < p0[0] && p_half[0] > 0.25,
            "should be between peaked and uniform"
        );
    }

    #[test]
    fn to_weights_length_matches_input() {
        let sc = LinearScorer {
            speed_weight: 1.0,
            success_weight: 1.0,
            min_ns_per_proposal: 1.0,
            bootstrap_success_rate: 0.3,
        };
        for n in [1usize, 2, 5, 10] {
            let scores: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let w = sc.to_weights(&scores, 1.0, 0.0);
            assert_eq!(w.len(), n);
        }
    }
}
