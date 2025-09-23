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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepairTrend {
    Better,
    Worse,
    Unchanged,
}

impl std::fmt::Display for RepairTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepairTrend::Better => write!(f, "Better"),
            RepairTrend::Worse => write!(f, "Worse"),
            RepairTrend::Unchanged => write!(f, "Unchanged"),
        }
    }
}

pub trait PenaltyModel: Send + Sync {
    fn score(&self, cost_f: f64, unassigned: usize, lambda: f64) -> f64;
    fn update_lambda(&self, lambda: f64, trend: RepairTrend) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct DefaultPenalty {
    pub use_penalty: bool,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub lambda_decay: f64,
    pub lambda_growth: f64,
}

impl PenaltyModel for DefaultPenalty {
    fn score(&self, cost_f: f64, u: usize, lambda: f64) -> f64 {
        if self.use_penalty {
            cost_f + lambda * (u as f64)
        } else {
            cost_f
        }
    }

    fn update_lambda(&self, lambda: f64, trend: RepairTrend) -> f64 {
        let l = match trend {
            RepairTrend::Better => lambda * self.lambda_decay,
            RepairTrend::Worse => lambda * self.lambda_growth,
            RepairTrend::Unchanged => lambda,
        };
        l.clamp(self.lambda_min, self.lambda_max)
    }
}

#[cfg(test)]
mod tests {
    use super::super::penalty::{PenaltyModel, RepairTrend};
    use crate::matheuristic::policies::energy::{EnergyModel, RelativeEnergy};

    /// Simple penalty: score = cost + lambda * unassigned
    #[derive(Clone, Copy, Debug, Default)]
    struct LinearPenalty;

    impl PenaltyModel for LinearPenalty {
        fn score(&self, cost_f: f64, unassigned: usize, lambda: f64) -> f64 {
            cost_f + lambda * (unassigned as f64)
        }
        fn update_lambda(&self, lambda: f64, _trend: RepairTrend) -> f64 {
            lambda
        }
    }

    fn e(
        cur_cost_f: f64,
        cur_u: usize,
        delta_f: f64,
        new_u: usize,
        lambda: f64,
        jitter: f64,
        jitter_sample: f64,
    ) -> f64 {
        EnergyModel::<f64>::energy(
            &RelativeEnergy,
            cur_cost_f,
            cur_u,
            delta_f,
            new_u,
            &LinearPenalty,
            lambda,
            jitter,
            jitter_sample,
        )
    }

    #[test]
    fn no_penalty_no_jitter_reduces_to_delta() {
        // lambda = 0, jitter = 0  => energy == delta_f
        let cur_cost = 1000.0;
        let cur_u = 5usize;
        let delta = -37.5;
        let new_u = 5usize; // same infeasibility
        let energy = e(cur_cost, cur_u, delta, new_u, 0.0, 0.0, 0.0);
        assert!(
            (energy - delta).abs() < 1e-12,
            "energy={energy}, delta={delta}"
        );
    }

    #[test]
    fn penalty_contributes_lambda_times_delta_unassigned() {
        // energy = (cur_cost + delta + lambda*new_u) - (cur_cost + lambda*cur_u)
        //        = delta + lambda*(new_u - cur_u)
        let delta = -10.0;
        let cur_u = 7usize;
        let new_u = 10usize; // worse by 3
        let lambda = 2.5;
        let energy = e(500.0, cur_u, delta, new_u, lambda, 0.0, 0.0);
        let expected = delta + lambda * ((new_u as f64) - (cur_u as f64));
        assert!(
            (energy - expected).abs() < 1e-12,
            "energy={energy}, expected={expected}"
        );
    }

    #[test]
    fn improving_feasibility_can_outweigh_positive_delta() {
        // Slightly worse cost (+1.0), but reduces unassigned by 2 with lambda=1.0:
        // energy = +1.0 + 1.0*(new_u - cur_u) = 1.0 + (3-5) = -1.0 < 0
        let energy = e(0.0, 5, 1.0, 3, 1.0, 0.0, 0.0);
        assert!(energy < 0.0, "energy should be negative, got {energy}");
    }

    #[test]
    fn jitter_is_zero_mean_and_bounded_by_half_width() {
        // energy_base without jitter
        let base = e(100.0, 4, -2.0, 4, 0.0, 0.0, 0.0);
        // jitter adds (jitter_sample - 0.5) * jitter
        let jitter = 0.8;

        let e_min = e(100.0, 4, -2.0, 4, 0.0, jitter, 0.0); // adds (-0.5)*0.8 = -0.4
        let e_mid = e(100.0, 4, -2.0, 4, 0.0, jitter, 0.5); // adds 0
        let e_max = e(100.0, 4, -2.0, 4, 0.0, jitter, 1.0); // adds (+0.5)*0.8 = +0.4

        assert!((e_mid - base).abs() < 1e-12);
        assert!(
            (e_min - (base - 0.4)).abs() < 1e-12,
            "e_min={e_min}, base-0.4={}",
            base - 0.4
        );
        assert!(
            (e_max - (base + 0.4)).abs() < 1e-12,
            "e_max={e_max}, base+0.4={}",
            base + 0.4
        );
    }

    #[test]
    fn monotone_in_delta_and_unassigned() {
        // energy = delta + lambda*(new_u - cur_u) + jitter_term
        let cur_cost = 0.0;
        let cur_u = 5usize;
        let lambda = 3.0;

        // Fix jitter term to zero
        let j = 0.0;
        let js = 0.5;

        // If we increase delta by +d, energy must increase by +d
        let e1 = e(cur_cost, cur_u, -4.0, 5, lambda, j, js);
        let e2 = e(cur_cost, cur_u, -3.0, 5, lambda, j, js);
        assert!(
            e2 > e1,
            "energy should increase with worse delta: e2={e2}, e1={e1}"
        );

        // If we increase new_u by +1, energy must increase by +lambda
        let e3 = e(cur_cost, cur_u, -4.0, 6, lambda, j, js);
        assert!(
            (e3 - (e1 + lambda)).abs() < 1e-12,
            "e3={e3}, e1+lambda={}",
            e1 + lambda
        );
    }

    #[test]
    fn exact_identity_against_definition() {
        // Compare with direct formula: (after - before)
        let cur_cost = 123.4;
        let cur_u = 9usize;
        let delta = -7.6;
        let new_u = 11usize;
        let lambda = 0.75;
        let jitter = 0.2;
        let js = 0.9;

        let before = LinearPenalty.score(cur_cost, cur_u, lambda);
        let after = LinearPenalty.score(cur_cost + delta, new_u, lambda);
        let expected = (after - before) + (js - 0.5) * jitter;

        let got = e(cur_cost, cur_u, delta, new_u, lambda, jitter, js);
        assert!(
            (got - expected).abs() < 1e-12,
            "got={got}, expected={expected}"
        );
    }
}
