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

pub trait TemperatureSchedule: Send + Sync {
    fn base_temp(&self, iter: usize) -> f64;
    fn effective_temp(&self, iter: usize, temp_scale: f64) -> f64 {
        (self.base_temp(iter) * temp_scale).clamp(self.min_temp(), self.max_temp())
    }
    fn min_temp(&self) -> f64;
    fn max_temp(&self) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct GeometricSchedule {
    pub initial: f64,
    pub rate: f64,
    pub min_t: f64,
    pub max_t: f64,
}

impl TemperatureSchedule for GeometricSchedule {
    fn base_temp(&self, iter: usize) -> f64 {
        (self.initial * self.rate.powi(iter as i32)).max(self.min_t)
    }
    fn min_temp(&self) -> f64 {
        self.min_t
    }
    fn max_temp(&self) -> f64 {
        self.max_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn base_temp_geometric_decay_with_floor() {
        // initial=100, rate=0.5, min_t=1 → geometric decay, floored at 1
        let sch = GeometricSchedule {
            initial: 100.0,
            rate: 0.5,
            min_t: 1.0,
            max_t: 1.0e9,
        };

        // iter=0: base = initial (since initial > min)
        assert!(approx_eq(sch.base_temp(0), 100.0, 1e-12));

        // iter=1: 100 * 0.5 = 50
        assert!(approx_eq(sch.base_temp(1), 50.0, 1e-12));

        // iter=10: 100 * 0.5^10 = 0.09765625 → floored to 1.0
        assert!(approx_eq(sch.base_temp(10), 1.0, 1e-12));

        // large iter stays at floor
        assert!(approx_eq(sch.base_temp(10_000), 1.0, 0.0));
    }

    #[test]
    fn effective_temp_scales_and_is_clamped_between_min_and_max() {
        // Same decaying schedule; min_t=1, max_t=150
        let sch = GeometricSchedule {
            initial: 100.0,
            rate: 0.5,
            min_t: 1.0,
            max_t: 150.0,
        };

        // iter=1: base=50; scale by 2 → 100 (within [1,150])
        let eff = sch.effective_temp(1, 2.0);
        assert!(approx_eq(eff, 100.0, 1e-12));

        // very large scale clamps to max_t
        let eff_hi = sch.effective_temp(1, 10.0); // base=50 → 500; clamp to 150
        assert!(approx_eq(eff_hi, 150.0, 1e-12));

        // very small scale clamps to min_t
        let eff_lo = sch.effective_temp(1, 1e-6); // base=50 → 5e-5; clamp to 1
        assert!(approx_eq(eff_lo, 1.0, 1e-12));
    }

    #[test]
    fn base_temp_is_nonincreasing_for_rate_below_one_until_floor() {
        let sch = GeometricSchedule {
            initial: 64.0,
            rate: 0.75,
            min_t: 2.0,
            max_t: 1.0e6,
        };

        // Check monotonic nonincreasing for first few iters
        let mut prev = sch.base_temp(0);
        for i in 1..20 {
            let cur = sch.base_temp(i);
            assert!(
                cur <= prev + 1e-15,
                "base_temp increased at iter {}: {} -> {}",
                i,
                prev,
                cur
            );
            prev = cur;
        }

        let mut hit_floor = false;
        for i in 0..100 {
            if approx_eq(sch.base_temp(i), sch.min_temp(), 0.0) {
                hit_floor = true;
                break;
            }
        }
        assert!(hit_floor, "expected to reach min_t (floor) eventually");
    }

    #[test]
    fn base_temp_can_exceed_max_when_rate_above_one_but_effective_is_capped() {
        // For rate>1, base_temp grows without an upper cap (only floored at min_t),
        // but effective_temp must clamp to max_t.
        let sch = GeometricSchedule {
            initial: 1.0,
            rate: 2.0,
            min_t: 0.1,
            max_t: 10.0,
        };

        // base grows: iter=5 → 32.0 (exceeds max_t)
        let base = sch.base_temp(5);
        assert!(
            base > sch.max_temp(),
            "base should exceed max_t when rate>1"
        );

        // effective clamps to max_t regardless of temp_scale=1.0
        let eff = sch.effective_temp(5, 1.0);
        assert!(approx_eq(eff, sch.max_temp(), 1e-12));
    }

    #[test]
    fn min_and_max_accessors_match_config() {
        let sch = GeometricSchedule {
            initial: 10.0,
            rate: 0.9,
            min_t: 0.5,
            max_t: 50.0,
        };
        assert!(approx_eq(sch.min_temp(), 0.5, 0.0));
        assert!(approx_eq(sch.max_temp(), 50.0, 0.0));
    }
}
