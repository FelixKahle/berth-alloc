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

pub trait ReheatPolicy: Send + Sync {
    /// Returns (new_temp_scale, explore_boost_until_iters, reset_operator_stats)
    fn on_stagnation(
        &self,
        iters_without_repair: usize,
        temp_scale: f64,
    ) -> (f64, Option<usize>, bool);
}

#[derive(Clone, Copy, Debug)]
pub struct DefaultReheat {
    pub iter_threshold: usize,
    pub reheat_multiplier: f64,
    pub explore_boost_iters: usize,
    pub reset_operator_stats_on_reheat: bool,
}

impl ReheatPolicy for DefaultReheat {
    fn on_stagnation(&self, iters: usize, temp_scale: f64) -> (f64, Option<usize>, bool) {
        if iters >= self.iter_threshold {
            (
                temp_scale * self.reheat_multiplier,
                Some(self.explore_boost_iters),
                self.reset_operator_stats_on_reheat,
            )
        } else {
            (temp_scale, None, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn below_threshold_no_change() {
        let rh = DefaultReheat {
            iter_threshold: 10,
            reheat_multiplier: 1.5,
            explore_boost_iters: 7,
            reset_operator_stats_on_reheat: true,
        };

        let (new_scale, until, reset) = rh.on_stagnation(9, 2.0);
        assert!(approx_eq(new_scale, 2.0), "temp_scale should be unchanged");
        assert!(until.is_none(), "no explore boost below threshold");
        assert!(!reset, "no reset below threshold");
    }

    #[test]
    fn at_threshold_triggers_reheat() {
        let rh = DefaultReheat {
            iter_threshold: 10,
            reheat_multiplier: 1.5,
            explore_boost_iters: 7,
            reset_operator_stats_on_reheat: true,
        };

        let (new_scale, until, reset) = rh.on_stagnation(10, 2.0);
        assert!(approx_eq(new_scale, 3.0), "2.0 * 1.5 = 3.0");
        assert_eq!(until, Some(7));
        assert!(reset, "reset flag is passed through");
    }

    #[test]
    fn above_threshold_also_triggers_reheat() {
        let rh = DefaultReheat {
            iter_threshold: 5,
            reheat_multiplier: 2.0,
            explore_boost_iters: 3,
            reset_operator_stats_on_reheat: false,
        };

        let (new_scale, until, reset) = rh.on_stagnation(8, 1.25);
        assert!(approx_eq(new_scale, 2.5), "1.25 * 2.0 = 2.5");
        assert_eq!(until, Some(3));
        assert!(!reset);
    }

    #[test]
    fn repeated_triggers_multiply_scale() {
        let rh = DefaultReheat {
            iter_threshold: 2,
            reheat_multiplier: 1.2,
            explore_boost_iters: 4,
            reset_operator_stats_on_reheat: true,
        };

        // First trigger
        let (s1, _, _) = rh.on_stagnation(2, 1.0);
        assert!(approx_eq(s1, 1.2));

        // Second trigger (e.g., still stagnating later)
        let (s2, _, _) = rh.on_stagnation(3, s1);
        assert!(approx_eq(s2, 1.2 * 1.2));
    }

    #[test]
    fn explore_boost_window_is_exact_value() {
        let rh = DefaultReheat {
            iter_threshold: 3,
            reheat_multiplier: 1.1,
            explore_boost_iters: 12,
            reset_operator_stats_on_reheat: false,
        };

        let (_, until, _) = rh.on_stagnation(3, 5.0);
        assert_eq!(until, Some(12));
    }
}
