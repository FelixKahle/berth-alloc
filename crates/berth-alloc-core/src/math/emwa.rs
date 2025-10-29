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

use num_traits::{Float, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul};

/// An Exponentially Weighted Moving Average (EWMA) calculator.
///
/// Generic over a float type `F` for the alpha value and a state type `T`
/// for the value being averaged. This version assumes that `T` can be multiplied
/// by `F` to produce an `F`, which is then converted back into `T`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ewma<F, T> {
    alpha: F,
    value: Option<T>,
}

/// Error type for an invalid alpha value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidAlphaError<F> {
    value: F,
}

impl<F: Float + Debug> InvalidAlphaError<F> {
    pub fn new(value: F) -> Self {
        Self { value }
    }
    pub fn value(&self) -> F {
        self.value
    }
}

impl<F: Float + Debug> std::fmt::Display for InvalidAlphaError<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid alpha value {:?}. Allowed range is 0.0..=1.0.",
            self.value
        )
    }
}

impl<F: Float + Debug> std::error::Error for InvalidAlphaError<F> {}

impl<F, T> Ewma<F, T>
where
    F: Float + Debug + Into<T>,
    T: Copy + Add<Output = T> + Mul<F, Output = F> + Zero, // T * F -> F
{
    /// Creates a new EWMA with a given smoothing factor `alpha`.
    #[inline]
    pub fn new(alpha: F) -> Result<Self, InvalidAlphaError<F>> {
        if !alpha.is_finite() || alpha <= F::zero() || alpha > F::one() {
            return Err(InvalidAlphaError::new(alpha));
        }
        Ok(Self { alpha, value: None })
    }

    #[inline]
    pub fn from_half_life(half_life_steps: F) -> Result<Self, InvalidAlphaError<F>> {
        if !half_life_steps.is_finite() || half_life_steps <= F::zero() {
            return Err(InvalidAlphaError::new(F::nan()));
        }

        let half = F::from(0.5).unwrap();
        let one = F::one();
        let alpha = one - half.powf(one / half_life_steps);
        Self::new(alpha)
    }

    #[inline]
    pub fn from_time_constant(tau_steps: F) -> Result<Self, InvalidAlphaError<F>> {
        if !tau_steps.is_finite() || tau_steps <= F::zero() {
            return Err(InvalidAlphaError::new(F::nan()));
        }

        let one = F::one();
        let alpha = one - (-one / tau_steps).exp();
        Self::new(alpha)
    }

    #[inline]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    #[inline]
    pub fn set_alpha(&mut self, alpha: F) -> Result<(), InvalidAlphaError<F>> {
        if !alpha.is_finite() || alpha <= F::zero() || alpha > F::one() {
            return Err(InvalidAlphaError::new(alpha));
        }
        self.alpha = alpha;
        Ok(())
    }

    #[inline]
    pub fn initialized(&self) -> bool {
        self.value.is_some()
    }

    #[inline]
    pub fn value(&self) -> Option<T> {
        self.value
    }

    #[inline]
    pub fn reset(&mut self) {
        self.value = None;
    }

    /// Observes a single value of type `T` and updates the EWMA.
    pub fn observe(&mut self, x: T) -> T {
        let new_value = match self.value {
            None => x,
            Some(current_value) => {
                let one_minus_alpha = F::one() - self.alpha;
                let term1: T = (x * self.alpha).into();
                let term2: T = (current_value * one_minus_alpha).into();
                term1 + term2
            }
        };
        self.value = Some(new_value);
        new_value
    }

    /// Observes a value `T` as if it occurred `steps` times in a row.
    pub fn observe_n(&mut self, x: T, steps: u64) -> T {
        if steps == 0 {
            return self.value.unwrap_or_else(T::zero);
        }

        let new_value = match self.value {
            None => x,
            Some(current_value) => {
                let one_minus_alpha = F::one() - self.alpha;
                let eff_alpha = F::one() - one_minus_alpha.powi(steps as i32);
                let eff_one_minus_alpha = F::one() - eff_alpha;

                let term1: T = (x * eff_alpha).into();
                let term2: T = (current_value * eff_one_minus_alpha).into();
                term1 + term2
            }
        };
        self.value = Some(new_value);
        new_value
    }

    /// Advances the EWMA by `steps` without a new observation (pure decay).
    pub fn decay(&mut self, steps: u64) -> T {
        if steps > 0 {
            self.value = self.value.map(|v| {
                let factor = (F::one() - self.alpha).powi(steps as i32);
                (v * factor).into()
            });
        }
        self.value.unwrap_or_else(T::zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Sub;

    // Type alias for convenience in tests. This is the key to fixing the test compile errors.
    type TestEwma = Ewma<f64, f64>;

    #[test]
    fn test_invalid_alpha_rejected() {
        assert!(TestEwma::new(0.0).is_err());
        assert!(TestEwma::new(1.1).is_err());
        assert!(TestEwma::new(f64::NAN).is_err());
    }

    #[test]
    fn test_basic_update() {
        let mut ew = TestEwma::new(0.5).unwrap();
        assert_eq!(ew.value(), None);
        assert_eq!(ew.observe(10.0), 10.0); // init
        let next_val = ew.observe(12.0);
        assert!(next_val.sub(11.0).abs() < 1e-12);
        assert_eq!(ew.initialized(), true);
    }

    #[test]
    fn test_multi_step_equivalent() {
        let mut a = TestEwma::new(0.2).unwrap();
        let mut b = TestEwma::new(0.2).unwrap();
        a.observe(0.0);
        b.observe(0.0);
        a.observe(10.0);
        a.observe(10.0);
        a.observe(10.0);
        b.observe_n(10.0, 3);
        assert!((a.value().unwrap() - b.value().unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_decay_only() {
        let mut ew = TestEwma::new(0.25).unwrap();
        ew.observe(8.0);
        let v = ew.decay(2);
        assert!((v - (0.75 * 0.75 * 8.0)).abs() < 1e-12);
    }

    #[test]
    fn test_half_life_constructor() {
        let ew = TestEwma::from_half_life(10.0).unwrap();
        assert!(ew.alpha() > 0.0 && ew.alpha() <= 1.0);
    }

    #[test]
    fn test_time_constant_constructor() {
        let ew = TestEwma::from_time_constant(5.0).unwrap();
        assert!(ew.alpha() > 0.0 && ew.alpha() <= 1.0);
    }

    #[test]
    fn test_alpha_boundary_values() {
        assert!(TestEwma::new(1.0).is_ok());
        assert!(TestEwma::new(f64::EPSILON).is_ok());
        assert!(TestEwma::new(-0.1).is_err());
    }

    #[test]
    fn test_set_alpha() {
        let mut ew = TestEwma::new(0.5).unwrap();
        assert!(ew.set_alpha(0.3).is_ok());
        assert_eq!(ew.alpha(), 0.3);
        assert!(ew.set_alpha(1.5).is_err());
        assert_eq!(ew.alpha(), 0.3);
    }

    #[test]
    fn test_reset_functionality() {
        let mut ew = TestEwma::new(0.4).unwrap();
        ew.observe(15.0);
        assert!(ew.initialized());
        ew.reset();
        assert!(!ew.initialized());
        assert_eq!(ew.value(), None);
    }

    #[test]
    fn test_observe_n_zero_steps() {
        let mut ew = TestEwma::new(0.3).unwrap();
        assert_eq!(ew.observe_n(10.0, 0), 0.0);
        assert!(!ew.initialized());
        ew.observe(20.0);
        assert_eq!(ew.observe_n(10.0, 0), 20.0);
        assert_eq!(ew.value(), Some(20.0));
    }

    #[test]
    fn test_decay_zero_steps() {
        let mut ew = TestEwma::new(0.2).unwrap();
        assert_eq!(ew.decay(0), 0.0);
        ew.observe(30.0);
        assert_eq!(ew.decay(0), 30.0);
    }

    #[test]
    fn test_decay_uninitialized() {
        let mut ew = TestEwma::new(0.1).unwrap();
        assert_eq!(ew.decay(5), 0.0);
    }

    #[test]
    fn test_observe_n_uninitialized() {
        let mut ew = TestEwma::new(0.6).unwrap();
        let result = ew.observe_n(100.0, 3);
        assert_eq!(result, 100.0);
        assert_eq!(ew.value(), Some(100.0));
    }

    #[test]
    fn test_half_life_invalid_params_rejected() {
        assert!(TestEwma::from_half_life(0.0).is_err());
        assert!(TestEwma::from_half_life(f64::NAN).is_err());
        assert!(TestEwma::from_half_life(f64::NEG_INFINITY).is_err());
        assert!(TestEwma::from_half_life(-1.0).is_err());
    }

    #[test]
    fn test_time_constant_invalid_params_rejected() {
        assert!(TestEwma::from_time_constant(0.0).is_err());
        assert!(TestEwma::from_time_constant(f64::NAN).is_err());
        assert!(TestEwma::from_time_constant(f64::INFINITY).is_err());
        assert!(TestEwma::from_time_constant(-1.0).is_err());
    }
}
