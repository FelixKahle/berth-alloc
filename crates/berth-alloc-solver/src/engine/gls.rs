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

use crate::state::chain_set::index::NodeIndex;
use fxhash::FxHashMap;

/// Guided Local Search penalties on features/arcs.
#[derive(Debug)]
pub struct Gls {
    // penalty(u,v) for "arc" u->v in the *current* solution graph.
    penalties: FxHashMap<(usize, usize), i32>,
    /// α controls strength of penalties added to base arc costs (same units as Cost).
    pub alpha: i64,
    /// geometric decay factor ∈ (0,1], applied at discrete times (e.g., on stagnation).
    decay: f64,
}

impl Gls {
    /// Create with given alpha; decay defaults to 1.0 (no decay).
    pub fn new(alpha: i64) -> Self {
        Self {
            penalties: FxHashMap::default(),
            alpha,
            decay: 1.0,
        }
    }

    #[inline]
    pub fn set_alpha(&mut self, alpha: i64) {
        self.alpha = alpha.max(0);
    }

    /// Set geometric decay ∈ (0,1]. Values <1 cause penalties to slowly fade.
    #[inline]
    pub fn set_decay(&mut self, decay: f64) {
        self.decay = if decay.is_finite() && decay > 0.0 {
            decay.min(1.0)
        } else {
            1.0
        };
    }

    /// Read current penalty p(u,v).
    #[inline]
    pub fn p(&self, u: NodeIndex, v: NodeIndex) -> i32 {
        *self.penalties.get(&(u.get(), v.get())).unwrap_or(&0)
    }

    /// Increase penalty by `by` (≥1).
    #[inline]
    pub fn bump(&mut self, u: NodeIndex, v: NodeIndex, by: i32) {
        *self.penalties.entry((u.get(), v.get())).or_insert(0) += by.max(1);
    }

    /// Apply geometric decay to all penalties (rounding down).
    pub fn decay_all(&mut self) {
        if (self.decay - 1.0).abs() < f64::EPSILON {
            return;
        }
        if self.penalties.is_empty() {
            return;
        }
        for val in self.penalties.values_mut() {
            let x = *val as f64 * self.decay;
            *val = x.floor() as i32;
        }
        // Optionally drop zeros:
        self.penalties.retain(|_, v| *v > 0);
    }

    /// Clear all penalties.
    pub fn clear(&mut self) {
        self.penalties.clear();
    }
}
