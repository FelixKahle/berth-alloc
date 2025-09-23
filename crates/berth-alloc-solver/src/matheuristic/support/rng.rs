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

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
pub struct SeedSequencer {
    pub base: u64,
}

impl SeedSequencer {
    pub fn new(base: u64) -> Self {
        Self { base }
    }

    /// Deterministic per-iteration seed.
    pub fn for_iter(&self, iter: usize) -> u64 {
        self.base ^ (iter as u64)
    }

    /// Deterministic per-job seed.
    pub fn for_job(&self, iter_seed: u64, j: usize) -> u64 {
        iter_seed ^ ((j as u64).rotate_left(17)) ^ 0x9E37_79B1_85EB_CA87u64
    }

    pub fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }
}
