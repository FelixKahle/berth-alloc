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

use std::num::NonZeroUsize;

/// A strongly-typed, non-zero handle to a node within a `PathArena`.
///
/// This wrapper around `NonZeroUsize` ensures that node keys are always valid
/// (i.e., not zero), which allows for `Option<NodeKey>` to be represented in
/// the same space as a `usize` due to niche optimization. This is crucial for
/// the arena's internal storage.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeKey(NonZeroUsize);

impl NodeKey {
    /// Returns the underlying `usize` value of the key (1-based).
    #[inline]
    pub fn get(self) -> usize {
        self.0.get()
    }

    /// Creates a `NodeKey` from a 1-based index.
    ///
    /// # Panics
    ///
    /// Panics if the input index `one_based_index` is zero.
    #[inline]
    pub(super) fn from_index(one_based_index: usize) -> Self {
        Self(NonZeroUsize::new(one_based_index).expect("NodeKey indices must be 1-based"))
    }
}

impl std::fmt::Display for NodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeKey({})", self.0)
    }
}

/// A strongly-typed handle to a path within a `PathArena`.
///
/// This is a simple wrapper around a `usize` to provide type safety when
/// interacting with the `PathArena` API.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PathKey(usize);

impl PathKey {
    /// Returns the raw `usize` index of the path.
    #[inline]
    pub const fn to_raw(self) -> usize {
        self.0
    }

    /// Creates a `PathKey` from a raw `usize` index.
    #[inline]
    pub(super) const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
}

impl From<PathKey> for usize {
    #[inline]
    fn from(val: PathKey) -> Self {
        val.0
    }
}

impl std::fmt::Display for PathKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PathKey({})", self.0)
    }
}
