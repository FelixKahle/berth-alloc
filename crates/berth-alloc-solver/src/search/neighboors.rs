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

use crate::model::index::RequestIndex;
use std::sync::Arc;

pub type NeighborFnVec = dyn Fn(RequestIndex) -> Vec<RequestIndex> + Send + Sync;
pub type NeighborFnSlice<'a> = dyn Fn(RequestIndex) -> &'a [RequestIndex] + Send + Sync + 'a;

#[derive(Clone)]
pub enum NeighborFn<'a> {
    Vec(Arc<NeighborFnVec>),
    Slice(Arc<NeighborFnSlice<'a>>),
}

#[derive(Clone)]
pub struct Neighboors<'n> {
    /// Symmetric, cheap surrogate across all pairs (e.g., index distance).
    pub generic: NeighborFn<'n>,
    /// Only neighbors sharing at least one berth.
    pub same_berth: NeighborFn<'n>,
    /// Only neighbors with time-window overlap.
    pub overlap_tw: NeighborFn<'n>,
    /// Logical AND: neighbors sharing a berth AND having overlapping time windows.
    pub direct_competitors: NeighborFn<'n>,
    /// Logical OR of generic, same_berth, and overlap_tw (broad, still pruned).
    pub neighbors: NeighborFn<'n>,
    /// Unfiltered “all but self” (useful for diversification).
    pub all: NeighborFn<'n>,
    /// Targeted views per berth (index by `BerthIndex`).
    pub per_berth: Vec<NeighborFn<'n>>,
}

impl<'n> Neighboors<'n> {
    #[inline]
    pub fn new(
        generic: NeighborFn<'n>,
        same_berth: NeighborFn<'n>,
        overlap_tw: NeighborFn<'n>,
        direct_competitors: NeighborFn<'n>,
        neighbors: NeighborFn<'n>,
        all: NeighborFn<'n>,
        per_berth: Vec<NeighborFn<'n>>,
    ) -> Self {
        Self {
            generic,
            same_berth,
            overlap_tw,
            direct_competitors,
            neighbors,
            all,
            per_berth,
        }
    }
}
