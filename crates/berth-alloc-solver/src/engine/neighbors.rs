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

use crate::{
    model::{
        index::{BerthIndex, RequestIndex},
        neighborhood::ProximityMap,
        view::NeighborView,
    },
    search::operator::NeighborFn,
};
use std::sync::Arc;

#[inline]
pub fn from_view(view: &NeighborView) -> Arc<NeighborFn> {
    let v = view.clone();
    Arc::new(move |seed: RequestIndex| {
        v.lists()
            .outgoing_for(seed)
            .map(|xs| xs.to_vec())
            .unwrap_or_default()
    })
}

#[inline]
pub fn generic(pm: &ProximityMap) -> Arc<NeighborFn> {
    from_view(pm.generic())
}
#[inline]
pub fn same_berth(pm: &ProximityMap) -> Arc<NeighborFn> {
    from_view(pm.same_berth())
}
#[inline]
pub fn overlap_tw(pm: &ProximityMap) -> Arc<NeighborFn> {
    from_view(pm.overlap_tw())
}
#[inline]
pub fn direct_competitors(pm: &ProximityMap) -> Arc<NeighborFn> {
    from_view(pm.direct_competitors())
}
#[inline]
pub fn any(pm: &ProximityMap) -> Arc<NeighborFn> {
    from_view(pm.any_feasibleish())
}

#[inline]
pub fn per_berth(pm: &ProximityMap, bi: BerthIndex) -> Arc<NeighborFn> {
    let v = pm.per_berth()[bi.get()].clone();
    Arc::new(move |seed: RequestIndex| {
        v.lists()
            .outgoing_for(seed)
            .map(|xs| xs.to_vec())
            .unwrap_or_default()
    })
}
