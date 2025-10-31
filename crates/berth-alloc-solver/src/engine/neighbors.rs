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
    search::operator::{NeighborFn, NeighborFnSlice},
};
use std::sync::Arc;

#[inline]
pub fn from_view<'a>(view: &'a NeighborView) -> NeighborFn<'a> {
    let f: Arc<NeighborFnSlice<'a>> =
        Arc::new(move |seed: RequestIndex| view.lists().outgoing_for(seed).unwrap_or(&[]));
    NeighborFn::Slice(f)
}

#[inline]
pub fn generic<'a>(pm: &'a ProximityMap) -> NeighborFn<'a> {
    from_view(pm.generic())
}

#[inline]
pub fn same_berth<'a>(pm: &'a ProximityMap) -> NeighborFn<'a> {
    from_view(pm.same_berth())
}

#[inline]
pub fn overlap_tw<'a>(pm: &'a ProximityMap) -> NeighborFn<'a> {
    // Method name in ProximityMap is overlap_time_window()
    from_view(pm.overlap_time_window())
}

#[inline]
pub fn direct_competitors<'a>(pm: &'a ProximityMap) -> NeighborFn<'a> {
    from_view(pm.direct_competitors())
}

#[inline]
pub fn any<'a>(pm: &'a ProximityMap) -> NeighborFn<'a> {
    from_view(pm.any_feasibleish())
}

#[inline]
pub fn per_berth<'a>(pm: &'a ProximityMap, bi: BerthIndex) -> NeighborFn<'a> {
    let view: &'a NeighborView = &pm.per_berth()[bi.get()];
    let f: Arc<NeighborFnSlice<'a>> =
        Arc::new(move |seed: RequestIndex| view.lists().outgoing_for(seed).unwrap_or(&[]));
    NeighborFn::Slice(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::solver_model::SolverModel;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::{prelude::*, problem::builder::ProblemBuilder};
    use std::collections::BTreeMap;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }
    #[inline]
    fn iv(a: i64, b: i64) -> TimeInterval<i64> {
        TimeInterval::new(tp(a), tp(b))
    }
    #[inline]
    fn td(v: i64) -> TimeDelta<i64> {
        TimeDelta::new(v)
    }
    #[inline]
    fn bid(n: u32) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: u32) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }

    fn berth(id: u32, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    fn flex_req(
        id: u32,
        window: (i64, i64),
        pts: &[(u32, i64)],
        weight: i64,
    ) -> Request<FlexibleKind, i64> {
        let mut m = BTreeMap::new();
        for (b, d) in pts {
            m.insert(bid(*b), td(*d));
        }
        Request::<FlexibleKind, i64>::new(rid(id), iv(window.0, window.1), weight, m).unwrap()
    }

    // Build a small problem where:
    // - r0 shares berth with r1 (B1),
    // - r1 shares berth with r2 (B2),
    // - Overlaps: r0<->r1 overlap, r1<->r2 overlap, r0<->r2 don't overlap.
    fn build_model() -> SolverModel<'static, i64> {
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);

        let r0 = flex_req(10, (0, 50), &[(1, 5)], 1); // only B1
        let r1 = flex_req(20, (25, 75), &[(1, 6), (2, 7)], 1); // B1 and B2
        let r2 = flex_req(30, (60, 100), &[(2, 8)], 1); // only B2

        let mut builder = ProblemBuilder::new();
        builder.add_berth(b1);
        builder.add_berth(b2);
        builder.add_flexible(r0);
        builder.add_flexible(r1);
        builder.add_flexible(r2);
        let problem = builder.build().expect("valid problem");

        // Leak problem to give it a 'static lifetime for the test scope
        let problem_static: &'static Problem<i64> = Box::leak(Box::new(problem));
        SolverModel::try_from(problem_static).expect("model ok")
    }

    #[test]
    fn test_from_view_slice_matches_neighbor_view() {
        let model = build_model();
        let pm = model.proximity_map();

        // Use the same_berth view to check slice behavior
        let nf = from_view(pm.same_berth());
        let im = model.index_manager();
        let r1 = im.request_index(rid(20)).unwrap();

        match nf {
            NeighborFn::Slice(f) => {
                let got = f(r1);
                let exp = pm.same_berth().lists().outgoing_for(r1).unwrap_or(&[]);
                assert_eq!(got, exp, "from_view slice should match NeighborView rows");

                // Out-of-bounds request should return empty slice
                let got_oob = f(RequestIndex::new(999));
                assert!(got_oob.is_empty(), "oob must return empty slice");
            }
            _ => panic!("from_view should produce Slice variant"),
        }
    }

    #[test]
    fn test_wiring_of_views_generic_overlap_any_direct() {
        let model = build_model();
        let pm = model.proximity_map();
        let im = model.index_manager();
        let r0 = im.request_index(rid(10)).unwrap();
        let r1 = im.request_index(rid(20)).unwrap();
        let r2 = im.request_index(rid(30)).unwrap();

        // generic
        match generic(pm) {
            NeighborFn::Slice(f) => {
                assert_eq!(f(r1), pm.generic().lists().outgoing_for(r1).unwrap_or(&[]));
            }
            _ => panic!("generic should be Slice"),
        }

        // overlap_tw (overlap_time_window)
        match overlap_tw(pm) {
            NeighborFn::Slice(f) => {
                assert_eq!(
                    f(r0),
                    pm.overlap_time_window()
                        .lists()
                        .outgoing_for(r0)
                        .unwrap_or(&[])
                );
                assert_eq!(
                    f(r1),
                    pm.overlap_time_window()
                        .lists()
                        .outgoing_for(r1)
                        .unwrap_or(&[])
                );
                assert_eq!(
                    f(r2),
                    pm.overlap_time_window()
                        .lists()
                        .outgoing_for(r2)
                        .unwrap_or(&[])
                );
            }
            _ => panic!("overlap_tw should be Slice"),
        }

        // direct_competitors
        match direct_competitors(pm) {
            NeighborFn::Slice(f) => {
                assert_eq!(
                    f(r1),
                    pm.direct_competitors()
                        .lists()
                        .outgoing_for(r1)
                        .unwrap_or(&[])
                );
            }
            _ => panic!("direct_competitors should be Slice"),
        }

        // any (union)
        match any(pm) {
            NeighborFn::Slice(f) => {
                assert_eq!(
                    f(r1),
                    pm.any_feasibleish().lists().outgoing_for(r1).unwrap_or(&[])
                );
            }
            _ => panic!("any should be Slice"),
        }
    }

    #[test]
    fn test_per_berth_slice_matches_view() {
        let model = build_model();
        let pm = model.proximity_map();
        let im = model.index_manager();
        let r0 = im.request_index(rid(10)).unwrap();
        let r1 = im.request_index(rid(20)).unwrap();

        let b1 = im.berth_index(bid(1)).unwrap();
        let b2 = im.berth_index(bid(2)).unwrap();

        // B1 view
        match per_berth(pm, b1) {
            NeighborFn::Slice(f) => {
                let got0 = f(r0);
                let exp0 = pm.per_berth()[b1.get()]
                    .lists()
                    .outgoing_for(r0)
                    .unwrap_or(&[]);
                assert_eq!(got0, exp0);

                let got1 = f(r1);
                let exp1 = pm.per_berth()[b1.get()]
                    .lists()
                    .outgoing_for(r1)
                    .unwrap_or(&[]);
                assert_eq!(got1, exp1);

                let g_oob = f(RequestIndex::new(999));
                assert!(g_oob.is_empty(), "oob must return empty slice");
            }
            _ => panic!("per_berth should be Slice variant"),
        }

        // B2 view
        match per_berth(pm, b2) {
            NeighborFn::Slice(f) => {
                let got1 = f(r1);
                let exp1 = pm.per_berth()[b2.get()]
                    .lists()
                    .outgoing_for(r1)
                    .unwrap_or(&[]);
                assert_eq!(got1, exp1);
            }
            _ => panic!("per_berth should be Slice variant"),
        }
    }
}
