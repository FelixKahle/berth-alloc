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
        neighborhood::{
            adjacency_mask::AdjacencyMask, neighbor_lists::NeighborLists, view::NeighborView,
        },
        solver_model::SolverModel,
    },
    state::chain_set::index::NodeIndex,
};
use num_traits::{CheckedAdd, CheckedSub};

pub mod adjacency_mask;
pub mod neighbor_lists;
pub mod view;

#[derive(Debug, Clone, Copy)]
pub struct ProximityMapParameter {
    topk_ratio: f64,
}

impl ProximityMapParameter {
    #[inline]
    pub const fn new(topk_ratio: f64) -> Self {
        Self {
            topk_ratio: topk_ratio.min(1.0).max(0.0),
        }
    }

    #[inline]
    pub const fn topk_ratio(&self) -> f64 {
        self.topk_ratio
    }
}

impl Default for ProximityMapParameter {
    fn default() -> Self {
        Self { topk_ratio: 1.0 }
    }
}

#[derive(Debug)]
pub struct ProximityMap {
    /// Symmetric, cheap surrogate across all pairs (e.g., index distance).
    generic: NeighborView,
    /// Only neighbors sharing at least one berth.
    same_berth: NeighborView,
    /// Only neighbors with time-window overlap.
    overlap_tw: NeighborView,
    /// Logical OR of the above three (broad, still pruned).
    any_feasibleish: NeighborView,
    /// Unfiltered “all but self” (useful for diversification).
    all: NeighborView,
    /// Optional targeted views per berth (index by `BerthIndex`).
    per_berth: Vec<NeighborView>,
}

impl ProximityMap {
    pub fn build<T: Copy + Ord + CheckedAdd + CheckedSub>(
        model: &SolverModel<'_, T>,
        params: ProximityMapParameter,
    ) -> Self {
        let n = model.flexible_requests_len();
        let b = model.berths_len();
        let ratio = params.topk_ratio();

        let all_lists = build_lists_all(n, ratio);
        let all_mask = AdjacencyMask::from_lists(n, &lists_to_usize(&all_lists));
        let all = NeighborView::new(all_lists, all_mask);

        let generic_lists = build_lists_with(
            n,
            |_i, _j| true,
            |i, j| Some(i.abs_diff(j)), // lower is better
            ratio,
        );
        let generic_mask = AdjacencyMask::from_lists(n, &lists_to_usize(&generic_lists));
        let generic = NeighborView::new(generic_lists, generic_mask);

        let same_berth_lists = build_lists_with(
            n,
            |i, j| share_any_berth(model, RequestIndex(i), RequestIndex(j)),
            |i, j| Some(i.abs_diff(j)),
            ratio,
        );
        let same_berth_mask = AdjacencyMask::from_lists(n, &lists_to_usize(&same_berth_lists));
        let same_berth = NeighborView::new(same_berth_lists, same_berth_mask);

        let feasible = model.feasible_intervals();
        let overlap_lists = build_lists_with(
            n,
            |i, j| feasible[i].intersects(&feasible[j]),
            |i, j| Some(i.abs_diff(j)), // tie-breaker
            ratio,
        );
        let overlap_mask = AdjacencyMask::from_lists(n, &lists_to_usize(&overlap_lists));
        let overlap_tw = NeighborView::new(overlap_lists, overlap_mask);

        let any_mask = generic.mask().or(same_berth.mask()).or(overlap_tw.mask());
        let any_lists = fuse_orders_with_mask(
            n,
            &[generic.lists(), same_berth.lists(), overlap_tw.lists()],
            &any_mask,
        );
        let any_feasibleish = NeighborView::new(any_lists, any_mask);

        let mut per_berth = Vec::with_capacity(b);
        for bi in 0..b {
            let lists = build_lists_with(
                n,
                |i, j| both_allowed_on(model, RequestIndex(i), RequestIndex(j), BerthIndex(bi)),
                |i, j| Some(i.abs_diff(j)),
                ratio,
            );
            let mask = AdjacencyMask::from_lists(n, &lists_to_usize(&lists));
            per_berth.push(NeighborView::new(lists, mask));
        }

        Self {
            generic,
            same_berth,
            overlap_tw,
            any_feasibleish,
            all,
            per_berth,
        }
    }

    #[inline]
    pub fn generic(&self) -> &NeighborView {
        &self.generic
    }

    #[inline]
    pub fn same_berth(&self) -> &NeighborView {
        &self.same_berth
    }

    #[inline]
    pub fn overlap_tw(&self) -> &NeighborView {
        &self.overlap_tw
    }

    #[inline]
    pub fn any_feasibleish(&self) -> &NeighborView {
        &self.any_feasibleish
    }

    #[inline]
    pub fn all(&self) -> &NeighborView {
        &self.all
    }

    #[inline]
    pub fn per_berth(&self) -> &[NeighborView] {
        &self.per_berth
    }
}

#[inline]
fn lists_to_usize(lists: &NeighborLists) -> Vec<Vec<usize>> {
    lists
        .outgoing()
        .iter()
        .map(|row| row.iter().map(|ni| ni.get()).collect::<Vec<_>>())
        .collect()
}

#[inline]
fn topk_len(len: usize, ratio: f64) -> usize {
    ((len as f64) * ratio).ceil() as usize
}

/// Build “all but self”, optionally truncated by topk ratio using |i-j| rank.
fn build_lists_all(n: usize, ratio: f64) -> NeighborLists {
    let mut outgoing: Vec<Vec<NodeIndex>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<NodeIndex> = (0..n).filter(|&j| j != i).map(NodeIndex::new).collect();

        // Sort by |i - j| as a generic “nearby” order.
        row.sort_unstable_by_key(|nj| i.abs_diff(nj.get()));

        // Truncate by ratio
        row.truncate(topk_len(row.len(), ratio));
        outgoing.push(row);
    }
    let incoming = invert_lists(n, &outgoing);
    NeighborLists::new(incoming, outgoing)
}

/// Generic builder with a filter and a score (lower is better).
fn build_lists_with<S: Copy + Ord>(
    n: usize,
    allow: impl Fn(usize, usize) -> bool,
    score: impl Fn(usize, usize) -> Option<S>,
    ratio: f64,
) -> NeighborLists {
    let mut outgoing: Vec<Vec<NodeIndex>> = Vec::with_capacity(n);
    let mut buf: Vec<(usize, S)> = Vec::with_capacity(n);

    for i in 0..n {
        buf.clear();
        for j in 0..n {
            if i == j {
                continue;
            }
            if allow(i, j)
                && let Some(s) = score(i, j)
            {
                buf.push((j, s));
            }
        }
        buf.sort_unstable_by_key(|&(_, s)| s);
        let k = topk_len(buf.len(), ratio);
        let row = buf
            .iter()
            .take(k)
            .map(|(j, _)| NodeIndex::new(*j))
            .collect();
        outgoing.push(row);
    }
    let incoming = invert_lists(n, &outgoing);
    NeighborLists::new(incoming, outgoing)
}

/// Build incoming lists from outgoing lists, reserving exact capacities to avoid reallocations.
fn invert_lists(n: usize, outgoing: &[Vec<NodeIndex>]) -> Vec<Vec<NodeIndex>> {
    let mut indeg = vec![0usize; n];
    for row in outgoing {
        for &j in row {
            indeg[j.get()] += 1;
        }
    }

    let mut incoming: Vec<Vec<NodeIndex>> = Vec::with_capacity(n);
    for (_, &deg) in indeg.iter().enumerate() {
        incoming.push(Vec::with_capacity(deg));
    }

    for (i, row) in outgoing.iter().enumerate() {
        let ni = NodeIndex::new(i);
        for &j in row {
            incoming[j.get()].push(ni);
        }
    }
    incoming
}

/// Fuse several preference orders under a membership mask.
/// The earlier lists in `views_in_priority` win ties.
fn fuse_orders_with_mask(
    n: usize,
    views_in_priority: &[&NeighborLists],
    mask: &AdjacencyMask,
) -> NeighborLists {
    let mut outgoing: Vec<Vec<NodeIndex>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::new();
        let mut seen = fixedbitset::FixedBitSet::with_capacity(n);
        seen.grow(n);
        for lists in views_in_priority {
            if let Some(cands) = lists.outgoing_for(NodeIndex::new(i)) {
                for &nj in cands {
                    let j = nj.get();
                    if !mask.contains(i, j) || seen.contains(j) {
                        continue;
                    }
                    seen.insert(j);
                    row.push(nj);
                }
            }
        }
        outgoing.push(row);
    }
    let incoming = invert_lists(n, &outgoing);
    NeighborLists::new(incoming, outgoing)
}

/// True iff there exists a berth where both requests are feasible.
fn share_any_berth<T: Copy + Ord + CheckedAdd + CheckedSub>(
    model: &SolverModel<'_, T>,
    i: RequestIndex,
    j: RequestIndex,
) -> bool {
    let b = model.berths_len();
    for bi in 0..b {
        let bi = BerthIndex(bi);
        if allowed_on(model, i, bi) && allowed_on(model, j, bi) {
            return true;
        }
    }
    false
}

/// True iff both requests are feasible on the same specific berth.
fn both_allowed_on<T: Copy + Ord + CheckedAdd + CheckedSub>(
    model: &SolverModel<'_, T>,
    i: RequestIndex,
    j: RequestIndex,
    b: BerthIndex,
) -> bool {
    allowed_on(model, i, b) && allowed_on(model, j, b)
}

/// True iff request is feasible on berth.
fn allowed_on<T: Copy + Ord + CheckedAdd + CheckedSub>(
    model: &SolverModel<'_, T>,
    r: RequestIndex,
    b: BerthIndex,
) -> bool {
    // processing_time() returns Option<Option<TimeDelta<T>>>
    //   None        -> OOB (shouldn’t happen here)
    //   Some(None)  -> not allowed on this berth
    //   Some(Some(_))-> allowed
    matches!(model.processing_time(r, b), Some(Some(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use berth_alloc_core::prelude::{TimeDelta, TimeInterval, TimePoint};
    use berth_alloc_model::common::FlexibleKind;
    use berth_alloc_model::prelude::{Berth, BerthIdentifier, Problem, RequestIdentifier};
    use berth_alloc_model::problem::builder::ProblemBuilder;
    use berth_alloc_model::problem::req::Request;
    use std::collections::BTreeMap;

    // ----- small helpers (mirror style of existing tests) -----
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
    fn bid(n: usize) -> BerthIdentifier {
        BerthIdentifier::new(n)
    }
    #[inline]
    fn rid(n: usize) -> RequestIdentifier {
        RequestIdentifier::new(n)
    }
    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex(n)
    }
    #[inline]
    fn ri(n: usize) -> RequestIndex {
        RequestIndex(n)
    }

    fn berth(id: usize, s: i64, e: i64) -> Berth<i64> {
        Berth::from_windows(bid(id), [iv(s, e)])
    }

    /// Make a small problem with:
    /// - berths: B1=[0,100), B2=[0,100)
    /// - requests:
    ///   r0: TW=[0,50), allowed {B1:10}
    ///   r1: TW=[25,75), allowed {B1:12, B2:20}
    ///   r2: TW=[60,90), allowed {B2:15}
    ///
    /// Overlaps: r0<->r1, r1<->r2. Same-berth: r0<->r1 on B1, r1<->r2 on B2.
    fn make_problem_for_proximity() -> Problem<i64> {
        let b1 = berth(1, 0, 100);
        let b2 = berth(2, 0, 100);

        // r0 on B1
        let mut pt_r0 = BTreeMap::new();
        pt_r0.insert(bid(1), td(10));
        let r0 = Request::<FlexibleKind, i64>::new(rid(10), iv(0, 50), 1, pt_r0)
            .expect("r0 must be feasible");

        // r1 on B1 and B2
        let mut pt_r1 = BTreeMap::new();
        pt_r1.insert(bid(1), td(12));
        pt_r1.insert(bid(2), td(20));
        let r1 = Request::<FlexibleKind, i64>::new(rid(20), iv(25, 75), 1, pt_r1)
            .expect("r1 must be feasible");

        // r2 only on B2
        let mut pt_r2 = BTreeMap::new();
        pt_r2.insert(bid(2), td(15));
        let r2 = Request::<FlexibleKind, i64>::new(rid(30), iv(60, 90), 1, pt_r2)
            .expect("r2 must be feasible");

        let mut builder = ProblemBuilder::new();
        // Add berths and requests (out of order on purpose)
        builder.add_berth(b2.clone());
        builder.add_berth(b1.clone());
        builder.add_flexible(r2.clone());
        builder.add_flexible(r0.clone());
        builder.add_flexible(r1.clone());
        builder.build().expect("problem should build")
    }

    fn model_from_problem(p: &Problem<i64>) -> SolverModel<'_, i64> {
        SolverModel::try_from(p).expect("solver model should build")
    }

    #[test]
    fn test_proximityparams_clamps_and_default() {
        let d = ProximityMapParameter::default();
        assert!((d.topk_ratio() - 1.0).abs() < 1e-6);

        let p0 = ProximityMapParameter::new(-5.0);
        assert!((p0.topk_ratio() - 0.0).abs() < 1e-6);

        let p1 = ProximityMapParameter::new(2.0);
        assert!((p1.topk_ratio() - 1.0).abs() < 1e-6);

        let p_half = ProximityMapParameter::new(0.5);
        assert!((p_half.topk_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_build_lists_all_order_and_ratio() {
        let n = 5;
        // ratio = 0.4 => ceil(4 * 0.4) = 2 neighbors per row
        let lists = super::build_lists_all(n, 0.4);

        // Node 0: neighbors sorted by |i-j| => [1,2,...], truncated to 2
        let row0 = lists.outgoing()[0]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(row0.len(), 2);
        assert_eq!(row0, vec![1, 2]);

        // Node 2: neighbors by |2-j| => [1,3,0,4], truncated to 2
        let row2 = lists.outgoing()[2]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(row2.len(), 2);
        assert_eq!(row2, vec![1, 3]);

        // Incoming should be consistent inverse
        let in_of_1 = lists.incoming()[1]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        // Who points to 1 among truncated rows? 0 and 2 should (as above).
        assert!(in_of_1.contains(&0));
        assert!(in_of_1.contains(&2));
    }

    #[test]
    fn test_build_lists_with_filter_and_score() {
        // n=4, allow only neighbors with j > i, score is (j - i)
        let lists = super::build_lists_with(4, |i, j| j > i, |i, j| Some(j - i), 1.0);

        // For i=1 => allowed {2,3}, increasing score => [2,3]
        let row1 = lists.outgoing()[1]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(row1, vec![2, 3]);

        // For i=3 => allowed {}, empty
        let row3 = lists.outgoing()[3]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert!(row3.is_empty());

        // Incoming inverse: node 3 should have incoming from 0,1,2 (only those with j>i)
        let in3 = lists.incoming()[3]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(in3, vec![0, 1, 2]);
    }

    #[test]
    fn test_invert_lists_consistency() {
        // Manually craft a small outgoing and check invert
        use crate::state::chain_set::index::NodeIndex;
        let n = 3;
        let outgoing = vec![
            vec![NodeIndex::new(1), NodeIndex::new(2)],
            vec![NodeIndex::new(2)],
            vec![],
        ];
        let incoming = super::invert_lists(n, &outgoing);

        // incoming[2] should have 0 and 1
        let in2 = incoming[2].iter().map(|x| x.get()).collect::<Vec<_>>();
        assert_eq!(in2, vec![0, 1]);
        // incoming[0] is empty (nobody points to 0)
        assert!(incoming[0].is_empty());
    }

    #[test]
    fn test_proximity_map_views_basic_properties() {
        let p = make_problem_for_proximity();
        let m = model_from_problem(&p);

        // Index mapping sanity: request ids {10,20,30} must be sorted into indices {0,1,2}
        let im = m.index_manager();
        assert_eq!(im.request_id(ri(0)), Some(rid(10)));
        assert_eq!(im.request_id(ri(1)), Some(rid(20)));
        assert_eq!(im.request_id(ri(2)), Some(rid(30)));

        let proximity = ProximityMap::build(&m, ProximityMapParameter::new(1.0));

        // ---- ALL view: every row contains all except self
        for i in 0..m.flexible_requests_len() {
            let row = proximity.all().lists().outgoing()[i]
                .iter()
                .map(|x| x.get())
                .collect::<Vec<_>>();
            assert_eq!(row.len(), m.flexible_requests_len() - 1);
            assert!(!row.contains(&i));
        }

        // ---- SAME BERTH view:
        // r0 (idx 0) shares a berth only with r1 (idx 1)
        let sb0 = proximity.same_berth().lists().outgoing()[0]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(sb0, vec![1]);

        // r1 shares with both r0 and r2 (B1 with r0, B2 with r2)
        let mut sb1 = proximity.same_berth().lists().outgoing()[1]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        sb1.sort();
        assert_eq!(sb1, vec![0, 2]);

        // r2 shares only with r1
        let sb2 = proximity.same_berth().lists().outgoing()[2]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(sb2, vec![1]);

        // ---- OVERLAP TW view:
        // r0 overlaps only with r1; r2 overlaps only with r1; r1 with both.
        let ov0 = proximity.overlap_tw().lists().outgoing()[0]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(ov0, vec![1]);

        let mut ov1 = proximity.overlap_tw().lists().outgoing()[1]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        ov1.sort();
        assert_eq!(ov1, vec![0, 2]);

        let ov2 = proximity.overlap_tw().lists().outgoing()[2]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(ov2, vec![1]);

        // ---- ANY FEASIBLEISH mask = union(generic, same_berth, overlap_tw)
        // Check that entries present in same_berth or overlap_tw are marked in the union mask.
        let union_mask = proximity.any_feasibleish().mask();
        assert!(union_mask.contains(0, 1)); // present in both same_berth + overlap
        assert!(union_mask.contains(1, 0)); // present
        assert!(union_mask.contains(1, 2)); // present
        assert!(union_mask.contains(2, 1)); // present
        // 0 <-> 2 are not same_berth nor overlap; they *will* be present via generic (index proximity),
        // but at minimum, union must include the above four.
        // We do a sanity check that 0->2 is also permitted by union since generic allows it:
        assert!(union_mask.contains(0, 2));
    }

    #[test]
    fn test_per_berth_views_respect_feasibility() {
        let p = make_problem_for_proximity();
        let m = model_from_problem(&p);
        let proximity = ProximityMap::build(&m, ProximityMapParameter::new(1.0));

        // B0 corresponds to berth id=1; B1 to id=2
        // On B0 (id=1): feasible pairs are r0<->r1 only
        let pb0 = &proximity.per_berth()[0];
        let mut pb0_0 = pb0.lists().outgoing()[0]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        pb0_0.sort();
        assert_eq!(pb0_0, vec![1]);
        let mut pb0_1 = pb0.lists().outgoing()[1]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        pb0_1.sort();
        assert_eq!(pb0_1, vec![0]);
        let pb0_2 = pb0.lists().outgoing()[2]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert!(pb0_2.is_empty());

        // On B1 (id=2): feasible pairs are r1<->r2 only
        let pb1 = &proximity.per_berth()[1];
        let pb1_0 = pb1.lists().outgoing()[0]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert!(pb1_0.is_empty());
        let mut pb1_1 = pb1.lists().outgoing()[1]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        pb1_1.sort();
        assert_eq!(pb1_1, vec![2]);
        let mut pb1_2 = pb1.lists().outgoing()[2]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        pb1_2.sort();
        assert_eq!(pb1_2, vec![1]);
    }

    #[test]
    fn test_topk_ratio_effect() {
        // n=3, each row has 2 candidates in ALL view. With 0.5 ratio => ceil(2*0.5)=1 kept.
        let p = make_problem_for_proximity();
        let m = model_from_problem(&p);
        let proximity = ProximityMap::build(&m, ProximityMapParameter::new(0.5));

        for i in 0..m.flexible_requests_len() {
            let row = proximity.all().lists().outgoing()[i]
                .iter()
                .map(|x| x.get())
                .collect::<Vec<_>>();
            assert_eq!(row.len(), 1, "row {} should be truncated to 1", i);
        }
    }

    #[test]
    fn test_generic_prefers_index_proximity() {
        // For n=5, generic should rank by |i-j|
        let p = {
            // simple 5 nodes w/ same full TW & all berths feasible so only index distance matters
            let b1 = berth(1, 0, 1000);
            let mut builder = ProblemBuilder::new();
            builder.add_berth(b1.clone());
            for ridx in [10, 11, 12, 13, 14] {
                let mut pt = BTreeMap::new();
                pt.insert(bid(1), td(5));
                let r = Request::<FlexibleKind, i64>::new(rid(ridx), iv(0, 1000), 1, pt)
                    .expect("request");
                builder.add_flexible(r);
            }
            builder.build().unwrap()
        };
        let m = model_from_problem(&p);
        let proximity = ProximityMap::build(&m, ProximityMapParameter::new(1.0));
        // Node 2 should prefer 1 and 3 first (closest indices)
        let row2 = proximity.generic().lists().outgoing()[2]
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>();
        assert_eq!(row2[0], 1);
        assert_eq!(row2[1], 3);
    }

    #[test]
    fn test_allowed_on_share_any_berth_helpers() {
        let p = make_problem_for_proximity();
        let m = model_from_problem(&p);

        // allowed_on checks
        assert!(super::allowed_on(&m, ri(0), bi(0))); // r0 on B1
        assert!(!super::allowed_on(&m, ri(0), bi(1))); // r0 not on B2
        assert!(super::allowed_on(&m, ri(1), bi(0))); // r1 on B1
        assert!(super::allowed_on(&m, ri(1), bi(1))); // r1 on B2
        assert!(!super::allowed_on(&m, ri(2), bi(0))); // r2 not on B1
        assert!(super::allowed_on(&m, ri(2), bi(1))); // r2 on B2

        // share_any_berth
        assert!(super::share_any_berth(&m, ri(0), ri(1))); // share B1
        assert!(super::share_any_berth(&m, ri(1), ri(2))); // share B2
        assert!(!super::share_any_berth(&m, ri(0), ri(2))); // share none
    }
}
