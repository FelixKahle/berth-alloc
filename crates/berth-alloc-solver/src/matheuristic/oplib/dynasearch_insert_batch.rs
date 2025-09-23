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

use crate::framework::planning::{
    BrandedAssignmentRef, BrandedFreeBerth, BrandedRequest, PlanBuilder, PlanningContext,
};
use crate::matheuristic::operator::Operator;
use crate::terminal::terminalocc::TerminalRead;
use berth_alloc_core::prelude::{Cost, TimeInterval, TimePoint};
use berth_alloc_model::common::FlexibleKind;
use berth_alloc_model::prelude::{BerthIdentifier, RequestIdentifier};
use berth_alloc_model::problem::asg::{AssignmentRef, AssignmentView};
use num_traits::{CheckedAdd, CheckedSub, Zero};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct EnhancedDynasearchOnceOperator<T> {
    /// Max neighbors to evaluate per call (keeps O(n^2) in check).
    pub max_neighbors: usize,
    /// Shuffle neighbor list before evaluation.
    pub randomize: bool,
    /// Rebuild policy: pack-left (true) or pack-right (false).
    pub pack_left: bool,
    /// Work cap to avoid long stalls during evaluation.
    pub work_budget: usize,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for EnhancedDynasearchOnceOperator<T> {
    fn default() -> Self {
        Self {
            max_neighbors: 6000,
            randomize: true,
            pack_left: true,
            work_budget: 50_000,
            _p: Default::default(),
        }
    }
}

impl<T> Operator for EnhancedDynasearchOnceOperator<T>
where
    T: Copy
        + Ord
        + Send
        + Sync
        + std::fmt::Debug
        + CheckedAdd
        + CheckedSub
        + Mul<Output = Cost>
        + Into<Cost>,
{
    type Time = T;

    fn name(&self) -> &'static str {
        "EnhancedDynasearchOnce"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: PlanningContext<'s, 'p, T>,
        rng: &mut ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        // Tokens for a global permutation that keeps berth segmentation via Delim markers.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Tok {
            Ship(RequestIdentifier),
            Delim,
        }

        // Snapshot current per-berth order (scalars only).
        let mut berth_order: Vec<BerthIdentifier> = Vec::new();
        let mut per_berth: BTreeMap<BerthIdentifier, Vec<(TimePoint<T>, RequestIdentifier)>> =
            BTreeMap::new();

        let snapshot_ok = ctx
            .with_builder(|builder| {
                builder.with_explorer(|ex| {
                    for a in ex.iter_assignments() {
                        per_berth
                            .entry(a.asg().berth_id())
                            .or_default()
                            .push((a.asg().start_time(), a.asg().request_id()));
                    }
                });
            })
            .is_ok();
        if !snapshot_ok || per_berth.is_empty() {
            return None;
        }

        berth_order.extend(per_berth.keys().cloned());
        for v in per_berth.values_mut() {
            v.sort_by_key(|(s, _)| *s);
        }

        // Build token sequence S.
        let mut s: Vec<Tok> = Vec::new();
        for (idx, b) in berth_order.iter().enumerate() {
            if let Some(v) = per_berth.get(b) {
                for &(_, rid) in v {
                    s.push(Tok::Ship(rid));
                }
            }
            if idx + 1 < berth_order.len() {
                s.push(Tok::Delim);
            }
        }

        // Split tokens to segments in berth_order.
        let split_segments = |tokens: &[Tok]| -> Vec<Vec<RequestIdentifier>> {
            let mut segs: Vec<Vec<RequestIdentifier>> = vec![Vec::new(); berth_order.len()];
            let mut k = 0usize;
            for &t in tokens {
                match t {
                    Tok::Delim => {
                        if k + 1 < segs.len() {
                            k += 1;
                        }
                    }
                    Tok::Ship(rid) => segs[k].push(rid),
                }
            }
            segs
        };
        let orig_segs = split_segments(&s);

        // Precompute berth index for each token position (ships and delims).
        let mut pos2berth_idx: Vec<usize> = Vec::with_capacity(s.len());
        {
            let mut k = 0usize;
            for t in &s {
                match *t {
                    Tok::Delim => {
                        pos2berth_idx.push(k);
                        if k + 1 < berth_order.len() {
                            k += 1;
                        }
                    }
                    Tok::Ship(_) => pos2berth_idx.push(k),
                }
            }
        }

        // Neighborhood over ship tokens only.
        #[derive(Clone, Copy, Debug)]
        enum Move {
            Swap(usize, usize),
            Insert(usize, usize), // move j before i
        }

        let n = s.len();
        if n < 2 {
            return None;
        }

        let mut neigh: Vec<Move> = Vec::new();
        for j in 0..n {
            for i in 0..n {
                if i == j {
                    continue;
                }
                if let (Some(Tok::Ship(_)), Some(Tok::Ship(_))) = (s.get(i), s.get(j)) {
                    if i < j {
                        neigh.push(Move::Swap(i, j));
                    }
                    neigh.push(Move::Insert(i, j));
                }
            }
        }
        if self.randomize {
            neigh.shuffle(rng);
        }
        if neigh.len() > self.max_neighbors {
            neigh.truncate(self.max_neighbors);
        }

        // Build mutated segments only for the two berths that change.
        let apply_move_to_segments =
            |mv: Move| -> Option<(usize, Vec<RequestIdentifier>, usize, Vec<RequestIdentifier>)> {
                match mv {
                    Move::Swap(i, j) => {
                        let (b_i, b_j) = (pos2berth_idx[i], pos2berth_idx[j]);
                        let ship_i = if let Tok::Ship(r) = s[i] {
                            r
                        } else {
                            return None;
                        };
                        let ship_j = if let Tok::Ship(r) = s[j] {
                            r
                        } else {
                            return None;
                        };

                        if b_i == b_j {
                            let mut a = orig_segs[b_i].clone();
                            let idx_i = a.iter().position(|&r| r == ship_i)?;
                            let idx_j = a.iter().position(|&r| r == ship_j)?;
                            a.swap(idx_i, idx_j);
                            Some((b_i, a, b_j, Vec::new()))
                        } else {
                            let mut a = orig_segs[b_i].clone();
                            let mut b = orig_segs[b_j].clone();
                            let idx_i = a.iter().position(|&r| r == ship_i)?;
                            let idx_j = b.iter().position(|&r| r == ship_j)?;
                            let _ = a.remove(idx_i);
                            let _ = b.remove(idx_j);
                            a.push(ship_j);
                            b.push(ship_i);
                            Some((b_i, a, b_j, b))
                        }
                    }
                    Move::Insert(i, j) => {
                        let (b_i, b_j) = (pos2berth_idx[i], pos2berth_idx[j]);
                        let ship_i = if let Tok::Ship(r) = s[i] {
                            r
                        } else {
                            return None;
                        };
                        let ship_j = if let Tok::Ship(r) = s[j] {
                            r
                        } else {
                            return None;
                        };

                        if b_i == b_j {
                            let mut seg = orig_segs[b_i].clone();
                            let from = seg.iter().position(|&r| r == ship_j)?;
                            let to_before = seg.iter().position(|&r| r == ship_i)?;
                            let x = seg.remove(from);
                            let ins_at = if from < to_before {
                                to_before - 1
                            } else {
                                to_before
                            };
                            seg.insert(ins_at, x);
                            Some((b_i, seg, b_i, Vec::new()))
                        } else {
                            let mut src = orig_segs[b_j].clone();
                            let mut dst = orig_segs[b_i].clone();
                            let from = src.iter().position(|&r| r == ship_j)?;
                            let to_before = dst.iter().position(|&r| r == ship_i)?;
                            let _ = src.remove(from);
                            dst.insert(to_before, ship_j);
                            Some((b_j, src, b_i, dst))
                        }
                    }
                }
            };

        // First-improvement: as soon as we find a feasible improving move, apply and return.
        let mut found_move: Option<Move> = None;
        let mut budget = self.work_budget;

        let eval_ok = ctx.with_builder(|builder| {
            // Unassign everything on these berths; return original scalar schedule for restore.
            let unassign_all = |pb: &mut crate::framework::planning::PlanBuilder<'_, 'p, T>,
                                berths: &BTreeSet<BerthIdentifier>| {
                // Snapshot scalars (ordered) for restoration.
                let perb: HashMap<BerthIdentifier, Vec<(RequestIdentifier, TimePoint<T>)>> = pb
                    .with_explorer(|ex| {
                        let mut m: HashMap<_, Vec<_>> = HashMap::new();
                        for a in ex.iter_assigned_requests() {
                            let bid = a.asg().berth_id();
                            if berths.contains(&bid) {
                                m.entry(bid)
                                    .or_default()
                                    .push((a.asg().request_id(), a.asg().start_time()));
                            }
                        }
                        for v in m.values_mut() {
                            v.sort_by_key(|&(_, s)| s);
                        }
                        m
                    });

                // Unassign all handles on those berths (one pass).
                let mut handles: Vec<_> = pb.with_explorer(|ex| {
                    ex.iter_assigned_requests()
                        .filter(|a| berths.contains(&a.asg().berth_id()))
                        .collect::<Vec<_>>()
                });
                handles.sort_by_key(|a| (a.asg().berth_id(), a.asg().start_time()));
                for h in handles {
                    let _ = pb.propose_unassignment(&h);
                }

                perb
            };

            // Pack a given segment to left/right on a fixed berth (streaming min instead of Vec+sort).
            let reassign_segment = |pb: &mut crate::framework::planning::PlanBuilder<'_, 'p, T>,
                                    bid: BerthIdentifier,
                                    order: &[RequestIdentifier]|
             -> bool {
                if order.is_empty() {
                    return true;
                }

                // Iterate in requested direction.
                let iter: Box<dyn Iterator<Item = RequestIdentifier>> = if self.pack_left {
                    Box::new(order.iter().cloned())
                } else {
                    Box::new(order.iter().rev().cloned())
                };

                for rid in iter {
                    // Read-only phase: find best (fb, start) via streaming argmin.
                    let trip = pb.with_explorer(|ex| {
                        let r = ex
                            .iter_unassigned_requests()
                            .find(|x| x.req().id() == rid)?;
                        let pt = r.req().processing_time_for(bid)?;
                        let w = r.req().feasible_window();

                        // streaming best: (start, iv_start) as tie-break
                        let mut best: Option<(
                            BrandedFreeBerth<'_, 'p, T>,
                            TimePoint<T>,
                            TimePoint<T>,
                        )> = None;

                        for fb in ex
                            .iter_free_for(r.clone())
                            .filter(|fb| fb.berth().id() == bid)
                        {
                            let iv = *fb.interval();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = iv.end().checked_sub(pt)?;
                            let hi_w = w.end().checked_sub(pt)?;
                            let hi = std::cmp::min(hi_iv, hi_w);
                            if lo <= hi {
                                let s = if self.pack_left { lo } else { hi };
                                let key_s = s;
                                let key_iv = iv.start();
                                match &best {
                                    None => best = Some((fb.clone(), s, key_iv)),
                                    Some((_, best_s, best_iv)) => {
                                        if (self.pack_left
                                            && (key_s < *best_s
                                                || (key_s == *best_s && key_iv < *best_iv)))
                                            || (!self.pack_left
                                                && (key_s > *best_s
                                                    || (key_s == *best_s && key_iv > *best_iv)))
                                        {
                                            best = Some((fb.clone(), s, key_iv));
                                        }
                                    }
                                }
                            }
                        }
                        let (fb, s, _) = best?;
                        Some((r, fb, s))
                    });

                    // Write phase
                    if let Some((r, fb, s)) = trip {
                        if pb.propose_assignment(r, s, &fb).is_err() {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            };

            // Restore to the exact original scalars (unassign current → reassign originals).
            let restore_original = |pb: &mut crate::framework::planning::PlanBuilder<'_, 'p, T>,
                                    perb: &HashMap<
                BerthIdentifier,
                Vec<(RequestIdentifier, TimePoint<T>)>,
            >| {
                // Unassign whatever is on those berths right now.
                let mut handles: Vec<_> = pb.with_explorer(|ex| {
                    ex.iter_assigned_requests()
                        .filter(|a| perb.contains_key(&a.asg().berth_id()))
                        .collect::<Vec<_>>()
                });
                handles.sort_by_key(|a| (a.asg().berth_id(), a.asg().start_time()));
                for h in handles {
                    let _ = pb.propose_unassignment(&h);
                }

                // Put back originals.
                for (bid, vecs) in perb {
                    for &(rid, start) in vecs {
                        #[allow(clippy::type_complexity)]
                        let trip: Option<(
                            BrandedRequest<'_, 'p, FlexibleKind, T>,
                            BrandedFreeBerth<'_, 'p, T>,
                            TimePoint<T>,
                        )> = pb.with_explorer(|ex| {
                            let r = ex
                                .iter_unassigned_requests()
                                .find(|x| x.req().id() == rid)?;
                            let pt = r.req().processing_time_for(*bid)?;
                            let end = start.checked_add(pt)?;
                            let target = TimeInterval::new(start, end);
                            let fb = ex.iter_free_for(r.clone()).find(|fb| {
                                fb.berth().id() == *bid && fb.interval().contains_interval(&target)
                            })?;
                            Some((r, fb.clone(), start))
                        });

                        if let Some((r, fb, s)) = trip {
                            if pb.propose_assignment(r, s, &fb).is_err() {
                                return false;
                            }
                        } else {
                            return false;
                        }
                    }
                }
                true
            };

            // Try neighbors (first-improvement): as soon as we find improvement, record it and stop.
            for mv in neigh.iter().cloned() {
                if budget == 0 || found_move.is_some() {
                    break;
                }

                let Some((b1, seg1_new, b2, seg2_new)) = apply_move_to_segments(mv) else {
                    continue;
                };

                let mut changed: BTreeSet<BerthIdentifier> = BTreeSet::new();
                changed.insert(berth_order[b1]);
                if b2 != b1 && !seg2_new.is_empty() {
                    changed.insert(berth_order[b2]);
                }

                // Trial: unassign changed berths and rebuild just those.
                let originals = unassign_all(builder, &changed);
                budget =
                    budget.saturating_sub(1 + originals.values().map(|v| v.len()).sum::<usize>());

                let mut feasible = true;
                feasible &= reassign_segment(builder, berth_order[b1], &seg1_new);
                budget = budget.saturating_sub(seg1_new.len());
                if feasible && b2 != b1 && !seg2_new.is_empty() {
                    feasible &= reassign_segment(builder, berth_order[b2], &seg2_new);
                    budget = budget.saturating_sub(seg2_new.len());
                }

                if feasible {
                    let improve = -builder.delta_cost(); // negative delta_cost => improvement
                    if improve > Cost::zero() {
                        found_move = Some(mv);
                    }
                }

                // Roll back the trial state.
                let _ = restore_original(builder, &originals);
                debug_assert!(builder.delta_cost() == Cost::zero());
            }
        });

        if eval_ok.is_err() || found_move.is_none() {
            return None;
        }
        let mv_best = found_move.unwrap();

        // === Apply the improving neighbor for real (touch only the two berths) ===
        let apply_ok = ctx.with_builder(|builder| {
            if let Some((b1, seg1_new, b2, seg2_new)) = apply_move_to_segments(mv_best) {
                let mut changed: BTreeSet<BerthIdentifier> = BTreeSet::new();
                changed.insert(berth_order[b1]);
                if b2 != b1 && !seg2_new.is_empty() {
                    changed.insert(berth_order[b2]);
                }

                // Unassign current on the changed berths.
                let current: HashMap<BerthIdentifier, Vec<(RequestIdentifier, TimePoint<T>)>> =
                    builder.with_explorer(|ex| {
                        let mut m: HashMap<_, Vec<_>> = HashMap::new();
                        for a in ex.iter_assigned_requests() {
                            let bid = a.asg().berth_id();
                            if changed.contains(&bid) {
                                m.entry(bid)
                                    .or_default()
                                    .push((a.asg().request_id(), a.asg().start_time()));
                            }
                        }
                        for v in m.values_mut() {
                            v.sort_by_key(|&(_, s)| s);
                        }
                        m
                    });
                for (bid, vecs) in current {
                    for (rid, start) in vecs {
                        let maybe_asg = builder.with_explorer(|ex| {
                            ex.iter_assigned_requests().find(|a| {
                                a.asg().berth_id() == bid
                                    && a.asg().request_id() == rid
                                    && a.asg().start_time() == start
                            })
                        });
                        if let Some(asg) = maybe_asg {
                            let _ = builder.propose_unassignment(&asg);
                        }
                    }
                }

                // Reassign helper (same streaming-min inside).
                let mut reassign_segment = |bid: BerthIdentifier, order: &[RequestIdentifier]| {
                    for &rid in order {
                        let trip = builder.with_explorer(|ex| {
                            let r = ex
                                .iter_unassigned_requests()
                                .find(|x| x.req().id() == rid)?;
                            let pt = r.req().processing_time_for(bid)?;
                            let w = r.req().feasible_window();

                            let mut best: Option<(
                                BrandedFreeBerth<'_, 'p, T>,
                                TimePoint<T>,
                                TimePoint<T>,
                            )> = None;

                            for fb in ex
                                .iter_free_for(r.clone())
                                .filter(|fb| fb.berth().id() == bid)
                            {
                                let iv = *fb.interval();
                                let lo = std::cmp::max(iv.start(), w.start());
                                let hi_iv = iv.end().checked_sub(pt)?;
                                let hi_w = w.end().checked_sub(pt)?;
                                let hi = std::cmp::min(hi_iv, hi_w);
                                if lo <= hi {
                                    let s = if self.pack_left { lo } else { hi };
                                    let key_s = s;
                                    let key_iv = iv.start();
                                    match &best {
                                        None => best = Some((fb.clone(), s, key_iv)),
                                        Some((_, best_s, best_iv)) => {
                                            if (self.pack_left
                                                && (key_s < *best_s
                                                    || (key_s == *best_s && key_iv < *best_iv)))
                                                || (!self.pack_left
                                                    && (key_s > *best_s
                                                        || (key_s == *best_s && key_iv > *best_iv)))
                                            {
                                                best = Some((fb.clone(), s, key_iv));
                                            }
                                        }
                                    }
                                }
                            }
                            let (fb, s, _) = best?;
                            Some((r, fb, s))
                        });

                        if let Some((r, fb, s)) = trip {
                            if builder.propose_assignment(r, s, &fb).is_err() {
                                return; // abort apply
                            }
                        } else {
                            return; // abort apply
                        }
                    }
                };

                reassign_segment(berth_order[b1], &seg1_new);
                if b2 != b1 && !seg2_new.is_empty() {
                    reassign_segment(berth_order[b2], &seg2_new);
                }
            }
        });

        apply_ok.ok()
    }
}

#[derive(Debug, Clone)]
pub struct BerthBlockRuinRecreateOperator<T> {
    /// Max number of consecutive assignments on a berth to ruin.
    pub max_block_len: usize,
    /// Pack-left or pack-right preference on rebuild.
    pub pack_left: bool,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for BerthBlockRuinRecreateOperator<T> {
    fn default() -> Self {
        Self {
            max_block_len: 6,
            pack_left: true,
            _p: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynasearchInsertBatchOperator<T> {
    /// Shuffle fallback (lo/hi) starts when trying to place on the target berth.
    pub try_randomize_starts: bool,
    /// Max number of distinct requests to try moving in one batch.
    pub max_moves_per_batch: usize,
    _p: std::marker::PhantomData<T>,
}

impl<T> Default for DynasearchInsertBatchOperator<T> {
    fn default() -> Self {
        Self {
            try_randomize_starts: true,
            max_moves_per_batch: 32,
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Operator for DynasearchInsertBatchOperator<T>
where
    T: Copy
        + Ord
        + Send
        + Sync
        + std::fmt::Debug
        + CheckedAdd
        + CheckedSub
        + Mul<Output = Cost>
        + Into<Cost>,
{
    type Time = T;

    fn name(&self) -> &'static str {
        "DynasearchInsertBatch"
    }

    fn propose<'s, 'p>(
        &self,
        _iteration: usize,
        ctx: PlanningContext<'s, 'p, T>,
        rng: &mut ChaCha8Rng,
    ) -> Option<crate::framework::planning::Plan<'p, T>> {
        #[derive(Clone, Debug)]
        struct Cand<T: Copy + Ord> {
            rid: RequestIdentifier,
            from_berth: BerthIdentifier,
            from_start: TimePoint<T>,
            to_berth: BerthIdentifier,
            to_start: TimePoint<T>,
            benefit: Cost,
        }

        let mut cands: Vec<Cand<T>> = Vec::new();
        let explored_ok = ctx
            .with_builder(|builder| {
                let assigned: Vec<(RequestIdentifier, BerthIdentifier, TimePoint<T>, Cost)> =
                    builder.with_explorer(|ex| {
                        ex.iter_assigned_requests()
                            .map(|a| {
                                (
                                    a.asg().request_id(),
                                    a.asg().berth_id(),
                                    a.asg().start_time(),
                                    a.asg().cost(),
                                )
                            })
                            .collect()
                    });

                for (rid, cur_bid, cur_start, cur_cost) in assigned {
                    let req_ref = builder
                        .ledger()
                        .problem()
                        .flexible_requests()
                        .get(rid)
                        .expect("request exists");
                    let window = req_ref.feasible_window();
                    let allowed: Vec<BerthIdentifier> = req_ref.iter_allowed_berths_ids().collect();

                    let local: Vec<Cand<T>> = builder.with_explorer(|ex| {
                        let mut out = Vec::new();
                        for fb in ex
                            .sandbox()
                            .inner()
                            .iter_free_intervals_for_berths_in(allowed.clone(), window)
                        {
                            let bid = fb.berth().id();
                            if let Some(pt) = req_ref.processing_time_for(bid) {
                                let iv = fb.interval();
                                let lo = std::cmp::max(iv.start(), window.start());
                                let hi_iv = iv.end().checked_sub(pt);
                                let hi_w = window.end().checked_sub(pt);
                                if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                                    let hi = std::cmp::min(hi_iv, hi_w);
                                    if lo <= hi {
                                        let starts = if lo == hi { vec![lo] } else { vec![lo, hi] };
                                        for s in starts {
                                            if let Ok(new_ref) =
                                                AssignmentRef::<FlexibleKind, T>::new(
                                                    req_ref,
                                                    fb.berth(),
                                                    s,
                                                )
                                            {
                                                let benefit = cur_cost - new_ref.cost();
                                                if benefit > Cost::zero() {
                                                    out.push(Cand {
                                                        rid,
                                                        from_berth: cur_bid,
                                                        from_start: cur_start,
                                                        to_berth: bid,
                                                        to_start: s,
                                                        benefit,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        out
                    });
                    cands.extend(local);
                }
            })
            .is_ok();

        if !explored_ok || cands.is_empty() {
            return None;
        }

        cands.sort_by(|a, b| b.benefit.cmp(&a.benefit));
        let mut seen: HashSet<RequestIdentifier> = HashSet::new();
        let mut picked: Vec<Cand<T>> = Vec::new();
        for c in cands.into_iter() {
            if seen.insert(c.rid) {
                picked.push(c);
                if picked.len() >= self.max_moves_per_batch {
                    break;
                }
            }
        }
        if picked.is_empty() {
            return None;
        }

        let mut placed_any = false;
        let apply_res = ctx.with_builder(|builder| {
            let rollback_original = |pb: &mut PlanBuilder<'_, 'p, T>,
                                     rid: RequestIdentifier,
                                     orig_bid: BerthIdentifier,
                                     orig_start: TimePoint<T>|
             -> bool {
                #[allow(clippy::type_complexity)]
                let triplet: Option<(
                    BrandedRequest<'_, 'p, FlexibleKind, T>,
                    BrandedFreeBerth<'_, 'p, T>,
                    TimePoint<T>,
                )> = pb.with_explorer(|ex| {
                    let r = get_unassigned_brand(ex, rid)?;
                    let pt = r.req().processing_time_for(orig_bid)?;
                    let end = orig_start.checked_add(pt)?;
                    let target = TimeInterval::new(orig_start, end);
                    let fb = ex.iter_free_for(r.clone()).find(|fb| {
                        fb.berth().id() == orig_bid && fb.interval().contains_interval(&target)
                    })?;
                    Some((r, fb.clone(), orig_start))
                });

                if let Some((r, fb, s)) = triplet {
                    return pb.propose_assignment(r, s, &fb).is_ok();
                }
                false
            };

            for cand in picked.into_iter() {
                let current_asg: Option<BrandedAssignmentRef<'_, 'p, FlexibleKind, T>> = builder
                    .with_explorer(|ex| {
                        ex.iter_assigned_requests()
                            .find(|a| a.asg().request_id() == cand.rid)
                    });

                let Some(asg_ref) = current_asg else {
                    continue;
                };

                if builder.propose_unassignment(&asg_ref).is_err() {
                    continue;
                }

                #[allow(clippy::type_complexity)]
                let exact: Option<(
                    BrandedRequest<'_, 'p, FlexibleKind, T>,
                    BrandedFreeBerth<'_, 'p, T>,
                    TimePoint<T>,
                )> = builder.with_explorer(|ex| {
                    let r = get_unassigned_brand(ex, cand.rid)?;
                    let pt = r.req().processing_time_for(cand.to_berth)?;
                    let end = cand.to_start.checked_add(pt)?;
                    let target = TimeInterval::new(cand.to_start, end);
                    let fb = ex.iter_free_for(r.clone()).find(|fb| {
                        fb.berth().id() == cand.to_berth && fb.interval().contains_interval(&target)
                    })?;
                    Some((r, fb.clone(), cand.to_start))
                });

                if let Some((r, fb, s)) = exact
                    && builder.propose_assignment(r, s, &fb).is_ok()
                {
                    placed_any = true;
                    continue;
                }

                #[allow(clippy::type_complexity)]
                let fallback: Option<(
                    BrandedRequest<'_, 'p, FlexibleKind, T>,
                    Vec<(BrandedFreeBerth<'_, 'p, T>, TimePoint<T>)>,
                )> = builder.with_explorer(|ex| {
                    let r = get_unassigned_brand(ex, cand.rid)?;
                    let mut opts: Vec<(BrandedFreeBerth<'_, 'p, T>, TimePoint<T>)> = Vec::new();
                    if let Some(pt) = r.req().processing_time_for(cand.to_berth) {
                        for fb in ex
                            .iter_free_for(r.clone())
                            .filter(|fb| fb.berth().id() == cand.to_berth)
                        {
                            let iv = *fb.interval();
                            let w = r.req().feasible_window();
                            let lo = std::cmp::max(iv.start(), w.start());
                            let hi_iv = iv.end().checked_sub(pt);
                            let hi_w = w.end().checked_sub(pt);
                            if let (Some(hi_iv), Some(hi_w)) = (hi_iv, hi_w) {
                                let hi = std::cmp::min(hi_iv, hi_w);
                                if lo <= hi {
                                    if lo == hi {
                                        opts.push((fb.clone(), lo));
                                    } else {
                                        opts.push((fb.clone(), lo));
                                        opts.push((fb.clone(), hi));
                                    }
                                }
                            }
                        }
                    }
                    if opts.is_empty() {
                        return None;
                    }
                    Some((r, opts))
                });

                let mut placed = false;
                if let Some((r, mut opts)) = fallback {
                    if self.try_randomize_starts {
                        opts.shuffle(rng);
                    } else {
                        opts.sort_by_key(|(_, s)| {
                            let a = s.value();
                            let b = cand.to_start.value();
                            if a >= b { a - b } else { b - a }
                        });
                    }
                    for (fb, s) in opts {
                        if builder.propose_assignment(r.clone(), s, &fb).is_ok() {
                            placed = true;
                            break;
                        }
                    }
                }

                if placed {
                    placed_any = true;
                    continue;
                }

                let restored =
                    rollback_original(builder, cand.rid, cand.from_berth, cand.from_start);
                if !restored {
                    break;
                }
            }
        });

        match (placed_any, apply_res) {
            (true, Ok(plan)) => Some(plan),
            _ => None,
        }
    }
}

fn get_unassigned_brand<'brand, 'pb, 'p, T: Copy + Ord + CheckedAdd + CheckedSub>(
    ex: &crate::framework::planning::PlanExplorer<'brand, 'pb, 'p, T>,
    rid: RequestIdentifier,
) -> Option<BrandedRequest<'brand, 'p, FlexibleKind, T>> {
    ex.iter_unassigned_requests().find(|r| r.req().id() == rid)
}
