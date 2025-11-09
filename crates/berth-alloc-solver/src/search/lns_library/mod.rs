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

pub mod repair;
pub mod ruin;

use crate::search::{
    eval::CostEvaluator,
    lns::{RandomRuinRepairPerturbPair, RepairProcedure, RuinProcedure},
    lns_library::{
        repair::{BestFitBySlackInsertionRepair, CheapestInsertionRepair, RegretInsertionRepair},
        ruin::{RandomSubsetRuin, RandomWalkRuin, RelatedRuin, SameBerthBlockRuin, TimeBandRuin},
    },
    neighboors::Neighboors,
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

#[derive(Debug, Clone, Copy)]
pub struct RuinSelectionConfig {
    pub use_random_subset: bool,
    pub use_time_band: bool,
    pub use_same_berth_block: bool,
    pub use_random_walk: bool,
    pub use_related_neighbors: bool,

    pub random_subset_k: usize,
    pub time_band_len: usize,
    pub same_berth_block_len: usize,
    pub random_walk_steps: usize,
    pub random_walk_same_berth_bias: f64,
    pub related_k: usize,
}

impl RuinSelectionConfig {
    #[inline]
    pub fn num_enabled(&self) -> usize {
        (self.use_random_subset as usize)
            + (self.use_time_band as usize)
            + (self.use_same_berth_block as usize)
            + (self.use_random_walk as usize)
            + (self.use_related_neighbors as usize)
    }
}

impl Default for RuinSelectionConfig {
    fn default() -> Self {
        Self {
            use_random_subset: true,
            use_time_band: true,
            use_same_berth_block: true,
            use_random_walk: true,
            use_related_neighbors: true,
            random_subset_k: 30,
            time_band_len: 150,
            same_berth_block_len: 30,
            random_walk_steps: 40,
            random_walk_same_berth_bias: 0.8,
            related_k: 20,
        }
    }
}

impl std::fmt::Display for RuinSelectionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RuinSelectionConfig {{ \
            use_random_subset: {}, use_time_band: {}, use_same_berth_block: {}, \
            use_random_walk: {}, use_related_neighbors: {}, \
            random_subset_k: {}, time_band_len: {}, same_berth_block_len: {}, \
            random_walk_steps: {}, random_walk_same_berth_bias: {}, related_k: {} \
            }}",
            self.use_random_subset,
            self.use_time_band,
            self.use_same_berth_block,
            self.use_random_walk,
            self.use_related_neighbors,
            self.random_subset_k,
            self.time_band_len,
            self.same_berth_block_len,
            self.random_walk_steps,
            self.random_walk_same_berth_bias,
            self.related_k,
        )
    }
}

/// Selection config for building a set of repair procedures.
#[derive(Debug, Clone, Copy)]
pub struct RepairSelectionConfig {
    pub use_cheapest_insertion: bool,
    pub use_best_fit_by_slack: bool,
    pub use_regret_insertion: bool,
}

impl RepairSelectionConfig {
    #[inline]
    pub fn num_enabled(&self) -> usize {
        (self.use_cheapest_insertion as usize)
            + (self.use_best_fit_by_slack as usize)
            + (self.use_regret_insertion as usize)
    }
}

impl Default for RepairSelectionConfig {
    fn default() -> Self {
        Self {
            use_cheapest_insertion: true,
            use_best_fit_by_slack: true,
            use_regret_insertion: true,
        }
    }
}

impl std::fmt::Display for RepairSelectionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RepairSelectionConfig {{ use_cheapest_insertion: {}, use_best_fit_by_slack: {}, use_regret_insertion: {} }}",
            self.use_cheapest_insertion, self.use_best_fit_by_slack, self.use_regret_insertion
        )
    }
}

/// Build a list of RuinProcedure trait objects according to the selection config.
/// Pass `Some(&neighboors)` to enable `RelatedRuin` (uses neighboors.neighbors).
#[inline]
pub fn make_ruin_list<'n, T, C, R>(
    config: &RuinSelectionConfig,
    neighboors: &Neighboors<'n>,
) -> Vec<Box<dyn RuinProcedure<T, C, R> + 'n>>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    let mut out: Vec<Box<dyn RuinProcedure<T, C, R> + 'n>> =
        Vec::with_capacity(config.num_enabled());

    if config.use_random_subset {
        out.push(Box::new(RandomSubsetRuin::new(config.random_subset_k)));
    }
    if config.use_time_band {
        out.push(Box::new(TimeBandRuin::new(config.time_band_len)));
    }
    if config.use_same_berth_block {
        out.push(Box::new(SameBerthBlockRuin::new(
            config.same_berth_block_len,
        )));
    }
    if config.use_random_walk {
        out.push(Box::new(RandomWalkRuin::new(
            config.random_walk_steps,
            config.random_walk_same_berth_bias,
        )));
    }
    if config.use_related_neighbors {
        out.push(Box::new(RelatedRuin::new(
            config.related_k,
            neighboors.neighbors.clone(),
        )));
    }

    out
}

/// Build a list of RepairProcedure trait objects according to the selection config.
#[inline]
pub fn make_repair_list<'n, T, C, R>(
    config: &RepairSelectionConfig,
) -> Vec<Box<dyn RepairProcedure<T, C, R> + 'n>>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    let mut out: Vec<Box<dyn RepairProcedure<T, C, R>>> = Vec::with_capacity(config.num_enabled());

    if config.use_cheapest_insertion {
        out.push(Box::new(CheapestInsertionRepair));
    }
    if config.use_best_fit_by_slack {
        out.push(Box::new(BestFitBySlackInsertionRepair::new()));
    }
    if config.use_regret_insertion {
        out.push(Box::new(RegretInsertionRepair));
    }

    out
}

#[inline]
pub fn make_random_ruin_repair_perturb_pair<'n, T, C, R>(
    ruin_cfg: &RuinSelectionConfig,
    repair_cfg: &RepairSelectionConfig,
    neighboors: &Neighboors<'n>,
) -> RandomRuinRepairPerturbPair<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    let ruins = make_ruin_list::<T, C, R>(ruin_cfg, neighboors);
    let repairs = make_repair_list::<T, C, R>(repair_cfg);
    RandomRuinRepairPerturbPair::new(ruins, repairs)
}
