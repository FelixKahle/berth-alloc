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

#[derive(Debug, Clone, PartialEq)]
pub struct StatsConfig {
    pub bootstrap_success_rate: f64,
    pub min_ns_per_proposal: f64,
    pub reward_alpha: f64,
    pub gen_time_alpha: f64,
    pub eval_time_alpha: f64,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            bootstrap_success_rate: 0.05,
            min_ns_per_proposal: 100.0,
            reward_alpha: 0.20,
            gen_time_alpha: 0.25,
            eval_time_alpha: 0.25,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllocationConfig {
    /// Target number of proposal attempts per meta-iteration (round).
    pub target_total_proposals_per_round: usize,
    /// Lower bound on proposals assigned to each operator (after softmax).
    pub min_per_op: usize,
    /// Upper bound on proposals assigned to each operator (after softmax).
    pub max_per_op: usize,
    /// Fraction of proposals reserved for uniform exploration (0..=1).
    pub explore_frac: f64,
    /// Weight of speed signal (1 / time per proposal) in softmax.
    pub speed_weight: f64,
    /// Weight of success rate in softmax.
    pub success_weight: f64,
    /// Minimum temperature for softmax (τ).
    pub softmax_tau_min: f64,
    /// Maximum temperature for softmax (τ).
    pub softmax_tau_max: f64,
    /// Numerical epsilon to guard division-by-zero / underflow.
    pub softmax_eps: f64,
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self {
            target_total_proposals_per_round: 4096,
            min_per_op: 128,
            max_per_op: 4096,
            explore_frac: 0.15,
            speed_weight: 0.7,
            success_weight: 0.3,
            softmax_tau_min: 0.02,
            softmax_tau_max: 0.25,
            softmax_eps: 1e-9,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnnealingConfig {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 1.0,
            cooling_rate: 0.9995,
            min_temperature: 1e-9,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomConfig {
    pub seed_base_task: u64,
    pub seed_base_select: u64,
}

impl Default for RandomConfig {
    fn default() -> Self {
        Self {
            seed_base_task: 828927561741,
            seed_base_select: 3735928559,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetaConfig {
    /// Hard time budget for the whole meta solve (milliseconds).
    pub max_solver_time_ms: u64,
    /// Stats / smoothing configuration.
    pub stats: StatsConfig,
    /// Allocation / softmax configuration.
    pub alloc: AllocationConfig,
    /// Simulated annealing parameters used in selection.
    pub anneal: AnnealingConfig,
    /// RNG seeds.
    pub random: RandomConfig,
}

impl Default for MetaConfig {
    fn default() -> Self {
        Self {
            max_solver_time_ms: 20000, // 10 seconds
            stats: StatsConfig::default(),
            alloc: AllocationConfig::default(),
            anneal: AnnealingConfig::default(),
            random: RandomConfig::default(),
        }
    }
}
