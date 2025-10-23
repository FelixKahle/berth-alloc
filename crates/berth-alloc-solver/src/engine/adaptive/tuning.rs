use num_traits::{Float, ToPrimitive};

use crate::{
    engine::adaptive::stats::OperatorStats,
    search::operator::{OperatorKind, OperatorTuning},
    state::solver_state::AdaptiveStats,
};

/// Engine-provided stagnation signal to influence tuning/exploration.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stagnation {
    pub stale_rounds: usize, // number of stale rounds since last accepted improvement
    /// 0..1 — how “stuck” the engine feels (strategy-specific heuristic).
    pub stuck_factor: f64,
}

/// Tuner that maps per-operator stats + global solver stats to a knob pack for the operator.
/// Generic on T at the trait level so it's object-safe for dyn.
pub trait OperatorTuner<T>: Send + Sync
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
{
    fn name(&self) -> &str;

    /// Compute a new tuning for a specific operator.
    ///
    /// - `kind`: which family (Local/Destroy/Repair)
    /// - `op_stats`: telemetry for this operator (EWMAs etc)
    /// - `stats`: global/problem stats derived from the current solver state
    /// - `stagnation`: engine-level stagnation signal for stronger diversification
    fn compute<'m, 'p>(
        &mut self,
        kind: OperatorKind,
        op_stats: &OperatorStats,
        stats: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> OperatorTuning;
}

pub struct DefaultOperatorTuner {
    pub target_runtime_ms: f64,
    pub min_intensity: f64,
    pub max_intensity: f64,
    pub min_greediness: f64,
    pub max_greediness: f64,
    pub min_locality: f64,
    pub max_locality: f64,
    pub min_perturb: f64,
    pub max_perturb: f64,
    pub low_acceptance: f64,
    pub good_acceptance: f64,
}

impl Default for DefaultOperatorTuner {
    fn default() -> Self {
        Self {
            target_runtime_ms: 6.0, // was 0.60
            min_intensity: 0.05,
            max_intensity: 1.00,
            min_greediness: 0.00,
            max_greediness: 1.00,
            min_locality: 0.00,
            max_locality: 1.00,
            min_perturb: 0.00,
            max_perturb: 1.00,
            low_acceptance: 0.10,
            good_acceptance: 0.35,
        }
    }
}

impl DefaultOperatorTuner {
    #[inline]
    fn clamp01(x: f64) -> f64 {
        x.max(0.0).min(1.0)
    }
}

impl<T> OperatorTuner<T> for DefaultOperatorTuner
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub + ToPrimitive,
{
    fn name(&self) -> &str {
        "DefaultOperatorTuner"
    }

    fn compute<'m, 'p>(
        &mut self,
        kind: OperatorKind,
        op: &OperatorStats,
        global: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> OperatorTuning {
        let ew_rt = op.ew_runtime_ms.value().unwrap_or(self.target_runtime_ms);
        let ew_delta_true = op.ew_delta_true.value().unwrap_or(0.0);
        let ar = op.ew_accept_prob.value().unwrap_or(0.0);

        let rt_ratio = (ew_rt / self.target_runtime_ms).max(0.0);
        let base_intensity = if ar < self.low_acceptance && rt_ratio > 1.2 {
            0.25 / rt_ratio
        } else if ar > self.good_acceptance && rt_ratio < 1.0 {
            (1.10 / rt_ratio).min(1.0)
        } else {
            (1.0 / rt_ratio).min(1.0)
        };
        let mut intensity = base_intensity.clamp(self.min_intensity, self.max_intensity);

        let improving_abs = (-ew_delta_true).max(0.0);
        let mut greediness = if ar > self.good_acceptance && improving_abs > 1e-6 {
            0.8
        } else if ar < self.low_acceptance {
            0.2
        } else {
            0.5
        }
        .clamp(self.min_greediness, self.max_greediness);

        // Locality uses utilization bias
        let util = global.berth_utilization();
        let kind_bias = match kind {
            OperatorKind::Local => 0.15,
            OperatorKind::Destroy => -0.10,
            OperatorKind::Repair => -0.05,
        };
        let mut locality = (util + kind_bias)
            .clamp(0.0, 1.0)
            .clamp(self.min_locality, self.max_locality);

        // Perturbation baseline
        let mut perturb = if ar < self.low_acceptance && improving_abs < 1e-3 {
            0.8
        } else if ar > self.good_acceptance && improving_abs > 1e-2 {
            0.1
        } else {
            0.3
        }
        .clamp(self.min_perturb, self.max_perturb);

        // Global stagnation blend: push diversification; favor larger destroys; lighten locals.
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);
        if sf > 0.0 {
            perturb = (perturb + 0.22 * sf).min(self.max_perturb);
            match kind {
                OperatorKind::Destroy => {
                    intensity = (intensity + 0.12 * sf).min(self.max_intensity);
                    greediness = greediness * (1.0 - 0.08 * sf);
                }
                OperatorKind::Local => {
                    intensity = (intensity * (1.0 - 0.10 * sf)).max(self.min_intensity);
                    greediness = greediness * (1.0 - 0.12 * sf);
                    locality =
                        (locality * (1.0 - 0.12 * sf)).clamp(self.min_locality, self.max_locality);
                }
                OperatorKind::Repair => {
                    greediness = greediness * (1.0 - 0.06 * sf);
                }
            }
        }

        OperatorTuning {
            intensity: Self::clamp01(intensity),
            greediness: Self::clamp01(greediness),
            locality: Self::clamp01(locality),
            perturb: Self::clamp01(perturb),
        }
    }
}

/// A wrapper tuner that enforces a per-operator soft time budget and clamps the
/// resulting intensity/greediness, to prevent "doing too much work" per propose().
pub struct WorkBudgetTuner {
    inner: DefaultOperatorTuner,
    /// Soft runtime budget per propose() in milliseconds.
    pub soft_time_ms: f64,
    /// Intensity bounds to constrain work size.
    pub min_intensity: f64,
    pub max_intensity: f64,
    /// Optional upper bound for greediness (helps keep sampling breadth).
    pub max_greediness: f64,
    /// Optional upper bound for locality (for Local ops).
    pub max_locality: f64,
}

impl Default for WorkBudgetTuner {
    fn default() -> Self {
        let inner = DefaultOperatorTuner::default(); // carries target_runtime_ms: 6.0
        Self {
            inner,
            soft_time_ms: 8.0, // was 0.90
            min_intensity: 0.08,
            max_intensity: 0.70,
            max_greediness: 0.90,
            max_locality: 0.95,
        }
    }
}

impl WorkBudgetTuner {
    pub fn with_soft_time_budget_ms(mut self, ms: f64) -> Self {
        self.soft_time_ms = ms.max(0.05);
        self
    }
    pub fn with_intensity_bounds(mut self, min_i: f64, max_i: f64) -> Self {
        self.min_intensity = min_i.clamp(0.0, 1.0);
        self.max_intensity = max_i.clamp(self.min_intensity, 1.0);
        self
    }
    pub fn with_max_greediness(mut self, g: f64) -> Self {
        self.max_greediness = g.clamp(0.0, 1.0);
        self
    }
    pub fn with_max_locality(mut self, l: f64) -> Self {
        self.max_locality = l.clamp(0.0, 1.0);
        self
    }

    /// Local heavy neighborhood operators (cross-exchanges, Or-Opt, hill climbers, RCL).
    pub fn local_heavy_strict() -> Self {
        Self::default()
            .with_soft_time_budget_ms(0.70)
            .with_intensity_bounds(0.05, 0.30)
            .with_max_greediness(0.70)
            .with_max_locality(0.70)
    }

    /// Heavy destroy operators (Shaw/time-cluster/band/processing-time, berth band, string-block).
    pub fn destroy_heavy_defaults() -> Self {
        Self::default()
            .with_soft_time_budget_ms(0.80)
            .with_intensity_bounds(0.05, 0.30)
            .with_max_greediness(0.75)
    }

    /// Repair operators (randomized greedy, k-regret).
    pub fn repair_defaults() -> Self {
        Self::default()
            .with_soft_time_budget_ms(0.80)
            .with_intensity_bounds(0.05, 0.25)
            .with_max_greediness(0.70)
    }
}

impl<T> OperatorTuner<T> for WorkBudgetTuner
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub + ToPrimitive,
{
    fn name(&self) -> &str {
        "WorkBudgetTuner"
    }

    fn compute<'m, 'p>(
        &mut self,
        kind: OperatorKind,
        op_stats: &OperatorStats,
        stats: &crate::state::solver_state::AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> crate::search::operator::OperatorTuning {
        // Start with default heuristic
        let mut t = self.inner.compute(kind, op_stats, stats, stagnation);

        // Gentler scaling when over budget (linear-ish)
        let ew_rt = op_stats.ew_runtime_ms.value().unwrap_or(self.soft_time_ms);
        if ew_rt.is_finite() && ew_rt > self.soft_time_ms {
            let over = (ew_rt / self.soft_time_ms).max(1.0);
            // scale ∈ (0,1], e.g. 1.0 at on-budget; ~0.62 at 1.6×; ~0.45 at 2.0×
            let scale = 1.0 / (1.0 + 0.6 * (over - 1.0));
            t.intensity *= scale;

            // Clamp greediness moderately when notably over
            if over > 1.6 {
                t.greediness = t.greediness.min(self.max_greediness);
            }
            if over > 2.0 {
                t.greediness = t.greediness.min(0.60);
            }

            if let OperatorKind::Local = kind {
                if over > 2.0 {
                    t.locality = t.locality.min(0.60);
                } else {
                    t.locality = t.locality.min(self.max_locality);
                }
            }
        }

        // Under-budget + improving → carefully ramp up
        let ar = op_stats.ew_accept_prob.value().unwrap_or(0.0);
        let ew_delta_true = op_stats.ew_delta_true.value().unwrap_or(0.0);
        let imp = (-ew_delta_true).max(0.0);

        if ew_rt.is_finite()
            && ew_rt <= 0.70 * self.soft_time_ms
            && ar >= self.inner.good_acceptance
            && imp > 1e-3
        {
            t.intensity = (t.intensity * 1.20).min(self.max_intensity);
            t.greediness = (t.greediness + 0.05).min(self.max_greediness);
            if let OperatorKind::Local = kind {
                t.locality = (t.locality + 0.05).min(self.max_locality);
            }
        }

        // Stuck → diversify more; amplify for destroy when utilization is high
        let util = stats.berth_utilization();
        if ar <= self.inner.low_acceptance && imp <= 1e-4 {
            let mut bump = 0.10 + 0.15 * util;
            if let OperatorKind::Destroy = kind {
                bump += 0.05;
            }
            t.perturb = (t.perturb + bump).min(1.0);
            t.greediness = (t.greediness * 0.85).min(self.max_greediness);
            if let OperatorKind::Local = kind {
                t.locality = t.locality.min(self.max_locality * 0.95);
            }
        }

        // Global stagnation (final blend)
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);
        if sf > 0.0 {
            t.perturb = (t.perturb + 0.20 * sf).min(1.0);
            match kind {
                OperatorKind::Destroy => {
                    t.intensity = (t.intensity + 0.10 * sf).min(self.max_intensity);
                    t.greediness = (t.greediness * (1.0 - 0.06 * sf)).min(self.max_greediness);
                }
                OperatorKind::Local => {
                    t.greediness = (t.greediness * (1.0 - 0.10 * sf)).min(self.max_greediness);
                    t.locality = (t.locality * (1.0 - 0.12 * sf)).min(self.max_locality);
                }
                OperatorKind::Repair => {
                    t.greediness = (t.greediness * (1.0 - 0.05 * sf)).min(self.max_greediness);
                }
            }
        }

        // Final profile clamps
        t.intensity = t.intensity.clamp(self.min_intensity, self.max_intensity);
        t.greediness = t.greediness.min(self.max_greediness);
        if let OperatorKind::Local = kind {
            t.locality = t.locality.min(self.max_locality);
        }

        t
    }
}

/// Map a desired absolute count of attempts k_desired for a family to an intensity in [0,1]
/// using the current assigned request count n (n>=1): intensity = k_desired / n.
#[inline]
fn intensity_for_target_count(n_assigned: usize, k_desired: f64) -> f64 {
    if n_assigned == 0 {
        0.0
    } else {
        (k_desired / (n_assigned as f64)).clamp(0.0, 1.0)
    }
}

/// Base policy helpers shared by the per-operator tuners.
#[inline]
fn accept_prob(op: &crate::engine::adaptive::stats::OperatorStats) -> f64 {
    op.ew_accept_prob.value().unwrap_or(0.0).clamp(0.0, 1.0)
}
#[inline]
fn improv_abs(op: &crate::engine::adaptive::stats::OperatorStats) -> f64 {
    (-op.ew_delta_true.value().unwrap_or(0.0)).max(0.0)
}
#[inline]
fn ew_runtime_ms(op: &crate::engine::adaptive::stats::OperatorStats, fallback: f64) -> f64 {
    op.ew_runtime_ms.value().unwrap_or(fallback)
}

/// Generic “absolute count target” tuner for local, candidate-based operators
/// like ShiftEarlierOnSameBerth, RelocateSingleBest, hill-climbers, and pair-attempts.
/// You provide a baseline [min,max] absolute attempt count band that you’d like to
/// emulate (e.g., the old hardcoded ranges), and this tuner converts it to an intensity
/// via intensity = target_k / n_assigned, then applies runtime/stagnation adjustments.
pub struct LocalCountTargetTuner {
    pub k_min: f64,
    pub k_max: f64,
    pub soft_time_ms: f64,
    pub min_intensity: f64,
    pub max_intensity: f64,
    pub max_greediness: f64,
    pub max_locality: f64,
}

impl Default for LocalCountTargetTuner {
    fn default() -> Self {
        Self {
            k_min: 8.0,
            k_max: 96.0,
            soft_time_ms: 0.75,
            min_intensity: 0.01,
            max_intensity: 0.40,
            max_greediness: 0.80,
            max_locality: 0.85,
        }
    }
}

impl LocalCountTargetTuner {
    pub fn new(k_min: f64, k_max: f64) -> Self {
        Self {
            k_min,
            k_max,
            ..Default::default()
        }
    }
    #[inline]
    fn pick_target_k(
        &self,
        op: &crate::engine::adaptive::stats::OperatorStats,
        stagnation: &Stagnation,
    ) -> f64 {
        let ar = accept_prob(op);
        let imp = improv_abs(op);
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);

        // Start mid-band; push up if improving and accepted; push down if stuck/slow.
        let mut t = 0.50;
        if ar >= 0.35 && imp > 1e-3 {
            t = 0.70;
        } else if ar <= 0.10 && imp <= 1e-4 {
            t = 0.35;
        }
        // When globally stuck, increase t moderately to sample a bit wider.
        t = (t + 0.20 * sf).clamp(0.30, 0.90);

        self.k_min + t * (self.k_max - self.k_min)
    }
}

impl<T> OperatorTuner<T> for LocalCountTargetTuner
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub + ToPrimitive,
{
    fn name(&self) -> &str {
        "LocalCountTargetTuner"
    }

    fn compute<'m, 'p>(
        &mut self,
        kind: OperatorKind,
        op_stats: &OperatorStats,
        stats: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> OperatorTuning {
        debug_assert!(matches!(kind, OperatorKind::Local));

        let n = stats.number_assigned_requests().max(1);
        let mut target_k = self.pick_target_k(op_stats, stagnation);

        // Runtime backoff or under-budget ramp
        let rt = ew_runtime_ms(op_stats, self.soft_time_ms);
        if rt.is_finite() && rt > self.soft_time_ms {
            let over = (rt / self.soft_time_ms).max(1.0);
            let gamma = if over > 2.5 {
                2.2
            } else if over > 1.6 {
                1.8
            } else {
                1.4
            };
            target_k *= (1.0 / over.powf(gamma)).clamp(0.20, 1.0);
        } else if rt.is_finite()
            && rt < 0.70 * self.soft_time_ms
            && accept_prob(op_stats) > 0.35
            && improv_abs(op_stats) > 1e-3
        {
            target_k *= 1.20;
        }

        // Convert absolute target to intensity
        let mut intensity =
            intensity_for_target_count(n, target_k).clamp(self.min_intensity, self.max_intensity);

        // Greediness/locality policy for local ops; pull toward exploration when stuck
        let mut greed = 0.55;
        if accept_prob(op_stats) <= 0.10 && improv_abs(op_stats) <= 1e-4 {
            greed *= 0.80;
        }
        let mut locality = (stats.berth_utilization() + 0.10).clamp(0.0, 1.0);
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);
        if sf > 0.0 {
            greed *= 1.0 - 0.10 * sf;
            locality *= 1.0 - 0.10 * sf;
        }

        // Final clamps
        intensity = intensity.clamp(self.min_intensity, self.max_intensity);
        greed = greed.min(self.max_greediness);
        locality = locality.min(self.max_locality);

        OperatorTuning {
            intensity,
            greediness: greed,
            locality,
            perturb: (0.25 + 0.25 * sf).min(1.0),
        }
    }
}

/// Or-Opt k-length tuner. Converts a desired small k-band (e.g., 6..10) into intensity via k/n.
pub struct OrOptBlockKTuner {
    pub k_min: f64,
    pub k_max: f64,
    pub soft_time_ms: f64,
    pub min_intensity: f64,
    pub max_intensity: f64,
    pub max_greediness: f64,
}

impl Default for OrOptBlockKTuner {
    fn default() -> Self {
        Self {
            k_min: 6.0,
            k_max: 10.0,
            soft_time_ms: 0.75,
            min_intensity: 0.002, // allow very small k/n
            max_intensity: 0.12,  // avoid huge k on large n
            max_greediness: 0.80,
        }
    }
}

impl<T> OperatorTuner<T> for OrOptBlockKTuner
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub + ToPrimitive,
{
    fn name(&self) -> &str {
        "OrOptBlockKTuner"
    }

    fn compute<'m, 'p>(
        &mut self,
        _kind: OperatorKind,
        op_stats: &OperatorStats,
        stats: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> OperatorTuning {
        let n = stats.number_assigned_requests().max(1);
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);

        // Favor upper k when improving; lower when stuck or slow.
        let ar = accept_prob(op_stats);
        let imp = improv_abs(op_stats);
        let mut k = if ar >= 0.35 && imp > 1e-3 {
            self.k_max
        } else {
            self.k_min
        };

        let rt = ew_runtime_ms(op_stats, self.soft_time_ms);
        if rt.is_finite() && rt > self.soft_time_ms {
            let over = (rt / self.soft_time_ms).max(1.0);
            k *= (1.0 / over.powf(1.6)).clamp(0.30, 1.0);
        }
        if sf > 0.0 && ar <= 0.10 {
            k *= 0.85;
        }

        let intensity =
            intensity_for_target_count(n, k).clamp(self.min_intensity, self.max_intensity);
        let greed = (0.55 + 0.10 * (ar >= 0.35) as i32 as f64).min(self.max_greediness);

        OperatorTuning {
            intensity,
            greediness: greed,
            locality: (stats.berth_utilization() + 0.10).clamp(0.0, 1.0),
            perturb: (0.25 + 0.20 * sf).min(1.0),
        }
    }
}

/// Destroy ratio tuner: maps desired removal ratio band directly to intensity (ratio=intensity).
pub struct DestroyRatioTuner {
    pub ratio_min: f64,
    pub ratio_max: f64,
    pub soft_time_ms: f64,
}

impl Default for DestroyRatioTuner {
    fn default() -> Self {
        Self {
            ratio_min: 0.22,
            ratio_max: 0.46,
            soft_time_ms: 0.90,
        }
    }
}

impl<T> OperatorTuner<T> for DestroyRatioTuner
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub + ToPrimitive,
{
    fn name(&self) -> &str {
        "DestroyRatioTuner"
    }

    fn compute<'m, 'p>(
        &mut self,
        _kind: OperatorKind,
        op_stats: &OperatorStats,
        stats: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> OperatorTuning {
        let ar = accept_prob(op_stats);
        let imp = improv_abs(op_stats);
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);

        // Choose ratio within band; higher when stuck, lower when runtime high.
        let mut t = 0.55;
        if ar >= 0.35 && imp > 1e-3 {
            t = 0.45;
        } else if ar <= 0.10 && imp <= 1e-4 {
            t = 0.65;
        }
        t = (t + 0.20 * sf).clamp(0.25, 0.90);
        let mut ratio = self.ratio_min + t * (self.ratio_max - self.ratio_min);

        // Runtime trim
        let rt = ew_runtime_ms(op_stats, self.soft_time_ms);
        if rt.is_finite() && rt > self.soft_time_ms {
            let over = (rt / self.soft_time_ms).max(1.0);
            ratio *= (1.0 / over.powf(1.5)).clamp(0.30, 1.0);
        }

        OperatorTuning {
            intensity: ratio.clamp(0.0, 1.0), // destroy ops read intensity as ratio
            greediness: 0.55,
            locality: (stats.berth_utilization() - 0.10).clamp(0.0, 1.0),
            perturb: (0.40 + 0.30 * sf).min(1.0),
        }
    }
}

/// K-Regret tuner: maps desired k band to intensity via k/n.
pub struct KRegretKTuner {
    pub k_min: f64,
    pub k_max: f64,
    pub soft_time_ms: f64,
    pub min_intensity: f64,
    pub max_intensity: f64,
}

impl Default for KRegretKTuner {
    fn default() -> Self {
        Self {
            k_min: 9.0,
            k_max: 11.0,
            soft_time_ms: 0.80,
            min_intensity: 0.002,
            max_intensity: 0.20,
        }
    }
}

impl<T> OperatorTuner<T> for KRegretKTuner
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub + ToPrimitive,
{
    fn name(&self) -> &str {
        "KRegretKTuner"
    }

    fn compute<'m, 'p>(
        &mut self,
        _kind: OperatorKind,
        op_stats: &OperatorStats,
        stats: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
    ) -> OperatorTuning {
        let n = stats.number_assigned_requests().max(1);
        let ar = accept_prob(op_stats);
        let imp = improv_abs(op_stats);
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);

        // Aim for upper k when productive; lower when stuck or slow.
        let mut k = if ar >= 0.35 && imp > 1e-3 {
            self.k_max
        } else {
            self.k_min
        };

        let rt = ew_runtime_ms(op_stats, self.soft_time_ms);
        if rt.is_finite() && rt > self.soft_time_ms {
            k *= (self.soft_time_ms / rt).clamp(0.30, 1.0);
        }
        if sf > 0.0 && ar <= 0.10 {
            k *= 0.90;
        }

        let intensity =
            intensity_for_target_count(n, k).clamp(self.min_intensity, self.max_intensity);
        OperatorTuning {
            intensity,
            greediness: 0.60,
            locality: 0.30,
            perturb: (0.25 + 0.20 * sf).min(1.0),
        }
    }
}
