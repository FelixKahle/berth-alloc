use crate::{
    engine::adaptive::{stats::OperatorStats, tuning::Stagnation},
    search::operator::OperatorKind,
    state::solver_state::AdaptiveStats,
};
use rand::Rng;

/// Selector decides which operator index to run from a given set.
pub trait OperatorSelector<T, R>: Send + Sync
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
    R: Rng,
{
    fn name(&self) -> &str;

    /// Return index in [0, n) to select.
    fn pick<'m, 'p>(
        &mut self,
        kind: OperatorKind,
        stats: &[OperatorStats],
        global: &AdaptiveStats<'m, 'p, T>,
        stagnation: &Stagnation,
        rng: &mut R,
    ) -> usize;
}

/// Softmax/Boltzmann selector over "expected improvement per millisecond".
/// Score_i = softmax( (impr_per_ms_i + exploration_bonus_i) / tau )
pub struct SoftmaxSelector {
    pub tau: f64,       // base temperature
    pub explore_c: f64, // exploration strength for small samples
    pub min_runtime_ms: f64,
}

impl Default for SoftmaxSelector {
    fn default() -> Self {
        Self {
            tau: 0.25,       // slightly cooler start; warms with stagnation
            explore_c: 0.35, // gentler UCB
            min_runtime_ms: 0.20,
        }
    }
}

impl SoftmaxSelector {
    /// Expected gain per *call* per millisecond.
    /// E[gain/ms] = p(produce) * p(accept | produce) * mean_gain_when_accepted / runtime_ms
    /// where gain = -delta_true (improvement ≥ 0).
    fn score(op: &OperatorStats, min_rt: f64) -> f64 {
        let rt = op.ew_runtime_ms.value().unwrap_or(min_rt).max(min_rt);

        // mean accepted delta (negative is good → gain is -delta)
        let mean_delta = op.ew_delta_true.value().unwrap_or(0.0);
        let gain_when_acc = (-mean_delta).max(0.0);

        let p_prod = op.production_ratio(); // successes / calls
        let p_acc = op.acceptance_ratio_when_produced(); // accepts / successes

        // No extra slow penalty; we already divide by runtime.
        (p_prod * p_acc * gain_when_acc) / rt
    }
}

impl<T, R> OperatorSelector<T, R> for SoftmaxSelector
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
    R: Rng,
{
    fn name(&self) -> &str {
        "SoftmaxSelector"
    }

    fn pick<'m, 'p>(
        &mut self,
        _kind: OperatorKind,
        ops: &[OperatorStats],
        _global: &AdaptiveStats<'m, 'p, T>,
        stagnation: &crate::engine::adaptive::tuning::Stagnation,
        rng: &mut R,
    ) -> usize {
        if ops.is_empty() {
            return 0;
        }
        if ops.len() == 1 {
            return 0;
        }

        // Cold-start: prefer never-called ops
        let mut cold = Vec::new();
        let mut total_calls: u64 = 0;
        for (i, op) in ops.iter().enumerate() {
            total_calls = total_calls.saturating_add(op.calls);
            if op.calls == 0 {
                cold.push(i);
            }
        }
        if !cold.is_empty() {
            return cold[rng.random_range(0..cold.len())];
        }

        let min_rt = self.min_runtime_ms.max(1e-6);
        let total_calls_f = (total_calls as f64).max(1.0);
        let log_total = (total_calls_f + 1.0).ln();

        let mut scores = Vec::with_capacity(ops.len());
        let mut calls_f = Vec::with_capacity(ops.len());
        let mut calls_sum = 0.0;

        for op in ops {
            let base = Self::score(op, min_rt);
            // UCB-style exploration (moderate)
            let c_i = op.calls as f64;
            let bonus = self.explore_c * ((log_total / (c_i + 1.0)).sqrt());
            scores.push(base + bonus);
            calls_f.push(c_i);
            calls_sum += c_i;
        }

        // Stagnation-aware nudge: lightly bias against dominant ops
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);
        if sf > 0.0 {
            let mean_calls = (calls_sum / (ops.len() as f64)).max(1.0);
            for (i, s) in scores.iter_mut().enumerate() {
                let usage_penalty = 1.0 / (1.0 + (calls_f[i] / mean_calls).sqrt());
                *s *= (1.0 - 0.10 * sf) + 0.10 * sf * usage_penalty;
            }
        }

        // Reduce least-used bias cap (less random thrash)
        let least_used_bias_cap = (0.06 + 0.18 * sf).min(0.15);
        if sf > 0.0 && rng.random::<f64>() < least_used_bias_cap {
            // pick from least-used quartile
            let mut idxs: Vec<usize> = (0..ops.len()).collect();
            idxs.sort_unstable_by_key(|&i| ops[i].calls);
            let q = ((ops.len() as f64) * 0.25)
                .round()
                .clamp(1.0, ops.len() as f64) as usize;
            let least = &idxs[..q];
            return least[rng.random_range(0..least.len())];
        }

        // Temperature schedule
        let mut tau = self.tau.max(1e-6);
        if total_calls < 200 {
            tau = 0.25;
        } else if total_calls < 2000 {
            tau = 0.20;
        } else {
            tau = tau.max(0.12);
        }
        tau = (tau * (1.0 + 2.0 * sf)).min(1.0);

        // Softmax over scores
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| ((s - max_s) / tau).exp()).collect();
        let sum: f64 = exps.iter().sum();

        // Best-of-two sampling
        let draw = |rng: &mut R| {
            let mut r = rng.random::<f64>() * sum;
            for (i, w) in exps.iter().enumerate() {
                if r <= *w {
                    return i;
                }
                r -= *w;
            }
            exps.len() - 1
        };
        let i1 = draw(rng);
        let i2 = draw(rng);
        if scores[i2] > scores[i1] { i2 } else { i1 }
    }
}

/// Epsilon-greedy on acceptance-weighted improvement per ms.
pub struct EpsilonGreedySelector {
    pub epsilon: f64,
    pub min_runtime_ms: f64,
}

impl Default for EpsilonGreedySelector {
    fn default() -> Self {
        Self {
            epsilon: 0.10,
            min_runtime_ms: 0.20,
        }
    }
}

impl EpsilonGreedySelector {
    fn score(op: &OperatorStats, min_rt: f64) -> f64 {
        let rt = op.ew_runtime_ms.value().unwrap_or(min_rt).max(min_rt);
        let mean_delta = op.ew_delta_true.value().unwrap_or(0.0);
        let gain_when_acc = (-mean_delta).max(0.0);
        let p_prod = op.production_ratio();
        let p_acc = op.acceptance_ratio_when_produced();
        (p_prod * p_acc * gain_when_acc) / rt
    }
}

impl<T, R> OperatorSelector<T, R> for EpsilonGreedySelector
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
    R: Rng,
{
    fn name(&self) -> &str {
        "EpsilonGreedySelector"
    }

    fn pick<'m, 'p>(
        &mut self,
        _kind: OperatorKind,
        ops: &[OperatorStats],
        _global: &AdaptiveStats<'m, 'p, T>,
        stagnation: &crate::engine::adaptive::tuning::Stagnation,
        rng: &mut R,
    ) -> usize {
        if ops.is_empty() {
            return 0;
        }
        if ops.len() == 1 {
            return 0;
        }

        // Cold-start preference
        let mut total_calls: u64 = 0;
        let mut cold = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            total_calls = total_calls.saturating_add(op.calls);
            if op.calls == 0 {
                cold.push(i);
            }
        }

        // Decaying epsilon with stagnation escalator
        let base_eps = (self.epsilon / (1.0 + (total_calls as f64)).sqrt()).max(0.02);
        let sf = stagnation.stuck_factor.clamp(0.0, 1.0);
        let eps = (base_eps * (1.0 + 1.5 * sf)).min(0.40);

        if rng.random::<f64>() < eps {
            if !cold.is_empty() {
                return cold[rng.random_range(0..cold.len())];
            }
            // Bias exploration to least-used quartile when stuck
            if sf > 0.0 {
                let mut idxs: Vec<usize> = (0..ops.len()).collect();
                idxs.sort_unstable_by_key(|&i| ops[i].calls);
                let q = ((ops.len() as f64) * 0.25).max(1.0) as usize;
                let least = &idxs[..q];
                return least[rng.random_range(0..least.len())];
            }
            return rng.random_range(0..ops.len());
        }

        // Exploit
        let mut best_i = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        let min_rt = self.min_runtime_ms.max(1e-6);
        for (i, op) in ops.iter().enumerate() {
            let s = Self::score(op, min_rt);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        best_i
    }
}
