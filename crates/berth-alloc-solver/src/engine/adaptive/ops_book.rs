use rand::Rng;

use crate::{
    engine::adaptive::{selection::OperatorSelector, stats::OperatorStats, tuning::OperatorTuner},
    search::operator::{OperatorKind, OperatorTuning},
    state::solver_state::AdaptiveStats,
};

/// Per-kind (Local/Destroy/Repair) book that tracks per-operator stats, tuners and a selector.
pub struct OperatorBook<T, R>
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
    R: Rng,
{
    kind: OperatorKind,
    pub stats: Vec<OperatorStats>,
    tuners: Vec<Box<dyn OperatorTuner<T>>>,
    selector: Box<dyn OperatorSelector<T, R>>,
    last_tuning: Vec<OperatorTuning>,
}

impl<T, R> OperatorBook<T, R>
where
    T: Copy + Ord + num_traits::CheckedAdd + num_traits::CheckedSub,
    R: Rng,
{
    pub fn new(kind: OperatorKind, selector: Box<dyn OperatorSelector<T, R>>) -> Self {
        Self {
            kind,
            stats: Vec::new(),
            tuners: Vec::new(),
            selector,
            last_tuning: Vec::new(),
        }
    }

    /// Register an operator with its tuner. Returns its index.
    pub fn register_operator(&mut self, tuner: Box<dyn OperatorTuner<T>>) -> usize {
        let idx = self.stats.len();
        self.stats.push(OperatorStats::default());
        self.tuners.push(tuner);
        self.last_tuning.push(OperatorTuning::default());
        idx
    }

    /// Replace selector at runtime.
    pub fn set_selector(&mut self, selector: Box<dyn OperatorSelector<T, R>>) {
        self.selector = selector;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// Compute fresh tunings for all operators using their registered tuner.
    pub fn retune_all<'m, 'p>(
        &mut self,
        global: &AdaptiveStats<'m, 'p, T>,
        stagnation: &crate::engine::adaptive::tuning::Stagnation,
    ) {
        for i in 0..self.stats.len() {
            self.last_tuning[i] =
                self.tuners[i].compute(self.kind, &self.stats[i], global, stagnation);
        }
    }

    /// Get the last computed tuning for an operator.
    #[inline]
    pub fn tuning_for(&self, idx: usize) -> &OperatorTuning {
        &self.last_tuning[idx]
    }

    /// Select the next operator index using the current selector.
    pub fn select<'m, 'p>(
        &mut self,
        global: &AdaptiveStats<'m, 'p, T>,
        stagnation: &crate::engine::adaptive::tuning::Stagnation,
        rng: &mut R,
    ) -> usize {
        if self.len() == 0 {
            return 0;
        }
        self.selector
            .pick(self.kind, &self.stats, global, stagnation, rng)
            .min(self.len().saturating_sub(1))
    }

    /// Record timing for propose() start for a specific operator (index).
    #[inline]
    pub fn propose_started(&self) -> std::time::Instant {
        OperatorStats::propose_started()
    }

    /// Record whether propose() produced a plan for this operator.
    #[inline]
    pub fn record_propose(&mut self, idx: usize, start: std::time::Instant, produced: bool) {
        if let Some(s) = self.stats.get_mut(idx) {
            s.record_propose(start, produced);
        }
    }

    /// Record acceptance outcome and the true/base delta (new - old).
    #[inline]
    pub fn record_outcome(&mut self, idx: usize, accepted: bool, delta_true: f64) {
        if let Some(s) = self.stats.get_mut(idx) {
            s.record_outcome(accepted, delta_true);
        }
    }
}
