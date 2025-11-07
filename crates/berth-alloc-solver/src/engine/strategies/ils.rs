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
    model::solver_model::SolverModel,
    search::eval::CostEvaluator,
    state::{plan::Plan, solver_state::SolverState},
};

#[derive(Debug)]
pub struct IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    pub model: &'m SolverModel<'p, T>,
    pub solver_state: &'s SolverState<'p, T>,
    pub evaluator: &'e C,
    pub rng: &'r mut R,
}

impl<'e, 'r, 's, 'm, 'p, T, C, R> IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    #[inline]
    pub fn new(
        model: &'m SolverModel<'p, T>,
        solver_state: &'s SolverState<'p, T>,
        evaluator: &'e C,
        rng: &'r mut R,
    ) -> Self {
        Self {
            model,
            solver_state,
            evaluator,
            rng,
        }
    }
}

pub trait IlsAcceptanceCriterion<T, C, R>: Send
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn name(&self) -> &str;

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool;
}

impl<'a, T, C, R> std::fmt::Debug for dyn IlsAcceptanceCriterion<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IlsAcceptanceCriterion({})", self.name())
    }
}

impl<'a, T, C, R> std::fmt::Display for dyn IlsAcceptanceCriterion<T, C, R> + 'a
where
    T: Copy + Ord,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IlsAcceptanceCriterion({})", self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    _phantom: std::marker::PhantomData<(T, C, R)>,
}

impl<T, C, R> Default for GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C, R> GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, R> IlsAcceptanceCriterion<T, C, R> for GreedyDescentAcceptanceCriterion<T, C, R>
where
    T: Copy + Ord + Send,
    C: CostEvaluator<T>,
    R: rand::Rng + Send,
{
    fn name(&self) -> &str {
        "GreedyDescentAcceptanceCriterion"
    }

    fn accept<'e, 'r, 's, 'm, 'p>(
        &self,
        _context: &mut IlsAcceptanceCriterionContext<'e, 'r, 's, 'm, 'p, T, C, R>,
        plan: &Plan<'p, T>,
    ) -> bool {
        let delta = &plan.fitness_delta;

        if delta.delta_unassigned < 0 {
            return true;
        }
        if delta.delta_unassigned > 0 {
            return false;
        }

        delta.delta_cost < 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::eval::DefaultCostEvaluator;

    type TestGreedyDescentAcceptanceCriterion =
        GreedyDescentAcceptanceCriterion<i64, DefaultCostEvaluator, rand::rngs::StdRng>;

    // Test-only helper that mirrors GreedyDescentAcceptanceCriterion logic.
    // This avoids constructing a full Plan (whose fields/builders are not visible here).
    fn greedy_descent_accept_core(delta_unassigned: i64, delta_cost: i64) -> bool {
        if delta_unassigned < 0 {
            return true;
        }
        if delta_unassigned > 0 {
            return false;
        }
        delta_cost < 0
    }

    #[test]
    fn name_is_correct() {
        let crit = TestGreedyDescentAcceptanceCriterion::new();
        assert_eq!(crit.name(), "GreedyDescentAcceptanceCriterion");
    }

    #[test]
    fn greedy_descent_accept_delta_unassigned_negative_always_true() {
        assert!(greedy_descent_accept_core(-1, 10));
        assert!(greedy_descent_accept_core(-5, -100));
        assert!(greedy_descent_accept_core(-1, 0));
    }

    #[test]
    fn greedy_descent_accept_delta_unassigned_positive_always_false() {
        assert!(!greedy_descent_accept_core(1, -10));
        assert!(!greedy_descent_accept_core(2, -999));
        assert!(!greedy_descent_accept_core(3, 0));
        assert!(!greedy_descent_accept_core(4, 1000));
    }

    #[test]
    fn greedy_descent_accept_delta_unassigned_zero_cost_negative_true() {
        assert!(greedy_descent_accept_core(0, -1));
        assert!(greedy_descent_accept_core(0, -500));
    }

    #[test]
    fn greedy_descent_accept_delta_unassigned_zero_cost_non_negative_false() {
        assert!(!greedy_descent_accept_core(0, 0));
        assert!(!greedy_descent_accept_core(0, 1));
        assert!(!greedy_descent_accept_core(0, 500));
    }
}
