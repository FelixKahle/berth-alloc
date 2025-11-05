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

use crate::search::{eval::CostEvaluator, neighboors::Neighboors, operator::LocalSearchOperator};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

pub mod compound;
pub mod relocate;
pub mod slide;
pub mod swap;

#[derive(Debug, Clone, Copy)]
pub struct OperatorSelectionConfig {
    pub use_relocate: bool,
    pub use_slide: bool,
    pub use_swap: bool,
}

impl OperatorSelectionConfig {
    #[inline]
    pub fn all_enabled() -> Self {
        Self {
            use_relocate: true,
            use_slide: true,
            use_swap: true,
        }
    }

    #[inline]
    pub fn num_enabled(&self) -> usize {
        (self.use_relocate as usize) + (self.use_slide as usize) + (self.use_swap as usize)
    }
}

impl Default for OperatorSelectionConfig {
    fn default() -> Self {
        Self::all_enabled()
    }
}

impl std::fmt::Display for OperatorSelectionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OperatorSelectionConfig {{ use_relocate: {}, use_slide: {}, use_swap: {} }}",
            self.use_relocate, self.use_slide, self.use_swap
        )
    }
}

#[inline(always)]
fn make_operator_list<'n, T, C, R>(
    config: &OperatorSelectionConfig,
    neighboors: &Neighboors<'n>,
) -> Vec<Box<dyn LocalSearchOperator<T, C, R> + 'n>>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    let mut operators: Vec<Box<dyn LocalSearchOperator<T, C, R>>> =
        Vec::with_capacity(config.num_enabled());

    if config.use_relocate {
        operators.push(Box::new(relocate::RelocateOp::new()));
    }

    if config.use_slide {
        operators.push(Box::new(slide::SlideOp::new()));
    }

    if config.use_swap {
        operators.push(Box::new(swap::SwapSlotOp::with_neighbors(
            neighboors.neighbors.clone(),
        )));
    }

    operators
}

#[inline]
pub fn make_compound_operator<'n, T, C, R>(
    config: &OperatorSelectionConfig,
    neighboors: &Neighboors<'n>,
) -> compound::CompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    compound::CompoundOperator::concatenate_restart(make_operator_list(config, neighboors))
}

#[inline]
pub fn make_random_compound_operator<'n, T, C, R>(
    config: &OperatorSelectionConfig,
    neighboors: &Neighboors<'n>,
) -> compound::RandomCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    compound::RandomCompoundOperator::new(make_operator_list(config, neighboors))
}

#[inline]
pub fn make_multi_armed_bandit_compound_operator<'n, T, C, R>(
    config: &OperatorSelectionConfig,
    neighboors: &Neighboors<'n>,
    memory_coefficient: f64,
    exploration_coefficient: f64,
) -> compound::MultiArmedBanditCompoundOperator<'n, T, C, R>
where
    T: Copy + Ord + std::fmt::Debug + CheckedAdd + CheckedSub + Mul<Output = Cost> + Into<Cost>,
    C: CostEvaluator<T>,
    R: rand::Rng,
{
    compound::MultiArmedBanditCompoundOperator::new(
        make_operator_list(config, neighboors),
        memory_coefficient,
        exploration_coefficient,
    )
}
