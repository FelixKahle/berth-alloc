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

pub mod berth_block_ruin_recreate;
pub mod dynasearch_insert_batch;
pub mod hill_climb;
pub mod hole_filler;
pub mod insert_move;
pub mod longest_ruin_recreate;
pub mod nuke;
pub mod pack_left_on_berth;
pub mod random_destroy_insert;
pub mod reconstruct_greedy;
pub mod relocate_one;
pub mod shaw_destroy_insert;
pub mod swap_pair;
pub mod swap_same_berth;
pub mod target_repair;
pub mod temporal_slice_destroy_insert;

pub mod prelude {
    use crate::matheuristic::{
        operator::Operator,
        oplib::{
            berth_block_ruin_recreate::BerthBlockRuinRecreateOperator,
            dynasearch_insert_batch::{
                DynasearchInsertBatchOperator, EnhancedDynasearchOnceOperator,
            },
            hill_climb::HillClimbRelocateOperator,
            hole_filler::HoleFillerOperator,
            insert_move::InsertMoveOperator,
            longest_ruin_recreate::LongestRuinRecreateOperator,
            nuke::NukeOperator,
            pack_left_on_berth::PackLeftOnBerthOperator,
            random_destroy_insert::RandomDestroyInsertOperator,
            reconstruct_greedy::ReconstructGreedyOperator,
            relocate_one::RelocateOneOperator,
            shaw_destroy_insert::ShawDestroyInsertOperator,
            swap_pair::SwapPairOperator,
            swap_same_berth::SwapOrderSameBerthOperator,
            target_repair::TargetedRepairOperator,
            temporal_slice_destroy_insert::TemporalSliceDestroyInsertOperator,
        },
    };
    use berth_alloc_core::prelude::Cost;
    use berth_alloc_model::prelude::Problem;
    use num_traits::{CheckedAdd, CheckedSub};
    use std::ops::Mul;

    pub fn op_list<T>(_: &Problem<T>) -> Vec<Box<dyn Operator<Time = T>>>
    where
        T: Copy
            + Ord
            + Send
            + Sync
            + std::fmt::Debug
            + CheckedAdd
            + CheckedSub
            + Mul<Output = Cost>
            + Into<Cost>
            + 'static,
    {
        vec![
            Box::new(RandomDestroyInsertOperator::default()),
            Box::new(SwapPairOperator::default()),
            Box::new(RelocateOneOperator::default()),
            Box::new(PackLeftOnBerthOperator::default()),
            Box::new(ShawDestroyInsertOperator::default()),
            Box::new(HoleFillerOperator::default()),
            Box::new(TemporalSliceDestroyInsertOperator::default()),
            Box::new(TargetedRepairOperator::default()),
            Box::new(NukeOperator::default()),
            Box::new(ReconstructGreedyOperator::default()),
            Box::new(HillClimbRelocateOperator::default()),
            Box::new(LongestRuinRecreateOperator::default()),
            Box::new(BerthBlockRuinRecreateOperator::default()),
            Box::new(InsertMoveOperator::default()),
            Box::new(SwapOrderSameBerthOperator::default()),
            Box::new(EnhancedDynasearchOnceOperator::default()),
            Box::new(DynasearchInsertBatchOperator::default()),
        ]
    }
}
