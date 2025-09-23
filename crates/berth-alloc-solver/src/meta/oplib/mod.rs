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

pub mod holefiller;
pub mod packleftonberth;
pub mod randomdestroyinsert;
pub mod relocateone;
pub mod shawdestroyinsert;
pub mod swappair;
pub mod targetrepair;
pub mod temporalslicedestroyinsert;

pub mod prelude {
    use std::ops::Mul;

    use crate::meta::{
        operator::Operator,
        oplib::{
            holefiller::HoleFillerOperator, packleftonberth::PackLeftOnBerthOperator,
            relocateone::RelocateOneOperator, shawdestroyinsert::ShawDestroyInsertOperator,
            swappair::SwapPairOperator, targetrepair::TargetedRepairOperator,
            temporalslicedestroyinsert::TemporalSliceDestroyInsertOperator,
        },
    };
    use berth_alloc_core::prelude::Cost;
    use berth_alloc_model::prelude::Problem;
    use num_traits::{CheckedAdd, CheckedSub};

    pub use super::randomdestroyinsert::*;

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
        ]
    }
}
