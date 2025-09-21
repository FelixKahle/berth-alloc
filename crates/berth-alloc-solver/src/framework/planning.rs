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
    framework::err::{BerthNotFreeError, ProposeAssignmentError},
    registry::{
        err::LedgerUncomitError,
        ledger::Ledger,
    },
    terminal::{
        delta::TerminalDelta,
        terminalocc::{TerminalOccupancy, TerminalRead},
    },
};
use berth_alloc_core::{
    prelude::{Cost, TimeInterval, TimePoint},
    utils::marker::Brand,
};
use berth_alloc_model::{
    common::FlexibleKind,
    prelude::{Berth, Request},
    problem::asg::{AssignmentRef, AssignmentView},
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone)]
pub struct BrandedFreeBerth<'brand, 'p, T: Copy + Ord> {
    interval: TimeInterval<T>,
    berth: &'p Berth<T>,
    _brand: Brand<'brand>,
}

impl<'brand, 'p, T: Copy + Ord> BrandedFreeBerth<'brand, 'p, T> {
    pub fn new(interval: TimeInterval<T>, berth: &'p Berth<T>) -> Self {
        Self {
            interval,
            berth,
            _brand: Brand::new(),
        }
    }

    pub fn interval(&self) -> &TimeInterval<T> {
        &self.interval
    }

    pub fn berth(&self) -> &'p Berth<T> {
        self.berth
    }
}

#[derive(Debug, Clone)]
pub struct Plan<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_delta: TerminalDelta<'p, T>,
    delta_cost: Cost,
}

#[derive(Debug, Clone)]
pub struct PlanBuilder<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_occupancy: TerminalOccupancy<'p, T>,
}

impl<'p, T: Copy + Ord> PlanBuilder<'p, T> {
    pub fn new(ledger: Ledger<'p, T>, terminal_occupancy: TerminalOccupancy<'p, T>) -> Self {
        Self {
            ledger,
            terminal_occupancy,
        }
    }

    #[inline]
    pub fn propose_assignment(
        &mut self,
        request: &'p Request<FlexibleKind, T>,
        start_time: TimePoint<T>,
        free_berth: &BrandedFreeBerth<'_, 'p, T>,
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, ProposeAssignmentError<T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let berth = free_berth.berth();
        let free_interval = free_berth.interval();

        let assignment = AssignmentRef::new(request, berth, start_time)?;
        let assignment_interval = assignment.interval();

        if !free_interval.contains_interval(&assignment_interval) {
            return Err(ProposeAssignmentError::NotFree(BerthNotFreeError::new(
                berth.id(),
                assignment_interval,
                *free_interval,
            )));
        }

        self.ledger
            .commit_assignment(request, berth, start_time)
            .map_err(ProposeAssignmentError::from)
    }

    #[inline]
    pub fn propose_unassignment(
        &mut self,
        assignment: &AssignmentRef<'p, 'p, FlexibleKind, T>,
    ) -> Result<AssignmentRef<'p, 'p, FlexibleKind, T>, LedgerUncomitError>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger.uncommit_assignment(assignment)
    }

    #[inline]
    pub fn iter_free_within<'a>(
        &'a self,
        berths: &'a [&'p Berth<T>],
        window: TimeInterval<T>,
    ) -> impl Iterator<Item = BrandedFreeBerth<'a, 'p, T>> + 'a
    where
        T: CheckedAdd + CheckedSub,
    {
        self.terminal_occupancy
            .iter_free_within(berths, window)
            .map(|free_berth| BrandedFreeBerth::new(free_berth.interval(), free_berth.berth()))
    }

    #[inline]
    pub fn iter_unassigned_requests(&self) -> impl Iterator<Item = &'p Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger.iter_unassigned_requests()
    }

    #[inline]
    pub fn iter_assigned_requests(&self) -> impl Iterator<Item = &Request<FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger.iter_assigned_requests()
    }

    #[inline]
    pub fn iter_assignments(
        &self,
    ) -> impl Iterator<Item = &AssignmentRef<'p, 'p, FlexibleKind, T>> {
        self.ledger.iter_assignments()
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    macro_rules! test_integer_types {
        ($($t:ty),*) => {
            $(
                assert_impl_all!(Plan<'static, $t>: Send, Sync);
            )*
        };
    }

    test_integer_types!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
}
