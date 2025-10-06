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
    common::{FixedKind, FlexibleKind},
    problem::{
        asg::{Assignment, AssignmentContainer, AssignmentRef, AssignmentView},
        req::RequestIdentifier,
    },
};
use berth_alloc_core::prelude::Cost;
use num_traits::{CheckedAdd, CheckedSub};
use std::ops::Mul;

pub trait SolutionView<T>
where
    T: Copy + Ord,
{
    type FixedAssignmentView: AssignmentView<FixedKind, T>;
    type FlexibleAssignmentView: AssignmentView<FlexibleKind, T>;

    fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Self::FixedAssignmentView>;
    fn flexible_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, Self::FlexibleAssignmentView>;

    fn fixed_assignments_len(&self) -> usize {
        self.fixed_assignments().len()
    }
    fn flexible_assignments_len(&self) -> usize {
        self.flexible_assignments().len()
    }
    fn total_assignments_len(&self) -> usize {
        self.fixed_assignments_len() + self.flexible_assignments_len()
    }

    fn is_empty(&self) -> bool {
        self.fixed_assignments().is_empty() && self.flexible_assignments().is_empty()
    }

    fn contains_fixed(&self, rid: RequestIdentifier) -> bool {
        self.fixed_assignments().get(rid).is_some()
    }
    fn contains_flexible(&self, rid: RequestIdentifier) -> bool {
        self.flexible_assignments().get(rid).is_some()
    }

    fn fixed_cost(&self) -> Cost
    where
        T: Mul<Output = Cost> + CheckedAdd + CheckedSub + Into<Cost>,
    {
        self.fixed_assignments().iter().map(|a| a.cost()).sum()
    }

    fn flexible_cost(&self) -> Cost
    where
        T: Mul<Output = Cost> + CheckedAdd + CheckedSub + Into<Cost>,
    {
        self.flexible_assignments().iter().map(|a| a.cost()).sum()
    }

    fn cost(&self) -> Cost
    where
        T: Mul<Output = Cost> + CheckedAdd + CheckedSub + Into<Cost>,
    {
        self.fixed_cost() + self.flexible_cost()
    }
}

#[derive(Debug, Clone)]
pub struct Solution<T: Copy + Ord> {
    fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
    flexible_assignments: AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>,
}

impl<T: Copy + Ord + CheckedAdd + CheckedSub> Solution<T> {
    #[inline]
    pub fn new(
        fixed_assignments: AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>,
        flexible_assignments: AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>,
    ) -> Self {
        Self {
            fixed_assignments,
            flexible_assignments,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> SolutionRef<'_, T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        let fixed = self
            .fixed_assignments
            .iter()
            .map(|a| a.to_ref())
            .collect::<AssignmentContainer<FixedKind, T, AssignmentRef<'_, '_, FixedKind, T>>>();

        let flex = self
                    .flexible_assignments
                    .iter()
                    .map(|a| a.to_ref())
                    .collect::<AssignmentContainer<
                        FlexibleKind,
                        T,
                        AssignmentRef<'_, '_, FlexibleKind, T>,
                    >>();

        SolutionRef {
            fixed_assignments: fixed,
            flexible_assignments: flex,
        }
    }
}

impl<T: Copy + Ord> SolutionView<T> for Solution<T> {
    type FixedAssignmentView = Assignment<FixedKind, T>;
    type FlexibleAssignmentView = Assignment<FlexibleKind, T>;

    fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>> {
        &self.fixed_assignments
    }

    fn flexible_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>> {
        &self.flexible_assignments
    }
}

#[derive(Debug, Clone)]
pub struct SolutionRef<'p, T: Copy + Ord> {
    fixed_assignments: AssignmentContainer<FixedKind, T, AssignmentRef<'p, 'p, FixedKind, T>>,
    flexible_assignments:
        AssignmentContainer<FlexibleKind, T, AssignmentRef<'p, 'p, FlexibleKind, T>>,
}

impl<'p, T: Copy + Ord + CheckedAdd + CheckedSub> SolutionRef<'p, T> {
    #[inline]
    pub fn new(
        fixed_assignments: AssignmentContainer<FixedKind, T, AssignmentRef<'p, 'p, FixedKind, T>>,
        flexible_assignments: AssignmentContainer<
            FlexibleKind,
            T,
            AssignmentRef<'p, 'p, FlexibleKind, T>,
        >,
    ) -> Self {
        Self {
            fixed_assignments,
            flexible_assignments,
        }
    }

    #[inline]
    pub fn to_owned(&self) -> Solution<T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        let fixed = self
            .fixed_assignments
            .iter()
            .map(|a| a.to_owned())
            .collect::<AssignmentContainer<FixedKind, T, Assignment<FixedKind, T>>>();

        let flex = self
            .flexible_assignments
            .iter()
            .map(|a| a.to_owned())
            .collect::<AssignmentContainer<FlexibleKind, T, Assignment<FlexibleKind, T>>>();

        Solution {
            fixed_assignments: fixed,
            flexible_assignments: flex,
        }
    }

    #[inline]
    pub fn into_owned(self) -> Solution<T>
    where
        T: CheckedAdd + CheckedSub + std::hash::Hash,
    {
        self.to_owned()
    }
}

impl<'p, T: Copy + Ord> SolutionView<T> for SolutionRef<'p, T> {
    type FixedAssignmentView = AssignmentRef<'p, 'p, FixedKind, T>;
    type FlexibleAssignmentView = AssignmentRef<'p, 'p, FlexibleKind, T>;

    fn fixed_assignments(&self) -> &AssignmentContainer<FixedKind, T, Self::FixedAssignmentView> {
        &self.fixed_assignments
    }

    fn flexible_assignments(
        &self,
    ) -> &AssignmentContainer<FlexibleKind, T, Self::FlexibleAssignmentView> {
        &self.flexible_assignments
    }
}
