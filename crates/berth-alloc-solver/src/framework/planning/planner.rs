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
    framework::planning::tok::PlanningToken,
    registry::ledger::Ledger,
    terminal::{
        delta::TerminalDelta,
        sandbox::TerminalSandbox,
        terminalocc::{FreeBerth, TerminalRead},
    },
};
use berth_alloc_core::prelude::Cost;
use berth_alloc_model::{
    common::{FlexibleKind, Kind},
    prelude::Request,
    problem::{asg::AssignmentRef, req::RequestView},
};
use num_traits::{CheckedAdd, CheckedSub};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanningResource<T> {
    token: PlanningToken,
    value: T,
}

impl<T> PlanningResource<T> {
    #[inline]
    fn new(token: PlanningToken, value: T) -> Self {
        Self { token, value }
    }

    #[inline]
    pub fn token(&self) -> PlanningToken {
        self.token
    }

    #[inline]
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: std::fmt::Display> std::fmt::Display for PlanningResource<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Resource({}, {})", self.token, self.value)
    }
}

pub type PlanningFreeBerth<'p, T> = PlanningResource<FreeBerth<'p, T>>;
pub type PlanningRequest<'p, K, T> = PlanningResource<&'p Request<K, T>>;
pub type PlanningAssignmentRef<'p, K, T> = PlanningResource<AssignmentRef<'p, 'p, K, T>>;

#[derive(Debug, Clone)]
pub struct Plan<'p, T: Copy + Ord> {
    ledger: Ledger<'p, T>,
    terminal_delta: TerminalDelta<'p, T>,
    delta_cost: Cost,
}

impl<'p, T: Copy + Ord> Plan<'p, T> {
    #[allow(dead_code)]
    #[inline]
    fn new(ledger: Ledger<'p, T>, terminal_delta: TerminalDelta<'p, T>, delta_cost: Cost) -> Self {
        Self {
            ledger,
            terminal_delta,
            delta_cost,
        }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'p, T> {
        &self.ledger
    }

    #[inline]
    pub fn terminal_delta(&self) -> &TerminalDelta<'p, T> {
        &self.terminal_delta
    }

    #[inline]
    pub fn into_inner(self) -> (Ledger<'p, T>, TerminalDelta<'p, T>, Cost) {
        (self.ledger, self.terminal_delta, self.delta_cost)
    }

    #[inline]
    pub fn delta_cost(&self) -> Cost {
        self.delta_cost
    }
}

#[derive(Debug, Clone)]
pub struct PlanningExplorer<'builder, 'problem, T: Copy + Ord> {
    ledger: &'builder Ledger<'problem, T>,
    sandbox: &'builder TerminalSandbox<'problem, T>,
    current_token: PlanningToken,
}

impl<'builder, 'problem, T: Copy + Ord> PlanningExplorer<'builder, 'problem, T> {
    #[allow(dead_code)]
    #[inline]
    fn new(
        ledger: &'builder Ledger<'problem, T>,
        sandbox: &'builder TerminalSandbox<'problem, T>,
        current_token: PlanningToken,
    ) -> Self {
        Self {
            ledger,
            sandbox,
            current_token,
        }
    }

    #[inline]
    pub fn ledger(&self) -> &Ledger<'problem, T> {
        self.ledger
    }

    #[inline]
    pub fn sandbox(&self) -> &TerminalSandbox<'problem, T> {
        self.sandbox
    }

    #[inline]
    pub fn iter_unassigned_requests(
        &self,
    ) -> impl Iterator<Item = PlanningRequest<'problem, FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger
            .iter_unassigned_requests()
            .map(|req| PlanningRequest::new(self.current_token, req))
    }

    #[inline]
    pub fn iter_assigned_requests(
        &self,
    ) -> impl Iterator<Item = PlanningRequest<'problem, FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger
            .iter_assigned_requests()
            .map(|req| PlanningRequest::new(self.current_token, req))
    }

    #[inline]
    pub fn iter_flexible_assignments(
        &self,
    ) -> impl Iterator<Item = PlanningAssignmentRef<'problem, FlexibleKind, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        self.ledger
            .iter_flexible_assignments()
            .map(|asg| PlanningAssignmentRef::new(self.current_token, *asg))
    }

    #[inline]
    pub fn iter_free_for<K: Kind>(
        &self,
        req: &'problem Request<K, T>,
    ) -> impl Iterator<Item = PlanningFreeBerth<'problem, T>>
    where
        T: CheckedAdd + CheckedSub,
    {
        let window = req.feasible_window();
        let allowed = req.iter_allowed_berths_ids();

        self.sandbox
            .inner()
            .iter_free_intervals_for_berths_in(allowed, window)
            .map(|fb| PlanningFreeBerth::new(PlanningToken::new(0), fb))
    }
}
