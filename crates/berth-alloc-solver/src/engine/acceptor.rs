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

use crate::state::fitness::Fitness;

pub trait Acceptor {
    fn name(&self) -> &str;
    fn accept(&self, current: &Fitness, new: &Fitness) -> bool;
}

impl std::fmt::Display for dyn Acceptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Default, Clone)]
pub struct LexStrictAcceptor;
impl Acceptor for LexStrictAcceptor {
    fn name(&self) -> &str {
        "LexStrictAcceptor"
    }
    fn accept(&self, cur: &Fitness, cand: &Fitness) -> bool {
        (cand.unassigned_requests < cur.unassigned_requests)
            || (cand.unassigned_requests == cur.unassigned_requests && cand.cost < cur.cost)
    }
}

/// For repair vs baseline: allow any strict drop in unassigned, and when
/// unassigned is equal, require a strict cost drop (no plateaus).
#[derive(Debug, Default, Clone)]
pub struct RepairAcceptor;
impl Acceptor for RepairAcceptor {
    fn name(&self) -> &str {
        "RepairAcceptor"
    }
    fn accept(&self, cur: &Fitness, cand: &Fitness) -> bool {
        (cand.unassigned_requests < cur.unassigned_requests)
            || (cand.unassigned_requests == cur.unassigned_requests && cand.cost < cur.cost)
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_obj_safe;

    assert_obj_safe!(Acceptor);
}
