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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PlanningToken {
    value: u64,
}

impl PlanningToken {
    #[inline]
    pub fn new(value: u64) -> Self {
        Self { value }
    }

    #[inline]
    pub fn value(&self) -> u64 {
        self.value
    }
}

impl std::fmt::Display for PlanningToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PlanningToken({})", self.value)
    }
}

impl PartialOrd for PlanningToken {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PlanningToken {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenFactory {
    next: u64,
}

impl TokenFactory {
    #[inline]
    pub fn new() -> Self {
        Self { next: 0 }
    }

    #[inline]
    pub fn with_start(start: u64) -> Self {
        Self { next: start }
    }

    #[inline]
    pub fn next_token(&mut self) -> PlanningToken {
        let token = PlanningToken::new(self.next);
        self.next += 1;
        token
    }

    #[inline]
    pub fn current_token(&self) -> PlanningToken {
        PlanningToken::new(self.next)
    }
}

impl Default for TokenFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TokenFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenFactory(next={})", self.next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_planning_token_creation_and_value() {
        let tok = PlanningToken::new(42);
        assert_eq!(tok.value(), 42);
        assert_eq!(format!("{}", tok), "PlanningToken(42)");
        // Ordering
        let lower = PlanningToken::new(1);
        assert!(lower < tok);
    }

    #[test]
    fn token_factory_default_and_sequence() {
        let mut factory = TokenFactory::new();
        assert_eq!(format!("{}", factory), "TokenFactory(next=0)");
        let t0 = factory.next_token();
        let t1 = factory.next_token();
        let t2 = factory.next_token();
        assert_eq!(t0.value(), 0);
        assert_eq!(t1.value(), 1);
        assert_eq!(t2.value(), 2);
        assert!(t0 < t1 && t1 < t2);
        // All unique
        let mut set = HashSet::new();
        assert!(set.insert(t0));
        assert!(set.insert(t1));
        assert!(set.insert(t2));
    }

    #[test]
    fn test_token_factory_with_start() {
        let mut factory = TokenFactory::with_start(10);
        let t10 = factory.next_token();
        let t11 = factory.next_token();
        assert_eq!(t10.value(), 10);
        assert_eq!(t11.value(), 11);
    }

    #[test]
    fn test_token_factory_clone_is_independent() {
        let mut original = TokenFactory::with_start(5);
        let mut cloned = original.clone();
        // Both should yield the same first value independently
        let o_first = original.next_token();
        let c_first = cloned.next_token();
        assert_eq!(o_first.value(), 5);
        assert_eq!(c_first.value(), 5);
        // Subsequent calls diverge independently
        assert_eq!(original.next_token().value(), 6);
        assert_eq!(cloned.next_token().value(), 6);
    }

    #[test]
    fn test_planning_token_equality_and_hash() {
        use std::hash::{Hash, Hasher};
        let a1 = PlanningToken::new(7);
        let a2 = PlanningToken::new(7);
        let b = PlanningToken::new(8);
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        // Hash equality for equal tokens
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        a1.hash(&mut h1);
        a2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_peek_non_advancing() {
        let mut f = TokenFactory::with_start(100);

        // peek should show the next to-be-issued token
        let p1 = f.current_token();
        assert_eq!(p1.value(), 100);

        // Repeated peeks do not advance
        assert_eq!(f.current_token(), p1);
        assert_eq!(f.current_token(), p1);

        // next returns exactly that token and advances
        let n1 = f.next_token();
        assert_eq!(n1, p1);

        // After advancing, peek reflects the new upcoming token
        let p2 = f.current_token();
        assert_eq!(p2.value(), 101);
        assert_eq!(f.current_token(), p2);

        // next now yields p2 and advances again
        let n2 = f.next_token();
        assert_eq!(n2, p2);

        // Verify another step
        assert_eq!(f.current_token().value(), 102);
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    assert_impl_all!(PlanningToken: Send, Sync);
}
