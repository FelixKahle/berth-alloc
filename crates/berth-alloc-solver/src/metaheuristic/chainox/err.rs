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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeNotInactiveError<K>(K);

impl<K> NodeNotInactiveError<K> {
    #[inline]
    pub fn new(node: K) -> Self {
        Self(node)
    }

    #[inline]
    pub fn node(&self) -> K
    where
        K: Copy,
    {
        self.0
    }
}

impl<K: std::fmt::Display> std::fmt::Display for NodeNotInactiveError<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {} is not inactive", self.0)
    }
}

impl<K: std::fmt::Debug + std::fmt::Display> std::error::Error for NodeNotInactiveError<K> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeAlreadyActiveError<K>(K);

impl<K> NodeAlreadyActiveError<K> {
    #[inline]
    pub fn new(node: K) -> Self {
        Self(node)
    }

    #[inline]
    pub fn node(&self) -> K
    where
        K: Copy,
    {
        self.0
    }
}

impl<K: std::fmt::Display> std::fmt::Display for NodeAlreadyActiveError<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {} is already active", self.0)
    }
}

impl<K: std::fmt::Debug + std::fmt::Display> std::error::Error for NodeAlreadyActiveError<K> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeInactiveError<K>(K);

impl<K> NodeInactiveError<K> {
    #[inline]
    pub fn new(node: K) -> Self {
        Self(node)
    }

    #[inline]
    pub fn node(&self) -> K
    where
        K: Copy,
    {
        self.0
    }
}

impl<K: std::fmt::Display> std::fmt::Display for NodeInactiveError<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {} is inactive", self.0)
    }
}

impl<K: std::fmt::Debug + std::fmt::Display> std::error::Error for NodeInactiveError<K> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DifferentPathsError<P> {
    first: P,
    second: P,
}

impl<P> DifferentPathsError<P> {
    #[inline]
    pub fn new(first: P, second: P) -> Self {
        Self { first, second }
    }

    #[inline]
    pub fn first(&self) -> P
    where
        P: Copy,
    {
        self.first
    }

    #[inline]
    pub fn second(&self) -> P
    where
        P: Copy,
    {
        self.second
    }
}

impl<P: std::fmt::Display> std::fmt::Display for DifferentPathsError<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Nodes are in different paths: {} and {}",
            self.first, self.second
        )
    }
}

impl<P: std::fmt::Debug + std::fmt::Display> std::error::Error for DifferentPathsError<P> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DestInsideRangeError<K> {
    first: K,
    last: K,
    dest: K,
}

impl<K> DestInsideRangeError<K> {
    #[inline]
    pub fn new(first: K, last: K, dest: K) -> Self {
        Self { first, last, dest }
    }

    #[inline]
    pub fn first(&self) -> K
    where
        K: Copy,
    {
        self.first
    }

    #[inline]
    pub fn last(&self) -> K
    where
        K: Copy,
    {
        self.last
    }

    #[inline]
    pub fn dest(&self) -> K
    where
        K: Copy,
    {
        self.dest
    }
}

impl<K: std::fmt::Display> std::fmt::Display for DestInsideRangeError<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Destination {} is inside moved range [{}..={}]",
            self.dest, self.first, self.last
        )
    }
}

impl<K: std::fmt::Debug + std::fmt::Display> std::error::Error for DestInsideRangeError<K> {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpliceRangeError<K, P> {
    InactiveNode(NodeInactiveError<K>),
    DifferentPaths(DifferentPathsError<P>),
    DestInsideRange(DestInsideRangeError<K>),
}

impl<K: std::fmt::Display, P: std::fmt::Display> std::fmt::Display for SpliceRangeError<K, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpliceRangeError::InactiveNode(e) => write!(f, "SpliceRangeError: {}", e),
            SpliceRangeError::DifferentPaths(e) => write!(f, "SpliceRangeError: {}", e),
            SpliceRangeError::DestInsideRange(e) => write!(f, "SpliceRangeError: {}", e),
        }
    }
}

impl<K: std::fmt::Debug + std::fmt::Display, P: std::fmt::Debug + std::fmt::Display>
    std::error::Error for SpliceRangeError<K, P>
{
}

impl<K, P> From<NodeInactiveError<K>> for SpliceRangeError<K, P> {
    fn from(e: NodeInactiveError<K>) -> Self {
        SpliceRangeError::InactiveNode(e)
    }
}

impl<K, P> From<DifferentPathsError<P>> for SpliceRangeError<K, P> {
    fn from(e: DifferentPathsError<P>) -> Self {
        SpliceRangeError::DifferentPaths(e)
    }
}

impl<K, P> From<DestInsideRangeError<K>> for SpliceRangeError<K, P> {
    fn from(e: DestInsideRangeError<K>) -> Self {
        SpliceRangeError::DestInsideRange(e)
    }
}
