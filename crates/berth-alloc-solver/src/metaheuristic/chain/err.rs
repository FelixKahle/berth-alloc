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

#[derive(Debug, Copy, Clone)]
pub struct BerthOutOfBoundsError {
    index: usize,
    num_berths: usize,
}

impl BerthOutOfBoundsError {
    pub fn new(index: usize, num_berths: usize) -> Self {
        Self { index, num_berths }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn num_berths(&self) -> usize {
        self.num_berths
    }
}

impl std::fmt::Display for BerthOutOfBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth index {} out of bounds (num_berths = {})",
            self.index, self.num_berths
        )
    }
}

impl std::error::Error for BerthOutOfBoundsError {}

#[derive(Debug, Copy, Clone)]
pub struct NodeOutOfBoundsError {
    index: usize,
    num_nodes: usize,
}

impl NodeOutOfBoundsError {
    pub fn new(index: usize, num_nodes: usize) -> Self {
        Self { index, num_nodes }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl std::fmt::Display for NodeOutOfBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node index {} out of bounds (num_nodes = {})",
            self.index, self.num_nodes
        )
    }
}

impl std::error::Error for NodeOutOfBoundsError {}

#[derive(Debug, Copy, Clone)]
pub struct NodeNotSkippedError {
    index: usize,
}

impl NodeNotSkippedError {
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

impl std::fmt::Display for NodeNotSkippedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node index {} is not skipped", self.index)
    }
}

impl std::error::Error for NodeNotSkippedError {}

#[derive(Debug, Copy, Clone)]
pub struct NodeIsSentinelError {
    index: usize,
}

impl NodeIsSentinelError {
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

impl std::fmt::Display for NodeIsSentinelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node index {} is a sentinel", self.index)
    }
}

impl std::error::Error for NodeIsSentinelError {}

#[derive(Debug, Copy, Clone)]
pub enum NodeInsertionError {
    NodeOutOfBounds(NodeOutOfBoundsError),
    NodeNotSkipped(NodeNotSkippedError),
    NodeIsSentinel(NodeIsSentinelError),
}

impl std::fmt::Display for NodeInsertionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeInsertionError::NodeOutOfBounds(e) => write!(f, "Insertion error: {}", e),
            NodeInsertionError::NodeNotSkipped(e) => write!(f, "Insertion error: {}", e),
            NodeInsertionError::NodeIsSentinel(e) => write!(f, "Insertion error: {}", e),
        }
    }
}

impl std::error::Error for NodeInsertionError {}

impl From<NodeOutOfBoundsError> for NodeInsertionError {
    fn from(e: NodeOutOfBoundsError) -> Self {
        NodeInsertionError::NodeOutOfBounds(e)
    }
}

impl From<NodeNotSkippedError> for NodeInsertionError {
    fn from(e: NodeNotSkippedError) -> Self {
        NodeInsertionError::NodeNotSkipped(e)
    }
}

impl From<NodeIsSentinelError> for NodeInsertionError {
    fn from(e: NodeIsSentinelError) -> Self {
        NodeInsertionError::NodeIsSentinel(e)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ChainError {
    NodeInsertion(NodeInsertionError),
    NodeOutOfBounds(NodeOutOfBoundsError),
    BerthOutOfBounds(BerthOutOfBoundsError),
    SegmentError(SegmentError),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::NodeInsertion(e) => write!(f, "Chain error: {}", e),
            ChainError::NodeOutOfBounds(e) => write!(f, "Chain error: {}", e),
            ChainError::BerthOutOfBounds(e) => write!(f, "Chain error: {}", e),
            ChainError::SegmentError(e) => write!(f, "Chain error: {}", e),
        }
    }
}

impl std::error::Error for ChainError {}

impl From<NodeInsertionError> for ChainError {
    fn from(e: NodeInsertionError) -> Self {
        ChainError::NodeInsertion(e)
    }
}

impl From<NodeOutOfBoundsError> for ChainError {
    fn from(e: NodeOutOfBoundsError) -> Self {
        ChainError::NodeOutOfBounds(e)
    }
}

impl From<BerthOutOfBoundsError> for ChainError {
    fn from(e: BerthOutOfBoundsError) -> Self {
        ChainError::BerthOutOfBounds(e)
    }
}

impl From<SegmentError> for ChainError {
    fn from(e: SegmentError) -> Self {
        ChainError::SegmentError(e)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SentinelInSegmentError {
    a: usize,
    b: usize,
}

impl SentinelInSegmentError {
    #[inline]
    pub fn new(a: usize, b: usize) -> Self {
        Self { a, b }
    }

    #[inline]
    pub fn a(&self) -> usize {
        self.a
    }

    #[inline]
    pub fn b(&self) -> usize {
        self.b
    }
}

impl std::fmt::Display for SentinelInSegmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sentinel node {} is in segment ending at node {}",
            self.a, self.b
        )
    }
}

impl std::error::Error for SentinelInSegmentError {}

#[derive(Debug, Copy, Clone)]
pub enum SegmentError {
    SentinelInSegment(SentinelInSegmentError),
    NodeOutOfBounds(NodeOutOfBoundsError),
}

impl std::fmt::Display for SegmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SegmentError::SentinelInSegment(e) => write!(f, "Segment error: {}", e),
            SegmentError::NodeOutOfBounds(e) => write!(f, "Segment error: {}", e),
        }
    }
}

impl std::error::Error for SegmentError {}

impl From<SentinelInSegmentError> for SegmentError {
    fn from(e: SentinelInSegmentError) -> Self {
        SegmentError::SentinelInSegment(e)
    }
}

impl From<NodeOutOfBoundsError> for SegmentError {
    fn from(e: NodeOutOfBoundsError) -> Self {
        SegmentError::NodeOutOfBounds(e)
    }
}
