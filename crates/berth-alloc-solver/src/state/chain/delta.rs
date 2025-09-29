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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArcRewrite {
    tail: usize,
    expected_head: usize,
    new_head: usize,
}

impl ArcRewrite {
    #[inline]
    pub fn new(tail: usize, expected_head: usize, new_head: usize) -> Self {
        Self {
            tail,
            expected_head,
            new_head,
        }
    }

    #[inline]
    pub fn tail(&self) -> usize {
        self.tail
    }

    #[inline]
    pub fn expected_head(&self) -> usize {
        self.expected_head
    }

    #[inline]
    pub fn new_head(&self) -> usize {
        self.new_head
    }
}

impl std::fmt::Display for ArcRewrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArcRewrite(tail: {}, expected_head: {}, new_head: {})",
            self.tail, self.expected_head, self.new_head
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChainDelta {
    updates: Vec<ArcRewrite>,
}

impl ChainDelta {
    #[inline]
    pub fn new() -> Self {
        Self {
            updates: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            updates: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn updates(&self) -> &[ArcRewrite] {
        &self.updates
    }

    #[inline]
    pub fn push(&mut self, node: usize, expected_succ: usize, new_succ: usize) {
        self.updates
            .push(ArcRewrite::new(node, expected_succ, new_succ));
    }

    #[inline]
    pub fn push_update(&mut self, update: ArcRewrite) {
        self.updates.push(update);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.updates.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }
}
