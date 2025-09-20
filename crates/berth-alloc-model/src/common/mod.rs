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

pub trait IdentifierMarkerName: Copy {
    const NAME: &'static str;
}

#[repr(transparent)]
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Identifier<I, U>(I, core::marker::PhantomData<U>);

impl<I, U> Identifier<I, U> {
    #[inline]
    pub fn new(id: I) -> Self {
        Self(id, core::marker::PhantomData)
    }

    #[inline]
    pub fn value(&self) -> &I {
        &self.0
    }

    #[inline]
    pub fn into_inner(self) -> I {
        self.0
    }
}

impl<I, U> std::fmt::Display for Identifier<I, U>
where
    I: std::fmt::Display,
    U: IdentifierMarkerName,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", U::NAME, self.0)
    }
}

pub trait Kind: Clone {
    const NAME: &'static str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedKind;

impl Kind for FixedKind {
    const NAME: &'static str = "Fixed";
}

impl std::fmt::Display for FixedKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::NAME)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlexibleKind;

impl Kind for FlexibleKind {
    const NAME: &'static str = "Flexible";
}

impl std::fmt::Display for FlexibleKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::NAME)
    }
}
