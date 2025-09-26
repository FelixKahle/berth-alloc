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

use crate::framework::planning::tok::PlanningToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpiredTokenError {
    token: PlanningToken,
}

impl ExpiredTokenError {
    pub fn new(token: PlanningToken) -> Self {
        Self { token }
    }

    pub fn token(&self) -> PlanningToken {
        self.token
    }
}

impl std::fmt::Display for ExpiredTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Token {} has expired", self.token)
    }
}

impl std::error::Error for ExpiredTokenError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InvalidTokenError {
    token: PlanningToken,
}

impl InvalidTokenError {
    pub fn new(token: PlanningToken) -> Self {
        Self { token }
    }

    pub fn token(&self) -> PlanningToken {
        self.token
    }
}

impl std::fmt::Display for InvalidTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Token {} is invalid", self.token)
    }
}

impl std::error::Error for InvalidTokenError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PlanningTokenError {
    Expired(ExpiredTokenError),
    Invalid(InvalidTokenError),
}

impl std::fmt::Display for PlanningTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanningTokenError::Expired(err) => write!(f, "{}", err),
            PlanningTokenError::Invalid(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for PlanningTokenError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_token_error_display_and_accessors() {
        let tok = PlanningToken::new(3);
        let err = ExpiredTokenError::new(tok);
        assert_eq!(err.token(), tok);
        assert_eq!(format!("{}", err), "Token PlanningToken(3) has expired");
    }

    #[test]
    fn invalid_token_error_display_and_accessors() {
        let tok = PlanningToken::new(7);
        let err = InvalidTokenError::new(tok);
        assert_eq!(err.token(), tok);
        assert_eq!(format!("{}", err), "Token PlanningToken(7) is invalid");
    }

    #[test]
    fn planning_token_error_display_matches_inner() {
        let t1 = PlanningToken::new(1);
        let e1 = PlanningTokenError::Expired(ExpiredTokenError::new(t1));
        assert_eq!(format!("{}", e1), "Token PlanningToken(1) has expired");

        let t2 = PlanningToken::new(2);
        let e2 = PlanningTokenError::Invalid(InvalidTokenError::new(t2));
        assert_eq!(format!("{}", e2), "Token PlanningToken(2) is invalid");
    }

    // Minimal compile-time checks that these implement std::error::Error
    fn assert_error_trait<E: std::error::Error + Send + Sync + 'static>(_e: &E) {}

    #[test]
    fn error_traits_present() {
        let e1 = ExpiredTokenError::new(PlanningToken::new(0));
        let e2 = InvalidTokenError::new(PlanningToken::new(0));
        let e3 = PlanningTokenError::Expired(e1);
        assert_error_trait(&e1);
        assert_error_trait(&e2);
        assert_error_trait(&e3);
    }
}

#[cfg(test)]
mod static_assertions {
    use super::*;
    use ::static_assertions::assert_impl_all;

    assert_impl_all!(ExpiredTokenError: Send, Sync, std::error::Error);
    assert_impl_all!(InvalidTokenError: Send, Sync, std::error::Error);
    assert_impl_all!(PlanningTokenError: Send, Sync, std::error::Error);
}
