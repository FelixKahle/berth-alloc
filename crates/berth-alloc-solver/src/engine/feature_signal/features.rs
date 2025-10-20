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

use berth_alloc_core::prelude::TimePoint;
use smallvec::SmallVec;
use std::marker::PhantomData;

/// Penalizable feature keys (hashable, small, copy-on-write friendly via SmallVec).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Feature {
    RequestBerth { req: usize, berth: usize },
    Request { req: usize },
    Berth { berth: usize },
    TimeBucket { tb: i64 },
    BerthTime { berth: usize, tb: i64 },
    RequestTime { req: usize, tb: i64 },
}

/// Contract: enumerate features “touched” by assigning (request → berth @ start_time).
pub trait FeatureExtractor<T> {
    fn features_for(
        &self,
        request: crate::model::index::RequestIndex,
        berth: crate::model::index::BerthIndex,
        start_time: TimePoint<T>,
        out: &mut SmallVec<[Feature; 6]>,
    );
}

/// Configurable extractor with a caller-provided bucketizer.
/// NOTE: bucketizer is a function pointer `fn(TimePoint<T>) -> i64` to remain 'static-safe.
#[derive(Clone)]
pub struct DefaultFeatureExtractor<T> {
    bucketizer: fn(TimePoint<T>) -> i64,
    include_req_berth: bool,
    include_request: bool,
    include_berth: bool,
    include_time: bool,
    include_berth_time: bool,
    include_req_time: bool,
    _phantom: PhantomData<T>,
}

impl<T> DefaultFeatureExtractor<T> {
    pub fn new(bucketizer: fn(TimePoint<T>) -> i64) -> Self {
        Self {
            bucketizer,
            include_req_berth: true,
            include_request: false,
            include_berth: false,
            include_time: true,
            include_berth_time: true,
            include_req_time: false,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn set_include_req_berth(mut self, yes: bool) -> Self {
        self.include_req_berth = yes;
        self
    }
    #[inline]
    pub fn set_include_request(mut self, yes: bool) -> Self {
        self.include_request = yes;
        self
    }
    #[inline]
    pub fn set_include_berth(mut self, yes: bool) -> Self {
        self.include_berth = yes;
        self
    }
    #[inline]
    pub fn set_include_time(mut self, yes: bool) -> Self {
        self.include_time = yes;
        self
    }
    #[inline]
    pub fn set_include_berth_time(mut self, yes: bool) -> Self {
        self.include_berth_time = yes;
        self
    }
    #[inline]
    pub fn set_include_req_time(mut self, yes: bool) -> Self {
        self.include_req_time = yes;
        self
    }
}

impl<T> FeatureExtractor<T> for DefaultFeatureExtractor<T> {
    #[inline]
    fn features_for(
        &self,
        req: crate::model::index::RequestIndex,
        berth: crate::model::index::BerthIndex,
        start_time: TimePoint<T>,
        out: &mut SmallVec<[Feature; 6]>,
    ) {
        let r = req.get();
        let b = berth.get();
        let tb = (self.bucketizer)(start_time);

        if self.include_req_berth {
            out.push(Feature::RequestBerth { req: r, berth: b });
        }
        if self.include_request {
            out.push(Feature::Request { req: r });
        }
        if self.include_berth {
            out.push(Feature::Berth { berth: b });
        }
        if self.include_time {
            out.push(Feature::TimeBucket { tb });
        }
        if self.include_berth_time {
            out.push(Feature::BerthTime { berth: b, tb });
        }
        if self.include_req_time {
            out.push(Feature::RequestTime { req: r, tb });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::index::{BerthIndex, RequestIndex};
    use berth_alloc_core::prelude::TimePoint;

    #[inline]
    fn tp(v: i64) -> TimePoint<i64> {
        TimePoint::new(v)
    }

    // Bucketizer helpers
    fn bkt_10(tp: TimePoint<i64>) -> i64 {
        tp.value() / 10
    }
    fn bkt_const_42(_: TimePoint<i64>) -> i64 {
        42
    }

    #[test]
    fn test_default_flags_include_rb_time_and_bt_only() {
        // Defaults from new(): RB=true, Time=true, BxT=true; Request=false, Berth=false, RxT=false.
        let fx: DefaultFeatureExtractor<i64> = DefaultFeatureExtractor::new(bkt_10);

        let mut out = SmallVec::<[Feature; 6]>::new();
        let ri = RequestIndex::new(3);
        let bi = BerthIndex::new(7);

        fx.features_for(ri, bi, tp(25), &mut out);

        // tb = 25/10 = 2
        let expected = vec![
            Feature::RequestBerth { req: 3, berth: 7 },
            Feature::TimeBucket { tb: 2 },
            Feature::BerthTime { berth: 7, tb: 2 },
        ];
        assert_eq!(
            out.as_slice(),
            &expected[..],
            "feature order and content must match defaults"
        );
    }

    #[test]
    fn test_all_flags_true_includes_all_six_in_defined_order() {
        let fx: DefaultFeatureExtractor<i64> = DefaultFeatureExtractor::new(bkt_const_42)
            .set_include_req_berth(true)
            .set_include_request(true)
            .set_include_berth(true)
            .set_include_time(true)
            .set_include_berth_time(true)
            .set_include_req_time(true);

        let mut out = SmallVec::<[Feature; 6]>::new();
        let ri = RequestIndex::new(1);
        let bi = BerthIndex::new(2);

        fx.features_for(ri, bi, tp(0), &mut out);

        // Order must match the push order in implementation.
        let expected = vec![
            Feature::RequestBerth { req: 1, berth: 2 },
            Feature::Request { req: 1 },
            Feature::Berth { berth: 2 },
            Feature::TimeBucket { tb: 42 },
            Feature::BerthTime { berth: 2, tb: 42 },
            Feature::RequestTime { req: 1, tb: 42 },
        ];
        assert_eq!(
            out.as_slice(),
            &expected[..],
            "all six features must be present in order"
        );
    }

    #[test]
    fn test_all_flags_false_yields_empty_set() {
        let fx: DefaultFeatureExtractor<i64> = DefaultFeatureExtractor::new(bkt_const_42)
            .set_include_req_berth(false)
            .set_include_request(false)
            .set_include_berth(false)
            .set_include_time(false)
            .set_include_berth_time(false)
            .set_include_req_time(false);

        let mut out = SmallVec::<[Feature; 6]>::new();
        fx.features_for(RequestIndex::new(0), BerthIndex::new(0), tp(0), &mut out);

        assert!(
            out.is_empty(),
            "no features should be emitted when all flags are false"
        );
    }

    #[test]
    fn test_bucketizer_is_applied_to_start_time() {
        let fx: DefaultFeatureExtractor<i64> = DefaultFeatureExtractor::new(bkt_10)
            .set_include_req_berth(false)
            .set_include_request(false)
            .set_include_berth(false)
            .set_include_time(true)
            .set_include_berth_time(true)
            .set_include_req_time(true);

        let mut out = SmallVec::<[Feature; 6]>::new();
        let ri = RequestIndex::new(5);
        let bi = BerthIndex::new(9);

        // start_time 17 => tb = 1
        fx.features_for(ri, bi, tp(17), &mut out);
        assert!(
            out.contains(&Feature::TimeBucket { tb: 1 }),
            "TimeBucket must reflect bucketizer tb"
        );
        assert!(
            out.contains(&Feature::BerthTime { berth: 9, tb: 1 }),
            "BerthTime must reflect bucketizer tb"
        );
        assert!(
            out.contains(&Feature::RequestTime { req: 5, tb: 1 }),
            "RequestTime must reflect bucketizer tb"
        );

        // start_time 99 => tb = 9
        out.clear();
        fx.features_for(ri, bi, tp(99), &mut out);
        assert!(out.contains(&Feature::TimeBucket { tb: 9 }));
        assert!(out.contains(&Feature::BerthTime { berth: 9, tb: 9 }));
        assert!(out.contains(&Feature::RequestTime { req: 5, tb: 9 }));
    }

    #[test]
    fn test_clone_preserves_configuration_and_bucketizer() {
        let fx: DefaultFeatureExtractor<i64> = DefaultFeatureExtractor::new(bkt_const_42)
            .set_include_req_berth(false)
            .set_include_request(true)
            .set_include_berth(true)
            .set_include_time(false)
            .set_include_berth_time(false)
            .set_include_req_time(false);

        let fx2 = fx.clone();

        let mut a = SmallVec::<[Feature; 6]>::new();
        let mut b = SmallVec::<[Feature; 6]>::new();

        let ri = RequestIndex::new(4);
        let bi = BerthIndex::new(3);
        let st = tp(123);

        fx.features_for(ri, bi, st, &mut a);
        fx2.features_for(ri, bi, st, &mut b);

        assert_eq!(a, b, "cloned extractor must emit identical features");
        // Only Request and Berth should appear per our flags.
        assert_eq!(a.len(), 2);
        assert!(a.contains(&Feature::Request { req: 4 }));
        assert!(a.contains(&Feature::Berth { berth: 3 }));
    }
}
