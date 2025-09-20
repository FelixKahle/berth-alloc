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

use num_traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, SaturatingAdd, SaturatingSub, Zero,
};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait MarkerName {
    const NAME_POINT: &'static str;
    const NAME_DELTA: &'static str;
}

#[repr(transparent)]
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Point<T, U>(T, core::marker::PhantomData<U>);

impl<T, U> Point<T, U> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Point(value, core::marker::PhantomData)
    }

    #[inline]
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Point::new(T::zero())
    }

    #[inline]
    pub const fn value(&self) -> T
    where
        T: Copy,
    {
        self.0
    }

    #[inline]
    pub fn is_zero(&self) -> bool
    where
        T: Zero,
    {
        self.0.is_zero()
    }

    #[inline]
    pub fn checked_add(self, d: Delta<T, U>) -> Option<Self>
    where
        T: CheckedAdd,
    {
        self.0.checked_add(&d.0).map(Point::new)
    }

    #[inline]
    pub fn saturating_add(self, d: Delta<T, U>) -> Self
    where
        T: SaturatingAdd + CheckedAdd,
    {
        Point::new(self.0.saturating_add(&d.0))
    }

    #[inline]
    pub fn checked_sub(self, d: Delta<T, U>) -> Option<Self>
    where
        T: CheckedSub<Output = T>,
    {
        self.0.checked_sub(&d.0).map(Point::new)
    }

    #[inline]
    pub fn saturating_sub(self, d: Delta<T, U>) -> Self
    where
        T: SaturatingSub + CheckedSub,
    {
        Point::new(self.0.saturating_sub(&d.0))
    }
}

impl<T: std::fmt::Display, U: MarkerName> std::fmt::Display for Point<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", U::NAME_POINT, self.0)
    }
}

impl<T, U> Default for Point<T, U>
where
    T: Zero,
{
    #[inline]
    fn default() -> Self {
        Point::new(T::zero())
    }
}

impl<T, U> Add<Delta<T, U>> for Point<T, U>
where
    T: CheckedAdd,
{
    type Output = Point<T, U>;

    #[inline]
    fn add(self, rhs: Delta<T, U>) -> Self::Output {
        Point::new(self.0.checked_add(&rhs.0).expect("error in Point + Delta"))
    }
}

impl<T, U> AddAssign<Delta<T, U>> for Point<T, U>
where
    T: CheckedAdd,
{
    fn add_assign(&mut self, rhs: Delta<T, U>) {
        self.0 = self.0.checked_add(&rhs.0).expect("error in Point += Delta");
    }
}

impl<T, U> Sub<Delta<T, U>> for Point<T, U>
where
    T: CheckedSub<Output = T>,
{
    type Output = Point<T, U>;

    fn sub(self, rhs: Delta<T, U>) -> Self::Output {
        Point::new(self.0.checked_sub(&rhs.0).expect("error in Point - Delta"))
    }
}

impl<T, U> SubAssign<Delta<T, U>> for Point<T, U>
where
    T: CheckedSub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Delta<T, U>) {
        self.0 = self.0.checked_sub(&rhs.0).expect("error in Point -= Delta");
    }
}

impl<T, U> Sub<Point<T, U>> for Point<T, U>
where
    T: CheckedSub<Output = T>,
{
    type Output = Delta<T, U>;

    fn sub(self, rhs: Point<T, U>) -> Self::Output {
        Delta::new(self.0.checked_sub(&rhs.0).expect("error in Point - Point"))
    }
}

#[repr(transparent)]
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Delta<T, U>(T, core::marker::PhantomData<U>);

impl<T, U> Delta<T, U> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Delta(value, core::marker::PhantomData)
    }

    #[inline]
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Delta::new(T::zero())
    }

    #[inline]
    pub const fn value(self) -> T
    where
        T: Copy,
    {
        self.0
    }

    #[inline]
    pub fn is_positive(&self) -> bool
    where
        T: Zero + PartialOrd,
    {
        self.0 > T::zero()
    }

    #[inline]
    pub fn is_negative(&self) -> bool
    where
        T: Zero + PartialOrd,
    {
        self.0 < T::zero()
    }

    #[inline]
    pub fn abs(self) -> Self
    where
        T: Zero + PartialOrd + CheckedNeg + Copy,
    {
        if self.is_negative() { -self } else { self }
    }
}

impl<T: std::fmt::Display, U: MarkerName> std::fmt::Display for Delta<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", U::NAME_DELTA, self.0)
    }
}

impl<T, U> Zero for Delta<T, U>
where
    T: Zero + CheckedAdd,
{
    #[inline]
    fn zero() -> Self {
        Delta::new(T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T, U> Default for Delta<T, U>
where
    T: Zero,
{
    #[inline]
    fn default() -> Self {
        Delta::new(T::zero())
    }
}

impl<T, U> From<T> for Delta<T, U> {
    #[inline]
    fn from(v: T) -> Self {
        Delta::new(v)
    }
}

impl<T, U> Add<Point<T, U>> for Delta<T, U>
where
    T: CheckedAdd,
{
    type Output = Point<T, U>;

    #[inline]
    fn add(self, rhs: Point<T, U>) -> Self::Output {
        Point::new(rhs.0.checked_add(&self.0).expect("error in Delta + Point"))
    }
}

impl<T, U> AddAssign for Delta<T, U>
where
    T: CheckedAdd,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.checked_add(&rhs.0).expect("error in Delta += Delta");
    }
}

impl<T, U> Add for Delta<T, U>
where
    T: CheckedAdd,
{
    type Output = Delta<T, U>;

    fn add(self, rhs: Self) -> Self::Output {
        Delta::new(self.0.checked_add(&rhs.0).expect("error in Delta + Delta"))
    }
}

impl<T, U> CheckedAdd for Delta<T, U>
where
    T: CheckedAdd,
{
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_add(&rhs.0).map(Delta::new)
    }
}

impl<T, U> SaturatingAdd for Delta<T, U>
where
    T: SaturatingAdd + CheckedAdd,
{
    fn saturating_add(&self, rhs: &Self) -> Self {
        Delta::new(self.0.saturating_add(&rhs.0))
    }
}

impl<T, U> Sub for Delta<T, U>
where
    T: CheckedSub<Output = T>,
{
    type Output = Delta<T, U>;

    fn sub(self, rhs: Self) -> Self::Output {
        Delta::new(self.0.checked_sub(&rhs.0).expect("error in Delta - Delta"))
    }
}

impl<T, U> SubAssign for Delta<T, U>
where
    T: CheckedSub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.checked_sub(&rhs.0).expect("error in Delta -= Delta");
    }
}

impl<T, U> CheckedSub for Delta<T, U>
where
    T: CheckedSub,
{
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        self.0.checked_sub(&rhs.0).map(Delta::new)
    }
}

impl<T, U> SaturatingSub for Delta<T, U>
where
    T: SaturatingSub + CheckedSub,
{
    fn saturating_sub(&self, rhs: &Self) -> Self {
        Delta::new(self.0.saturating_sub(&rhs.0))
    }
}

impl<T, U> Neg for Delta<T, U>
where
    T: CheckedNeg,
{
    type Output = Delta<T, U>;

    fn neg(self) -> Self::Output {
        Delta::new(self.0.checked_neg().expect("error in -Delta"))
    }
}

impl<T, U> Mul<T> for Delta<T, U>
where
    T: CheckedMul,
{
    type Output = Delta<T, U>;

    fn mul(self, rhs: T) -> Self::Output {
        Delta::new(self.0.checked_mul(&rhs).expect("error in Delta * scalar"))
    }
}

impl<T, U> MulAssign<T> for Delta<T, U>
where
    T: CheckedMul,
{
    fn mul_assign(&mut self, rhs: T) {
        self.0 = self.0.checked_mul(&rhs).expect("error in Delta *= scalar");
    }
}

impl<T, U> Div<T> for Delta<T, U>
where
    T: CheckedDiv,
{
    type Output = Delta<T, U>;

    fn div(self, rhs: T) -> Self::Output {
        Delta::new(self.0.checked_div(&rhs).expect("error in Delta / scalar"))
    }
}

impl<T, U> DivAssign<T> for Delta<T, U>
where
    T: CheckedDiv,
{
    fn div_assign(&mut self, rhs: T) {
        self.0 = self.0.checked_div(&rhs).expect("error in Delta /= scalar");
    }
}

impl<T, U> Sum for Delta<T, U>
where
    T: Zero + CheckedAdd,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, T, U> Sum<&'a Delta<T, U>> for Delta<T, U>
where
    T: Zero + CheckedAdd + Copy,
    U: Copy,
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    fn tp(v: i32) -> Point<i32, ()> {
        Point::new(v)
    }
    fn dt(v: i32) -> Delta<i32, ()> {
        Delta::new(v)
    }

    #[test]
    fn test_size_and_repr_transparent() {
        assert_eq!(mem::size_of::<Point<i32, ()>>(), mem::size_of::<i32>());
        assert_eq!(mem::size_of::<Delta<i32, ()>>(), mem::size_of::<i32>());
    }

    #[test]
    fn test_display() {
        struct MyMarker;
        impl MarkerName for MyMarker {
            const NAME_POINT: &'static str = "MyPoint";
            const NAME_DELTA: &'static str = "MyDelta";
        }

        let p: Point<i32, MyMarker> = Point::new(42);
        let d: Delta<i32, MyMarker> = Delta::new(7);

        assert_eq!(format!("{}", p), "MyPoint(42)");
        assert_eq!(format!("{}", d), "MyDelta(7)");
    }

    #[test]
    fn test_zero_and_default() {
        let pz = Point::<i32, ()>::zero();
        let dz = Delta::<i32, ()>::zero();
        let pd: Point<i32, ()> = Default::default();
        let dd: Delta<i32, ()> = Default::default();
        assert_eq!(pz.value(), 0);
        assert_eq!(dz.value(), 0);
        assert_eq!(pd.value(), 0);
        assert_eq!(dd.value(), 0);
        assert!(pz.is_zero());
        assert!(dz.is_zero());
    }

    #[test]
    fn test_point_plus_minus_delta() {
        let p = tp(10);
        let d = dt(5);
        assert_eq!((p.value()), 10);
        let p2 = p + d;
        assert_eq!(p2.value(), 15);

        let p3 = p2 - d;
        assert_eq!(p3.value(), 10);
    }

    #[test]
    fn test_point_add_assign_and_sub_assign() {
        let mut p = tp(1);
        p += dt(2);
        assert_eq!(p.value(), 3);
        p -= dt(1);
        assert_eq!(p.value(), 2);
    }

    #[test]
    fn test_point_minus_point_is_delta() {
        let p1 = tp(20);
        let p2 = tp(5);
        let d = p1 - p2;
        assert_eq!(d.value(), 15);
    }

    #[test]
    fn test_delta_add_sub() {
        let a = dt(7);
        let b = dt(3);
        assert_eq!((a + b).value(), 10);
        assert_eq!((a - b).value(), 4);
    }

    #[test]
    fn test_delta_add_assign_and_sub_assign() {
        let mut a = dt(10);
        a += dt(5);
        assert_eq!(a.value(), 15);
        a -= dt(3);
        assert_eq!(a.value(), 12);
    }

    #[test]
    fn test_delta_neg() {
        assert_eq!((-dt(3)).value(), -3);
        assert_eq!((-dt(-8)).value(), 8);
    }

    #[test]
    fn test_delta_scalar_mul_div() {
        let d = dt(12);
        assert_eq!((d * 2).value(), 24);
        assert_eq!((d / 3).value(), 4);
    }

    #[test]
    fn test_delta_checked_add_sub_mul_div_ok() {
        let a = dt(10);
        let b = dt(20);
        assert_eq!(a.checked_add(&b).unwrap().value(), 30);
        assert_eq!(b.checked_sub(&a).unwrap().value(), 10);
    }

    #[test]
    fn test_delta_saturating_add_sub() {
        let almost_max = Delta::new(i32::MAX - 1);
        let one = dt(1);
        // SaturatingAdd trait method
        let s = SaturatingAdd::saturating_add(&almost_max, &one);
        assert_eq!(s.value(), i32::MAX);

        let almost_min = Delta::new(i32::MIN + 1);
        // SaturatingSub trait method
        let t = SaturatingSub::saturating_sub(&almost_min, &dt(2));
        assert_eq!(t.value(), i32::MIN);
    }

    #[test]
    fn test_delta_sum_owned() {
        let v = vec![dt(1), dt(2), dt(3), dt(4)];
        let sum: Delta<i32, ()> = v.into_iter().sum();
        assert_eq!(sum.value(), 10);
    }

    #[test]
    fn test_delta_sum_refs() {
        let v = vec![dt(5), dt(6), dt(-2)];
        let sum: Delta<i32, ()> = v.iter().sum();
        assert_eq!(sum.value(), 9);
    }

    #[test]
    #[should_panic(expected = "error in Point + Delta")]
    fn panic_point_add_overflow() {
        let p = tp(i32::MAX);
        let _ = p + dt(1);
    }

    #[test]
    #[should_panic(expected = "error in Point - Delta")]
    fn panic_point_sub_underflow() {
        let p = tp(i32::MIN);
        let _ = p - dt(1);
    }

    #[test]
    #[should_panic(expected = "error in Point - Point")]
    fn test_panic_point_minus_point_underflow() {
        let p1 = tp(i32::MIN + 1);
        let p2 = tp(i32::MAX);
        let _ = p1 - p2;
    }

    #[test]
    #[should_panic(expected = "error in Delta + Delta")]
    fn test_panic_delta_add_overflow() {
        let a = Delta::new(i32::MAX);
        let b = dt(1);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "error in Delta - Delta")]
    fn test_panic_delta_sub_underflow() {
        let a = Delta::new(i32::MIN);
        let b = dt(1);
        let _ = a - b;
    }

    #[test]
    #[should_panic(expected = "error in -Delta")]
    fn test_panic_neg_min() {
        let a: Delta<i32, ()> = Delta::new(i32::MIN);
        let _ = -a;
    }

    #[test]
    #[should_panic(expected = "error in Delta * scalar")]
    fn test_panic_mul_overflow() {
        let a: Delta<i32, ()> = Delta::new(i32::MAX / 2 + 1);
        let _ = a * 2;
    }

    #[test]
    #[should_panic(expected = "error in Delta / scalar")]
    fn test_panic_div_by_zero() {
        let a = dt(123);
        let _ = a / 0;
    }

    #[test]
    #[should_panic(expected = "error in Delta / scalar")]
    fn test_panic_div_min_by_minus_one() {
        let a: Delta<i32, ()> = Delta::new(i32::MIN);
        let _ = a / -1;
    }
}
