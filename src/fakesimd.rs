// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Fake SIMD module is a minimal subset of std::simd of "portable_simd"
//! feature in a nightly rust.

use std::array;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Simd<T: SimdElement, const LANES: usize>([T; LANES]);
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Mask<T: SimdElement, const LANES: usize> {
    mask: [bool; LANES],
    phantom_data: std::marker::PhantomData<T>,
}

impl<T, const N: usize> Simd<T, N>
where
    T: SimdElement,
{
    #[inline]
    pub const fn from_array(array: [T; N]) -> Self {
        Self(array)
    }

    #[inline]
    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }

    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [T; N] {
        &mut self.0
    }

    #[inline]
    pub fn splat(v: T) -> Self {
        Self([v; N])
    }

    #[inline]
    pub fn rotate_lanes_right<const OFFSET: usize>(self) -> Self {
        Self(array::from_fn(|i| self.0[(i + N - OFFSET) % N]))
    }
}

#[allow(non_camel_case_types)]
pub type i32x16 = Simd<i32, 16>;

#[allow(non_camel_case_types)]
pub type u32x16 = Simd<u32, 16>;

pub trait SimdUint {
    type Scalar;
    fn reduce_max(self) -> Self::Scalar;
    fn reduce_min(self) -> Self::Scalar;
}

pub trait SimdInt {
    type Scalar;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn reduce_sum(self) -> Self::Scalar;
}
pub trait SimdPartialEq {
    type Mask;
    fn simd_eq(self, other: Self) -> Self::Mask;
}
pub trait SimdPartialOrd {
    type Mask;
    fn simd_le(self, other: Self) -> Self::Mask;
}

pub trait SimdElement: Copy + std::fmt::Debug {
    type Mask;
}
impl SimdElement for i32 {
    type Mask = i32;
}
impl SimdElement for u32 {
    type Mask = u32;
}

pub trait SupportedLaneCount {}
pub struct LaneCount<const LANES: usize>();
impl SupportedLaneCount for LaneCount<16> {}
impl SupportedLaneCount for LaneCount<32> {}

impl<T, const N: usize> Mask<T, N>
where
    T: SimdElement,
{
    #[inline]
    pub fn select(self, true_values: Simd<T, N>, false_values: Simd<T, N>) -> Simd<T, N> {
        Simd(array::from_fn(|i| {
            if self.mask[i] {
                true_values.0[i]
            } else {
                false_values.0[i]
            }
        }))
    }
}

impl<T, const N: usize> SimdInt for Simd<T, N>
where
    T: SimdElement + num_traits::PrimInt + num_traits::Signed + std::iter::Sum,
{
    type Scalar = T;
    #[inline]
    fn abs(self) -> Self {
        Simd(array::from_fn(|i| num_traits::sign::abs(self.0[i])))
    }

    #[inline]
    fn signum(self) -> Self {
        Simd(array::from_fn(|i| num_traits::sign::signum(self.0[i])))
    }

    #[inline]
    fn reduce_sum(self) -> T {
        self.0.into_iter().sum()
    }
}

impl<T, const N: usize> SimdUint for Simd<T, N>
where
    T: SimdElement + num_traits::PrimInt,
{
    type Scalar = T;
    #[inline]
    fn reduce_max(self) -> T {
        self.0
            .into_iter()
            .max()
            .expect("INTERNAL ERROR in `reduce_max` of fakesimd.")
    }
    #[inline]
    fn reduce_min(self) -> T {
        self.0
            .into_iter()
            .min()
            .expect("INTERNAL ERROR in `reduce_min` of fakesimd.")
    }
}

impl<T, const N: usize> SimdPartialEq for Simd<T, N>
where
    T: SimdElement + PartialEq,
{
    type Mask = Mask<T, N>;
    #[inline]
    fn simd_eq(self, other: Self) -> Self::Mask {
        Mask {
            mask: array::from_fn(|i| self.0[i] == other.0[i]),
            phantom_data: std::marker::PhantomData,
        }
    }
}

impl<T, const N: usize> SimdPartialOrd for Simd<T, N>
where
    T: SimdElement + PartialOrd,
{
    type Mask = Mask<T, N>;
    #[inline]
    fn simd_le(self, other: Self) -> Self::Mask {
        Mask {
            mask: array::from_fn(|i| self.0[i] <= other.0[i]),
            phantom_data: std::marker::PhantomData,
        }
    }
}

impl<T, const N: usize> std::convert::AsRef<[T; N]> for Simd<T, N>
where
    T: SimdElement,
{
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> std::convert::From<Simd<T, N>> for [T; N]
where
    T: SimdElement,
{
    #[inline]
    fn from(t: Simd<T, N>) -> [T; N] {
        t.0
    }
}

// arithmatic ops

impl<I, T, const N: usize> std::ops::Index<I> for Simd<T, N>
where
    T: SimdElement,
    I: std::slice::SliceIndex<[T]>,
{
    type Output = <I as std::slice::SliceIndex<[T]>>::Output;
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.as_array()[index]
    }
}

impl<I, T, const N: usize> std::ops::IndexMut<I> for Simd<T, N>
where
    T: SimdElement,
    I: std::slice::SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut <I as std::slice::SliceIndex<[T]>>::Output {
        &mut self.as_mut_array()[index]
    }
}

impl<T, const N: usize> std::ops::Add<Simd<T, N>> for Simd<T, N>
where
    T: SimdElement + std::ops::Add<T, Output = T>,
{
    type Output = Simd<T, N>;
    #[inline]
    fn add(self, rhs: Simd<T, N>) -> Self::Output {
        Simd(array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<T, const N: usize> std::ops::Sub<Simd<T, N>> for Simd<T, N>
where
    T: SimdElement + std::ops::Sub<T, Output = T>,
{
    type Output = Simd<T, N>;
    #[inline]
    fn sub(self, rhs: Simd<T, N>) -> Self::Output {
        Simd(array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<T, const N: usize> std::ops::Mul<Simd<T, N>> for Simd<T, N>
where
    T: SimdElement + std::ops::Mul<T, Output = T>,
{
    type Output = Simd<T, N>;
    #[inline]
    fn mul(self, rhs: Simd<T, N>) -> Self::Output {
        Simd(array::from_fn(|i| self.0[i] * rhs.0[i]))
    }
}

impl<T, const N: usize> std::ops::Div<Simd<T, N>> for Simd<T, N>
where
    T: SimdElement + std::ops::Div<T, Output = T>,
{
    type Output = Simd<T, N>;
    #[inline]
    fn div(self, rhs: Simd<T, N>) -> Self::Output {
        Simd(array::from_fn(|i| self.0[i] / rhs.0[i]))
    }
}

impl<T, const N: usize> std::ops::BitAnd<Simd<T, N>> for Simd<T, N>
where
    T: SimdElement + std::ops::BitAnd<T, Output = T>,
{
    type Output = Simd<T, N>;
    #[inline]
    fn bitand(self, rhs: Simd<T, N>) -> Self::Output {
        Simd(array::from_fn(|i| self.0[i] & rhs.0[i]))
    }
}

impl<U, T, const N: usize> std::ops::SubAssign<U> for Simd<T, N>
where
    T: SimdElement,
    Simd<T, N>: std::ops::Sub<U, Output = Simd<T, N>>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: U) {
        *self = *self - rhs;
    }
}
