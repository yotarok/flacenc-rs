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

//! Fake SIMD module is a minimal subset of `std::simd` of `portable_simd`
//! feature in a nightly rust. This module only implement the functions that
//! are used in flacenc.

use std::array;

// ===
// TYPE DEFINITION
// ===
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Simd<T: SimdElement, const LANES: usize>([T; LANES]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub struct Mask<T: SimdElement, const LANES: usize> {
    mask: [bool; LANES],
    phantom_data: std::marker::PhantomData<T>,
}

// ===
// SIMD ELEMENT TRAITS and `SupportedLaneCount`
// ===
pub trait SimdElement: Copy + std::fmt::Debug {
    type Mask;
}

impl SimdElement for i32 {
    type Mask = Self;
}

impl SimdElement for u32 {
    type Mask = Self;
}

impl SimdElement for f32 {
    type Mask = Self;
}

pub trait SupportedLaneCount {}
pub struct LaneCount<const LANES: usize>();
impl SupportedLaneCount for LaneCount<16> {}
impl SupportedLaneCount for LaneCount<32> {}

// ===
// SIMD OP TRAITS
// ===
pub trait SimdUint {
    type Scalar;
    fn reduce_max(self) -> Self::Scalar;
    fn reduce_min(self) -> Self::Scalar;
}

pub trait SimdInt {
    type Scalar;
    #[allow(clippy::return_self_not_must_use)]
    fn abs(self) -> Self;
    #[allow(clippy::return_self_not_must_use)]
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

// ===
// TYPE ALIASES
// ===
#[allow(non_camel_case_types)]
pub type i32x16 = Simd<i32, 16>;

#[allow(non_camel_case_types)]
pub type u32x16 = Simd<u32, 16>;

#[allow(non_camel_case_types)]
pub type f32x32 = Simd<f32, 32>;

// ===
// IMPLEMENTATION OF `fakesimd::Simd` (non-trait methods)
// ===
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

    #[allow(clippy::return_self_not_must_use)]
    #[inline]
    pub fn rotate_lanes_right<const OFFSET: usize>(self) -> Self {
        Self(array::from_fn(|i| self.0[(i + N - OFFSET) % N]))
    }
}

// ===
// IMPLEMENTATION OF `fakesimd::Mask`
// ===
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

// ===
// IMPLEMENTATION OF SIMD-SPECIFIC OPS
// ===
impl<T, const N: usize> SimdInt for Simd<T, N>
where
    T: SimdElement + num_traits::PrimInt + num_traits::Signed + std::iter::Sum,
{
    type Scalar = T;
    #[inline]
    fn abs(self) -> Self {
        Self(array::from_fn(|i| num_traits::sign::abs(self.0[i])))
    }

    #[inline]
    fn signum(self) -> Self {
        Self(array::from_fn(|i| num_traits::sign::signum(self.0[i])))
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

// ===
// IMPLEMENTATION OF OPERATOR OVERRIDES
// ===
macro_rules! def_binop {
    ($trait_name:ident, $fn_name:ident, $expr:expr) => {
        impl<T, const N: usize> std::ops::$trait_name<Self> for Simd<T, N>
        where
            T: SimdElement + std::ops::$trait_name<T, Output = T>,
        {
            type Output = Self;
            #[allow(clippy::redundant_closure_call)]
            #[inline]
            #[allow(clippy::redundant_closure_call)]
            fn $fn_name(self, rhs: Self) -> Self::Output {
                Self(array::from_fn(|i| ($expr)(self.0[i], rhs.0[i])))
            }
        }
    };
}

macro_rules! def_binop_assign {
    ($trait_name:ident, $binop_name: ident, $fn_name:ident, $expr:expr) => {
        impl<U, T, const N: usize> std::ops::$trait_name<U> for Simd<T, N>
        where
            T: SimdElement,
            Self: std::ops::$binop_name<U, Output = Self>,
        {
            #[inline]
            #[allow(clippy::redundant_closure_call)]
            fn $fn_name(&mut self, rhs: U) {
                *self = ($expr)(*self, rhs);
            }
        }
    };
}

def_binop!(Add, add, |x, y| x + y);
def_binop!(Sub, sub, |x, y| x - y);
def_binop!(Mul, mul, |x, y| x * y);
def_binop!(Div, div, |x, y| x / y);
def_binop!(BitAnd, bitand, |x, y| x & y);
def_binop!(Shr, shr, |x, y| x >> y);

def_binop_assign!(AddAssign, Add, add_assign, |x, y| x + y);
def_binop_assign!(SubAssign, Sub, sub_assign, |x, y| x - y);

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
