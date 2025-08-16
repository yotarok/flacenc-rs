// Copyright 2022-2024 Google LLC
// Copyright 2025- flacenc-rs developers
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

//! Algorithms for quantized linear-prediction coding (QLPC).

use std::collections::BTreeMap;
use std::rc::Rc;

use num_traits::AsPrimitive;
use num_traits::Float;

use super::arrayutils::find_max_abs;
use super::arrayutils::unaligned_map_and_update;
use super::arrayutils::SimdVec;
use super::component::QuantizedParameters;
use super::config::Window;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::qlpc::MAX_SHIFT as QLPC_MAX_SHIFT;
use super::constant::qlpc::MIN_SHIFT as QLPC_MIN_SHIFT;
use super::repeat;
use super::repeat::repeat;

import_simd!(as simd);

/// Trait for a type that can be used for storing LPC statistics/ parameters.
///
/// Currently, it is only implemented for f32/ f64.
#[allow(clippy::module_name_repetitions)]
pub trait LpcFloat:
    Float
    + std::ops::AddAssign
    + std::ops::MulAssign
    + std::iter::Sum
    + std::fmt::Debug
    + std::fmt::Display
    + simd::SimdElement
    + simd::SimdCast
    + From<f32>
    + From<i16>
    + AsPrimitive<f32>
    + AsPrimitive<i16>
{
    #[allow(dead_code)]
    type Simd<const N: usize>: SimdFloat<Scalar = Self>
        + StdFloat
        + Copy
        + From<simd::Simd<Self, N>>
    where
        simd::LaneCount<N>: simd::SupportedLaneCount;

    /// Solves symetric positive-definite linear equation in-place.
    ///
    /// This computes `v = matmul(inverse(mat), v)` where `mat` is assumed to be
    /// symmetric positive-definite (SPD), and if not it returns `false`.
    /// Otherwise, it returns `true` and `v` is overwritten by the solution.
    #[allow(dead_code)]
    #[cfg(feature = "experimental")]
    fn solve_sym_mut(mat: &nalgebra::DMatrix<Self>, v: &mut nalgebra::DVector<Self>) -> bool;
}

macro_rules! def_lpc_float {
    ($ty:ty) => {
        impl self::LpcFloat for $ty {
            type Simd<const N: usize>
                = simd::Simd<$ty, N>
            where
                simd::LaneCount<N>: simd::SupportedLaneCount;

            #[cfg(feature = "experimental")]
            #[inline]
            fn solve_sym_mut(
                mat: &nalgebra::DMatrix<Self>,
                v: &mut nalgebra::DVector<Self>,
            ) -> bool {
                mat.clone().cholesky().map_or(false, |decompose| {
                    decompose.solve_mut(v);
                    true
                })
            }
        }
    };
}
def_lpc_float!(f32);
def_lpc_float!(f64);

/// Precomputes window function given the window config `win`.
#[inline]
pub fn window_weights(win: &Window, len: usize) -> Vec<f32> {
    match *win {
        Window::Rectangle => vec![1.0f32; len],
        Window::Tukey { alpha: 0.0 } => {
            vec![1.0f32; len]
        }
        Window::Tukey { alpha } => {
            let max_t = len as f32 - 1.0;
            let alpha_len = alpha * max_t;
            let mut ret = Vec::with_capacity(len);
            for t in 0..len {
                let t = t as f32;
                let w = if t < alpha_len / 2.0 {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * t / alpha_len).cos())
                } else if t < max_t - alpha_len / 2.0 {
                    1.0
                } else {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * (max_t - t) / alpha_len).cos())
                };
                ret.push(w);
            }
            ret
        }
    }
}

/// Quantizes and fingerprints the window function for caching.
fn fingerprint_window(w: &Window) -> u64 {
    match *w {
        Window::Rectangle => 0x01_00_00_00_00_00_00_00u64,
        Window::Tukey { alpha } => {
            let qalpha = (alpha * 65535.0) as u64;
            assert!(qalpha < 65536, "alpha is larger than 1");
            0x02_00_00_00_00_00_00_00u64 + qalpha
        }
    }
}

/// A struct used for indexing window cache.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WindowKey {
    /// Size of the window cache.
    size: usize,
    /// A fingerprint computed from window-specific hyper-parameters.
    fingerprint: u64,
}

impl WindowKey {
    /// Constructs `WindowKey` with given window size and parameters.
    fn new(size: usize, params: &Window) -> Self {
        Self {
            size,
            fingerprint: fingerprint_window(params),
        }
    }
}

/// Trait for a weighting function when collecting the second order statistics.
///
/// It is only interesting in "experimental" build, so far, only `NoWeight` is used
/// in non-experimental build.
pub trait Weight {
    /// Apply weight to a sample `x` at time-offset `t`.
    fn apply(&self, t: usize, x: f32) -> f32;
    /// Apply weights to a vector of samples `x` starting at time-offset `t0`.
    #[cfg(feature = "experimental")]
    #[allow(dead_code)]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N>
    where
        simd::LaneCount<N>: simd::SupportedLaneCount;
}

struct NoWeight;
#[cfg(feature = "experimental")]
struct VecWeight(Vec<f32>);
#[cfg(feature = "experimental")]
struct ShiftedWeight<const M: usize, W: Weight>(W);

impl<W: Weight> Weight for &W {
    #[inline]
    fn apply(&self, t: usize, x: f32) -> f32 {
        (*self).apply(t, x)
    }
    #[cfg(feature = "experimental")]
    #[inline]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N>
    where
        simd::LaneCount<N>: simd::SupportedLaneCount,
    {
        (*self).apply_simd(t0, x)
    }
}

impl Weight for NoWeight {
    #[inline]
    fn apply(&self, _t: usize, x: f32) -> f32 {
        x
    }
    #[cfg(feature = "experimental")]
    #[inline]
    fn apply_simd<const N: usize>(&self, _t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N>
    where
        simd::LaneCount<N>: simd::SupportedLaneCount,
    {
        x
    }
}

#[cfg(feature = "experimental")]
impl Weight for VecWeight {
    #[inline]
    fn apply(&self, t: usize, x: f32) -> f32 {
        self.0[t] * x
    }
    #[inline]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N>
    where
        simd::LaneCount<N>: simd::SupportedLaneCount,
    {
        x * simd::Simd::<f32, N>::from_slice(&self.0[t0..(t0 + N)])
    }
}

#[cfg(feature = "experimental")]
impl<W: Weight, const M: usize> Weight for ShiftedWeight<M, W> {
    #[inline]
    fn apply(&self, t: usize, x: f32) -> f32 {
        self.0.apply(t + M, x)
    }
    #[inline]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N>
    where
        simd::LaneCount<N>: simd::SupportedLaneCount,
    {
        self.0.apply_simd(t0 + M, x)
    }
}

const QLPC_WIN_SIMD_N: usize = 16;
type WindowMap = BTreeMap<WindowKey, Rc<SimdVec<f32, QLPC_WIN_SIMD_N>>>;
reusable!(WINDOW_CACHE: WindowMap);

/// Gets the window function for the given config and size.
fn get_window(window: &Window, size: usize) -> Rc<SimdVec<f32, QLPC_WIN_SIMD_N>> {
    let key = WindowKey::new(size, window);
    reuse!(WINDOW_CACHE, |caches: &mut WindowMap| {
        if caches.get(&key).is_none() {
            let v = window_weights(window, size);
            caches.insert(key.clone(), Rc::from(SimdVec::from_slice(&v)));
        }
        Rc::clone(caches.get(&key).expect(panic_msg::ERROR_NOT_EXPECTED))
    })
}

/// Finds shift parameter for quantizing the given set of coefficients.
fn find_shift<T>(coefs: &[T], precision: usize) -> i8
where
    T: LpcFloat,
{
    assert!(precision <= 15);
    assert!(!coefs.is_empty());
    let max_abs_coef: T = coefs
        .iter()
        .copied()
        .map(Float::abs)
        .reduce(T::max)
        .unwrap();
    // location of MSB in binary representations of absolute values.
    let abs_log2: i16 = Float::max(
        Float::ceil(Float::log2(max_abs_coef)),
        <T as From<i16>>::from(i16::MIN + 16),
    )
    .as_();
    let shift: i16 = (precision as i16 - 1) - abs_log2;
    shift.clamp(i16::from(QLPC_MIN_SHIFT), i16::from(QLPC_MAX_SHIFT)) as i8
}

/// Quantizes LPC parameter with the given shift parameter.
#[inline]
fn quantize_parameter<T>(p: T, shift: i8) -> i16
where
    T: LpcFloat,
{
    let scalefac = Float::powi(<T as From<i16>>::from(2), i32::from(shift));
    let scaled_int = Float::round(p * scalefac);
    num_traits::clamp(
        scaled_int,
        <T as From<i16>>::from(i16::MIN),
        <T as From<i16>>::from(i16::MAX),
    )
    .as_()
}

/// Creates [`QuantizedParameters`] by quantizing the given coefficients.
pub fn quantize_parameters<T>(coefs: &[T], precision: usize) -> QuantizedParameters
where
    T: LpcFloat,
{
    if coefs.is_empty() {
        return QuantizedParameters::from_parts(&[], 0, 0, precision);
    }
    let shift = find_shift(coefs, precision);
    let mut q_coefs = [0i16; MAX_LPC_ORDER];

    for (n, coef) in coefs.iter().enumerate() {
        // This clamp op is mainly for safety, but actually required
        // because the shift-width estimation `find_shift` used here is not
        // perfect, and quantization may yields "2^(p-1)" quantized value
        // for precision "p" configuration, that is larger than a maximum
        // p-bits signed integer "2^(p-1) - 1".
        q_coefs[n] = std::cmp::min(
            std::cmp::max(quantize_parameter(*coef, shift), -(1 << (precision - 1))),
            (1 << (precision - 1)) - 1,
        );
    }

    let tail_zeros = q_coefs
        .rsplitn(2, |&x| x != 0)
        .next()
        .map_or(0, <[i16]>::len);
    let order = std::cmp::max(1, q_coefs.len() - tail_zeros);

    QuantizedParameters::from_parts(&q_coefs[0..order], order, shift, precision)
}

/// Implementation of `compute_error` for each SIMD config.
#[inline]
fn compute_error_impl<T, const N: usize>(qps: &QuantizedParameters, signal: &[T], errors: &mut [T])
where
    T: simd::SimdElement + num_traits::int::PrimInt + From<i8> + From<i16> + std::ops::AddAssign<T>,
    simd::Simd<T, N>: std::ops::Shr<simd::Simd<T, N>, Output = simd::Simd<T, N>>
        + std::ops::Sub<simd::Simd<T, N>, Output = simd::Simd<T, N>>
        + std::ops::Mul<simd::Simd<T, N>, Output = simd::Simd<T, N>>
        + std::ops::AddAssign<simd::Simd<T, N>>,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    let block_size = signal.len();
    debug_assert!(errors.len() >= block_size);
    errors.fill(T::zero());

    for order in 0..qps.order() {
        let w = qps.coefs[order].into();
        let wv = simd::Simd::<T, N>::splat(w);
        unaligned_map_and_update(
            &signal[0..block_size - order - 1],
            &mut errors[order + 1..],
            #[inline]
            |px, x| {
                *px += w * x;
            },
            #[inline]
            |pv, v| {
                *pv += wv * v;
            },
        );
    }

    let shift = qps.shift() as usize;
    let shift_v = simd::Simd::<T, N>::splat(qps.shift().into());
    unaligned_map_and_update::<T, N, _, _, _>(
        signal,
        errors,
        #[inline]
        |px, x| {
            *px = x - (*px >> shift);
        },
        #[inline]
        |pv, v| {
            *pv = v - (*pv >> shift_v);
        },
    );
    errors[0..qps.order()].fill(T::zero());
}

/// Compute error signal from `QuantizedParameters`.
///
/// # Panics
///
/// This function panics if `errors.len()` is smaller than `signal.len()`.
#[allow(clippy::collapsible_else_if)]
pub fn compute_error(qps: &QuantizedParameters, signal: &[i32], errors: &mut [i32]) {
    assert!(errors.len() >= signal.len());
    let maxabs_signal: u64 = find_max_abs::<16>(signal).into();
    // `Simd::reduce_sum` is avoided to mitigate overflow error.
    // NOTE: If we restrict the precision to be 11 bit, 24-additions of 11-bit
    //       ints are 16-bit safe. we assume it's reasonably fast.
    let sumabs_coefs: i64 = {
        let mut acc: i64 = 0i64;
        let abs_coefs = qps.coefs.abs();
        repeat!(lane to 32 => {
            acc += i64::from(abs_coefs.as_array()[lane]);
        });
        acc
    };
    let maxabs = maxabs_signal * sumabs_coefs as u64;
    if maxabs < i32::MAX as u64 {
        // larger lanes here can alleviate inefficiency of unaligned reads.
        compute_error_impl::<i32, 64>(qps, signal, errors);
    } else {
        // This is very inefficient, but should rarely happen in BPS=16bit case.
        let signal64: Vec<i64> = signal.iter().map(|v| (*v).into()).collect();
        let mut errors64 = vec![0i64; signal64.len()];
        compute_error_impl::<i64, 64>(qps, &signal64, &mut errors64);
        for (v, p) in errors64
            .into_iter()
            .map(|v| v as i32)
            .zip(errors.iter_mut())
        {
            *p = v;
        }
    }
}

/// Compute auto-correlation coefficients.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[allow(dead_code)]
pub fn auto_correlation<T: LpcFloat>(order: usize, signal: &[f32], dest: &mut [T]) {
    weighted_auto_correlation(order, signal, dest, NoWeight);
}

/// Computes the sum of outer products of lagged vectors.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "experimental")]
#[allow(dead_code)]
pub fn lagged_outer_prod_sum<T>(order: usize, signal: &[f32], dest: &mut nalgebra::DMatrix<T>)
where
    T: LpcFloat,
{
    weighted_lagged_outer_prod_sum(order, signal, dest, NoWeight);
}

/// Computes sum of `x[t] * y[t] * weight(t_offset + t)`s.
#[inline]
#[cfg(feature = "simd-nightly")]
fn weighted_prod_sum<T, W>(t_offset: usize, x: &[f32], y: &[f32], weight: W) -> T
where
    T: LpcFloat,
    W: Weight,
{
    let mut acc = T::zero();
    for (tau, (x, delayed_x)) in x.iter().copied().zip(y.iter().copied()).enumerate() {
        let wx = Into::<T>::into(weight.apply(t_offset + tau, x));
        acc = Float::mul_add(delayed_x.into(), wx, acc);
    }
    acc
}

/// Internal function that computes the sum of `signal[t] * signal[t-DELAY] * weight(t)`s.
///
/// This function takes arguments as const generics, and this necessitates us to have a
/// redundant parameter `LANES_MINUS_DELAY` which is assumed to be always `LANES - DELAY`.
/// This is due to a current limitation of constant computation in Rust.
#[cfg(feature = "simd-nightly")]
#[inline]
fn weighted_delay_prod_sum_impl<
    T,
    W,
    const LANES: usize,
    const DELAY: usize,
    const LANES_MINUS_DELAY: usize,
>(
    warm_up: usize,
    signal: &[f32],
    weight: W,
) -> T
where
    T: LpcFloat,
    simd::LaneCount<LANES>: simd::SupportedLaneCount,
    W: Weight,
{
    assert!(DELAY <= LANES);
    assert!(LANES_MINUS_DELAY == LANES - DELAY);
    let mut acc = T::zero();

    let delayed_signal = &signal[warm_up - DELAY..];
    let (head, body, foot) = signal[warm_up..].as_simd();
    let mut t_offset = warm_up;

    acc += weighted_prod_sum(t_offset, head, delayed_signal, &weight);
    t_offset += head.len();

    // this is a bit awkward to use f32 for `indices`, but this can reduce some complexity of
    // implementing `fakesimd::Mask::cast`. this is required for compilation even though this
    // loop is not actually used. We can resort conditional compilation as well, but conditional
    // compilation is also not very clean.
    let indices = simd::Simd::from_array(std::array::from_fn(|n| n as f32));
    let mask = indices.simd_lt(simd::Simd::splat(DELAY as f32));
    // ^ first `DELAY` lanes are true.

    let mut prev_v = simd::Simd::from_array(std::array::from_fn(|n| {
        if warm_up + n < LANES {
            0.0
        } else {
            signal.get(t_offset + n - LANES).copied().unwrap_or(0.0)
        }
    }));
    let mut acc_v: T::Simd<LANES> = simd::Simd::splat(T::zero()).into();
    for v in body.iter().copied::<simd::Simd<f32, LANES>>() {
        prev_v = mask.select(
            prev_v.rotate_elements_left::<LANES_MINUS_DELAY>(),
            v.rotate_elements_right::<DELAY>(),
        );
        let wv = weight.apply_simd(t_offset, v);

        acc_v = T::Simd::mul_add(wv.cast().into(), prev_v.cast().into(), acc_v);
        prev_v = v;
        t_offset += LANES; // this needs to be updated in each iteration since weight refers it.
    }

    acc += weighted_prod_sum(
        t_offset,
        foot,
        &delayed_signal[t_offset - warm_up..],
        &weight,
    );
    acc + acc_v.reduce_sum()
}

/// Compute weighted auto-correlation coefficients.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "simd-nightly")]
#[inline]
#[allow(clippy::cognitive_complexity)] // so far complexity is hidden by seq macros.
pub fn weighted_auto_correlation_simd<T, W>(order: usize, signal: &[f32], dest: &mut [T], weight: W)
where
    T: LpcFloat,
    W: Weight,
{
    let warmup = order - 1;
    let weight = &weight;
    seq_macro::seq!(DELAY in 0..=32 {
        #[allow(clippy::unnecessary_semicolon)]
        if DELAY < order {
            // `LANES` is starting from 8.
            #[allow(clippy::identity_op)] // delay may be zero.
            #[allow(clippy::eq_op)] // delay may be 0x07.
            const LANES: usize = usize::next_power_of_two((DELAY | 0x07) - 1);
            #[allow(clippy::identity_op)] // delay may be zero.
            const LANES_MINUS_DELAY: usize = LANES - DELAY;
            dest[DELAY] = weighted_delay_prod_sum_impl::<
                T, _, LANES, DELAY, LANES_MINUS_DELAY
            >(warmup, signal, weight);
        }
    });
}

pub fn weighted_auto_correlation_nosimd<T, W>(
    order: usize,
    signal: &[f32],
    dest: &mut [T],
    weight: W,
) where
    T: LpcFloat,
    W: Weight,
{
    for t in (order - 1)..signal.len() {
        let wy: T = weight.apply(t, signal[t]).into();
        repeat!(tau to { MAX_LPC_ORDER + 1 } ; while tau < order => {
            dest[tau] = Float::mul_add(Into::<T>::into(signal[t - tau]), wy, dest[tau]);
        });
    }
}

/// Computes auto-correlation function up to `order`.
pub fn weighted_auto_correlation<T, W>(order: usize, signal: &[f32], dest: &mut [T], weight: W)
where
    T: LpcFloat,
    W: Weight,
{
    assert!(dest.len() >= order);
    for p in &mut *dest {
        *p = T::zero();
    }
    #[cfg(feature = "simd-nightly")]
    weighted_auto_correlation_simd(order, signal, dest, weight);
    #[cfg(not(feature = "simd-nightly"))]
    weighted_auto_correlation_nosimd(order, signal, dest, weight);
}

/// Compute weighted lagged-outer-prod-sum statistics.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "experimental")]
#[inline]
pub fn weighted_lagged_outer_prod_sum<T, W>(
    order: usize,
    signal: &[f32],
    dest: &mut nalgebra::DMatrix<T>,
    weight: W,
) where
    W: Weight,
    T: LpcFloat,
{
    assert!(dest.ncols() >= order);
    assert!(dest.nrows() >= order);

    dest.fill(T::zero());

    for t in (order - 1)..signal.len() {
        for i in 0..order {
            for j in i..order {
                let wx = Into::<T>::into(weight.apply(t, signal[t - j]));
                dest[(i, j)] = Float::mul_add(signal[t - i].into(), wx, dest[(i, j)]);
            }
        }
    }
    for i in 0..order {
        for j in (i + 1)..order {
            dest[(j, i)] = dest[(i, j)];
        }
    }
}

/// Computes raw errors from unquantized LPC coefficients.
///
/// This function computes "prediction - signal" in floating-point numbers.
#[allow(dead_code)] // Used either in experimental or tests of non-experimental.
fn compute_raw_errors<T>(signal: &[i32], lpc_coefs: &[T], errors: &mut [f32])
where
    T: LpcFloat,
{
    let lpc_order = lpc_coefs.len();
    for t in lpc_order..signal.len() {
        errors[t] = -signal[t] as f32;
        for j in 0..lpc_order {
            let coef: f32 = lpc_coefs[j].as_();
            errors[t] += coef * signal[t - 1 - j] as f32;
        }
    }
}

/// Solves "y = T x" where T is a Toeplitz matrix with the given coefficients.
///
/// The (i, j)-th element of the Toeplitz matrix "T" is defined by
/// `coefs[(i - j).abs()]`, and the i-th element of "y" is defined as `ys[i]`.
/// The solution "x" will be stored in `dest`.
///
/// # Panics
///
/// Panics if `dest` or `coefs` is shorter than `ys`. In addition to that,
/// the following preconditions are checked.
/// 1. Signal energy `coefs[0]` is non-negative.
/// 2. If signal-energy is zero, all `coefs` and `ys` must be zero.
#[inline]
pub fn symmetric_levinson_recursion<T, const N: usize>(coefs: &[T], ys: &[T], dest: &mut [T])
where
    T: LpcFloat,
    repeat::Count<N>: repeat::Repeat,
{
    assert!(dest.len() >= ys.len());
    assert!(coefs.len() >= ys.len());

    for p in &mut *dest {
        *p = T::zero();
    }

    // coefs[0] is energy of the signal, so must be non-negative.
    assert!(coefs[0] >= T::zero());
    if coefs[0].is_zero() {
        let allzero = ys
            .iter()
            .chain(coefs.iter())
            .fold(true, |f, &v| f & v.is_zero());
        assert!(
            allzero,
            "If signal is digital silence, all coefficients must be zero."
        );
        return;
    }

    let order = ys.len();
    let mut forward = [T::zero(); N];
    let mut forward_next = [T::zero(); N];
    let mut diagonal_loading = T::zero();

    // this actually should use a go-to statement.
    #[allow(clippy::never_loop)]
    loop {
        forward[0] = Float::recip(coefs[0] + diagonal_loading);
        dest[0] = ys[0] / (coefs[0] + diagonal_loading);

        for n in 1..order {
            let error = {
                let mut acc = T::zero();
                repeat!(d to N ; while d < n => {
                    acc = Float::mul_add(coefs[n - d], forward[d], acc);
                });
                acc
            };
            let denom = Float::mul_add(error, -error, T::one());
            if denom.is_zero() {
                diagonal_loading = T::one().max(diagonal_loading + diagonal_loading);
                continue;
            }
            let alpha = Float::recip(denom);
            let beta = -alpha * error;
            repeat!(d to N ; while d <= n => {
                forward_next[d] = Float::mul_add(alpha, forward[d], beta * forward[n - d]);
            });
            repeat!(d to N ; while d <= n => {
                forward[d] = forward_next[d];
            });

            let delta = {
                let mut acc = T::zero();
                repeat!(d to N ; while d < n => {
                    acc = Float::mul_add(coefs[n - d], dest[d], acc);
                });
                acc
            };
            repeat!(d to N ; while d <= n => {
                dest[d] = Float::mul_add(ys[n] - delta, forward[n - d], dest[d]);
            });
        }
        break;
    }
}

/// Working buffer for (unquantized) LPC estimation.
struct LpcEstimator<T> {
    /// Buffer for storing windowed signal.
    windowed_signal: SimdVec<f32, QLPC_WIN_SIMD_N>,
    /// Buffer for storing auto-correlation coefficients.
    corr_coefs: Vec<T>,
    /// Buffer for delay-sum matrix and it's inverse. (not used in auto-correlation mode.)
    #[cfg(feature = "experimental")]
    lagged_outer_prod_sum: nalgebra::DMatrix<T>,
    /// Weights for IRLS.
    #[cfg(feature = "experimental")]
    weights: Vec<f32>,
}

reusable!(CAST_BUFFER: SimdVec<i32, QLPC_WIN_SIMD_N> = SimdVec::new());

impl<T> LpcEstimator<T>
where
    T: LpcFloat,
{
    pub fn new() -> Self {
        Self {
            windowed_signal: SimdVec::new(),
            corr_coefs: vec![],
            #[cfg(feature = "experimental")]
            lagged_outer_prod_sum: nalgebra::DMatrix::zeros(MAX_LPC_ORDER, MAX_LPC_ORDER),
            #[cfg(feature = "experimental")]
            weights: vec![],
        }
    }

    #[allow(clippy::identity_op)] // false-alarm when OFFSET == 0
    fn fill_windowed_signal(
        &mut self,
        signal: &[i32],
        window: &[simd::Simd<f32, QLPC_WIN_SIMD_N>],
    ) {
        debug_assert!(window.len() * QLPC_WIN_SIMD_N >= signal.len());
        reuse!(CAST_BUFFER, |cast_buf: &mut SimdVec<
            i32,
            QLPC_WIN_SIMD_N,
        >| {
            cast_buf.reset_from_slice(signal);

            self.windowed_signal.reset_from_iter_simd(
                signal.len(),
                cast_buf.iter_simd().zip(window).map(|(s, w)| s.cast() * *w),
            );
        });
    }

    /// Performs weighted LPC via auto-correlation coefficients.
    #[allow(clippy::range_plus_one)]
    pub fn weighted_lpc_from_auto_corr<W>(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        weight: W,
    ) -> heapless::Vec<T, MAX_LPC_ORDER>
    where
        W: Weight,
    {
        let mut ret = heapless::Vec::new();
        if lpc_order == 0 {
            return ret;
        }
        ret.resize(lpc_order, T::zero())
            .expect("INTERNAL ERROR: lpc_order specified exceeded max.");
        self.corr_coefs.resize(lpc_order + 1, T::zero());
        self.corr_coefs.fill(T::zero());
        self.fill_windowed_signal(signal, get_window(window, signal.len()).as_ref_simd());

        weighted_auto_correlation(
            lpc_order + 1,
            self.windowed_signal.as_ref(),
            &mut self.corr_coefs,
            weight,
        );
        for &v in &self.corr_coefs {
            assert!(
                !(v.is_nan() || v.is_infinite()),
                "corr_coefs[_] = {v} must be normal or zero."
            );
        }
        symmetric_levinson_recursion::<T, MAX_LPC_ORDER>(
            &self.corr_coefs[0..lpc_order],
            &self.corr_coefs[1..lpc_order + 1],
            &mut ret,
        );
        for &v in &ret {
            assert!(!(v.is_nan() || v.is_infinite()));
        }
        ret
    }

    pub fn lpc_from_auto_corr(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
    ) -> heapless::Vec<T, MAX_LPC_ORDER> {
        self.weighted_lpc_from_auto_corr(signal, window, lpc_order, NoWeight)
    }

    /// Optimizes LPC with Mean-Absolute-Error criterion.
    #[cfg(feature = "experimental")]
    pub fn lpc_with_irls_mae(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        steps: usize,
    ) -> heapless::Vec<T, MAX_LPC_ORDER> {
        self.weights.clear();
        self.weights.resize(signal.len(), 1.0f32);
        let mut raw_errors = vec![0.0f32; signal.len()];
        let mut best_coefs = None;
        let mut best_error = f32::MAX;

        let normalizer = signal.iter().map(|x| x.abs()).max().unwrap() as f32;
        let weight_fn = |err: f32| (err.abs().max(1.0) / normalizer).max(0.01).powf(-1.2);

        for _t in 0..=steps {
            let coefs = self.weighted_lpc_with_direct_mse(
                signal,
                window,
                lpc_order,
                VecWeight(self.weights.clone()),
            );
            compute_raw_errors(signal, &coefs, &mut raw_errors);

            let sum_abs_err: f32 = raw_errors.iter().copied().map(f32::abs).sum::<f32>();
            if sum_abs_err < best_error {
                best_error = sum_abs_err;
                best_coefs = Some(coefs);
            }

            for (p, &err) in self.weights.iter_mut().zip(&raw_errors).skip(lpc_order) {
                *p = weight_fn(err);
            }
        }
        best_coefs.unwrap()
    }

    #[cfg(feature = "experimental")]
    fn weighted_lpc_with_direct_mse<W>(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        weight: W,
    ) -> heapless::Vec<T, MAX_LPC_ORDER>
    where
        W: Weight,
    {
        self.corr_coefs.resize(lpc_order + 1, T::zero());
        self.corr_coefs.fill(T::zero());

        self.fill_windowed_signal(signal, get_window(window, signal.len()).as_ref_simd());

        self.lagged_outer_prod_sum.fill(T::zero());
        self.lagged_outer_prod_sum
            .resize_mut(lpc_order, lpc_order, T::zero());

        weighted_auto_correlation_nosimd(
            lpc_order + 1,
            self.windowed_signal.as_ref(),
            &mut self.corr_coefs,
            &weight,
        );
        weighted_lagged_outer_prod_sum(
            lpc_order,
            &self.windowed_signal.as_ref()[0..self.windowed_signal.len() - 1],
            &mut self.lagged_outer_prod_sum,
            ShiftedWeight::<1, _>(weight),
        );

        let mut xy = nalgebra::DVector::<T>::from(self.corr_coefs[1..].to_vec());

        let mut regularizer = T::zero();
        while !T::solve_sym_mut(&self.lagged_outer_prod_sum, &mut xy) {
            let old_regularizer = regularizer;
            regularizer = T::one().max(regularizer + regularizer);
            for i in 0..lpc_order {
                self.lagged_outer_prod_sum[(i, i)] += regularizer - old_regularizer;
            }
        }

        let mut ret = heapless::Vec::new();
        ret.resize(lpc_order, T::zero())
            .expect("INTERNAL ERROR: lpc_order specified exceeded max.");
        for i in 0..lpc_order {
            ret[i] = xy[i];
        }
        ret
    }

    #[cfg(feature = "experimental")]
    fn lpc_with_direct_mse(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
    ) -> heapless::Vec<T, MAX_LPC_ORDER> {
        self.weighted_lpc_with_direct_mse(signal, window, lpc_order, NoWeight)
    }
}

reusable!(LPC_ESTIMATOR: LpcEstimator<f64> = LpcEstimator::new());

/// Estimates LPC coefficients with auto-correlation method.
#[allow(clippy::module_name_repetitions)]
pub fn lpc_from_autocorr(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    LPC_ESTIMATOR.with(|estimator| {
        estimator
            .borrow_mut()
            .lpc_from_auto_corr(signal, window, lpc_order)
    })
}

/// Estimates LPC coefficients with direct MSE method.
#[allow(clippy::module_name_repetitions)]
#[cfg(feature = "experimental")]
pub fn lpc_with_direct_mse(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    LPC_ESTIMATOR.with(|estimator| {
        estimator
            .borrow_mut()
            .lpc_with_direct_mse(signal, window, lpc_order)
    })
}

#[allow(clippy::module_name_repetitions)]
#[cfg(not(feature = "experimental"))]
pub fn lpc_with_direct_mse(
    _signal: &[i32],
    _window: &Window,
    _lpc_order: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    unimplemented!("not built with \"experimental\" feature flag.")
}

/// Estimates LPC coefficients with IRLS-MAE method.
#[allow(clippy::module_name_repetitions)]
#[cfg(feature = "experimental")]
pub fn lpc_with_irls_mae(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
    steps: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    LPC_ESTIMATOR.with(|estimator| {
        estimator
            .borrow_mut()
            .lpc_with_irls_mae(signal, window, lpc_order, steps)
    })
}

#[allow(clippy::module_name_repetitions)]
#[cfg(not(feature = "experimental"))]
pub fn lpc_with_irls_mae(
    _signal: &[i32],
    _window: &Window,
    _lpc_order: usize,
    _steps: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    unimplemented!("not built with \"experimental\" feature flag.")
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::nursery, clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::assert_close;
    use crate::assert_finite;
    use crate::sigen;
    use crate::sigen::Signal;
    use crate::test_helper;

    use rstest::rstest;
    use std::f32::consts::PI;

    #[test]
    fn auto_correlation_computation() {
        let mut signal = [0f32; 128];
        for t in 0..signal.len() {
            signal[t] = (t as f32 / 32.0 * 2.0 * PI).sin() * 1024.0;
        }
        let mut corr = [0f32; 64];
        auto_correlation(32, &signal, &mut corr);

        let mut max_corr: f32 = 0.0;
        let mut min_corr: f32 = 0.0;
        let mut argmax_corr: usize = 0;
        let mut argmin_corr: usize = 0;
        for t in 0..32 {
            if corr[t] > max_corr {
                argmax_corr = t;
                max_corr = corr[t];
            }
            if corr[t] < min_corr {
                argmin_corr = t;
                min_corr = corr[t];
            }
        }
        assert_eq!(argmax_corr, 0);
        assert_eq!(argmin_corr, 16);
    }

    #[test]
    fn auto_correlation_computation_with_known_samples() {
        let signal: [f32; 64] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
            1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
            -1.0, // warmup ends
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let mut corr = [0f64; 33];
        weighted_auto_correlation(33, &signal, &mut corr, NoWeight);

        assert_eq!(corr[0], 24.0);
        assert_eq!(corr[1], -4.0);
        assert_eq!(corr[2], 2.0);
        assert_eq!(corr[32], 0.0);
    }

    #[test]
    fn symmetric_levinson_algorithm() {
        let coefs: [f32; 4] = [1.0, 0.5, 0.0, 0.25];
        let ys: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
        let expect_xs: [f32; 4] = [8.0, -10.0, 10.0, -8.0];

        let mut xs: [f32; 4] = [0.0; 4];

        symmetric_levinson_recursion::<f32, 8>(&coefs, &ys, &mut xs);
        eprintln!("Found solution = {xs:?}");
        assert_eq!(xs, expect_xs);

        let coefs: [f32; 5] = [1.0, -0.5, -1.0, -0.5, 0.5];
        let ys: [f32; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
        let expect_xs: [f32; 5] = [0.80833, -0.26458, -0.36667, -0.45208, -1.06667];

        let mut xs: [f32; 5] = [0.0; 5];

        symmetric_levinson_recursion::<f32, MAX_LPC_ORDER>(&coefs, &ys, &mut xs);
        eprintln!("Found solution = {xs:?}");
        for (x, expected_x) in xs.iter().zip(expect_xs.iter()) {
            assert_close!(x, expected_x);
        }
    }

    #[test]
    fn shift_finder() {
        // max abs is [0.01]
        // shifting this 9 bits left yields [10000000], and it hits the MSB of
        // 8-bit integer representation.
        assert_eq!(find_shift(&[0.25, 0.125, 0.000001, 0.0], 8), 9);
    }

    #[test]
    fn parameter_quantizer() {
        let qp = quantize_parameters(&[0.0, 0.5, 0.1], 4);
        eprintln!("{qp:?}");
        assert_eq!(qp.coefs(), vec![0i16, 7i16, 2i16]);

        let qp = quantize_parameters(&[1.0, -0.5, 0.5], 2);
        eprintln!("{qp:?}");
        assert_eq!(qp.coefs(), vec![1, -1, 1]);
        assert_eq!(qp.dequantized(), vec![0.5, -0.5, 0.5]);
    }

    #[test]
    fn qlpc_auto_truncation() {
        let coefs = [1.0, 0.5, 0.0, 0.0];
        let qp = quantize_parameters(&coefs, 8);
        assert_eq!(qp.order(), 2);
    }

    #[rstest]
    fn qlpc_recovery(#[values(2, 12, 24)] lpc_order: usize) {
        let coef_prec: usize = 15;
        let signal = sigen::Sine::new(32, 0.8)
            .noise_with_seed(123, 0.01)
            .to_vec_quantized(16, 1024);

        let lpc_coefs = LPC_ESTIMATOR.with(|estimator| {
            estimator.borrow_mut().lpc_from_auto_corr(
                &signal,
                &Window::Tukey { alpha: 0.1 },
                lpc_order,
            )
        });
        assert_finite!(lpc_coefs);
        let mut errors = vec![0i32; signal.len()];
        eprintln!("{signal:?}");
        let qlpc = quantize_parameters(&lpc_coefs[0..lpc_order], coef_prec);

        // QLPC coefs can be shorter than the specified order because it truncates tail
        // zeroes.
        assert!(qlpc.coefs().len() <= lpc_order);
        eprintln!("Raw coefs: {:?}", &lpc_coefs[0..lpc_order]);
        eprintln!("QLPC params: {:?}", &qlpc);
        compute_error(&qlpc, &signal, &mut errors);

        let mut signal_energy = 0.0f64;
        let mut error_energy = 0.0f64;
        for t in lpc_order..signal.len() {
            signal_energy += signal[t] as f64 * signal[t] as f64;
            error_energy += errors[t] as f64 * errors[t] as f64;
        }
        // expect some prediction efficiency.
        eprintln!(
            "Prediction error ratio = {} dB",
            10.0 * (signal_energy / error_energy).log10()
        );
        assert!(error_energy < signal_energy);

        eprintln!("Recover with coefs: {:?}", qlpc.coefs());
        for t in lpc_order..signal.len() {
            let mut pred: i64 = 0;
            for (tau, ref_qcoef) in qlpc.coefs().iter().enumerate() {
                pred += i64::from(signal[t - tau - 1]) * i64::from(*ref_qcoef)
            }
            pred >>= qlpc.shift();
            assert_eq!(errors[t] + (pred as i32), signal[t], "Failed at t={t}");
        }
    }

    #[test]
    fn lpc_with_pure_dc() {
        const LPC_ORDER: usize = 1; // Overdetermined when order > 1
        let signal = [12345, 12345, 12345, 12345, 12345, 12345, 12345];
        let signal_float = signal.iter().map(|&x| x as f32).collect::<Vec<f32>>();

        let mut corr = [0f32; LPC_ORDER + 1];
        auto_correlation(LPC_ORDER + 1, &signal_float, &mut corr);

        let mut coefs = [0f32; LPC_ORDER];
        symmetric_levinson_recursion::<f32, LPC_ORDER>(
            &corr[0..LPC_ORDER],
            &corr[1..LPC_ORDER + 1],
            &mut coefs,
        );
        assert_close!(coefs[0], 1.0f32);

        let qlpc = quantize_parameters(&coefs[0..LPC_ORDER], 15);
        eprintln!("{qlpc:?}");
        let mut errors = vec![0i32; signal.len()];
        compute_error(&qlpc, &signal, &mut errors);
        for t in 0..errors.len() {
            assert!(errors[t] < 2);
        }
    }

    #[test]
    fn lpc_with_known_coefs() {
        // [1, -1, 0.5]
        let lpc_order: usize = 3;
        let signal = vec![
            0, -512, 0, 512, 256, -256, -256, 128, 256, 0, -192, -64, 128, 96, -64, -96, 16, 80,
            16, -56, -32, 32, 36, -12,
        ];

        let coefs = LPC_ESTIMATOR.with(|estimator| {
            estimator.borrow_mut().lpc_from_auto_corr(
                &signal,
                &Window::Tukey { alpha: 0.25 },
                lpc_order,
            )
        });
        eprintln!("{coefs:?}");
        // Actual auto-correlation function is not Toeplitz due to boundaries.
        assert!(coefs[0] > 0.0);
        assert!(coefs[1] < 0.0);
        assert!(coefs[2] > 0.0);
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn lpc_with_known_coefs_dmse() {
        let lpc_order: usize = 3;
        let signal = vec![
            0, -512, 0, 512, 256, -256, -256, 128, 256, 0, -192, -64, 128, 96, -64, -96, 16, 80,
            16, -56, -32, 32, 36, -12,
        ];
        let coefs = LPC_ESTIMATOR.with(|estimator| {
            estimator
                .borrow_mut()
                .lpc_with_direct_mse(&signal, &Window::Rectangle, lpc_order)
        });
        eprintln!("{coefs:?}");
        // Direct MSE can recover the oracle more accurately
        assert!(0.9 < coefs[0] && coefs[0] < 1.1);
        assert!(-1.1 < coefs[1] && coefs[1] < -0.9);
        assert!(0.4 < coefs[2] && coefs[2] < 0.6);
    }

    #[test]
    fn tukey_window() {
        // reference computed with scipy as `scipy.signal.windows.tukey(32, 0.3)`.
        let reference = [
            0., 0.1098376, 0.39109322, 0.720197, 0.95255725, 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.95255725, 0.720197, 0.39109322,
            0.1098376, 0.,
        ];
        let win = Window::Tukey { alpha: 0.3 };
        let win_vec = get_window(&win, reference.len());
        let win_vec = win_vec.as_ref().as_ref();
        for (t, &expected_w) in reference.iter().enumerate() {
            assert_close!(win_vec[t], expected_w);
        }
    }

    #[test]
    fn tukey_window_range() {
        for alpha in &[0.0, 0.3, 0.5, 0.8, 1.0] {
            let win = Window::Tukey { alpha: *alpha };
            let win_vec = get_window(&win, 4096);
            let win_vec = win_vec.as_ref().as_ref();
            for (t, v) in win_vec.iter().enumerate() {
                assert!(
                    v.is_normal() || *v == 0.0,
                    "window({alpha})[{t}] = {v} must be normal or zero."
                );
            }
        }
    }

    /// Computes squared sum (energy) of the slice.
    fn compute_energy<T>(signal: &[T]) -> f64
    where
        f64: From<T>,
        T: Copy,
    {
        let mut ret: f64 = 0.0;
        for v in signal.iter() {
            ret += f64::from(*v) * f64::from(*v);
        }
        ret
    }

    #[test]
    fn qlpc_with_test_signal() {
        let mut signal = test_helper::test_signal("sus109", 0);
        signal.truncate(4096);
        let lpc_order = 8;
        let coef_prec = 12;
        let signal_energy = compute_energy(&signal[lpc_order..]);

        let lpc_coefs = LPC_ESTIMATOR.with(|estimator| {
            estimator.borrow_mut().lpc_from_auto_corr(
                &signal,
                &Window::Tukey { alpha: 0.1 },
                lpc_order,
            )
        });
        let mut raw_errors = vec![0.0f32; signal.len()];
        compute_raw_errors(&signal, &lpc_coefs[0..lpc_order], &mut raw_errors);

        let raw_error_energy = compute_energy(&raw_errors[lpc_order..]);
        eprintln!(
            "Raw prediction error ratio = {} dB",
            10.0 * (signal_energy / raw_error_energy).log10()
        );

        let mut errors = vec![0i32; signal.len()];
        let qlpc = quantize_parameters(&lpc_coefs[0..lpc_order], coef_prec);
        assert_eq!(qlpc.coefs().len(), lpc_order);
        eprintln!("Raw coefs: {:?}", &lpc_coefs[0..lpc_order]);
        compute_error(&qlpc, &signal, &mut errors);

        let error_energy = compute_energy(&errors[lpc_order..]);
        // expect some prediction efficiency.
        eprintln!(
            "Prediction error ratio = {} dB",
            10.0 * (signal_energy / error_energy).log10()
        );
        assert!(error_energy < signal_energy);
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn if_direct_mse_is_better_than_autocorr() {
        let lpc_order: usize = 24;
        let mut signal = test_helper::test_signal("sus109", 0);

        // Difference is more visible when window size is small.
        signal.truncate(128);

        let window_autocorr = Window::Tukey { alpha: 0.1 };
        let window_direct_mse = Window::Rectangle;
        let mut errors_autocorr = vec![0f32; signal.len()];
        let mut errors_direct_mse = vec![0f32; signal.len()];

        let coefs_autocorr = LPC_ESTIMATOR.with(|estimator| {
            estimator
                .borrow_mut()
                .lpc_from_auto_corr(&signal, &window_autocorr, lpc_order)
        });
        let coefs_direct_mse = LPC_ESTIMATOR.with(|estimator| {
            estimator
                .borrow_mut()
                .lpc_with_direct_mse(&signal, &window_direct_mse, lpc_order)
        });

        compute_raw_errors(&signal, &coefs_autocorr, &mut errors_autocorr);
        compute_raw_errors(&signal, &coefs_direct_mse, &mut errors_direct_mse);

        let signal_energy = compute_energy(&signal);
        let error_energy_autocorr = compute_energy(&errors_autocorr[lpc_order..]);
        let error_energy_direct_mse = compute_energy(&errors_direct_mse[lpc_order..]);

        let snr_autocorr = 10.0 * (signal_energy / error_energy_autocorr).log10();
        let snr_direct_mse = 10.0 * (signal_energy / error_energy_direct_mse).log10();

        eprintln!("SNR of auto-correlation method = {snr_autocorr} dB");
        eprintln!("coefs_autocorr = {coefs_autocorr:?}");
        eprintln!("SNR of direct MSE method = {snr_direct_mse} dB");
        eprintln!("coefs_direct_mse = {coefs_direct_mse:?}");
        assert!(snr_autocorr < snr_direct_mse);
    }

    #[test]
    #[allow(clippy::identity_op, clippy::neg_multiply)]
    #[cfg(feature = "experimental")]
    fn lagged_outer_prod_sum_computation() {
        let signal = vec![4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0];
        let mut result = nalgebra::DMatrix::zeros(2, 2);
        lagged_outer_prod_sum::<f64>(2, &signal, &mut result);
        eprintln!("{result:?}");
        assert_eq!(
            result[(0, 0)],
            (-4 * -4 + 3 * 3 + -3 * -3 + 2 * 2 + -2 * -2 + 1 * 1 + -1 * -1) as f64
        );
        assert_eq!(
            result[(0, 1)],
            (4 * -4 + -4 * 3 + 3 * -3 + -3 * 2 + 2 * -2 + -2 * 1 + 1 * -1) as f64
        );
        assert_eq!(
            result[(1, 1)],
            (4 * 4 + -4 * -4 + 3 * 3 + -3 * -3 + 2 * 2 + -2 * -2 + 1 * 1) as f64
        );
        assert_eq!(result[(1, 0)], result[(0, 1)])
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn solve_mut_sym() {
        let signal: Vec<f32> = sigen::Sine::new(32, 0.8)
            .noise(0.01)
            .to_vec_quantized(16, 1024)
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let order = 12;
        let mut autocorr = vec![0.0f64; order + 1];

        auto_correlation(order + 1, &signal, &mut autocorr);
        let mut covar = nalgebra::DMatrix::zeros(order, order);
        lagged_outer_prod_sum(order, &signal, &mut covar);

        let mut x =
            nalgebra::DVector::<f64>::from(autocorr.iter().copied().skip(1).collect::<Vec<_>>());
        f64::solve_sym_mut(&covar, &mut x);

        eprintln!("x = {x:?}");
        let y = covar * x;

        for (dim, (y, y_expected)) in y.iter().zip(autocorr.iter().skip(1)).enumerate() {
            eprintln!("{y} == {y_expected} @ {dim}");
            assert_close!(y, y_expected);
        }
    }

    #[test]
    #[cfg(feature = "simd-nightly")]
    fn parity_of_auto_correlation_functions_for_simd_and_nosimd() {
        let signal: Vec<f32> = sigen::Sine::new(32, 0.8)
            .noise(0.01)
            .to_vec_quantized(16, 1024)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let order: usize = 25;

        let mut dest_simd = vec![0.0f64; MAX_LPC_ORDER + 1];
        let mut dest_nosimd = vec![0.0f64; MAX_LPC_ORDER + 1];

        weighted_auto_correlation_simd(order, &signal, &mut dest_simd, NoWeight);
        weighted_auto_correlation_nosimd(order, &signal, &mut dest_nosimd, NoWeight);

        for (d, (x_simd, x_nosimd)) in dest_simd.iter().zip(dest_nosimd.iter()).enumerate() {
            eprintln!("dim={d}, x_simd={x_simd}, x_nosimd={x_nosimd}, {order}");
            assert_close!(x_simd, x_nosimd);
        }
    }

    #[test]
    fn overflow_patterns() {
        let signal = vec![
            127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
            127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 29, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let lpc_order = 15;
        let quant_precision = 13;
        let lpc_coefs = lpc_from_autocorr(&signal, &Window::Rectangle, lpc_order);
        let qlpc = quantize_parameters(&lpc_coefs[0..lpc_order], quant_precision);

        let mut errors = vec![0i32; signal.len()];
        compute_error(&qlpc, &signal, &mut errors);
    }

    #[test]
    fn order_zero_lpc() {
        let signal = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];
        let lpc_order = 0;
        let quant_precision = 13;
        let lpc_coefs = lpc_from_autocorr(&signal, &Window::Rectangle, lpc_order);
        let qlpc = quantize_parameters(&lpc_coefs[0..lpc_order], quant_precision);

        let mut errors = vec![0i32; signal.len()];
        compute_error(&qlpc, &signal, &mut errors);
        assert_eq!(&errors, &vec![0i32; signal.len()]);
    }

    #[rstest]
    #[cfg(feature = "experimental")]
    fn comparing_mse_vs_mae(#[values(256, 512, 1024, 2048, 4096)] block_size: usize) {
        let lpc_order: usize = 16;
        let mut signal = test_helper::test_signal("sus109", 0);

        signal.truncate(block_size);

        let mut errors_mae = vec![0f32; signal.len()];
        let mut errors_mse = vec![0f32; signal.len()];

        let coefs_mse = LPC_ESTIMATOR.with(|estimator| {
            estimator
                .borrow_mut()
                .lpc_with_direct_mse(&signal, &Window::Rectangle, lpc_order)
        });

        let coefs_mae = LPC_ESTIMATOR.with(|estimator| {
            estimator
                .borrow_mut()
                .lpc_with_irls_mae(&signal, &Window::Rectangle, lpc_order, 4)
        });

        compute_raw_errors(&signal, &coefs_mse, &mut errors_mse);
        compute_raw_errors(&signal, &coefs_mae, &mut errors_mae);

        let mae_mse: f32 = errors_mse
            .iter()
            .map(|&x| x.abs() / signal.len() as f32)
            .sum();
        let mae_mae: f32 = errors_mae
            .iter()
            .map(|&x| x.abs() / signal.len() as f32)
            .sum();

        eprintln!("MAE of MSE-estimated parameters: {mae_mse}");
        eprintln!("MAE of MAE-estimated parameters: {mae_mae}");
        assert!(mae_mse >= mae_mae);
    }
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;
    use crate::sigen;
    use crate::sigen::Signal;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    #[allow(clippy::semicolon_if_nothing_returned)] // for blackboxing the return value
    fn tukey_window_zero(b: &mut Bencher) {
        let window_cfg = Window::Tukey { alpha: 0.1 };
        let size = 4096usize;
        let window = get_window(&window_cfg, size);
        let mut lpc_estimator = LpcEstimator::<f64>::new();
        let signal = [0i32; 4096];

        lpc_estimator.fill_windowed_signal(&signal, window.as_ref_simd());
        b.iter(|| {
            lpc_estimator.fill_windowed_signal(black_box(&signal), black_box(window.as_ref_simd()))
        });
    }

    #[bench]
    fn auto_corr_order14_zero(b: &mut Bencher) {
        let signal = [0.0f32; 4096];
        let mut dest = [0.0f32; 14];
        b.iter(|| {
            auto_correlation(black_box(14usize), black_box(&signal), black_box(&mut dest));
        });
    }

    #[bench]
    fn levinson_recursion(b: &mut Bencher) {
        let bps = 16;
        let lpc_order = 14;
        let block_size = 4096;
        let signal: Vec<_> = sigen::Noise::new(0.6)
            .to_vec_quantized(bps, block_size)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let mut corr_coefs = vec![0.0f64; lpc_order + 1];
        let mut lpc_coefs = vec![0.0f64; lpc_order];
        auto_correlation(lpc_order + 1, &signal, &mut corr_coefs);

        b.iter(|| {
            symmetric_levinson_recursion::<f64, 24>(
                black_box(&corr_coefs[..lpc_order]),
                black_box(&corr_coefs[1..]),
                black_box(&mut lpc_coefs),
            );
        });
    }

    #[bench]
    fn quantized_parameter_error_dc(b: &mut Bencher) {
        let signal = [10000i32; 4096];
        let mut errors = [0i32; 4096];
        let qp = quantize_parameters(&[1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0], 12);
        b.iter(|| compute_error(black_box(&qp), black_box(&signal), black_box(&mut errors)));
    }
}
