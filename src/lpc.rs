// Copyright 2022 Google LLC
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

use super::arrayutils::find_max_abs;
use super::arrayutils::unaligned_map_and_update;
use super::arrayutils::SimdVec;
use super::component::QuantizedParameters;
use super::config::Window;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::qlpc::MAX_SHIFT as QLPC_MAX_SHIFT;
use super::constant::qlpc::MIN_SHIFT as QLPC_MIN_SHIFT;
use super::repeat::repeat;

import_simd!(as simd);

/// Precomputes window function given the window config `win`.
#[inline]
pub fn window_weights(win: &Window, len: usize) -> Vec<f32> {
    match *win {
        Window::Rectangle => vec![1.0f32; len],
        Window::Tukey { alpha } if alpha == 0.0 => {
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
fn find_shift(coefs: &[f32], precision: usize) -> i8 {
    assert!(precision <= 15);
    assert!(!coefs.is_empty());
    let max_abs_coef: f32 = coefs.iter().map(|x| x.abs()).reduce(f32::max).unwrap();
    // location of MSB in binary representations of absolute values.
    let abs_log2: i16 = max_abs_coef.log2().ceil().max(f32::from(i16::MIN + 16)) as i16;
    let shift: i16 = (precision as i16 - 1) - abs_log2;
    shift.clamp(i16::from(QLPC_MIN_SHIFT), i16::from(QLPC_MAX_SHIFT)) as i8
}

/// Quantizes LPC parameter with the given shift parameter.
#[inline]
fn quantize_parameter(p: f32, shift: i8) -> i16 {
    let scalefac = 2.0f32.powi(i32::from(shift));
    (p * scalefac)
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

/// Creates [`QuantizedParameters`] by quantizing the given coefficients.
pub fn quantize_parameters(coefs: &[f32], precision: usize) -> QuantizedParameters {
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
pub fn auto_correlation(order: usize, signal: &[f32], dest: &mut [f32]) {
    weighted_auto_correlation(order, signal, dest, |_t| 1.0f32);
}

/// Compute delay sum.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "experimental")]
#[allow(dead_code)]
pub fn delay_sum(order: usize, signal: &[f32], dest: &mut nalgebra::DMatrix<f32>) {
    weighted_delay_sum(order, signal, dest, |_t| 1.0f32);
}

#[inline]
#[allow(clippy::needless_range_loop)] // for readability
#[allow(dead_code)] // not used in "fakesimd" but should still be compilable.
fn weighted_auto_correlation_simd<F, const N: usize>(
    order: usize,
    signal: &[f32],
    dest: &mut [f32],
    weight_fn: F,
) where
    F: Fn(usize) -> f32,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    assert!(dest.len() >= order);
    assert!(order <= N);

    let mut lagged = simd::Simd::<f32, N>::splat(0f32);
    let mut acc = simd::Simd::<f32, N>::splat(0f32);
    for tau in 0..(order - 1) {
        lagged[tau] = signal[order - 2 - tau];
    }
    for t in (order - 1)..signal.len() {
        let w = weight_fn(t);
        let y = signal[t];
        lagged = lagged.rotate_elements_right::<1>();
        lagged[0] = y;
        acc += simd::Simd::<f32, N>::splat(w * y) * lagged;
    }
    for (tau, mut_p) in dest.iter_mut().enumerate() {
        *mut_p = if tau < order { acc[tau] } else { 0.0 };
    }
}

/// Compute weighted auto-correlation coefficients.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
pub fn weighted_auto_correlation<F>(order: usize, signal: &[f32], dest: &mut [f32], weight_fn: F)
where
    F: Fn(usize) -> f32,
{
    #[cfg(feature = "simd-nightly")]
    {
        // The current implementation is inefficient with fakesimd when order
        // is low. So, here we still have a scalar version of it.
        if order <= 4 {
            weighted_auto_correlation_simd::<_, 4>(order, signal, dest, weight_fn);
        } else if order <= 8 {
            weighted_auto_correlation_simd::<_, 8>(order, signal, dest, weight_fn);
        } else if order <= 16 {
            weighted_auto_correlation_simd::<_, 16>(order, signal, dest, weight_fn);
        } else if order <= 32 {
            weighted_auto_correlation_simd::<_, 32>(order, signal, dest, weight_fn);
        } else {
            assert!(order == 33);
            // this is inefficient because there's only 1 element that doesn't fit
            // to Simd with LANE==32. However, this is so far okay as it is rather
            // rare to use order == 33 (i.e. lpc_order == 32).
            weighted_auto_correlation_simd::<_, 64>(order, signal, dest, weight_fn);
        }
    }
    #[cfg(not(feature = "simd-nightly"))]
    {
        assert!(dest.len() >= order);
        for p in &mut *dest {
            *p = 0.0;
        }
        for t in (order - 1)..signal.len() {
            let w = weight_fn(t);
            for tau in 0..order {
                let v: f32 = signal[t] * signal[t - tau] * w;
                dest[tau] += v;
            }
        }
    }
}

/// Compute weighted delay-sum statistics.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "experimental")]
pub fn weighted_delay_sum<F>(
    order: usize,
    signal: &[f32],
    dest: &mut nalgebra::DMatrix<f32>,
    weight_fn: F,
) where
    F: Fn(usize) -> f32,
{
    assert!(dest.ncols() >= order);
    assert!(dest.nrows() >= order);

    dest.fill(0.0f32);

    for t in (order - 1)..signal.len() {
        let w = weight_fn(t);
        for i in 0..order {
            for j in i..order {
                let v = signal[t - i] * signal[t - j] * w;
                dest[(i, j)] += v;
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
fn compute_raw_errors(signal: &[i32], lpc_coefs: &[f32], errors: &mut [f32]) {
    let lpc_order = lpc_coefs.len();
    for t in lpc_order..signal.len() {
        errors[t] = -signal[t] as f32;
        for j in 0..lpc_order {
            errors[t] += lpc_coefs[j] * signal[t - 1 - j] as f32;
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
pub fn symmetric_levinson_recursion(coefs: &[f32], ys: &[f32], dest: &mut [f32]) {
    assert!(dest.len() >= ys.len());
    assert!(coefs.len() >= ys.len());

    for p in &mut *dest {
        *p = 0.0;
    }

    // coefs[0] is energy of the signal, so must be non-negative.
    assert!(coefs[0] >= 0.0);
    if coefs[0] == 0.0 {
        let allzero = ys
            .iter()
            .chain(coefs.iter())
            .fold(true, |f, &v| f & (v == 0.0));
        assert!(
            allzero,
            "If signal is digital silence, all coefficients must be zero."
        );
        return;
    }

    let order = ys.len();
    let mut forward = vec![0f32; order];
    let mut forward_next = vec![0f32; order];

    forward[0] = 1.0 / coefs[0];
    dest[0] = ys[0] / coefs[0];

    for n in 1..order {
        let error: f32 = coefs[1..=n]
            .iter()
            .rev()
            .zip(forward.iter())
            .map(|(x, y)| x * y)
            .sum();
        let denom = error.mul_add(-error, 1.0);
        let (alpha, beta): (f32, f32) = if denom == 0.0 {
            // TODO: check if this is mathematically sound.  From the definition of
            //       levinson-recurssion, when error^2 == 1.0, we can only say
            //       alpha + beta = 1.0, or alpha - beta = 1.0 (depending on sign(error)).
            //       due to rank deficiency.
            (1.0, 0.0)
        } else {
            let a = 1.0 / denom;
            (a, -a * error)
        };
        for d in 0..=n {
            forward_next[d] = alpha.mul_add(forward[d], beta * forward[n - d]);
        }
        forward.copy_from_slice(&forward_next);

        let delta: f32 = coefs[1..=n]
            .iter()
            .rev()
            .zip(dest.iter())
            .map(|(x, y)| *x * *y)
            .sum();
        for d in 0..=n {
            dest[d] += (ys[n] - delta) * forward[n - d];
        }
    }
}

/// Working buffer for (unquantized) LPC estimation.
struct LpcEstimator {
    /// Buffer for storing windowed signal.
    windowed_signal: SimdVec<f32, QLPC_WIN_SIMD_N>,
    /// Buffer for storing auto-correlation coefficients.
    corr_coefs: Vec<f32>,
    /// Buffer for delay-sum matrix and it's inverse. (not used in auto-correlation mode.)
    #[cfg(feature = "experimental")]
    delay_sum: nalgebra::DMatrix<f32>,
    /// Weights for IRLS.
    #[cfg(feature = "experimental")]
    weights: Vec<f32>,
}

reusable!(CAST_BUFFER: SimdVec<i32, QLPC_WIN_SIMD_N> = SimdVec::new());

impl LpcEstimator {
    pub fn new() -> Self {
        Self {
            windowed_signal: SimdVec::new(),
            corr_coefs: vec![],
            #[cfg(feature = "experimental")]
            delay_sum: nalgebra::DMatrix::zeros(MAX_LPC_ORDER, MAX_LPC_ORDER),
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

    #[allow(clippy::range_plus_one)]
    pub fn weighted_lpc_from_auto_corr<F>(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        weight_fn: F,
    ) -> heapless::Vec<f32, MAX_LPC_ORDER>
    where
        F: Fn(usize) -> f32,
    {
        let mut ret = heapless::Vec::new();
        if lpc_order == 0 {
            return ret;
        }
        ret.resize(lpc_order, 0.0)
            .expect("INTERNAL ERROR: lpc_order specified exceeded max.");
        self.corr_coefs.resize(lpc_order + 1, 0.0);
        self.corr_coefs.fill(0f32);
        self.fill_windowed_signal(signal, get_window(window, signal.len()).as_ref_simd());

        weighted_auto_correlation(
            lpc_order + 1,
            self.windowed_signal.as_ref(),
            &mut self.corr_coefs,
            weight_fn,
        );
        for &v in &self.corr_coefs {
            assert!(
                v.is_normal() || v == 0.0,
                "corr_coefs[_] = {v} must be normal or zero."
            );
        }
        symmetric_levinson_recursion(
            &self.corr_coefs[0..lpc_order],
            &self.corr_coefs[1..lpc_order + 1],
            &mut ret,
        );
        for &v in &ret {
            assert!(v.is_normal() || v.is_subnormal() || v == 0.0);
        }
        ret
    }

    pub fn lpc_from_auto_corr(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
    ) -> heapless::Vec<f32, MAX_LPC_ORDER> {
        self.weighted_lpc_from_auto_corr(signal, window, lpc_order, |_t| 1.0f32)
    }

    /// Optimizes LPC with Mean-Absolute-Error criterion.
    #[cfg(feature = "experimental")]
    pub fn lpc_with_irls_mae(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        steps: usize,
    ) -> heapless::Vec<f32, MAX_LPC_ORDER> {
        self.weights.clear();
        self.weights.resize(signal.len(), 1.0f32);
        let mut raw_errors = vec![0.0f32; signal.len()];
        let mut best_coefs = None;
        let mut best_error = f32::MAX;

        let normalizer = signal.iter().map(|x| x.abs()).max().unwrap() as f32;
        let weight_fn = |err: f32| (err.abs().max(1.0) / normalizer).max(0.01).powf(-1.2);

        for _t in 0..=steps {
            let ws = self.weights.clone();
            let coefs = self.weighted_lpc_with_direct_mse(signal, window, lpc_order, |t| {
                ws.get(t).copied().unwrap_or(0.0)
            });
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
    fn weighted_lpc_with_direct_mse<F>(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        weight_fn: F,
    ) -> heapless::Vec<f32, MAX_LPC_ORDER>
    where
        F: Fn(usize) -> f32,
    {
        self.corr_coefs.resize(lpc_order + 1, 0.0);
        self.corr_coefs.fill(0f32);

        self.fill_windowed_signal(signal, get_window(window, signal.len()).as_ref_simd());

        self.delay_sum.fill(0.0f32);
        self.delay_sum.resize_mut(lpc_order, lpc_order, 0.0f32);
        weighted_auto_correlation(
            lpc_order + 1,
            self.windowed_signal.as_ref(),
            &mut self.corr_coefs,
            &weight_fn,
        );
        weighted_delay_sum(
            lpc_order,
            &self.windowed_signal.as_ref()[0..self.windowed_signal.len() - 1],
            &mut self.delay_sum,
            |t| weight_fn(t + 1),
        );

        let mut xy = nalgebra::DVector::<f32>::from(self.corr_coefs[1..].to_vec());

        let mut regularizer = f32::EPSILON;
        loop {
            if let Some(decompose) = self.delay_sum.clone().cholesky() {
                decompose.solve_mut(&mut xy);
                break;
            }
            for i in 0..lpc_order {
                self.delay_sum[(i, i)] += regularizer;
            }
            regularizer *= 10.0;
        }

        let mut ret = heapless::Vec::new();
        ret.resize(lpc_order, 0.0)
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
    ) -> heapless::Vec<f32, MAX_LPC_ORDER> {
        self.weighted_lpc_with_direct_mse(signal, window, lpc_order, |_t| 1.0f32)
    }
}

reusable!(LPC_ESTIMATOR: LpcEstimator = LpcEstimator::new());

/// Estimates LPC coefficients with auto-correlation method.
#[allow(clippy::module_name_repetitions)]
pub fn lpc_from_autocorr(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
) -> heapless::Vec<f32, MAX_LPC_ORDER> {
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
) -> heapless::Vec<f32, MAX_LPC_ORDER> {
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
) -> heapless::Vec<f32, MAX_LPC_ORDER> {
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
) -> heapless::Vec<f32, MAX_LPC_ORDER> {
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
) -> heapless::Vec<f32, MAX_LPC_ORDER> {
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
    fn symmetric_levinson_algorithm() {
        let coefs: [f32; 4] = [1.0, 0.5, 0.0, 0.25];
        let ys: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
        let expect_xs: [f32; 4] = [8.0, -10.0, 10.0, -8.0];

        let mut xs: [f32; 4] = [0.0; 4];

        symmetric_levinson_recursion(&coefs, &ys, &mut xs);
        eprintln!("Found solution = {xs:?}");
        assert_eq!(xs, expect_xs);

        let coefs: [f32; 5] = [1.0, -0.5, -1.0, -0.5, 0.5];
        let ys: [f32; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
        let expect_xs: [f32; 5] = [0.80833, -0.26458, -0.36667, -0.45208, -1.06667];

        let mut xs: [f32; 5] = [0.0; 5];

        symmetric_levinson_recursion(&coefs, &ys, &mut xs);
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
        let coef_prec: usize = 12;
        let signal = sigen::Sine::new(32, 0.8)
            .noise(0.01)
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
        symmetric_levinson_recursion(&corr[0..LPC_ORDER], &corr[1..LPC_ORDER + 1], &mut coefs);
        assert_close!(coefs[0], 1.0f32);

        let qlpc = quantize_parameters(&coefs[0..LPC_ORDER], 15);
        eprintln!("{:?}", qlpc);
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
    fn delay_sum_computation() {
        let signal = vec![4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0];
        let mut result = nalgebra::DMatrix::zeros(2, 2);
        delay_sum(2, &signal, &mut result);
        eprintln!("{result:?}");
        assert_eq!(
            result[(0, 0)],
            (-4 * -4 + 3 * 3 + -3 * -3 + 2 * 2 + -2 * -2 + 1 * 1 + -1 * -1) as f32
        );
        assert_eq!(
            result[(0, 1)],
            (4 * -4 + -4 * 3 + 3 * -3 + -3 * 2 + 2 * -2 + -2 * 1 + 1 * -1) as f32
        );
        assert_eq!(
            result[(1, 1)],
            (4 * 4 + -4 * -4 + 3 * 3 + -3 * -3 + 2 * 2 + -2 * -2 + 1 * 1) as f32
        );
        assert_eq!(result[(1, 0)], result[(0, 1)])
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

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    #[allow(clippy::semicolon_if_nothing_returned)] // for blackboxing the return value
    fn tukey_window_zero(b: &mut Bencher) {
        let window_cfg = Window::Tukey { alpha: 0.1 };
        let size = 4096usize;
        let window = get_window(&window_cfg, size);
        let mut lpc_estimator = LpcEstimator::new();
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
    fn quantized_parameter_error_dc(b: &mut Bencher) {
        let signal = [10000i32; 4096];
        let mut errors = [0i32; 4096];
        let qp = quantize_parameters(&[1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0], 12);
        b.iter(|| compute_error(black_box(&qp), black_box(&signal), black_box(&mut errors)));
    }
}
