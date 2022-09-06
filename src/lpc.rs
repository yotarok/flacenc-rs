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

use std::simd::SimdInt;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use serde::Deserialize;
use serde::Serialize;

use super::constant::MAX_LPC_ORDER;
use super::constant::QLPC_MAX_SHIFT;
use super::constant::QLPC_MIN_SHIFT;

/// Analysis window descriptor.
///
/// This enum is `Serializable` and `Deserializable` because this will be
/// directly used in config structs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Window {
    Rectangle,
    Tukey { alpha: f32 },
}

impl Eq for Window {}

impl PartialOrd for Window {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(format!("{:?}", self).cmp(&format!("{:?}", other)))
    }
}

impl Ord for Window {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .expect("INTERNAL ERROR: This should not happen.")
    }
}

impl Window {
    #[inline]
    pub fn weights(&self, len: usize) -> Vec<f32> {
        match *self {
            Self::Rectangle => vec![1.0f32; len],
            Self::Tukey { alpha } => {
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
}

impl Default for Window {
    fn default() -> Self {
        Window::Tukey { alpha: 0.1 }
    }
}

type WindowMap = BTreeMap<(usize, Window), Rc<[f32]>>;
thread_local! {
    static WINDOW_CACHE: RefCell<WindowMap> = RefCell::new(BTreeMap::new());
}

fn get_window(window: &Window, size: usize) -> Rc<[f32]> {
    let key = (size, window.clone());
    WINDOW_CACHE.with(|caches| {
        if caches.borrow().get(&key).is_none() {
            caches
                .borrow_mut()
                .insert(key.clone(), Rc::from(window.weights(size)));
        }
        Rc::clone(
            caches
                .borrow()
                .get(&key)
                .expect("INTERNAL ERROR: window cache was not properly populated"),
        )
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

/// Dequantizes QLPC parameter. (Only used for debug/ test currently.)
#[inline]
fn dequantize_parameter(coef: i16, shift: i8) -> f32 {
    let scalefac = 2.0f32.powi(-i32::from(shift));
    f32::from(coef) * scalefac
}

const QLPC_SIMD_LANES: usize = 16usize;
const MAX_COEF_VECTORS: usize = (MAX_LPC_ORDER + (QLPC_SIMD_LANES - 1)) / QLPC_SIMD_LANES;
#[allow(dead_code)]

/// Shifts elements in a vector of `T` represented as a slice of `Simd<T, N>`.
#[inline]
fn shift_lanes_right<T, const N: usize>(val: T, vecs: &mut [std::simd::Simd<T, N>])
where
    T: std::simd::SimdElement,
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
{
    let mut carry = val;
    for v in vecs {
        let mut shifted = v.rotate_lanes_right::<1>();
        (shifted[0], carry) = (carry, shifted[0]);
        *v = shifted;
    }
}

/// Quantized LPC coefficients.
#[derive(Clone, Debug)]
pub struct QuantizedParameters {
    coefs: heapless::Vec<std::simd::i32x16, MAX_COEF_VECTORS>,
    order: usize,
    shift: i8,
    precision: usize,
}

impl QuantizedParameters {
    /// Constructs `QuantizedParameters` from the parameters and precision.
    pub fn with_coefs(coefs: &[f32], precision: usize) -> Self {
        let shift = find_shift(coefs, precision);
        let mut q_coefs = [0i32; MAX_LPC_ORDER];

        for (n, coef) in coefs.iter().enumerate() {
            // This clamp op is mainly for safety, but actually required
            // because the shift-width estimation `find_shift` used here is not
            // perfect, and quantization may yields "2^(p-1)" quantized value
            // for precision "p" configuration, that is larger than a maximum
            // p-bits signed integer "2^(p-1) - 1".
            q_coefs[n] = std::cmp::min(
                std::cmp::max(
                    i32::from(quantize_parameter(*coef, shift)),
                    -(1 << (precision - 1)),
                ),
                (1 << (precision - 1)) - 1,
            );
        }

        let tail_zeros = q_coefs
            .rsplitn(2, |&x| x != 0)
            .next()
            .map_or(0, <[i32]>::len);
        let order = q_coefs.len() - tail_zeros;

        let mut coefs_v = heapless::Vec::new();
        for arr in q_coefs.chunks(QLPC_SIMD_LANES) {
            let mut v = std::simd::i32x16::splat(0);
            v.as_mut_array()[0..arr.len()].copy_from_slice(arr);
            coefs_v
                .push(v)
                .expect("INTERNAL ERROR: Length of coefs exceeded QLPC_SIMD_LANES.");
        }

        Self {
            coefs: coefs_v,
            order,
            shift,
            precision,
        }
    }

    /// Returns the order of LPC specified by this parameter.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Compute error signal from `QuantizedParameters`.
    ///
    /// # Panics
    ///
    /// This function panics if `errors.len()` is smaller than `signal.len()`.
    pub fn compute_error(&self, signal: &[i32], errors: &mut [i32]) {
        assert!(errors.len() >= signal.len());
        for p in errors.iter_mut().take(self.order()) {
            *p = 0;
        }
        let mut window_h = heapless::Vec::<std::simd::i32x16, MAX_COEF_VECTORS>::new();
        let mut window_l = heapless::Vec::<std::simd::i32x16, MAX_COEF_VECTORS>::new();
        // TODO: Make these const again
        let low_word_mask: std::simd::i32x16 = std::simd::i32x16::splat(0x0000_FFFFi32);
        let low_word_denom: std::simd::i32x16 = std::simd::i32x16::splat(0x0001_0000i32);

        for i in 0..MAX_COEF_VECTORS {
            let tau: isize = (self.order() as isize - 1) - (i * QLPC_SIMD_LANES) as isize;
            if tau < 0 {
                break;
            }
            let mut v = std::simd::i32x16::splat(0);
            for j in 0..QLPC_SIMD_LANES {
                let j = j as isize;
                if tau - j < 0 {
                    break;
                }
                v[j as usize] = signal[(tau - j) as usize];
            }

            window_l
                .push((v.abs() & low_word_mask) * v.signum())
                .expect("INTERNAL ERROR: Couldn't push to window_l");
            window_h
                .push(v.abs() / low_word_denom * v.signum())
                .expect("INTERNAL ERROR: Couldn't push to window_h");
        }

        for t in self.order()..signal.len() {
            let mut pred = 0i64;
            for j in 0..window_l.len() {
                pred += i64::from((self.coefs[j] * window_l[j]).reduce_sum());
                pred += i64::from((self.coefs[j] * window_h[j]).reduce_sum()) << 16;
            }

            let shifted: i32 = (pred >> self.shift) as i32;
            errors[t] = (signal[t] - shifted) as i32;

            // shift window
            shift_lanes_right(
                (signal[t].abs() & 0xFFFFi32) * signal[t].signum(),
                &mut window_l,
            );
            shift_lanes_right((signal[t].abs() >> 16) * signal[t].signum(), &mut window_h);
        }
    }

    /// Returns precision.
    pub const fn precision(&self) -> usize {
        self.precision
    }

    /// Returns the shift parameter.
    pub const fn shift(&self) -> i8 {
        self.shift
    }

    /// Returns an individual coefficient in quantized form.
    pub fn coef(&self, idx: usize) -> i16 {
        let q = idx / QLPC_SIMD_LANES;
        let r = idx % QLPC_SIMD_LANES;
        self.coefs[q][r] as i16
    }

    /// Returns `Vec` containing quantized coefficients.
    pub fn coefs(&self) -> Vec<i16> {
        (0..self.order()).map(|j| self.coef(j)).collect()
    }

    /// Returns `Vec` containing dequantized coefficients.
    #[allow(dead_code)]
    pub fn dequantized(&self) -> Vec<f32> {
        self.coefs()
            .iter()
            .map(|x| dequantize_parameter(*x, self.shift))
            .collect()
    }
}

/// Compute auto-correlation coefficients.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
pub fn auto_correlation(order: usize, signal: &[f32], dest: &mut [f32]) {
    weighted_auto_correlation(order, signal, dest, |_t| 1.0f32);
}

/// Compute delay sum.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
pub fn delay_sum(order: usize, signal: &[f32], dest: &mut nalgebra::DMatrix<f32>) {
    weighted_delay_sum(order, signal, dest, |_t| 1.0f32);
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
    assert!(dest.len() >= order);
    for p in dest.iter_mut() {
        *p = 0.0;
    }
    for t in (order - 1)..signal.len() {
        let w = weight_fn(t);
        for tau in 0..order {
            let v: f32 = (signal[t] * signal[t - tau]) as f32 * w;
            dest[tau] += v;
        }
    }
}

/// Compute weighted delay-sum statistics.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
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
/// Panics if `dest` or `coefs` is shorter than `ys`.
pub fn symmetric_levinson_recursion(coefs: &[f32], ys: &[f32], dest: &mut [f32]) {
    assert!(dest.len() >= ys.len());
    assert!(coefs.len() >= ys.len());

    for p in dest.iter_mut() {
        *p = 0.0;
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
        let alpha: f32 = 1.0 / (1.0 - error * error);
        let beta: f32 = -alpha * error;
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
    windowed_signal: Vec<f32>,
    /// Buffer for storing auto-correlation coefficients.
    corr_coefs: Vec<f32>,
    /// Buffer for delay-sum matrix and it's inverse. (not used in auto-correlation mode.)
    delay_sum: nalgebra::DMatrix<f32>,
    /// Weights for IRLS.
    weights: Vec<f32>,
}

impl LpcEstimator {
    pub fn new() -> Self {
        Self {
            windowed_signal: vec![],
            corr_coefs: vec![],
            delay_sum: nalgebra::DMatrix::zeros(MAX_LPC_ORDER, MAX_LPC_ORDER),
            weights: vec![],
        }
    }

    fn fill_windowed_signal(&mut self, signal: &[i32], window: &[f32]) {
        self.windowed_signal.clear();
        for (t, &v) in signal.iter().enumerate() {
            self.windowed_signal.push(v as f32 * window[t]);
        }
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
        ret.resize(lpc_order, 0.0)
            .expect("INTERNAL ERROR: lpc_order specified exceeded max.");
        self.corr_coefs.resize(lpc_order + 1, 0.0);
        self.corr_coefs.fill(0f32);
        self.fill_windowed_signal(signal, &get_window(window, signal.len()));

        weighted_auto_correlation(
            lpc_order + 1,
            &self.windowed_signal,
            &mut self.corr_coefs,
            weight_fn,
        );
        symmetric_levinson_recursion(
            &self.corr_coefs[0..lpc_order],
            &self.corr_coefs[1..lpc_order + 1],
            &mut ret,
        );
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
            let coefs = self.weighted_lpc_with_direct_mse(signal, window, lpc_order, |t| ws[t]);
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

        self.fill_windowed_signal(signal, &get_window(window, signal.len()));

        self.delay_sum.fill(0.0f32);
        self.delay_sum.resize_mut(lpc_order, lpc_order, 0.0f32);
        weighted_auto_correlation(
            lpc_order + 1,
            &self.windowed_signal,
            &mut self.corr_coefs,
            &weight_fn,
        );
        weighted_delay_sum(
            lpc_order,
            &self.windowed_signal[..signal.len() - 1],
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

    fn lpc_with_direct_mse(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
    ) -> heapless::Vec<f32, MAX_LPC_ORDER> {
        self.weighted_lpc_with_direct_mse(signal, window, lpc_order, |_t| 1.0f32)
    }
}

thread_local! {
    /// Global (thread-local) working buffer for LPC estimation.
    static LPC_ESTIMATOR: RefCell<LpcEstimator> = RefCell::new(LpcEstimator::new());
}

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

/// Estimates LPC coefficients with IRLS-MAE method.
#[allow(clippy::module_name_repetitions)]
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

#[cfg(test)]
#[allow(clippy::pedantic, clippy::nursery, clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::assert_close;
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
        eprintln!("Found solution = {:?}", xs);
        assert_eq!(xs, expect_xs);

        let coefs: [f32; 5] = [1.0, -0.5, -1.0, -0.5, 0.5];
        let ys: [f32; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
        let expect_xs: [f32; 5] = [0.80833, -0.26458, -0.36667, -0.45208, -1.06667];

        let mut xs: [f32; 5] = [0.0; 5];

        symmetric_levinson_recursion(&coefs, &ys, &mut xs);
        eprintln!("Found solution = {:?}", xs);
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
        let qp = QuantizedParameters::with_coefs(&[0.0, 0.5, 0.1], 4);
        eprintln!("{:?}", qp);
        assert_eq!(qp.coefs(), vec![0i16, 7i16, 2i16]);

        let qp = QuantizedParameters::with_coefs(&[1.0, -0.5, 0.5], 2);
        eprintln!("{:?}", qp);
        assert_eq!(qp.coefs(), vec![1, -1, 1]);
        assert_eq!(qp.dequantized(), vec![0.5, -0.5, 0.5]);
    }

    #[test]
    fn qlpc_auto_truncation() {
        let coefs = [1.0, 0.5, 0.0, 0.0];
        let qp = QuantizedParameters::with_coefs(&coefs, 8);
        assert_eq!(qp.order(), 2);
    }

    #[rstest]
    fn qlpc_recovery(#[values(2, 12, 24)] lpc_order: usize) {
        let coef_prec: usize = 12;
        let signal = test_helper::sinusoid_plus_noise(1024, 32, 30000.0, 128);

        let lpc_coefs = LPC_ESTIMATOR.with(|estimator| {
            estimator.borrow_mut().lpc_from_auto_corr(
                &signal,
                &Window::Tukey { alpha: 0.1 },
                lpc_order,
            )
        });
        let mut errors = vec![0i32; signal.len()];
        eprintln!("{:?}", signal);
        let qlpc = QuantizedParameters::with_coefs(&lpc_coefs[0..lpc_order], coef_prec);
        assert_eq!(qlpc.coefs().len(), lpc_order);
        eprintln!("Raw coefs: {:?}", &lpc_coefs[0..lpc_order]);
        qlpc.compute_error(&signal, &mut errors);

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
            assert_eq!(errors[t] + (pred as i32), signal[t], "Failed at t={}", t);
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

        let qlpc = QuantizedParameters::with_coefs(&coefs[0..LPC_ORDER], 15);
        let mut errors = vec![0i32; signal.len()];
        qlpc.compute_error(&signal, &mut errors);
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
        eprintln!("{:?}", coefs);
        // Actual auto-correlation function is not Toeplitz due to boundaries.
        assert!(coefs[0] > 0.0);
        assert!(coefs[1] < 0.0);
        assert!(coefs[2] > 0.0);

        let coefs = LPC_ESTIMATOR.with(|estimator| {
            estimator
                .borrow_mut()
                .lpc_with_direct_mse(&signal, &Window::Rectangle, lpc_order)
        });
        eprintln!("{:?}", coefs);
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
        for (t, &expected_w) in reference.iter().enumerate() {
            assert_close!(win_vec[t], expected_w);
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
        let qlpc = QuantizedParameters::with_coefs(&lpc_coefs[0..lpc_order], coef_prec);
        assert_eq!(qlpc.coefs().len(), lpc_order);
        eprintln!("Raw coefs: {:?}", &lpc_coefs[0..lpc_order]);
        qlpc.compute_error(&signal, &mut errors);

        let error_energy = compute_energy(&errors[lpc_order..]);
        // expect some prediction efficiency.
        eprintln!(
            "Prediction error ratio = {} dB",
            10.0 * (signal_energy / error_energy).log10()
        );
        assert!(error_energy < signal_energy);
    }

    #[test]
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

        eprintln!("SNR of auto-correlation method = {} dB", snr_autocorr);
        eprintln!("coefs_autocorr = {:?}", coefs_autocorr);
        eprintln!("SNR of direct MSE method = {} dB", snr_direct_mse);
        eprintln!("coefs_direct_mse = {:?}", coefs_direct_mse);
        assert!(snr_autocorr < snr_direct_mse);
    }

    #[test]
    #[allow(clippy::identity_op, clippy::neg_multiply)]
    fn delay_sum_computation() {
        let signal = vec![4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0];
        let mut result = nalgebra::DMatrix::zeros(2, 2);
        delay_sum(2, &signal, &mut result);
        eprintln!("{:?}", result);
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

    #[rstest]
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

        eprintln!("MAE of MSE-estimated parameters: {}", mae_mse);
        eprintln!("MAE of MAE-estimated parameters: {}", mae_mae);
        assert!(mae_mse >= mae_mae);
    }
}
