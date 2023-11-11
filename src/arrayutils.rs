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

//! Module for array processing utility functions.

use seq_macro::seq;

use super::repeat::repeat;

import_simd!(as simd);

/// A wrapper for Vec of `simd::Simd` that can be viewed as a scalar slice.
#[derive(Debug, Default)]
pub struct SimdVec<T, const N: usize>
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    inner: Vec<simd::Simd<T, N>>,
    len: usize,
}

type IterSimd<'a, T, const N: usize> = std::slice::Iter<'a, simd::Simd<T, N>>;

impl<T, const N: usize> SimdVec<T, N>
where
    T: simd::SimdElement + Default,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    /// Constructs new `SimdVec` with length `0` and the default capacity.
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            len: 0usize,
        }
    }

    /// Constructs `SimdVec` from the given slice of scalars.
    #[inline]
    pub fn from_slice(data: &[T]) -> Self {
        let mut ret = Self::new();
        ret.reset_from_slice(data);
        ret
    }

    /// Returns the number of scalar elements.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of internal simd-vectors.
    #[inline]
    #[allow(dead_code)]
    pub fn simd_len(&self) -> usize {
        self.inner.len()
    }

    /// Resets the size and contents with values in the given slice.
    #[inline]
    pub fn reset_from_slice(&mut self, data: &[T]) {
        pack_into_simd_vec(data, &mut self.inner);
        self.len = data.len();
    }

    /// Returns an iterator for the simd-vectors.
    #[inline]
    pub fn iter_simd(&self) -> IterSimd<T, N> {
        self.inner.iter()
    }

    /// Resizes the innew storage so it can contain `new_len` scalars.
    ///
    /// If resizing needs to add a new simd-vector to the internal storage,
    /// `value` is used as an initial value.
    #[inline]
    #[allow(dead_code)]
    pub fn resize(&mut self, new_len: usize, value: simd::Simd<T, N>) {
        let len_v = (new_len + N - 1) / N;
        self.inner.resize(len_v, value);
        self.len = new_len;
    }

    /// Resest the size and values from the iterator of simd-vectors.
    ///
    /// NOTE: Currently, it's undefined behavior if `new_len` does not
    /// satisfy `new_len >= iter.len() / N` and
    /// `new_len < (iter.len() + 1) / N`.
    #[inline]
    pub fn reset_from_iter_simd<I>(&mut self, new_len: usize, iter: I)
    where
        I: Iterator<Item = simd::Simd<T, N>>,
    {
        self.inner.clear();
        let capacity_v = (new_len + N - 1) / N;
        self.inner.reserve(capacity_v);
        self.inner.extend(iter.take(capacity_v));
        self.len = new_len;
    }

    /// Returns a slice of the scalar values.
    #[inline]
    pub fn as_ref(&self) -> &[T] {
        &transmute_and_flatten_simd(&self.inner)[0..self.len]
    }

    /// Returns a slice of the simd vectors.
    #[inline]
    #[allow(dead_code)]
    pub fn as_ref_simd(&self) -> &[simd::Simd<T, N>] {
        &self.inner
    }

    /// Returns a mutable slice of the simd vectors.
    #[inline]
    #[allow(dead_code)]
    pub fn as_mut_simd(&mut self) -> &mut [simd::Simd<T, N>] {
        &mut self.inner
    }
}

// deinterleaver is often used in the I/O thread which can be a performance
// bottleneck. So, hoping that LLVM optimizer can automatically SIMD-ize,
// `seq_macro` is extensively used to define chennel-specific implementations
// with unrolled loops.

/// Generic impl for `deinterleave` used when there's no special impl.
#[inline]
#[allow(dead_code)]
fn deinterleave_gen(interleaved: &[i32], channels: usize, channel_stride: usize, dest: &mut [i32]) {
    let samples = dest.len() / channels;
    let src_samples = interleaved.len() / channels;
    for t in 0..samples {
        for ch in 0..channels {
            dest[channel_stride * ch + t] = if t < src_samples {
                interleaved[channels * t + ch]
            } else {
                0i32
            }
        }
    }
}

seq!(N in 2..=8 {
    #[inline]
    #[allow(dead_code)]
    #[allow(clippy::cognitive_complexity)]
    fn deinterleave_ch~N(interleaved: &[i32], channel_stride: usize, dest: &mut [i32]) {
        let dst_samples = dest.len() / N;
        let src_samples = interleaved.len() / N;
        let mut t0 = 0;
        while t0 < dst_samples {
            repeat!(offset to 32 ; while (t0 + offset) < dst_samples => {
                repeat!(ch to N => {
                    dest[channel_stride * ch + t0 + offset] = if (t0 + offset) < src_samples {
                        interleaved[N * (t0 + offset) + ch]
                    } else {
                        0i32
                    };
                });
            });
            t0 += 32;
        }
    }
});

fn deinterleave_ch1(interleaved: &[i32], _channel_stride: usize, dest: &mut [i32]) {
    let n = std::cmp::min(dest.len(), interleaved.len());
    dest[0..n].copy_from_slice(&interleaved[0..n]);
}

/// Deinterleaves channel interleaved samples to the channel-major order.
pub fn deinterleave(interleaved: &[i32], channels: usize, channel_stride: usize, dest: &mut [i32]) {
    seq!(CH in 1..=8 {
        if channels == CH {
            return deinterleave_ch~CH(interleaved, channel_stride, dest);
        }
    });
    // This is not going to be used in FLAC, but just trying to make it
    // complete.
    deinterleave_gen(interleaved, channels, channel_stride, dest);
}

/// Implementation for each bytes-per-sample (BPS) setting.
///
/// # Panics
///
/// This function panics when the length of `bytes` is not a multiple of
/// `bytes_per_sample`, or `dest` does not have enough elements to store the
/// results.
fn le_bytes_to_i32s_impl<const BPS: usize>(bytes: &[u8], dest: &mut [i32]) {
    let t_end = bytes.len();
    assert!(t_end % BPS == 0, "t_end={t_end}, BPS={BPS}");
    assert!(dest.len() >= t_end / BPS);
    let mut t = 0;
    let mut n = 0;
    while t < t_end {
        dest[n] = i32::from_le_bytes(std::array::from_fn(|i| {
            if i < (4 - BPS) {
                0u8
            } else {
                bytes[t + i - (4 - BPS)]
            }
        })) >> ((4 - BPS) * 8);
        n += 1;
        t += BPS;
    }
}

/// Converts a byte-sequence of little-endian integers to integers (i32).
///
/// NOTE: This can also be done in a zero-copy manner by introducing a wrapper
/// that behaves like a slice and performs padding on the fly. At the moment,
/// I don't expect much improvement by doing so, but this might be an option.
///
/// # Panics
///
/// This function panics when `bytes_per_sample` is not in range `1..=4`, or
/// the length of `bytes` is not a multiple of `bytes_per_sample`, or `dest`
/// does not have enough elements to store the results.
pub fn le_bytes_to_i32s(bytes: &[u8], dest: &mut [i32], bytes_per_sample: usize) {
    if bytes_per_sample == 2 {
        le_bytes_to_i32s_impl::<2>(bytes, dest);
    } else if bytes_per_sample == 1 {
        le_bytes_to_i32s_impl::<1>(bytes, dest);
    } else if bytes_per_sample == 3 {
        le_bytes_to_i32s_impl::<3>(bytes, dest);
    } else if bytes_per_sample == 4 {
        le_bytes_to_i32s_impl::<4>(bytes, dest);
    } else {
        panic!("bytes_per_samples > 4 or bytes_per_samples == 0");
    }
}

/// Converts i32s to little-endian bytes.
///
/// NOTE: Currenty, this function is not used in "flacenc-bin", and therefore
/// this might be slow.
pub fn i32s_to_le_bytes(ints: &[i32], dest: &mut [u8], bytes_per_sample: usize) {
    let mut n = 0;
    for v in ints {
        for offset in 0..bytes_per_sample {
            dest[n] = v.to_le_bytes()[offset];
            n += 1;
        }
    }
}

/// Returns true if all elements are equal.
pub fn is_constant<T: PartialEq>(samples: &[T]) -> bool {
    for t in 1..samples.len() {
        if samples[0] != samples[t] {
            return false;
        }
    }
    true
}

/// Pack unaligned scalars into `Vec` of `Simd`s.
#[inline]
pub fn pack_into_simd_vec<T, const LANES: usize>(src: &[T], dest: &mut Vec<simd::Simd<T, LANES>>)
where
    T: simd::SimdElement + Default,
    simd::LaneCount<LANES>: simd::SupportedLaneCount,
{
    let zero_v = simd::Simd::<T, LANES>::default();
    dest.clear();
    let len = src.len();
    let len_v = (len + LANES - 1) / LANES;
    dest.resize(len_v, zero_v);

    transmute_and_flatten_simd_mut(dest)[0..len].copy_from_slice(src);
}

/// Unpack slice of `Simd` into `Vec` of scalars.
///
/// NOTE: This is currently not used. Since this function is expected to be
/// useful in future enhancements, it is kept here.
/// NOTE: This performs unnecessary copy. It might be better to consider
/// another zero-copy solution.
#[allow(dead_code)]
pub fn unpack_simds<T, const LANES: usize>(src: &[simd::Simd<T, LANES>], dest: &mut Vec<T>)
where
    T: simd::SimdElement + Default,
    simd::LaneCount<LANES>: simd::SupportedLaneCount,
{
    dest.resize(src.len() * LANES, T::default());
    let mut offset = 0;
    for v in src {
        let arr = <[T; LANES]>::from(*v);
        dest[offset..offset + LANES].copy_from_slice(&arr);
        offset += LANES;
    }
}

/// A wrapper of `slice::as_simd` with a fallback behavior in stable toolchain.
fn slice_as_simd<T, const N: usize>(data: &[T]) -> (&[T], &[simd::Simd<T, N>], &[T])
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    #[cfg(feature = "simd-nightly")]
    {
        data.as_simd()
    }
    #[cfg(not(feature = "simd-nightly"))]
    {
        (data, &[], &[])
    }
}

/// A wrapper of `slice::as_simd_mut` supporting stable toolchain.
fn slice_as_simd_mut<T, const N: usize>(
    data: &mut [T],
) -> (&mut [T], &mut [simd::Simd<T, N>], &mut [T])
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    #[cfg(feature = "simd-nightly")]
    {
        data.as_simd_mut()
    }
    #[cfg(not(feature = "simd-nightly"))]
    {
        (data, &mut [], &mut [])
    }
}

/// Maps and reduces the array with SIMD acceleration.
fn simd_map_and_reduce<U, const N: usize, T, F, G, Q, R, S>(
    data: &[T],
    scalar_fn: F,
    vector_fn: G,
    scalar_reduce_fn: Q,
    vector_reduce_fn: R,
    vector_to_scalar_fn: S,
    init: U,
) -> U
where
    T: simd::SimdElement,
    U: simd::SimdElement,
    F: Fn(T) -> U,
    G: Fn(simd::Simd<T, N>) -> simd::Simd<U, N>,
    Q: Fn(U, U) -> U,
    R: Fn(simd::Simd<U, N>, simd::Simd<U, N>) -> simd::Simd<U, N>,
    S: Fn(simd::Simd<U, N>) -> U,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    let mut acc_v = simd::Simd::splat(init);
    let mut acc = init;
    let (head, body, foot) = slice_as_simd(data);
    for x in head {
        let y = scalar_fn(*x);
        acc = scalar_reduce_fn(y, acc);
    }
    for v in body {
        let w = vector_fn(*v);
        acc_v = vector_reduce_fn(w, acc_v);
    }
    for x in foot {
        let y = scalar_fn(*x);
        acc = scalar_reduce_fn(y, acc);
    }
    scalar_reduce_fn(acc, vector_to_scalar_fn(acc_v))
}

/// Finds the element with maximum absolute value from the data.
pub fn find_max_abs<const N: usize>(data: &[i32]) -> u32
where
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    simd_map_and_reduce(
        data,
        i32::unsigned_abs,
        |v| v.abs().cast(),
        std::cmp::max,
        simd::SimdOrd::simd_max,
        simd::SimdUint::reduce_max,
        0,
    )
}

/// Updates `dest` using the given functions and their arguments `src`.
///
/// This function updates `dest` using a scalar update function `scalar_fn`
/// that is called as `scalar_fn(&mut dest[t], src[t])` and a vector function
/// that is called as `vector_fn(&mut dest_v[n], src_v[n])` where `dest_v[n]`
/// and `src_v[n]` are the n-th simd vectors in `dest` and `src` respectively.
/// Both `dest` and `src` do not need to be SIMD aligned. This function
/// splits `dest` into unaligned prefix/ suffix and aligned body, and performs
/// unaligned load from `src` for preparing the matching simd vector from
/// `src`.
#[inline]
pub fn unaligned_map_and_update<T, const N: usize, U, F, G>(
    src: &[U],
    dest: &mut [T],
    mut scalar_fn: F,
    mut vector_fn: G,
) where
    T: simd::SimdElement,
    U: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
    F: FnMut(&mut T, U),
    G: FnMut(&mut simd::Simd<T, N>, simd::Simd<U, N>),
{
    let (head, body, foot) = slice_as_simd_mut(dest);

    let mut t = 0;
    for p in head {
        scalar_fn(p, src[t]);
        t += 1;
    }

    for pv in body {
        let src_v = simd::Simd::from_slice(&src[t..t + N]);
        vector_fn(pv, src_v);
        t += N;
    }

    for p in foot {
        scalar_fn(p, src[t]);
        t += 1;
    }
}

/// Transmutes a slice of `[Simd]`s into a slice of scalars.
///
/// This hides an unsafe block. The operation is basically safe as rust ensures
/// that:
///   1. array is contiguous, i.e. `size_of::<[T; N]>() == size_of::<T>() * N`.
///      <https://doc.rust-lang.org/beta/reference/type-layout.html#array-layout>
///   2. `Simd` is bit-equivalent with array, i.e.
///      `size_of::<Simd<T, N>>() == size_of::<T>() * N`.
///      <https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#size-alignment-and-unsafe-code>
/// Still, there's a possibility that some architecture uses a different
/// element (or bit) order inside since the above conditions only for there
/// will be no memory violations. For the orders, we only rely on unittests.
///
/// In future, we may rely on `zerocopy` crate that only support raw
/// (non-portable) simd types, currently.
pub fn transmute_and_flatten_simd<T, const N: usize>(simds: &[simd::Simd<T, N>]) -> &[T]
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    let newlen = simds.len() * N;
    unsafe { std::slice::from_raw_parts(simds.as_ptr().cast(), newlen) }
}

/// Transmutes a slice of `[Simd]`s into a slice of scalars.
///
/// This hides an unsafe block. See documents for `transmute_and_flatten_simd`
/// for safety discussions.
pub fn transmute_and_flatten_simd_mut<T, const N: usize>(simds: &mut [simd::Simd<T, N>]) -> &mut [T]
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    let newlen = simds.len() * N;
    unsafe { std::slice::from_raw_parts_mut(simds.as_mut_ptr().cast(), newlen) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn do_deinterleave() {
        let interleaved = [0, 0, -1, -2, 1, 2, -3, 6];
        let mut dest = vec![0i32; interleaved.len()];
        deinterleave(&interleaved, 2, 4, &mut dest);
        assert_eq!(&dest, &[0, -1, 1, -3, 0, -2, 2, 6]);

        let interleaved = [0, 0, -1, -2, 1, 2, -3, 6];
        let mut dest = vec![-123i32; interleaved.len() + 4];
        deinterleave(&interleaved, 4, 3, &mut dest);
        assert_eq!(&dest, &[0, 1, 0, 0, 2, 0, -1, -3, 0, -2, 6, 0]);

        let mut dest = vec![-123i32; interleaved.len() + 4];
        deinterleave_gen(&interleaved, 4, 3, &mut dest);
        assert_eq!(&dest, &[0, 1, 0, 0, 2, 0, -1, -3, 0, -2, 6, 0]);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn deinterleave_ch2_direct_call() {
        let block_size: usize = 4096;
        let src_len: usize = 123;
        let mut src = Vec::new();
        src.extend(0i32..(src_len as i32 * 2));

        let mut dest = vec![0i32; block_size * 2];
        deinterleave_ch2(&src, block_size, &mut dest);

        let mut expected = 0i32;
        for t in 0..src_len {
            assert_eq!(dest[t], expected);
            expected += 2;
        }
        let mut expected = 1i32;
        for t in block_size..(block_size + src_len) {
            assert_eq!(dest[t], expected);
            expected += 2;
        }
    }

    #[test]
    fn convert_le_bytes_to_ints() {
        let bytes = [
            0x56, 0x34, 0x12, 0x9B, 0x57, 0x13, 0xFF, 0xFF, 0xFF, 0xAC, 0x68, 0x24,
        ];
        let mut dest = [0i32; 4];
        le_bytes_to_i32s(&bytes, &mut dest, 3);
        assert_eq!(dest, [0x12_3456, 0x13_579B, -1, 0x24_68AC]);
    }

    #[test]
    fn constant_detector() {
        let signal = vec![5; 64];
        assert!(super::is_constant(&signal));

        let signal = vec![-3; 192];
        assert!(super::is_constant(&signal));

        let mut signal = vec![8.2f32; 192];
        signal[191] = f32::NAN;
        assert!(!super::is_constant(&signal));
    }

    #[test]
    fn packing_into_simd_vec() {
        let vals: Vec<i16> = (0i16..7i16).collect();
        let mut dest: Vec<simd::i16x4> = vec![];
        pack_into_simd_vec(&vals, &mut dest);

        assert_eq!(
            dest,
            [
                simd::i16x4::from_array([0, 1, 2, 3]),
                simd::i16x4::from_array([4, 5, 6, 0])
            ]
        );

        let vals: Vec<i16> = (0i16..12i16).collect();
        let mut dest: Vec<simd::i16x4> = vec![];
        pack_into_simd_vec(&vals, &mut dest);

        assert_eq!(
            dest,
            [
                simd::i16x4::from_array([0, 1, 2, 3]),
                simd::i16x4::from_array([4, 5, 6, 7]),
                simd::i16x4::from_array([8, 9, 10, 11])
            ]
        );

        let vals = [];
        let mut dest: Vec<simd::i16x4> = vec![];
        pack_into_simd_vec(&vals, &mut dest);

        assert_eq!(dest, []);
    }

    #[test]
    fn unpacking_simd_vectors() {
        let vs = [
            simd::i16x4::from_array([0, 1, 2, 3]),
            simd::i16x4::from_array([4, 5, 6, 0]),
        ];
        let mut dest = vec![];
        unpack_simds(&vs, &mut dest);
        assert_eq!(dest, [0, 1, 2, 3, 4, 5, 6, 0]);

        let vs: [simd::i16x4; 0] = [];
        let mut dest = vec![1, 2, 3];
        unpack_simds(&vs, &mut dest);
        assert_eq!(dest, []);
    }

    #[test]
    fn simd_ref_can_be_flattened() {
        let vs = [
            simd::i16x4::from_array([0, 1, 2, 3]),
            simd::i16x4::from_array([4, 5, 6, 0]),
        ];
        assert_eq!(transmute_and_flatten_simd(&vs), &[0, 1, 2, 3, 4, 5, 6, 0]);

        // small simd struct
        let vs = [
            simd::i8x1::from_array([-1]),
            simd::i8x1::from_array([-2]),
            simd::i8x1::from_array([-3]),
        ];
        assert_eq!(transmute_and_flatten_simd(&vs), &[-1, -2, -3]);

        // large simd struct
        let vs = [
            simd::u64x64::from_array(std::array::from_fn(|d| d as u64)),
            simd::u64x64::from_array(std::array::from_fn(|d| d as u64 + 1)),
            simd::u64x64::from_array(std::array::from_fn(|d| d as u64 + 2)),
        ];
        let flat_view = transmute_and_flatten_simd(&vs);
        for (i, v) in flat_view.iter().enumerate() {
            let offset = i as u64 / 64;
            assert_eq!(*v, offset + i as u64 % 64);
        }
    }

    #[test]
    fn find_max_abs_works() {
        let vs = [
            0,
            0,
            0,
            i32::MIN,
            i32::MAX,
            1,
            2,
            3,
            4,
            5,
            i32::MIN,
            i32::MAX,
            0,
            0,
            0,
        ];
        assert_eq!(find_max_abs::<4>(&vs), i32::MIN.unsigned_abs());
    }
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    fn simd_packing(b: &mut Bencher) {
        let mut src = Vec::new();
        src.extend(0i32..4096i32);
        let mut dest: Vec<simd::Simd<i32, 16>> = Vec::new();

        b.iter(|| pack_into_simd_vec(black_box(&src), black_box(&mut dest)));
    }

    #[bench]
    fn deinterleave_ch2_direct_call(b: &mut Bencher) {
        let mut src = Vec::new();
        src.extend(0i32..(4096i32 * 2));
        let mut dest = vec![0i32; 4096 * 2];

        b.iter(|| deinterleave_ch2(black_box(&src), black_box(4096), black_box(&mut dest)));
    }
}
