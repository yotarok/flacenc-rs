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

//! Module for input source handling.

use std::fmt;

use md5::Digest;

use super::arrayutils::deinterleave;
use super::arrayutils::find_min_and_max;
use super::arrayutils::le_bytes_to_i32s;
use super::constant::MAX_BLOCK_SIZE;
use super::constant::MAX_CHANNELS;
use super::constant::MIN_BLOCK_SIZE;
use super::error::verify_range;
use super::error::verify_true;
use super::error::SourceError;
use super::error::VerifyError;

/// Traits for buffer-like objects that can be filled by [`Source`].
///
/// An implementation of [`Source::read_samples`] is expected to call one
/// of the `fill_*` method declared in this trait.
///
/// An impl of `Fill` must accept the samples that is shorter than the pre-
/// defined length for e.g. the last frame handling. On the other hand,
/// `Fill` is expected to return an error if the number of samples is larger
/// than the block size.
pub trait Fill {
    /// Fills the target variable with the given interleaved samples.
    ///
    /// # Errors
    ///
    /// This may fail when configuration of `Fill` is not consistent with the
    /// input `interleaved` values.
    ///
    /// # Examples
    ///
    /// [`FrameBuf`] implements `Fill`.
    ///
    /// ```
    /// # use flacenc::source::{Fill, FrameBuf};
    /// let mut fb = FrameBuf::with_size(8, 1024).unwrap();
    /// fb.fill_interleaved(&[0i32; 8 * 1024]);
    /// ```
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError>;

    /// Fills target with the little-endian bytes that represent samples.
    ///
    /// # Errors
    ///
    /// This may fail when configuration of `Fill` is not consistent with the
    /// input `bytes` or `bytes_per_sample` values.
    ///
    /// # Examples
    ///
    /// [`FrameBuf`] implements `Fill`.
    ///
    /// ```
    /// # use flacenc::source::{Fill, FrameBuf};
    /// let mut fb = FrameBuf::with_size(2, 64).unwrap();
    /// // Note that `FrameBuf` (or `Fill` in general) accepts shorter inputs.
    /// fb.fill_le_bytes(&[0x12, 0x34, 0x54, 0x76, 0x56, 0x78, 0x10, 0x32], 2);
    /// // this FrameBuf now has 2 channels with elements:
    /// //   - channel-1 (left) : [0x3412, 0x7856]
    /// //   - channel-2 (right): [0x7654, 0x3210]
    /// ```
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError>;
}

impl<T: Fill, U: Fill> Fill for (T, U) {
    #[inline]
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        self.0.fill_interleaved(interleaved)?;
        self.1.fill_interleaved(interleaved)
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        self.0.fill_le_bytes(bytes, bytes_per_sample)?;
        self.1.fill_le_bytes(bytes, bytes_per_sample)
    }
}

impl<T> Fill for &mut T
where
    T: Fill,
{
    #[inline]
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        T::fill_interleaved(self, interleaved)
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        T::fill_le_bytes(self, bytes, bytes_per_sample)
    }
}

/// Reusable buffer for multi-channel framed signals.
#[derive(Clone, Debug)]
pub struct FrameBuf {
    samples: Vec<i32>,
    size: usize,
    /// The number of loaded inter-channel samples.
    ///
    /// this can be smaller than `self.samples.len() / self.channels` for the last block of the
    /// stream.
    filled_size: usize,
    /// Working buffer.
    ///
    /// This is currently only used in `read_le_bytes` (for storing `i32`-upcasted samples).
    readbuf: Vec<i32>,
}

impl FrameBuf {
    /// Constructs new 2-channel `FrameBuf` that will be later resized.
    ///
    /// This is a safe constructor that never fails, and always produce a valid
    /// `FrameBuf`. This constructor is intended to be used for preparing
    /// reusable buffer for stereo coding.
    pub(crate) fn new_stereo_buffer() -> Self {
        Self {
            samples: vec![0i32; 256 * 2],
            size: 256,
            filled_size: 0,
            readbuf: vec![],
        }
    }

    /// Constructs `FrameBuf` of the specified size.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if arguments are out of the ranges of FLAC
    /// specifications. Specifically `channels` must be in `1..=[MAX_CHANNELS]` and
    /// size must be in `MIN_BLOCK_SIZE]..=MAX_BLOCK_SIZE`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::FrameBuf;
    /// let fb = FrameBuf::with_size(2, 1024).unwrap();
    /// assert_eq!(fb.size(), 1024);
    /// ```
    pub fn with_size(channels: usize, size: usize) -> Result<Self, VerifyError> {
        verify_range!("FrameBuf::with_size (channels)", channels, 1..=MAX_CHANNELS)?;
        verify_range!(
            "FrameBuf::with_size (block size)",
            size,
            MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE
        )?;
        Ok(Self {
            samples: vec![0i32; size * channels],
            size,
            filled_size: 0,
            readbuf: vec![],
        })
    }

    /// Returns the size in the number of per-channel samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::FrameBuf;
    /// let fb = FrameBuf::with_size(2, 1024).unwrap();
    /// assert_eq!(fb.size(), 1024);
    /// ```
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the number of inter-channel samples written to this `FrameBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::Fill;
    /// # use flacenc::source::FrameBuf;
    /// let mut fb = FrameBuf::with_size(1, 1024).unwrap();
    /// fb.fill_interleaved(&[0, 1, 2, 3]);
    /// assert_eq!(fb.filled_size(), 4);
    /// ```
    pub const fn filled_size(&self) -> usize {
        self.filled_size
    }

    /// Fill stereo buffer with the stereo samples from the given iterator.
    ///
    /// This is currently only used for making M/S framebuffer from the L/R buffer.
    pub(crate) fn fill_stereo_with_iter<I>(&mut self, iter: I)
    where
        I: Iterator<Item = (i32, i32)>,
    {
        assert_eq!(2, self.channels());
        let (m_slice, s_slice) = self.samples.split_at_mut(self.size);
        self.filled_size = 0;
        let dest_iter = m_slice.iter_mut().zip(s_slice.iter_mut());
        for ((m, s), (dest_m, dest_s)) in iter.take(self.size).zip(dest_iter) {
            *dest_m = m;
            *dest_s = s;
            self.filled_size += 1;
        }
    }

    /// Resizes `FrameBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::FrameBuf;
    /// let mut fb = FrameBuf::with_size(2, 1024).unwrap();
    /// assert_eq!(fb.size(), 1024);
    /// fb.resize(2048);
    /// assert_eq!(fb.size(), 2048);
    /// ```
    pub fn resize(&mut self, new_size: usize) {
        let channels = self.channels();
        self.size = new_size;
        self.samples.resize(self.size * channels, 0i32);
    }

    /// Returns the number of channels
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::FrameBuf;
    /// let fb = FrameBuf::with_size(8, 1024).unwrap();
    /// assert_eq!(fb.channels(), 8);
    /// ```
    pub fn channels(&self) -> usize {
        self.samples.len() / self.size
    }

    /// Returns samples from the given channel.
    pub(crate) fn channel_slice(&self, ch: usize) -> &[i32] {
        &self.samples[ch * self.size..(ch * self.size + self.filled_size)]
    }

    /// Returns the internal representation of multichannel signals.
    #[cfg(test)]
    pub(crate) fn raw_slice(&self) -> &[i32] {
        &self.samples
    }

    /// Verifies data consistency with the given stream info.
    pub(crate) fn verify_samples(&self, bits_per_sample: usize) -> Result<(), VerifyError> {
        let max_allowed = (1i32 << (bits_per_sample - 1)) - 1;
        let min_allowed = -(1i32 << (bits_per_sample - 1));
        for ch in 0..self.channels() {
            let (min, max) = find_min_and_max::<64>(self.channel_slice(ch), 0i32);
            if min < min_allowed || max > max_allowed {
                return Err(VerifyError::new(
                    "input.framebuf",
                    &format!("input sample must be in the range of bits={bits_per_sample}"),
                ));
            }
        }
        Ok(())
    }
}

impl Fill for FrameBuf {
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        let stride = self.size();
        let channels = self.channels();
        deinterleave(interleaved, channels, stride, &mut self.samples);
        self.filled_size = interleaved.len() / channels;
        Ok(())
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        let sample_count = bytes.len() / bytes_per_sample;
        self.readbuf.resize(sample_count, 0);
        le_bytes_to_i32s(bytes, &mut self.readbuf, bytes_per_sample);

        let stride = self.size();
        let channels = self.channels();
        deinterleave(&self.readbuf, self.channels(), stride, &mut self.samples);
        self.filled_size = sample_count / channels;
        Ok(())
    }
}

/// Context information being updated while reading from [`Source`].
///
/// Some information such as MD5 of the input waveform is better handled in
/// [`Source`]-side rather than via frame buffers. `Context` is for handling
/// such context variables.
#[derive(Clone)]
pub struct Context {
    md5: md5::Md5,
    bytes_per_sample: usize,
    channels: usize,
    sample_count: usize,
    frame_count: usize,
}

impl Context {
    /// Creates new context.
    ///
    /// # Panics
    ///
    /// Panics if `bits_per_sample > 32`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::Context;
    /// let ctx = Context::new(16, 2);
    /// assert!(ctx.current_frame_number().is_none());
    /// assert_eq!(ctx.total_samples(), 0);
    /// ```
    pub fn new(bits_per_sample: usize, channels: usize) -> Self {
        let bytes_per_sample = (bits_per_sample + 7) / 8;
        assert!(
            bytes_per_sample <= 4,
            "bits_per_sample={bits_per_sample} cannot be larger than 32."
        );
        Self {
            md5: md5::Md5::new(),
            bytes_per_sample,
            channels,
            sample_count: 0,
            frame_count: 0,
        }
    }

    /// Returns bytes-per-sample configuration of this `Context`.
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        self.bytes_per_sample
    }

    /// Returns the count of the last frame loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::{Context, Fill};
    /// let mut ctx = Context::new(16, 2);
    /// assert!(ctx.current_frame_number().is_none());
    ///
    /// ctx.fill_interleaved(&[0, -1, -2, 3]);
    /// assert_eq!(ctx.current_frame_number(), Some(0usize));
    /// ```
    #[inline]
    #[allow(clippy::unnecessary_lazy_evaluations)] // false-alarm
    pub fn current_frame_number(&self) -> Option<usize> {
        (self.frame_count > 0).then(|| self.frame_count - 1)
    }

    /// Returns MD5 digest of the consumed samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::Context;
    /// let ctx = Context::new(16, 2);
    /// let zero_md5 = [
    ///     0xD4, 0x1D, 0x8C, 0xD9, 0x8F, 0x00, 0xB2, 0x04,
    ///     0xE9, 0x80, 0x09, 0x98, 0xEC, 0xF8, 0x42, 0x7E,
    /// ];
    /// assert_eq!(ctx.md5_digest(), zero_md5);
    /// // it doesn't change if you don't call "update" functions.
    /// assert_eq!(ctx.md5_digest(), zero_md5);
    /// ```
    #[inline]
    pub fn md5_digest(&self) -> [u8; 16] {
        self.md5.clone().finalize().into()
    }

    /// Returns the number of samples loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::{Context, Fill};
    /// let mut ctx = Context::new(16, 2);
    ///
    /// ctx.fill_interleaved(&[0, -1, -2, 3]);
    /// assert_eq!(ctx.total_samples(), 2);
    /// ```
    #[inline]
    pub fn total_samples(&self) -> usize {
        self.sample_count
    }
}

impl Fill for Context {
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        if interleaved.is_empty() {
            return Ok(());
        }
        for v in interleaved {
            self.md5.update(&v.to_le_bytes()[0..self.bytes_per_sample]);
        }
        self.sample_count += interleaved.len() / self.channels;
        self.frame_count += 1;
        Ok(())
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.md5.update(bytes);
        self.sample_count += bytes.len() / self.channels / bytes_per_sample;
        self.frame_count += 1;
        Ok(())
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digest = format!("{:?}", self.md5.clone().finalize());
        f.debug_struct("Context")
            .field("bytes_per_sample", &self.bytes_per_sample)
            .field("channels", &self.channels)
            .field("sample_count", &self.sample_count)
            .field("frame_count", &self.frame_count)
            .field("md5", &digest)
            .finish()
    }
}

/// Trait representing the input source for the encoder.
pub trait Source {
    /// Returns the number of channels.
    fn channels(&self) -> usize;
    /// Returns the number of bits per sample;
    fn bits_per_sample(&self) -> usize;
    /// Returns sampling rate in Hz.
    fn sample_rate(&self) -> usize;
    /// Reads samples to [`T: Fill`](Fill).
    ///
    /// Implementation of this function must call either
    /// [`dest.fill_interleaved`](Fill::fill_interleaved) or
    /// [`dest.fill_le_bytes`](Fill::fill_le_bytes),
    /// and returns the number of per-channel samples read.
    ///
    /// # Errors
    ///
    /// This function can return [`SourceError`] when read is failed.
    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError>;
    /// Returns length of source if it's finite and defined.
    fn len_hint(&self) -> Option<usize> {
        None
    }
}

impl<T: Source> Source for &mut T {
    fn channels(&self) -> usize {
        T::channels(self)
    }
    fn bits_per_sample(&self) -> usize {
        T::bits_per_sample(self)
    }
    fn sample_rate(&self) -> usize {
        T::sample_rate(self)
    }
    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        T::read_samples(self, block_size, dest)
    }
    fn len_hint(&self) -> Option<usize> {
        T::len_hint(self)
    }
}

/// Trait representing seekable variant of [`Source`].
///
/// This trait is not currently used in the encoder, but some encoding algorithm
/// in future may require that a source is seekable.
pub trait Seekable: Source {
    /// Returns `true` if the source contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length in per-channel samples
    fn len(&self) -> usize;

    /// Seeks to the specified offset from the beginning.
    ///
    /// # Errors
    ///
    /// This function can return [`SourceError`] when read is failed.
    fn read_samples_from<F: Fill>(
        &mut self,
        offset: usize,
        block_size: usize,
        context: &mut F,
    ) -> Result<usize, SourceError>;
}

impl<T: Seekable> Seekable for &mut T {
    fn is_empty(&self) -> bool {
        T::is_empty(self)
    }

    fn len(&self) -> usize {
        T::len(self)
    }

    fn read_samples_from<F: Fill>(
        &mut self,
        offset: usize,
        block_size: usize,
        context: &mut F,
    ) -> Result<usize, SourceError> {
        T::read_samples_from(self, offset, block_size, context)
    }
}

/// Source with preloaded samples.
#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct MemSource {
    channels: usize,
    bits_per_sample: usize,
    sample_rate: usize,
    samples: Vec<i32>,
    read_head: usize,
}

impl MemSource {
    /// Constructs `MemSource` from samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use flacenc::source::MemSource;
    ///
    /// let src = MemSource::from_samples(&[0, 0, 1, -1, 2, -2, 3, -3], 2, 16, 8000);
    /// assert_eq!(src.as_slice(), &[0, 0, 1, -1, 2, -2, 3, -3]);
    /// ```
    pub fn from_samples(
        samples: &[i32],
        channels: usize,
        bits_per_sample: usize,
        sample_rate: usize,
    ) -> Self {
        Self {
            channels,
            bits_per_sample,
            sample_rate,
            samples: samples.to_owned(),
            read_head: 0,
        }
    }

    /// Returns sample buffer as a raw slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::MemSource;
    /// let src = MemSource::from_samples(&[0, 0, 1, -1, 2, -2, 3, -3], 2, 16, 8000);
    /// assert_eq!(src.as_slice(), &[0, 0, 1, -1, 2, -2, 3, -3]);
    /// ```
    pub fn as_slice(&self) -> &[i32] {
        &self.samples
    }
}

impl Source for MemSource {
    fn channels(&self) -> usize {
        self.channels
    }

    fn bits_per_sample(&self) -> usize {
        self.bits_per_sample
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        self.read_samples_from(self.read_head, block_size, dest)
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl Seekable for MemSource {
    fn len(&self) -> usize {
        self.samples.len() / self.channels()
    }

    fn read_samples_from<F: Fill>(
        &mut self,
        offset: usize,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        let to_read = block_size * self.channels;
        let begin = std::cmp::min(offset * self.channels, self.samples.len());
        let end = std::cmp::min(offset * self.channels + to_read, self.samples.len());
        let src = &self.samples[begin..end];

        dest.fill_interleaved(src)?;

        let read_samples = (end - begin) / self.channels;
        self.read_head += read_samples;
        Ok(read_samples)
    }
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::nursery, clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn reading_and_deinterleaving() {
        let mut signal = vec![];
        let block_size = 512;
        let channels = 4;
        for t in 0..block_size {
            for _ch in 0..channels {
                signal.push(t as i32);
            }
        }

        let mut src = MemSource::from_samples(&signal, channels, 16, 16000);
        let mut framebuf_and_ctx = (
            FrameBuf::with_size(channels, block_size).unwrap(),
            Context::new(16, channels),
        );
        let read = src
            .read_samples_from(0, block_size, &mut framebuf_and_ctx)
            .expect("Read error");
        assert_eq!(read, block_size);
        let (framebuf, _ctx) = framebuf_and_ctx;
        let mut head = 0;
        for _ch in 0..channels {
            for t in 0..block_size {
                assert_eq!(framebuf.raw_slice()[head], t as i32);
                head += 1;
            }
        }
    }

    #[test]
    fn sequential_read() {
        let mut signal = vec![];
        let total_size = 1100;
        let channels = 3;
        for t in 0..total_size {
            for ch in 0..channels {
                let sign: i32 = if ch == 0 { 1 } else { -1 };
                signal.push(sign * t);
            }
        }

        let block_size = 128;
        let mut src = MemSource::from_samples(&signal, channels, 16, 16000);
        let ctx = Context::new(16, channels);
        let framebuf = FrameBuf::with_size(channels, block_size).unwrap();
        let mut framebuf_and_ctx = (framebuf, ctx);

        for step in 0..8 {
            let read = src
                .read_samples(block_size, &mut framebuf_and_ctx)
                .expect("Read error");
            assert_eq!(read, 128);
            assert_eq!(src.read_head, 128 * (step + 1));
            for t in 0..block_size {
                assert_eq!(
                    framebuf_and_ctx.0.channel_slice(0)[t],
                    (block_size * step + t) as i32
                );
                assert_eq!(
                    framebuf_and_ctx.0.channel_slice(1)[t],
                    -((block_size * step + t) as i32)
                );
            }
        }
        let read = src
            .read_samples(block_size, &mut framebuf_and_ctx)
            .expect("Read error");
        assert_eq!(read, 76);
        for t in 0..76 {
            assert_eq!(framebuf_and_ctx.0.channel_slice(0)[t], (1024 + t) as i32);
            assert_eq!(framebuf_and_ctx.0.channel_slice(1)[t], -((1024 + t) as i32));
            assert_eq!(framebuf_and_ctx.0.channel_slice(2)[t], -((1024 + t) as i32));
        }
    }

    #[test]
    fn md5_computation() {
        let mut ctx = Context::new(16, 2);
        ctx.fill_interleaved(&[0i32; 32 * 2])
            .expect("update failed");

        // Reference computed with Python's hashlib.
        assert_eq!(
            ctx.md5_digest(),
            [
                0xF0, 0x9F, 0x35, 0xA5, 0x63, 0x78, 0x39, 0x45, 0x8E, 0x46, 0x2E, 0x63, 0x50, 0xEC,
                0xBC, 0xE4
            ]
        );

        let mut ctx = Context::new(16, 2);
        ctx.fill_interleaved(&[0xABCDi32; 32 * 2])
            .expect("update failed");
        // Reference computed by a reliable version of this library.
        assert_eq!(
            ctx.md5_digest(),
            [
                0x02, 0x3D, 0x3A, 0xE9, 0x26, 0x0B, 0xB0, 0xC9, 0x51, 0xF6, 0x5B, 0x25, 0x24, 0x62,
                0xB1, 0xFA
            ]
        );
    }
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    fn feeding_bytes_to_context(b: &mut Bencher) {
        let (bytes_per_sample, channels, block_size) = (2, 2, 4096);
        let mut ctx = Context::new(bytes_per_sample, channels);
        let signal_bytes = vec![0u8; bytes_per_sample * channels * block_size];
        b.iter(|| ctx.fill_le_bytes(black_box(&signal_bytes), bytes_per_sample));
    }
}
