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

//! Module for input source handling.

use seq_macro::seq;
use std::fmt;

use super::error::SourceError;
use super::error::SourceErrorReason;

// deinterleaver is often used in the I/O thread which can be a performance
// bottleneck. So, hoping that LLVM optimizer can automatically SIMD-ize,
// `seq_macro` is extensively used to define chennel-specific implementations
// with unrolled loops.

#[inline]
#[allow(dead_code)]
fn deinterleave_gen(interleaved: &[i32], channels: usize, dest: &mut [i32]) {
    let samples = dest.len() / channels;
    let src_samples = interleaved.len() / channels;
    for t in 0..samples {
        for ch in 0..channels {
            dest[samples * ch + t] = if t < src_samples {
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
    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    fn deinterleave_ch~N(interleaved: &[i32], dest: &mut [i32]) {
        let samples = dest.len() / N;
        let src_samples = interleaved.len() / N;
        let mut t = 0;
        while t < samples {
            let t0 = t;
            seq!(UNROLL in 0..32 {
                seq!(CH in 0..N {
                    dest[samples * CH + t0 + UNROLL] = if t < src_samples {
                        interleaved[N * (t0 + UNROLL) + CH]
                    } else {
                        0i32
                    };
                });
                t += 1;
                if t >= samples {
                    break;
                }
            });
        }
    }
});

fn deinterleave_ch1(interleaved: &[i32], dest: &mut [i32]) {
    let n = std::cmp::min(dest.len(), interleaved.len());
    dest[0..n].copy_from_slice(&interleaved[0..n]);
}

/// Deinterleaves channel interleaved samples to the channel-major order.
pub(crate) fn deinterleave(interleaved: &[i32], channels: usize, dest: &mut [i32]) {
    seq!(CH in 1..=8 {
        if channels == CH {
            return deinterleave_ch~CH(interleaved, dest);
        }
    });
    // This is not going to be used in FLAC, but just trying to make it
    // complete.
    deinterleave_gen(interleaved, channels, dest);
}

/// Reusable buffer for multi-channel framed signals.
#[derive(Clone, Debug)]
pub struct FrameBuf {
    samples: Vec<i32>,
    channels: usize,
    size: usize,
}

impl FrameBuf {
    /// Constructs `FrameBuf` of the specified size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    /// let fb = FrameBuf::with_size(2, 1024);
    /// assert_eq!(fb.size(), 1024);
    /// ```
    pub fn with_size(channels: usize, size: usize) -> Self {
        Self {
            samples: vec![0i32; size * channels],
            channels,
            size,
        }
    }

    /// Returns the size in the number of per-channel samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    /// let fb = FrameBuf::with_size(2, 1024);
    /// assert_eq!(fb.size(), 1024);
    /// ```
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Resizes `FrameBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    /// let mut fb = FrameBuf::with_size(2, 1024);
    /// assert_eq!(fb.size(), 1024);
    /// fb.resize(2048);
    /// assert_eq!(fb.size(), 2048);
    /// ```
    pub fn resize(&mut self, new_size: usize) {
        self.size = new_size;
        self.samples.resize(self.size * self.channels, 0i32);
    }

    /// Fill first samples from the interleaved slice, and resets rest.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    /// let mut fb = FrameBuf::with_size(8, 1024);
    /// fb.fill_from_interleaved(&[0i32; 8 * 1024]);
    /// ```
    pub fn fill_from_interleaved(&mut self, interleaved: &[i32]) {
        deinterleave(interleaved, self.channels, &mut self.samples);
    }

    /// Returns the number of channels
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    /// let fb = FrameBuf::with_size(8, 1024);
    /// assert_eq!(fb.channels(), 8);
    /// ```
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Returns samples from the given channel.
    pub(crate) fn channel_slice(&self, ch: usize) -> &[i32] {
        &self.samples[ch * self.size..(ch + 1) * self.size]
    }

    /// Returns mutable samples from the given channel.
    pub(crate) fn channel_slice_mut(&mut self, ch: usize) -> &mut [i32] {
        &mut self.samples[ch * self.size..(ch + 1) * self.size]
    }

    /// Returns the internal representation of multichannel signals.
    #[cfg(test)]
    pub(crate) fn raw_slice(&self) -> &[i32] {
        &self.samples
    }
}

/// Context information being updated while reading from [`Source`].
///
/// Some information such as MD5 of the input waveform is better handled in
/// [`Source`]-side rather than via frame buffers. `Context` is for handling
/// such context variables.
#[derive(Clone)]
pub struct Context {
    md5: md5::Context,
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
    /// # use flacenc::source::*;
    ///
    /// let ctx = Context::new(16, 2);
    /// assert!(ctx.current_frame_number().is_none());;
    /// assert_eq!(ctx.total_samples(), 0);
    /// ```
    pub fn new(bits_per_sample: usize, channels: usize) -> Self {
        let bytes_per_sample = (bits_per_sample + 7) / 8;
        assert!(
            bytes_per_sample <= 4,
            "bits_per_sample={bits_per_sample} cannot be larger than 32."
        );
        Self {
            md5: md5::Context::new(),
            bytes_per_sample,
            channels,
            sample_count: 0,
            frame_count: 0,
        }
    }

    /// Updates MD5 context for input hash (checksum) computation.
    #[inline]
    fn update_md5(&mut self, interleaved: &[i32], block_size: usize) {
        const ZEROS: [u8; 4] = [0u8; 4];
        for v in interleaved {
            self.md5.consume(&v.to_le_bytes()[0..self.bytes_per_sample]);
        }
        for _t in interleaved.len()..(block_size * self.channels) {
            self.md5.consume(&ZEROS[0..self.bytes_per_sample]);
        }
    }

    /// Updates the context with the read samples.
    ///
    /// This function is intended to be called from [`Source::read_samples`].
    ///
    /// # Errors
    ///
    /// This function currently does not return an error.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    ///
    /// let mut ctx = Context::new(16, 2);
    ///
    /// // it's okay to feed shorter blocks (`interleaved.len() < block_size == 30`).
    /// ctx.update(&[0, -1, -2, 3], 30);
    ///
    /// assert_eq!(ctx.total_samples(), 2);
    /// ```
    pub fn update(&mut self, interleaved: &[i32], block_size: usize) -> Result<(), SourceError> {
        self.update_md5(interleaved, block_size);
        self.sample_count += interleaved.len() / self.channels;
        self.frame_count += 1;
        Ok(())
    }

    /// Updates the context with the read bytes.
    ///
    /// This function is a short-cut version of [`update`] that can be used when
    /// [`Source`] already have a WAV-file-like sample buffer.
    ///
    /// [`update`]: Self::update
    ///
    /// # Errors
    ///
    /// This function currently does not return an error.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    ///
    /// let mut ctx = Context::new(16, 2);
    /// let mut ctx_ref = Context::new(16, 2);
    ///
    /// ctx_ref.update(&[0, 1, 2, 3, 4, 5], 10);
    /// ctx.update_with_le_bytes(
    ///     &[0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
    ///       0x03, 0x00, 0x04, 0x00, 0x05, 0x00],
    ///     10);
    ///
    /// assert_eq!(ctx.md5_digest(), ctx_ref.md5_digest());
    /// ```
    pub fn update_with_le_bytes(
        &mut self,
        packed_samples: &[u8],
        block_size: usize,
    ) -> Result<(), SourceError> {
        self.md5.consume(packed_samples);
        let block_byte_count = block_size * self.channels * self.bytes_per_sample;
        if packed_samples.len() < block_byte_count {
            self.md5
                .consume(vec![0u8; block_byte_count - packed_samples.len()]);
        }
        self.sample_count += packed_samples.len() / self.channels / self.bytes_per_sample;
        self.frame_count += 1;
        Ok(())
    }

    /// Returns the count of the last frame loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    ///
    /// let mut ctx = Context::new(16, 2);
    /// assert!(ctx.current_frame_number().is_none());
    ///
    /// ctx.update(&[0, -1, -2, 3], 16);
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
    /// # use flacenc::source::*;
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
        self.md5.clone().compute().into()
    }

    /// Returns the number of samples loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::source::*;
    ///
    /// let mut ctx = Context::new(16, 2);
    ///
    /// ctx.update(&[0, -1, -2, 3], 30);
    /// assert_eq!(ctx.total_samples(), 2);
    /// ```
    #[inline]
    pub fn total_samples(&self) -> usize {
        self.sample_count
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digest = format!("{:x}", self.md5.clone().compute());
        f.debug_struct("Context")
            .field("bytes_per_sample", &self.bytes_per_sample)
            .field("channels", &self.channels)
            .field("sample_count", &self.sample_count)
            .field("frame_count", &self.frame_count)
            .field("md5", &digest)
            .finish()
    }
}

/// Trait representing the input stream for the encoder.
pub trait Source {
    /// Returns the number of channels.
    fn channels(&self) -> usize;
    /// Returns the number of bits per sample;
    fn bits_per_sample(&self) -> usize;
    /// Returns sampling rate in Hz.
    fn sample_rate(&self) -> usize;
    /// Reads samples to [`FrameBuf`].
    ///
    /// Typical implementation of this function must do two things:
    ///
    /// 1.  Call [`dest.fill_from_interleaved`] for loading samples to
    ///     `FrameBuf`.
    /// 2.  Either [`context.update`] or [`context.update_with_le_bytes`] for
    ///     updating the load context information.
    ///
    /// and returns the number of per-channel samples read.
    ///
    /// [`dest.fill_from_interleaved`]: FrameBuf::fill_from_interleaved
    /// [`context.update`]: Context::update
    /// [`context.update_with_le_bytes]: Context::update_with_le_bytes
    ///
    /// # Errors
    ///
    /// This function can return [`SourceError`] when read is failed.
    fn read_samples(
        &mut self,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError>;
    /// Returns length of source if it's finite and defined.
    fn len_hint(&self) -> Option<usize> {
        None
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
    fn read_samples_from(
        &mut self,
        offset: usize,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError>;
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
    /// # use flacenc::source::*;
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
    /// # use flacenc::source::*;
    ///
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

    fn read_samples(
        &mut self,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError> {
        self.read_samples_from(self.read_head, dest, context)
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl Seekable for MemSource {
    fn len(&self) -> usize {
        self.samples.len() / self.channels()
    }

    fn read_samples_from(
        &mut self,
        offset: usize,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError> {
        if dest.channels() != self.channels {
            return Err(SourceError::by_reason(SourceErrorReason::InvalidBuffer));
        }
        let to_read = dest.size() * self.channels;
        let begin = std::cmp::min(offset * self.channels, self.samples.len());
        let end = std::cmp::min(offset * self.channels + to_read, self.samples.len());
        let src = &self.samples[begin..end];

        dest.fill_from_interleaved(src);
        if !src.is_empty() {
            context.update(src, dest.size())?;
        }
        self.read_head += dest.size();
        Ok(src.len() / self.channels)
    }
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::nursery, clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn simple_deinterleave() {
        let interleaved = [0, 0, -1, -2, 1, 2, -3, 6];
        let mut dest = vec![0i32; interleaved.len()];
        deinterleave(&interleaved, 2, &mut dest);
        assert_eq!(&dest, &[0, -1, 1, -3, 0, -2, 2, 6]);
    }

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
        let mut framebuf = FrameBuf::with_size(channels, block_size);
        let mut ctx = Context::new(16, channels);
        let read = src
            .read_samples_from(0, &mut framebuf, &mut ctx)
            .expect("Read error");
        assert_eq!(read, block_size);

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
        let mut ctx = Context::new(16, channels);
        let mut framebuf = FrameBuf::with_size(channels, block_size);

        for step in 0..8 {
            let read = src
                .read_samples(&mut framebuf, &mut ctx)
                .expect("Read error");
            assert_eq!(read, 128);
            assert_eq!(src.read_head, 128 * (step + 1));
            for t in 0..block_size {
                assert_eq!(framebuf.channel_slice(0)[t], (block_size * step + t) as i32);
                assert_eq!(
                    framebuf.channel_slice(1)[t],
                    -((block_size * step + t) as i32)
                );
            }
        }
        let read = src
            .read_samples(&mut framebuf, &mut ctx)
            .expect("Read error");
        assert_eq!(read, 76);
        for t in 0..76 {
            assert_eq!(framebuf.channel_slice(0)[t], (1024 + t) as i32);
            assert_eq!(framebuf.channel_slice(1)[t], -((1024 + t) as i32));
            assert_eq!(framebuf.channel_slice(2)[t], -((1024 + t) as i32));
        }
    }

    #[test]
    fn md5_computation() {
        let mut ctx = Context::new(16, 2);
        ctx.update(&[0i32; 32 * 2], 32).expect("update failed");

        // Reference computed with Python's hashlib.
        assert_eq!(
            ctx.md5_digest(),
            [
                0xF0, 0x9F, 0x35, 0xA5, 0x63, 0x78, 0x39, 0x45, 0x8E, 0x46, 0x2E, 0x63, 0x50, 0xEC,
                0xBC, 0xE4
            ]
        );

        let mut ctx = Context::new(16, 2);
        ctx.update(&[0xABCDi32; 32 * 2], 32).expect("update failed");
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
