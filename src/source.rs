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
pub fn deinterleave(interleaved: &[i32], channels: usize, dest: &mut [i32]) {
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
    pub fn with_size(channels: usize, size: usize) -> Self {
        Self {
            samples: vec![0i32; size * channels],
            channels,
            size,
        }
    }

    /// Returns the size in the number of inter-channel samples.
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Resizes `FrameBuf`.
    pub fn resize(&mut self, new_size: usize) {
        self.size = new_size;
        self.samples.resize(self.size * self.channels, 0i32);
    }

    /// Fill first samples from the interleaved slice, and resets rest.
    pub fn fill_from_interleaved(&mut self, interleaved: &[i32]) {
        deinterleave(interleaved, self.channels, &mut self.samples);
    }

    /// Returns the number of channels
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Returns samples from the given channel.
    pub fn channel_slice(&self, ch: usize) -> &[i32] {
        &self.samples[ch * self.size..(ch + 1) * self.size]
    }

    /// Returns mutable samples from the given channel.
    pub fn channel_slice_mut(&mut self, ch: usize) -> &mut [i32] {
        &mut self.samples[ch * self.size..(ch + 1) * self.size]
    }

    /// Returns the internal representation of multichannel signals.
    #[cfg(test)]
    pub fn raw_slice(&self) -> &[i32] {
        &self.samples
    }
}

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
    /// This function is typically called from `Source::read_samples`.
    ///
    /// # Errors
    ///
    /// This function currently does not return an error.
    pub fn update(&mut self, interleaved: &[i32], block_size: usize) -> Result<(), SourceError> {
        self.update_md5(interleaved, block_size);
        self.sample_count += interleaved.len() / self.channels;
        self.frame_count += 1;
        Ok(())
    }

    /// Updates the context with the read bytes.
    ///
    /// This function is a short-cut version of `update` that can be used when
    /// `Source` already have a WAV-file-like sample buffer.
    ///
    /// # Errors
    ///
    /// This function currently does not return an error.
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
    /// # Panics
    ///
    /// This panics when it is called before `update` is called (typically via
    /// `Source::read_samples`) at least once.
    #[inline]
    pub fn current_frame_number(&self) -> usize {
        assert!(self.frame_count > 0);
        self.frame_count - 1
    }

    /// Returns MD5 digest of the consumed samples.
    #[inline]
    pub fn md5_digest(&self) -> [u8; 16] {
        self.md5.clone().compute().into()
    }

    /// Returns the number of samples consumed.
    #[inline]
    #[allow(clippy::missing_const_for_fn)] // for API robustness.
    pub fn total_samples(&self) -> usize {
        self.sample_count
    }
}

pub trait Source {
    /// Returns the number of channels.
    fn channels(&self) -> usize;
    /// Returns the number of bits per sample;
    fn bits_per_sample(&self) -> usize;
    /// Returns sampling rate in Hz.
    fn sample_rate(&self) -> usize;
    /// Reads samples to the buffer.
    #[allow(clippy::missing_errors_doc)]
    fn read_samples(
        &mut self,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError>;
    /// Returns length of source if it's defined.
    fn len_hint(&self) -> Option<usize> {
        None
    }
}

pub trait Seekable: Source {
    /// Returns `true` if the source contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the length in inter-channel samples
    fn len(&self) -> usize;

    /// Seeks to the specified offset from the beginning.
    #[allow(clippy::missing_errors_doc)]
    fn read_samples_from(
        &mut self,
        offset: usize,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError>;
}

/// Source with preloaded samples.
#[derive(Clone, Debug)]
pub struct PreloadedSignal {
    pub channels: usize,
    pub bits_per_sample: usize,
    pub sample_rate: usize,
    pub samples: Vec<i32>,
    pub read_head: usize,
}

impl PreloadedSignal {
    /// Constructs `PreloadedSignal` from samples.
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
    pub fn as_raw_slice(&self) -> &[i32] {
        &self.samples
    }
}

impl Source for PreloadedSignal {
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

impl Seekable for PreloadedSignal {
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
    fn reading_and_deinterleaving() {
        let mut signal = vec![];
        let block_size = 512;
        let channels = 4;
        for t in 0..block_size {
            for _ch in 0..channels {
                signal.push(t as i32);
            }
        }

        let mut src = PreloadedSignal::from_samples(&signal, channels, 16, 16000);
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
        let mut src = PreloadedSignal::from_samples(&signal, channels, 16, 16000);
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
