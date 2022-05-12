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

//! Controller connecting coding algorithms.

use std::cell::RefCell;

use super::component::BitRepr;
use super::component::ChannelAssignment;
use super::component::Constant;
use super::component::FixedLpc;
use super::component::Frame;
use super::component::Lpc;
use super::component::Residual;
use super::component::Stream;
use super::component::StreamInfo;
use super::component::SubFrame;
use super::component::Verbatim;
use super::config;
use super::lpc;
use super::rice;
use super::source::FrameBuf;
use super::source::Source;

/// Returns true if samples are all same.
pub fn is_constant<T: PartialEq>(samples: &[T]) -> bool {
    for t in 1..samples.len() {
        if samples[0] != samples[t] {
            return false;
        }
    }
    true
}

/// Constructs `Residual` component given the error signal.
pub fn encode_residual(config: &config::Prc, errors: &[i32], warmup_length: usize) -> Residual {
    let block_size = errors.len();
    let prc_p = rice::find_partitioned_rice_parameter(errors, warmup_length, config.max_parameter);
    let nparts = 1 << prc_p.order;
    let part_size = errors.len() / nparts;

    let mut quotinents = vec![0u32; block_size];
    let mut remainders = vec![0u32; block_size];

    for (p, rice_p) in prc_p.ps.iter().enumerate().take(nparts) {
        let start = std::cmp::max(p * part_size, warmup_length);
        let end = (p + 1) * part_size;
        let remainder_mask = (1u32 << rice_p) - 1;

        for t in start..end {
            let err = rice::encode_signbit(errors[t]);
            quotinents[t] = if t < warmup_length { 0 } else { err >> rice_p };
            remainders[t] = if t < warmup_length {
                0
            } else {
                err & remainder_mask
            };
        }
    }
    Residual::new(
        prc_p.order,
        block_size,
        warmup_length,
        &prc_p.ps,
        &quotinents,
        &remainders,
    )
}

/// Pack scalars into `Vec` of `Simd`s.
pub fn pack_into_simd_vec<T, const LANES: usize>(
    src: &[T],
    dest: &mut Vec<std::simd::Simd<T, LANES>>,
) where
    T: std::simd::SimdElement + From<i8>,
    std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
{
    dest.clear();
    let mut v = std::simd::Simd::<T, LANES>::splat(0i8.into());
    for slice in src.chunks(LANES) {
        if slice.len() < LANES {
            v.as_mut_array()[0..slice.len()].copy_from_slice(slice);
            v.as_mut_array()[slice.len()..LANES].fill(0i8.into());
        } else {
            v.as_mut_array().copy_from_slice(slice);
        }
        dest.push(v);
    }
}

/// Unpack slice of `Simd` into `Vec` of elements.
pub fn unpack_simds<T, const LANES: usize>(src: &[std::simd::Simd<T, LANES>], dest: &mut Vec<T>)
where
    T: std::simd::SimdElement + From<i8>,
    std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
{
    dest.resize(src.len() * LANES, 0i8.into());
    let mut offset = 0;
    for v in src {
        let arr = <[T; LANES]>::from(*v);
        dest[offset..offset + LANES].copy_from_slice(&arr);
        offset += LANES;
    }
}

/// Helper struct holding working memory for fixed LPC.
struct FixedLpcEncoder {
    errors: Vec<std::simd::i32x16>,
    /// Length of errors in the number of samples (scalars).
    error_len_in_samples: usize,
    /// Temporary buffer for unpacked error signal.
    unpacked_errors: Vec<i32>,
}

impl FixedLpcEncoder {
    pub const fn new() -> Self {
        Self {
            errors: vec![],
            error_len_in_samples: 0,
            unpacked_errors: vec![],
        }
    }

    /// Constructs `FixedLpc` using the current `self.errors`.
    fn make_subframe(
        &mut self,
        config: &config::Prc,
        warmup_samples: &[i32],
        bits_per_sample: u8,
    ) -> FixedLpc {
        let order = warmup_samples.len();
        unpack_simds(&self.errors, &mut self.unpacked_errors);
        self.unpacked_errors.truncate(self.error_len_in_samples);
        let residual = encode_residual(config, &self.unpacked_errors, order);
        FixedLpc::new(warmup_samples, residual, bits_per_sample as usize)
    }

    /// Performs `e_{t} -= e_{t-1}`, where `e` is unpacked `self.errors`.
    fn increment_error_order(&mut self) {
        let mut carry = 0i32;
        for i in 0..self.errors.len() {
            let mut shifted = self.errors[i].rotate_lanes_right::<1>();
            (shifted[0], carry) = (carry, shifted[0]);
            self.errors[i] -= shifted;
        }
    }

    /// Finds the smalest config for `FixedLpc`.
    pub fn apply(
        &mut self,
        config: &config::SubFrameCoding,
        signal: &[i32],
        bits_per_sample: u8,
        baseline_bits: usize,
    ) -> Option<SubFrame> {
        let mut minimizer = None;
        let mut min_bits = baseline_bits;

        pack_into_simd_vec(signal, &mut self.errors);
        self.error_len_in_samples = signal.len();

        for order in 0..=4 {
            let subframe: SubFrame = self
                .make_subframe(&config.prc, &signal[0..order], bits_per_sample)
                .into();
            let bits = subframe.count_bits();
            if bits < min_bits {
                minimizer = Some(subframe);
                min_bits = bits;
            }
            self.increment_error_order();
        }
        minimizer
    }
}

thread_local! {
    /// Global (thread-local) working buffer for fixed LPC.
    static FIXED_LPC_ENCODER: RefCell<FixedLpcEncoder> = RefCell::new(FixedLpcEncoder::new());
}

/// Tries 0..4-th order fixed LPC and returns the smallest `SubFrame`.
///
/// # Panics
///
/// The current implementation may cause overflow error if `bits_per_sample` is
/// larger than 29. Therefore, it panics when `bits_per_sample` is larger than
/// this.
pub fn fixed_lpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
    baseline_bits: usize,
) -> Option<SubFrame> {
    assert!(bits_per_sample < 30);
    FIXED_LPC_ENCODER.with(|encoder| {
        encoder
            .borrow_mut()
            .apply(config, signal, bits_per_sample, baseline_bits)
    })
}

/// Estimates the optimal LPC coefficients and returns `SubFrame`s with these.
///
/// # Panics
///
/// It panics if `signal` is shorter than `MAX_LPC_ORDER_PLUS_1`.
pub fn estimated_qlpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
) -> SubFrame {
    let mut errors = vec![0i32; signal.len()];
    let qlpc = lpc::qlpc(
        config.qlpc.lpc_order,
        config.qlpc.quant_precision,
        signal,
        &mut errors,
    );
    let residual = encode_residual(&config.prc, &errors, qlpc.order());
    Lpc::new(
        &signal[0..qlpc.order()],
        qlpc,
        residual,
        bits_per_sample as usize,
    )
    .into()
}

/// Finds the best method to encode the given samples, and returns `SubFrame`.
pub fn encode_subframe(
    config: &config::SubFrameCoding,
    samples: &[i32],
    bits_per_sample: u8,
) -> SubFrame {
    if config.use_constant && is_constant(samples) {
        // Assuming constant is always best if it's applicable.
        Constant::new(samples[0], bits_per_sample).into()
    } else {
        let verb: SubFrame = Verbatim::from_samples(samples, bits_per_sample).into();
        let baseline_bits = verb.count_bits();
        let fixed = if config.use_fixed {
            fixed_lpc(config, samples, bits_per_sample, baseline_bits)
        } else {
            None
        };

        let baseline_bits = fixed.as_ref().map_or(baseline_bits, |x| {
            std::cmp::min(baseline_bits, x.count_bits())
        });
        let est_lpc = if config.use_lpc {
            let candidate = estimated_qlpc(config, samples, bits_per_sample);
            if candidate.count_bits() < baseline_bits {
                Some(candidate)
            } else {
                None
            }
        } else {
            None
        };

        est_lpc.or(fixed).unwrap_or(verb)
    }
}

/// Encode frame with the given channel assignment.
pub fn encode_frame_impl(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    offset: usize,
    stream_info: &StreamInfo,
    ch_info: &ChannelAssignment,
) -> Frame {
    let nchannels = stream_info.channels();
    let bits_per_sample = stream_info.bits_per_sample();
    let mut frame = Frame::new(ch_info.clone(), offset, framebuf.size());
    for ch in 0..nchannels {
        frame.add_subframe(encode_subframe(
            &config.subframe_coding,
            framebuf.channel_slice(ch),
            (bits_per_sample + ch_info.bits_per_sample_offset(ch)) as u8,
        ));
    }
    frame
}

/// Helper struct holding working memory for stereo coding (mid-side).
struct StereoCodingHelper {
    midside_framebuf: FrameBuf,
}

impl StereoCodingHelper {
    pub fn new() -> Self {
        Self {
            midside_framebuf: FrameBuf::with_size(2, 4096),
        }
    }

    pub fn apply(
        &mut self,
        config: &config::Encoder,
        framebuf: &FrameBuf,
        indep: Frame,
        offset: usize,
        stream_info: &StreamInfo,
    ) -> Frame {
        let mut stereo_frame = None;
        let mut min_bits = indep.count_bits();
        let envelope_bits = indep.header().count_bits() + 16;

        self.midside_framebuf.resize(framebuf.size());

        for t in 0..framebuf.size() {
            let l = framebuf.channel_slice(0)[t];
            let r = framebuf.channel_slice(1)[t];
            let (ch0, ch1) = ((l + r) >> 1, l - r);
            self.midside_framebuf.channel_slice_mut(0)[t] = ch0;
            self.midside_framebuf.channel_slice_mut(1)[t] = ch1;
        }

        let ms_frame = encode_frame_impl(
            config,
            &self.midside_framebuf,
            offset,
            stream_info,
            &ChannelAssignment::MidSide,
        );

        let combinations = [
            (
                config.stereo_coding.use_leftside,
                ChannelAssignment::LeftSide,
                indep.subframe(0),
                ms_frame.subframe(1),
            ),
            (
                config.stereo_coding.use_rightside,
                ChannelAssignment::RightSide,
                ms_frame.subframe(1),
                indep.subframe(1),
            ),
            (
                config.stereo_coding.use_midside,
                ChannelAssignment::MidSide,
                ms_frame.subframe(0),
                ms_frame.subframe(1),
            ),
        ];

        for (allowed, ch_info, subframe0, subframe1) in &combinations {
            if !allowed {
                continue;
            }
            let bits = envelope_bits + subframe0.count_bits() + subframe1.count_bits();
            if bits < min_bits {
                let mut header = ms_frame.header().clone();
                header.reset_channel_assignment(ch_info.clone());
                min_bits = bits;
                stereo_frame = Some(Frame::from_parts(
                    header,
                    [*subframe0, *subframe1].into_iter().cloned(),
                ));
            }
        }
        stereo_frame.unwrap_or(indep)
    }
}

thread_local! {
    /// Global (thread-local) working buffer for stereo coding algorithms.
    static STEREO_CODING_HELPER: RefCell<StereoCodingHelper> = RefCell::new(StereoCodingHelper::new());
}

/// Finds the best configuration for encoding samples and returns a `Frame`.
pub fn encode_frame(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    offset: usize,
    stream_info: &StreamInfo,
) -> Frame {
    let nchannels = stream_info.channels();
    let ch_info = ChannelAssignment::Independent(nchannels as u8);
    let mut ret = encode_frame_impl(config, framebuf, offset, stream_info, &ch_info);

    if nchannels == 2 {
        ret = STEREO_CODING_HELPER.with(|helper| {
            helper
                .borrow_mut()
                .apply(config, framebuf, ret, offset, stream_info)
        });
    }
    ret
}

/// Encoder entry function for fixed block-size encoding.
pub fn encode_with_fixed_block_size<T: Source>(
    config: &config::Encoder,
    mut src: T,
    block_size: usize,
) -> Stream {
    let mut stream = Stream::new(src.sample_rate(), src.channels(), src.bits_per_sample());
    let channels = src.channels();
    let mut framebuf = FrameBuf::with_size(channels, block_size);

    let mut md5_context = md5::Context::new();
    let mut offset: usize = 0;
    loop {
        let read_samples = src.read_samples(&mut framebuf).expect("Loading failed.");
        if read_samples == 0 {
            break;
        }
        // For the last frame, FLAC uses all the samples in the zero-padded block.
        framebuf.update_md5(src.bits_per_sample(), block_size, &mut md5_context);
        let frame = encode_frame(config, &framebuf, offset, stream.stream_info());
        stream.add_frame(frame);
        offset += read_samples;
    }
    stream
        .stream_info_mut()
        .set_total_samples(src.len_hint().unwrap_or(offset));
    stream
        .stream_info_mut()
        .set_md5_digest(&md5_context.compute().into());
    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source;
    use crate::test_helper;

    #[test]
    fn constant_detector() {
        let signal = vec![5; 64];
        assert!(super::is_constant(&signal));

        let signal = vec![-3; 192];
        assert!(super::is_constant(&signal));
    }

    #[test]
    fn fixed_lpc_encoder() {
        let mut encoder = FixedLpcEncoder::new();
        let signal = test_helper::sinusoid_plus_noise(64, 32, 10000.0, 128);
        pack_into_simd_vec(&signal, &mut encoder.errors);
        encoder.error_len_in_samples = signal.len();
        encoder.increment_error_order();
        let mut unpacked = vec![];
        unpack_simds(&encoder.errors, &mut unpacked);
        for t in 1..signal.len() {
            assert_eq!(unpacked[t], signal[t] - signal[t - 1]);
        }
        encoder.increment_error_order();
        unpack_simds(&encoder.errors, &mut unpacked);
        for t in 2..signal.len() {
            assert_eq!(unpacked[t], signal[t] - 2 * signal[t - 1] + signal[t - 2]);
        }
    }

    #[test]
    fn md5_invariance() {
        let channels = 2;
        let bits_per_sample = 24;
        let sample_rate = 16000;
        let block_size = 128;
        let constant = 23;
        let signal_len = 1024;
        let signal = test_helper::constant_plus_noise(signal_len * channels, constant, 0);
        let source =
            source::PreloadedSignal::from_samples(&signal, channels, bits_per_sample, sample_rate);
        let stream = encode_with_fixed_block_size(&config::Encoder::default(), source, block_size);
        eprintln!(
            "MD5 of DC signal ({}) with len={} and ch={} was",
            constant, signal_len, channels
        );
        eprint!("[");
        for &b in stream.stream_info().md5() {
            eprint!("0x{:02X}, ", b);
        }
        eprintln!("]");
        assert_eq!(
            stream.stream_info().md5(),
            &[
                0xEE, 0x78, 0x7A, 0x6E, 0x99, 0x01, 0x36, 0x79, 0xA5, 0xBB, 0x6D, 0x5C, 0x10, 0xAF,
                0x0B, 0x87
            ]
        );
    }
}
