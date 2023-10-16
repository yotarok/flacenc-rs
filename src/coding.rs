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

use super::arrayutils::is_constant;
use super::arrayutils::pack_into_simd_vec;
use super::arrayutils::unpack_simds;
use super::component::BitRepr;
use super::component::ChannelAssignment;
use super::component::Constant;
use super::component::FixedLpc;
use super::component::Frame;
use super::component::FrameHeader;
use super::component::Lpc;
use super::component::Residual;
use super::component::Stream;
use super::component::StreamInfo;
use super::component::SubFrame;
use super::component::Verbatim;
use super::config;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::error::EncodeError;
use super::lpc;
#[cfg(feature = "par")]
use super::par;
use super::rice;
use super::source::Context;
use super::source::FrameBuf;
use super::source::Source;
use crate::reusable;
use crate::reuse;

#[cfg(feature = "fakesimd")]
use super::fakesimd as simd;
#[cfg(not(feature = "fakesimd"))]
use std::simd;

/// Computes rice encoding of a scalar (used in `encode_residual`.)
#[inline]
const fn quotients_and_remainders(err: i32, rice_p: u8) -> (u32, u32) {
    let remainder_mask = (1u32 << rice_p) - 1;
    let err = rice::encode_signbit(err);
    (err >> rice_p, err & remainder_mask)
}

/// Computes rice encoding of a SIMD vector (used in `encode_residual`.)
#[inline]
#[cfg(not(feature = "fakesimd"))]
fn quotients_and_remainders_simd<const N: usize>(
    err_v: simd::Simd<i32, N>,
    rice_p: u8,
    quotients: &mut [u32],
    remainders: &mut [u32],
) where
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    let rice_p_v = simd::Simd::splat(u32::from(rice_p));
    let remainder_mask_v = simd::Simd::splat((1u32 << rice_p) - 1);
    let err_v = rice::encode_signbit_simd(err_v);
    quotients.copy_from_slice((err_v >> rice_p_v).as_ref());
    remainders.copy_from_slice((err_v & remainder_mask_v).as_ref());
}

/// Computes encoding of each residual partition.
///
/// This function is moved out from the main loop for avoiding messy conditoinal
/// compilation due to fakesimd. We had to resort conditional compilation
/// because `as_simd` operation that is provided as an extension to standard
/// types (slice/ Vec) is still there even if we use stable version, so the
/// approach we used in "fakesimd" is not suitable for mimicking this.
///
/// TODO: Probably, it's better to introduce another abstraction for `as_simd`
/// e.g. SIMD-version of `map` so we can do conditional compilation there.
#[cfg(not(feature = "fakesimd"))]
#[inline]
fn encode_residual_partition(
    start: usize,
    end: usize,
    rice_p: u8,
    errors: &[i32],
    quotients: &mut [u32],
    remainders: &mut [u32],
) {
    const SIMD_N: usize = 8;
    // note that t >= warmup_length because start >= warmup_length.
    let mut t = start;
    let (head, body, tail) = errors[start..end].as_simd::<SIMD_N>();
    for err in head {
        (quotients[t], remainders[t]) = quotients_and_remainders(*err, rice_p);
        t += 1;
    }
    for err_v in body {
        quotients_and_remainders_simd::<SIMD_N>(
            *err_v,
            rice_p,
            &mut quotients[t..t + SIMD_N],
            &mut remainders[t..t + SIMD_N],
        );
        t += SIMD_N;
    }
    for err in tail {
        (quotients[t], remainders[t]) = quotients_and_remainders(*err, rice_p);
        t += 1;
    }
}

/// Computes encoding of each residual partition. (without SIMD)
#[cfg(feature = "fakesimd")]
#[inline]
fn encode_residual_partition(
    start: usize,
    end: usize,
    rice_p: u8,
    errors: &[i32],
    quotients: &mut [u32],
    remainders: &mut [u32],
) {
    let mut t = start;
    for err in &errors[start..end] {
        (quotients[t], remainders[t]) = quotients_and_remainders(*err, rice_p);
        t += 1;
    }
}

/// Constructs `Residual` component given the error signal.
fn encode_residual(config: &config::Prc, errors: &[i32], warmup_length: usize) -> Residual {
    let block_size = errors.len();
    let prc_p = rice::find_partitioned_rice_parameter(errors, warmup_length, config.max_parameter);
    let nparts = 1 << prc_p.order;
    let part_size = errors.len() / nparts;

    let mut quotients = vec![0u32; block_size];
    let mut remainders = vec![0u32; block_size];

    for (p, rice_p) in prc_p.ps.iter().enumerate().take(nparts) {
        let start = std::cmp::max(p * part_size, warmup_length);
        let end = (p + 1) * part_size;
        // ^ this is okay because partitions are larger than warmup_length
        encode_residual_partition(start, end, *rice_p, errors, &mut quotients, &mut remainders);
    }
    Residual::from_parts(
        prc_p.order as u8,
        block_size,
        warmup_length,
        prc_p.ps,
        quotients,
        remainders,
    )
}

/// Helper struct holding working memory for fixed LPC.
#[derive(Default)]
struct FixedLpcEncoder {
    errors: Vec<simd::i32x16>,
    /// Length of errors in the number of samples (scalars).
    error_len_in_samples: usize,
    /// Temporary buffer for unpacked error signal.
    unpacked_errors: Vec<i32>,
}

impl FixedLpcEncoder {
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

    /// Finds the smallest config for `FixedLpc`.
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

reusable!(FIXED_LPC_ENCODER: FixedLpcEncoder);

/// Tries 0..4-th order fixed LPC and returns the smallest `SubFrame`.
///
/// # Panics
///
/// The current implementation may cause overflow error if `bits_per_sample` is
/// larger than 29. Therefore, it panics when `bits_per_sample` is larger than
/// this.
fn fixed_lpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
    baseline_bits: usize,
) -> Option<SubFrame> {
    assert!(bits_per_sample < 30);
    reuse!(FIXED_LPC_ENCODER, |encoder: &mut FixedLpcEncoder| {
        encoder.apply(config, signal, bits_per_sample, baseline_bits)
    })
}

fn perform_qlpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
) -> heapless::Vec<f32, MAX_LPC_ORDER> {
    if config.qlpc.use_direct_mse {
        if config.qlpc.mae_optimization_steps > 0 {
            lpc::lpc_with_irls_mae(
                signal,
                &config.qlpc.window,
                config.qlpc.lpc_order,
                config.qlpc.mae_optimization_steps,
            )
        } else {
            lpc::lpc_with_direct_mse(signal, &config.qlpc.window, config.qlpc.lpc_order)
        }
    } else {
        lpc::lpc_from_autocorr(signal, &config.qlpc.window, config.qlpc.lpc_order)
    }
}

/// Estimates the optimal LPC coefficients and returns `SubFrame`s with these.
///
/// # Panics
///
/// It panics if `signal` is shorter than `MAX_LPC_ORDER_PLUS_1`.
fn estimated_qlpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
) -> SubFrame {
    let mut errors = vec![0i32; signal.len()];
    let lpc_order = config.qlpc.lpc_order;
    let lpc_coefs = perform_qlpc(config, signal);
    let qlpc =
        lpc::QuantizedParameters::with_coefs(&lpc_coefs[0..lpc_order], config.qlpc.quant_precision);
    qlpc.compute_error(signal, &mut errors);

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
fn encode_subframe(
    config: &config::SubFrameCoding,
    samples: &[i32],
    bits_per_sample: u8,
) -> SubFrame {
    if config.use_constant && is_constant(samples) {
        // Assuming constant is always best if it's applicable.
        Constant::new(samples[0], bits_per_sample).into()
    } else {
        let baseline_bits =
            Verbatim::count_bits_from_metadata(samples.len(), bits_per_sample as usize);

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
            (candidate.count_bits() < baseline_bits).then_some(candidate)
        } else {
            None
        };

        est_lpc
            .or(fixed)
            .unwrap_or_else(|| Verbatim::from_samples(samples, bits_per_sample).into())
    }
}

/// Encode frame with the given channel assignment.
fn encode_frame_impl(
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

    // for Claxon compatibility.
    frame.header_mut().reset_sample_size(stream_info);

    frame
}

// Recombines stereo frame.
#[inline]
fn recombine_stereo_frame(header: FrameHeader, indep: Frame, ms: Frame) -> Frame {
    let (_header, l, r) = indep
        .into_stereo_channels()
        .expect(panic_msg::DATA_INCONSISTENT);
    let (_header, m, s) = ms
        .into_stereo_channels()
        .expect(panic_msg::DATA_INCONSISTENT);

    let chans = header.channel_assignment().select_channels(l, r, m, s);
    Frame::from_parts(
        header,
        <(SubFrame, SubFrame) as Into<[SubFrame; 2]>>::into(chans).into_iter(),
    )
}

reusable!(MSFRAMEBUF: FrameBuf = FrameBuf::with_size(2, 4096));

/// Tries several stereo channel recombinations and returns the best.
fn try_stereo_coding(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    indep: Frame,
    offset: usize,
    stream_info: &StreamInfo,
) -> Frame {
    reuse!(MSFRAMEBUF, |ms_framebuf: &mut FrameBuf| {
        ms_framebuf.resize(framebuf.size());

        for t in 0..framebuf.size() {
            let l = framebuf.channel_slice(0)[t];
            let r = framebuf.channel_slice(1)[t];
            let (ch0, ch1) = ((l + r) >> 1, l - r);
            ms_framebuf.channel_slice_mut(0)[t] = ch0;
            ms_framebuf.channel_slice_mut(1)[t] = ch1;
        }

        let ms_frame = encode_frame_impl(
            config,
            ms_framebuf,
            offset,
            stream_info,
            &ChannelAssignment::MidSide,
        );

        let (bits_l, bits_r, bits_m, bits_s) = (
            indep.subframe(0).unwrap().count_bits(),
            indep.subframe(1).unwrap().count_bits(),
            ms_frame.subframe(0).unwrap().count_bits(),
            ms_frame.subframe(1).unwrap().count_bits(),
        );

        let combinations = [
            config
                .stereo_coding
                .use_leftside
                .then_some((ChannelAssignment::LeftSide, bits_l + bits_s)),
            config
                .stereo_coding
                .use_rightside
                .then_some((ChannelAssignment::RightSide, bits_r + bits_s)),
            config
                .stereo_coding
                .use_midside
                .then_some((ChannelAssignment::MidSide, bits_m + bits_s)),
        ];

        let envelope_bits = indep.header().count_bits() + 16;
        let mut min_bits = indep.count_bits();
        let mut min_ch_info = ChannelAssignment::Independent(2);
        for (ch_info, body_bits) in combinations.iter().flatten() {
            let bits = envelope_bits + body_bits;
            if bits < min_bits {
                min_bits = bits;
                min_ch_info = ch_info.clone();
            }
        }
        let mut header = ms_frame.header().clone();
        header.reset_channel_assignment(min_ch_info);
        recombine_stereo_frame(header, indep, ms_frame)
    })
}

/// Finds the best configuration for encoding samples and returns a `Frame`.
fn encode_frame(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    offset: usize,
    stream_info: &StreamInfo,
) -> Frame {
    let nchannels = stream_info.channels();
    let ch_info = ChannelAssignment::Independent(nchannels as u8);
    let mut ret = encode_frame_impl(config, framebuf, offset, stream_info, &ch_info);

    if nchannels == 2 {
        ret = try_stereo_coding(config, framebuf, ret, offset, stream_info);
    }
    ret
}

/// Encodes [`FrameBuf`] to [`Frame`].
///
/// # Errors
///
/// This function currently doesn't return error. However, in future this can
/// emit an error when `config` is not verified and in inconsistent state.
/// Currently, behavior when `config` is errorneous is undefined.
///
/// # Examples
///
/// ```
/// # use flacenc::*;
/// use flacenc::config;
/// use flacenc::component::StreamInfo;
/// use flacenc::source::{Context, FrameBuf, MemSource, Source};
///
/// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
/// let signal = vec![0i32; signal_len * channels];
/// let bits_per_sample = 16;
///
/// let mut source = MemSource::from_samples(&signal, channels, bits_per_sample, sample_rate);
/// let mut fb = FrameBuf::with_size(channels, block_size);
/// let stream_info = StreamInfo::new(sample_rate, channels, bits_per_sample);
/// assert!(source.read_samples(block_size, &mut fb).is_ok());
///
/// let frame = encode_fixed_size_frame(
///     &config::Encoder::default(), // block-size in config will be overridden.
///     &fb,
///     0,
///     &stream_info
/// ).expect("encoder error");
/// ```
pub fn encode_fixed_size_frame(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    frame_number: usize,
    stream_info: &StreamInfo,
) -> Result<Frame, EncodeError> {
    // A bit awkward, but this function is implemented by overwriting relevant
    // fields of `Frame` generated by `encode_frame`.
    let mut ret = encode_frame(config, framebuf, 0, stream_info);
    ret.header_mut().set_frame_number(frame_number as u32);
    Ok(ret)
}

/// Encodes [`Source`] to [`Stream`].
///
/// This is the main entry point of this library crate.
///
/// # Errors
///
/// This function returns [`EncodeError`] that contains a [`SourceError`] when
/// it failed to read samples from `src`.
///
/// [`SourceError`]: crate::error::SourceError
///
/// # Panics
///
/// This function panics only by an internal error.
///
/// # Examples
///
/// ```
/// # use flacenc::*;
/// # #[path = "doctest_helper.rs"]
/// # mod doctest_helper;
/// # use doctest_helper::*;
/// use flacenc::config;
/// use flacenc::source::MemSource;
///
/// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
/// let signal = vec![0i32; signal_len * channels];
/// let bits_per_sample = 16;
/// let source = MemSource::from_samples(&signal, channels, bits_per_sample, sample_rate);
/// let result = encode_with_fixed_block_size(
///     &config::Encoder::default(), source, block_size
/// );
/// assert!(result.is_ok());
/// ```
pub fn encode_with_fixed_block_size<T: Source>(
    config: &config::Encoder,
    mut src: T,
    block_size: usize,
) -> Result<Stream, EncodeError> {
    #[cfg(feature = "par")]
    {
        if config.multithread {
            return par::encode_with_fixed_block_size(config, src, block_size);
        }
    }
    let mut stream = Stream::new(src.sample_rate(), src.channels(), src.bits_per_sample());
    let mut framebuf_and_context = (
        FrameBuf::with_size(src.channels(), block_size),
        Context::new(src.bits_per_sample(), src.channels(), block_size),
    );
    loop {
        let read_samples = src.read_samples(block_size, &mut framebuf_and_context)?;
        if read_samples == 0 {
            break;
        }
        let frame = encode_fixed_size_frame(
            config,
            &framebuf_and_context.0,
            framebuf_and_context.1.current_frame_number().unwrap(),
            stream.stream_info(),
        )?;
        stream.add_frame(frame);
    }

    let (_, context) = framebuf_and_context;
    stream
        .stream_info_mut()
        .set_md5_digest(&context.md5_digest());
    stream
        .stream_info_mut()
        .set_total_samples(src.len_hint().unwrap_or_else(|| context.total_samples()));
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source;
    use crate::test_helper;

    #[test]
    fn fixed_lpc_encoder() {
        let mut encoder = FixedLpcEncoder::default();
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
            source::MemSource::from_samples(&signal, channels, bits_per_sample, sample_rate);
        let stream = encode_with_fixed_block_size(&config::Encoder::default(), source, block_size)
            .expect("Source read error");
        eprintln!("MD5 of DC signal ({constant}) with len={signal_len} and ch={channels} was",);
        eprint!("[");
        for &b in stream.stream_info().md5_digest() {
            eprint!("0x{b:02X}, ");
        }
        eprintln!("]");
        assert_eq!(
            stream.stream_info().md5_digest(),
            &[
                0xEE, 0x78, 0x7A, 0x6E, 0x99, 0x01, 0x36, 0x79, 0xA5, 0xBB, 0x6D, 0x5C, 0x10, 0xAF,
                0x0B, 0x87
            ]
        );
    }
}
