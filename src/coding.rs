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

use super::arrayutils::find_min_and_max;
use super::arrayutils::find_sum_abs_f32;
use super::arrayutils::is_constant;
use super::arrayutils::SimdVec;
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
use super::constant::fixed::MAX_LPC_ORDER as MAX_FIXED_LPC_ORDER;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::MAX_BLOCK_SIZE;
use super::constant::MIN_BLOCK_SIZE;
use super::error::EncodeError;
use super::error::VerifyError;
use super::lpc;
#[cfg(feature = "par")]
use super::par;
use super::rice;
use super::source::Context;
use super::source::FrameBuf;
use super::source::Source;

import_simd!(as simd);

/// Computes rice encoding of a scalar (used in `encode_residual`.)
#[inline]
const fn quotients_and_remainders(err: i32, rice_p: u8) -> (u32, u32) {
    let remainder_mask = (1u32 << rice_p) - 1;
    let err = rice::encode_signbit(err);
    (err >> rice_p, err & remainder_mask)
}

/// Computes rice encoding of a SIMD vector (used in `encode_residual`.)
#[inline]
#[cfg(feature = "simd-nightly")]
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
#[cfg(feature = "simd-nightly")]
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
#[cfg(not(feature = "simd-nightly"))]
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

/// Computes `Residual` from the given error signal and PRC parameters.
fn encode_residual_with_prc_parameter(
    _config: &config::Prc,
    errors: &[i32],
    warmup_length: usize,
    prc_p: rice::PrcParameter,
) -> Residual {
    let block_size = errors.len();
    let nparts = 1 << prc_p.order;
    let part_size = errors.len() >> prc_p.order;
    debug_assert!(part_size >= warmup_length);

    let mut quotients = vec![0u32; block_size];
    let mut remainders = vec![0u32; block_size];

    let mut offset = 0;
    for rice_p in &prc_p.ps[0..nparts] {
        let start = std::cmp::max(offset, warmup_length);
        offset += part_size;
        let end = offset;
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

/// Constructs `Residual` component given the error signal.
fn encode_residual(config: &config::Prc, errors: &[i32], warmup_length: usize) -> Residual {
    let prc_p = rice::find_partitioned_rice_parameter(errors, warmup_length, config.max_parameter);
    encode_residual_with_prc_parameter(config, errors, warmup_length, prc_p)
}

type FixedLpcErrors = [SimdVec<i32, 16>; MAX_FIXED_LPC_ORDER + 1];
reusable!(FIXED_LPC_ERRORS: FixedLpcErrors);

/// Resets `FixedLpcErrors` from the given signal.
fn reset_fixed_lpc_errors(errors: &mut FixedLpcErrors, signal: &[i32]) {
    errors[0].reset_from_slice(signal);

    for order in 0..MAX_FIXED_LPC_ORDER {
        let next_order = order + 1;

        let mut carry = 0i32;
        errors[next_order].resize(signal.len(), simd::Simd::default());
        for t in 0..errors[order].simd_len() {
            let x = errors[order].as_ref_simd()[t];
            let mut shifted = x.rotate_elements_right::<1>();
            (shifted[0], carry) = (carry, shifted[0]);
            errors[next_order].as_mut_simd()[t] = x - shifted;
        }
    }
}

/// Estimate bit count from the error.
fn estimate_entropy(errors: &[i32], warmup_len: usize, partitions: usize) -> usize {
    // this function computes partition average of:
    //   (1 + e) log (1 + e) - e * log e
    // where log-base is 2 and e is the average error.
    // This can further be approximated (by Stirling's formula) as:
    //   log(1 + e) + constant
    // given e >> 1, it can further be approximated as log(e); however we don't
    // use this formula as it is anyway cheap to compute.
    let block_size = errors.len();
    let partition_size = (block_size + partitions - 1) / partitions;

    let mut offset = 0;
    let mut acc = 0;
    for _p in 0..partitions {
        let end = std::cmp::min(block_size, offset + partition_size);
        let partition_len = end - offset;
        if end >= warmup_len {
            let sample_count = std::cmp::min(end - warmup_len, partition_len);
            let sum_errors = find_sum_abs_f32::<16>(&errors[offset..end]);
            let avg_errors = sum_errors * 2.0 / (sample_count as f32 + 0.00001);
            let geom_p = 1.0 / (avg_errors + 1.0);
            let xent = avg_errors.mul_add(-(1.0 - geom_p).log2(), -geom_p.log2());
            acc += (xent * sample_count as f32) as usize;
        }
        offset = end;
    }
    acc
}

/// Selects the best LPC order from error signals and encode `Residual`.
fn select_order_and_encode_residual<'a, I>(
    order_sel: &config::OrderSel,
    prc_config: &config::Prc,
    errors: I,
    bits_per_sample: usize,
    baseline_bits: usize,
) -> Option<(usize, Residual)>
where
    I: Iterator<Item = (usize, &'a [i32])>,
{
    let max_rice_p = prc_config.max_parameter;
    match *order_sel {
        config::OrderSel::BitCount => errors
            .map(
                #[inline]
                |(order, err)| {
                    let prc_p = rice::find_partitioned_rice_parameter(err, order, max_rice_p);
                    let bits = bits_per_sample * order + prc_p.code_bits;
                    (order, err, prc_p, bits)
                },
            )
            .min_by_key(|(_order, _err, _prc_p, bits)| *bits)
            .and_then(
                #[inline]
                |(order, err, prc_p, bits)| {
                    (bits < baseline_bits).then(
                        #[inline]
                        || {
                            (
                                order,
                                encode_residual_with_prc_parameter(prc_config, err, order, prc_p),
                            )
                        },
                    )
                },
            ),
        config::OrderSel::ApproxEnt { partitions } => errors
            .map(
                #[inline]
                |(order, err)| {
                    (
                        order,
                        err,
                        estimate_entropy(err, order, partitions) + bits_per_sample * order,
                    )
                },
            )
            .min_by_key(
                #[inline]
                |(_order, _err, bits)| *bits,
            )
            .and_then(
                #[inline]
                |(order, err, bits)| {
                    (bits < baseline_bits).then(|| (order, encode_residual(prc_config, err, order)))
                },
            ),
    }
}

/// Tries `0..=4`-th order fixed LPC and returns the smallest `SubFrame`.
///
/// # Panics
///
/// The current implementation may cause overflow error if `bits_per_sample` is
/// larger than 29. Therefore, it panics when `bits_per_sample` is larger than
/// this.
#[inline]
fn fixed_lpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
    baseline_bits: usize,
) -> Option<SubFrame> {
    assert!(bits_per_sample < 30);
    let max_order = config.fixed.max_order;

    reuse!(FIXED_LPC_ERRORS, |errors: &mut FixedLpcErrors| {
        reset_fixed_lpc_errors(errors, signal);
        let errors = errors
            .iter()
            .map(SimdVec::as_ref)
            .take(max_order + 1)
            .enumerate();
        select_order_and_encode_residual(
            &config.fixed.order_sel,
            &config.prc,
            errors,
            bits_per_sample as usize,
            baseline_bits,
        )
        .map(|(order, residual)| {
            FixedLpc::from_parts(
                heapless::Vec::from_slice(&signal[..order])
                    .expect("Exceeded maximum order for FixedLpc component."),
                residual,
                bits_per_sample,
            )
            .into()
        })
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

reusable!(QLPC_ERROR_BUFFER: Vec<i32>);

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
    let lpc_order = config.qlpc.lpc_order;
    let lpc_coefs = perform_qlpc(config, signal);
    let qlpc = lpc::quantize_parameters(&lpc_coefs[0..lpc_order], config.qlpc.quant_precision);
    let residual = reuse!(QLPC_ERROR_BUFFER, |errors: &mut Vec<i32>| {
        errors.resize(signal.len(), 0i32);
        lpc::compute_error(&qlpc, signal, errors);
        encode_residual(&config.prc, errors, qlpc.order())
    });
    Lpc::from_parts(
        heapless::Vec::from_slice(&signal[0..qlpc.order()])
            .expect("LPC order exceeded the maximum"),
        qlpc,
        residual,
        bits_per_sample,
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
        Constant::from_parts(samples.len(), samples[0], bits_per_sample).into()
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
    frame.header_mut().reset_sample_size_spec(stream_info);

    frame
}

// Recombines stereo frame.
#[allow(clippy::tuple_array_conversions)] // recommended conversion methods are not supported in MSRV
#[inline]
fn recombine_stereo_frame(header: FrameHeader, indep: Frame, ms: Frame) -> Frame {
    let (_header, l, r) = indep
        .into_stereo_channels()
        .expect(panic_msg::DATA_INCONSISTENT);
    let (_header, m, s) = ms
        .into_stereo_channels()
        .expect(panic_msg::DATA_INCONSISTENT);

    let chans = header.channel_assignment().select_channels(l, r, m, s);
    Frame::from_parts(header, [chans.0, chans.1].into_iter())
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

        let mut min_bits = bits_l + bits_r;
        let mut min_ch_info = ChannelAssignment::Independent(2);
        for (ch_info, bits) in combinations.iter().flatten() {
            if *bits < min_bits {
                min_bits = *bits;
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
/// The block size is taken from `FrameBuf::size`.
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
/// let stream_info = StreamInfo::new(sample_rate, channels, bits_per_sample).unwrap();
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
    let block_size = framebuf.size();
    if block_size < MIN_BLOCK_SIZE {
        return Err(VerifyError::new(
            "input.framebuf.size",
            &format!("must be greater than or equal to {MIN_BLOCK_SIZE}"),
        )
        .into());
    }
    if block_size > MAX_BLOCK_SIZE {
        return Err(VerifyError::new(
            "input.framebuf.size",
            &format!("must be lesser than or equal to {MAX_BLOCK_SIZE}"),
        )
        .into());
    }

    let bps = stream_info.bits_per_sample();
    let max_allowed = (1i32 << (bps - 1)) - 1;
    let min_allowed = -(1i32 << (bps - 1));
    for ch in 0..framebuf.channels() {
        let (min, max) = find_min_and_max::<64>(framebuf.channel_slice(ch), 0i32);
        if min < min_allowed || max > max_allowed {
            return Err(VerifyError::new(
                "input.framebuf",
                &format!("input sample must be in the range of bits={bps}"),
            )
            .into());
        }
    }

    // TODO: No update `config.block_sizes` is made as the algorithm below
    // doesn't use `config.block_sizes` in the current implementation.
    // However, this is fragile to extension. A solution that is performant
    // and future-safe is required.

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
    let mut stream = Stream::new(src.sample_rate(), src.channels(), src.bits_per_sample())?;
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
    use crate::component::Decode;
    use crate::error::Verify;
    use crate::sigen;
    use crate::sigen::Signal;
    use crate::source;
    use crate::source::Fill;

    #[test]
    fn fixed_lpc_error_computation() {
        let mut errors = FixedLpcErrors::default();
        let signal = sigen::Sine::new(32, 0.3)
            .noise(0.1)
            .to_vec_quantized(16, 64);
        reset_fixed_lpc_errors(&mut errors, &signal);
        let unpacked = errors[1].as_ref();
        for t in 1..signal.len() {
            assert_eq!(unpacked[t], signal[t] - signal[t - 1]);
        }
        let unpacked = errors[2].as_ref();
        for t in 2..signal.len() {
            assert_eq!(unpacked[t], signal[t] - 2 * signal[t - 1] + signal[t - 2]);
        }
    }

    #[test]
    fn fixed_lpc_of_sine() {
        let signal = sigen::Sine::new(100, 0.6).to_vec_quantized(8, 1024);
        let mut config = config::SubFrameCoding::default();
        for order in 0..=MAX_FIXED_LPC_ORDER {
            config.fixed.max_order = order;
            let subframe = fixed_lpc(&config, &signal, 8, usize::MAX)
                .expect("Should return Some because `baseline_bits` is usize::MAX.");
            subframe.verify().expect("Should return valid subframe.");
            assert_eq!(subframe.decode(), signal);
        }
    }

    #[test]
    fn md5_invariance() {
        let channels = 2;
        let bits_per_sample = 24;
        let sample_rate = 16000;
        let block_size = 128;
        let constant: f32 = (23f64 / f64::from(1 << 23)) as f32;
        let signal_len = 1024;
        let signal =
            sigen::Dc::new(constant).to_vec_quantized(bits_per_sample, signal_len * channels);
        assert_eq!(signal[0], 23);
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

    #[test]
    fn losslessness_residual_coding() {
        let signal = sigen::Noise::new(0.4).to_vec_quantized(8, 64);
        let residual = encode_residual(&config::Prc::default(), &signal, 0);
        let decoded = residual.decode();
        assert_eq!(decoded, signal);

        let signal = sigen::Noise::new(0.9)
            .concat(2048, sigen::Sine::new(40, 0.1))
            .to_vec_quantized(8, 4096);
        let residual = encode_residual(&config::Prc::default(), &signal, 0);
        let decoded = residual.decode();
        assert_eq!(decoded, signal);
    }

    #[test]
    fn losslessness_subframe_coding() {
        let bits_per_sample = 8;
        let config = config::SubFrameCoding::default();
        let signal = sigen::Noise::new(0.4).to_vec_quantized(bits_per_sample, 64);
        let subframe = encode_subframe(&config, &signal, bits_per_sample as u8);
        let decoded = subframe.decode();
        assert_eq!(decoded, signal);

        let signal = sigen::Sine::new(40, 0.9).to_vec_quantized(bits_per_sample, 64);
        let subframe = encode_subframe(&config, &signal, bits_per_sample as u8);
        let decoded = subframe.decode();
        assert_eq!(decoded, signal);
    }

    #[test]
    fn encoding_zeros() {
        let channel_count = 1;
        let block_size = 64;
        let bits_per_sample = 8;
        let sample_rate = 88200;
        let stream_info = StreamInfo::new(sample_rate, channel_count, bits_per_sample).unwrap();
        let mut fb = FrameBuf::with_size(channel_count, block_size);
        fb.fill_interleaved(&vec![0; block_size]).unwrap();
        let frame =
            encode_fixed_size_frame(&config::Encoder::default(), &fb, 0, &stream_info).unwrap();
        frame.verify().unwrap();

        assert_eq!(frame.decode(), vec![0; block_size]);
    }

    #[test]
    fn order_selector_bitcount() {
        let block_size = 256;
        let bits_per_sample = 16;
        let prc_config = config::Prc::default();
        let errors = [
            vec![255i32; block_size],
            vec![256i32; block_size],
            vec![128i32; block_size],
        ];
        let select_result = select_order_and_encode_residual(
            &config::OrderSel::BitCount,
            &prc_config,
            errors.iter().map(AsRef::as_ref).enumerate(),
            bits_per_sample,
            usize::MAX,
        );
        let (selected_order, residual) =
            select_result.expect("should be `Some` because baseline_bits == usize::MAX.");
        residual.verify().expect("should return a valid residual.");

        assert_eq!(selected_order, 0);
        let selected_count = residual.count_bits() + selected_order * bits_per_sample;

        for (order, err) in errors.iter().enumerate() {
            let ref_residual = encode_residual(&prc_config, err, order);
            let ref_count = ref_residual.count_bits() + bits_per_sample * order;
            assert!(
                ref_count >= selected_count,
                "should select the error sequence that minimizes the bit count."
            );
            if order == selected_order {
                assert!(ref_residual.decode() == residual.decode());
            }
        }
    }

    #[test]
    fn order_selector_approxent() {
        let block_size = 256;
        let bits_per_sample = 16;
        let prc_config = config::Prc::default();
        let errors = [
            vec![255i32; block_size],
            vec![256i32; block_size],
            vec![128i32; block_size],
            vec![127i32; block_size],
        ];
        let select_result = select_order_and_encode_residual(
            &config::OrderSel::ApproxEnt { partitions: 32 },
            &prc_config,
            errors.iter().map(AsRef::as_ref).enumerate(),
            bits_per_sample,
            usize::MAX,
        );
        let (selected_order, residual) =
            select_result.expect("should be `Some` because baseline_bits == usize::MAX.");
        residual.verify().expect("should return a valid residual.");

        assert_eq!(selected_order, 2);
    }

    // The block comment below is left intentionally for future when we found
    // another fuzz test failure.
    /*
    #[test]
    fn encode_failed_fuzz_pattern_1() {
        let signals = vec![
            sigen::Sine::with_initial_phase(21893, 0.92941177, 5.8396664),
            sigen::Sine::with_initial_phase(21892, 0.92941177, 5.8396664),
        ];
        let block_size = 23556;
        let channel_count = 2;
        let bits_per_sample = 8;
        let mut config = config::Encoder::default();
        let stream_info = crate::component::StreamInfo::new(8000, channel_count, bits_per_sample);
        config.subframe_coding.qlpc.window = config::Window::Tukey { alpha: 0.8 };

        let sample_count = channel_count * block_size;
        let mut buffer = vec![0i32; sample_count];
        for (ch, sig) in signals.iter().enumerate() {
            for (t, x) in sig
                .to_vec_quantized(bits_per_sample, block_size)
                .into_iter()
                .enumerate()
            {
                buffer[t * channel_count + ch] = x;
            }
        }

        let mut fb = FrameBuf::with_size(channel_count, block_size);
        fb.fill_interleaved(&buffer).unwrap();
        encode_fixed_size_frame(&config, &fb, 0, &stream_info).expect("should be ok");
    }
    */
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;
    use crate::source::Fill;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    use crate::sigen;
    use crate::sigen::Signal;

    #[bench]
    fn residual_encoder_zero(b: &mut Bencher) {
        let errors = [0i32; 4096];
        let cfg = &config::Prc::default();
        b.iter(|| encode_residual(black_box(cfg), black_box(&errors), black_box(3)));
    }

    #[bench]
    fn residual_partition_encoder_zero(b: &mut Bencher) {
        let errors = [0i32; 4096];
        let mut quotients = [0u32; 4096];
        let mut remainders = [0u32; 4096];
        b.iter(|| {
            encode_residual_partition(
                black_box(1024),
                black_box(2048),
                black_box(10u8),
                black_box(&errors),
                black_box(&mut quotients),
                black_box(&mut remainders),
            );
            (quotients, remainders)
        });
    }

    #[bench]
    fn fixed_lpc_encoding_zero(b: &mut Bencher) {
        let signal = [0i32; 4096];
        let cfg = &config::SubFrameCoding::default();
        b.iter(|| {
            fixed_lpc(
                black_box(cfg),
                black_box(&signal),
                black_box(16u8),
                black_box(0usize),
            )
        });
    }

    #[bench]
    fn fixed_size_frame_encoder_zero(b: &mut Bencher) {
        let cfg = &config::Encoder::default();
        let stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let mut fb = FrameBuf::with_size(2, 4096);
        // input is always zero, so it should use Constant and fast.
        fb.fill_interleaved(&[0i32; 4096 * 2]).unwrap();
        b.iter(|| {
            encode_fixed_size_frame(
                black_box(cfg),
                black_box(&fb),
                black_box(123usize),
                &stream_info,
            )
        });
    }

    fn bench_stereo_frame_encoder_impl<S: Signal>(
        b: &mut Bencher,
        signal: &S,
        use_constant: bool,
        use_fixed: bool,
        use_lpc: bool,
    ) {
        let mut cfg = config::Encoder::default();
        cfg.subframe_coding.use_constant = use_constant;
        cfg.subframe_coding.use_fixed = use_fixed;
        cfg.subframe_coding.use_lpc = use_lpc;
        let stream_info = StreamInfo::new(48000, 2, 16).unwrap();
        let mut fb = FrameBuf::with_size(2, 4096);
        let signal = signal.to_vec_quantized(16, 4096 * 2);
        fb.fill_interleaved(&signal).unwrap();
        b.iter(|| {
            encode_fixed_size_frame(
                black_box(&cfg),
                black_box(&fb),
                black_box(123usize),
                &stream_info,
            )
        });
    }

    #[bench]
    fn stereo_frame_encoder_pure_sine_fixed(b: &mut Bencher) {
        bench_stereo_frame_encoder_impl(b, &sigen::Sine::new(200, 0.4), false, true, false);
    }

    #[bench]
    fn stereo_frame_encoder_pure_sine_lpc(b: &mut Bencher) {
        bench_stereo_frame_encoder_impl(b, &sigen::Sine::new(200, 0.4), false, false, true);
    }

    #[bench]
    fn stereo_frame_encoder_noisy_sine_fixed(b: &mut Bencher) {
        bench_stereo_frame_encoder_impl(
            b,
            &sigen::Sine::new(200, 0.4).noise(0.4),
            false,
            true,
            false,
        );
    }

    #[bench]
    fn stereo_frame_encoder_noisy_sine_lpc(b: &mut Bencher) {
        bench_stereo_frame_encoder_impl(
            b,
            &sigen::Sine::new(200, 0.4).noise(0.4),
            false,
            false,
            true,
        );
    }

    #[bench]
    fn normal_qlpc_noise(b: &mut Bencher) {
        let cfg = &config::SubFrameCoding::default();
        let bits_per_sample = 16u8;
        let signal = sigen::Noise::new(0.6).to_vec_quantized(bits_per_sample as usize, 4096);

        b.iter(|| {
            estimated_qlpc(
                black_box(cfg),
                black_box(&signal),
                black_box(bits_per_sample),
            )
        });
    }
}
