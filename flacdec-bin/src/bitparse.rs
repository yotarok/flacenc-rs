// Copyright 2024 Google LLC
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

use flacenc::component;

use crate::bitsource::BitSource;
use crate::error::Error;
use crate::error::FormatError;

/// Reads `component::StreamInfo` from the bit source.
pub fn parse_stream_info<S: BitSource>(src: &mut S) -> Result<component::StreamInfo, Error> {
    let min_block_size = src.read_u64(16)? as usize;
    let max_block_size = src.read_u64(16)? as usize;
    let min_frame_size = src.read_u64(24)? as usize;
    let max_frame_size = src.read_u64(24)? as usize;
    let sample_rate = src.read_u64(20)? as usize;
    let channels = src.read_u64(3)? as usize + 1;
    let bits_per_sample = src.read_u64(5)? as usize + 1;
    let total_samples = src.read_u64(36)? as usize;
    let mut md5: [u8; 16] = [0u8; 16];
    src.read_bytes_aligned(16, &mut md5)?;
    let mut ret = component::StreamInfo::new(sample_rate, channels, bits_per_sample)?;
    ret.set_md5_digest(&md5);
    ret.set_block_sizes(min_block_size, max_block_size)?;
    ret.set_frame_sizes(min_frame_size, max_frame_size)?;
    ret.set_total_samples(total_samples);
    Ok(ret)
}

/// Reads `component::MetadataBlockData` from the bit source.
fn parse_metadata_block<S: BitSource>(
    src: &mut S,
) -> Result<(bool, component::MetadataBlockData), Error> {
    let mut header = [0u8; 1];
    src.read_bytes_aligned(1, &mut header)?;
    let header = header[0];
    let is_last = (header & 0x80u8) != 0;
    let block_type = header & 0x7Fu8;
    let block_size = src.read_u64(24)? as usize;

    log::debug!(
        target: "flacdec-bin::parse_metadata_block",
        "{{ msg: \"MetadataBlock found\", is_last: {}, block_type: {}, block_size: {} }}",
        is_last, block_type, block_size,
    );

    Ok((
        is_last,
        match block_type {
            0 => component::MetadataBlockData::from(parse_stream_info(src)?),
            _ => component::MetadataBlockData::new_unknown(
                block_type,
                &src.read_bytevec_aligned(block_size)?,
            )?,
        },
    ))
}

/// Reads `component::Residual` from the bit source.
fn parse_residual<S: BitSource>(
    src: &mut S,
    block_size: usize,
    warmup_length: usize,
) -> Result<component::Residual, Error> {
    if src.read_u64(2)? != 0 {
        return Err(FormatError::new(src.current_bit_offset(), "Only 4-bit rice supported").into());
    }

    let part_order = src.read_u64(4)? as usize;
    let nparts = 1usize << part_order;
    let part_len = block_size / nparts;

    let mut rice_ps = Vec::with_capacity(nparts);
    let mut quotients = Vec::with_capacity(block_size);
    let mut remainders = Vec::with_capacity(block_size);

    for p in 0..nparts {
        let rice_p = src.read_u64(4)? as u8;
        if rice_p == 0x0F {
            return Err(FormatError::new(
                src.current_bit_offset(),
                "escaped rice coding is unsupported",
            )
            .into());
        }
        rice_ps.push(rice_p);
        for t in 0..part_len {
            let (q, r) = if p == 0 && t < warmup_length {
                (0, 0)
            } else {
                (
                    src.read_unary_code()? as u32,
                    src.read_u64(rice_p as usize)? as u32,
                )
            };
            quotients.push(q);
            remainders.push(r);
        }
    }

    Ok(component::Residual::new(
        part_order,
        block_size,
        warmup_length,
        &rice_ps,
        &quotients,
        &remainders,
    )?)
}

/// Reads `component::Constant` from the bit source.
fn parse_constant<S: BitSource>(
    src: &mut S,
    block_size: usize,
    bits_per_sample: usize,
) -> Result<component::Constant, Error> {
    let dc_offset = src.read_i64(bits_per_sample)? as i32;
    Ok(component::Constant::new(
        block_size,
        dc_offset,
        bits_per_sample,
    )?)
}

/// Reads `component::Verbatim` from the bit source.
fn parse_verbatim<S: BitSource>(
    src: &mut S,
    block_size: usize,
    bits_per_sample: usize,
) -> Result<component::Verbatim, Error> {
    let mut data = Vec::with_capacity(block_size);
    for _t in 0..block_size {
        data.push(src.read_i64(bits_per_sample)? as i32);
    }
    Ok(component::Verbatim::new(&data, bits_per_sample)?)
}

/// Reads `component::FixedLpc` from the bit source.
fn parse_fixed<S: BitSource>(
    src: &mut S,
    block_size: usize,
    order: usize,
    bits_per_sample: usize,
) -> Result<component::FixedLpc, Error> {
    let mut warm_up = Vec::with_capacity(order);
    for _t in 0..order {
        warm_up.push(src.read_i64(bits_per_sample)? as i32);
    }
    let residual = parse_residual(src, block_size, order)?;

    Ok(component::FixedLpc::new(
        &warm_up,
        residual,
        bits_per_sample,
    )?)
}

/// Reads `component::Lpc` from the bit source.
fn parse_lpc<S: BitSource>(
    src: &mut S,
    block_size: usize,
    order: usize,
    bits_per_sample: usize,
) -> Result<component::Lpc, Error> {
    let mut warm_up = Vec::with_capacity(order);
    for _t in 0..order {
        warm_up.push(src.read_i64(bits_per_sample)? as i32);
    }

    // parse parameters
    let precision = src.read_u64(4)? as usize + 1;
    let shift = src.read_i64(5)?;
    let mut coefs = Vec::with_capacity(order);
    for _t in 0..order {
        coefs.push(src.read_i64(precision)? as i16);
    }
    let parameters = component::QuantizedParameters::new(&coefs, order, shift as i8, precision)?;
    let residual = parse_residual(src, block_size, order)?;

    Ok(component::Lpc::new(
        &warm_up,
        parameters,
        residual,
        bits_per_sample,
    )?)
}

/// Reads `component::SubFrame` from the bit source.
fn parse_subframe<S: BitSource>(
    src: &mut S,
    block_size: usize,
    bits_per_sample: usize,
) -> Result<component::SubFrame, Error> {
    let subframe_type = src.read_u64(7)?;
    if subframe_type > 0x3F {
        return Err(FormatError::new(src.current_bit_offset(), "Reserved bit set").into());
    }
    let has_wasted_bits = src.read_u64(1)? != 0;
    if has_wasted_bits {
        return Err(FormatError::new(src.current_bit_offset(), "Wasted bits not supported").into());
    }

    let subframe = if subframe_type == 0 {
        component::SubFrame::Constant(parse_constant(src, block_size, bits_per_sample)?)
    } else if subframe_type == 1 {
        component::SubFrame::Verbatim(parse_verbatim(src, block_size, bits_per_sample)?)
    } else if (8..=12).contains(&subframe_type) {
        let order = (subframe_type - 8) as usize;
        component::SubFrame::FixedLpc(parse_fixed(src, block_size, order, bits_per_sample)?)
    } else if subframe_type >= 32 {
        let order = (subframe_type - 31) as usize;
        component::SubFrame::Lpc(parse_lpc(src, block_size, order, bits_per_sample)?)
    } else {
        return Err(FormatError::new(src.current_bit_offset(), "Unsupported frame type").into());
    };
    Ok(subframe)
}

/// Reads block size specified in `FrameHeader`.
fn parse_and_decode_block_size<S: BitSource>(src: &mut S, spec: u8) -> Result<usize, Error> {
    if spec == 0 {
        Err(FormatError::new(src.current_bit_offset(), "reserved block size used").into())
    } else if spec == 1 {
        Ok(192)
    } else if spec <= 5 {
        let scale = 1usize << (spec - 2);
        Ok(576 * scale)
    } else if spec == 6 {
        Ok(src.read_u64(8)? as usize + 1)
    } else if spec == 7 {
        Ok(src.read_u64(16)? as usize + 1)
    } else if spec < 16 {
        let scale = 1usize << (spec - 8);
        Ok(256 * scale)
    } else {
        unreachable!();
    }
}

/// Reads sampling rate specified in `FrameHeader`.
fn parse_and_decode_sample_rate<S: BitSource>(
    src: &mut S,
    spec: u8,
) -> Result<component::SampleRateSpec, Error> {
    match spec {
        0 => Ok(component::SampleRateSpec::Unspecified),
        1 => Ok(component::SampleRateSpec::R88_2kHz),
        2 => Ok(component::SampleRateSpec::R176_4kHz),
        3 => Ok(component::SampleRateSpec::R192kHz),
        4 => Ok(component::SampleRateSpec::R8kHz),
        5 => Ok(component::SampleRateSpec::R16kHz),
        6 => Ok(component::SampleRateSpec::R22_05kHz),
        7 => Ok(component::SampleRateSpec::R24kHz),
        8 => Ok(component::SampleRateSpec::R32kHz),
        9 => Ok(component::SampleRateSpec::R44_1kHz),
        10 => Ok(component::SampleRateSpec::R48kHz),
        11 => Ok(component::SampleRateSpec::R96kHz),
        12 => Ok(component::SampleRateSpec::KHz(src.read_u64(8)? as u8)),
        13 => Ok(component::SampleRateSpec::Hz(src.read_u64(16)? as u16)),
        14 => Ok(component::SampleRateSpec::DaHz(src.read_u64(16)? as u16)),
        15 => Err(FormatError::new(src.current_bit_offset(), "invalid sample rate").into()),
        _ => unreachable!(),
    }
}

/// Reads `component::FrameHeader` from the bit source.
fn parse_frame_header<S: BitSource>(src: &mut S) -> Result<component::FrameHeader, Error> {
    let marker = src.read_u64(16)? as u16;
    if marker & 0xFFFC != 0xFFF8 {
        return Err(FormatError::new(src.current_bit_offset(), "Sync code not found").into());
    }
    if marker & 0x02 != 0 {
        return Err(FormatError::new(src.current_bit_offset(), "Reserved bit is set").into());
    }
    let variable_block_size = (marker & 0x01) != 0;

    let block_size_tag = src.read_u64(4)? as u8;
    let sample_rate_tag = src.read_u64(4)? as u8;
    let channel_assignment = component::ChannelAssignment::from_tag(src.read_u64(4)? as u8)
        .ok_or_else(|| FormatError::new(src.current_bit_offset(), "Reserved channel assignment"))?;
    let bits_per_sample_tag = src.read_u64(3)? as u8;
    let reserved = src.read_u64(1)? != 0;
    if reserved {
        return Err(FormatError::new(src.current_bit_offset(), "Reserved bit is set").into());
    }

    let (frame_number, start_sample_number) = if variable_block_size {
        (0, src.read_utf8_aligned()?)
    } else {
        (src.read_utf8_aligned()?, 0)
    };

    let block_size = parse_and_decode_block_size(src, block_size_tag)?;
    let sample_rate = parse_and_decode_sample_rate(src, sample_rate_tag)?;
    let bits_per_sample = component::SampleSizeSpec::from_tag(bits_per_sample_tag)
        .ok_or_else(|| FormatError::new(src.current_bit_offset(), "sample size out of range"))?;
    if bits_per_sample == component::SampleSizeSpec::Reserved {
        return Err(FormatError::new(src.current_bit_offset(), "reserved sample size").into());
    }

    let _crc = src.read_u64(8)?;

    let mut header = if variable_block_size {
        component::FrameHeader::new_variable_size(
            block_size,
            channel_assignment,
            bits_per_sample,
            start_sample_number as usize,
        )?
    } else {
        component::FrameHeader::new_fixed_size(
            block_size,
            channel_assignment,
            bits_per_sample,
            frame_number as usize,
        )?
    };
    header.set_sample_rate_spec(sample_rate);
    Ok(header)
}

/// Reads `component::Frame` from the bit source.
fn parse_frame<S: BitSource>(
    src: &mut S,
    stream_info: &component::StreamInfo,
) -> Result<component::Frame, Error> {
    let header = parse_frame_header(src)?;
    let mut subframes = Vec::with_capacity(stream_info.channels());
    let bits_per_sample = header
        .bits_per_sample()
        .unwrap_or_else(|| stream_info.bits_per_sample());
    for ch in 0..stream_info.channels() {
        let subframe = parse_subframe(
            src,
            header.block_size(),
            bits_per_sample + header.channel_assignment().bits_per_sample_offset(ch),
        )?;
        subframes.push(subframe);
    }
    src.skip_to_next_byte();
    let _frame_crc = src.read_u64(16)? as u16;
    // so far no CRC check done.

    Ok(component::Frame::new(header, subframes.into_iter())?)
}

/// Reads `component::Stream` from the bit source.
pub fn parse_stream<S: BitSource>(src: &mut S) -> Result<component::Stream, Error> {
    let mut header = [0u8; 4];
    src.read_bytes_aligned(4, &mut header)?;
    if header != [0x66, 0x4c, 0x61, 0x43] {
        return Err(FormatError::new(
            src.current_bit_offset(),
            "Magic number doesn't match. (Probably, non FLAC input given?)",
        )
        .into());
    }

    let (mut is_last, first_metadata) = parse_metadata_block(src)?;
    let stream_info = first_metadata.as_stream_info().ok_or_else::<Error, _>(|| {
        FormatError::new(
            src.current_bit_offset(),
            "First metadata block must be StreamInfo.",
        )
        .into()
    })?;
    let mut stream = component::Stream::with_stream_info(stream_info.clone());

    while !is_last {
        let (b, metadata) = parse_metadata_block(src)?;
        is_last = b;
        stream.add_metadata_block(metadata);
    }

    loop {
        let frame = match parse_frame(src, stream_info) {
            Ok(f) => f,
            Err(err) => {
                if err.is_stream_ended() {
                    break;
                }
                return Err(err);
            }
        };
        stream.add_frame(frame);
    }
    Ok(stream)
}
