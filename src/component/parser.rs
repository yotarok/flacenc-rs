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

use nom::bits::bits;
use nom::bits::streaming::tag as bit_tag;
use nom::bits::streaming::take as bit_take;
use nom::branch::alt;
use nom::bytes::streaming::tag as byte_tag;
use nom::bytes::streaming::take as byte_take;
use nom::combinator::into;
use nom::combinator::map;
use nom::combinator::verify;
use nom::error::ParseError;
use nom::error_position;
use nom::multi::many_m_n;
use nom::multi::many_till;
use nom::number::streaming::be_u16;
use nom::number::streaming::be_u24;
use nom::number::streaming::be_u8;
use nom::IResult;
use nom::Offset;

use crate::component;
use crate::component::bitrepr::FRAME_CRC;
use crate::component::bitrepr::HEADER_CRC;
use crate::component::FrameOffset;
use crate::constant::MAX_BITS_PER_SAMPLE;
use crate::error::VerifyError;

type BitInput<'a> = (&'a [u8], usize);

/// Recognizes [`component::Stream`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn stream<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], component::Stream, E>
where
    E: ParseError<&'a [u8]>,
{
    let remaining_input = input;
    let (remaining_input, _) = byte_tag("fLaC")(remaining_input)?;
    let (remaining_input, stream_info_block) = metadata_block(remaining_input)?;
    let stream_info = stream_info_block
        .data
        .as_stream_info()
        .ok_or_else(|| nom::Err::Error(error_position!(input, nom::error::ErrorKind::Verify)))?;

    let (remaining_input, rest_mdblocks): (_, Vec<component::MetadataBlock>) =
        if stream_info_block.is_last {
            (remaining_input, vec![])
        } else {
            let mut is_last = stream_info_block.is_last;
            let mut blocks = vec![];
            let mut remaining_input = remaining_input;
            while !is_last {
                let (i, b) = metadata_block(remaining_input)?;
                is_last = b.is_last;
                remaining_input = i;
                blocks.push(b);
            }
            (remaining_input, blocks)
        };

    let (remaining_input, (frames, _)) =
        many_till(frame(stream_info, true), nom::combinator::eof)(remaining_input)?;
    let mut stream = component::Stream::with_stream_info(stream_info.clone());
    for mdblock in &rest_mdblocks {
        stream.add_metadata_block(mdblock.data.clone());
    }
    for f in &frames {
        stream.frames_mut().push(f.clone());
    }
    Ok((remaining_input, stream))
}

/// Recognizes [`component::MetadataBlock`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn metadata_block<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], component::MetadataBlock, E>
where
    E: ParseError<&'a [u8]>,
{
    let remaining_input = input;
    let (remaining_input, first_byte) = be_u8(remaining_input)?;
    let is_last = (first_byte >> 7) != 0;
    let block_type = first_byte & 0x7F;
    let (remaining_input, length) = be_u24(remaining_input)?;

    #[allow(clippy::single_match_else)]
    let (remaining_input, body) = match block_type {
        0 => map(stream_info, Into::into)(remaining_input)?,
        _ => {
            let (i, blob) = byte_take(length)(remaining_input)?;
            (
                i,
                component::MetadataBlockData::new_unknown(block_type, blob).map_err(|_e| {
                    nom::Err::Error(error_position!(
                        remaining_input,
                        nom::error::ErrorKind::TagBits
                    ))
                })?,
            )
        }
    };
    Ok((
        remaining_input,
        component::MetadataBlock::from_parts(is_last, body),
    ))
}

/// Recognizes [`component::StreamInfo`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
///
/// # Panics
///
/// Only panics by internal errors.
pub fn stream_info<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], component::StreamInfo, E>
where
    E: ParseError<&'a [u8]>,
{
    let remaining_input = input;
    let (remaining_input, min_block_size) = be_u16(remaining_input)?;
    let (remaining_input, max_block_size) = be_u16(remaining_input)?;
    let (remaining_input, min_frame_size) = be_u24(remaining_input)?;
    let (remaining_input, max_frame_size) = be_u24(remaining_input)?;
    let (remaining_input, (sample_rate, channels, bits_per_sample, total_samples)) =
        bits(|input| {
            let remaining_input = input;
            let (remaining_input, sr) = bit_take(20usize)(remaining_input)?;
            let (remaining_input, ch): (_, usize) = bit_take(3usize)(remaining_input)?;
            let (remaining_input, bps): (_, usize) = bit_take(5usize)(remaining_input)?;
            let (remaining_input, total): (_, usize) = bit_take(36usize)(remaining_input)?;
            let ret: IResult<_, _, (BitInput<'a>, nom::error::ErrorKind)> =
                Ok((remaining_input, (sr, ch + 1, bps + 1, total)));
            ret
        })(remaining_input)
        .map_err(convert_bits_err)?;
    let (remaining_input, md5) = byte_take(16usize)(remaining_input)?;
    let info_fn = || {
        let mut info = component::StreamInfo::new(sample_rate, channels, bits_per_sample)?;
        info.set_total_samples(total_samples);
        info.set_md5_digest(md5.try_into().expect("Internal error"));
        info.set_block_sizes(min_block_size as usize, max_block_size as usize)?;
        info.set_frame_sizes(min_frame_size as usize, max_frame_size as usize)?;
        let ret: Result<_, VerifyError> = Ok(info);
        ret
    };
    let info = info_fn().map_err(|_e| {
        nom::Err::Error(error_position!(
            remaining_input,
            nom::error::ErrorKind::Verify
        ))
    })?;
    Ok((remaining_input, info))
}

/// Recognizes [`component::Frame`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn frame<'a, E>(
    stream_info: &component::StreamInfo,
    check_crc: bool,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], component::Frame, E>
where
    E: ParseError<&'a [u8]>,
{
    let channels_from_header = stream_info.channels();
    let bits_per_sample_from_header = stream_info.bits_per_sample();

    move |input| {
        let input_start = input;
        let remaining_input = input;
        let (remaining_input, header) = frame_header(true)(remaining_input)?;
        let channels = header.channel_assignment().channels();
        if channels != channels_from_header {
            return Err(nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            )));
        }
        let block_size = header.block_size();
        let bits_per_sample = header
            .bits_per_sample()
            .unwrap_or(bits_per_sample_from_header);
        if bits_per_sample != bits_per_sample_from_header {
            return Err(nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            )));
        }

        let mut ch = 0;
        let (remaining_input, subframes) = bits(many_m_n(channels, channels, |i| {
            let ret = subframe::<(BitInput<'a>, nom::error::ErrorKind)>(
                block_size,
                bits_per_sample + header.channel_assignment().bits_per_sample_offset(ch),
            )(i);
            ch += 1;
            ret
        }))(remaining_input)
        .map_err(convert_bits_err)?;
        let test_crc16 = check_crc.then(|| {
            let frame_bytes = &input_start[..input_start.offset(remaining_input)];
            FRAME_CRC.checksum(frame_bytes)
        });
        let (remaining_input, _) =
            verify(be_u16, |crc| test_crc16.map_or(true, |x| x == *crc))(remaining_input)?;

        let frame = component::Frame::from_parts(header, subframes);
        Ok((remaining_input, frame))
    }
}

fn convert_bits_err<'a, E>(e: nom::Err<(&'a [u8], nom::error::ErrorKind)>) -> nom::Err<E>
where
    E: ParseError<&'a [u8]>,
{
    e.map(|(inp, kind)| E::from_error_kind(inp, kind))
}

/// Recognizes [`component::FrameHeader`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn frame_header<'a, E>(
    check_crc: bool,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], component::FrameHeader, E>
where
    E: ParseError<&'a [u8]>,
{
    move |input| {
        let input_start = input;
        let (
            remaining_input,
            (blocking_type, block_size_tag, sample_rate_tag, channel_tag, sample_size_tag),
        ) = bits(|bit_input| {
            let remaining_input = bit_input;
            let (remaining_input, _sync) = bit_tag(0x7FFCu16, 15usize)(remaining_input)?;
            let (remaining_input, blocking_type): (_, u8) = bit_take(1usize)(remaining_input)?;

            let (remaining_input, block_size_tag): (_, u8) = bit_take(4usize)(remaining_input)?;
            let (remaining_input, sample_rate_tag): (_, u8) = bit_take(4usize)(remaining_input)?;
            let (remaining_input, channel_tag): (_, u8) = bit_take(4usize)(remaining_input)?;
            let (remaining_input, sample_size_tag): (_, u8) = bit_take(3usize)(remaining_input)?;
            let (remaining_input, _reserved) = bit_tag(0, 1usize)(remaining_input)?;
            let ret: IResult<BitInput<'a>, _, (BitInput<'a>, nom::error::ErrorKind)> = Ok((
                remaining_input,
                (
                    blocking_type,
                    block_size_tag,
                    sample_rate_tag,
                    channel_tag,
                    sample_size_tag,
                ),
            ));
            ret
        })(input)
        .map_err(convert_bits_err)?;

        let bits_per_sample =
            component::SampleSizeSpec::from_tag(sample_size_tag).ok_or_else(|| {
                nom::Err::Error(error_position!(
                    remaining_input,
                    nom::error::ErrorKind::TagBits
                ))
            })?;
        let channel_assignment =
            component::ChannelAssignment::from_tag(channel_tag).ok_or_else(|| {
                nom::Err::Error(error_position!(
                    remaining_input,
                    nom::error::ErrorKind::TagBits
                ))
            })?;
        let (remaining_input, offset) = if blocking_type == 0 {
            map(utf8_code, |x| FrameOffset::Frame(x as u32))(remaining_input)?
        } else {
            map(utf8_code, FrameOffset::StartSample)(remaining_input)?
        };

        let (remaining_input, block_size_spec): (&[u8], component::BlockSizeSpec) =
            block_size_code(block_size_tag)(remaining_input)?;
        let (remaining_input, sample_rate) = sample_rate_code(sample_rate_tag)(remaining_input)?;

        let test_crc8 = check_crc.then(|| {
            let header_bytes = &input_start[..input_start.offset(remaining_input)];
            HEADER_CRC.checksum(header_bytes)
        });
        let (remaining_input, _) =
            verify(be_u8, |crc| test_crc8.map_or(true, |x| x == *crc))(remaining_input)?;

        let mut frame_header = component::FrameHeader::from_specs(
            block_size_spec,
            channel_assignment,
            bits_per_sample,
            sample_rate,
        );
        frame_header.set_frame_offset(offset);

        Ok((remaining_input, frame_header))
    }
}

fn block_size_code<'a, E>(
    tag: u8,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], component::BlockSizeSpec, E>
where
    E: ParseError<&'a [u8]>,
{
    move |input| match tag {
        0b0001 => Ok((input, component::BlockSizeSpec::S192)),
        0b0010..=0b0101 => Ok((input, component::BlockSizeSpec::Pow2Mul576(tag - 0b0010))),
        0b0110 => {
            let (i, x) = be_u8(input)?;
            Ok((i, component::BlockSizeSpec::ExtraByte(x)))
        }
        0b0111 => {
            let (i, x) = be_u16(input)?;
            Ok((i, component::BlockSizeSpec::ExtraTwoBytes(x)))
        }
        0b1000..=0b1111 => Ok((input, component::BlockSizeSpec::Pow2Mul256(tag - 0b1000))),
        _ => Err(nom::Err::Error(error_position!(
            input,
            nom::error::ErrorKind::TagBits
        ))),
    }
}

fn sample_rate_code<'a, E>(
    tag: u8,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], component::SampleRateSpec, E>
where
    E: ParseError<&'a [u8]>,
{
    debug_assert!(tag <= 0b1110);
    move |input| {
        let remaining_input = input;
        let (remaining_input, data) = if tag == 0b1100 {
            let (r, x) = be_u8(remaining_input)?;
            (r, Some(x as usize))
        } else if tag == 0b1101 || tag == 0b1110 {
            let (r, x) = be_u16(remaining_input)?;
            (r, Some(x as usize))
        } else {
            (remaining_input, None)
        };
        let spec = component::SampleRateSpec::from_tag_and_data(tag, data).ok_or_else(|| {
            nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            ))
        })?;
        Ok((remaining_input, spec))
    }
}

fn utf8_code<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], u64, E>
where
    E: ParseError<&'a [u8]>,
{
    let remaining_input = input;
    let (remaining_input, head): (_, u64) =
        map(byte_take(1usize), |x: &[u8]| x[0].into())(remaining_input)?;

    let (tail_count, mut acc) = if head < 128 {
        (0usize, head & 0x7F)
    } else if head < 0xE0 {
        (1, head & 0x1F)
    } else if head < 0xF0 {
        (2, head & 0x0F)
    } else if head < 0xF8 {
        (3, head & 0x07)
    } else if head < 0xFC {
        (4, head & 0x03)
    } else if head < 0xFE {
        (5, head & 0x01)
    } else if head == 0xFE {
        (6, 0)
    } else {
        return Err(nom::Err::Error(error_position!(
            remaining_input,
            nom::error::ErrorKind::TagBits
        )));
    };

    let (remaining_input, tail): (_, &[u8]) = byte_take(tail_count)(remaining_input)?;
    for b in tail {
        acc = acc << 6 | u64::from(*b & 0x3F);
    }
    Ok((remaining_input, acc))
}

/// Recognizes [`component::SubFrame`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn subframe<'a, E>(
    block_size: usize,
    bits_per_sample: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::SubFrame, E>
where
    E: ParseError<BitInput<'a>>,
{
    debug_assert!(bits_per_sample <= MAX_BITS_PER_SAMPLE);
    alt((
        into(constant::<E>(block_size, bits_per_sample)),
        into(fixed_lpc::<E>(block_size, bits_per_sample)),
        into(lpc::<E>(block_size, bits_per_sample)),
        into(verbatim::<E>(block_size, bits_per_sample)),
    ))
}

fn subframe_header<'a, E>(input: BitInput<'a>) -> IResult<BitInput<'a>, (u8, bool), E>
where
    E: ParseError<BitInput<'a>>,
{
    let remaining_input = input;

    let (remaining_input, typetag) = bit_take(7usize)(remaining_input)?;
    let (remaining_input, wasted_flag): (_, u8) = bit_take(1usize)(remaining_input)?;

    assert!(wasted_flag == 0); // not supported

    Ok((remaining_input, (typetag, wasted_flag != 0)))
}

/// Recognizes [`component::Constant`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn constant<'a, E>(
    block_size: usize,
    bits_per_sample: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::Constant, E>
where
    E: ParseError<BitInput<'a>>,
{
    debug_assert!(bits_per_sample <= MAX_BITS_PER_SAMPLE);
    move |input| {
        let remaining_input = input;
        let (remaining_input, (typetag, _wasted_flag)) = subframe_header(remaining_input)?;
        if typetag != 0x00 {
            return Err(nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            )));
        }

        let (remaining_input, dc_offset) =
            map(bit_take(bits_per_sample), |u| u_to_i(u, bits_per_sample))(remaining_input)?;

        Ok((
            remaining_input,
            component::Constant::from_parts(block_size, dc_offset, bits_per_sample as u8),
        ))
    }
}

/// Recognizes [`component::FixedLpc`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
///
/// # Panics
///
/// Only panics by an internal error.
pub fn fixed_lpc<'a, E>(
    block_size: usize,
    bits_per_sample: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::FixedLpc, E>
where
    E: ParseError<BitInput<'a>>,
{
    debug_assert!(bits_per_sample <= MAX_BITS_PER_SAMPLE);
    move |input| {
        let remaining_input = input;
        let (remaining_input, (typetag, _wasted_flag)) = subframe_header(remaining_input)?;
        if !(0x08..=0x0C).contains(&typetag) {
            return Err(nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            )));
        }
        let order = (typetag as usize) - 0x08;
        let (remaining_input, warm_up) = raw_samples(bits_per_sample, order)(remaining_input)?;
        let warm_up = heapless::Vec::try_from(warm_up.as_slice()).expect("Unexpected error");

        let (remaining_input, residual) = residual(block_size, order)(remaining_input)?;

        Ok((
            remaining_input,
            component::FixedLpc::from_parts(warm_up, residual, bits_per_sample as u8),
        ))
    }
}

/// Recognizes [`component::Lpc`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
///
/// # Panics
///
/// Only panics by an internal error.
pub fn lpc<'a, E>(
    block_size: usize,
    bits_per_sample: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::Lpc, E>
where
    E: ParseError<BitInput<'a>>,
{
    debug_assert!(bits_per_sample <= MAX_BITS_PER_SAMPLE);
    move |input| {
        let remaining_input = input;
        let (remaining_input, (typetag, _wasted_flag)) = subframe_header(remaining_input)?;
        if !(0x20..0x40).contains(&typetag) {
            return Err(nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            )));
        }
        let order = (typetag as usize) - 0x20 + 1;
        let (remaining_input, warm_up) = raw_samples(bits_per_sample, order)(remaining_input)?;
        let warm_up = heapless::Vec::try_from(warm_up.as_slice()).expect("Unexpected error");

        let (remaining_input, parameters) = quantized_parameters(order)(remaining_input)?;
        let (remaining_input, residual) = residual(block_size, order)(remaining_input)?;

        Ok((
            remaining_input,
            component::Lpc::from_parts(warm_up, parameters, residual, bits_per_sample as u8),
        ))
    }
}

/// Recognizes [`component::QuantizedParameters`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
///
/// # Panics
///
/// Only panics by an internal error.
pub fn quantized_parameters<'a, E>(
    order: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::QuantizedParameters, E>
where
    E: ParseError<BitInput<'a>>,
{
    move |input| {
        let remaining_input = input;
        let (remaining_input, precision): (_, usize) =
            map(bit_take(4usize), |p: u8| (p as usize + 1))(remaining_input)?;
        let (remaining_input, shift): (_, i8) =
            map(bit_take(5usize), |x: u8| u_to_i(u32::from(x), 5) as i8)(remaining_input)?;
        let (remaining_input, coefs) = raw_samples(precision, order)(remaining_input)?;

        let coefs: Vec<i16> = coefs.into_iter().map(|x| x as i16).collect();
        let ret = component::QuantizedParameters::new(&coefs, order, shift, precision)
            .expect("Unexpected error");
        Ok((remaining_input, ret))
    }
}

/// Recognizes [`component::Verbatim`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn verbatim<'a, E>(
    block_size: usize,
    bits_per_sample: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::Verbatim, E>
where
    E: ParseError<BitInput<'a>>,
{
    debug_assert!(bits_per_sample <= MAX_BITS_PER_SAMPLE);
    move |input| {
        let remaining_input = input;
        let (remaining_input, (typetag, _wasted_flag)) = subframe_header(remaining_input)?;
        if typetag != 0x01 {
            return Err(nom::Err::Error(error_position!(
                remaining_input,
                nom::error::ErrorKind::TagBits
            )));
        }

        let (remaining_input, data) = raw_samples(bits_per_sample, block_size)(remaining_input)?;

        // TODO: might be better to add `from_parts` method to `Verbatim`.
        Ok((
            remaining_input,
            component::Verbatim::from_samples(&data, bits_per_sample as u8),
        ))
    }
}

/// Recognizes [`component::Residual`].
///
/// # Errors
///
/// Same as other nom parsers, this returns [`nom::Err`] if `input` doesn't conforms the format.
pub fn residual<'a, E>(
    block_size: usize,
    warmup_length: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, component::Residual, E>
where
    E: ParseError<BitInput<'a>>,
{
    move |input| {
        let remaining_input = input;

        let (remaining_input, method): (_, u8) = bit_take(2usize)(remaining_input)?;
        let p_bits = match method {
            0b00 => 4usize,
            0b01 => 5usize,
            _ => {
                return Err(nom::Err::Error(error_position!(
                    remaining_input,
                    nom::error::ErrorKind::TagBits
                )));
            }
        };
        let (remaining_input, partition_order): (_, u8) = bit_take(4usize)(remaining_input)?;

        let partition_count = 1usize << (partition_order as usize);
        let partition_len = block_size / partition_count;

        let mut rice_params = Vec::with_capacity(partition_count);
        let mut quotients = Vec::with_capacity(block_size);
        let mut remainders = Vec::with_capacity(block_size);

        let mut remaining_input = remaining_input;
        for part in 0..partition_count {
            let (i, rice_p) = bit_take(p_bits)(remaining_input)?;
            remaining_input = i;
            rice_params.push(rice_p);

            for t in (partition_len * part)..(partition_len * (part + 1)) {
                if t < warmup_length {
                    quotients.push(0);
                    remainders.push(0);
                    continue;
                }

                let (i, q): (_, usize) = unary_code(remaining_input)?;
                remaining_input = i;
                let (i, r): (_, u32) = bit_take(rice_p as usize)(remaining_input)?;
                remaining_input = i;
                quotients.push(q as u32);
                remainders.push(r);
            }
        }
        let parsed = component::Residual::from_parts(
            partition_order,
            block_size,
            warmup_length,
            rice_params,
            quotients,
            remainders,
        );

        Ok((remaining_input, parsed))
    }
}

fn u_to_i(x: u32, bits: usize) -> i32 {
    let x: u64 = x.into(); // widen
    let msb: u64 = 1u64 << (bits - 1);
    let offset: i32 = if x >= msb { (1u32 << bits) as i32 } else { 0 };
    i32::try_from(x).unwrap() - offset
}

/// Utility parser for reading a sequence of samples with an arbitrary bit-width.
fn raw_samples<'a, E>(
    bits_per_sample: usize,
    size: usize,
) -> impl FnMut(BitInput<'a>) -> IResult<BitInput<'a>, Vec<i32>, E>
where
    E: ParseError<BitInput<'a>>,
{
    debug_assert!(bits_per_sample <= MAX_BITS_PER_SAMPLE);
    move |input| {
        let mut remaining_input = input;
        let mut data = Vec::with_capacity(size);

        for _t in 0..size {
            let (i, u): (_, u32) = bit_take(bits_per_sample)(remaining_input)?;
            remaining_input = i;
            data.push(u_to_i(u, bits_per_sample));
        }
        Ok((remaining_input, data))
    }
}

/// Recognizes unary-code of unsigned integers.
fn unary_code<'a, E>(input: BitInput<'a>) -> IResult<BitInput<'a>, usize, E>
where
    E: ParseError<BitInput<'a>>,
{
    let remaining_input = input;
    let (remaining_input, ret) = nom::multi::many0_count(bit_tag(0, 1usize))(remaining_input)?;
    let (remaining_input, _) = bit_tag(1, 1usize)(remaining_input)?;
    Ok((remaining_input, ret))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::coding;
    use crate::component::bitrepr::encode_to_utf8like;
    use crate::component::BitRepr;
    use crate::config;
    use crate::config::Encoder as EncoderConfig;
    use crate::config::Window;
    use crate::constant;
    use crate::error::Verify;
    use crate::lpc;
    use crate::sigen;
    use crate::sigen::Signal;
    use crate::source;
    use crate::test_helper::make_random_residual;
    use crate::test_helper::make_verbatim_frame;

    use nom::error::VerboseError;

    use rand::distributions::Distribution;
    use rand::distributions::Uniform;
    use rand::Rng;

    #[test]
    fn decoding_stream() {
        let channels = 2;
        let bits_per_sample = 16;
        let sample_rate = 16000;
        let signal_len = 102;

        let mut channel_signals = vec![];
        for _ch in 0..channels {
            channel_signals.push(
                sigen::Sine::new(36, 0.4)
                    .noise(0.04)
                    .to_vec_quantized(bits_per_sample, signal_len),
            );
        }

        let mut signal = vec![];
        for t in 0..signal_len {
            for s in &channel_signals {
                signal.push(s[t]);
            }
        }

        let source =
            source::MemSource::from_samples(&signal, channels, bits_per_sample, sample_rate);

        let config = EncoderConfig::default().into_verified().unwrap();
        let comp = coding::encode_with_fixed_block_size(&config, source, config.block_size)
            .expect("encoding error");

        let bytes = comp.to_bytes();
        let (_remaining_input, decoded) =
            stream::<VerboseError<&[u8]>>(&bytes).expect("Unexpected parse error");

        assert_eq!(
            comp.stream_info().to_bytes(),
            decoded.stream_info().to_bytes()
        );
        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_metadata_block_and_stream_info() {
        for is_last in [true, false] {
            let nchannels: usize = 2;
            let sample_rate: usize = 44100;
            let bits_per_sample: usize = 24;
            let mut stream_info =
                component::StreamInfo::new(sample_rate, nchannels, bits_per_sample).unwrap();
            stream_info.set_block_sizes(128, 1024).unwrap();
            stream_info.set_frame_sizes(123, 4567).unwrap();
            let comp = component::MetadataBlock::from_parts(is_last, stream_info.into());

            let bytes = comp.to_bytes();
            let (_remaining_input, decoded) =
                metadata_block::<VerboseError<&[u8]>>(&bytes).expect("Unexpected parse error");

            assert_eq!(comp.to_bytes(), decoded.to_bytes());
        }
    }

    fn decoding_frame_testimpl(block_size: usize, bits_per_sample: usize, sample_rate: usize) {
        let nchannels: usize = 2;
        let stream_info =
            component::StreamInfo::new(sample_rate, nchannels, bits_per_sample).unwrap();
        let framebuf = vec![-1i32; block_size * nchannels];
        let comp = make_verbatim_frame(&stream_info, &framebuf, 0);

        let bytes = comp.to_bytes();
        let (_remaining_input, decoded) = frame::<VerboseError<&[u8]>>(&stream_info, true)(&bytes)
            .expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_frame() {
        for block_size in [1152, 1024] {
            let bits_per_sample = 16;
            let sample_rate = 65535;
            decoding_frame_testimpl(block_size, bits_per_sample, sample_rate);
        }
    }

    fn decoding_frame_header_testimpl(
        block_size: usize,
        bits_per_sample: usize,
        sample_rate: usize,
    ) {
        let nchannels: usize = 2;
        let stream_info =
            component::StreamInfo::new(sample_rate, nchannels, bits_per_sample).unwrap();
        let framebuf = vec![-1i32; block_size * nchannels];
        let frame = make_verbatim_frame(&stream_info, &framebuf, 0);
        let comp = frame.header().clone();

        let bytes = comp.to_bytes();

        let (_remaining_input, decoded) =
            frame_header::<VerboseError<&[u8]>>(true)(&bytes).expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_frame_header() {
        for block_size in [192, 1152, 127, 298, 1024] {
            for bits_per_sample in [8, 16, 24] {
                for sample_rate in [88200, 3, 65535, 95900] {
                    decoding_frame_header_testimpl(block_size, bits_per_sample, sample_rate);
                }
            }
        }
    }

    #[test]
    fn decoding_utf8_code() {
        for x in &[
            0u64,
            76,
            195,
            256,
            257,
            1000,
            1023,
            1024,
            65535,
            65536,
            68000,
            68_719_476_735, // 2^36 - 1
        ] {
            let code = encode_to_utf8like(*x).expect("encode error");
            let (remaining_input, y) =
                utf8_code::<VerboseError<&[u8]>>(&code).expect("decode error");
            assert_eq!(remaining_input, &[]);
            assert_eq!(*x, y);
        }
    }

    fn random_lpc<R: Rng>(mut rng: R) -> component::Lpc {
        let block_size = Uniform::from(64..=256).sample(&mut rng);
        let order = Uniform::from(1..=constant::qlpc::MAX_ORDER).sample(&mut rng);
        let precision = Uniform::from(1..=constant::qlpc::MAX_PRECISION).sample(&mut rng);
        let mut signal = Vec::with_capacity(block_size);
        for _t in 0..block_size {
            signal.push(Uniform::from(-127..=127).sample(&mut rng));
        }
        let lpc_coefs = lpc::lpc_from_autocorr(&signal, &Window::default(), order);
        let qlpc = lpc::quantize_parameters(&lpc_coefs[0..order], precision);
        let mut errors = vec![0i32; signal.len()];
        lpc::compute_error(&qlpc, &signal, &mut errors);
        let residual = coding::encode_residual(&config::Prc::default(), &errors, qlpc.order());

        component::Lpc::from_parts(
            heapless::Vec::from_slice(&signal[0..qlpc.order()])
                .expect("LPC order exceeded the maximum"),
            qlpc,
            residual,
            8,
        )
    }

    #[test]
    fn decoding_constant() {
        let mut rng = rand::thread_rng();
        let block_size = Uniform::from(64..=256).sample(&mut rng);
        let offset = Uniform::from(-1000..1000).sample(&mut rng);
        let comp = component::Constant::new(block_size, offset, 16).expect("construction error");
        let bytes = comp.to_bytes();

        let (_remaining_input, decoded) = constant::<VerboseError<BitInput>>(
            comp.block_size(),
            comp.bits_per_sample(),
        )((&bytes, 0))
        .expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_fixedlpc() {
        let residual = make_random_residual(rand::thread_rng(), 2);
        let comp = component::FixedLpc::new(&[0, 0], residual, 16).expect("");
        let bytes = comp.to_bytes();

        let (_remaining_input, decoded) = fixed_lpc::<VerboseError<BitInput>>(
            comp.residual().block_size(),
            comp.bits_per_sample(),
        )((&bytes, 0))
        .expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_lpc() {
        let comp = random_lpc(rand::thread_rng());
        let bytes = comp.to_bytes();

        let (_remaining_input, decoded) = lpc::<VerboseError<BitInput>>(
            comp.residual().block_size(),
            comp.bits_per_sample(),
        )((&bytes, 0))
        .expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_verbatim() {
        let mut rng = rand::thread_rng();
        let block_size = Uniform::from(1..=128).sample(&mut rng);
        let mut samples = Vec::with_capacity(block_size);
        for _t in 0..block_size {
            samples.push(Uniform::from(-127..=127).sample(&mut rng));
        }
        let bits_per_sample = 12;

        let comp = component::Verbatim::from_samples(samples.as_slice(), bits_per_sample as u8);
        let bytes = comp.to_bytes();
        let (_remaining_input, decoded) =
            verbatim::<VerboseError<BitInput>>(block_size, bits_per_sample)((&bytes, 0))
                .expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_residual() {
        let comp = make_random_residual(rand::thread_rng(), 0);
        let bytes = comp.to_bytes();
        let (_remaining_input, decoded) = residual::<VerboseError<BitInput>>(
            comp.block_size(),
            comp.warmup_length(),
        )((&bytes, 0))
        .expect("Unexpected parse error");

        assert_eq!(comp.to_bytes(), decoded.to_bytes());
    }

    #[test]
    fn decoding_unary_code() {
        let (remaining_input, decoded) = unary_code::<VerboseError<BitInput>>((&[0x01], 0))
            .expect("Unexpected error from `unary_code`.");
        assert_eq!(decoded, 7);
        assert_eq!(remaining_input, ([].as_slice(), 0));

        let (remaining_input, decoded) = unary_code::<VerboseError<BitInput>>((&[0x81], 1))
            .expect("Unexpected error from `unary_code`.");
        assert_eq!(decoded, 6);
        assert_eq!(remaining_input, ([].as_slice(), 0));

        let (remaining_input, decoded) = unary_code::<VerboseError<BitInput>>((&[0x80, 0x0F], 1))
            .expect("Unexpected error from `unary_code`.");
        assert_eq!(decoded, 11);
        assert_eq!(remaining_input, ([0x0F].as_slice(), 5));

        let (_remaining_input, decoded) = unary_code::<VerboseError<BitInput>>((&[0xDF], 2))
            .expect("Unexpected error from `unary_code`.");
        assert_eq!(decoded, 1);
    }

    #[test]
    fn unsigned_to_signed_conversion() {
        assert_eq!(127, u_to_i(127, 8));
        assert_eq!(-128, u_to_i(128, 8));
    }
}
