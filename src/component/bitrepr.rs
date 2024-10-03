// Copyright 2022-2024 Google LLC
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

use std::cmp::max;

use super::super::bitsink::BitSink;
use super::super::bitsink::ByteSink;
use super::super::bitsink::MemSink;
use super::super::error::OutputError;
use super::super::error::RangeError;
use super::super::repeat::try_repeat;

use super::datatype::ChannelAssignment;
use super::datatype::Constant;
use super::datatype::FixedLpc;
use super::datatype::Frame;
use super::datatype::FrameHeader;
use super::datatype::Lpc;
use super::datatype::MetadataBlock;
use super::datatype::MetadataBlockData;
use super::datatype::Residual;
use super::datatype::Stream;
use super::datatype::StreamInfo;
use super::datatype::SubFrame;
use super::datatype::Verbatim;

const CRC_8_FLAC: crc::Algorithm<u8> = crc::CRC_8_SMBUS;
const CRC_16_FLAC: crc::Algorithm<u16> = crc::CRC_16_UMTS;

pub mod seal_bit_repr {
    pub trait Sealed {}
    impl Sealed for super::Stream {}
    impl Sealed for super::MetadataBlock {}
    impl Sealed for super::MetadataBlockData {}
    impl Sealed for super::StreamInfo {}
    impl Sealed for super::Frame {}
    impl Sealed for super::FrameHeader {}
    impl Sealed for super::ChannelAssignment {}
    impl Sealed for super::SubFrame {}
    impl Sealed for super::Constant {}
    impl Sealed for super::FixedLpc {}
    impl Sealed for super::Verbatim {}
    impl Sealed for super::Lpc {}
    impl Sealed for super::Residual {}
}

/// FLAC components that can be represented in a bit sequence.
pub trait BitRepr: seal_bit_repr::Sealed {
    /// Counts the number of bits required to store the component.
    fn count_bits(&self) -> usize;

    /// Writes the bit sequence to `BitSink`.
    ///
    /// # Errors
    ///
    /// This function returns error if `self` contains an invalid value that
    /// does not fit to FLAC's bitstream format, or if a `BitSink` method
    /// returned an error.
    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>>;

    /// Test utility for obtaining bits as a [`Vec`] of [`u8`].
    #[cfg(test)]
    fn to_bytes(&self) -> Vec<u8> {
        let mut sink = MemSink::<u8>::new();
        self.write(&mut sink).expect("No error expected");
        sink.into_inner()
    }

    /// Test utility for obtaining bits as eight-bit separated `String`.
    #[cfg(test)]
    fn to_bitstring(&self) -> String {
        let mut sink = MemSink::<u8>::new();
        self.write(&mut sink).expect("No error expected");
        sink.to_bitstring()
    }

    #[cfg(test)]
    /// Checks if the number of bits actually written equals to the expected number of bits.
    ///
    /// # Errors
    ///
    /// If the check passed i.e. the number of bits actually written is as same as the expected
    /// number, it returns `Ok(bits)`. Otherwise, it returns `Err((expected_bits, actual_bits))`.
    fn verify_bit_counter(&self) -> Result<usize, (usize, usize)> {
        let expected = self.count_bits();
        let mut sink = MemSink::<u8>::new();
        self.write(&mut sink).expect("No error expected");
        if expected == sink.len() {
            Ok(expected)
        } else {
            Err((expected, sink.len()))
        }
    }
}

/// Lookup table for `encode_to_utf8like`.
const UTF8_HEADS: [u8; 7] = [0x80, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC, 0xFE];

/// Encodes the given integer into UTF-8-like byte sequence.
///
/// # Panics
///
/// It will not panic.
///
/// # Errors
///
/// It returns an error if `val` exceeds 36-bit value.
#[inline]
pub fn encode_to_utf8like(val: u64) -> Result<heapless::Vec<u8, 7>, RangeError> {
    let val_size = u64::BITS as usize;
    let code_bits: usize = val_size - val.leading_zeros() as usize;
    let mut ret = heapless::Vec::new();
    if code_bits <= 7 {
        ret.push(val as u8).unwrap();
    } else if code_bits > 36 {
        return Err(RangeError::from_display(
            "input",
            "cannot exceed 36 bits.",
            &val,
        ));
    } else {
        // capacity = n * 6 + 6 - n = n * 5 + 6
        // n = ceil(capacity - 6 / 5)
        let trailing_bytes: usize = (code_bits - 2) / 5;
        assert!(trailing_bytes >= 1);
        assert!(trailing_bytes <= 6);
        let capacity = trailing_bytes * 6 + 6 - trailing_bytes;
        assert!(capacity >= code_bits);

        let first_bits = 6 - trailing_bytes;
        let mut val = val << (val_size - capacity);
        let head_byte: u8 = if trailing_bytes == 6 {
            0xFEu8
        } else {
            UTF8_HEADS[trailing_bytes] | ((val >> (64 - first_bits)) & 0xFF) as u8
        };
        ret.push(head_byte).unwrap();
        val <<= first_bits;

        for _i in 0..trailing_bytes {
            let b = 0x80u8 | (val >> 58) as u8;
            ret.push(b).unwrap();
            val <<= 6;
        }
    }
    Ok(ret)
}

/// Computes the number of bytes required for UTF-8-like encoding of `val`.
const fn utf8like_bytesize(val: usize) -> usize {
    let val_size = usize::BITS as usize;
    let code_bits: usize = val_size - val.leading_zeros() as usize;
    if code_bits <= 7 {
        1
    } else {
        1 + (code_bits - 2) / 5
    }
}

impl BitRepr for Stream {
    #[inline]
    fn count_bits(&self) -> usize {
        let mut ret = 32 + self.stream_info_block().count_bits();
        for elem in self.metadata() {
            ret += elem.count_bits();
        }
        for frame in self.frames() {
            ret += frame.count_bits();
        }
        ret
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write_bytes_aligned(&[0x66, 0x4c, 0x61, 0x43]) // fLaC
            .map_err(OutputError::<S>::from_sink)?;
        self.stream_info_block().write(dest)?;
        for elem in self.metadata() {
            elem.write(dest)?;
        }
        for frame in self.frames() {
            frame.write(dest)?;
        }
        Ok(())
    }
}

impl BitRepr for MetadataBlock {
    #[inline]
    fn count_bits(&self) -> usize {
        // This is a bit tricky, but `self.data.count_bits` doesn't include the
        // number of bits used for storing typetag.
        32 + self.data.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        let block_type: u8 = self.data.typetag() + if self.is_last { 0x80 } else { 0x00 };
        dest.write(block_type)
            .map_err(OutputError::<S>::from_sink)?;
        let data_size: u32 = (self.data.count_bits() / 8) as u32;
        dest.write_lsbs(data_size, 24)
            .map_err(OutputError::<S>::from_sink)?;
        self.data.write(dest)?;
        Ok(())
    }
}

impl BitRepr for MetadataBlockData {
    #[inline]
    fn count_bits(&self) -> usize {
        match self {
            Self::StreamInfo(info) => info.count_bits(),
            Self::Unknown { data, .. } => data.len() * 8,
        }
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        match self {
            Self::StreamInfo(info) => info.write(dest)?,
            Self::Unknown { data, .. } => {
                dest.write_bytes_aligned(data)
                    .map_err(OutputError::<S>::from_sink)?;
            }
        };
        Ok(())
    }
}

impl BitRepr for StreamInfo {
    #[inline]
    fn count_bits(&self) -> usize {
        272
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write::<u16>(self.min_block_size() as u16)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write::<u16>(self.max_block_size() as u16)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.min_frame_size() as u32, 24)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.max_frame_size() as u32, 24)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.sample_rate() as u32, 20)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs((self.channels() - 1) as u8, 3)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs((self.bits_per_sample() - 1) as u8, 5)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.total_samples() as u64, 36)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_bytes_aligned(self.md5_digest())
            .map_err(OutputError::<S>::from_sink)?;
        Ok(())
    }
}

reusable!(FRAME_CRC_BUFFER: (MemSink<u64>, Vec<u8>) = (MemSink::new(), Vec::new()));
pub static FRAME_CRC: crc::Crc<u16, crc::Table<16>> =
    crc::Crc::<u16, crc::Table<16>>::new(&CRC_16_FLAC);

impl BitRepr for Frame {
    #[inline]
    fn count_bits(&self) -> usize {
        self.precomputed_bitstream().as_ref().map_or_else(
            || {
                let header = self.header().count_bits();
                let body: usize = self.subframes().iter().map(BitRepr::count_bits).sum();

                let aligned = ((header + body + 7) >> 3) << 3;
                let footer = 16;
                aligned + footer
            },
            |bytes| bytes.len() << 3,
        )
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        if let Some(ref bytes) = self.precomputed_bitstream() {
            dest.write_bytes_aligned(bytes)
                .map_err(OutputError::<S>::from_sink)?;
            Ok(())
        } else {
            reuse!(FRAME_CRC_BUFFER, |buf: &mut (MemSink<u64>, Vec<u8>)| {
                let frame_sink = &mut buf.0;
                let bytebuf = &mut buf.1;

                frame_sink.clear();
                frame_sink.reserve(self.count_bits());

                self.header()
                    .write(frame_sink)
                    .map_err(OutputError::<S>::ignore_sink_error)?;
                for sub in self.subframes() {
                    sub.write(frame_sink)
                        .map_err(OutputError::<S>::ignore_sink_error)?;
                }
                frame_sink.align_to_byte().unwrap();

                bytebuf.resize(frame_sink.len() >> 3, 0u8);
                frame_sink.write_to_byte_slice(&mut *bytebuf);

                dest.write_bytes_aligned(&*bytebuf).unwrap();

                dest.write(FRAME_CRC.checksum(&*bytebuf))
                    .map_err(OutputError::<S>::from_sink)
            })
        }
    }
}

impl BitRepr for ChannelAssignment {
    #[inline]
    fn count_bits(&self) -> usize {
        4
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        match *self {
            Self::Independent(ch) => {
                if ch > 8 {
                    return Err(RangeError::from_display("#channel", "cannot exceed 8", &ch).into());
                }
                dest.write_lsbs(ch - 1, 4)
                    .map_err(OutputError::<S>::from_sink)?;
            }
            Self::LeftSide => {
                dest.write_lsbs(0x8u64, 4)
                    .map_err(OutputError::<S>::from_sink)?;
            }
            Self::RightSide => {
                dest.write_lsbs(0x9u64, 4)
                    .map_err(OutputError::<S>::from_sink)?;
            }
            Self::MidSide => {
                dest.write_lsbs(0xAu64, 4)
                    .map_err(OutputError::<S>::from_sink)?;
            }
        }
        Ok(())
    }
}

reusable!(HEADER_CRC_BUFFER: ByteSink = ByteSink::new());
pub static HEADER_CRC: crc::Crc<u8, crc::Table<16>> =
    crc::Crc::<u8, crc::Table<16>>::new(&CRC_8_FLAC);

impl BitRepr for FrameHeader {
    #[inline]
    fn count_bits(&self) -> usize {
        let mut ret = 40;
        if self.is_variable_blocking() {
            ret += 8 * utf8like_bytesize(self.start_sample_number() as usize);
        } else {
            ret += 8 * utf8like_bytesize(self.frame_number() as usize);
        }
        ret += self.block_size_spec().count_extra_bits();
        ret += self.sample_rate_spec().count_extra_bits();
        ret
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        reuse!(HEADER_CRC_BUFFER, |header_buffer: &mut ByteSink| {
            header_buffer.clear();
            header_buffer.reserve(self.count_bits());

            // sync-code + reserved 1-bit + variable-block indicator
            let header_word = 0xFFF8u16 + u16::from(self.is_variable_blocking());
            // ^ `from` converts true to 1 and false to 0.
            header_buffer.write_lsbs(header_word, 16).unwrap();

            // block_size_spec tag + 4-bit sample rate specifier.
            header_buffer
                .write_lsbs(
                    self.block_size_spec().tag() << 4 | self.sample_rate_spec().tag(),
                    8,
                )
                .unwrap();
            self.channel_assignment()
                .write(header_buffer)
                .map_err(OutputError::<S>::ignore_sink_error)?;

            // sample size specifier + 1-bit reserved (zero)
            header_buffer
                .write_lsbs(self.sample_size_spec().into_tag() << 1, 4)
                .unwrap();

            if self.is_variable_blocking() {
                let v = encode_to_utf8like(self.start_sample_number())?;
                header_buffer.write_bytes_aligned(&v).unwrap();
            } else {
                let v = encode_to_utf8like(self.frame_number().into())?;
                header_buffer.write_bytes_aligned(&v).unwrap();
            }
            self.block_size_spec()
                .write_extra_bits(header_buffer)
                .unwrap();
            self.sample_rate_spec()
                .write_extra_bits(header_buffer)
                .unwrap();

            dest.write_bytes_aligned(header_buffer.as_slice())
                .map_err(OutputError::<S>::from_sink)?;
            dest.write(HEADER_CRC.checksum(header_buffer.as_slice()))
                .map_err(OutputError::<S>::from_sink)?;
            Ok(())
        })
    }
}

impl BitRepr for SubFrame {
    #[inline]
    fn count_bits(&self) -> usize {
        match self {
            Self::Verbatim(c) => c.count_bits(),
            Self::Constant(c) => c.count_bits(),
            Self::FixedLpc(c) => c.count_bits(),
            Self::Lpc(c) => c.count_bits(),
        }
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        match self {
            Self::Verbatim(c) => c.write(dest),
            Self::Constant(c) => c.write(dest),
            Self::FixedLpc(c) => c.write(dest),
            Self::Lpc(c) => c.write(dest),
        }
    }
}

impl BitRepr for Constant {
    #[inline]
    fn count_bits(&self) -> usize {
        8 + self.bits_per_sample()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write(0u8).map_err(OutputError::<S>::from_sink)?;
        dest.write_twoc(self.dc_offset(), self.bits_per_sample())
            .map_err(OutputError::<S>::from_sink)?;
        Ok(())
    }
}

impl BitRepr for Verbatim {
    #[inline]
    fn count_bits(&self) -> usize {
        Self::count_bits_from_metadata(self.samples().len(), self.bits_per_sample())
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write(0x02u8).map_err(OutputError::<S>::from_sink)?;
        for i in 0..self.samples().len() {
            dest.write_twoc(self.samples()[i], self.bits_per_sample())
                .map_err(OutputError::<S>::from_sink)?;
        }
        Ok(())
    }
}

impl BitRepr for FixedLpc {
    #[inline]
    fn count_bits(&self) -> usize {
        8 + self.bits_per_sample() * self.order() + self.residual().count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        let head_byte = 0x10u8 | (self.order() << 1) as u8;
        dest.write(head_byte).map_err(OutputError::<S>::from_sink)?;
        for v in self.warm_up() {
            dest.write_twoc(*v, self.bits_per_sample())
                .map_err(OutputError::<S>::from_sink)?;
        }
        self.residual().write(dest)
    }
}

impl BitRepr for Lpc {
    #[inline]
    fn count_bits(&self) -> usize {
        let warm_up_bits = self.bits_per_sample() * self.order();
        8 + warm_up_bits
            + 4
            + 5
            + self.parameters().precision() * self.order()
            + self.residual().count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        let head_byte = 0x40 | (((self.order() - 1) as u8) << 1);
        dest.write(head_byte).map_err(OutputError::<S>::from_sink)?;

        for i in 0..self.order() {
            dest.write_twoc(self.warm_up()[i], self.bits_per_sample())
                .map_err(OutputError::<S>::from_sink)?;
        }

        assert!((self.parameters().precision() as u8) < 16u8);
        dest.write_lsbs((self.parameters().precision() - 1) as u64, 4)
            .map_err(OutputError::<S>::from_sink)?;

        // FLAC reference decoder doesn't support this.
        assert!(self.parameters().shift() >= 0);
        dest.write_twoc(self.parameters().shift(), 5)
            .map_err(OutputError::<S>::from_sink)?;

        for ref_coef in &self.parameters().coefs() {
            debug_assert!(*ref_coef < (1 << (self.parameters().precision() - 1)));
            debug_assert!(*ref_coef >= -(1 << (self.parameters().precision() - 1)));
            dest.write_twoc(*ref_coef, self.parameters().precision())
                .map_err(OutputError::<S>::from_sink)?;
        }

        self.residual().write(dest)
    }
}

const RESIDUAL_WRITE_UNROLL_N: usize = 4;
impl BitRepr for Residual {
    #[inline]
    fn count_bits(&self) -> usize {
        let nparts = 1usize << self.partition_order();
        let quotient_bits: usize = self.sum_quotients() + self.block_size() - self.warmup_length();

        let mut remainder_bits: usize =
            self.sum_rice_params() * (self.block_size() >> self.partition_order());
        remainder_bits -= self.warmup_length() * self.rice_params()[0] as usize;
        2 + 4 + nparts * 4 + quotient_bits + remainder_bits
    }

    /// Writes `Residual` to the [`BitSink`].
    ///
    /// This is the most inner-loop of the output part of the encoder, so
    /// computational efficiency is prioritized more than readability.
    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        // The number of partitions with 00 (indicating 4-bit mode) prepended.
        dest.write_lsbs(self.partition_order() as u32, 6)
            .map_err(OutputError::<S>::from_sink)?;
        let nparts = 1usize << self.partition_order();

        // unforunatelly, the overhead due to the use of iterators is visible
        // here. so we back-off to integer-based loops. (drawback of index-loop
        // is boundary check, but this will be skipped in release builds.)
        let part_len = self.block_size() >> self.partition_order();
        let mut p = 0;
        let mut offset = 0;
        while p < nparts {
            let rice_p = self.rice_params()[p];
            dest.write_lsbs(rice_p, 4)
                .map_err(OutputError::<S>::from_sink)?;
            let start = max(self.warmup_length(), offset);
            offset += part_len;
            let end = offset;

            let startbit: u32 = 1u32 << rice_p;
            let rice_p_plus_1: usize = (rice_p + 1) as usize;
            let mut t0 = start;
            while t0 < end {
                try_repeat!(
                    offset to RESIDUAL_WRITE_UNROLL_N;
                    while t0 + offset < end => {
                        let t = t0 + offset;
                        let q = self.quotients()[t] as usize;
                        let r_plus_startbit =
                            (self.remainders()[t] | startbit) << (32 - rice_p_plus_1);
                        dest.write_zeros(q)?;
                        dest.write_msbs(r_plus_startbit, rice_p_plus_1)?;
                        Ok::<(), S::Error>(())
                    }
                )
                .map_err(OutputError::<S>::from_sink)?;
                t0 += RESIDUAL_WRITE_UNROLL_N;
            }
            p += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::BlockSizeSpec;
    use crate::component::SampleRateSpec;
    use crate::component::SampleSizeSpec;
    use crate::error::Verify;
    use crate::test_helper::make_random_residual;
    use crate::test_helper::make_verbatim_frame;

    #[test]
    fn write_empty_stream() {
        let stream = Stream::new(44100, 2, 16).expect("`Stream::new` should not fail.");
        let stream_bytes = stream.to_bytes();
        assert_eq!(
            stream_bytes.len() * 8,
            32 // fLaC
      + 1 + 7 + 24 // METADATA_BLOCK_HEADER
      + 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128 // METADATA_BLOCK_STREAMINFO
        );
        assert_eq!(stream.count_bits(), stream_bytes.len() * 8);
    }

    #[test]
    fn write_stream_info() {
        let stream_info = StreamInfo::new(44100, 2, 16).expect("`Stream::new` should not fail.");
        let stream_info_bytes = stream_info.to_bytes();
        assert_eq!(
            stream_info_bytes.len() * 8,
            16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128
        );
        assert_eq!(stream_info.count_bits(), stream_info_bytes.len() * 8);
    }

    #[test]
    fn write_frame_header() {
        let header = FrameHeader::from_specs(
            BlockSizeSpec::from_size(2304),
            ChannelAssignment::Independent(2),
            SampleSizeSpec::Unspecified,
            SampleRateSpec::Unspecified,
        );
        header.to_bytes(); // just checking it doesn't panic.

        // test with canonical frame
        let header = FrameHeader::from_specs(
            BlockSizeSpec::from_size(192),
            ChannelAssignment::Independent(2),
            SampleSizeSpec::Unspecified,
            SampleRateSpec::Unspecified,
        );
        header
            .verify_bit_counter()
            .expect("`FrameHeader::count_bits` should be accurate.");
        assert_eq!(
            header.to_bitstring(),
            concat!(
                "11111111_111110", // sync
                "01_",             // reserved/ blocking strategy (const in this impl)
                "00010000_",       // block size/ sample_rate (0=header)
                "00010000_",       // channel/ bps (0=header)/ reserved
                "00000000_",       // sample number
                "01101001",        // crc8
            )
        );

        assert_eq!(header.count_bits(), 48);
    }

    #[test]
    fn channel_assignment_encoding() {
        let ch = ChannelAssignment::Independent(8);
        assert_eq!(ch.to_bitstring(), "0111****");
        let ch = ChannelAssignment::RightSide;
        assert_eq!(ch.to_bitstring(), "1001****");
        ch.verify_bit_counter()
            .expect("`ChanneAssignment::count_bits` should be accurate.");
    }

    #[test]
    fn write_verbatim_frame() {
        let nchannels: usize = 3;
        let nsamples: usize = 17;
        let bits_per_sample: usize = 16;
        let stream_info = StreamInfo::new(16000, nchannels, bits_per_sample)
            .expect("`StreamInfo::new` should not return error");
        let framebuf = vec![-1i32; nsamples * nchannels];
        let frame = make_verbatim_frame(&stream_info, &framebuf, 0);
        frame
            .header()
            .verify_bit_counter()
            .expect("`FrameHeader::count_bits` should be accurate.");

        for ch in 0..3 {
            frame.subframe(ch).unwrap().to_bytes();
            frame
                .subframe(ch)
                .unwrap()
                .verify_bit_counter()
                .expect("`SubFrame::count_bits` should be accurate.");
        }

        frame
            .verify_bit_counter()
            .expect("`Frame::count_bits` should be accurate.");
    }

    #[test]
    #[allow(clippy::cast_lossless)]
    fn bit_count_residual() {
        let residual = make_random_residual(rand::thread_rng(), 0);
        residual
            .verify()
            .expect("should construct a valid Residual");
        residual
            .verify_bit_counter()
            .expect("`Residual::count_bits` should be accurate");
    }
}
