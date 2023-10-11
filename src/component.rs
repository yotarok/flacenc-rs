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

//! Components to be written in the output file.

use std::cell::RefCell;
use std::cmp::max;
use std::cmp::min;

use super::bitsink::BitSink;
use super::bitsink::ByteSink;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::MAX_CHANNELS;
use super::error::OutputError;
use super::error::RangeError;
use super::lpc;
use super::rice;

// re-export quantized parameters
pub use lpc::QuantizedParameters;

const CRC_8_FLAC: crc::Algorithm<u8> = crc::CRC_8_SMBUS;
const CRC_16_FLAC: crc::Algorithm<u16> = crc::CRC_16_UMTS;

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
}

/// Lookup table for `encode_to_utf8like`.
const UTF8_HEADS: [u8; 7] = [0x80, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC, 0xFE];

/// Encodes the given integer into UTF-8-like byte sequence.
#[inline]
fn encode_to_utf8like(val: u64) -> Result<heapless::Vec<u8, 7>, RangeError> {
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

/// Returns header bits for the given block size.
const fn block_size_spec(block_size: u16) -> (u8, u16, usize) {
    match block_size {
        192 => (0x01, 0x0000, 0),
        576 | 1152 | 2304 | 4608 => {
            let n: usize = block_size as usize / 576;
            let pow = n.trailing_zeros() as u8;
            let head: u8 = 2 + pow;
            (head, 0x0000, 0)
        }
        256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 => {
            let n: usize = block_size as usize / 256;
            let pow = n.trailing_zeros() as u8;
            let head: u8 = 8 + pow;
            (head, 0x0000, 0)
        }
        _ => {
            if block_size < 256 {
                let footer: u8 = (block_size - 1) as u8;
                (0x06, footer as u16, 8)
            } else {
                // block_size is always < 65536 as it is u16.
                let footer: u16 = block_size - 1;
                (0x07, footer, 16)
            }
        }
    }
}

/// [`STREAM`](https://xiph.org/flac/format.html#stream) component.
pub struct Stream {
    stream_info: MetadataBlock,
    metadata: Vec<MetadataBlock>,
    frames: Vec<Frame>,
}

impl Stream {
    /// Constructs `Stream` with the given meta information.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let stream = Stream::new(16000, 1, 16);
    /// assert_eq!(stream.stream_info().channels(), 1);
    /// ```
    pub fn new(sample_rate: usize, channels: usize, bits_per_sample: usize) -> Self {
        let stream_info = StreamInfo::new(sample_rate, channels, bits_per_sample);
        Self {
            stream_info: MetadataBlock::from_stream_info(stream_info, true),
            metadata: vec![],
            frames: vec![],
        }
    }

    /// Returns a reference to `StreamInfo` associated with `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is corrupted by manually modifying fields.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let stream = Stream::new(16000, 1, 24);
    /// assert_eq!(stream.stream_info().bits_per_sample(), 24);
    /// ```
    pub fn stream_info(&self) -> &StreamInfo {
        // This "allow" is required because `MetadataBlockData` variants other
        // than `StreamInfo` are not implemented yet.
        #[allow(unreachable_patterns)]
        match self.stream_info.data {
            MetadataBlockData::StreamInfo(ref info) => info,
            _ => panic!("Stream is not properly initialized."),
        }
    }

    /// Returns a mutable reference to `StreamInfo` associated with `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is corrupted by manually modifying fields.
    pub(crate) fn stream_info_mut(&mut self) -> &mut StreamInfo {
        // This "allow" is required because `MetadataBlockData` variants other
        // than `StreamInfo` are not implemented yet.
        #[allow(unreachable_patterns)]
        match self.stream_info.data {
            MetadataBlockData::StreamInfo(ref mut info) => info,
            _ => panic!("Stream is not properly initialized."),
        }
    }

    /// Appends `Frame` to this `Stream` and updates `StreamInfo`.
    ///
    /// This also updates frame statistics in `stream_info` but does not update
    /// MD5 checksums and the total number of samples.  For updating those,
    /// please manually call `set_total_samples` and `set_md5_digest`,
    /// respectively, via `self.stream_info_mut`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// let mut stream = Stream::new(16000, 1, 24);
    /// stream.add_frame(frame);
    /// assert_eq!(stream.frame_count(), 1);
    /// ```
    pub fn add_frame(&mut self, frame: Frame) {
        // TODO: Add example section to the doc. Currently, it's not
        // straightforward as we don't have a public method for generating
        // a single `Frame` except for encoders.
        self.stream_info_mut().update_frame_info(&frame);
        self.frames.push(frame);
    }

    /// Returns `Frame` for the given frame number.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let frame0 = stream.frame(0).expect("0-th frame is not found.");
    /// let frame19 = stream.frame(19).expect("19-th frame is not found.");
    /// assert!(frame0.count_bits() > 0);
    /// assert!(frame19.count_bits() > 0);
    /// ```
    pub fn frame(&self, n: usize) -> Option<&Frame> {
        self.frames.get(n)
    }

    /// Returns the number of `Frame`s in the stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// assert_eq!(stream.frame_count(), 200);
    /// ```
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns `Frame`s as a slice.
    #[allow(dead_code)]
    pub(crate) fn frames(&self) -> &[Frame] {
        &self.frames
    }
}

impl BitRepr for Stream {
    #[inline]
    fn count_bits(&self) -> usize {
        let mut ret = 32 + self.stream_info.count_bits();
        for elem in &self.metadata {
            ret += elem.count_bits();
        }
        for frame in &self.frames {
            ret += frame.count_bits();
        }
        ret
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write_bytes_aligned(&[0x66, 0x4c, 0x61, 0x43]) // fLaC
            .map_err(OutputError::<S>::from_sink)?;
        self.stream_info.write(dest)?;
        for elem in &self.metadata {
            elem.write(dest)?;
        }
        for frame in &self.frames {
            frame.write(dest)?;
        }
        Ok(())
    }
}

/// [`METADATA_BLOCK`](https://xiph.org/flac/format.html#metadata_block) component.
#[derive(Clone, Debug)]
struct MetadataBlock {
    // METADATA_BLOCK_HEADER
    is_last: bool,
    block_type: MetadataBlockType,
    // METADATA_BLOCK_DATA
    data: MetadataBlockData,
}

impl MetadataBlock {
    const fn from_stream_info(info: StreamInfo, is_last: bool) -> Self {
        Self {
            is_last,
            block_type: MetadataBlockType::StreamInfo,
            data: MetadataBlockData::StreamInfo(info),
        }
    }
}

impl BitRepr for MetadataBlock {
    #[inline]
    fn count_bits(&self) -> usize {
        32 + self.data.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        let block_type: u8 = self.block_type as u8 + if self.is_last { 0x80 } else { 0x00 };
        dest.write(block_type)
            .map_err(OutputError::<S>::from_sink)?;
        let data_size: u32 = (self.data.count_bits() / 8) as u32;
        dest.write_lsbs(data_size, 24)
            .map_err(OutputError::<S>::from_sink)?;
        self.data.write(dest)?;
        Ok(())
    }
}

/// Enum for `BLOCK_TYPE` in `METADATA_BLOCK_HEADER`.
#[allow(dead_code)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
enum MetadataBlockType {
    StreamInfo = 0,
    Padding,
    Application,
    SeekTable,
    VorbisComment,
    CueSheet,
    Picture,
    ReservedBegin = 7,
    ReservedEnd = 126,
    Invalid = 127,
}

#[derive(Clone, Debug)]
/// Enum that covers all variants of `METADATA_BLOCK`.
///
/// Currently only `StreamInfo` is covered in this enum.
#[non_exhaustive]
enum MetadataBlockData {
    StreamInfo(StreamInfo),
}

impl BitRepr for MetadataBlockData {
    #[inline]
    fn count_bits(&self) -> usize {
        match self {
            Self::StreamInfo(info) => info.count_bits(),
        }
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        match self {
            Self::StreamInfo(info) => info.write(dest)?,
        }
        Ok(())
    }
}

/// [`METADATA_BLOCK_STREAM_INFO`](https://xiph.org/flac/format.html#metadata_block_streaminfo) component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamInfo {
    min_block_size: u16, // 16 bits: Minimum block size in samples.
    max_block_size: u16, // 16 bits: Maximum block size in samples.
    min_frame_size: u32, // 24 bits: Minimum frame size in bytes.
    max_frame_size: u32, // 24 bits: Maximum frame size in bytes.
    sample_rate: u32,    // 20 bits: Sample rate in Hz.
    channels: u8,        // 3 bits: will be written with a bias (-1)
    bits_per_sample: u8, // 5 bits: will be written with a bias (-1)
    total_samples: u64,  // 36 bits: Can be zero (unknown)
    md5: [u8; 16],
}

impl StreamInfo {
    /// Constructs new `StreamInfo`.
    ///
    /// For unspecified fields, the following default values are used:
    ///
    /// -  `min_block_size`: `u16::MAX`,
    /// -  `max_block_size`: `0`,
    /// -  `min_frame_size`: `u32::MAX`,
    /// -  `max_frame_size`: `0`,
    /// -  `total_samples`: `0`,
    /// -  `md5_digest`: all-zero (indicating verification disabled.)
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16);
    /// assert_eq!(info.max_frame_size(), 0);
    /// ```
    pub fn new(sample_rate: usize, channels: usize, bits_per_sample: usize) -> Self {
        Self {
            min_block_size: u16::MAX,
            max_block_size: 0,
            min_frame_size: u32::MAX,
            max_frame_size: 0,
            sample_rate: sample_rate as u32,
            channels: channels as u8,
            bits_per_sample: bits_per_sample as u8,
            total_samples: 0,
            md5: [0; 16],
        }
    }

    /// Updates `StreamInfo` with values from the given Frame.
    ///
    /// This function updates `{min|max}_{block|frame}_size` and
    /// `total_samples`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let mut info = StreamInfo::new(16000, 2, 16);
    ///
    /// for n in 0..other_stream.frame_count() {
    ///     info.update_frame_info(other_stream.frame(n).unwrap());
    /// }
    /// assert_eq!(info.max_block_size(), 160);
    /// assert_eq!(info.min_block_size(), 160);
    ///
    /// // `Frame` doesn't hold the original signal length, so `total_samples`
    /// // becomes a multuple of block_size == 160.
    /// assert_eq!(info.total_samples(), 31360);
    /// ```
    pub fn update_frame_info(&mut self, frame: &Frame) {
        let block_size = frame.block_size() as u16;
        self.min_block_size = min(block_size, self.min_block_size);
        self.max_block_size = max(block_size, self.max_block_size);
        let frame_size_in_bytes = (frame.count_bits() / 8) as u32;
        self.min_frame_size = min(frame_size_in_bytes, self.min_frame_size);
        self.max_frame_size = max(frame_size_in_bytes, self.max_frame_size);

        self.total_samples += u64::from(block_size);
    }

    /// Returns `min_frame_size` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let mut info = StreamInfo::new(16000, 2, 16);
    ///
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// info.update_frame_info(other_stream.frame(0).unwrap());
    ///
    /// assert!(info.min_frame_size() > 0);
    /// ```
    pub fn min_frame_size(&self) -> usize {
        self.min_frame_size as usize
    }

    /// Returns `max_frame_size` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16);
    ///
    /// assert_eq!(info.max_frame_size(), 0);
    /// ```
    pub fn max_frame_size(&self) -> usize {
        self.max_frame_size as usize
    }

    /// Returns `min_block_size` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let mut info = StreamInfo::new(16000, 2, 16);
    ///
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// info.update_frame_info(other_stream.frame(0).unwrap());
    ///
    /// assert_eq!(info.min_block_size(), 160);
    /// ```
    pub fn min_block_size(&self) -> usize {
        self.min_block_size as usize
    }

    /// Returns `max_block_size` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16);
    ///
    /// assert_eq!(info.max_block_size(), 0);
    /// ```
    pub fn max_block_size(&self) -> usize {
        self.max_block_size as usize
    }

    /// Returns `sample_rate` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16);
    /// assert_eq!(info.sample_rate(), 16000);
    /// ```
    pub fn sample_rate(&self) -> usize {
        self.sample_rate as usize
    }

    /// Returns `channels` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16);
    /// assert_eq!(info.channels(), 2);
    /// ```
    pub fn channels(&self) -> usize {
        self.channels as usize
    }

    /// Returns `bits_per_sample` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16);
    /// assert_eq!(info.bits_per_sample(), 16);
    /// ```
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }

    /// Returns `total_samples` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let mut info = StreamInfo::new(16000, 2, 16);
    ///
    /// for n in 0..other_stream.frame_count() {
    ///     info.update_frame_info(other_stream.frame(n).unwrap());
    /// }
    ///
    /// // `Frame` doesn't hold the original signal length, so `total_samples`
    /// // becomes a multuple of block_size == 160.
    /// assert_eq!(info.total_samples(), 31360);
    /// ```
    pub fn total_samples(&self) -> usize {
        self.total_samples as usize
    }

    /// Sets `total_samples` field.
    ///
    /// `total_samples` is updated during the encoding. However, since `Frame`
    /// only knows its frame size, the effective number of samples is not
    /// visible after paddings.  Similar to `set_md5_digest`, this field should
    /// be finalized by propagating information from `Context`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::*;
    /// let mut ctx = source::Context::new(16, 2);
    /// let mut info = StreamInfo::new(16000, 2, 16);
    /// ctx.update(&[0x0000_0FFFi32; 246], 123);
    /// info.set_total_samples(ctx.total_samples());
    /// assert_ne!(info.total_samples(), 246);
    /// ```
    pub fn set_total_samples(&mut self, n: usize) {
        self.total_samples = n as u64;
    }

    /// Returns `md5_digest` field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let mut info = StreamInfo::new(16000, 2, 16);
    ///
    /// assert_eq!(info.md5_digest(), &[0u8; 16]); // default
    ///
    /// for n in 0..other_stream.frame_count() {
    ///     info.update_frame_info(other_stream.frame(n).unwrap());
    /// }
    ///
    /// // `update_frame_info` doesn't update MD5
    /// assert_eq!(info.md5_digest(), &[0u8; 16]);
    /// ```
    pub fn md5_digest(&self) -> &[u8; 16] {
        &self.md5
    }

    /// Resets MD5 digest value by the given slice.
    ///
    /// MD5 computation is not performed in in `update_frame_info`, and is
    /// expected to be done externally (by `source::Context`). This function
    /// is called to set MD5 bytes after we read all input samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::*;
    /// let mut ctx = source::Context::new(16, 2);
    /// let mut info = StreamInfo::new(16000, 2, 16);
    /// assert_eq!(info.md5_digest(), &[0x00u8; 16]);
    /// ctx.update(&[0x0000_0FFFi32; 256], 128);
    /// info.set_md5_digest(&ctx.md5_digest());
    /// assert_ne!(info.md5_digest(), &[0x00u8; 16]);
    /// ```
    pub fn set_md5_digest(&mut self, digest: &[u8; 16]) {
        self.md5.copy_from_slice(digest);
    }
}

impl BitRepr for StreamInfo {
    #[inline]
    fn count_bits(&self) -> usize {
        272
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write::<u16>(self.min_block_size)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write::<u16>(self.max_block_size)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.min_frame_size, 24)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.max_frame_size, 24)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.sample_rate, 20)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.channels - 1, 3)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.bits_per_sample - 1, 5)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_lsbs(self.total_samples, 36)
            .map_err(OutputError::<S>::from_sink)?;
        dest.write_bytes_aligned(&self.md5)
            .map_err(OutputError::<S>::from_sink)?;
        Ok(())
    }
}

/// [`FRAME`](https://xiph.org/flac/format.html#frame) component.
#[derive(Clone, Debug)]
pub struct Frame {
    header: FrameHeader,
    subframes: heapless::Vec<SubFrame, MAX_CHANNELS>,
    precomputed_bitstream: Option<Vec<u8>>,
}

impl Frame {
    /// Returns block size of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// assert_eq!(frame.block_size(), 160);
    /// ```
    pub fn block_size(&self) -> usize {
        self.header.block_size as usize
    }

    /// Constructs an empty `Frame`.
    ///
    /// This makes an invalid `Frame`; therefore this shouldn't be "pub" so far.
    pub(crate) fn new(ch_info: ChannelAssignment, offset: usize, block_size: usize) -> Self {
        let header = FrameHeader::new(block_size, ch_info, offset);
        Self {
            header,
            subframes: heapless::Vec::new(),
            precomputed_bitstream: None,
        }
    }

    /// Constructs Frame from `FrameHeader` and `SubFrame`s.
    pub(crate) fn from_parts<I>(header: FrameHeader, subframes: I) -> Self
    where
        I: Iterator<Item = SubFrame>,
    {
        Self {
            header,
            subframes: subframes.collect(),
            precomputed_bitstream: None,
        }
    }

    /// Deconstructs frame and transfers ownership of the data structs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// let (header, subframes) = frame.into_parts();
    ///
    /// assert_eq!(subframes.len(), 2);
    /// ```
    pub fn into_parts(self) -> (FrameHeader, heapless::Vec<SubFrame, MAX_CHANNELS>) {
        (self.header, self.subframes)
    }

    /// Adds subframe.
    ///
    /// # Panics
    ///
    /// Panics when the number of subframes added exceeded the `MAX_CHANNELS`.
    pub(crate) fn add_subframe(&mut self, subframe: SubFrame) {
        self.subframes
            .push(subframe)
            .expect("Exceeded maximum number of channels.");
    }

    /// Returns `FrameHeader` of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    /// assert_eq!(frame.header().block_size(), 160);
    /// ```
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns a mutable reference to `FrameHeader` of this frame.
    pub(crate) fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    /// Returns `SubFrame` for the given channel.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    /// for ch in 0..frame.subframe_count() {
    ///     assert!(frame.subframe(ch).is_some());
    /// }
    /// assert!(frame.subframe(2).is_none());
    /// ```
    #[inline]
    pub fn subframe(&self, ch: usize) -> Option<&SubFrame> {
        self.subframes.get(ch)
    }

    /// Returns the number of `SubFrame`s in this `Frame`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    /// for ch in 0..frame.subframe_count() {
    ///     assert!(frame.subframe(ch).is_some());
    /// }
    /// assert!(frame.subframe(2).is_none());
    /// ```
    #[inline]
    pub fn subframe_count(&self) -> usize {
        self.subframes.len()
    }

    /// Allocates precomputed bitstream buffer, and precomputes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let mut frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    /// let frame_cloned = frame.clone();
    ///
    /// // This method is idempotent, and doesn't affect to results.
    /// let mut sink = ByteSink::new();
    /// frame.precompute_bitstream();
    /// frame.write(&mut sink);
    ///
    /// let mut sink_ref = ByteSink::new();
    /// frame_cloned.write(&mut sink_ref);
    ///
    /// assert_eq!(sink.as_byte_slice(), sink_ref.as_byte_slice());
    /// ```
    pub fn precompute_bitstream(&mut self) {
        if self.precomputed_bitstream.is_some() {
            return;
        }
        let mut dest = ByteSink::with_capacity(self.count_bits());
        if self.write(&mut dest).is_ok() {
            self.precomputed_bitstream = Some(dest.into_bytes());
        }
    }

    #[cfg(test)]
    const fn is_bitstream_precomputed(&self) -> bool {
        self.precomputed_bitstream.is_some()
    }
}

thread_local! {
    static FRAME_CRC_BUFFER: RefCell<ByteSink> = RefCell::new(ByteSink::new());
}

impl BitRepr for Frame {
    #[inline]
    fn count_bits(&self) -> usize {
        let header = self.header.count_bits();
        let body: usize = self.subframes.iter().map(BitRepr::count_bits).sum();

        let aligned = (header + body + 7) / 8 * 8;
        let footer = 16;
        aligned + footer
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        if let Some(ref bytes) = &self.precomputed_bitstream {
            dest.write_bytes_aligned(bytes)
                .map_err(OutputError::<S>::from_sink)?;
            Ok(())
        } else {
            FRAME_CRC_BUFFER.with(|frame_buffer| {
                let frame_buffer: &mut ByteSink = &mut frame_buffer.borrow_mut();
                frame_buffer.clear();
                frame_buffer.reserve(self.count_bits());

                self.header
                    .write(frame_buffer)
                    .map_err(OutputError::<S>::ignore_sink_error)?;
                for sub in &self.subframes {
                    sub.write(frame_buffer)
                        .map_err(OutputError::<S>::ignore_sink_error)?;
                }
                frame_buffer.align_to_byte().unwrap();

                dest.write_bytes_aligned(frame_buffer.as_byte_slice())
                    .unwrap();

                dest.write(
                    crc::Crc::<u16>::new(&CRC_16_FLAC).checksum(frame_buffer.as_byte_slice()),
                )
                .map_err(OutputError::<S>::from_sink)
            })
        }
    }
}

/// Enum for channel assignment in `FRAME_HEADER`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChannelAssignment {
    Independent(u8),
    LeftSide,
    RightSide,
    MidSide,
}

impl ChannelAssignment {
    /// Returns the number of extra bit required to store the channel samples.
    pub(crate) const fn bits_per_sample_offset(&self, ch: usize) -> usize {
        #[allow(clippy::match_same_arms, clippy::bool_to_int_with_if)]
        match *self {
            Self::Independent(_) => 0,
            Self::LeftSide => {
                if ch == 1 {
                    1 // side
                } else {
                    0 // left
                }
            }
            Self::RightSide => {
                if ch == 0 {
                    1 // side
                } else {
                    0 // right
                }
            }
            Self::MidSide => {
                if ch == 1 {
                    1 // side
                } else {
                    0 // mid
                }
            }
        }
    }

    #[inline]
    pub(crate) fn select_channels(
        &self,
        l: SubFrame,
        r: SubFrame,
        m: SubFrame,
        s: SubFrame,
    ) -> (SubFrame, SubFrame) {
        match *self {
            Self::Independent(_) => (l, r),
            Self::LeftSide => (l, s),
            Self::RightSide => (s, r),
            Self::MidSide => (m, s),
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

/// [`FRAME_HEADER`](https://xiph.org/flac/format.html#frame_header) component.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    variable_block_size: bool, // must be same in all frames
    block_size: u16,           // encoded with special function
    channel_assignment: ChannelAssignment,
    sample_size: u8,          // if set, it must be consistent with StreamInfo
    frame_number: u32,        // written when variable_block_size == false
    start_sample_number: u64, // written when variable_block_size == true
}

impl FrameHeader {
    pub(crate) const fn new(
        block_size: usize,
        channel_assignment: ChannelAssignment,
        start_sample_number: usize,
    ) -> Self {
        Self {
            variable_block_size: true,
            block_size: block_size as u16,
            channel_assignment,
            sample_size: 0,
            frame_number: 0,
            start_sample_number: start_sample_number as u64,
        }
    }

    /// Clear `variable_block_size` flag, and set `frame_number`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let mut header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// header.set_frame_number(12);
    ///
    /// let mut sink = ByteSink::new();
    /// header.write(&mut sink);
    ///
    /// // 16-th bit denotes blocking strategy and it should be 0 (fixed blocking mode)
    /// // after setting the frame number.
    /// assert_eq!(sink.as_byte_slice()[1] & 0x01u8, 0u8);
    /// assert_eq!(sink.as_byte_slice()[4], 12u8);
    /// ```
    pub fn set_frame_number(&mut self, frame_number: u32) {
        self.variable_block_size = false;
        self.frame_number = frame_number;
    }

    /// Overwrites channel assignment information of the frame.
    pub(crate) fn reset_channel_assignment(&mut self, channel_assignment: ChannelAssignment) {
        self.channel_assignment = channel_assignment;
    }

    /// Resets `sample_size` field from `StreamInfo`.
    ///
    /// This field must be specified for Claxon compatibility.
    pub(crate) fn reset_sample_size(&mut self, stream_info: &StreamInfo) {
        let bits = match stream_info.bits_per_sample {
            8 => 1,
            12 => 2,
            16 => 4,
            20 => 5,
            24 => 6,
            32 => 7,
            _ => 0,
        };
        self.sample_size = bits;
    }

    /// Returns block size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// assert_eq!(header.block_size(), 160);
    /// ```
    pub fn block_size(&self) -> usize {
        self.block_size as usize
    }

    /// Returns `ChannelAssignment` of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # use flacenc::test_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 1, 16000);
    /// let header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// // this is only used for stereo signal, and it will be always `Independent` for
    /// // non-stereo signals.
    /// assert_eq!(header.channel_assignment(), &ChannelAssignment::Independent(1));
    /// ```
    pub fn channel_assignment(&self) -> &ChannelAssignment {
        &self.channel_assignment
    }
}

thread_local! {
    static HEADER_CRC_BUFFER: RefCell<ByteSink> = RefCell::new(ByteSink::new());
}

impl BitRepr for FrameHeader {
    #[inline]
    fn count_bits(&self) -> usize {
        let mut ret = 40;
        if self.variable_block_size {
            ret += 8 * utf8like_bytesize(self.start_sample_number as usize);
        } else {
            ret += 8 * utf8like_bytesize(self.frame_number as usize);
        }
        let (_head, _foot, footsize) = block_size_spec(self.block_size);
        ret += footsize;
        ret
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        HEADER_CRC_BUFFER.with(|header_buffer| {
            {
                let dest: &mut ByteSink = &mut header_buffer.borrow_mut();
                dest.clear();
                dest.reserve(self.count_bits());

                // sync-code + reserved 1-bit + variable-block indicator
                let header_word = 0xFFF8u16 + u16::from(self.variable_block_size);
                // ^ `from` converts true to 1 and false to 0.
                dest.write_lsbs(header_word, 16).unwrap();

                let (head, foot, footsize) = block_size_spec(self.block_size);
                // head + 4-bit sample rate specifier.
                dest.write_lsbs(head << 4, 8).unwrap();
                self.channel_assignment
                    .write(dest)
                    .map_err(OutputError::<S>::ignore_sink_error)?;

                // sample size specifier + 1-bit reserved (zero)
                dest.write_lsbs(self.sample_size << 1, 4).unwrap();

                if self.variable_block_size {
                    let v = encode_to_utf8like(self.start_sample_number)?;
                    dest.write_bytes_aligned(&v).unwrap();
                } else {
                    let v = encode_to_utf8like(self.frame_number.into())?;
                    dest.write_bytes_aligned(&v).unwrap();
                }
                dest.write_lsbs(foot, footsize).unwrap();
            }

            dest.write_bytes_aligned(header_buffer.borrow().as_byte_slice())
                .map_err(OutputError::<S>::from_sink)?;
            dest.write(
                crc::Crc::<u8>::new(&CRC_8_FLAC).checksum(header_buffer.borrow().as_byte_slice()),
            )
            .map_err(OutputError::<S>::from_sink)?;
            Ok(())
        })
    }
}

/// [`SUBFRAME`](https://xiph.org/flac/format.html#subframe) component.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SubFrame {
    /// This variant contains `Constant` sub-frame.
    Constant(Constant),
    /// This variant contains `Verbatim` sub-frame.
    Verbatim(Verbatim),
    /// This variant contains `FixedLpc` sub-frame.
    FixedLpc(FixedLpc),
    /// This variant contains `Lpc` sub-frame.
    Lpc(Lpc),
}

impl From<Constant> for SubFrame {
    fn from(c: Constant) -> Self {
        Self::Constant(c)
    }
}

impl From<Verbatim> for SubFrame {
    fn from(c: Verbatim) -> Self {
        Self::Verbatim(c)
    }
}

impl From<FixedLpc> for SubFrame {
    fn from(c: FixedLpc) -> Self {
        Self::FixedLpc(c)
    }
}

impl From<Lpc> for SubFrame {
    fn from(c: Lpc) -> Self {
        Self::Lpc(c)
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

/// [`SUBFRAME_CONSTANT`](https://xiph.org/flac/format.html#subframe_constant) component.
#[derive(Clone, Debug)]
pub struct Constant {
    dc_offset: i32,
    bits_per_sample: u8,
}

impl Constant {
    pub(crate) const fn new(dc_offset: i32, bits_per_sample: u8) -> Self {
        Self {
            dc_offset,
            bits_per_sample,
        }
    }

    /// Returns offset value.
    pub fn dc_offset(&self) -> i32 {
        self.dc_offset
    }

    /// Returns bits-per-sample.
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }
}

impl BitRepr for Constant {
    #[inline]
    fn count_bits(&self) -> usize {
        8 + self.bits_per_sample as usize
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write(0u8).map_err(OutputError::<S>::from_sink)?;
        dest.write_twoc(self.dc_offset, self.bits_per_sample as usize)
            .map_err(OutputError::<S>::from_sink)?;
        Ok(())
    }
}

/// [`SUBFRAME_VERBATIM`](https://xiph.org/flac/format.html#subframe_verbatim) component.
#[derive(Clone, Debug)]
pub struct Verbatim {
    data: Vec<i32>,
    bits_per_sample: u8,
}

impl Verbatim {
    pub(crate) fn from_samples(samples: &[i32], bits_per_sample: u8) -> Self {
        Self {
            data: Vec::from(samples),
            bits_per_sample,
        }
    }

    #[inline]
    pub(crate) const fn count_bits_from_metadata(
        block_size: usize,
        bits_per_sample: usize,
    ) -> usize {
        8 + block_size * bits_per_sample
    }

    /// Returns a slice for the verbatim samples.
    pub fn samples(&self) -> &[i32] {
        &self.data
    }

    /// Returns bits-per-sample.
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }
}

impl BitRepr for Verbatim {
    #[inline]
    fn count_bits(&self) -> usize {
        Self::count_bits_from_metadata(self.data.len(), self.bits_per_sample as usize)
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        dest.write(0x02u8).map_err(OutputError::<S>::from_sink)?;
        for i in 0..self.data.len() {
            dest.write_twoc(self.data[i], self.bits_per_sample as usize)
                .map_err(OutputError::<S>::from_sink)?;
        }
        Ok(())
    }
}

/// [`SUBFRAME_FIXED`](https://xiph.org/flac/format.html#subframe_fixed) component.
#[derive(Clone, Debug)]
pub struct FixedLpc {
    warm_up: heapless::Vec<i32, 4>,
    residual: Residual,
    bits_per_sample: u8,
}

impl FixedLpc {
    /// Creates `FixedLpc`.
    ///
    /// # Panics
    ///
    /// Panics when `warm_up.len()`, i.e. the order of LPC, is larger than the
    /// maximum fixed-LPC order (4).
    pub(crate) fn new(warm_up: &[i32], residual: Residual, bits_per_sample: usize) -> Self {
        let warm_up = heapless::Vec::from_slice(warm_up)
            .expect("Exceeded maximum order for FixedLPC component.");

        Self {
            warm_up,
            residual,
            bits_per_sample: bits_per_sample as u8,
        }
    }

    /// Returns the order of LPC (of fixed LPC).
    pub fn order(&self) -> usize {
        self.warm_up.len()
    }

    /// Returns warm-up samples as a slice.
    pub fn warm_up(&self) -> &[i32] {
        &self.warm_up
    }

    /// Returns a reference to the internal `Residual` component.
    pub fn residual(&self) -> &Residual {
        &self.residual
    }

    /// Returns bits-per-sample.
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }
}

impl BitRepr for FixedLpc {
    #[inline]
    fn count_bits(&self) -> usize {
        8 + self.bits_per_sample as usize * self.order() + self.residual.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        let head_byte = 0x10u8 | (self.order() << 1) as u8;
        dest.write(head_byte).map_err(OutputError::<S>::from_sink)?;
        for v in &self.warm_up {
            dest.write_twoc(*v, self.bits_per_sample as usize)
                .map_err(OutputError::<S>::from_sink)?;
        }
        self.residual.write(dest)
    }
}

/// [`SUBFRAME_LPC`](https://xiph.org/flac/format.html#subframe_lpc) component.
#[derive(Clone, Debug)]
pub struct Lpc {
    parameters: lpc::QuantizedParameters,
    warm_up: heapless::Vec<i32, MAX_LPC_ORDER>,
    residual: Residual,
    bits_per_sample: u8,
}

impl Lpc {
    /// Constructs `Lpc`.
    ///
    /// # Panics
    ///
    /// Panics if the length of `warm_up` is not equal to `parameters.order()`.
    pub(crate) fn new(
        warm_up: &[i32],
        parameters: lpc::QuantizedParameters,
        residual: Residual,
        bits_per_sample: usize,
    ) -> Self {
        assert_eq!(warm_up.len(), parameters.order());
        let warm_up = heapless::Vec::from_slice(warm_up).expect("LPC order exceeded the maximum");
        Self {
            warm_up,
            parameters,
            residual,
            bits_per_sample: bits_per_sample as u8,
        }
    }

    /// Returns the order of LPC (of fixed LPC).
    pub const fn order(&self) -> usize {
        self.parameters.order()
    }

    /// Returns warm-up samples as a slice.
    pub fn warm_up(&self) -> &[i32] {
        &self.warm_up
    }

    /// Returns a reference to parameter struct.
    pub fn parameters(&self) -> &lpc::QuantizedParameters {
        &self.parameters
    }

    /// Returns a reference to the internal `Residual` component.
    pub fn residual(&self) -> &Residual {
        &self.residual
    }

    /// Returns bits-per-sample.
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }
}

impl BitRepr for Lpc {
    #[inline]
    fn count_bits(&self) -> usize {
        let warm_up_bits = self.bits_per_sample as usize * self.order();
        8 + warm_up_bits
            + 4
            + 5
            + self.parameters.precision() * self.order()
            + self.residual.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        let head_byte = 0x40 | (((self.order() - 1) as u8) << 1);
        dest.write(head_byte).map_err(OutputError::<S>::from_sink)?;

        for i in 0..self.order() {
            dest.write_twoc(self.warm_up[i], self.bits_per_sample as usize)
                .map_err(OutputError::<S>::from_sink)?;
        }

        assert!((self.parameters.precision() as u8) < 16u8);
        dest.write_lsbs((self.parameters.precision() - 1) as u64, 4)
            .map_err(OutputError::<S>::from_sink)?;

        // FLAC reference decoder doesn't support this.
        assert!(self.parameters.shift() >= 0);
        dest.write_twoc(self.parameters.shift(), 5)
            .map_err(OutputError::<S>::from_sink)?;

        for ref_coef in &self.parameters.coefs() {
            debug_assert!(*ref_coef < (1 << (self.parameters.precision() - 1)));
            debug_assert!(*ref_coef >= -(1 << (self.parameters.precision() - 1)));
            dest.write_twoc(*ref_coef, self.parameters.precision())
                .map_err(OutputError::<S>::from_sink)?;
        }

        self.residual.write(dest)
    }
}

/// [`RESIDUAL`](https://xiph.org/flac/format.html#residual) component.
#[derive(Clone, Debug)]
pub struct Residual {
    // TODO: Currently only supports 4-bit parameters
    partition_order: u8,
    block_size: usize,
    warmup_length: usize,
    // TODO: Escaped partition rice_params=0b(1)1111 is not supported.
    rice_params: Vec<u8>,

    // Here, raw-value is expected to have the sign bits encoded as its LSB.
    quotients: Vec<u32>,  // This should have left-padded for warm up samples
    remainders: Vec<u32>, // This should have left-padded for warm up samples
}

impl Residual {
    #[cfg(test)]
    pub(crate) fn new(
        partition_order: usize,
        block_size: usize,
        warmup_length: usize,
        rice_params: &[u8],
        quotients: &[u32],
        remainders: &[u32],
    ) -> Self {
        Self::from_parts(
            partition_order as u8,
            block_size,
            warmup_length,
            rice_params.to_owned(),
            quotients.to_owned(),
            remainders.to_owned(),
        )
    }

    /// Constructs `Residual` with consuming parts.
    pub(crate) fn from_parts(
        partition_order: u8,
        block_size: usize,
        warmup_length: usize,
        rice_params: Vec<u8>,
        quotients: Vec<u32>,
        remainders: Vec<u32>,
    ) -> Self {
        Self {
            partition_order,
            block_size,
            warmup_length,
            rice_params,
            quotients,
            remainders,
        }
    }

    /// Returns the partition order for the PRC.
    pub fn partition_order(&self) -> usize {
        self.partition_order as usize
    }

    /// Returns the rice parameter for the `p`-th partition
    pub fn rice_param(&self, p: usize) -> usize {
        self.rice_params[p] as usize
    }

    /// Returns the residual value for the `t`-th sample.
    pub fn residual(&self, t: usize) -> i32 {
        let nparts = 1usize << self.partition_order as usize;
        let part_id = t * nparts / self.block_size;
        let quotient = self.quotients[t];
        let shift = u32::from(self.rice_params[part_id]);
        let remainder = self.remainders[t];
        let v = (quotient << shift) + remainder;
        rice::decode_signbit(v)
    }
}

impl BitRepr for Residual {
    #[inline]
    fn count_bits(&self) -> usize {
        let nparts = 1usize << self.partition_order as usize;

        // using SIMD for `sum` here didn't help much.
        let quotient_bits: usize = self.quotients.iter().map(|x| *x as usize).sum::<usize>()
            + self.block_size
            - self.warmup_length;

        let mut remainder_bits: usize = 0;
        let part_len = self.block_size / nparts;
        for p in 0..nparts {
            remainder_bits += self.rice_params[p] as usize * part_len;
        }
        remainder_bits -= self.warmup_length * self.rice_params[0] as usize;
        2 + 4 + nparts * 4 + quotient_bits + remainder_bits
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        // The number of partitions with 00 (indicating 4-bit mode) prepended.
        dest.write_lsbs(self.partition_order, 6)
            .map_err(OutputError::<S>::from_sink)?;
        let nparts = 1usize << self.partition_order as usize;

        // NOTE:
        // It might be an option to store `self.rice_params` first to an
        // expanded buffer that has the same length as `self.quotients` and
        // `self.remainders`, and do serialization with an unrolled loop for
        // encouraging auto-vectorization.
        for p in 0..nparts {
            dest.write_lsbs(self.rice_params[p], 4)
                .map_err(OutputError::<S>::from_sink)?;
            let start = std::cmp::max(
                self.warmup_length,
                (p * self.block_size) >> self.partition_order,
            );
            let end = ((p + 1) * self.block_size) >> self.partition_order;

            let startbit = 1 << self.rice_params[p];
            for t in start..end {
                let q = self.quotients[t] as usize;
                dest.write_zeros(q).map_err(OutputError::<S>::from_sink)?;
                let r_plus_startbit = self.remainders[t] | startbit;
                dest.write_lsbs(r_plus_startbit, self.rice_params[p] as usize + 1)
                    .map_err(OutputError::<S>::from_sink)?;
            }
        }
        Ok(())
    }
}

mod seal_bit_repr {
    use super::*;

    pub trait Sealed {}
    impl Sealed for Stream {}
    impl Sealed for MetadataBlock {}
    impl Sealed for MetadataBlockData {}
    impl Sealed for StreamInfo {}
    impl Sealed for Frame {}
    impl Sealed for FrameHeader {}
    impl Sealed for ChannelAssignment {}
    impl Sealed for SubFrame {}
    impl Sealed for Constant {}
    impl Sealed for FixedLpc {}
    impl Sealed for Verbatim {}
    impl Sealed for Lpc {}
    impl Sealed for Residual {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

    use bitvec::bits;
    use bitvec::prelude::BitVec;
    use bitvec::prelude::Lsb0;

    use rand::distributions::Distribution;
    use rand::distributions::Uniform;

    fn make_frame(stream_info: &StreamInfo, samples: &[i32], offset: usize) -> Frame {
        let channels = stream_info.channels as usize;
        let block_size = samples.len() / channels;
        let bits_per_sample: u8 = stream_info.bits_per_sample;
        let ch_info = ChannelAssignment::Independent(channels as u8);
        let mut frame = Frame::new(ch_info, offset, block_size);
        for ch in 0..channels {
            frame.add_subframe(
                Verbatim::from_samples(
                    &samples[block_size * ch..block_size * (ch + 1)],
                    bits_per_sample,
                )
                .into(),
            );
        }
        frame
    }

    #[test]
    fn write_empty_stream() -> Result<(), OutputError<BitVec<u8>>> {
        let stream = Stream::new(44100, 2, 16);
        let mut bv: BitVec<u8> = BitVec::new();
        stream.write(&mut bv)?;
        assert_eq!(
            bv.len(),
            32 // fLaC
      + 1 + 7 + 24 // METADATA_BLOCK_HEADER
      + 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128 // METADATA_BLOCK_STREAMINFO
        );
        assert_eq!(stream.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn write_stream_info() -> Result<(), OutputError<BitVec<u8>>> {
        let stream_info = StreamInfo::new(44100, 2, 16);
        let mut bv: BitVec<u8> = BitVec::new();
        stream_info.write(&mut bv)?;
        assert_eq!(bv.len(), 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128);
        assert_eq!(stream_info.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn write_frame_header() -> Result<(), OutputError<BitVec<usize>>> {
        let header = FrameHeader::new(2304, ChannelAssignment::Independent(2), 192);
        let mut bv: BitVec<usize> = BitVec::new();
        header.write(&mut bv)?;

        // test with canonical frame
        let header = FrameHeader::new(192, ChannelAssignment::Independent(2), 0);
        let mut bv: BitVec<usize> = BitVec::new();
        header.write(&mut bv)?;

        assert_eq!(
            bv,
            bits![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // sync
                0, 1, // reserved/ blocking strategy (const in this impl)
                0, 0, 0, 1, 0, 0, 0, 0, // block size/ sample_rate (0=header)
                0, 0, 0, 1, 0, 0, 0, 0, // channel/ bps (0=header)/ reserved
                0, 0, 0, 0, 0, 0, 0, 0, // sample number
                0, 1, 1, 0, 1, 0, 0, 1, // crc-8
            ]
        );

        assert_eq!(header.count_bits(), bv.len());

        Ok(())
    }

    #[test]
    fn write_verbatim_frame() -> Result<(), OutputError<BitVec>> {
        let nchannels: usize = 3;
        let nsamples: usize = 17;
        let bits_per_sample: usize = 16;
        let stream_info = StreamInfo::new(16000, nchannels, bits_per_sample);
        let framebuf = vec![-1i32; nsamples * nchannels];
        let frame = make_frame(&stream_info, &framebuf, 0);
        let mut bv: BitVec<usize> = BitVec::new();
        frame.write(&mut bv)?;

        assert_eq!(frame.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn utf8_encoding() -> Result<(), RangeError> {
        let v = 0x56;
        let bs = encode_to_utf8like(v)?;
        assert_eq!(bs, &[0x56]);

        let v = 0x1024;
        let bs = encode_to_utf8like(v)?;
        assert_eq!(bs, &[0xE1, 0x80, 0xA4]);

        let v = 0xF_FFFF_FFFFu64; // 36 bits of ones
        let bs = encode_to_utf8like(v)?;
        assert_eq!(bs, &[0xFE, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]);

        let v = 0x10_0000_0000u64; //  out of domain
        encode_to_utf8like(v).expect_err("Should be out of domain");

        Ok(())
    }

    #[test]
    fn block_size_encoding() {
        let (head, _foot, footsize) = block_size_spec(192);
        assert_eq!(head, 0x01);
        assert_eq!(footsize, 0);

        let (head, _foot, footsize) = block_size_spec(2048);
        assert_eq!(head, 0x0B);
        assert_eq!(footsize, 0);

        let (head, _foot, footsize) = block_size_spec(1152);
        assert_eq!(head, 0x03);
        assert_eq!(footsize, 0);

        let (head, foot, footsize) = block_size_spec(193);
        assert_eq!(head, 0x06);
        assert_eq!(footsize, 8);
        assert_eq!(foot, 0xC0);

        let (head, foot, footsize) = block_size_spec(1151);
        assert_eq!(head, 0x07);
        assert_eq!(footsize, 16);
        assert_eq!(foot, 0x047E);
    }

    #[test]
    fn channel_assignment_encoding() -> Result<(), OutputError<BitVec<usize>>> {
        let ch = ChannelAssignment::Independent(8);
        let mut bv: BitVec<usize> = BitVec::new();
        ch.write(&mut bv)?;
        assert_eq!(bv, bits![0, 1, 1, 1]);
        let ch = ChannelAssignment::RightSide;
        let mut bv: BitVec<usize> = BitVec::new();
        ch.write(&mut bv)?;
        assert_eq!(bv, bits![1, 0, 0, 1]);
        assert_eq!(ch.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn stream_info_update() {
        let mut stream_info = StreamInfo::new(44100, 2, 16);
        let framebuf = test_helper::constant_plus_noise(256 * 2, 123, 21);
        let frame1 = make_frame(&stream_info, &framebuf, 0);
        stream_info.update_frame_info(&frame1);
        let framebuf = test_helper::constant_plus_noise(192 * 2, 234, 32);
        let frame2 = make_frame(&stream_info, &framebuf, 256);
        stream_info.update_frame_info(&frame2);

        assert_eq!(stream_info.min_block_size, 192);
        assert_eq!(stream_info.max_block_size, 256);

        // header_size = 5 + sample_number + block_size + sample_rate
        // sample_rate = 0 (in this implementation)
        // block_size = 0 (since preset sizes (192 and 256) are used)
        // sample_number = 1 for the first frame, 2 for the second frame
        // footer_size = 2
        // verbatim_subframe_size = 2 * block_size * 2 + subframe_header_size
        // subframe_header_size = 1
        // so overall:
        // first_frame_size = 5 + 1 + 0 + 0 + 2 + 1 + 2 * 256 * 2 + 1 = 1034
        // second_frame_size = 5 + 2 + 0 + 0 + 2 + 1 + 2 * 192 * 2 + 1 = 779
        assert_eq!(stream_info.min_frame_size, 779);
        assert_eq!(stream_info.max_frame_size, 1034);
    }

    #[test]
    #[allow(clippy::cast_lossless)]
    fn bit_count_residual() -> Result<(), OutputError<BitVec<usize>>> {
        let mut rng = rand::thread_rng();
        let block_size = 4 * Uniform::from(4..=1024).sample(&mut rng);
        let partition_order: usize = 2;
        let nparts = 2usize.pow(partition_order as u32);
        let part_len = block_size / nparts;
        let params = vec![7, 8, 6, 7];
        let mut quotients: Vec<u32> = vec![];
        let mut remainders: Vec<u32> = vec![];

        for t in 0..block_size {
            let part_id = t / part_len;
            let p = params[part_id];

            quotients.push((255 / p) as u32);
            remainders.push((255 % p) as u32);
        }
        let residual = Residual::new(
            partition_order,
            block_size,
            0,
            &params,
            &quotients,
            &remainders,
        );

        let mut bv: BitVec<usize> = BitVec::new();
        residual.write(&mut bv)?;

        assert_eq!(residual.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn frame_bitstream_precomputataion() -> Result<(), OutputError<BitVec<usize>>> {
        let stream_info = StreamInfo::new(44100, 2, 16);
        let samples = test_helper::sinusoid_plus_noise(256 * 2, 128, 200.0, 100);
        let mut frame = make_frame(&stream_info, &samples, 0);
        let mut bv_ref: BitVec<usize> = BitVec::new();
        let frame_cloned = frame.clone();
        frame_cloned.write(&mut bv_ref)?;
        assert!(bv_ref.len() % 8 == 0); // frame must be byte-aligned.

        frame.precompute_bitstream();
        assert!(frame.is_bitstream_precomputed());
        assert!(!frame_cloned.is_bitstream_precomputed());

        let mut bv: BitVec<usize> = BitVec::new();
        frame.write(&mut bv)?;
        assert_eq!(bv, bv_ref);
        Ok(())
    }
}
