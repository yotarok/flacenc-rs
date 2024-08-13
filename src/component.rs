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

use std::cmp::max;
use std::cmp::min;

#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use super::arrayutils::find_max;
use super::arrayutils::wrapping_sum;
use super::bitsink::BitSink;
use super::bitsink::ByteSink;
use super::bitsink::MemSink;
#[cfg(any(test, feature = "__export_decode"))]
use super::constant::fixed::MAX_LPC_ORDER as MAX_FIXED_LPC_ORDER;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::qlpc::MAX_PRECISION as MAX_LPC_PRECISION;
use super::constant::qlpc::MAX_SHIFT as MAX_LPC_SHIFT;
use super::constant::qlpc::MIN_SHIFT as MIN_LPC_SHIFT;
use super::constant::MAX_BITS_PER_SAMPLE;
use super::constant::MAX_BLOCK_SIZE;
use super::constant::MAX_CHANNELS;
use super::constant::MIN_BITS_PER_SAMPLE;
use super::constant::MIN_BLOCK_SIZE;
use super::error::verify_range;
use super::error::verify_true;
use super::error::OutputError;
use super::error::RangeError;
use super::error::Verify;
use super::error::VerifyError;
use super::repeat::try_repeat;
use super::rice;

import_simd!(as simd);

const CRC_8_FLAC: crc::Algorithm<u8> = crc::CRC_8_SMBUS;
const CRC_16_FLAC: crc::Algorithm<u16> = crc::CRC_16_UMTS;

/// Proxy type for serializing/ deserializing `simd::Simd`.
#[cfg(feature = "serde")]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(remote = "simd::Simd"))]
struct SimdDef<T, const N: usize>(#[serde(getter = "simd::Simd::as_array")] [T; N])
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
    [T; N]: for<'a> Deserialize<'a> + Serialize;

#[cfg(feature = "serde")]
impl<T, const N: usize> From<SimdDef<T, N>> for simd::Simd<T, N>
where
    T: simd::SimdElement,
    simd::LaneCount<N>: simd::SupportedLaneCount,
    [T; N]: for<'a> Deserialize<'a> + Serialize,
{
    fn from(def: SimdDef<T, N>) -> Self {
        Self::from_array(def.0)
    }
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
}

/// Traits for FLAC components containing signals (represented in `[i32]`).
///
/// "Signal" here has slightly different meaning depending on the component
/// that implements this trait. For example, for `Residual`, signal is a
/// prediction error signal. For `SubFrame`, signal means a single-channel
/// sequence of samples whereas for `Frame`, signal is an interleaved multi-
/// channel samples.
#[cfg(any(test, feature = "__export_decode"))]
pub trait Decode: seal_bit_repr::Sealed {
    /// Decodes and copies signal to the specified buffer.
    ///
    /// # Panics
    ///
    /// Implementations of this method should panic when `dest` doesn't have
    /// a sufficient length.
    fn copy_signal(&self, dest: &mut [i32]);

    /// Returns number of elements in the decoded signal.
    fn signal_len(&self) -> usize;

    /// Returns signal represented as `Vec<i32>`.
    fn decode(&self) -> Vec<i32> {
        let mut ret = vec![0i32; self.signal_len()];
        self.copy_signal(&mut ret);
        ret
    }
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

// Some (internal) utility macros for value verification.
macro_rules! verify_block_size {
    ($varname:literal, $size:expr) => {
        verify_range!($varname, $size, MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE)
    };
}
macro_rules! verify_bps {
    ($varname:literal, $bps:expr) => {
        verify_range!(
            $varname,
            $bps,
            MIN_BITS_PER_SAMPLE..=(MAX_BITS_PER_SAMPLE + 1)
        )
        .and_then(|()| {
            verify_true!(
                $varname,
                ($bps as usize) % 4 == 0 || ($bps as usize) % 4 == 1,
                "must be a multiple of 4 (or 4n + 1 for side-channel)"
            )
        })
    };
}
macro_rules! verify_sample_range {
    ($varname:literal, $sample:expr, $bps:expr) => {{
        let min_sample = -((1usize << ($bps as usize - 1)) as i32);
        let max_sample = (1usize << ($bps as usize - 1)) as i32 - 1;
        verify_range!($varname, $sample, min_sample..=max_sample)
    }};
}

/// [`STREAM`](https://xiph.org/flac/format.html#stream) component.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Stream {
    stream_info: MetadataBlock,
    metadata: Vec<MetadataBlock>,
    frames: Vec<Frame>,
}

impl Stream {
    /// Constructs `Stream` with the given meta information.
    ///
    /// # Errors
    ///
    /// Returns error if an input argument is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let stream = Stream::new(16000, 1, 16).unwrap();
    /// assert_eq!(stream.stream_info().channels(), 1);
    /// ```
    pub fn new(
        sample_rate: usize,
        channels: usize,
        bits_per_sample: usize,
    ) -> Result<Self, VerifyError> {
        Ok(Self::with_stream_info(StreamInfo::new(
            sample_rate,
            channels,
            bits_per_sample,
        )?))
    }

    /// Constructs `Stream` with the given `StreamInfo`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let stream_info = StreamInfo::new(16000, 1, 16).unwrap();
    /// let stream = Stream::with_stream_info(stream_info);
    /// assert_eq!(stream.stream_info().sample_rate(), 16000);
    /// ```
    pub fn with_stream_info(stream_info: StreamInfo) -> Self {
        Self {
            stream_info: MetadataBlock::from_stream_info(stream_info, true),
            metadata: vec![],
            frames: vec![],
        }
    }

    /// Returns a reference to [`StreamInfo`] associated with `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is corrupted by manually modifying fields.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let stream = Stream::new(16000, 1, 24).unwrap();
    /// assert_eq!(stream.stream_info().bits_per_sample(), 24);
    /// ```
    pub fn stream_info(&self) -> &StreamInfo {
        if let MetadataBlockData::StreamInfo(ref info) = self.stream_info.data {
            info
        } else {
            panic!("Stream is not properly initialized.")
        }
    }

    /// Returns a mutable reference to [`StreamInfo`] associated with `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is corrupted by manually modifying fields.
    pub(crate) fn stream_info_mut(&mut self) -> &mut StreamInfo {
        if let MetadataBlockData::StreamInfo(ref mut info) = self.stream_info.data {
            info
        } else {
            panic!("Stream is not properly initialized.")
        }
    }

    /// Appends [`Frame`] to this `Stream` and updates [`StreamInfo`].
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// let mut stream = Stream::new(16000, 1, 24).unwrap();
    /// stream.add_frame(frame);
    /// assert_eq!(stream.frame_count(), 1);
    /// ```
    pub fn add_frame(&mut self, frame: Frame) {
        self.stream_info_mut().update_frame_info(&frame);
        self.frames.push(frame);
    }

    /// Add [`MetadataBlockData`] to this `Stream`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::bitsink::*;
    /// let mut stream = Stream::new(16000, 1, 24).unwrap();
    /// stream.add_metadata_block(
    ///     MetadataBlockData::new_unknown(2, &[0xFF]).unwrap()
    /// );
    /// let mut sink = ByteSink::new();
    /// stream.write(&mut sink);
    /// let bytes = sink.as_slice();
    /// let first_header_byte = bytes[4];
    /// assert_eq!(first_header_byte, 0x00); // lastflag + type (STREAMINFO=0)
    /// let second_header_byte = bytes[bytes.len() - 5];
    /// assert_eq!(second_header_byte, 0x82); // lastflag + type (=2)
    /// ```
    pub fn add_metadata_block(&mut self, metadata: MetadataBlockData) {
        let metadata = MetadataBlock::from_parts(true, metadata);
        if let Some(x) = self.metadata.last_mut() {
            x.is_last = false;
        } else {
            self.stream_info.is_last = false;
        }
        self.metadata.push(metadata);
    }

    /// Returns [`Frame`] for the given frame number.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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

    /// Returns the number of [`Frame`]s in the stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// assert_eq!(stream.frame_count(), 200);
    /// ```
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns [`Frame`]s as a slice.
    #[allow(dead_code)]
    pub(crate) fn frames(&self) -> &[Frame] {
        &self.frames
    }

    fn verify_frames_in_variable_block_size_mode(&self) -> Result<(), VerifyError> {
        let mut current = 0u64;

        for (i, frame) in self.frames.iter().enumerate() {
            verify_true!(
                "variable_block_size",
                frame.header.variable_block_size,
                "must be same for all frames"
            )
            .and_then(|()| {
                verify_true!(
                    "start_sample_number",
                    frame.header.start_sample_number == current,
                    "must be the sum of the block sizes of the preceding frames"
                )
            })
            .map_err(|e| e.within("header").within(&format!("frames[{i}]")))?;
            frame
                .verify()
                .map_err(|e| e.within(&format!("frames[{i}]")))?;
            current = current.wrapping_add(frame.header.block_size.into());
        }
        Ok(())
    }

    fn verify_frames_in_fixed_block_size_mode(&self) -> Result<(), VerifyError> {
        let mut current = 0u32;

        for (i, frame) in self.frames.iter().enumerate() {
            verify_true!(
                "variable_block_size",
                !frame.header.variable_block_size,
                "must be same for all frames"
            )
            .and_then(|()| {
                verify_true!(
                    "frame_number",
                    frame.header.frame_number == current,
                    "must be the count of the preceding frames"
                )
            })
            .map_err(|e| e.within("header").within(&format!("frames[{i}]")))?;
            frame
                .verify()
                .map_err(|e| e.within(&format!("frames[{i}]")))?;
            current = current.wrapping_add(1);
        }
        Ok(())
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

impl Verify for Stream {
    fn verify(&self) -> Result<(), VerifyError> {
        self.stream_info
            .verify()
            .map_err(|e| e.within("stream_info"))?;
        for (i, md) in self.metadata.iter().enumerate() {
            md.verify()
                .map_err(|e| e.within(&format!("metadata[{i}]")))?;
            let is_last = i + 1 == self.metadata.len();

            verify_true!(
                "is_last",
                is_last || !md.is_last,
                "should be unset for non-last metdata blocks"
            )
            .and_then(|()| {
                verify_true!(
                    "is_last",
                    !is_last || md.is_last,
                    "should be set for the last metdata block"
                )
            })
            .map_err(|e| e.within(&format!("metadata[{i}]")))?;
        }

        if self.frames.is_empty() {
            Ok(())
        } else if self.frames[0].header.variable_block_size {
            self.verify_frames_in_variable_block_size_mode()
        } else {
            self.verify_frames_in_fixed_block_size_mode()
        }
    }
}

/// [`METADATA_BLOCK`](https://xiph.org/flac/format.html#metadata_block) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetadataBlock {
    is_last: bool,
    data: MetadataBlockData,
}

impl MetadataBlock {
    pub(crate) const fn from_parts(is_last: bool, data: MetadataBlockData) -> Self {
        Self { is_last, data }
    }

    const fn from_stream_info(info: StreamInfo, is_last: bool) -> Self {
        Self {
            is_last,
            data: MetadataBlockData::StreamInfo(info),
        }
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

impl Verify for MetadataBlock {
    fn verify(&self) -> Result<(), VerifyError> {
        self.data.verify()
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", content = "data"))]
/// Enum that covers variants of `METADATA_BLOCK`.
///
/// Currently only [`StreamInfo`] is covered in this enum.
#[non_exhaustive]
pub enum MetadataBlockData {
    /// Variant that contains [`StreamInfo`].
    ///
    /// This variant can be obtained via [`From<StreamInfo>::from`].
    StreamInfo(StreamInfo),
    /// Variant that contains unknown data.
    ///
    /// This variant can be obtained via [`MetadataBlockData::new_unknown`].
    Unknown {
        /// 7-bit metadata type tag.
        typetag: u8,
        /// Metadata content represented in `Vec<u8>`.
        data: Vec<u8>,
    },
}

impl MetadataBlockData {
    /// Constructs new `MetadataBlockData::Unknown` from the content (in `[u8]`).
    ///
    /// # Errors
    ///
    /// Emits errors when `tag` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// use flacenc::bitsink::MemSink;
    /// let metadata: MetadataBlockData = MetadataBlockData::new_unknown(
    ///     0x12, &[0x34, 0x56]
    /// ).expect("failed to construct unknown metadata.");
    /// let mut sink = MemSink::<u8>::new();
    /// metadata.write(&mut sink).unwrap();
    ///
    /// // Note that typetag will be written only after it is wrapped in
    /// // `MetadataBlock`.
    /// assert_eq!(&[0x34, 0x56], sink.as_slice());
    /// ```
    pub fn new_unknown(tag: u8, data: &[u8]) -> Result<Self, VerifyError> {
        verify_range!("tag", tag, 0..=126)?;
        Ok(Self::Unknown {
            typetag: tag,
            data: data.to_owned(),
        })
    }

    pub(crate) fn typetag(&self) -> u8 {
        match self {
            Self::StreamInfo(_) => 0,
            Self::Unknown { typetag, .. } => *typetag,
        }
    }

    /// Obtain inner [`StreamInfo`] if `self` contains `StreamInfo`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let stream_info = StreamInfo::new(16000, 1, 8).unwrap();
    /// let metadata: MetadataBlockData = stream_info.clone().into();
    /// assert_eq!(metadata.as_stream_info(), Some(&stream_info));
    /// ```
    pub fn as_stream_info(&self) -> Option<&StreamInfo> {
        if let Self::StreamInfo(ref info) = self {
            Some(info)
        } else {
            None
        }
    }
}

impl From<StreamInfo> for MetadataBlockData {
    fn from(value: StreamInfo) -> Self {
        Self::StreamInfo(value)
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

impl Verify for MetadataBlockData {
    fn verify(&self) -> Result<(), VerifyError> {
        match self {
            Self::StreamInfo(info) => info.verify(),
            Self::Unknown { .. } => Ok(()),
        }
    }
}

/// [`METADATA_BLOCK_STREAM_INFO`](https://xiph.org/flac/format.html#metadata_block_streaminfo) component.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    /// -  `min_block_size`: [`u16::MAX`],
    /// -  `max_block_size`: `0`,
    /// -  `min_frame_size`: [`u32::MAX`],
    /// -  `max_frame_size`: `0`,
    /// -  `total_samples`: `0`,
    /// -  `md5_digest`: `[0u8; 16]` (indicating verification disabled.)
    ///
    /// # Errors
    ///
    /// Returns an error if an input argument is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
    /// assert_eq!(info.max_frame_size(), 0);
    /// ```
    pub fn new(
        sample_rate: usize,
        channels: usize,
        bits_per_sample: usize,
    ) -> Result<Self, VerifyError> {
        let ret = Self {
            min_block_size: u16::MAX,
            max_block_size: 0,
            min_frame_size: u32::MAX,
            max_frame_size: 0,
            sample_rate: sample_rate as u32,
            channels: channels as u8,
            bits_per_sample: bits_per_sample as u8,
            total_samples: 0,
            md5: [0; 16],
        };
        ret.verify()?;
        Ok(ret)
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    ///
    /// for n in 0..other_stream.frame_count() {
    ///     info.update_frame_info(other_stream.frame(n).unwrap());
    /// }
    /// assert_eq!(info.max_block_size(), 160);
    /// assert_eq!(info.min_block_size(), 160);
    ///
    /// // `Frame` doesn't hold the original signal length, so `total_samples`
    /// // becomes a multiple of block_size == 160.
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    ///
    /// for n in 0..other_stream.frame_count() {
    ///     info.update_frame_info(other_stream.frame(n).unwrap());
    /// }
    ///
    /// // `Frame` doesn't hold the original signal length, so `total_samples`
    /// // becomes a multiple of block_size == 160.
    /// assert_eq!(info.total_samples(), 31360);
    /// ```
    pub fn total_samples(&self) -> usize {
        self.total_samples as usize
    }

    /// Sets `total_samples` field.
    ///
    /// `total_samples` is updated during the encoding. However, since [`Frame`]
    /// only knows its frame size, the effective number of samples is not
    /// visible after paddings.  Similar to [`set_md5_digest`], this
    /// field should be finalized by propagating information from [`Context`].
    ///
    /// [`set_md5_digest`]: StreamInfo::set_md5_digest
    /// [`Context`]: crate::source::Context
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::source::{Context, Fill};
    /// # use flacenc::*;
    /// let mut ctx = Context::new(16, 2, 123);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    /// ctx.fill_interleaved(&[0x0000_0FFFi32; 246]);
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let other_stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
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
    /// MD5 computation is not performed in in [`update_frame_info`], and is
    /// expected to be done externally (by [`Context`]). This function is called
    /// to set MD5 bytes after we read all input samples.
    ///
    /// [`Context`]: crate::source::Context
    /// [`update_frame_info`]: StreamInfo::update_frame_info
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::*;
    /// # use flacenc::source::{Context, Fill};
    /// let mut ctx = Context::new(16, 2, 128);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    /// assert_eq!(info.md5_digest(), &[0x00u8; 16]);
    /// ctx.fill_interleaved(&[0x0000_0FFFi32; 256]);
    /// info.set_md5_digest(&ctx.md5_digest());
    /// assert_ne!(info.md5_digest(), &[0x00u8; 16]);
    /// ```
    pub fn set_md5_digest(&mut self, digest: &[u8; 16]) {
        self.md5.copy_from_slice(digest);
    }

    /// Resets `min_block_size` and `max_block_size` fields.
    ///
    /// # Errors
    ///
    /// Returns error when `min_value` or `max_value` is not a valid block size, or when
    /// `min_value > max_value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::*;
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    /// info.set_block_sizes(128, 1024).unwrap();
    /// assert_eq!(info.min_block_size(), 128);
    /// assert_eq!(info.max_block_size(), 1024);
    /// ```
    pub fn set_block_sizes(
        &mut self,
        min_value: usize,
        max_value: usize,
    ) -> Result<(), VerifyError> {
        self.min_block_size = min_value
            .try_into()
            .map_err(|_| VerifyError::new("min_block_size", "must be a valid block size."))?;
        self.max_block_size = max_value
            .try_into()
            .map_err(|_| VerifyError::new("max_block_size", "must be a valid block size."))?;
        verify_block_size!("min_block_size", self.min_block_size as usize)?;
        verify_block_size!("max_block_size", self.max_block_size as usize)?;
        verify_true!(
            "min_block_size",
            self.min_block_size <= self.max_block_size,
            "must be smaller than `max_block_size`"
        )?;
        Ok(())
    }

    /// Resets `min_frame_size` and `max_frame_size` fields.
    ///
    /// # Errors
    ///
    /// Returns error when `min_value` or `max_value` is not 32-bit representable, or when
    /// `min_value > max_value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::*;
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    /// info.set_frame_sizes(128, 2048).unwrap();
    /// assert_eq!(info.min_frame_size(), 128);
    /// assert_eq!(info.max_frame_size(), 2048);
    /// ```
    pub fn set_frame_sizes(
        &mut self,
        min_value: usize,
        max_value: usize,
    ) -> Result<(), VerifyError> {
        self.min_frame_size = min_value
            .try_into()
            .map_err(|_| VerifyError::new("min_frame_size", "must be a 32-bit integer."))?;
        self.max_frame_size = max_value
            .try_into()
            .map_err(|_| VerifyError::new("min_frame_size", "must be a 32-bit integer."))?;
        verify_true!(
            "min_frame_size",
            self.min_frame_size <= self.max_frame_size,
            "must be smaller than `max_frame_size`"
        )?;
        Ok(())
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

impl Verify for StreamInfo {
    fn verify(&self) -> Result<(), VerifyError> {
        if self.total_samples != 0 {
            verify_true!(
                "min_block_size",
                self.min_block_size <= self.max_block_size,
                "must be smaller than `max_block_size`"
            )?;
            verify_block_size!("min_block_size", self.min_block_size as usize)?;
            verify_block_size!("max_block_size", self.max_block_size as usize)?;
            verify_true!(
                "min_frame_size",
                self.min_frame_size <= self.max_frame_size,
                "must be smaller than `max_frame_size`"
            )?;
        }
        verify_range!("sample_rate", self.sample_rate as usize, ..=96_000)?;
        verify_range!("channels", self.channels as usize, 1..=8)?;
        verify_bps!("bits_per_sample", self.bits_per_sample as usize)
    }
}

/// [`FRAME`](https://xiph.org/flac/format.html#frame) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Frame {
    header: FrameHeader,
    // this making this `heapless` is inefficient in typical use cases.
    // as there're only few use cases that requires `MAX_CHANNELS`. It is
    // shown that with `mimalloc` the performance deficit by making it on
    // heap was negligible.
    subframes: Vec<SubFrame>,
    precomputed_bitstream: Option<Vec<u8>>,
}

impl Frame {
    /// Returns block size of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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
    pub(crate) fn new_empty(ch_info: ChannelAssignment, offset: usize, block_size: usize) -> Self {
        let header = FrameHeader::new(block_size, ch_info, offset);
        Self {
            header,
            subframes: Vec::with_capacity(MAX_CHANNELS),
            precomputed_bitstream: None,
        }
    }

    /// Constructs `Frame` from header and subframes.
    ///
    /// # Errors
    ///
    /// Emits error if the number of channel specified in `header` does not match
    /// to the length of `subframes`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let chs = ChannelAssignment::Independent(1);
    /// let header = FrameHeader::new_fixed_size(192, chs, SampleSizeSpec::B8, 0).unwrap();
    /// let subframe = Constant::new(192, -1, 8).unwrap();
    /// let frame = Frame::new(header, [subframe.into()].into_iter()).unwrap();
    /// ```
    pub fn new<I>(header: FrameHeader, subframes: I) -> Result<Self, VerifyError>
    where
        I: Iterator<Item = SubFrame>,
    {
        let subframes: Vec<SubFrame> = subframes.collect();
        verify_true!(
            "subframes.len()",
            header.channel_assignment().channels() == subframes.len(),
            "must match to the channel specification in the header"
        )?;
        Ok(Self::from_parts(header, subframes))
    }

    /// Constructs Frame from [`FrameHeader`] and [`SubFrame`]s.
    pub(crate) fn from_parts(header: FrameHeader, subframes: Vec<SubFrame>) -> Self {
        Self {
            header,
            subframes,
            precomputed_bitstream: None,
        }
    }

    /// Deconstructs frame and transfers ownership of the data structs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// let (header, subframes) = frame.into_parts();
    ///
    /// assert_eq!(subframes.len(), 2);
    /// ```
    pub fn into_parts(self) -> (FrameHeader, Vec<SubFrame>) {
        (self.header, self.subframes)
    }

    /// Adds subframe.
    ///
    /// # Panics
    ///
    /// Panics when the number of subframes added exceeded the `MAX_CHANNELS`.
    pub(crate) fn add_subframe(&mut self, subframe: SubFrame) {
        self.precomputed_bitstream = None;
        self.subframes.push(subframe);
        assert!(self.subframes.len() <= MAX_CHANNELS);
    }

    /// Returns a reference to [`FrameHeader`] of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    /// assert_eq!(frame.header().block_size(), 160);
    /// ```
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns a mutable reference to [`FrameHeader`] of this frame.
    pub(crate) fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    /// Returns [`SubFrame`] for the given channel.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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

    /// Returns the number of [`SubFrame`]s in this `Frame`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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
    /// assert_eq!(sink.as_slice(), sink_ref.as_slice());
    /// ```
    pub fn precompute_bitstream(&mut self) {
        if self.precomputed_bitstream.is_some() {
            return;
        }
        let mut dest = ByteSink::with_capacity(self.count_bits());
        if self.write(&mut dest).is_ok() {
            self.precomputed_bitstream = Some(dest.into_inner());
        }
    }

    /// Consumes `self` and returns the parts if `self` is a stereo frame.
    ///
    /// # Errors
    ///
    /// When `self.subframe_count() != 2`, this function returns the
    /// reconstructed self. On error, this allocates from the heap, and it is
    /// not efficient.
    ///
    /// # Panics
    ///
    /// Should not panic except for memory error.
    #[inline]
    pub fn into_stereo_channels(self) -> Result<(FrameHeader, SubFrame, SubFrame), Self> {
        if self.subframe_count() != 2 {
            return Err(self);
        }
        let (header, subframes) = self.into_parts();
        let mut iter = subframes.into_iter();
        let ch0 = iter.next().expect(panic_msg::DATA_INCONSISTENT);
        let ch1 = iter.next().expect(panic_msg::DATA_INCONSISTENT);
        Ok((header, ch0, ch1))
    }

    #[cfg(test)]
    const fn is_bitstream_precomputed(&self) -> bool {
        self.precomputed_bitstream.is_some()
    }
}

reusable!(FRAME_CRC_BUFFER: (MemSink<u64>, Vec<u8>) = (MemSink::new(), Vec::new()));
static FRAME_CRC: crc::Crc<u16, crc::Table<16>> =
    crc::Crc::<u16, crc::Table<16>>::new(&CRC_16_FLAC);

impl BitRepr for Frame {
    #[inline]
    fn count_bits(&self) -> usize {
        self.precomputed_bitstream.as_ref().map_or_else(
            || {
                let header = self.header.count_bits();
                let body: usize = self.subframes.iter().map(BitRepr::count_bits).sum();

                let aligned = ((header + body + 7) >> 3) << 3;
                let footer = 16;
                aligned + footer
            },
            |bytes| bytes.len() << 3,
        )
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        if let Some(ref bytes) = &self.precomputed_bitstream {
            dest.write_bytes_aligned(bytes)
                .map_err(OutputError::<S>::from_sink)?;
            Ok(())
        } else {
            reuse!(FRAME_CRC_BUFFER, |buf: &mut (MemSink<u64>, Vec<u8>)| {
                let frame_sink = &mut buf.0;
                let bytebuf = &mut buf.1;

                frame_sink.clear();
                frame_sink.reserve(self.count_bits());

                self.header
                    .write(frame_sink)
                    .map_err(OutputError::<S>::ignore_sink_error)?;
                for sub in &self.subframes {
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

impl Verify for Frame {
    fn verify(&self) -> Result<(), VerifyError> {
        for (ch, sf) in self.subframes.iter().enumerate() {
            sf.verify()
                .map_err(|e| e.within(&format!("subframe[{ch}]")))?;
        }
        if let Some(ref buf) = self.precomputed_bitstream {
            let mut dest = ByteSink::with_capacity(self.count_bits());
            self.write(&mut dest).map_err(|_| {
                VerifyError::new(
                    "self",
                    "erroroccured while computing verification reference.",
                )
            })?;
            let reference = dest.into_inner();
            verify_true!(
                "precomputed_bitstream.len",
                buf.len() == reference.len(),
                "must be identical with the recomputed bitstream"
            )?;
            for (t, (testbyte, refbyte)) in buf.iter().zip(reference.iter()).enumerate() {
                verify_true!(
                    "precomputed_bitstream[{t}]",
                    testbyte == refbyte,
                    "must be identical with the recomputed bitstream"
                )?;
            }
        }
        self.header.verify().map_err(|e| e.within("header"))
    }
}

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for Frame {
    fn signal_len(&self) -> usize {
        self.block_size() * self.subframe_count()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.signal_len());

        let mut channels = vec![];
        for sf in &self.subframes {
            channels.push(sf.decode());
        }

        match self.header.channel_assignment() {
            ChannelAssignment::Independent(_) => {}
            ChannelAssignment::LeftSide => {
                for t in 0..self.block_size() {
                    channels[1][t] = channels[0][t] - channels[1][t];
                }
            }
            ChannelAssignment::RightSide => {
                for t in 0..self.block_size() {
                    channels[0][t] += channels[1][t];
                }
            }
            ChannelAssignment::MidSide => {
                for t in 0..self.block_size() {
                    let s = channels[1][t];
                    let m = (channels[0][t] << 1) + (s & 0x01);
                    channels[0][t] = (m + s) >> 1;
                    channels[1][t] = (m - s) >> 1;
                }
            }
        };

        // interleave
        let channel_count = channels.len();
        for (ch, sig) in channels.iter().enumerate() {
            for (t, x) in sig.iter().enumerate() {
                dest[t * channel_count + ch] = *x;
            }
        }
    }
}

/// Enum for channel assignment in `FRAME_HEADER`.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", content = "data"))]
pub enum ChannelAssignment {
    /// Indicates that the frame contains multiple channels independently.
    ///
    /// The `u8` field indicates the number of channels. This is the only
    /// option if the number of channels is not two.
    Independent(u8),
    /// Indicates that the frame contains left and side channels.
    LeftSide,
    /// Indicates that the frame contains right and side channels.
    RightSide,
    /// Indicates that the frame contains mid and side channels.
    MidSide,
}

impl ChannelAssignment {
    /// Constructs `ChannelAssignment` from the tag.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// assert_eq!(
    ///     ChannelAssignment::from_tag(5),
    ///     Some(ChannelAssignment::Independent(6)),
    /// );
    /// assert_eq!(
    ///     ChannelAssignment::from_tag(10),
    ///     Some(ChannelAssignment::MidSide),
    /// );
    /// ```
    pub const fn from_tag(tag: u8) -> Option<Self> {
        if tag < 8 {
            Some(Self::Independent(tag + 1))
        } else if tag == 8 {
            Some(Self::LeftSide)
        } else if tag == 9 {
            Some(Self::RightSide)
        } else if tag == 10 {
            Some(Self::MidSide)
        } else {
            None
        }
    }

    /// Returns the number of extra bit required to store the channel samples.
    ///
    /// "Side" signal (as used in mid-side coding) requires an extra bit for
    /// storing large values such as `i32::MAX - i32::MIN`. This function maps
    /// `ChannelAssignment` and channel id `ch` to the number of extra bits
    /// required (0 or 1).
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let rs = ChannelAssignment::RightSide;
    /// // Ride-side coding stores the side signal in channel-0.
    /// assert_eq!(rs.bits_per_sample_offset(0), 1);
    /// assert_eq!(rs.bits_per_sample_offset(1), 0);
    /// ```
    pub const fn bits_per_sample_offset(&self, ch: usize) -> usize {
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

    pub(crate) fn channels(&self) -> usize {
        if let Self::Independent(n) = self {
            *n as usize
        } else {
            2
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

impl Verify for ChannelAssignment {
    fn verify(&self) -> Result<(), VerifyError> {
        match *self {
            Self::Independent(ch) => {
                verify_range!("Independent(ch)", ch as usize, 1..=MAX_CHANNELS)
            }
            Self::LeftSide | Self::RightSide | Self::MidSide => Ok(()),
        }
    }
}

/// Enum for supported sample sizes.
///
/// Refer [`FRAME_HEADER`](https://xiph.org/flac/format.html#frame_header)
/// specification for details.
///
/// TODO: Hide this from public API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum SampleSizeSpec {
    /// `Unspecified` can be used in `FrameHeader` to instruct decoders to get
    /// sample size information from `StreamInfo`.
    Unspecified = 0,
    /// 8 bits-per-second
    B8 = 1,
    /// 12 bits-per-second
    B12 = 2,
    /// `tag == 3` is reserved.
    Reserved = 3,
    /// 16 bits-per-second
    B16 = 4,
    /// 20 bits-per-second
    B20 = 5,
    /// 24 bits-per-second
    B24 = 6,
    /// 32 bits-per-second
    B32 = 7,
}

impl SampleSizeSpec {
    /// Constructs `SampleSizeSpec` from the tag (an integer in the bitstream).
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// assert_eq!(SampleSizeSpec::from_tag(4), Some(SampleSizeSpec::B16));
    /// assert_eq!(SampleSizeSpec::from_tag(8), None);
    /// ```
    pub const fn from_tag(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unspecified),
            1 => Some(Self::B8),
            2 => Some(Self::B12),
            3 => Some(Self::Reserved),
            4 => Some(Self::B16),
            5 => Some(Self::B20),
            6 => Some(Self::B24),
            7 => Some(Self::B32),
            _ => None,
        }
    }

    /// Returns the tag (an integer in the bitstream) corresponding to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// assert_eq!(SampleSizeSpec::from_tag(4).unwrap().into_tag(), 4);
    /// ```
    pub const fn into_tag(self) -> u8 {
        self as u8
    }

    /// Constructs `SampleSizeSpec` from the bits-per-sample value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// assert_eq!(SampleSizeSpec::from_bits(8), Some(SampleSizeSpec::B8));
    /// assert_eq!(SampleSizeSpec::from_bits(13), None);
    /// ```
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            8 => Some(Self::B8),
            12 => Some(Self::B12),
            16 => Some(Self::B16),
            20 => Some(Self::B20),
            24 => Some(Self::B24),
            32 => Some(Self::B32),
            _ => None,
        }
    }

    /// Returns the bits-per-sample value corresponding to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// assert_eq!(SampleSizeSpec::from_bits(8).unwrap().into_bits(), Some(8));
    /// ```
    pub const fn into_bits(self) -> Option<u8> {
        match self {
            Self::Unspecified | Self::Reserved => None,
            Self::B8 => Some(8),
            Self::B12 => Some(12),
            Self::B16 => Some(16),
            Self::B20 => Some(20),
            Self::B24 => Some(24),
            Self::B32 => Some(32),
        }
    }
}

/// Enum for supported sampling rates.
///
/// Refer [`FRAME_HEADER`](https://xiph.org/flac/format.html#frame_header)
/// specification for details.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum SampleRateSpec {
    /// `Unspecified` can be used in `FrameHeader` to instruct decoders to get
    /// sample rate information from `StreamInfo`.
    Unspecified,
    /// 88.2kHz.
    R88_2kHz,
    /// 176.4kHz.
    R176_4kHz,
    /// 192kHz.
    R192kHz,
    /// 8kHz.
    R8kHz,
    /// 16kHZ.
    R16kHz,
    /// 22.05kHz.
    R22_05kHz,
    /// 24kHz.
    R24kHz,
    /// 32kHz.
    R32kHz,
    /// 44.1kHz.
    R44_1kHz,
    /// 48kHz.
    R48kHz,
    /// 96kHz.
    R96kHz,
    /// An immediate value specifying kHz up to 255kHz.
    KHz(u8),
    /// An immediate value specifying Hz up to 65535Hz.
    Hz(u16),
    /// An immediate value specifying deca-Hz up to 655.35kHz.
    DaHz(u16),
}

impl SampleRateSpec {
    /// Constructs `SampleRateSpec` from frequency in Hz.
    ///
    /// This method returns None if the specified `freq` is higher than the maximum representable
    /// value. When this function is called with a non-typical frequency (that is representable in
    /// `SampleRateSpec::R*` variants), this function tries to use `KHz`, `DaHz`, and `Hz` in this
    /// order. This function never returns `Self::Unspecified`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// assert_eq!(SampleRateSpec::from_freq(44100), Some(SampleRateSpec::R44_1kHz));
    /// assert_eq!(SampleRateSpec::from_freq(44084), Some(SampleRateSpec::Hz(44084u16)));
    /// assert_eq!(SampleRateSpec::from_freq(44080), Some(SampleRateSpec::DaHz(4408u16)));
    /// assert_eq!(SampleRateSpec::from_freq(44000), Some(SampleRateSpec::KHz(44u8)));
    /// assert_eq!(SampleRateSpec::from_freq(65537), None);
    /// ```
    pub fn from_freq(freq: u32) -> Option<Self> {
        match freq {
            88_200 => Some(Self::R88_2kHz),
            176_400 => Some(Self::R176_4kHz),
            192_000 => Some(Self::R192kHz),
            8_000 => Some(Self::R8kHz),
            16_000 => Some(Self::R16kHz),
            22_050 => Some(Self::R22_05kHz),
            24_000 => Some(Self::R24kHz),
            32_000 => Some(Self::R32kHz),
            44_100 => Some(Self::R44_1kHz),
            48_000 => Some(Self::R48kHz),
            96_000 => Some(Self::R96kHz),
            _ => None,
        }
        .or_else(|| {
            (0 == freq % 1000)
                .then(|| (freq / 1000).try_into().ok().map(Self::KHz))
                .flatten()
        })
        .or_else(|| {
            (0 == freq % 10)
                .then(|| (freq / 10).try_into().ok().map(Self::DaHz))
                .flatten()
        })
        .or_else(|| freq.try_into().ok().map(Self::Hz))
    }

    /// Returns the number of extra bits required to store the specification.
    fn count_extra_bits(self) -> usize {
        match self {
            Self::KHz(_) => 8,
            Self::DaHz(_) | Self::Hz(_) => 16,
            Self::Unspecified
            | Self::R88_2kHz
            | Self::R176_4kHz
            | Self::R192kHz
            | Self::R8kHz
            | Self::R16kHz
            | Self::R22_05kHz
            | Self::R24kHz
            | Self::R32kHz
            | Self::R44_1kHz
            | Self::R48kHz
            | Self::R96kHz => 0,
        }
    }

    /// Returns 4-bit indicator for the sample-rate specifier.
    fn tag(self) -> u8 {
        match self {
            Self::Unspecified => 0,
            Self::R88_2kHz => 1,
            Self::R176_4kHz => 2,
            Self::R192kHz => 3,
            Self::R8kHz => 4,
            Self::R16kHz => 5,
            Self::R22_05kHz => 6,
            Self::R24kHz => 7,
            Self::R32kHz => 8,
            Self::R44_1kHz => 9,
            Self::R48kHz => 10,
            Self::R96kHz => 11,
            Self::KHz(_) => 12,
            Self::Hz(_) => 13,
            Self::DaHz(_) => 14,
        }
    }

    /// Writes
    fn write_extra_bits<S: BitSink>(self, dest: &mut S) -> Result<(), S::Error> {
        match self {
            Self::KHz(v) => dest.write_lsbs(v, 8),
            Self::DaHz(v) | Self::Hz(v) => dest.write_lsbs(v, 16),
            Self::Unspecified
            | Self::R88_2kHz
            | Self::R176_4kHz
            | Self::R192kHz
            | Self::R8kHz
            | Self::R16kHz
            | Self::R22_05kHz
            | Self::R24kHz
            | Self::R32kHz
            | Self::R44_1kHz
            | Self::R48kHz
            | Self::R96kHz => Ok(()),
        }
    }
}

/// [`FRAME_HEADER`](https://xiph.org/flac/format.html#frame_header) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrameHeader {
    variable_block_size: bool, // must be same in all frames
    block_size: u16,           // encoded with special function
    channel_assignment: ChannelAssignment,
    sample_size_spec: SampleSizeSpec,
    sample_rate_spec: SampleRateSpec,
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
            sample_size_spec: SampleSizeSpec::Unspecified,
            sample_rate_spec: SampleRateSpec::Unspecified,
            frame_number: 0,
            start_sample_number: start_sample_number as u64,
        }
    }

    /// Constructs `FrameHeader` in variable-length mode.
    ///
    /// # Errors
    ///
    /// Returns error when `block_size` or `start_sample_number` is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::bitsink::*;
    /// let header = FrameHeader::new_variable_size(
    ///     192,
    ///     ChannelAssignment::Independent(1),
    ///     SampleSizeSpec::B8,
    /// 123456).unwrap();
    /// let mut sink = ByteSink::new();
    /// header.write(&mut sink);
    /// assert_eq!(&sink.as_slice()[..8], &[
    ///     0xFF, 0xF9, // sync-code + fixed/var
    ///     0x10, 0x02, // block size + rate + channel + sample size + reserved
    ///     0xF0, 0x9E, 0x89, 0x80 // start sample number encoded in utf-8
    /// ]);
    /// ```
    pub fn new_variable_size(
        block_size: usize,
        channel_assignment: ChannelAssignment,
        bits_per_sample: SampleSizeSpec,
        start_sample_number: usize,
    ) -> Result<Self, VerifyError> {
        verify_block_size!("block_size", block_size)?;
        // TODO: `channel_assignment` is not following the verifocation guideline.
        //       So, it needs to be checked here.
        channel_assignment.verify()?;
        let mut ret = Self::new(block_size, channel_assignment, start_sample_number);
        ret.sample_size_spec = bits_per_sample;
        Ok(ret)
    }

    /// Constructs `FrameHeader` in fixed-length mode.
    ///
    /// # Errors
    ///
    /// Returns error when `block_size` or `frame_number` is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::bitsink::*;
    /// let header = FrameHeader::new_fixed_size(
    ///     192,
    ///     ChannelAssignment::Independent(1),
    ///     SampleSizeSpec::B8,
    /// 12345).unwrap();
    /// let mut sink = ByteSink::new();
    /// header.write(&mut sink);
    /// assert_eq!(&sink.as_slice()[..7], &[
    ///     0xFF, 0xF8, // sync-code + fixed/var
    ///     0x10, 0x02, // block size + rate + channel + sample size + reserved
    ///     0xE3, 0x80, 0xB9 // frame number encoded in utf-8
    /// ]);
    /// ```
    pub fn new_fixed_size(
        block_size: usize,
        channel_assignment: ChannelAssignment,
        bits_per_sample: SampleSizeSpec,
        frame_number: usize,
    ) -> Result<Self, VerifyError> {
        verify_block_size!("block_size", block_size)?;
        verify_range!("frame_number", frame_number, 0..=(u32::MAX as usize))?;
        // TODO: `channel_assignment` is not following the verifocation guideline.
        //       So, it needs to be checked here.
        channel_assignment.verify()?;
        let mut ret = Self::new(block_size, channel_assignment, 0);
        ret.sample_size_spec = bits_per_sample;
        ret.set_frame_number(frame_number as u32);
        Ok(ret)
    }

    /// Clear `variable_block_size` flag, and set `frame_number`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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
    /// assert_eq!(sink.as_slice()[1] & 0x01u8, 0u8);
    /// assert_eq!(sink.as_slice()[4], 12u8);
    /// ```
    pub fn set_frame_number(&mut self, frame_number: u32) {
        self.variable_block_size = false;
        self.frame_number = frame_number;
    }

    /// Hidden temporary function for resetting `sample_rate_spec`.
    ///
    /// TODO: Streamline how to specify SampleBlockSpec and SampleRateSpec and remove this hacky
    /// shortcut.
    #[cfg(feature = "__export_decode")]
    pub fn set_sample_rate_spec(&mut self, spec: SampleRateSpec) {
        self.sample_rate_spec = spec;
    }

    /// Overwrites channel assignment information of the frame.
    pub(crate) fn reset_channel_assignment(&mut self, channel_assignment: ChannelAssignment) {
        self.channel_assignment = channel_assignment;
    }

    /// Resets `sample_size_spec` field using [`StreamInfo`].
    ///
    /// This field must be specified for Claxon compatibility.
    pub(crate) fn reset_sample_size_spec(&mut self, stream_info: &StreamInfo) {
        self.sample_size_spec = SampleSizeSpec::from_bits(stream_info.bits_per_sample)
            .unwrap_or(SampleSizeSpec::Unspecified);
    }

    /// Resets `sample_rate_spec` field using [`StreamInfo`].
    ///
    /// This field must be specified for Claxon compatibility.
    pub(crate) fn reset_sample_rate_spec(&mut self, stream_info: &StreamInfo) {
        self.sample_rate_spec = SampleRateSpec::from_freq(stream_info.sample_rate)
            .unwrap_or(SampleRateSpec::Unspecified);
    }

    /// Returns block size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// assert_eq!(header.block_size(), 160);
    /// ```
    pub fn block_size(&self) -> usize {
        self.block_size as usize
    }

    /// Returns bits-per-sample.
    ///
    /// This function returns `None` when bits-per-sample specification is
    /// given in the `FrameHeader`.  Otherwise, it returns `None` and bits-per-sample
    /// should be retrieved from [`StreamInfo`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let chs = ChannelAssignment::Independent(1);
    /// let header = FrameHeader::new_fixed_size(192, chs, SampleSizeSpec::Unspecified, 0).unwrap();
    /// assert!(header.bits_per_sample().is_none());
    ///
    /// let chs = ChannelAssignment::Independent(1);
    /// let header = FrameHeader::new_fixed_size(192, chs, SampleSizeSpec::B12, 0).unwrap();
    /// assert_eq!(header.bits_per_sample().unwrap(), 12);
    /// ```
    pub fn bits_per_sample(&self) -> Option<usize> {
        self.sample_size_spec.into_bits().map(|x| x as usize)
    }

    /// Returns [`ChannelAssignment`] of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # #[path = "doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
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

reusable!(HEADER_CRC_BUFFER: ByteSink = ByteSink::new());
static HEADER_CRC: crc::Crc<u8, crc::Table<16>> = crc::Crc::<u8, crc::Table<16>>::new(&CRC_8_FLAC);

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
        ret += self.sample_rate_spec.count_extra_bits();
        ret
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        reuse!(HEADER_CRC_BUFFER, |header_buffer: &mut ByteSink| {
            header_buffer.clear();
            header_buffer.reserve(self.count_bits());

            // sync-code + reserved 1-bit + variable-block indicator
            let header_word = 0xFFF8u16 + u16::from(self.variable_block_size);
            // ^ `from` converts true to 1 and false to 0.
            header_buffer.write_lsbs(header_word, 16).unwrap();

            let (head, foot, footsize) = block_size_spec(self.block_size);
            // head + 4-bit sample rate specifier.
            header_buffer
                .write_lsbs(head << 4 | self.sample_rate_spec.tag(), 8)
                .unwrap();
            self.channel_assignment
                .write(header_buffer)
                .map_err(OutputError::<S>::ignore_sink_error)?;

            // sample size specifier + 1-bit reserved (zero)
            header_buffer
                .write_lsbs(self.sample_size_spec.into_tag() << 1, 4)
                .unwrap();

            if self.variable_block_size {
                let v = encode_to_utf8like(self.start_sample_number)?;
                header_buffer.write_bytes_aligned(&v).unwrap();
            } else {
                let v = encode_to_utf8like(self.frame_number.into())?;
                header_buffer.write_bytes_aligned(&v).unwrap();
            }
            header_buffer.write_lsbs(foot, footsize).unwrap();
            self.sample_rate_spec
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

impl Verify for FrameHeader {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_block_size!("block_size", self.block_size as usize)?;

        self.channel_assignment
            .verify()
            .map_err(|e| e.within("channel_assignment"))
    }
}

/// [`SUBFRAME`](https://xiph.org/flac/format.html#subframe) component.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum SubFrame {
    /// This variant contains [`Constant`] sub-frame.
    Constant(Constant),
    /// This variant contains [`Verbatim`] sub-frame.
    Verbatim(Verbatim),
    /// This variant contains [`FixedLpc`] sub-frame.
    FixedLpc(FixedLpc),
    /// This variant contains [`Lpc`] sub-frame.
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

impl Verify for SubFrame {
    fn verify(&self) -> Result<(), VerifyError> {
        match self {
            Self::Verbatim(c) => c.verify(),
            Self::Constant(c) => c.verify(),
            Self::FixedLpc(c) => c.verify(),
            Self::Lpc(c) => c.verify(),
        }
    }
}

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for SubFrame {
    fn signal_len(&self) -> usize {
        match self {
            Self::Verbatim(c) => c.signal_len(),
            Self::Constant(c) => c.signal_len(),
            Self::FixedLpc(c) => c.signal_len(),
            Self::Lpc(c) => c.signal_len(),
        }
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        match self {
            Self::Verbatim(c) => c.copy_signal(dest),
            Self::Constant(c) => c.copy_signal(dest),
            Self::FixedLpc(c) => c.copy_signal(dest),
            Self::Lpc(c) => c.copy_signal(dest),
        }
    }
}

/// [`SUBFRAME_CONSTANT`](https://xiph.org/flac/format.html#subframe_constant) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Constant {
    block_size: usize,
    dc_offset: i32,
    bits_per_sample: u8,
}

impl Constant {
    /// Constructs new `Constant`.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if an argument is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let subframe = Constant::new(1024, 3, 16)?;
    /// let mut sink = ByteSink::new();
    /// subframe.write(&mut sink)?;
    /// assert_eq!(sink.as_slice(), [
    ///     0x00, /* tag */
    ///     0x00, 0x03,  /* 16bits written from MSB to LSB */
    /// ]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        block_size: usize,
        dc_offset: i32,
        bits_per_sample: usize,
    ) -> Result<Self, VerifyError> {
        verify_block_size!("block_size", block_size)?;
        verify_bps!("bits_per_sample", bits_per_sample)?;
        verify_sample_range!("dc_offset", dc_offset, bits_per_sample)?;
        Ok(Self::from_parts(
            block_size,
            dc_offset,
            bits_per_sample as u8,
        ))
    }

    /// Constructs new `Constant`. (unverified version)
    pub(crate) fn from_parts(block_size: usize, dc_offset: i32, bits_per_sample: u8) -> Self {
        Self {
            block_size,
            dc_offset,
            bits_per_sample,
        }
    }

    /// Returns block size.
    pub fn block_size(&self) -> usize {
        self.block_size
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

impl Verify for Constant {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_block_size!("block_size", self.block_size)?;
        verify_bps!("bits_per_sample", self.bits_per_sample as usize)?;
        verify_sample_range!("dc_offset", self.dc_offset, self.bits_per_sample)
    }
}

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for Constant {
    fn signal_len(&self) -> usize {
        self.block_size
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.block_size);
        dest[0..self.signal_len()].fill(self.dc_offset);
    }
}

/// [`SUBFRAME_VERBATIM`](https://xiph.org/flac/format.html#subframe_verbatim) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Verbatim {
    data: Vec<i32>,
    bits_per_sample: u8,
}

impl Verbatim {
    /// Constructs new `Verbatim`.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if an argument is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let subframe = Verbatim::new(&[0xAB; 64], 16)?;
    /// let mut sink = ByteSink::new();
    /// subframe.write(&mut sink)?;
    /// assert_eq!(sink.as_slice()[0], 0x02); /* tag */
    /// for t in 0..64 {
    ///     assert_eq!(
    ///         sink.as_slice()[(1 + t * 2)..][..2],
    ///         [0x00, 0xAB]
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(samples: &[i32], bits_per_sample: usize) -> Result<Self, VerifyError> {
        verify_bps!("bits_per_sample", bits_per_sample)?;
        for v in samples {
            verify_sample_range!("samples", *v, bits_per_sample)?;
        }
        Ok(Self::from_samples(samples, bits_per_sample as u8))
    }

    /// Constructs new `Verbatim`. (unverified version)
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

impl Verify for Verbatim {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_block_size!("data.len", self.data.len())?;
        verify_bps!("bits_per_sample", self.bits_per_sample as usize)?;
        for (t, v) in self.data.iter().enumerate() {
            verify_sample_range!("data[{t}]", *v, self.bits_per_sample)?;
        }
        Ok(())
    }
}

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for Verbatim {
    fn signal_len(&self) -> usize {
        self.data.len()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.signal_len());
        dest[0..self.signal_len()].copy_from_slice(&self.data);
    }
}

/// [`SUBFRAME_FIXED`](https://xiph.org/flac/format.html#subframe_fixed) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixedLpc {
    warm_up: heapless::Vec<i32, 4>,
    residual: Residual,
    bits_per_sample: u8,
}

impl FixedLpc {
    /// Constructs new `FixedLpc`.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if an argument is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let residual = Residual::new(0, 64, 1, &[8], &[0; 64], &[0; 64])?; // zero
    /// let subframe = FixedLpc::new(&[0xCDi32], residual, 16)?;
    /// let mut sink = ByteSink::new();
    /// subframe.write(&mut sink)?;
    /// assert_eq!(sink.as_slice()[0], 0x12); /* tag */
    /// assert_eq!(sink.as_slice()[1..3], [0x00, 0xCD]); /* warmup */
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        warm_up: &[i32],
        residual: Residual,
        bits_per_sample: usize,
    ) -> Result<Self, VerifyError> {
        verify_bps!("bits_per_sample", bits_per_sample)?;
        for v in warm_up {
            verify_sample_range!("warm_up", *v, bits_per_sample)?;
        }
        let warm_up = heapless::Vec::from_slice(warm_up)
            .map_err(|()| VerifyError::new("warm_up", "must be shorter than (or equal to) 4"))?;
        let ret = Self::from_parts(warm_up, residual, bits_per_sample as u8);
        Ok(ret)
    }

    /// Creates `FixedLpc`.
    ///
    /// # Panics
    ///
    /// Panics when `warm_up.len()`, i.e. the order of LPC, is larger than the
    /// maximum fixed-LPC order (4).
    pub(crate) fn from_parts(
        warm_up: heapless::Vec<i32, 4>,
        residual: Residual,
        bits_per_sample: u8,
    ) -> Self {
        Self {
            warm_up,
            residual,
            bits_per_sample,
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

    /// Returns a reference to the internal [`Residual`] component.
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

impl Verify for FixedLpc {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_bps!("bits_per_sample", self.bits_per_sample as usize)?;
        for (t, v) in self.warm_up.iter().enumerate() {
            verify_sample_range!("warm_up[{t}]", *v, self.bits_per_sample)?;
        }
        self.residual.verify().map_err(|err| err.within("residual"))
    }
}

/// Common utility function for decoding of both `FixedLpc` and `Lpc`.
#[cfg(any(test, feature = "__export_decode"))]
fn decode_lpc<T: Into<i64> + Copy>(
    warm_up: &[i32],
    coefs: &[T],
    shift: usize,
    residual: &Residual,
    dest: &mut [i32],
) {
    residual.copy_signal(dest);
    for (t, x) in warm_up.iter().enumerate() {
        dest[t] = *x;
    }
    for t in warm_up.len()..residual.signal_len() {
        let mut pred: i64 = 0i64;
        for (tau, w) in coefs.iter().enumerate() {
            pred += <T as Into<i64>>::into(*w) * i64::from(dest[t - 1 - tau]);
        }
        dest[t] += (pred >> shift) as i32;
    }
}

#[cfg(any(test, feature = "__export_decode"))]
const FIXED_LPC_COEFS: [[i32; MAX_FIXED_LPC_ORDER]; MAX_FIXED_LPC_ORDER + 1] = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [2, -1, 0, 0],
    [3, -3, 1, 0],
    [4, -6, 4, -1],
];

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for FixedLpc {
    fn signal_len(&self) -> usize {
        self.residual.signal_len()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        let order = self.warm_up.len();
        decode_lpc(
            &self.warm_up,
            &FIXED_LPC_COEFS[order][0..order],
            0usize,
            &self.residual,
            dest,
        );
    }
}

/// [`SUBFRAME_LPC`](https://xiph.org/flac/format.html#subframe_lpc) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lpc {
    parameters: QuantizedParameters,
    warm_up: heapless::Vec<i32, MAX_LPC_ORDER>,
    residual: Residual,
    bits_per_sample: u8,
}

impl Lpc {
    /// Constructs new `Lpc`.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if an argument is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let residual = Residual::new(0, 64, 1, &[8], &[0; 64], &[0; 64])?; // zero
    /// let p = QuantizedParameters::new(&[1], 1, 0, 7)?; // designed to be 16 bits
    /// let subframe = Lpc::new(&[0xEFi32], p, residual, 16)?;
    /// let mut sink = ByteSink::new();
    /// subframe.write(&mut sink)?;
    /// assert_eq!(sink.as_slice()[0], 0x40); // tag
    /// assert_eq!(sink.as_slice()[1..3], [0x00, 0xEF]); // warm-up
    /// assert_eq!(sink.as_slice()[3..5], [0x60, 0x01]); // warm-up
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        warm_up: &[i32],
        parameters: QuantizedParameters,
        residual: Residual,
        bits_per_sample: usize,
    ) -> Result<Self, VerifyError> {
        verify_bps!("bits_per_sample", bits_per_sample)?;
        for v in warm_up {
            verify_sample_range!("warm_up", *v, bits_per_sample)?;
        }
        let warm_up = heapless::Vec::from_slice(warm_up).map_err(|()| {
            VerifyError::new(
                "warm_up",
                "must be shorter than (or equal to) `qlpc::MAX_ORDER`",
            )
        })?;
        let ret = Self::from_parts(warm_up, parameters, residual, bits_per_sample as u8);
        ret.verify()?;
        Ok(ret)
    }

    /// Constructs `Lpc`.
    ///
    /// # Panics
    ///
    /// Panics if the length of `warm_up` is not equal to `parameters.order()`.
    pub(crate) fn from_parts(
        warm_up: heapless::Vec<i32, MAX_LPC_ORDER>,
        parameters: QuantizedParameters,
        residual: Residual,
        bits_per_sample: u8,
    ) -> Self {
        assert_eq!(warm_up.len(), parameters.order());
        Self {
            parameters,
            warm_up,
            residual,
            bits_per_sample,
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
    pub fn parameters(&self) -> &QuantizedParameters {
        &self.parameters
    }

    /// Returns a reference to the internal [`Residual`] component.
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

impl Verify for Lpc {
    fn verify(&self) -> Result<(), VerifyError> {
        self.parameters
            .verify()
            .map_err(|err| err.within("parameters"))?;
        verify_bps!("bits_per_sample", self.bits_per_sample as usize)?;
        for (t, v) in self.warm_up.iter().enumerate() {
            verify_sample_range!("warm_up[{t}]", *v, self.bits_per_sample)?;
        }
        self.residual.verify().map_err(|err| err.within("residual"))
    }
}

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for Lpc {
    fn signal_len(&self) -> usize {
        self.residual.signal_len()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        decode_lpc(
            &self.warm_up,
            &self.parameters.coefs(),
            self.parameters.shift() as usize,
            &self.residual,
            dest,
        );
    }
}

/// Quantized LPC coefficients.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantizedParameters {
    #[cfg_attr(feature = "serde", serde(with = "SimdDef"))]
    pub(crate) coefs: simd::i16x32,
    order: usize,
    shift: i8,
    precision: usize,
}

/// Dequantizes QLPC parameter. (Only used for debug/ test currently.)
#[inline]
fn dequantize_parameter(coef: i16, shift: i8) -> f32 {
    let scalefac = 2.0f32.powi(-i32::from(shift));
    f32::from(coef) * scalefac
}

impl QuantizedParameters {
    /// Constructs new `QuantizedParameters`.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if an argument is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let p = QuantizedParameters::new(&[1], 1, 0, 7)?;
    /// assert_eq!(p.coefficient(0), Some(1));
    /// assert_eq!(p.coefficient(1), None);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        coefs: &[i16],
        order: usize,
        shift: i8,
        precision: usize,
    ) -> Result<Self, VerifyError> {
        let ret = Self::from_parts(coefs, order, shift, precision);
        // `QuantizedParameter` doesn't have a child component, so calling
        // `verify` here is not redundant whereas it incurs redundant checks
        // when the struct has a child component.
        ret.verify()?;
        Ok(ret)
    }

    /// Constructs new `QuantizedParameters` from parts without data verification.
    pub(crate) fn from_parts(coefs: &[i16], order: usize, shift: i8, precision: usize) -> Self {
        debug_assert!(coefs.len() == order);
        let mut coefs_v = simd::i16x32::default();
        coefs_v[0..order].copy_from_slice(coefs);
        Self {
            coefs: coefs_v,
            order,
            shift,
            precision,
        }
    }

    /// Returns the order of LPC specified by this parameter.
    #[inline]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns precision.
    #[inline]
    pub const fn precision(&self) -> usize {
        self.precision
    }

    /// Returns the shift parameter.
    #[inline]
    pub const fn shift(&self) -> i8 {
        self.shift
    }

    /// Returns an individual coefficient in quantized form.
    pub fn coefficient(&self, idx: usize) -> Option<i16> {
        (idx < self.order()).then(|| self.coefs[idx])
    }

    /// Returns `Vec` containing quantized coefficients.
    pub(crate) fn coefs(&self) -> Vec<i16> {
        (0..self.order()).map(|j| self.coefs[j]).collect()
    }

    /// Returns `Vec` containing dequantized coefficients.
    #[allow(dead_code)]
    pub(crate) fn dequantized(&self) -> Vec<f32> {
        self.coefs()
            .iter()
            .map(|x| dequantize_parameter(*x, self.shift))
            .collect()
    }
}

impl Verify for QuantizedParameters {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_range!("order", self.order, ..=MAX_LPC_ORDER)?;
        verify_range!("shift", self.shift, MIN_LPC_SHIFT..=MAX_LPC_SHIFT)?;
        verify_range!("precision", self.precision, ..=MAX_LPC_PRECISION)?;
        Ok(())
    }
}

/// [`RESIDUAL`](https://xiph.org/flac/format.html#residual) component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    // Some pre-computed values.
    sum_quotients: usize,
    sum_rice_params: usize,
}

impl Residual {
    /// Constructs `Residual` from loaded encodes.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if an argument is invalid.
    pub fn new(
        partition_order: usize,
        block_size: usize,
        warmup_length: usize,
        rice_params: &[u8],
        quotients: &[u32],
        remainders: &[u32],
    ) -> Result<Self, VerifyError> {
        // Some pre-construction verification
        let ret = Self::from_parts(
            partition_order as u8,
            block_size,
            warmup_length,
            rice_params.to_owned(),
            quotients.to_owned(),
            remainders.to_owned(),
        );
        ret.verify()?;
        Ok(ret)
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
        debug_assert!(rice_params.len() == 1usize << partition_order as usize);

        let max_quotients: usize = find_max::<64>(&quotients) as usize;
        let sum_quotients: usize = if max_quotients * block_size < u32::MAX as usize {
            // If overflow-safe, use SIMD.
            wrapping_sum::<u32, 32>(&quotients) as usize
        } else {
            quotients.iter().map(|x| *x as usize).sum()
        };
        let sum_rice_params: usize = rice_params.iter().map(|x| *x as usize).sum();

        Self {
            partition_order,
            block_size,
            warmup_length,
            rice_params,
            quotients,
            remainders,
            sum_quotients,
            sum_rice_params,
        }
    }

    /// Returns the partition order for the PRC.
    pub fn partition_order(&self) -> usize {
        self.partition_order as usize
    }

    /// Returns the rice parameter for the `p`-th partition
    pub fn rice_parameter(&self, p: usize) -> usize {
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

const RESIDUAL_WRITE_UNROLL_N: usize = 4;
impl BitRepr for Residual {
    #[inline]
    fn count_bits(&self) -> usize {
        let nparts = 1usize << self.partition_order as usize;
        let quotient_bits: usize = self.sum_quotients + self.block_size - self.warmup_length;

        let mut remainder_bits: usize =
            self.sum_rice_params * (self.block_size >> self.partition_order);
        remainder_bits -= self.warmup_length * self.rice_params[0] as usize;
        2 + 4 + nparts * 4 + quotient_bits + remainder_bits
    }

    /// Writes `Residual` to the [`BitSink`].
    ///
    /// This is the most inner-loop of the output part of the encoder, so
    /// computational efficiency is prioritized more than readability.
    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), OutputError<S>> {
        // The number of partitions with 00 (indicating 4-bit mode) prepended.
        dest.write_lsbs(self.partition_order, 6)
            .map_err(OutputError::<S>::from_sink)?;
        let nparts = 1usize << self.partition_order as usize;

        // unforunatelly, the overhead due to the use of iterators is visible
        // here. so we back-off to integer-based loops. (drawback of index-loop
        // is boundary check, but this will be skipped in release builds.)
        let part_len = self.block_size >> self.partition_order;
        let mut p = 0;
        let mut offset = 0;
        while p < nparts {
            let rice_p = self.rice_params[p];
            dest.write_lsbs(rice_p, 4)
                .map_err(OutputError::<S>::from_sink)?;
            let start = std::cmp::max(self.warmup_length, offset);
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
                        let q = self.quotients[t] as usize;
                        let r_plus_startbit =
                            (self.remainders[t] | startbit) << (32 - rice_p_plus_1);
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

impl Verify for Residual {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_true!(
            "self.quotients",
            self.quotients.len() == self.remainders.len(),
            "quotients and remainders must have the same number of elements"
        )?;
        verify_block_size!("quotients.len", self.quotients.len())?;

        verify_true!(
            "quotients.len",
            self.quotients.len() == self.block_size,
            "must have the same length as the block size"
        )?;
        verify_true!(
            "remainders.len",
            self.remainders.len() == self.block_size,
            "must have the same length as the block size"
        )?;
        for t in 0..self.warmup_length {
            verify_true!(
                "quotients[{t}]",
                self.quotients[t] == 0,
                "must be zero for warmup samples"
            )?;
            verify_true!(
                "remainders[{t}]",
                self.remainders[t] == 0,
                "must be zero for warmup samples"
            )?;
        }

        let partition_count = 1 << self.partition_order;
        let partition_len = self.block_size / partition_count;
        for t in 0..self.block_size {
            let rice_p = self.rice_params[t / partition_len];
            verify_range!("remainders[{t}]", self.remainders[t], ..(1 << rice_p))?;
        }

        let sum_quotients_check: usize = self
            .quotients
            .iter()
            .fold(0usize, |acc, x| acc + *x as usize);
        let sum_rice_params_check: usize = self
            .rice_params
            .iter()
            .fold(0usize, |acc, x| acc + *x as usize);
        verify_true!(
            "sum_quotients",
            self.sum_quotients == sum_quotients_check,
            "must be identical with the actual sum of quotients"
        )?;
        verify_true!(
            "sum_rice_params",
            self.sum_rice_params == sum_rice_params_check,
            "must be identical with the actual sum of rice parameters"
        )?;
        Ok(())
    }
}

#[cfg(any(test, feature = "__export_decode"))]
impl Decode for Residual {
    fn signal_len(&self) -> usize {
        self.block_size
    }

    #[allow(clippy::needless_range_loop)]
    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.signal_len());

        let part_len = self.block_size >> self.partition_order;
        assert!(part_len > 0);

        for t in 0..self.block_size {
            dest[t] = rice::decode_signbit(
                (self.quotients[t] << self.rice_params[t / part_len]) + self.remainders[t],
            );
        }
    }
}

mod seal_bit_repr {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigen;
    use crate::sigen::Signal;

    use rand::distributions::Distribution;
    use rand::distributions::Uniform;

    fn make_frame(stream_info: &StreamInfo, samples: &[i32], offset: usize) -> Frame {
        let channels = stream_info.channels as usize;
        let block_size = samples.len() / channels;
        let bits_per_sample: u8 = stream_info.bits_per_sample;
        let ch_info = ChannelAssignment::Independent(channels as u8);
        let mut frame = Frame::new_empty(ch_info, offset, block_size);
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
    fn write_empty_stream() -> Result<(), OutputError<MemSink<u8>>> {
        let stream = Stream::new(44100, 2, 16).unwrap();
        let mut bv: MemSink<u8> = MemSink::new();
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
    fn write_stream_info() -> Result<(), OutputError<MemSink<u8>>> {
        let stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let mut bv: MemSink<u8> = MemSink::new();
        stream_info.write(&mut bv)?;
        assert_eq!(bv.len(), 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128);
        assert_eq!(stream_info.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn write_frame_header() -> Result<(), OutputError<MemSink<u8>>> {
        let header = FrameHeader::new(2304, ChannelAssignment::Independent(2), 192);
        let mut bv: MemSink<u8> = MemSink::new();
        header.write(&mut bv)?;

        // test with canonical frame
        let header = FrameHeader::new(192, ChannelAssignment::Independent(2), 0);
        let mut bv: MemSink<u8> = MemSink::new();
        header.write(&mut bv)?;

        assert_eq!(
            bv.to_bitstring(),
            concat!(
                "11111111_111110", // sync
                "01_",             // reserved/ blocking strategy (const in this impl)
                "00010000_",       // block size/ sample_rate (0=header)
                "00010000_",       // channel/ bps (0=header)/ reserved
                "00000000_",       // sample number
                "01101001",        // crc8
            )
        );

        assert_eq!(header.count_bits(), bv.len());

        Ok(())
    }

    #[test]
    fn write_verbatim_frame() -> Result<(), OutputError<MemSink<u64>>> {
        let nchannels: usize = 3;
        let nsamples: usize = 17;
        let bits_per_sample: usize = 16;
        let stream_info = StreamInfo::new(16000, nchannels, bits_per_sample).unwrap();
        let framebuf = vec![-1i32; nsamples * nchannels];
        let frame = make_frame(&stream_info, &framebuf, 0);
        let mut bv: MemSink<u64> = MemSink::new();

        frame.header().write(&mut bv)?;
        assert_eq!(frame.header().count_bits(), bv.len());

        for ch in 0..3 {
            bv.clear();
            frame.subframe(ch).unwrap().write(&mut bv)?;
            assert_eq!(frame.subframe(ch).unwrap().count_bits(), bv.len());
        }

        bv.clear();
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
    fn channel_assignment_encoding() -> Result<(), OutputError<MemSink<u8>>> {
        let ch = ChannelAssignment::Independent(8);
        let mut bv: MemSink<u8> = MemSink::new();
        ch.write(&mut bv)?;
        assert_eq!(bv.to_bitstring(), "0111****");
        let ch = ChannelAssignment::RightSide;
        let mut bv: MemSink<u8> = MemSink::new();
        ch.write(&mut bv)?;
        assert_eq!(bv.to_bitstring(), "1001****");
        assert_eq!(ch.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn stream_info_update() {
        let mut stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let framebuf = sigen::Dc::new(0.01)
            .noise(0.002)
            .to_vec_quantized(16, 256 * 2);
        let frame1 = make_frame(&stream_info, &framebuf, 0);
        stream_info.update_frame_info(&frame1);
        let framebuf = sigen::Dc::new(0.02)
            .noise(0.1)
            .to_vec_quantized(16, 192 * 2);
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
    fn bit_count_residual() -> Result<(), OutputError<MemSink<u64>>> {
        let mut rng = rand::thread_rng();
        let block_size = 4 * Uniform::from(16..=1024).sample(&mut rng);
        let partition_order: usize = 2;
        let nparts = 2usize.pow(partition_order as u32);
        let part_len = block_size / nparts;
        let params = vec![7, 8, 6, 7];
        let mut quotients: Vec<u32> = vec![];
        let mut remainders: Vec<u32> = vec![];

        for t in 0..block_size {
            let part_id = t / part_len;
            let p = params[part_id];
            let denom = 1u32 << p;

            quotients.push((255 / denom) as u32);
            remainders.push((255 % denom) as u32);
        }
        let residual = Residual::new(
            partition_order,
            block_size,
            0,
            &params,
            &quotients,
            &remainders,
        )
        .expect("Residual construction failed.");
        residual
            .verify()
            .expect("should construct a valid Residual");

        let mut bv: MemSink<u64> = MemSink::new();
        residual.write(&mut bv)?;

        assert_eq!(residual.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn frame_bitstream_precomputataion() -> Result<(), OutputError<MemSink<u64>>> {
        let stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let samples = sigen::Sine::new(128, 0.2)
            .noise(0.1)
            .to_vec_quantized(12, 512);
        let mut frame = make_frame(&stream_info, &samples, 0);
        let mut bv_ref: MemSink<u64> = MemSink::new();
        let frame_cloned = frame.clone();
        frame_cloned.write(&mut bv_ref)?;
        assert!(bv_ref.len() % 8 == 0); // frame must be byte-aligned.

        frame.precompute_bitstream();
        assert!(frame.is_bitstream_precomputed());
        assert!(!frame_cloned.is_bitstream_precomputed());

        let mut bv: MemSink<u64> = MemSink::new();
        frame.write(&mut bv)?;
        assert_eq!(bv.to_bitstring(), bv_ref.to_bitstring());

        // this makes `Frame` broken as the header says it has two channels.
        frame.add_subframe(frame.subframe(0).unwrap().clone());
        // anyway cache should be discarded.
        assert!(!frame.is_bitstream_precomputed());
        Ok(())
    }

    #[test]
    fn channel_assignment_is_small_enough() {
        let size = std::mem::size_of::<ChannelAssignment>();
        assert_eq!(size, 2);
    }
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    fn residual_write_to_u64s(b: &mut Bencher) {
        let warmup_len = 13;
        let mut quotients = [2u32; 4096];
        let mut remainders = [0u32; 4096];
        for t in 0..warmup_len {
            quotients[t] = 0u32;
            remainders[t] = 0u32;
        }
        let residual = Residual::new(8, 4096, warmup_len, &[8u8; 256], &quotients, &remainders)
            .expect("Residual construction failed.");
        let mut sink = MemSink::<u64>::with_capacity(4096 * 2 * 8);

        b.iter(|| {
            sink.clear();
            residual.write(black_box(&mut sink))
        });
    }

    #[bench]
    fn residual_bit_counter(b: &mut Bencher) {
        let warmup_len = 13;
        let mut quotients = [2u32; 4096];
        let mut remainders = [0u32; 4096];
        for t in 0..warmup_len {
            quotients[t] = 0u32;
            remainders[t] = 0u32;
        }
        let residual = Residual::new(8, 4096, warmup_len, &[8u8; 256], &quotients, &remainders)
            .expect("Residual construction failed.");

        b.iter(|| black_box(&residual).count_bits());
    }
}
