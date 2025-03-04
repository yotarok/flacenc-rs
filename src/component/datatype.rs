// Copyright 2022-2024 Google LLC
// Copyright 2025- flacenc-rs developers
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
use std::cmp::min;

#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use super::bitrepr::BitRepr;
use super::verify::verify_block_size;
use super::verify::verify_bps;
use super::verify::verify_sample_range;
use crate::arrayutils::find_max;
use crate::arrayutils::wrapping_sum;
use crate::bitsink::BitSink;
use crate::bitsink::MemSink;
use crate::constant::panic_msg;
use crate::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use crate::constant::MAX_CHANNELS;
use crate::error::verify_range;
use crate::error::verify_true;
use crate::error::Verify;
use crate::error::VerifyError;
use crate::rice;

import_simd!(as simd);

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

    /// Returns [`StreamInfo`] wrapped in a [`MetadataBlock`].
    pub(crate) fn stream_info_block(&self) -> &MetadataBlock {
        &self.stream_info
    }

    /// Returns a mutable reference to [`StreamInfo`] associated with `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is corrupted by manually modifying fields.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let mut stream = Stream::new(16000, 1, 24).unwrap();
    /// stream.stream_info_mut().set_total_samples(123456);
    /// assert_eq!(stream.stream_info().total_samples(), 123456);
    /// ```
    pub fn stream_info_mut(&mut self) -> &mut StreamInfo {
        if let MetadataBlockData::StreamInfo(ref mut info) = self.stream_info.data {
            info
        } else {
            panic!("Stream is not properly initialized.")
        }
    }

    /// Appends [`Frame`] to this `Stream` and updates [`StreamInfo`].
    ///
    /// This also updates frame statistics in `stream_info` but does not update
    /// MD5 checksums. For updating those, call `set_md5_digest` manually via
    /// [`Self::stream_info_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// let frame0 = stream.frame(0).expect("0-th frame is not found.");
    /// let frame19 = stream.frame(19).expect("19-th frame is not found.");
    /// assert!(frame0.count_bits() > 0);
    /// assert!(frame19.count_bits() > 0);
    /// assert!(stream.frame(200).is_none());
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
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
    /// let stream = make_example_stream(signal_len, block_size, channels, sample_rate);
    /// assert_eq!(stream.frame_count(), 200);
    /// ```
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub(crate) fn metadata(&self) -> &[MetadataBlock] {
        &self.metadata
    }

    /// Returns [`Frame`]s as a slice.
    #[allow(dead_code)]
    pub(crate) fn frames(&self) -> &[Frame] {
        &self.frames
    }

    pub(crate) fn verify_variable_blocking_frames(&self) -> Result<(), VerifyError> {
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
            current = current.wrapping_add(frame.header.block_size() as u64);
        }
        Ok(())
    }

    pub(crate) fn verify_fixed_blocking_frames(&self) -> Result<(), VerifyError> {
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

    /// Returns [`Frame`]s as mutable Vec.
    #[cfg(any(test, feature = "decode"))]
    pub(crate) fn frames_mut(&mut self) -> &mut Vec<Frame> {
        &mut self.frames
    }
}

/// [`METADATA_BLOCK`](https://xiph.org/flac/format.html#metadata_block) component.
///
/// It is currently hidden from the public interface because [`MetadataBlockData`]
/// and relevant accessor methods in [`Stream`] can hide this implementation detail.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetadataBlock {
    pub(crate) is_last: bool,
    pub(crate) data: MetadataBlockData,
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
    /// Constructs new `MetadataBlockData::Unknown` from the content (in [`u8`]).
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

    /// Updates `StreamInfo` with values from the given [`Frame`].
    ///
    /// This function updates `{min|max}_{block|frame}_size` and
    /// `total_samples`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    /// assert_eq!(info.min_block_size(), 31234 % 160);
    ///
    /// assert_eq!(info.total_samples(), 31234);
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

    /// Returns the minimum frame size in bytes.
    ///
    /// [`METADATA_BLOCK_STREAM_INFO`]: https://xiph.org/flac/format.html#metadata_block_streaminfo
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    #[inline]
    pub fn min_frame_size(&self) -> usize {
        self.min_frame_size as usize
    }

    /// Returns the maximum frame size in bytes.
    ///
    /// [`METADATA_BLOCK_STREAM_INFO`]: https://xiph.org/flac/format.html#metadata_block_streaminfo
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
    ///
    /// assert_eq!(info.max_frame_size(), 0);
    /// ```
    #[inline]
    pub fn max_frame_size(&self) -> usize {
        self.max_frame_size as usize
    }

    /// Returns the minimum block size in samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    #[inline]
    pub fn min_block_size(&self) -> usize {
        self.min_block_size as usize
    }

    /// Returns the maximum block size in samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
    ///
    /// assert_eq!(info.max_block_size(), 0);
    /// ```
    #[inline]
    pub fn max_block_size(&self) -> usize {
        self.max_block_size as usize
    }

    /// Returns sampling rate of the stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
    /// assert_eq!(info.sample_rate(), 16000);
    /// ```
    #[inline]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate as usize
    }

    /// Returns the number of channels of the stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
    /// assert_eq!(info.channels(), 2);
    /// ```
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels as usize
    }

    /// Returns bits-per-sample of the stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let info = StreamInfo::new(16000, 2, 16).unwrap();
    /// assert_eq!(info.bits_per_sample(), 16);
    /// ```
    #[inline]
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }

    /// Returns the number of samples of the stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    /// assert_eq!(info.total_samples(), 31234);
    /// ```
    #[inline]
    pub fn total_samples(&self) -> usize {
        self.total_samples as usize
    }

    /// Sets the number of samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::source::{Context, Fill};
    /// # use flacenc::*;
    /// let mut ctx = Context::new(16, 2);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    /// ctx.fill_interleaved(&[0x0000_0FFFi32; 246]);
    /// info.set_total_samples(ctx.total_samples());
    /// assert_ne!(info.total_samples(), 246); // not sample count
    /// assert_eq!(info.total_samples(), 123); // but inter-channel sample count
    /// ```
    #[inline]
    pub fn set_total_samples(&mut self, n: usize) {
        self.total_samples = n as u64;
    }

    /// Returns md5 digest of the input waveform.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    #[inline]
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
    /// let mut ctx = Context::new(16, 2);
    /// let mut info = StreamInfo::new(16000, 2, 16).unwrap();
    /// assert_eq!(info.md5_digest(), &[0x00u8; 16]);
    /// ctx.fill_interleaved(&[0x0000_0FFFi32; 256]);
    /// info.set_md5_digest(&ctx.md5_digest());
    /// assert_ne!(info.md5_digest(), &[0x00u8; 16]);
    /// ```
    pub fn set_md5_digest(&mut self, digest: &[u8; 16]) {
        self.md5.copy_from_slice(digest);
    }

    /// Resets the minimum/ maximum block sizes.
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

    /// Resets the minimum/ maximum frame sizes.
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
    /// Constructs an empty `Frame`.
    ///
    /// This makes an invalid `Frame`; therefore this shouldn't be "pub" so far. Specifically,
    /// the frame offset to the header must be set before actually written to the bitstream.
    pub(crate) fn new_empty(
        block_size_spec: BlockSizeSpec,
        ch_info: ChannelAssignment,
        sample_size_spec: SampleSizeSpec,
        sample_rate_spec: SampleRateSpec,
    ) -> Self {
        let header =
            FrameHeader::from_specs(block_size_spec, ch_info, sample_size_spec, sample_rate_spec);
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
    /// let header = FrameHeader::new(192, chs, 8, 44100, FrameOffset::Frame(0)).unwrap();
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
    #[inline]
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
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// let (header, subframes) = frame.into_parts();
    ///
    /// assert_eq!(subframes.len(), 2);
    /// ```
    #[inline]
    pub fn into_parts(self) -> (FrameHeader, Vec<SubFrame>) {
        (self.header, self.subframes)
    }

    /// Adds subframe.
    ///
    /// # Panics
    ///
    /// Panics when the number of subframes added exceeded the `MAX_CHANNELS`.
    #[inline]
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
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    /// assert_eq!(frame.header().block_size(), 160);
    /// ```
    #[inline]
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns a mutable reference to [`FrameHeader`] of this frame.
    #[inline]
    pub(crate) fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    /// Returns [`SubFrame`] for the given channel.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
    /// # #[path = "../doctest_helper.rs"]
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

    #[inline]
    pub(crate) fn subframes(&self) -> &[SubFrame] {
        &self.subframes
    }

    /// Returns block size of this frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let frame = make_example_frame(signal_len, block_size, channels, sample_rate);
    ///
    /// assert_eq!(frame.block_size(), 160);
    /// ```
    #[inline]
    pub fn block_size(&self) -> usize {
        self.header.block_size()
    }

    /// Allocates precomputed bitstream buffer, and precomputes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
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
        let mut dest = MemSink::<u8>::with_capacity(self.count_bits());
        if self.write(&mut dest).is_ok() {
            self.precomputed_bitstream = Some(dest.into_inner());
        }
    }

    #[inline]
    pub(crate) fn precomputed_bitstream(&self) -> Option<&Vec<u8>> {
        self.precomputed_bitstream.as_ref()
    }

    /// Consumes `self` and returns the parts if `self` is a stereo frame.
    ///
    /// # Errors
    ///
    /// When `self.subframe_count() != 2`, this function returns the reconstructed self.
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
    #[inline]
    pub(crate) const fn is_bitstream_precomputed(&self) -> bool {
        self.precomputed_bitstream.is_some()
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
    #[inline]
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
    #[inline]
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

    #[inline]
    pub(crate) fn channels(&self) -> usize {
        if let Self::Independent(n) = self {
            *n as usize
        } else {
            2
        }
    }
}

/// Enum representing the location of frame either by a frame count or starting-sample number.
///
/// The use of `Self::Frame` implies fixed-blocking mode, and `Self::StartSample` implies variable
/// blocking mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameOffset {
    /// Frame offset specifier based on the number of frames preceding.
    Frame(u32),
    /// Frame offset specifier based on the number of samples preceding.
    StartSample(u64),
}

/// Reimplementation of `u32::ilog2` for older rust compilers.
///
/// # Panics
///
/// It panics when `x == 0`.
#[inline]
fn ilog2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// Enum for block size specifier in [`FrameHeader`].
///
/// Refer [`FRAME_HEADER`](https://xiph.org/flac/format.html#frame_header)
/// specification for details.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum BlockSizeSpec {
    /// Reserved.
    #[allow(dead_code)]
    Reserved,
    /// Special case when `size = 192`.
    S192,
    /// Size that can be represented as `size = 576 * 2^n` where `n` in `0..=3`.
    Pow2Mul576(u8),
    /// Size that is stored in a byte at the end of [`FrameHeader`].
    ExtraByte(u8),
    /// Size that is stored in two bytes at the end of [`FrameHeader`].
    ExtraTwoBytes(u16),
    /// Size that can be represented as `size = 256 * 2^n` where `n` in `0..=8`.
    Pow2Mul256(u8),
}

impl BlockSizeSpec {
    /// Constructs `BlockSizeSpec` from frequency in Hz.
    ///
    /// This function never returns `Self::Reserved`.
    #[inline]
    pub fn from_size(size: u16) -> Self {
        match size {
            192 => Self::S192,
            576 | 1152 | 2304 | 4608 => Self::Pow2Mul576(ilog2(u32::from(size / 576)) as u8),
            256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 => {
                Self::Pow2Mul256(ilog2(u32::from(size / 256)) as u8)
            }
            x if x <= 256 => Self::ExtraByte((x - 1) as u8),
            x => Self::ExtraTwoBytes(x - 1),
        }
    }

    /// Returns the number of extra bits required to store the specification.
    #[inline]
    pub(crate) fn count_extra_bits(self) -> usize {
        match self {
            Self::ExtraByte(_) => 8,
            Self::ExtraTwoBytes(_) => 16,
            Self::Reserved | Self::S192 | Self::Pow2Mul576(_) | Self::Pow2Mul256(_) => 0,
        }
    }

    #[inline]
    pub fn block_size(self) -> Option<usize> {
        match self {
            Self::Reserved => None,
            Self::S192 => Some(192),
            Self::Pow2Mul576(x) => Some(576usize * (1usize << x as usize)),
            Self::ExtraByte(x) => Some((x + 1) as usize),
            Self::ExtraTwoBytes(x) => Some((x + 1) as usize),
            Self::Pow2Mul256(x) => Some(256usize * (1usize << x as usize)),
        }
    }

    /// Returns 4-bit indicator for the sample-rate specifier.
    #[inline]
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::Reserved => 0,
            Self::S192 => 1,
            Self::Pow2Mul576(x) => 2 + x,
            Self::ExtraByte(_) => 6,
            Self::ExtraTwoBytes(_) => 7,
            Self::Pow2Mul256(x) => 8 + x,
        }
    }

    /// Writes extra data field to `dest`.
    #[inline]
    pub(crate) fn write_extra_bits<S: BitSink>(self, dest: &mut S) -> Result<(), S::Error> {
        match self {
            Self::ExtraByte(v) => dest.write_lsbs(v, 8),
            Self::ExtraTwoBytes(v) => dest.write_lsbs(v, 16),
            Self::Reserved | Self::S192 | Self::Pow2Mul576(_) | Self::Pow2Mul256(_) => Ok(()),
        }
    }
}

/// Enum for supported sample sizes.
///
/// Refer [`FRAME_HEADER`](https://xiph.org/flac/format.html#frame_header)
/// specification for details.
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
    #[inline]
    #[allow(dead_code)]
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
    #[inline]
    pub const fn into_tag(self) -> u8 {
        self as u8
    }

    /// Constructs `SampleSizeSpec` from the bits-per-sample value.
    #[inline]
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
    #[inline]
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
    #[inline]
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

    #[cfg(any(test, feature = "decode"))]
    #[inline]
    pub(crate) fn from_tag_and_data(tag: u8, value: Option<usize>) -> Option<Self> {
        if tag > 0b1110 {
            return None;
        }
        Some(match tag {
            0b0000 => Self::Unspecified,
            0b0001 => Self::R88_2kHz,
            0b0010 => Self::R176_4kHz,
            0b0011 => Self::R192kHz,
            0b0100 => Self::R8kHz,
            0b0101 => Self::R16kHz,
            0b0110 => Self::R22_05kHz,
            0b0111 => Self::R24kHz,
            0b1000 => Self::R32kHz,
            0b1001 => Self::R44_1kHz,
            0b1010 => Self::R48kHz,
            0b1011 => Self::R96kHz,
            0b1100 => Self::KHz(value? as u8),
            0b1101 => Self::Hz(value? as u16),
            0b1110 => Self::DaHz(value? as u16),
            _ => unreachable!(), // this arm is covered in the first if-stmt of this fn.
        })
    }

    /// Returns the number of extra bits required to store the specification.
    #[inline]
    pub(crate) fn count_extra_bits(self) -> usize {
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
    #[inline]
    pub(crate) fn tag(self) -> u8 {
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

    /// Writes extra data field to `dest`.
    #[inline]
    pub(crate) fn write_extra_bits<S: BitSink>(self, dest: &mut S) -> Result<(), S::Error> {
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
    block_size_spec: BlockSizeSpec,
    channel_assignment: ChannelAssignment,
    sample_size_spec: SampleSizeSpec,
    sample_rate_spec: SampleRateSpec,
    frame_number: u32,        // written when variable_block_size == false
    start_sample_number: u64, // written when variable_block_size == true
}

impl FrameHeader {
    #[inline]
    pub(crate) const fn from_specs(
        block_size_spec: BlockSizeSpec,
        channel_assignment: ChannelAssignment,
        sample_size_spec: SampleSizeSpec,
        sample_rate_spec: SampleRateSpec,
    ) -> Self {
        Self {
            variable_block_size: true,
            block_size_spec,
            channel_assignment,
            sample_size_spec,
            sample_rate_spec,
            frame_number: 0,
            start_sample_number: 0,
        }
    }

    /// Constructs `FrameHeader` from the given metadata.
    ///
    /// # Errors
    ///
    /// Returns error when `block_size` or `bits_per_sample` is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// # use flacenc::bitsink::*;
    /// let header = FrameHeader::new(
    ///     192, ChannelAssignment::Independent(1), 8, 44100, FrameOffset::StartSample(123456)
    /// ).unwrap();
    /// let mut sink = ByteSink::new();
    /// header.write(&mut sink);
    /// assert_eq!(&sink.as_slice()[..8], &[
    ///     0xFF, 0xF9, // sync-code + fixed/var
    ///     0x19, 0x02, // block size + rate + channel + sample size + reserved
    ///     0xF0, 0x9E, 0x89, 0x80 // start sample number encoded in utf-8
    /// ]);
    /// ```
    #[inline]
    pub fn new(
        block_size: usize,
        channel_assignment: ChannelAssignment,
        bits_per_sample: usize,
        sample_rate: usize,
        offset: FrameOffset,
    ) -> Result<Self, VerifyError> {
        verify_block_size!("block_size", block_size)?;
        let block_size_spec = BlockSizeSpec::from_size(block_size as u16);
        let sample_size_spec =
            SampleSizeSpec::from_bits(bits_per_sample as u8).ok_or_else(|| {
                VerifyError::new("bits_per_sample", "must be one of a supported value.")
            })?;
        verify_true!(
            "bits_per_sample",
            sample_size_spec != SampleSizeSpec::B32,
            "32-bit encoding is not supported currently."
        )?;
        channel_assignment.verify()?;
        let sample_rate_spec = SampleRateSpec::from_freq(sample_rate as u32)
            .ok_or_else(|| VerifyError::new("sample_rate", "must be in a supported range."))?;
        let mut ret = Self::from_specs(
            block_size_spec,
            channel_assignment,
            sample_size_spec,
            sample_rate_spec,
        );
        ret.set_frame_offset(offset);
        Ok(ret)
    }

    #[inline]
    pub(crate) fn is_variable_blocking(&self) -> bool {
        self.variable_block_size
    }

    /// Sets the location of frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let mut header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// header.set_frame_offset(FrameOffset::Frame(3));
    ///
    /// header.set_frame_offset(FrameOffset::StartSample(480));
    /// ```
    #[inline]
    pub fn set_frame_offset(&mut self, offset: FrameOffset) {
        match offset {
            FrameOffset::Frame(n) => self.set_frame_number(n),
            FrameOffset::StartSample(n) => self.set_start_sample_number(n),
        }
    }

    /// Enters to fixed blocking mode, and sets `frame_number`.
    #[inline]
    fn set_frame_number(&mut self, frame_number: u32) {
        self.variable_block_size = false;
        self.frame_number = frame_number;
    }

    /// Enters to variable blocking mode, and sets `start_sample_number`.
    #[inline]
    fn set_start_sample_number(&mut self, start_sample_number: u64) {
        self.variable_block_size = true;
        self.start_sample_number = start_sample_number;
    }

    #[inline]
    pub(crate) fn frame_number(&self) -> u32 {
        self.frame_number
    }

    #[inline]
    pub(crate) fn start_sample_number(&self) -> u64 {
        self.start_sample_number
    }

    #[inline]
    pub(crate) fn sample_rate_spec(&self) -> &SampleRateSpec {
        &self.sample_rate_spec
    }

    #[inline]
    pub(crate) fn sample_size_spec(&self) -> &SampleSizeSpec {
        &self.sample_size_spec
    }

    /// Overwrites channel assignment information of the frame.
    #[inline]
    pub(crate) fn reset_channel_assignment(&mut self, channel_assignment: ChannelAssignment) {
        self.channel_assignment = channel_assignment;
    }

    /// Returns block size.
    ///
    /// # Panics
    ///
    /// It panics if `self` is deserialized from external data, and in an invalid state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// # use flacenc::component::*;
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 2, 16000);
    /// let header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// assert_eq!(header.block_size(), 160);
    /// ```
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size_spec
            .block_size()
            .expect("Reserved block-size tag should not be used.")
    }

    /// Returns block size spec.
    #[inline]
    pub(crate) fn block_size_spec(&self) -> BlockSizeSpec {
        self.block_size_spec
    }

    /// Returns bits-per-sample.
    ///
    /// This function returns `None` when bits-per-sample specification is not
    /// given in the `FrameHeader`. Currently, flacenc cannot construct `FrameHeader` without
    /// valid bits-per-sample specification.  Therefore, this function returns `None` only
    /// when `self` is deserialized from external data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::component::*;
    /// let chs = ChannelAssignment::Independent(1);
    /// let header = FrameHeader::new(192, chs, 12, 44100, FrameOffset::Frame(0)).unwrap();
    /// assert_eq!(header.bits_per_sample().unwrap(), 12);
    /// ```
    #[inline]
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
    /// # #[path = "../doctest_helper.rs"]
    /// # mod doctest_helper;
    /// # use doctest_helper::*;
    /// let (signal_len, block_size, channels, sample_rate) = (31234, 160, 1, 16000);
    /// let header = make_example_frame_header(signal_len, block_size, channels, sample_rate);
    ///
    /// // this is only used for stereo signal, and it will be always `Independent` for
    /// // non-stereo signals.
    /// assert_eq!(header.channel_assignment(), &ChannelAssignment::Independent(1));
    /// ```
    #[inline]
    pub fn channel_assignment(&self) -> &ChannelAssignment {
        &self.channel_assignment
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
    #[inline]
    pub(crate) fn from_parts(block_size: usize, dc_offset: i32, bits_per_sample: u8) -> Self {
        Self {
            block_size,
            dc_offset,
            bits_per_sample,
        }
    }

    /// Returns block size.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns offset value.
    #[inline]
    pub fn dc_offset(&self) -> i32 {
        self.dc_offset
    }

    /// Returns bits-per-sample.
    #[inline]
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
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
    #[inline]
    pub fn samples(&self) -> &[i32] {
        &self.data
    }

    /// Returns bits-per-sample.
    #[inline]
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
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
    #[inline]
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
    #[inline]
    pub fn order(&self) -> usize {
        self.warm_up.len()
    }

    /// Returns warm-up samples as a slice.
    #[inline]
    pub fn warm_up(&self) -> &[i32] {
        &self.warm_up
    }

    /// Returns a reference to the internal [`Residual`] component.
    #[inline]
    pub fn residual(&self) -> &Residual {
        &self.residual
    }

    /// Returns bits-per-sample.
    #[inline]
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
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
    #[inline]
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
    #[inline]
    pub const fn order(&self) -> usize {
        self.parameters.order()
    }

    /// Returns warm-up samples as a slice.
    #[inline]
    pub fn warm_up(&self) -> &[i32] {
        &self.warm_up
    }

    /// Returns a reference to parameter struct.
    #[inline]
    pub fn parameters(&self) -> &QuantizedParameters {
        &self.parameters
    }

    /// Returns a reference to the internal [`Residual`] component.
    #[inline]
    pub fn residual(&self) -> &Residual {
        &self.residual
    }

    /// Returns bits-per-sample.
    #[inline]
    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
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
    #[inline]
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
    #[inline]
    pub(crate) fn coefs(&self) -> Vec<i16> {
        (0..self.order()).map(|j| self.coefs[j]).collect()
    }

    /// Returns `Vec` containing dequantized coefficients.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn dequantized(&self) -> Vec<f32> {
        self.coefs()
            .iter()
            .map(|x| dequantize_parameter(*x, self.shift))
            .collect()
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
    #[inline]
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
    #[inline]
    pub fn partition_order(&self) -> usize {
        self.partition_order as usize
    }

    /// Returns the rice parameter for the `p`-th partition
    #[inline]
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

    /// Returns the block size of this `Residual`.
    ///
    /// In common use cases, this accessor is not necessary as `block_size` is normally known
    /// before constructing `Residual`. However, in some use cases like tests, it's convenient to
    /// have it here.
    #[inline]
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the warmup length of this `Residual`.
    ///
    /// In common use cases, this accessor is not necessary as `warmup_length` is normally known
    /// before constructing `Residual`. However, in some use cases like tests, it's convenient to
    /// have it here.
    #[inline]
    pub(crate) fn warmup_length(&self) -> usize {
        self.warmup_length
    }

    #[inline]
    pub(crate) fn sum_quotients(&self) -> usize {
        self.sum_quotients
    }

    #[inline]
    pub(crate) fn sum_rice_params(&self) -> usize {
        self.sum_rice_params
    }

    #[inline]
    pub(crate) fn rice_params(&self) -> &[u8] {
        &self.rice_params
    }

    #[inline]
    pub(crate) fn quotients(&self) -> &[u32] {
        &self.quotients
    }

    #[inline]
    pub(crate) fn remainders(&self) -> &[u32] {
        &self.remainders
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Maybe, this test is not very informative but it's still good to ensure some conditions for
    // avoiding future performance regression.
    #[test]
    fn channel_assignment_is_small_enough() {
        let size = std::mem::size_of::<ChannelAssignment>();
        assert_eq!(size, 2);
    }
}
