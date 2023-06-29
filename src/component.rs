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

use bitvec::bits;
use bitvec::prelude::BitVec;
use bitvec::prelude::Lsb0;
use bitvec::prelude::Msb0;
use bitvec::view::BitView;

use super::bitsink::BitSink;
use super::constant::MAX_CHANNELS;
use super::constant::MAX_LPC_ORDER;
use super::error::EncodeError;
use super::error::RangeError;
use super::lpc;
use super::rice;

const CRC_8_FLAC: crc::Algorithm<u8> = crc::CRC_8_SMBUS;
const CRC_16_FLAC: crc::Algorithm<u16> = crc::CRC_16_UMTS;

/// FLAC components that can be represented in a bit sequence.
pub trait BitRepr {
    /// Counts the number of bits required to store the component.
    fn count_bits(&self) -> usize;

    /// Writes the bit sequence to `BitSink`.
    #[allow(clippy::missing_errors_doc)]
    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError>;
}

/// Encodes the given integer into UTF-8-like byte sequence.
fn encode_to_utf8like<S: BitSink>(val: u64, dest: &mut S) -> Result<(), RangeError> {
    let val_size = usize::BITS as usize;
    let code_bits: usize = val_size - val.leading_zeros() as usize;
    if code_bits <= 7 {
        dest.write_lsbs(val, 8);
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

        let mut val_bv: BitVec = BitVec::new();
        val_bv.extend_from_bitslice(&val.view_bits::<Msb0>()[val_size - capacity..]);

        dest.write_lsbs(0xFFu8, trailing_bytes);
        dest.write_bitslice(bits![1, 0]);
        let first_bits = 6 - trailing_bytes;
        let mut off = 0;
        dest.write_bitslice(&val_bv[off..first_bits]);

        off = first_bits;
        for _i in 0..trailing_bytes {
            dest.write_bitslice(bits![1, 0]);
            dest.write_bitslice(&val_bv[off..off + 6]);
            off += 6;
        }
    }
    Ok(())
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
    pub const fn new(sample_rate: usize, channels: usize, bits_per_sample: usize) -> Self {
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
    pub const fn stream_info(&self) -> &StreamInfo {
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
    pub fn stream_info_mut(&mut self) -> &mut StreamInfo {
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
    pub fn add_frame(&mut self, frame: Frame) {
        // Consume frame and add to stream.
        self.stream_info_mut().update_frame_info(&frame);
        self.frames.push(frame);
    }

    /// Returns `Frame` for the given frame number.
    pub fn frame(&self, n: usize) -> &Frame {
        &self.frames[n]
    }

    #[cfg(test)]
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }
}

impl BitRepr for Stream {
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

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        dest.write_lsbs(0x66_4c_61_43u32, 32); // fLaC
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
    fn count_bits(&self) -> usize {
        32 + self.data.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        let block_type = self.block_type as u32 + if self.is_last { 0x80 } else { 0x00 };
        dest.write_lsbs(block_type, 8);
        let mut data_buf: BitVec<usize> = BitVec::with_capacity(self.data.count_bits());
        self.data.write(&mut data_buf)?;
        let data_size: u32 = (data_buf.len() / 8) as u32;
        dest.write_lsbs(data_size, 24);
        dest.write_bitslice(&data_buf);
        Ok(())
    }
}

/// Enum for `BLOCK_TYPE` in `METADATA_BLOCK_HEADER`.
#[allow(dead_code)]
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
enum MetadataBlockData {
    StreamInfo(StreamInfo),
}

impl BitRepr for MetadataBlockData {
    fn count_bits(&self) -> usize {
        match self {
            Self::StreamInfo(info) => info.count_bits(),
        }
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        match self {
            Self::StreamInfo(info) => info.write(dest)?,
        }
        Ok(())
    }
}

/// [`METADATA_BLOCK_STREAM_INFO`](https://xiph.org/flac/format.html#metadata_block_streaminfo) component.
#[derive(Clone, Debug)]
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
    pub const fn new(sample_rate: usize, channels: usize, bits_per_sample: usize) -> Self {
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

    /// Updates `StreamInfo` with information from the given Frame.
    pub fn update_frame_info(&mut self, frame: &Frame) {
        let block_size = frame.block_size() as u16;
        self.min_block_size = min(block_size, self.min_block_size);
        self.max_block_size = max(block_size, self.max_block_size);
        let frame_size_in_bytes = (frame.count_bits() / 8) as u32;
        self.min_frame_size = min(frame_size_in_bytes, self.min_frame_size);
        self.max_frame_size = max(frame_size_in_bytes, self.max_frame_size);

        self.total_samples += u64::from(block_size);
    }

    pub fn set_total_samples(&mut self, n: usize) {
        self.total_samples = n as u64;
    }

    pub fn set_md5_digest(&mut self, digest: &[u8; 16]) {
        self.md5.copy_from_slice(digest);
    }

    #[cfg(test)]
    pub const fn md5(&self) -> &[u8; 16] {
        &self.md5
    }

    pub const fn channels(&self) -> usize {
        self.channels as usize
    }

    pub const fn bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }
}

impl BitRepr for StreamInfo {
    fn count_bits(&self) -> usize {
        272
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        dest.write::<u16>(self.min_block_size);
        dest.write::<u16>(self.max_block_size);
        dest.write_lsbs(self.min_frame_size, 24);
        dest.write_lsbs(self.max_frame_size, 24);
        dest.write_lsbs(self.sample_rate, 20);
        dest.write_lsbs(self.channels - 1, 3);
        dest.write_lsbs(self.bits_per_sample - 1, 5);
        dest.write_lsbs(self.total_samples, 36);
        dest.write(self.md5);
        Ok(())
    }
}

/// [`FRAME`](https://xiph.org/flac/format.html#frame) component.
#[derive(Clone, Debug)]
pub struct Frame {
    header: FrameHeader,
    subframes: heapless::Vec<SubFrame, MAX_CHANNELS>,
}

impl Frame {
    pub const fn block_size(&self) -> usize {
        self.header.block_size as usize
    }

    /// Constructs an empty Frame.
    pub const fn new(ch_info: ChannelAssignment, offset: usize, block_size: usize) -> Self {
        let header = FrameHeader::new(block_size, ch_info, offset);
        Self {
            header,
            subframes: heapless::Vec::new(),
        }
    }

    /// Constructs Frame from `FrameHeader` and `SubFrame`s.
    pub fn from_parts<I>(header: FrameHeader, subframes: I) -> Self
    where
        I: Iterator<Item = SubFrame>,
    {
        Self {
            header,
            subframes: subframes.collect(),
        }
    }

    pub fn add_subframe(&mut self, subframe: SubFrame) {
        self.subframes
            .push(subframe)
            .expect("Exceeded maximum number of channels.");
    }

    /// Returns `FrameHeader` of this frame.
    pub const fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns a mutable reference to `FrameHeader` of this frame.
    pub fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    /// Returns `SubFrame` for the given channel.
    pub fn subframe(&self, ch: usize) -> &SubFrame {
        &self.subframes[ch]
    }
}

impl BitRepr for Frame {
    fn count_bits(&self) -> usize {
        let header = self.header.count_bits();
        let body: usize = self.subframes.iter().map(BitRepr::count_bits).sum();

        let aligned = (header + body + 7) / 8 * 8;
        let footer = 16;
        aligned + footer
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        let mut frame_buffer: BitVec<u8, Msb0> = BitVec::with_capacity(self.count_bits());

        self.header.write(&mut frame_buffer)?;
        for sub in self.subframes.iter() {
            sub.write(&mut frame_buffer)?;
        }
        frame_buffer.align_to_byte();

        dest.write_bitslice(&frame_buffer);
        dest.write(crc::Crc::<u16>::new(&CRC_16_FLAC).checksum(frame_buffer.as_raw_slice()));
        Ok(())
    }
}

/// Enum for channel assignment in `FRAME_HEADER`.
#[derive(Clone, Debug)]
pub enum ChannelAssignment {
    Independent(u8),
    LeftSide,
    RightSide,
    MidSide,
}

impl ChannelAssignment {
    /// Returns the number of extra bit required to store the channel samples.
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
}

impl BitRepr for ChannelAssignment {
    fn count_bits(&self) -> usize {
        4
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        match *self {
            Self::Independent(ch) => {
                if ch > 8 {
                    return Err(RangeError::from_display("#channel", "cannot exceed 8", &ch).into());
                }
                dest.write_lsbs(ch - 1, 4);
            }
            Self::LeftSide => {
                dest.write_bitslice(bits![1, 0, 0, 0]);
            }
            Self::RightSide => {
                dest.write_bitslice(bits![1, 0, 0, 1]);
            }
            Self::MidSide => {
                dest.write_bitslice(bits![1, 0, 1, 0]);
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
    // frame-wise sample_rate field is not supported
    channel_assignment: ChannelAssignment,
    // frame-wise sample_size in bits is not supported
    frame_number: u32,        // written when variable_block_size == false
    start_sample_number: u64, // written when variable_block_size == true
}

impl FrameHeader {
    pub const fn new(
        block_size: usize,
        channel_assignment: ChannelAssignment,
        start_sample_number: usize,
    ) -> Self {
        Self {
            variable_block_size: true,
            block_size: block_size as u16,
            channel_assignment,
            frame_number: 0,
            start_sample_number: start_sample_number as u64,
        }
    }

    /// Clear `variable_block_size` flag, and set `frame_number`.
    pub fn enter_fixed_size_mode(&mut self, frame_number: u32) {
        self.variable_block_size = false;
        self.frame_number = frame_number;
    }

    /// Overwrites channel assignment information of the frame.
    pub fn reset_channel_assignment(&mut self, channel_assignment: ChannelAssignment) {
        self.channel_assignment = channel_assignment;
    }

    /// Returns block size.
    #[cfg(test)]
    pub const fn block_size(&self) -> usize {
        self.block_size as usize
    }
}

impl BitRepr for FrameHeader {
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

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        let mut header_buffer: BitVec<u8, Msb0> = BitVec::with_capacity(128 * 8);
        {
            let dest = &mut header_buffer;
            // sync-code + reserved 1-bit + variable-block indicator
            let header_word = 0xFFF8u16 + u16::from(self.variable_block_size);
            // ^ `from` converts true to 1 and false to 0.
            dest.write_lsbs(header_word, 16);

            let (head, foot, footsize) = block_size_spec(self.block_size);
            dest.write_lsbs(head, 4);
            // sample rate must be declared in header, not here.
            dest.write_bitslice(bits![0, 0, 0, 0]);
            self.channel_assignment.write(dest)?;
            // sample size must be declared in header not here.
            dest.write_bitslice(bits![0, 0, 0, 0]); // and a reserved bit added

            if self.variable_block_size {
                encode_to_utf8like(self.start_sample_number, dest)?;
            } else {
                encode_to_utf8like(u64::from(self.frame_number), dest)?;
            }
            dest.write_lsbs(foot, footsize);
        }

        dest.write_bitslice(&header_buffer);
        dest.write(crc::Crc::<u8>::new(&CRC_8_FLAC).checksum(header_buffer.as_raw_slice()));

        Ok(())
    }
}

/// [SUBFRAME](https://xiph.org/flac/format.html#subframe) component.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SubFrame {
    Constant(Constant),
    Verbatim(Verbatim),
    FixedLpc(FixedLpc),
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
    fn count_bits(&self) -> usize {
        match self {
            Self::Verbatim(c) => c.count_bits(),
            Self::Constant(c) => c.count_bits(),
            Self::FixedLpc(c) => c.count_bits(),
            Self::Lpc(c) => c.count_bits(),
        }
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
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
    pub const fn new(dc_offset: i32, bits_per_sample: u8) -> Self {
        Self {
            dc_offset,
            bits_per_sample,
        }
    }
}

impl BitRepr for Constant {
    fn count_bits(&self) -> usize {
        8 + self.bits_per_sample as usize
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        dest.write_bitslice(bits![0, 0, 0, 0, 0, 0, 0, 0]);
        dest.write_twoc(self.dc_offset, self.bits_per_sample as usize);
        Ok(())
    }
}

/// [`SUBFRAME_VERBATIM`](https://xiph.org/flac/format.html#subframe_verbatim) component.
#[derive(Clone, Debug)]
pub struct Verbatim {
    pub data: Vec<i32>,
    bits_per_sample: u8,
}

impl Verbatim {
    pub fn from_samples(samples: &[i32], bits_per_sample: u8) -> Self {
        Self {
            data: Vec::from(samples),
            bits_per_sample,
        }
    }
}

impl BitRepr for Verbatim {
    fn count_bits(&self) -> usize {
        8 + self.data.len() * (self.bits_per_sample as usize)
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        dest.write_bitslice(bits![0, 0, 0, 0, 0, 0, 1, 0]);
        for i in 0..self.data.len() {
            dest.write_twoc(self.data[i], self.bits_per_sample as usize);
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
    pub fn new(warm_up: &[i32], residual: Residual, bits_per_sample: usize) -> Self {
        let warm_up = heapless::Vec::from_slice(warm_up)
            .expect("Exceeded maximum order for FixedLPC component.");

        Self {
            warm_up,
            residual,
            bits_per_sample: bits_per_sample as u8,
        }
    }

    pub fn order(&self) -> usize {
        self.warm_up.len()
    }
}

impl BitRepr for FixedLpc {
    fn count_bits(&self) -> usize {
        8 + self.bits_per_sample as usize * self.order() + self.residual.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        let head_byte = 0x10u8 | (self.order() << 1) as u8;
        dest.write(head_byte);
        for v in self.warm_up.iter() {
            dest.write_twoc(*v, self.bits_per_sample as usize);
        }
        self.residual.write(dest)?;
        Ok(())
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
    pub fn new(
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

    pub const fn order(&self) -> usize {
        self.parameters.order()
    }

    #[allow(dead_code)]
    pub const fn parameters(&self) -> &lpc::QuantizedParameters {
        &self.parameters
    }
}

impl BitRepr for Lpc {
    fn count_bits(&self) -> usize {
        let warm_up_bits = self.bits_per_sample as usize * self.order();
        8 + warm_up_bits
            + 4
            + 5
            + self.parameters.precision() * self.order()
            + self.residual.count_bits()
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        let head_byte = 0x40 | (((self.order() - 1) as u8) << 1);
        dest.write(head_byte);

        for i in 0..self.order() {
            dest.write_twoc(self.warm_up[i], self.bits_per_sample as usize);
        }

        assert!((self.parameters.precision() as u8) < 16u8);
        dest.write_lsbs((self.parameters.precision() - 1) as u64, 4);

        // FLAC reference decoder doesn't support this.
        assert!(self.parameters.shift() >= 0);
        dest.write_twoc(self.parameters.shift(), 5);

        for ref_coef in &self.parameters.coefs() {
            debug_assert!(*ref_coef < (1 << (self.parameters.precision() - 1)));
            debug_assert!(*ref_coef >= -(1 << (self.parameters.precision() - 1)));
            dest.write_twoc(*ref_coef, self.parameters.precision());
        }

        self.residual.write(dest)?;

        Ok(())
    }
}

/// [RESIDUAL](https://xiph.org/flac/format.html#residual) component.
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
    pub fn new(
        partition_order: usize,
        block_size: usize,
        warmup_length: usize,
        rice_params: &[u8],
        quotients: &[u32],
        remainders: &[u32],
    ) -> Self {
        Self {
            partition_order: partition_order as u8,
            block_size,
            warmup_length,
            rice_params: rice_params.to_owned(),
            quotients: quotients.to_owned(),
            remainders: remainders.to_owned(),
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, id: usize) -> i32 {
        let nparts = 1usize << self.partition_order as usize;
        let part_id = id * nparts / self.block_size;
        let quotient = self.quotients[id];
        let shift = u32::from(self.rice_params[part_id]);
        let remainder = self.remainders[id];
        let v = (quotient << shift) + remainder;
        rice::decode_signbit(v)
    }
}

impl BitRepr for Residual {
    fn count_bits(&self) -> usize {
        let nparts = 1usize << self.partition_order as usize;
        let mut quotient_bits: usize = 0;
        for t in self.warmup_length..self.block_size {
            // plus 1 for stop bits.
            quotient_bits += self.quotients[t] as usize + 1;
        }
        let mut remainder_bits: usize = 0;
        for p in 0..nparts {
            let part_len: usize =
                self.block_size / nparts - if p == 0 { self.warmup_length } else { 0 };
            remainder_bits += self.rice_params[p] as usize * part_len;
        }
        2 + 4 + nparts * 4 + quotient_bits + remainder_bits
    }

    fn write<S: BitSink>(&self, dest: &mut S) -> Result<(), EncodeError> {
        // The number of partitions with 00 (indicating 4-bit mode) prepended.
        dest.write_lsbs(self.partition_order, 6);
        let nparts = 1usize << self.partition_order as usize;
        for p in 0..nparts {
            dest.write_lsbs(self.rice_params[p], 4);
            let start = std::cmp::max(
                self.warmup_length,
                (p * self.block_size) >> self.partition_order,
            );
            let end = ((p + 1) * self.block_size) >> self.partition_order;
            for t in start..end {
                let mut q = self.quotients[t] as usize;
                while q >= 64 {
                    dest.write(0u64);
                    q -= 64;
                }
                dest.write_lsbs(1u64, q + 1);
                dest.write_lsbs(self.remainders[t], self.rice_params[p] as usize);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

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
    fn write_empty_stream() -> Result<(), EncodeError> {
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
    fn write_stream_info() -> Result<(), EncodeError> {
        let stream_info = StreamInfo::new(44100, 2, 16);
        let mut bv: BitVec<u8> = BitVec::new();
        stream_info.write(&mut bv)?;
        assert_eq!(bv.len(), 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128);
        assert_eq!(stream_info.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn write_frame_header() -> Result<(), EncodeError> {
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
    fn write_verbatim_frame() -> Result<(), EncodeError> {
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
    fn utf8_encoding() -> Result<(), EncodeError> {
        let v = 0x56;
        let mut bv: BitVec<usize> = BitVec::new();
        encode_to_utf8like(v, &mut bv)?;
        assert_eq!(bv, bits![0, 1, 0, 1, 0, 1, 1, 0]);

        let v = 0x1024;
        let mut bv: BitVec<usize> = BitVec::new();
        encode_to_utf8like(v, &mut bv)?;
        assert_eq!(
            bv,
            bits![1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
        );

        let v = 0xF_FFFF_FFFFu64; // 36 bits of ones
        let mut bv: BitVec<usize> = BitVec::new();
        encode_to_utf8like(v, &mut bv)?;
        assert_eq!(
            bv,
            bits![
                1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1
            ]
        );

        let v = 0x10_0000_0000u64; //  out of domain
        let mut bv: BitVec<usize> = BitVec::new();
        encode_to_utf8like(v, &mut bv).expect_err("Should be out of domain");

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
    fn channel_assignment_encoding() -> Result<(), EncodeError> {
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
    fn bit_count_residual() -> Result<(), EncodeError> {
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
}
