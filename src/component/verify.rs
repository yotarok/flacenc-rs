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

use crate::bitsink::MemSink;
use crate::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use crate::constant::qlpc::MAX_PRECISION as MAX_LPC_PRECISION;
use crate::constant::qlpc::MAX_SHIFT as MAX_LPC_SHIFT;
use crate::constant::qlpc::MIN_SHIFT as MIN_LPC_SHIFT;
use crate::constant::MAX_CHANNELS;
use crate::error::verify_range;
use crate::error::verify_true;
use crate::error::Verify;
use crate::error::VerifyError;

use super::bitrepr::BitRepr;
use super::datatype::ChannelAssignment;
use super::datatype::Constant;
use super::datatype::FixedLpc;
use super::datatype::Frame;
use super::datatype::FrameHeader;
use super::datatype::Lpc;
use super::datatype::MetadataBlock;
use super::datatype::MetadataBlockData;
use super::datatype::QuantizedParameters;
use super::datatype::Residual;
use super::datatype::Stream;
use super::datatype::StreamInfo;
use super::datatype::SubFrame;
use super::datatype::Verbatim;

// Some (internal) utility macros for value verification.
macro_rules! verify_block_size {
    ($varname:literal, $size:expr) => {
        verify_range!($varname, $size, ..=(crate::constant::MAX_BLOCK_SIZE))
    };
}
pub(crate) use verify_block_size;

macro_rules! verify_bps {
    ($varname:literal, $bps:expr) => {
        verify_range!(
            $varname,
            $bps,
            (crate::constant::MIN_BITS_PER_SAMPLE)..=(crate::constant::MAX_BITS_PER_SAMPLE + 1)
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
pub(crate) use verify_bps;

macro_rules! verify_sample_range {
    ($varname:literal, $sample:expr, $bps:expr) => {{
        let min_sample = -((1usize << ($bps as usize - 1)) as i32);
        let max_sample = (1usize << ($bps as usize - 1)) as i32 - 1;
        verify_range!($varname, $sample, min_sample..=max_sample)
    }};
}
pub(crate) use verify_sample_range;

impl Verify for Stream {
    fn verify(&self) -> Result<(), VerifyError> {
        self.stream_info()
            .verify()
            .map_err(|e| e.within("stream_info"))?;
        for (i, md) in self.metadata().iter().enumerate() {
            md.verify()
                .map_err(|e| e.within(&format!("metadata[{i}]")))?;
            let is_last = i + 1 == self.metadata().len();

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

        if self.frames().is_empty() {
            Ok(())
        } else if self.frames()[0].header().is_variable_blocking() {
            self.verify_variable_blocking_frames()
        } else {
            self.verify_fixed_blocking_frames()
        }
    }
}

impl Verify for MetadataBlock {
    fn verify(&self) -> Result<(), VerifyError> {
        self.data.verify()
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

impl Verify for StreamInfo {
    fn verify(&self) -> Result<(), VerifyError> {
        if self.total_samples() != 0 {
            verify_true!(
                "min_block_size",
                self.min_block_size() <= self.max_block_size(),
                "must be smaller than `max_block_size`"
            )?;
            verify_block_size!("min_block_size", self.min_block_size())?;
            verify_block_size!("max_block_size", self.max_block_size())?;
            verify_true!(
                "min_frame_size",
                self.min_frame_size() <= self.max_frame_size(),
                "must be smaller than `max_frame_size`"
            )?;
        }
        verify_range!("sample_rate", self.sample_rate(), ..=96_000)?;
        verify_range!("channels", self.channels(), 1..=8)?;
        verify_bps!("bits_per_sample", self.bits_per_sample())
    }
}

impl Verify for Frame {
    fn verify(&self) -> Result<(), VerifyError> {
        for (ch, sf) in self.subframes().iter().enumerate() {
            sf.verify()
                .map_err(|e| e.within(&format!("subframe[{ch}]")))?;
        }
        if let Some(buf) = self.precomputed_bitstream() {
            let mut dest = MemSink::<u8>::with_capacity(self.count_bits());
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
        self.header().verify().map_err(|e| e.within("header"))
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

impl Verify for FrameHeader {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_block_size!("block_size", self.block_size())?;

        self.channel_assignment()
            .verify()
            .map_err(|e| e.within("channel_assignment"))
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

impl Verify for Constant {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_block_size!("block_size", self.block_size())?;
        verify_bps!("bits_per_sample", self.bits_per_sample())?;
        verify_sample_range!("dc_offset", self.dc_offset(), self.bits_per_sample())
    }
}

impl Verify for Verbatim {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_block_size!("data.len", self.samples().len())?;
        verify_bps!("bits_per_sample", self.bits_per_sample())?;
        for (t, v) in self.samples().iter().enumerate() {
            verify_sample_range!("data[{t}]", *v, self.bits_per_sample())?;
        }
        Ok(())
    }
}

impl Verify for FixedLpc {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_bps!("bits_per_sample", self.bits_per_sample())?;
        for (t, v) in self.warm_up().iter().enumerate() {
            verify_sample_range!("warm_up[{t}]", *v, self.bits_per_sample())?;
        }
        self.residual()
            .verify()
            .map_err(|err| err.within("residual"))
    }
}

impl Verify for Lpc {
    fn verify(&self) -> Result<(), VerifyError> {
        self.parameters()
            .verify()
            .map_err(|err| err.within("parameters"))?;
        verify_bps!("bits_per_sample", self.bits_per_sample())?;
        for (t, v) in self.warm_up().iter().enumerate() {
            verify_sample_range!("warm_up[{t}]", *v, self.bits_per_sample())?;
        }
        self.residual()
            .verify()
            .map_err(|err| err.within("residual"))
    }
}

impl Verify for QuantizedParameters {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_range!("order", self.order(), ..=MAX_LPC_ORDER)?;
        verify_range!("shift", self.shift(), MIN_LPC_SHIFT..=MAX_LPC_SHIFT)?;
        verify_range!("precision", self.precision(), ..=MAX_LPC_PRECISION)?;
        Ok(())
    }
}

impl Verify for Residual {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_true!(
            "self.quotients",
            self.quotients().len() == self.remainders().len(),
            "quotients and remainders must have the same number of elements"
        )?;
        verify_block_size!("quotients.len", self.quotients().len())?;

        verify_true!(
            "quotients.len",
            self.quotients().len() == self.block_size(),
            "must have the same length as the block size"
        )?;
        verify_true!(
            "remainders.len",
            self.remainders().len() == self.block_size(),
            "must have the same length as the block size"
        )?;
        for t in 0..self.warmup_length() {
            verify_true!(
                "quotients[{t}]",
                self.quotients()[t] == 0,
                "must be zero for warmup samples"
            )?;
            verify_true!(
                "remainders[{t}]",
                self.remainders()[t] == 0,
                "must be zero for warmup samples"
            )?;
        }

        let partition_count = 1 << self.partition_order();
        let partition_len = self.block_size() / partition_count;
        for t in 0..self.block_size() {
            let rice_p = self.rice_params()[t / partition_len];
            verify_range!("remainders[{t}]", self.remainders()[t], ..(1 << rice_p))?;
        }

        let sum_quotients_check: usize = self
            .quotients()
            .iter()
            .fold(0usize, |acc, x| acc + *x as usize);
        let sum_rice_params_check: usize = self
            .rice_params()
            .iter()
            .fold(0usize, |acc, x| acc + *x as usize);
        verify_true!(
            "sum_quotients",
            self.sum_quotients() == sum_quotients_check,
            "must be identical with the actual sum of quotients"
        )?;
        verify_true!(
            "sum_rice_params",
            self.sum_rice_params() == sum_rice_params_check,
            "must be identical with the actual sum of rice parameters"
        )?;
        Ok(())
    }
}
