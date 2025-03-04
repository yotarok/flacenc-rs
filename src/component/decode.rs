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

use super::super::constant::fixed::MAX_LPC_ORDER as MAX_FIXED_LPC_ORDER;
use super::super::rice;
use super::bitrepr::seal_bit_repr;

use super::datatype::*;
/// Traits for FLAC components containing signals (represented in [`i32`]).
///
/// "Signal" here has slightly different meaning depending on the component
/// that implements this trait. For example, for `Residual`, signal is a
/// prediction error signal. For `SubFrame`, signal means a single-channel
/// sequence of samples whereas for `Frame`, signal is an interleaved multi-
/// channel samples.
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

impl Decode for Frame {
    fn signal_len(&self) -> usize {
        self.block_size() * self.subframe_count()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.signal_len());

        let mut channels = vec![];
        for sf in self.subframes() {
            channels.push(sf.decode());
        }

        match self.header().channel_assignment() {
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

impl Decode for Constant {
    fn signal_len(&self) -> usize {
        self.block_size()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.block_size());
        dest[0..self.signal_len()].fill(self.dc_offset());
    }
}

impl Decode for Verbatim {
    fn signal_len(&self) -> usize {
        self.samples().len()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.signal_len());
        dest[0..self.signal_len()].copy_from_slice(self.samples());
    }
}

/// Common utility function for decoding of both `FixedLpc` and `Lpc`.
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

const FIXED_LPC_COEFS: [[i32; MAX_FIXED_LPC_ORDER]; MAX_FIXED_LPC_ORDER + 1] = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [2, -1, 0, 0],
    [3, -3, 1, 0],
    [4, -6, 4, -1],
];

impl Decode for FixedLpc {
    fn signal_len(&self) -> usize {
        self.residual().signal_len()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        let order = self.warm_up().len();
        decode_lpc(
            self.warm_up(),
            &FIXED_LPC_COEFS[order][0..order],
            0usize,
            self.residual(),
            dest,
        );
    }
}

impl Decode for Lpc {
    fn signal_len(&self) -> usize {
        self.residual().signal_len()
    }

    fn copy_signal(&self, dest: &mut [i32]) {
        decode_lpc(
            self.warm_up(),
            &self.parameters().coefs(),
            self.parameters().shift() as usize,
            self.residual(),
            dest,
        );
    }
}

impl Decode for Residual {
    fn signal_len(&self) -> usize {
        self.block_size()
    }

    #[allow(clippy::needless_range_loop)]
    fn copy_signal(&self, dest: &mut [i32]) {
        assert!(dest.len() >= self.signal_len());

        let part_len = self.block_size() >> self.partition_order();
        assert!(part_len > 0);

        for t in 0..self.block_size() {
            dest[t] = rice::decode_signbit(
                (self.quotients()[t] << self.rice_params()[t / part_len]) + self.remainders()[t],
            );
        }
    }
}
