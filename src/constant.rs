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

//! Configuration constants

/// Exit code: Invalid config
pub enum ExitCode {
    #[allow(dead_code)]
    Normal = 0,
    InvalidConfig = -1,
}

/// Maximum length of a block.
pub const MIN_BLOCKSIZE_SUPPORTED: usize = 32;

/// Maximum length of a block.
pub const MAX_BLOCKSIZE_SUPPORTED: usize = 32768;

/// Maximum length of a block.
#[allow(dead_code)]
pub const MAX_BLOCKSIZE: usize = 65536;

/// Maximum number of channels supported.
pub const MAX_CHANNELS: usize = 8;

/// Maximum order of LPC supported. (32 by specification.)
pub const MAX_LPC_ORDER: usize = 24;

/// Maximum order of LPC supported. (32 by specification.)
pub const MAX_LPC_ORDER_PLUS_1: usize = MAX_LPC_ORDER + 1;

/// Maximum allowed rice parameters (incl.)
///
/// 5-bit rice coding is not supported currently, and 0b1111 is reserved for
/// verbatim encoding. So, 14 will be the maximum.
pub const MAX_RICE_PARAMETER: usize = 14;

/// Maximum order of rice parameter partitioning (incl.)
pub const MAX_RICE_PARTITION_ORDER: usize = 15;

/// Maximum number of rice partitions (excl.)
pub const MAX_RICE_PARTITIONS: usize = 1usize << MAX_RICE_PARTITION_ORDER;

/// Minimum rice partition size. (1 by specification)
///
/// For some source, degradation observed only when increasing this value
/// from 256 to 512.  Here, it is set to 64 for robustness.
pub const MIN_RICE_PARTITION_SIZE: usize = 64;

/// Maximum bits-per-sample.
#[allow(dead_code)]
pub const MAX_SAMPLE_BITS: usize = 32;

/// The number of `FrameBuf`s for each worker thread in par-mode.
pub const PAR_MODE_FRAMEBUF_MULTIPLICITY: usize = 2;

/// The number of bits used for encoding shift bits of QLPC.
pub const QLPC_SHIFT_BITS: usize = 5;

/// Max (by spec) precision for storing QLPC coefficients.
pub const QLPC_MAX_PRECISION: usize = 16;

/// Maximum shift parameter defined by the specification.
pub const QLPC_MAX_SHIFT: i8 = (1i8 << (QLPC_SHIFT_BITS - 1)) - 1;

/// Minimum shift parameter.
///
/// According to the bitstream specification, it can be -16
/// (`-(1 << QLPC_SHIFT_BITS)`), but the reference decoder doesn't support
/// negative shift case.
pub const QLPC_MIN_SHIFT: i8 = 0;

/// Default LPC order for QLPC module.
pub const QLPC_DEFAULT_ORDER: usize = 10;

/// Default precision for storing QLPC coefficients.
pub const QLPC_DEFAULT_PRECISION: usize = 12;
