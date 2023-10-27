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

#![allow(dead_code)] // it's okay if some FLAC-spec constants are not used.

// Constance sorted in an alphabetical-order.  Top-level constants first, and
// then sub-modules. Constants that are used only in a specific sub-module or
// its caller should be placed in the corresponding submodule.

/// Minimum length of a block supported.
pub const MIN_BLOCKSIZE: usize = 32;

/// Maximum length of a block supported (65536 in the specification.)
pub const MAX_BLOCKSIZE: usize = 32768;

/// Maximum number of channels.
pub const MAX_CHANNELS: usize = 8;

/// Maximum bits-per-sample supported.
pub const MAX_BITS_PER_SAMPLE: usize = 24;

/// Sub-module containing constants related to build-time information.
pub mod build_info {
    pub const CRATE_VERSION: &str = match option_env!("CARGO_PKG_VERSION") {
        Some(v) => v,
        None => "unknown",
    };
}

/// Constants related to keys for the environment variables.
pub(crate) mod envvar_key {
    /// Environment variable name for specifying the number of threads.
    pub const DEFAULT_PARALLELISM: &str = "FLACENC_WORKERS";
}

/// Constants related to fixed-parameter LPC encoding.
pub mod fixed {
    /// Maximum order of fixed LPC supported.
    pub const MAX_LPC_ORDER: usize = 4;
}

/// Constants related to par-mode (multithreading.)
pub mod par {
    /// The number of [`FrameBuf`]s for each worker thread in par-mode.
    ///
    /// [`FrameBuf`]: crate::source::FrameBuf
    pub const FRAMEBUF_MULTIPLICITY: usize = 2;
}

/// Constants related to quantized linear predictive coding (QLPC).
pub mod qlpc {
    /// The number of bits used for encoding shift bits of QLPC.
    pub const SHIFT_BITS: usize = 5;

    /// Maximum order of LPC supported. (32 in the specification.)
    pub const MAX_ORDER: usize = 24;

    /// Max number of bits (precision) for storing QLPC coefficients.
    pub const MAX_PRECISION: usize = 16;

    /// Maximum shift parameter of QLPC defined in the specification.
    pub const MAX_SHIFT: i8 = (1i8 << (SHIFT_BITS - 1)) - 1;

    /// Minimum shift parameter of QLPC.
    ///
    /// According to the bitstream specification, it can be -16
    /// (`-(1 << QLPC_SHIFT_BITS)`), but the reference decoder doesn't support
    /// negative shift case.
    pub const MIN_SHIFT: i8 = 0;

    /// Default LPC order for QLPC module.
    pub const DEFAULT_ORDER: usize = 10;

    /// Default precision for storing QLPC coefficients.
    pub const DEFAULT_PRECISION: usize = 12;

    /// Default alpha parameter for Tukey window.
    pub const DEFAULT_TUKEY_ALPHA: f32 = 0.1;
}

/// Constants related to partitioned rice coding (PRC).
pub mod rice {
    /// Maximum allowed value for the Rice parameters.
    ///
    /// 5-bit rice coding is not supported currently, and 0b1111 is reserved for
    /// verbatim encoding. So, 14 will be the maximum.
    pub const MAX_RICE_PARAMETER: usize = 14;

    /// Maximum order of Rice parameter partitioning.
    pub const MAX_PARTITION_ORDER: usize = 15;

    /// Maximum number of Rice partitions.
    pub const MAX_PARTITIONS: usize = 1usize << MAX_PARTITION_ORDER;

    /// Minimum rice partition size. (1 in the specification)
    pub const MIN_PARTITION_SIZE: usize = 64;
}

/// Module for internal error messages.
///
/// Use `panic!` and those messages only for env-related unrecoverable errors.
/// It's okay to use them in tests, but it's not okay to add another variable
/// only for test functions.
pub(crate) mod panic_msg {
    pub const ARC_DESTRUCT_FAILED: &str = "INTERNAL ERROR: Arc destruction failed.";
    pub const DATA_INCONSISTENT: &str = "INTERNAL ERROR: Internal variable inconsistency detected.";
    pub const ERROR_NOT_EXPECTED: &str =
        "INTERNAL ERROR: Error occured in the function where it is not expected.";
    pub const FRAMENUM_NOT_SET: &str =
        "INTERNAL ERROR: Frame buffer is not properly initialized. (FrameNo. not set).";
    pub const MPMC_SEND_FAILED: &str =
        "INTERNAL ERROR: Critical error occured in multi-thread communication channel.";
    pub const MPMC_RECV_FAILED: &str =
        "INTERNAL ERROR: Critical error occured in multi-thread communication channel.";
    pub const MUTEX_LOCK_FAILED: &str = "INTERNAL ERROR: Couldn't get lock for mutex.";
    pub const MUTEX_DROP_FAILED: &str = "INTERNAL ERROR: Couldn't discard mutex.";
    pub const NO_ERROR_EXPECTED: &str =
        "INTERNAL ERROR: Error emitted from the function designed not to return err.";
    pub const THREAD_JOIN_FAILED: &str = "INTERNAL ERROR: Failed to wait thread termination.";
}
