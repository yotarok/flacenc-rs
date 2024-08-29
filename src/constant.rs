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

//! Configuration constants
//!
//! All boundary (prefixed with `MIN_` or `MAX_`) values defined in this module
//! are inclusive, e.g. in a case of `{MIN|MAX}_BLOCK_SIZE`, a block size `b` is
//! valid when `MIN_BLOCK_SIZE <= b` and `b <= MAX_BLOCK_SIZE`.

#![allow(dead_code)] // it's okay if some FLAC-spec constants are not used.

// Constance sorted in an alphabetical-order.  Top-level constants first, and
// then sub-modules.

mod built {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

/// Default block size.
pub const DEFAULT_BLOCK_SIZE: usize = 4096;

/// The number of partitions used in entropy estimation algorithm.
pub(crate) const DEFAULT_ENTROPY_ESTIMATOR_PARTITIONS: usize = 16;

/// Minimum bits-per-sample supported. (4 in the specification.)
pub const MIN_BITS_PER_SAMPLE: usize = 8;

/// Minimum length of a block supported.
///
/// Note that a FLAC stream may have a block that is shorter than this value at the end of the
/// stream, and decoders must be able to decode it. This value is for restricting the block sizes
/// of the frames except for the last one, and only used in the config-file verification.
pub const MIN_BLOCK_SIZE: usize = 32;

/// Minimum length of a block supported in prediction modules.
///
/// Frames shorter than this constant will be encoded in [`component::Constant`] or
/// [`component::Verbatim`].
pub(crate) const MIN_BLOCK_SIZE_FOR_PREDICTION: usize = 64;

/// Maximum bits-per-sample supported. (32 in the specification.)
pub const MAX_BITS_PER_SAMPLE: usize = 24;

/// Maximum length of a block supported (65535 in the specification.)
pub const MAX_BLOCK_SIZE: usize = 32767;

/// Maximum number of channels.
pub const MAX_CHANNELS: usize = 8;

/// The max number of partitions used in entropy estimation algorithm.
pub(crate) const MAX_ENTROPY_ESTIMATOR_PARTITIONS: usize = 64;

/// Constants related to build-time information.
///
/// Information in this module is gathered using [`built`] crate.
///
/// [`built`]: https://crates.io/crates/built
pub mod build_info {
    use super::built;

    /// Build profile. "debug" or "release".
    pub const BUILD_PROFILE: &str = built::PROFILE;

    /// Version of `flacenc` library from Cargo.toml.
    pub const CRATE_VERSION: &str = built::PKG_VERSION;

    /// Comma-separated strings of features activated.
    pub const FEATURES: &str = built::FEATURES_LOWERCASE_STR;

    /// `rustc` version used for building this crate.
    pub const RUSTC_VERSION: &str = built::RUSTC_VERSION;
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
    /// Default LPC order for QLPC module.
    pub const DEFAULT_ORDER: usize = 10;

    /// Default precision for storing QLPC coefficients.
    pub const DEFAULT_PRECISION: usize = 15;

    /// Default alpha parameter for Tukey window.
    pub const DEFAULT_TUKEY_ALPHA: f32 = 0.4;

    /// Maximum order of LPC supported. (32 in the specification.)
    pub const MAX_ORDER: usize = 24;

    /// Max number of bits (precision) for storing QLPC coefficients.
    pub const MAX_PRECISION: usize = 15;

    /// Maximum shift parameter of QLPC defined in the specification.
    pub const MAX_SHIFT: i8 = (1i8 << (SHIFT_BITS - 1)) - 1;

    /// Minimum shift parameter of QLPC.
    ///
    /// According to the bitstream specification, it can be -16
    /// (`-(1 << QLPC_SHIFT_BITS)`), but the reference decoder doesn't support
    /// negative shift case.
    pub const MIN_SHIFT: i8 = 0;

    /// The number of bits used for encoding shift bits of QLPC.
    pub const SHIFT_BITS: usize = 5;
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
