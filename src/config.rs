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

//! Encoder configuration structs.
//!
//! This module defines a deserializable config variables and those default
//! values. Typically, the top-level variable [`Encoder`] is represented in
//! a toml file like below:
//!
//! ```toml
//! block_size = 4096
//! multithread = true
//!
//! [stereo_coding]
//! use_leftside = true
//! use_rightside = true
//! use_midside = true
//!
//! [subframe_coding]
//! use_constant = true
//! use_fixed = true
//! use_lpc = true
//!
//! [subframe_coding.fixed]
//! max_order = 4
//!
//! [subframe_coding.fixed.order_sel]
//! type = "ApproxEnt"
//! partitions = 32
//!
//! [subframe_coding.qlpc]
//! lpc_order = 10
//! quant_precision = 15
//! use_direct_mse = false
//! mae_optimization_steps = 0
//!
//! [subframe_coding.qlpc.window]
//! type = "Tukey"
//! alpha = 0.4
//!
//! [subframe_coding.prc]
//! max_parameter = 14
//! ```

use std::num::NonZeroUsize;

#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use super::constant;
use super::constant::qlpc::DEFAULT_ORDER as QLPC_DEFAULT_ORDER;
use super::constant::qlpc::DEFAULT_PRECISION as QLPC_DEFAULT_PRECISION;
use super::constant::qlpc::DEFAULT_TUKEY_ALPHA;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::qlpc::MAX_PRECISION as QLPC_MAX_PRECISION;
use super::constant::rice::MAX_RICE_PARAMETER;
use super::constant::DEFAULT_ENTROPY_ESTIMATOR_PARTITIONS;
use super::constant::MAX_BLOCK_SIZE;
use super::constant::MAX_ENTROPY_ESTIMATOR_PARTITIONS;
use super::constant::MIN_BLOCK_SIZE;
use super::error::verify_range;
use super::error::verify_true;
use super::error::Verify;
use super::error::VerifyError;

/// Configuration for encoder.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Encoder {
    /// Encoder block size. (default: [`constant::DEFAULT_BLOCK_SIZE`])
    pub block_size: usize,
    /// Whether encoder runs on multi-thread mode. (default: `true` when `"par"` feature is used)
    pub multithread: bool,
    /// The number of threads used in multithread mode. (default: `None`)
    ///
    /// If None, the number of workers is set to be identical with the number
    /// of the logical CPU cores in the running environment.
    pub workers: Option<NonZeroUsize>,
    /// Configuration for stereo-coding module.
    pub stereo_coding: StereoCoding,
    /// Configuration for individual channels.
    pub subframe_coding: SubFrameCoding,
}

#[allow(clippy::derivable_impls)]
impl Default for Encoder {
    fn default() -> Self {
        Self {
            stereo_coding: StereoCoding::default(),
            subframe_coding: SubFrameCoding::default(),
            block_size: constant::DEFAULT_BLOCK_SIZE,
            multithread: cfg!(feature = "par"),
            workers: None,
        }
    }
}

impl Verify for Encoder {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_range!(
            "block_size",
            self.block_size,
            MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE
        )?;

        self.stereo_coding
            .verify()
            .map_err(|err| err.within("stereo_coding"))?;
        self.subframe_coding
            .verify()
            .map_err(|err| err.within("subframe_coding"))?;
        Ok(())
    }
}

/// Configuration for stereo coding algorithms.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct StereoCoding {
    /// If set to false, left-side coding will not be used. (default: `true`)
    pub use_leftside: bool,
    /// If set to false, right-side coding will not be used. (default: `true`)
    pub use_rightside: bool,
    /// If set to false, mid-side coding will not be used. (default: `true`)
    pub use_midside: bool,
}

impl Default for StereoCoding {
    fn default() -> Self {
        Self {
            use_leftside: true,
            use_rightside: true,
            use_midside: true,
        }
    }
}

impl Verify for StereoCoding {
    fn verify(&self) -> Result<(), VerifyError> {
        Ok(())
    }
}

/// Configuration for sub-frame (individual channel) coding.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct SubFrameCoding {
    // NOTE: Disabling verbatim coding is intentionally prohibited as we want
    //  to ensure that there's at least one possible FLAC representation for
    //  every possible input signal.
    /// If set to false, constant mode will not be used. (default: `true`)
    pub use_constant: bool,
    /// If set to false, fixed-LPC mode will not be used. (default: `true`)
    pub use_fixed: bool,
    /// If set to false, LPC mode will not be used. (default: `true`)
    pub use_lpc: bool,
    /// Configuration for fixed LPC encoder.
    pub fixed: Fixed,
    /// Configuration for quantized LPC encoder.
    pub qlpc: Qlpc,
    /// Configuration for partitioned Rice coding.
    pub prc: Prc,
}

impl Default for SubFrameCoding {
    fn default() -> Self {
        Self {
            use_constant: true,
            use_fixed: true,
            use_lpc: true,
            fixed: Fixed::default(),
            qlpc: Qlpc::default(),
            prc: Prc::default(),
        }
    }
}

impl Verify for SubFrameCoding {
    fn verify(&self) -> Result<(), VerifyError> {
        self.qlpc.verify().map_err(|err| err.within("qlpc"))?;
        self.prc.verify().map_err(|err| err.within("prc"))?;
        Ok(())
    }
}

/// Configuration for partitioned-rice coding (PRC).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Prc {
    /// Max value for the parameter of rice coding.
    pub max_parameter: usize,
}

impl Default for Prc {
    fn default() -> Self {
        Self {
            max_parameter: MAX_RICE_PARAMETER,
        }
    }
}

impl Verify for Prc {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_range!("max_parameter", self.max_parameter, ..=MAX_RICE_PARAMETER)?;
        Ok(())
    }
}

/// Configuration for fixed-parameter linear-predictive coding.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Fixed {
    /// Maximum LPC order. (default: [`constant::fixed::MAX_LPC_ORDER`])
    ///
    /// This value must be less than or equal to [`constant::fixed::MAX_LPC_ORDER`]
    pub max_order: usize,

    /// Configuration for the algorithm for selecting order.
    pub order_sel: OrderSel,
}

impl Verify for Fixed {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_range!(
            "max_order",
            self.max_order,
            ..=(constant::fixed::MAX_LPC_ORDER)
        )?;
        Ok(())
    }
}

impl Default for Fixed {
    fn default() -> Self {
        Self {
            max_order: constant::fixed::MAX_LPC_ORDER,
            order_sel: OrderSel::default(),
        }
    }
}

/// Configuration for quantized linear-predictive coding (QLPC).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Qlpc {
    /// LPC order. (default: [`QLPC_DEFAULT_ORDER`])
    pub lpc_order: usize,
    /// Precision for quantized LPC coefficients. (default: [`QLPC_DEFAULT_PRECISION`])
    pub quant_precision: usize,
    /// If set, use a direct MSE method for LPC estimation. (default: `false`)
    ///
    /// This is for an encoder with `experimental` feature. In
    /// non-`experimental` encoders, this setting is simply ignored.
    pub use_direct_mse: bool,
    /// If set, iteratively optimizes LPC parameters with the given steps.
    ///
    /// This is for an encoder with `experimental` feature. In
    /// non-`experimental` encoders, this setting is simply ignored.
    pub mae_optimization_steps: usize,
    /// Window function to be used for LPC estimation.
    pub window: Window,
}

impl Default for Qlpc {
    fn default() -> Self {
        Self {
            lpc_order: QLPC_DEFAULT_ORDER,
            quant_precision: QLPC_DEFAULT_PRECISION,
            use_direct_mse: false,
            mae_optimization_steps: 0,
            window: Window::default(),
        }
    }
}

impl Verify for Qlpc {
    fn verify(&self) -> Result<(), VerifyError> {
        verify_range!("lpc_order", self.lpc_order, 1..=MAX_LPC_ORDER)?;
        verify_range!(
            "quant_precision",
            self.quant_precision,
            1..=QLPC_MAX_PRECISION
        )?;
        if cfg!(not(feature = "experimental")) {
            verify_true!(
                "use_direct_mse",
                !self.use_direct_mse,
                "this feature is only available in `experimental` build."
            )?;
            verify_true!(
                "mae_optimization_steps",
                self.mae_optimization_steps == 0,
                "this feature is only available in `experimental` build."
            )?;
        }

        self.window.verify().map_err(|err| err.within("window"))?;
        Ok(())
    }
}

/// Analysis window descriptor.
///
/// This enum can be deserialized from toml table with a special tag "type"
/// specifying the type of window, for example like below:
///
/// ```toml
/// type = "Tukey"
/// alpha = 0.4
/// ```
///
/// The current default value for [`Window`] is "Tukey" with alpha =
/// [`DEFAULT_TUKEY_ALPHA`].
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
#[non_exhaustive]
pub enum Window {
    /// [Rectangular] window.
    ///
    /// [Rectangular]: https://en.wikipedia.org/wiki/Window_function#Rectangular_window
    Rectangle,
    /// [Tukey] window.
    ///
    /// `alpha` parameter must satisfy both `0.0 <= alpha` and `alpha <= 1.0`.
    /// `alpha == 0.0` is equivalent with using a rectangular window.
    ///
    /// [Tukey]: https://en.wikipedia.org/wiki/Window_function#Tukey_window
    Tukey {
        /// `alpha` parameter of Tukey window.
        alpha: f32,
    },
}

impl Eq for Window {}

impl Default for Window {
    fn default() -> Self {
        Self::Tukey {
            alpha: DEFAULT_TUKEY_ALPHA,
        }
    }
}

impl Verify for Window {
    fn verify(&self) -> Result<(), VerifyError> {
        match *self {
            Self::Rectangle => Ok(()),
            Self::Tukey { alpha } => {
                if (0.0..=1.0).contains(&alpha) {
                    Ok(())
                } else {
                    Err(VerifyError::new(
                        "tukey.alpha",
                        "alpha must be in range between 0 and 1",
                    ))
                }
            }
        }
    }
}

/// Helper fn for serde.
#[cfg(feature = "serde")]
const fn default_partition_count() -> usize {
    DEFAULT_ENTROPY_ESTIMATOR_PARTITIONS
}

/// Configuration for (LPC) order selection algorithms.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum OrderSel {
    /// Performs actual encoding and count bits.
    BitCount,
    /// Estimates the number of bits using partitioned entropy estimation.
    ApproxEnt {
        /// The number of partitions used for estimation.
        #[cfg_attr(feature = "serde", serde(default = "default_partition_count"))]
        partitions: usize,
    },
}

impl Default for OrderSel {
    fn default() -> Self {
        Self::ApproxEnt {
            partitions: DEFAULT_ENTROPY_ESTIMATOR_PARTITIONS,
        }
    }
}

impl Verify for OrderSel {
    fn verify(&self) -> Result<(), VerifyError> {
        match *self {
            Self::BitCount => Ok(()),
            Self::ApproxEnt { partitions } => {
                verify_range!(
                    "ApproxEnt.partitions",
                    partitions,
                    1..=MAX_ENTROPY_ESTIMATOR_PARTITIONS
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_for_encoder() {
        {
            let config = Encoder::default();
            config.verify().unwrap();
        }
        {
            let config = Encoder {
                block_size: 1234,
                ..Default::default()
            };
            config.verify().unwrap();
        }
        {
            let config = Encoder {
                block_size: 1,
                ..Default::default()
            };
            config.verify().unwrap_err();
        }
        {
            let config = Encoder {
                block_size: 123_456,
                ..Default::default()
            };
            config.verify().unwrap_err();
        }
    }

    #[test]
    fn verification_for_stereo_coding() {
        let config = StereoCoding::default();
        config.verify().unwrap();
    }

    #[test]
    fn verification_for_subframe_coding() {
        {
            let config = SubFrameCoding::default();
            config.verify().unwrap();
        }
        {
            // test error propagation.
            let mut config = SubFrameCoding::default();
            config.prc.max_parameter = 1234;
            config.verify().unwrap_err();
        }
    }

    #[test]
    fn verification_for_prc() {
        {
            let config = Prc::default();
            config.verify().unwrap();
        }
        {
            let config = Prc {
                max_parameter: 18,
                ..Default::default()
            };
            config.verify().unwrap_err();
        }
    }

    #[test]
    fn verification_for_qlpc() {
        {
            let config = Qlpc::default();
            config.verify().unwrap();
        }
        {
            let config = Qlpc {
                lpc_order: 39,
                ..Default::default()
            };
            config.verify().unwrap_err();
        }
        {
            let config = Qlpc {
                quant_precision: 256,
                ..Default::default()
            };
            config.verify().unwrap_err();
        }
        {
            let config = Qlpc {
                use_direct_mse: true,
                ..Default::default()
            };
            if cfg!(feature = "experimental") {
                config.verify().unwrap();
            } else {
                config.verify().unwrap_err();
            }
        }
        {
            let config = Qlpc {
                mae_optimization_steps: 20,
                ..Default::default()
            };
            if cfg!(feature = "experimental") {
                config.verify().unwrap();
            } else {
                config.verify().unwrap_err();
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization() -> Result<(), toml::ser::Error> {
        let config = Encoder::default();
        toml::to_string(&config)?;
        Ok(())
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization() {
        let src = "
[subframe_coding.qlpc]
lpc_order = 7
";
        let config: Encoder = toml::from_str(src).expect("Parse error.");
        assert_eq!(config.subframe_coding.qlpc.lpc_order, 7);
        assert_eq!(
            config.subframe_coding.qlpc.quant_precision,
            QLPC_DEFAULT_PRECISION
        );

        // Check the rest is default.
        assert_eq!(
            config.subframe_coding.qlpc.quant_precision,
            QLPC_DEFAULT_PRECISION
        );
        assert_eq!(config.block_size, constant::DEFAULT_BLOCK_SIZE);
        assert!(config.subframe_coding.use_lpc);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn if_empty_source_yields_default_config() {
        let empty_src = "";
        let config: Encoder = toml::from_str(empty_src).expect("Parse error.");
        let default_config: Encoder = Encoder::default();
        eprintln!(
            "## Current default config\n\n{}",
            toml::to_string(&config).unwrap()
        );
        assert_eq!(toml::to_string(&config), toml::to_string(&default_config));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_and_verify() {
        let src = "
[subframe_coding.qlpc]
lpc_order = 256
";
        let config: Encoder = toml::from_str(src).expect("Parse error.");
        let verify_result = config.verify();
        assert!(verify_result.is_err());
        eprintln!("{}", verify_result.err().unwrap());
    }
}
