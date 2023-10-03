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

use serde::Deserialize;
use serde::Serialize;

use super::constant::MAX_BLOCKSIZE_SUPPORTED;
use super::constant::MAX_LPC_ORDER;
use super::constant::MAX_RICE_PARAMETER;
use super::constant::MIN_BLOCKSIZE_SUPPORTED;
use super::constant::QLPC_DEFAULT_ORDER;
use super::constant::QLPC_DEFAULT_PRECISION;
use super::constant::QLPC_MAX_PRECISION;
use super::error::Verify;
use super::error::VerifyError;
use super::lpc::Window;

/// Configuration for encoder.
#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Encoder {
    /// The possible block sizes encoder can use.
    pub block_sizes: Vec<usize>,
    /// Whether encoder runs on multi-thread mode.
    pub multithread: bool,
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
            block_sizes: vec![4096usize],
            multithread: cfg!(feature = "par"),
        }
    }
}

impl Verify for Encoder {
    fn verify(&self) -> Result<(), VerifyError> {
        if self.block_sizes.is_empty() {
            return Err(VerifyError::new(
                "block_sizes",
                "Must specify at least one block size.",
            ));
        }
        if self.block_sizes.len() > 1 {
            return Err(VerifyError::new(
                "block_sizes",
                "Multiple blocksize mode is not supported currently.",
            ));
        }
        for (i, &bs) in self.block_sizes.iter().enumerate() {
            if bs > MAX_BLOCKSIZE_SUPPORTED {
                return Err(VerifyError::new(
                    &format!("block_sizes[{i}]"),
                    &format!("Must be less than {MAX_BLOCKSIZE_SUPPORTED}"),
                ));
            } else if bs < MIN_BLOCKSIZE_SUPPORTED {
                return Err(VerifyError::new(
                    &format!("block_sizes[{i}]"),
                    &format!("Must be more than {MIN_BLOCKSIZE_SUPPORTED}"),
                ));
            }
        }

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
#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct StereoCoding {
    /// If set to false, left-side coding will not be used.
    pub use_leftside: bool,
    /// If set to false, right-side coding will not be used.
    pub use_rightside: bool,
    /// If set to false, mid-side coding will not be used.
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
#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct SubFrameCoding {
    // Disabling verbatim coding is intentionally prohibited.
    /// If set to false, constant mode will not be used.
    pub use_constant: bool,
    /// If set to false, fixed-LPC mode will not be used.
    pub use_fixed: bool,
    /// If set to false, LPC mode will not be used.
    pub use_lpc: bool,
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
#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
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
        if self.max_parameter > MAX_RICE_PARAMETER {
            return Err(VerifyError::new(
                "max_parameter",
                &format!("Must not exceed {MAX_RICE_PARAMETER}"),
            ));
        }
        Ok(())
    }
}

/// Configuration for quantized linear-predictive coding (QLPC).
#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Qlpc {
    /// LPC order.
    pub lpc_order: usize,
    /// Precision for quantized LPC coefficients.
    pub quant_precision: usize,
    /// If set, use a direct MSE method for LPC estimation.
    pub use_direct_mse: bool,
    /// If set, iteratively optimizes LPC parameters with the given steps.
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
        if self.lpc_order > MAX_LPC_ORDER {
            return Err(VerifyError::new(
                "lpc_order",
                &format!("Must not exceed {MAX_LPC_ORDER}"),
            ));
        }
        if self.quant_precision > QLPC_MAX_PRECISION {
            return Err(VerifyError::new(
                "quant_precision",
                &format!("Must not exceed {QLPC_MAX_PRECISION}"),
            ));
        }
        if self.quant_precision == 0 {
            return Err(VerifyError::new("quant_precision", "Must not be zero"));
        }
        if cfg!(not(feature = "experimental")) && self.use_direct_mse {
            return Err(VerifyError::new(
                "use_direct_mse",
                "Can only be used when \"experimental\" feature enabled",
            ));
        }
        if cfg!(not(feature = "experimental")) && self.mae_optimization_steps > 0 {
            return Err(VerifyError::new(
                "mae_optimization_steps",
                "Can only be used when \"experimental\" feature enabled",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::error::Verify;
    use super::*;

    #[test]
    fn serialization() -> Result<(), toml::ser::Error> {
        let config = Encoder::default();
        toml::to_string(&config)?;
        Ok(())
    }

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
        assert_eq!(config.block_sizes, &[4096]);
        assert!(config.subframe_coding.use_lpc);
    }

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
