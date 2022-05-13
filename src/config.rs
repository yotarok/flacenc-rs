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

use super::constant::MAX_RICE_PARAMETER;
use super::constant::QLPC_DEFAULT_ORDER;
use super::constant::QLPC_DEFAULT_PRECISION;
use super::lpc::Window;

/// Configuration for encoder.
#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Encoder {
    /// If specified, the encoder operates on the fixed block size.
    ///
    /// Currently. variable block encoding is not supported. Therefore, this
    /// must be always set (or keep default).
    pub fixed_block_size: Option<usize>,
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
            fixed_block_size: Some(4096),
        }
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
    /// Window function to be used for LPC estimation.
    pub window: Window,
}

impl Default for Qlpc {
    fn default() -> Self {
        Self {
            lpc_order: QLPC_DEFAULT_ORDER,
            quant_precision: QLPC_DEFAULT_PRECISION,
            use_direct_mse: false,
            window: Window::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialization() {
        let config = Encoder::default();
        assert!(toml::to_string(&config).is_ok());
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
        assert_eq!(config.fixed_block_size, Some(4096));
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
}
