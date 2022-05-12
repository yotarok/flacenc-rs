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


const fn default_fixed_block_size() -> Option<usize> {
    Some(4096)
}

/// Configuration for encoder.
#[derive(Serialize, Deserialize, Debug)]
pub struct Encoder {
    #[serde(default="default_fixed_block_size")]
    pub fixed_block_size: Option<usize>,
    #[serde(default)]
    pub stereo_coding: StereoCoding,
    #[serde(default)]
    pub subframe_coding: SubFrameCoding,
}

#[allow(clippy::derivable_impls)]
impl Default for Encoder {
    fn default() -> Self {
        Self {
            stereo_coding: StereoCoding::default(),
            subframe_coding: SubFrameCoding::default(),
            fixed_block_size: default_fixed_block_size(),
        }
    }
}

const fn default_use_leftside() -> bool {
    true
}
const fn default_use_rightside() -> bool {
    true
}
const fn default_use_midside() -> bool {
    true
}

/// Configuration for stereo coding algorithms.
#[derive(Serialize, Deserialize, Debug)]
pub struct StereoCoding {
    #[serde(default = "default_use_leftside")]
    pub use_leftside: bool,
    #[serde(default = "default_use_rightside")]
    pub use_rightside: bool,
    #[serde(default = "default_use_midside")]
    pub use_midside: bool,
}

impl Default for StereoCoding {
    fn default() -> Self {
        Self {
            use_leftside: default_use_leftside(),
            use_rightside: default_use_rightside(),
            use_midside: default_use_midside(),
        }
    }
}

const fn default_use_constant() -> bool {
    true
}
const fn default_use_fixed() -> bool {
    true
}
const fn default_use_lpc() -> bool {
    true
}

/// Configuration for sub-frame (individual channel) coding.
#[derive(Serialize, Deserialize, Debug)]
pub struct SubFrameCoding {
    // Disabling verbatim coding is intentionally prohibited.
    #[serde(default)]
    pub use_constant: bool,
    #[serde(default)]
    pub use_fixed: bool,
    #[serde(default)]
    pub use_lpc: bool,
    #[serde(default)]
    pub qlpc: Qlpc,
    #[serde(default)]
    pub prc: Prc,
}

impl Default for SubFrameCoding {
    fn default() -> Self {
        Self {
            use_constant: default_use_constant(),
            use_fixed: default_use_fixed(),
            use_lpc: default_use_lpc(),
            qlpc: Qlpc::default(),
            prc: Prc::default(),
        }
    }
}

const fn default_max_parameter() -> usize {
    MAX_RICE_PARAMETER
}

/// Configuration for partitioned-rice coding (PRC).
#[derive(Serialize, Deserialize, Debug)]
pub struct Prc {
    #[serde(default = "default_max_parameter")]
    pub max_parameter: usize,
}

impl Default for Prc {
    fn default() -> Self {
        Self {
            max_parameter: default_max_parameter(),
        }
    }
}

const fn default_lpc_order() -> usize {
    QLPC_DEFAULT_ORDER
}
const fn default_quant_precision() -> usize {
    QLPC_DEFAULT_PRECISION
}

/// Configuration for quantized linear-predictive coding (QLPC).
#[derive(Serialize, Deserialize, Debug)]
pub struct Qlpc {
    #[serde(default = "default_lpc_order")]
    pub lpc_order: usize,
    #[serde(default = "default_quant_precision")]
    pub quant_precision: usize,
}

impl Default for Qlpc {
    fn default() -> Self {
        Self {
            lpc_order: default_lpc_order(),
            quant_precision: default_quant_precision(),
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

        // Check the rest is default.
        assert_eq!(
            config.subframe_coding.qlpc.quant_precision,
            QLPC_DEFAULT_PRECISION
        );
        assert_eq!(
            config.fixed_block_size,
            default_fixed_block_size()
        );
    }
}
