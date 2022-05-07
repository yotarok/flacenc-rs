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

/// Configuration for encoder.
#[derive(Serialize, Deserialize, Debug)]
pub struct Encoder {
    pub stereo_coding: StereoCoding,
    pub subframe_coding: SubFrameCoding,
}

#[allow(clippy::derivable_impls)]
impl Default for Encoder {
    fn default() -> Self {
        Self {
            stereo_coding: StereoCoding::default(),
            subframe_coding: SubFrameCoding::default(),
        }
    }
}

/// Configuration for stereo coding algorithms.
#[derive(Serialize, Deserialize, Debug)]
pub struct StereoCoding {
    pub allow_leftside: bool,
    pub allow_rightside: bool,
    pub allow_midside: bool,
}

impl Default for StereoCoding {
    fn default() -> Self {
        Self {
            allow_leftside: true,
            allow_rightside: true,
            allow_midside: true,
        }
    }
}

/// Configuration for sub-frame (individual channel) coding.
#[derive(Serialize, Deserialize, Debug)]
pub struct SubFrameCoding {
    // Disabling verbatim coding is intentionally prohibited.
    pub allow_constant: bool,
    pub allow_fixed: bool,
    pub allow_lpc: bool,
    // This struct will be enhanced like follows:
    // rice_coding: RiceCoding,
    // lpc: Lpc,
    // fixed_lpc: FixedLpc,
}

impl Default for SubFrameCoding {
    fn default() -> Self {
        Self {
            allow_constant: true,
            allow_fixed: true,
            allow_lpc: true,
        }
    }
}
