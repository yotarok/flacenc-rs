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

#![feature(portable_simd)]
// Note that clippy attributes should be in sync with those declared in "main.rs"
#![warn(clippy::all, clippy::nursery, clippy::pedantic, clippy::cargo)]
// Some of clippy::pedantic rules are actually useful, so use it with a lot of
// ad-hoc exceptions.
#![allow(
    // Reactivate "use_self" once false-positive issue is gone.
    // https://github.com/rust-lang/rust-clippy/issues/6902
    clippy::use_self,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::multiple_crate_versions,
    clippy::must_use_candidate
)]
// Some from restriction lint-group
#![warn(
    clippy::clone_on_ref_ptr,
    clippy::create_dir,
    clippy::dbg_macro,
    clippy::empty_structs_with_brackets,
    clippy::exit,
    clippy::let_underscore_must_use,
    clippy::lossy_float_literal,
    clippy::print_stdout,
    clippy::rc_buffer,
    clippy::rc_mutex,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::separated_literal_suffix,
    clippy::str_to_string,
    clippy::string_add,
    clippy::string_to_string,
    clippy::try_err,
    clippy::unnecessary_self_imports,
    clippy::wildcard_enum_match_arm
)]

pub mod bitsink;
pub mod coding;
pub mod component;
pub mod config;
pub mod constant;
pub mod error;
pub mod lpc;
pub mod rice;
pub mod source;

#[cfg(test)]
mod test_helper;

#[cfg(test)]
mod test {
    // end-to-end, but transparent test.
    use super::*;
    use rstest::rstest;

    const FIXED_BLOCK_CONFIGS: [&str; 4] = [
        "",
        r#"
block_sizes = [512]
        "#,
        r#"
block_sizes = [123]
        "#,
        r#"
block_sizes = [1024]
[subframe_coding.qlpc]
use_direct_mse = true
mae_optimization_steps = 2
        "#,
    ];

    #[rstest]
    fn e2e_with_generated_sinusoids(
        #[values(1, 2, 3, 5, 8)] channels: usize,
        #[values(FIXED_BLOCK_CONFIGS[0],
                 FIXED_BLOCK_CONFIGS[1],
                 FIXED_BLOCK_CONFIGS[2],
                 FIXED_BLOCK_CONFIGS[3])]
        config: &str,
    ) {
        let signal_len = 16123;
        let bits_per_sample = 16;
        let sample_rate = 16000;

        let mut channel_signals = vec![];
        for _ch in 0..channels {
            channel_signals.push(test_helper::sinusoid_plus_noise(
                signal_len, 36, 10000.0, 123,
            ));
        }

        let mut signal = vec![];
        for t in 0..signal_len {
            for s in &channel_signals {
                signal.push(s[t]);
            }
        }
        let config = toml::from_str(config).expect("config parsing error");
        let source =
            source::PreloadedSignal::from_samples(&signal, channels, bits_per_sample, sample_rate);

        test_helper::integrity_test(
            |s| {
                coding::encode_with_fixed_block_size(&config, s, config.block_sizes[0])
                    .expect("source error")
            },
            &source,
        );
    }
}
