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

#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "simd-nightly", feature(portable_simd))]
#![cfg_attr(feature = "simd-nightly", feature(test))]
// Note that clippy attributes should be in sync with those declared in "main.rs"
#![warn(clippy::all, clippy::nursery, clippy::pedantic, clippy::cargo)]
// Some of clippy::pedantic rules are actually useful, so use it with a lot of
// ad-hoc exceptions.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    clippy::multiple_crate_versions,
    clippy::must_use_candidate,
    clippy::wildcard_dependencies
)]
// Some from restriction lint-group
#![warn(
    clippy::clone_on_ref_ptr,
    clippy::create_dir,
    clippy::dbg_macro,
    clippy::empty_structs_with_brackets,
    clippy::exit,
    clippy::if_then_some_else_none,
    clippy::impl_trait_in_params,
    clippy::let_underscore_must_use,
    clippy::lossy_float_literal,
    clippy::multiple_inherent_impl,
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

/// Expands import statements for `fakesimd` or `std::simd`.
macro_rules! import_simd {
    (as $modalias:ident) => {
        #[cfg(feature = "simd-nightly")]
        use std::simd as $modalias;
        #[cfg(not(feature = "simd-nightly"))]
        use $crate::fakesimd as $modalias;

        #[allow(unused_imports)]
        use simd::prelude::*;

        #[allow(unused_imports)]
        use simd::StdFloat;
    };
}

/// Sets up the thread-local re-usable storage for avoiding reallocation.
///
/// This provides a short-cut for the common pattern using [`thread_local!`]
/// and [`RefCell`].  Currently, this is just for removing small repetition in
/// code.
///
/// NOTE: This is tentatively brought to the global space for avoiding
/// unintended exposure in the document. Probably, this should be moved to an
/// appropriate sub-modules after we learn how to control visibility of macros
/// defined in a sub-module.
///
/// [`RefCell`]: std::cell::RefCell
macro_rules! reusable {
    ($key:ident: $t:ty) => {
        thread_local! {
            static $key: std::cell::RefCell<$t> = std::cell::RefCell::new(Default::default());
        }
    };
    ($key:ident: $t:ty = $init:expr) => {
        thread_local! {
            static $key: std::cell::RefCell<$t> = std::cell::RefCell::new($init);
        }
    };
}

/// Macro used when using a storage declared using [`reusable!`].
///
/// NOTE: This is tentatively brought to the global space for avoiding
/// unintended exposure in the document. Probably, this should be moved to an
/// appropriate sub-modules after we learn how to control visibility of macros
/// defined in a sub-module.
macro_rules! reuse {
    ($key:ident, $fn:expr) => {{
        #[allow(clippy::redundant_closure_call)]
        $key.with(|cell| $fn(&mut cell.borrow_mut()))
    }};
}

// internal log macros
#[allow(unused)]
macro_rules! info { (target: $t:literal, $($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::info!(target: $t, $($x)*)
    }
) }

pub(crate) mod arrayutils;
pub mod bitsink;
pub(crate) mod coding;
pub mod component;
pub mod config;
pub mod constant;
pub mod error;
#[cfg(not(feature = "simd-nightly"))]
pub(crate) mod fakesimd;
pub(crate) mod lpc;
#[cfg(feature = "par")]
pub(crate) mod par;
pub(crate) mod repeat;
pub(crate) mod rice;
#[cfg(any(test, feature = "__export_sigen"))]
pub mod sigen;
pub mod source;

#[cfg(test)]
pub mod test_helper;

// this is for including "doctest_helper.rs" in lintting and auto-formating.
#[cfg(clippy)]
mod doctest_helper;

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// import global entry points
pub use coding::encode_fixed_size_frame;

pub use coding::encode_with_fixed_block_size;

#[cfg(test)]
mod tests {
    // end-to-end, but transparent test.
    #[cfg(feature = "serde")]
    use super::error::Verify;
    #[cfg(feature = "serde")]
    use super::*;
    #[cfg(feature = "serde")]
    use crate::sigen::Signal;
    #[cfg(feature = "serde")]
    use rstest::rstest;

    #[allow(dead_code)] // only used when `cfg(feature = "serde")`.
    const FIXED_BLOCK_CONFIGS: [&str; 5] = [
        "",
        r"
block_sizes = [512]
        ",
        r"
block_sizes = [123]
        ",
        r"
block_sizes = [1024]
[subframe_coding.qlpc]
use_direct_mse = true
mae_optimization_steps = 2
        ",
        r"
multithread = false
        ",
    ];

    /// E2E testing involving integrity check with `claxon` decoder.
    ///
    /// The test signals used in this test are just sinusoids mixed with small white noise. All
    /// possible combinations of the selected numbers of channels, sampling rates, and config
    /// strings are evaluated.
    #[cfg(feature = "serde")]
    #[rstest]
    fn e2e_with_generated_sinusoids(
        #[values(1, 2, 3, 5, 8)] channels: usize,
        #[values(16000, 16001, 95800)] sample_rate: usize,
        #[values(FIXED_BLOCK_CONFIGS[0],
                 FIXED_BLOCK_CONFIGS[1],
                 FIXED_BLOCK_CONFIGS[2],
                 FIXED_BLOCK_CONFIGS[3],
                 FIXED_BLOCK_CONFIGS[4])]
        config: &str,
    ) {
        let signal_len = 16123;
        let bits_per_sample = 16;

        let mut channel_signals = vec![];
        for _ch in 0..channels {
            channel_signals.push(
                sigen::Sine::new(36, 0.4)
                    .noise(0.04)
                    .to_vec_quantized(bits_per_sample, signal_len),
            );
        }

        let mut signal = vec![];
        for t in 0..signal_len {
            for s in &channel_signals {
                signal.push(s[t]);
            }
        }
        let mut config: config::Encoder = toml::from_str(config).expect("config parsing error");

        if !cfg!(feature = "experimental") {
            // disable experimental features
            config.subframe_coding.qlpc.use_direct_mse = false;
            config.subframe_coding.qlpc.mae_optimization_steps = 0;
        }

        let config = config.into_verified().expect("config value error");

        let source =
            source::MemSource::from_samples(&signal, channels, bits_per_sample, sample_rate);

        test_helper::integrity_test(
            |s| {
                coding::encode_with_fixed_block_size(&config, s, config.block_size)
                    .expect("source error")
            },
            &source,
        );
    }

    reusable!(REUSABLE_BUF: Vec<i32>);

    #[test]
    fn call_twice() {
        fn fn1() {
            reuse!(REUSABLE_BUF, |buf: &mut Vec<i32>| {
                assert_eq!(buf.len(), 0);
                buf.resize(5, 0i32);
            });
        }

        fn fn2() {
            reuse!(REUSABLE_BUF, |buf: &mut Vec<i32>| {
                assert_eq!(buf.len(), 5);
            });
        }

        fn1();
        fn2();
    }
}
