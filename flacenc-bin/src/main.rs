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

// Note that clippy attributes should be in sync with those declared in "lib.rs"
#![warn(clippy::all, clippy::nursery, clippy::pedantic, clippy::cargo)]
#![allow(
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

use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use clap::Parser;
use log::info;
#[cfg(feature = "pprof")]
use pprof::protos::Message;

use flacenc::component::BitRepr;
use flacenc::component::Stream;
use flacenc::config;
use flacenc::error::EncodeError;
use flacenc::error::Verify;
use flacenc::source::Source;

mod display;
mod source;

use display::Progress;
use source::HoundSource;

/// FLAC encoder.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path for the output FLAC file.
    #[clap(short, long)]
    output: String,
    /// Path for the input audio file.
    source: String,
    /// If set, load config from the specified file.
    #[clap(short, long)]
    config: Option<String>,
    /// If set, dump the config used to the specified path.
    #[clap(long)]
    dump_config: Option<String>,
    /// If set, dump profiler output to the specified path.
    #[cfg(feature = "pprof")]
    #[clap(long)]
    pprof_output: Option<String>,
}

/// Exit codes of the encoder process.
enum ExitCode {
    #[allow(dead_code)]
    Normal = 0,
    InvalidConfig = -1,
}

/// Serializes `Stream` to a file.
#[allow(clippy::expect_used)]
fn write_stream<F: Write>(stream: &Stream, file: &mut F) -> usize {
    let bits = stream.count_bits();
    let mut bv = flacenc::bitsink::ByteSink::with_capacity(bits);
    stream.write(&mut bv).expect("Bitstream formatting failed.");
    file.write_all(bv.as_slice())
        .expect("Failed to write a bitstream to the file.");
    bits
}

fn run_encoder<S: Source>(
    encoder_config: &config::Encoder,
    source: S,
) -> Result<Stream, EncodeError> {
    let block_size = encoder_config.block_sizes[0];
    flacenc::encode_with_fixed_block_size(encoder_config, source, block_size)
}

fn log_build_constants() {
    info!(
        target: "flacenc-bin::build_info::jsonl",
        "{{ version: \"{}\", features: \"{}\", profile: \"{}\", rustc: \"{}\" }}",
        flacenc::constant::build_info::CRATE_VERSION,
        flacenc::constant::build_info::FEATURES,
        flacenc::constant::build_info::BUILD_PROFILE,
        flacenc::constant::build_info::RUSTC_VERSION,
    );
}

#[allow(clippy::let_underscore_must_use)]
fn main_body(args: Args) -> Result<(), i32> {
    let io_info = display::IoArgs::new(&args.config, &args.source, &args.output);
    let _ = display::show_banner();
    log_build_constants();
    let encoder_config = args.config.map_or_else(config::Encoder::default, |path| {
        let conf_str = std::fs::read_to_string(path).expect("Config file read error.");
        toml::from_str(&conf_str).expect("Config file syntax error.")
    });
    if let Err(e) = encoder_config.verify() {
        eprintln!("Error: {}", e.within("encoder_config"));
        return Err(ExitCode::InvalidConfig as i32);
    }

    let _ = display::show_progress(&io_info, &Progress::Started);

    let source = HoundSource::from_path(&args.source).expect("Failed to load input source.");
    let source_bytes = source.file_size();
    let source_duration_secs = Some(source.duration_as_secs());
    let encoder_start = Instant::now();

    let stream = run_encoder(&encoder_config, source).expect("Encoder error.");

    if let Some(path) = args.dump_config {
        let mut file = File::create(path).expect("Failed to create a file.");
        file.write_all(toml::to_string(&encoder_config).unwrap().as_bytes())
            .expect("File write failed.");
    }

    let mut file = File::create(args.output).expect("Failed to create a file.");
    let bits_written = write_stream(&stream, &mut file);

    let encode_time = encoder_start.elapsed();
    let _ = display::show_progress(
        &io_info,
        &Progress::Done {
            encode_time,
            bits_written,
            source_bytes,
            source_duration_secs,
        },
    );
    Ok(())
}

#[cfg(feature = "pprof")]
fn run_with_profiler_if_requested<F>(args: Args, body: F) -> Result<(), i32>
where
    F: FnOnce(Args) -> Result<(), i32>,
{
    if let Some(ref profiler_out) = args.pprof_output {
        let profiler_out = profiler_out.clone();
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .unwrap();
        let result = body(args);
        if let Ok(report) = guard.report().build() {
            let mut file = File::create(&profiler_out).unwrap();
            let profile = report.pprof().unwrap();

            let mut content = Vec::new();
            profile.write_to_vec(&mut content).unwrap();
            file.write_all(&content).unwrap();
        };
        result
    } else {
        body(args)
    }
}

#[cfg(not(feature = "pprof"))]
#[inline]
fn run_with_profiler_if_requested<F>(args: Args, body: F) -> Result<(), i32>
where
    F: FnOnce(Args) -> Result<(), i32>,
{
    body(args)
}

#[allow(clippy::expect_used)]
fn main() -> Result<(), i32> {
    env_logger::Builder::from_env("FLACENC_LOG")
        .format_timestamp(None)
        .init();
    run_with_profiler_if_requested(Args::parse(), main_body)
}
