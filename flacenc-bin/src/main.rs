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
    clippy::missing_const_for_fn,
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

use std::fmt::Debug;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use clap::Parser;
#[cfg(feature = "pprof")]
use pprof::protos::Message;

use flacenc::coding;
use flacenc::component::BitRepr;
use flacenc::component::Stream;
use flacenc::config;
use flacenc::constant::ExitCode;
use flacenc::error::SourceError;
use flacenc::error::Verify;
use flacenc::source;

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

/// Serializes `Stream` to a file.
#[allow(clippy::expect_used)]
fn write_stream<F: Write>(stream: &Stream, file: &mut F) {
    eprintln!("{} bits to be written", stream.count_bits());
    let mut bv = flacenc::bitsink::ByteVec::new();
    stream.write(&mut bv).expect("Bitstream formatting failed.");
    let mut writer = BufWriter::new(file);
    writer.write_all(&bv.bytes).expect("");
}

/// Collect iterator of `Result`s, and returns values or the first error.
fn collect_results<I, T, E>(iter: I) -> Result<Vec<T>, E>
where
    I: Iterator<Item = Result<T, E>>,
    E: std::error::Error,
    T: Debug,
{
    let mut error: Option<E> = None;
    let mut samples = vec![];
    for r in iter {
        if let Ok(v) = r {
            samples.push(v);
        } else {
            error = Some(r.unwrap_err());
            break;
        }
    }
    error.map_or_else(|| Ok(samples), Err)
}

/// Loads wave file and constructs `PreloadedSignal`.
///
/// # Errors
///
/// This function propagates errors emitted by the backend wave parser.
pub fn load_input_wav<P: AsRef<Path>>(
    path: P,
) -> Result<source::PreloadedSignal, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path).map_err(Box::new)?;
    let spec = reader.spec();
    let samples: Vec<i32> = collect_results(reader.samples())?;
    Ok(source::PreloadedSignal::from_samples(
        &samples,
        spec.channels as usize,
        spec.bits_per_sample as usize,
        spec.sample_rate as usize,
    ))
}

#[cfg(feature = "experimental")]
fn run_encoder(
    encoder_config: &config::Encoder,
    source: source::PreloadedSignal,
) -> Result<Stream, SourceError> {
    if encoder_config.block_sizes.len() == 1 {
        let block_size = encoder_config.block_sizes[0];
        coding::encode_with_fixed_block_size(&encoder_config, source, block_size)
    } else {
        coding::encode_with_multiple_block_sizes(&encoder_config, source)
    }
}

#[cfg(not(feature = "experimental"))]
fn run_encoder(
    encoder_config: &config::Encoder,
    source: source::PreloadedSignal,
) -> Result<Stream, SourceError> {
    let block_size = encoder_config.block_sizes[0];
    coding::encode_with_fixed_block_size(encoder_config, source, block_size)
}

fn main_body(args: Args) -> Result<(), i32> {
    let encoder_config = args.config.map_or_else(config::Encoder::default, |path| {
        let conf_str = std::fs::read_to_string(path).expect("Config file read error.");
        toml::from_str(&conf_str).expect("Config file syntax error.")
    });

    if let Err(e) = encoder_config.verify() {
        eprintln!("Error: {}", e.within("encoder_config"));
        return Err(ExitCode::InvalidConfig as i32);
    }

    let source = load_input_wav(&args.source).expect("Failed to load input source.");

    let stream = run_encoder(&encoder_config, source).expect("Encoder error.");

    if let Some(path) = args.dump_config {
        let mut file = File::create(path).expect("Failed to create a file.");
        file.write_all(toml::to_string(&encoder_config).unwrap().as_bytes())
            .expect("File write failed.");
    }

    let mut file = File::create(args.output).expect("Failed to create a file.");
    write_stream(&stream, &mut file);
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
    run_with_profiler_if_requested(Args::parse(), main_body)
}
