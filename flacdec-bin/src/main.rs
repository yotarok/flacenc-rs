// Copyright 2024 Google LLC
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

use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use clap::Parser;
use flacenc::component::Decode;
use log::info;

use termcolor::ColorChoice;
use termcolor::ColorSpec;
use termcolor::StandardStream;
use termcolor::WriteColor;

mod bitparse;
mod bitsource;
mod error;

/// Version of the decoder binary.
const CRATE_VERSION: &str = match option_env!("CARGO_PKG_VERSION") {
    Some(v) => v,
    None => "unknown",
};

/// FLAC decoder.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path for the output WAV file.
    #[clap(short, long)]
    output: String,
    /// If set, dump an intermediate structs to the specified path.
    #[clap(short, long)]
    dump_struct: Option<String>,
    /// Path for the input FLAC file.
    source: String,
}

/// Obtains `termcolor::StandardStream` for the default outputs.
fn terminal_output() -> Arc<termcolor::StandardStream> {
    Arc::new(StandardStream::stderr(ColorChoice::Auto))
}

/// Shows program banner.
///
/// # Errors
///
/// Propagates unknown I/O errors.
pub fn show_banner() -> Result<(), std::io::Error> {
    let termout = terminal_output();
    let mut termout = termout.lock();
    termout.set_color(ColorSpec::new().set_bold(true))?;
    write!(termout, "\n{:>10} ", "flacdec")?;
    termout.reset()?;
    writeln!(
        termout,
        "(decCLI v{}, engine v{})",
        CRATE_VERSION,
        flacenc::constant::build_info::CRATE_VERSION,
    )?;
    termout.set_color(ColorSpec::new().set_dimmed(true))?;
    writeln!(
        termout,
        "{:>10} [{}]",
        "",
        flacenc::constant::build_info::FEATURES
    )?;
    termout.reset()
}

#[allow(clippy::unnecessary_wraps)] // Return code will be used in future.
fn main_body(args: Args) -> Result<(), i32> {
    let _ = show_banner();

    let mut source =
        bitsource::MemSource::from_path(args.source).expect("Could not read the input file.");
    let stream = bitparse::parse_stream(&mut source).expect("No error");

    let stream_info = stream.stream_info();
    let frame_count = stream.frame_count();
    info!("Parsed {frame_count} frames.");

    if let Some(path) = args.dump_struct {
        let data = rmp_serde::to_vec_named(&stream).expect("Failed to serialize into msgpack.");
        let mut file = File::create(path).expect("Failed to create file.");
        file.write_all(&data).expect("Failed to write");
    }

    let mut writer = hound::WavWriter::create(
        args.output,
        hound::WavSpec {
            channels: stream_info.channels() as u16,
            sample_rate: stream_info.sample_rate() as u32,
            bits_per_sample: stream_info.bits_per_sample() as u16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .expect("Failed to create writer");
    for n in 0..frame_count {
        let frame_signal = stream.frame(n).unwrap().decode();
        for v in &frame_signal {
            let v = *v;
            writer.write_sample(v).expect("internal decoding error.");
        }
    }

    Ok(())
}

fn main() -> Result<(), i32> {
    env_logger::Builder::from_env("FLACENC_LOG")
        .format_timestamp(None)
        .init();
    main_body(Args::parse())
}
