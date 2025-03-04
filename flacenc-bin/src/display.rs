// Copyright 2023-2024 Google LLC
// Copyright 2025- flacenc-rs developers
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

//! A module for a fancy output for "flacenc-bin".

use std::borrow::Borrow;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use termcolor::Color;
use termcolor::ColorChoice;
use termcolor::ColorSpec;
use termcolor::StandardStream;
use termcolor::WriteColor;

const CRATE_VERSION: &str = match option_env!("CARGO_PKG_VERSION") {
    Some(v) => v,
    None => "unknown",
};
const DEFAULT_CONFIG_FILE_STEM: &str = "[default]";
const DEFAULT_INPUT_FILE_NAME: &str = "[default]";
const DEFAULT_OUTPUT_FILE_NAME: &str = "[default]";
const UNKNOWN_CONFIG_FILE_STEM: &str = "[unknown]";
const UNKNOWN_INPUT_FILE_NAME: &str = "[unknown]";
const UNKNOWN_OUTPUT_FILE_NAME: &str = "[unknown]";

/// Some data type that contains input and output file locations for display.
pub trait IoArgs {
    fn input_path(&self) -> Option<impl Borrow<Path>>;
    fn output_path(&self) -> Option<impl Borrow<Path>>;
    fn config_path(&self) -> Option<impl Borrow<Path>>;

    fn input_file_name(&self) -> String {
        self.input_path().as_ref().map_or_else(
            || DEFAULT_INPUT_FILE_NAME.to_owned(),
            |p| {
                p.borrow().file_name().map_or_else(
                    || UNKNOWN_INPUT_FILE_NAME.to_owned(),
                    |s| s.to_string_lossy().to_string(),
                )
            },
        )
    }

    fn output_file_name(&self) -> String {
        self.output_path().as_ref().map_or_else(
            || DEFAULT_OUTPUT_FILE_NAME.to_owned(),
            |p| {
                p.borrow().file_name().map_or_else(
                    || UNKNOWN_OUTPUT_FILE_NAME.to_owned(),
                    |s| s.to_string_lossy().to_string(),
                )
            },
        )
    }

    fn config_file_stem(&self) -> String {
        self.config_path().as_ref().map_or_else(
            || DEFAULT_CONFIG_FILE_STEM.to_owned(),
            |p| {
                p.borrow().file_stem().map_or_else(
                    || UNKNOWN_CONFIG_FILE_STEM.to_owned(),
                    |n| n.to_string_lossy().to_string(),
                )
            },
        )
    }
}

pub enum Progress {
    Started,
    // Encoding { current: usize, bits: usize, source_bytes: Option<usize>, offset_time:
    // Instantaneous }
    Encoded {
        encode_time: Duration,
        bits_written: usize,
        source_duration_secs: Option<f32>,
        source_bytes: Option<usize>,
    },
    Decoded {
        decode_time: Duration,
        source_duration_secs: f32,
        source_bytes: usize,
        samples_written: usize,
    },
}

fn terminal_output() -> Arc<termcolor::StandardStream> {
    Arc::new(StandardStream::stderr(ColorChoice::Auto))
}

pub enum Banner {
    EncoderSmall,
    DecoderSmall,
}

/// Outputs the initial banner.
pub fn show_banner(mode: &Banner) -> Result<(), std::io::Error> {
    let termout = terminal_output();
    let mut termout = termout.lock();
    termout.set_color(ColorSpec::new().set_bold(true))?;
    write!(termout, "\n{:>10} ", "flacenc")?;
    termout.reset()?;
    write!(
        termout,
        "(CLI v{}, engine v{})",
        CRATE_VERSION,
        flacenc::constant::build_info::CRATE_VERSION,
    )?;
    match *mode {
        Banner::EncoderSmall => {
            writeln!(termout)?;
        }
        Banner::DecoderSmall => {
            writeln!(termout, " -- decoder mode")?;
        }
    }
    termout.set_color(ColorSpec::new().set_dimmed(true))?;
    writeln!(
        termout,
        "{:>10} [{}]",
        "",
        flacenc::constant::build_info::FEATURES
    )?;
    termout.reset()
}

/// Outputs after-encode summary to the terminal.
#[allow(clippy::uninlined_format_args)] // for readability
fn show_progress_encoded<T>(
    io: &T,
    bits_written: usize,
    encode_time: Duration,
    source_bytes: Option<usize>,
    source_duration_secs: Option<f32>,
) -> Result<(), std::io::Error>
where
    T: IoArgs,
{
    let termout = terminal_output();
    let mut termout = termout.lock();
    termout.set_color(ColorSpec::new().set_fg(Some(Color::Green)).set_bold(true))?;
    write!(termout, "{:>10} ", "Encoded")?;
    termout.reset()?;

    let bytes_written = (bits_written + 7) >> 3;
    let encode_secs = encode_time.as_secs_f32();
    let output_throughput = bytes_written as f32 / 1024.0 / 1024.0 / encode_secs;
    writeln!(
        termout,
        "{} [{} bytes / {:.3} s, {:.1} MiB/s]",
        io.output_file_name(),
        bytes_written,
        encode_secs,
        output_throughput
    )?;
    if let Some(source_bytes) = source_bytes {
        let compression_rate = bytes_written as f32 / source_bytes as f32;
        write!(termout, "{:>10} ", "")?;
        writeln!(
            termout,
            "compression ratio = {} / {} = {:.1}%",
            bytes_written,
            source_bytes,
            compression_rate * 100.0
        )?;
    }
    if let Some(source_duration_secs) = source_duration_secs {
        let irtf = source_duration_secs / encode_secs;
        write!(termout, "{:>10} ", "")?;
        writeln!(
            termout,
            "inverse RTF = {:.3}s / {:.3}s = {:.1}x",
            source_duration_secs, encode_secs, irtf,
        )?;
    }
    writeln!(termout)
}

/// Outputs after-encode summary to the terminal.
#[allow(clippy::uninlined_format_args)] // for readability
fn show_progress_decoded<T>(
    io: &T,
    samples_written: usize,
    decode_time: Duration,
    _source_bytes: usize,
    source_duration_secs: f32,
) -> Result<(), std::io::Error>
where
    T: IoArgs,
{
    let termout = terminal_output();
    let mut termout = termout.lock();
    termout.set_color(ColorSpec::new().set_fg(Some(Color::Green)).set_bold(true))?;
    write!(termout, "{:>10} ", "Decoded")?;
    termout.reset()?;

    let decode_secs = decode_time.as_secs_f32();
    writeln!(
        termout,
        "{} => {} [{} samples / {:.3} s]",
        io.input_file_name(),
        io.output_file_name(),
        samples_written,
        decode_secs,
    )?;
    let irtf = source_duration_secs / decode_secs;
    write!(termout, "{:>10} ", "")?;
    writeln!(
        termout,
        "inverse RTF = {:.3}s / {:.3}s = {:.1}x",
        source_duration_secs, decode_secs, irtf,
    )?;
    writeln!(termout)
}

/// Outputs progress to the terminal.
pub fn show_progress<T>(io: &T, progress: &Progress) -> Result<(), std::io::Error>
where
    T: IoArgs,
{
    match *progress {
        Progress::Started => {
            let termout = terminal_output();
            let mut termout = termout.lock();
            termout.set_color(ColorSpec::new().set_fg(Some(Color::Cyan)).set_bold(true))?;
            write!(termout, "{:>10} ", "Encoding")?;
            termout.reset()?;
            writeln!(
                termout,
                "{} => {} [{}]",
                io.input_file_name(),
                io.output_file_name(),
                io.config_file_stem()
            )
        }
        Progress::Encoded {
            bits_written,
            encode_time,
            source_bytes,
            source_duration_secs,
        } => show_progress_encoded(
            io,
            bits_written,
            encode_time,
            source_bytes,
            source_duration_secs,
        ),
        Progress::Decoded {
            decode_time,
            samples_written,
            source_bytes,
            source_duration_secs,
        } => show_progress_decoded(
            io,
            samples_written,
            decode_time,
            source_bytes,
            source_duration_secs,
        ),
    }
}

/// Shows an error message.
#[allow(clippy::uninlined_format_args)] // for readability
#[allow(clippy::needless_pass_by_value)] // for explicitly say err cannot be reused.
pub fn show_error_msg<E>(msg: &str, err: Option<E>)
where
    E: std::error::Error + Sized,
{
    let termout = terminal_output();
    let mut termout = termout.lock();
    let _ = termout.set_color(ColorSpec::new().set_fg(Some(Color::Red)).set_bold(true));
    let _ = write!(termout, "{:>10} ", "ERROR");
    let _ = termout.reset();
    let _ = writeln!(termout, "{}", msg);
    if let Some(e) = err {
        let _ = termout.set_color(ColorSpec::new().set_dimmed(true));
        let _ = write!(termout, "{:>10} ", "");
        let _ = writeln!(termout, "{:?}", e);
    }
    let _ = termout.reset();
}
