// Copyright 2023 Google LLC
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

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
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
const DEFAULT_CONFIG_NAME: &str = "[default]";
const UNKNOWN_CONFIG_NAME: &str = "[unknown]";
const UNKNOWN_INPUT_NAME: &str = "[unknown]";
const UNKNOWN_OUTPUT_NAME: &str = "[unknown]";

// might be better if we handle Args and Config directly.
#[allow(clippy::struct_field_names)] // tentatively allowed
pub struct IoArgs {
    config_path: Option<PathBuf>,
    input_path: PathBuf,
    output_path: PathBuf,
}

impl IoArgs {
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        config_path: &Option<P>,
        input_path: Q,
        output_path: R,
    ) -> Self {
        Self {
            config_path: config_path.as_ref().map(|x| x.as_ref().to_path_buf()),
            input_path: input_path.as_ref().to_path_buf(),
            output_path: output_path.as_ref().to_path_buf(),
        }
    }

    pub fn output_name(&self) -> String {
        self.output_path.file_name().map_or_else(
            || UNKNOWN_OUTPUT_NAME.to_owned(),
            |s| s.to_string_lossy().to_string(),
        )
    }

    pub fn input_name(&self) -> String {
        self.input_path.file_name().map_or_else(
            || UNKNOWN_INPUT_NAME.to_owned(),
            |s| s.to_string_lossy().to_string(),
        )
    }

    pub fn config_name(&self) -> String {
        self.config_path.as_ref().map_or_else(
            || DEFAULT_CONFIG_NAME.to_owned(),
            |p| {
                p.file_stem().map_or_else(
                    || UNKNOWN_CONFIG_NAME.to_owned(),
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
    Done {
        encode_time: Duration,
        bits_written: usize,
        source_duration_secs: Option<f32>,
        source_bytes: Option<usize>,
    },
}

fn terminal_output() -> Arc<termcolor::StandardStream> {
    Arc::new(StandardStream::stderr(ColorChoice::Auto))
}

/// Outputs the initial banner.
pub fn show_banner() -> Result<(), std::io::Error> {
    let termout = terminal_output();
    let mut termout = termout.lock();
    termout.set_color(ColorSpec::new().set_bold(true))?;
    write!(termout, "\n{:>10} ", "flacenc")?;
    termout.reset()?;
    writeln!(
        termout,
        "(CLI v{}, engine v{})",
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

/// Outputs after-encode summary to the terminal.
#[allow(clippy::uninlined_format_args)] // for readability
fn show_progress_done(
    io: &IoArgs,
    bits_written: usize,
    encode_time: Duration,
    source_bytes: Option<usize>,
    source_duration_secs: Option<f32>,
) -> Result<(), std::io::Error> {
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
        io.output_name(),
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

/// Outputs progress to the terminal.
pub fn show_progress(io: &IoArgs, progress: &Progress) -> Result<(), std::io::Error> {
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
                io.input_name(),
                io.output_name(),
                io.config_name()
            )
        }
        Progress::Done {
            bits_written,
            encode_time,
            source_bytes,
            source_duration_secs,
        } => show_progress_done(
            io,
            bits_written,
            encode_time,
            source_bytes,
            source_duration_secs,
        ),
    }
}
