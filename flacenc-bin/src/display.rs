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
    Done { bits_written: usize },
}

fn terminal_output() -> Arc<termcolor::StandardStream> {
    Arc::new(StandardStream::stderr(ColorChoice::Auto))
}

/// Show the initial banner.
pub fn show_banner() -> Result<(), std::io::Error> {
    let termout = terminal_output();
    let mut termout = termout.lock();
    termout.set_color(ColorSpec::new().set_bold(true))?;
    write!(termout, "\n{:>10} ", "flacenc")?;
    termout.reset()?;
    writeln!(
        termout,
        "(engine v{}, CLI v{})",
        flacenc::constant::build_info::CRATE_VERSION,
        CRATE_VERSION
    )
}

pub fn show_progress(io: &IoArgs, progress: &Progress) -> Result<(), std::io::Error> {
    let termout = terminal_output();
    let mut termout = termout.lock();
    match *progress {
        Progress::Started => {
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
        Progress::Done { bits_written, .. } => {
            termout.set_color(ColorSpec::new().set_fg(Some(Color::Green)).set_bold(true))?;
            write!(termout, "{:>10} ", "Encoded")?;
            termout.reset()?;
            writeln!(
                termout,
                "{} [{} bytes]",
                io.output_name(),
                ((bits_written + 7) >> 3)
            )?;
            writeln!(termout)
        }
    }
}
