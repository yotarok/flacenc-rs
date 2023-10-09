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
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use clap::Parser;
#[cfg(feature = "pprof")]
use pprof::protos::Message;

use flacenc::component::BitRepr;
use flacenc::component::Stream;
use flacenc::config;
use flacenc::error::EncodeError;
use flacenc::error::SourceError;
use flacenc::error::SourceErrorReason;
use flacenc::error::Verify;
use flacenc::source::Context;
use flacenc::source::FrameBuf;
use flacenc::source::Source;

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
fn write_stream<F: Write>(stream: &Stream, file: &mut F) {
    let bits = stream.count_bits();
    eprintln!("{bits} bits to be written");
    let mut bv = flacenc::bitsink::ByteSink::with_capacity(bits);
    stream.write(&mut bv).expect("Bitstream formatting failed.");
    file.write_all(bv.as_byte_slice())
        .expect("Failed to write a bitstream to the file.");
}

/// Converts a byte-sequence of little-endian integers to integers (i32).
fn bytes_to_ints(bytes: &[u8], dest: &mut [i32], bytes_per_sample: usize) {
    let bitshift = bytes_per_sample * 8;
    let pad = 4 - bytes_per_sample;
    for (vals, dest_p) in bytes.chunks(bytes_per_sample).zip(dest.iter_mut()) {
        let bs = std::array::from_fn(|n| if n < pad { 0 } else { vals[n - pad] });
        *dest_p = i32::from_le_bytes(bs) >> bitshift;
    }
}

/// An example of `flacenc::source::Source` based on `hound::WavReader`.
///
/// To mitigate I/O overhead due to sample-by-sample retrieval in hound API,
/// this source only uses hound to parse WAV header and seeks offset for the
/// first sample. After parsing the header, the inside `BufReader` is obtained
/// via `WavReader::into_inner` and it is used to retrieve blocks of samples.
struct HoundSource {
    spec: hound::WavSpec,
    duration: usize,
    reader: BufReader<File>,
    bytes_per_sample: usize,
    buf: Vec<i32>,
    bytebuf: Vec<u8>,
    current_offset: usize,
}

impl HoundSource {
    /// Constructs `HoundSource` from `path`.
    ///
    /// # Errors
    ///
    /// The function fails when file is not found or has invalid format. This
    /// function currently do not support WAVs with IEEE float samples, and it
    /// returns `SourceError` with `SourceErrorReason::InvalidFormat` if the
    /// samples are floats.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = Box::new(hound::WavReader::open(path).map_err(Box::new)?);
        let spec = reader.spec();
        let duration = reader.duration() as usize;
        reader.seek(0).unwrap();
        if spec.sample_format == hound::SampleFormat::Int {
            Ok(Self {
                spec,
                duration,
                reader: reader.into_inner(),
                bytes_per_sample: (spec.bits_per_sample as usize + 7) / 8,
                buf: Vec::new(),
                bytebuf: Vec::new(),
                current_offset: 0,
            })
        } else {
            Err(Box::new(SourceError::by_reason(
                SourceErrorReason::InvalidFormat,
            )))
        }
    }
}

impl Source for HoundSource {
    #[inline]
    fn channels(&self) -> usize {
        self.spec.channels as usize
    }

    #[inline]
    fn bits_per_sample(&self) -> usize {
        self.spec.bits_per_sample as usize
    }

    #[inline]
    fn sample_rate(&self) -> usize {
        self.spec.sample_rate as usize
    }

    #[inline]
    fn read_samples(
        &mut self,
        dest: &mut FrameBuf,
        context: &mut Context,
    ) -> Result<usize, SourceError> {
        self.buf.clear();
        self.bytebuf.clear();
        let to_read = std::cmp::min(self.duration - self.current_offset, dest.size());
        let to_read_bytes = to_read * self.bytes_per_sample * self.channels();

        self.bytebuf.resize(to_read_bytes, 0u8);
        let read_bytes = self
            .reader
            .read(&mut self.bytebuf)
            .map_err(SourceError::from_io_error)?;
        self.current_offset += to_read;

        self.buf.resize(read_bytes / self.bytes_per_sample, 0);
        bytes_to_ints(&self.bytebuf, &mut self.buf, self.bytes_per_sample);

        dest.fill_from_interleaved(&self.buf);
        if !self.buf.is_empty() {
            context.update_with_le_bytes(&self.bytebuf, dest.size())?;
        }
        Ok(self.buf.len() / self.channels())
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.duration)
    }
}

fn run_encoder<S: Source>(
    encoder_config: &config::Encoder,
    source: S,
) -> Result<Stream, EncodeError> {
    let block_size = encoder_config.block_sizes[0];
    flacenc::encode_with_fixed_block_size(encoder_config, source, block_size)
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

    let source = HoundSource::from_path(&args.source).expect("Failed to load input source.");

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    pub fn test_bytes_to_ints() {
        let bs = [0x34, 0x12, 0x56, 0x34, 0x78, 0x56, 0xCC, 0xED, 0x00, 0x00];
        let mut ints = [0i32; 5];
        bytes_to_ints(&bs, &mut ints, 2);
        assert_eq!(ints, [0x1234, 0x3456, 0x5678, -0x1234, 0x0000]);
    }
}
