// Copyright 2022-2024 Google LLC
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
    clippy::multiple_crate_versions
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

use std::borrow::Borrow;
use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::num::NonZeroU8;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Args;
use clap::Parser;
use clap::Subcommand;
use log::info;
use md5::Digest;
#[cfg(feature = "pprof")]
use pprof::protos::Message;

use flacenc::component::parser;
use flacenc::component::BitRepr;
use flacenc::component::Decode;
use flacenc::component::Stream;
use flacenc::config;
use flacenc::error::EncodeError;
use flacenc::error::Verified;
use flacenc::error::Verify;
use flacenc::source::Source;

mod display;
mod source;

use display::Progress;
use source::HoundSource;

/// FLAC encoder.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct ProgramArgs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Args)]
struct EncodeArgs {
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
    /// Path for the output FLAC file.
    #[clap(short, long)]
    output: String,
    /// Path for the input audio file.
    source: String,
}

impl display::IoArgs for EncodeArgs {
    fn input_path(&self) -> Option<impl Borrow<Path>> {
        Some(PathBuf::from(&self.source))
    }
    fn output_path(&self) -> Option<impl Borrow<Path>> {
        Some(PathBuf::from(&self.output))
    }
    fn config_path(&self) -> Option<impl Borrow<Path>> {
        self.config.as_ref().map(PathBuf::from)
    }
}

#[derive(Debug, Args)]
struct DecodeArgs {
    #[clap(long)]
    dump_struct: Option<String>,
    /// Path for the output FLAC file.
    #[clap(short, long)]
    output: String,
    /// Path for the input audio file.
    source: String,
}

impl display::IoArgs for DecodeArgs {
    fn input_path(&self) -> Option<impl Borrow<Path>> {
        Some(PathBuf::from(&self.source))
    }
    fn output_path(&self) -> Option<impl Borrow<Path>> {
        Some(PathBuf::from(&self.output))
    }
    fn config_path(&self) -> Option<impl Borrow<Path>> {
        Option::<PathBuf>::None
    }
}

const DEFAULT_COMMAND_STR: &str = "encode";
const KNOWN_COMMAND_STRS: [&str; 2] = ["encode", "decode"];

#[derive(Debug, Subcommand)]
enum Commands {
    Encode(EncodeArgs),
    Decode(DecodeArgs),
}

// Error code taken from "sysexits.h"
//
// The unsafe blocks below is only a tentative workaround before "const_option" feature
// is stabilized. See [[https://github.com/rust-lang/rust/issues/67441]].
const EX_DATAERR: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(65) };
const EX_NOINPUT: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(66) };
const EX_SOFTWARE: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(70) };
const EX_CANTCREAT: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(73) };
const EX_IOERR: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(74) };

/// Serializes `Stream` to a file.
fn write_stream<F: Write>(stream: &Stream, file: &mut F) -> Result<usize, std::io::Error> {
    let bits = stream.count_bits();
    let mut bv = flacenc::bitsink::ByteSink::with_capacity(bits);

    // Currently, it's difficult to distinguish ifthe error is happened in bitstream formatting
    // (should lead to EX_SOFTWARE exit code), or file writing (lead to EX_IOERR). Currently,
    // the error from this function is always mapped to EX_IOERR though it's not very accurate.
    stream.write(&mut bv).map_err(std::io::Error::other)?;
    file.write_all(bv.as_slice())?;
    Ok(bits)
}

fn run_encoder<S: Source>(
    encoder_config: &Verified<config::Encoder>,
    source: S,
) -> Result<Stream, EncodeError> {
    flacenc::encode_with_fixed_block_size(encoder_config, source, encoder_config.block_size)
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

#[allow(clippy::needless_pass_by_value)]
fn main_enc_body(args: EncodeArgs) -> Result<(), NonZeroU8> {
    let _ = display::show_banner(&display::Banner::EncoderSmall);
    log_build_constants();
    let encoder_config = args
        .config
        .as_ref()
        .map_or_else(config::Encoder::default, |path| {
            let conf_str = std::fs::read_to_string(path).expect("Config file read error.");
            toml::from_str(&conf_str).expect("Config file syntax error.")
        });
    let encoder_config = encoder_config.into_verified().map_err(|(_, e)| {
        display::show_error_msg("Invalid config parameter is detected.", Some(e));
        EX_DATAERR
    })?;

    let _ = display::show_progress(&args, &Progress::Started);

    let source = HoundSource::from_path(&args.source).map_err(|e| {
        // note that `Box<dyn Error>` is not `Error`. afaik this is for avoiding implement
        // overlap. `From<E: Error> for Box<dyn Error>` for convenience, if `Box<E>` is also
        // `Error`, it overlaps with `impl From<T> for T`.
        display::show_error_msg("Failed to read the source WAV file.", Some(e.as_ref()));
        EX_NOINPUT
    })?;
    let source_bytes = source.file_size();
    let source_duration_secs = Some(source.duration_as_secs());
    let encoder_start = Instant::now();

    let stream = run_encoder(&encoder_config, source).map_err(|e| {
        display::show_error_msg("Failed to read the source WAV file.", Some(e));
        EX_SOFTWARE
    })?;

    if let Some(ref path) = args.dump_config {
        let mut file = File::create(path).map_err(|e| {
            display::show_error_msg("Failed to create the config dump file.", Some(e));
            EX_CANTCREAT
        })?;
        file.write_all(toml::to_string(&encoder_config).unwrap().as_bytes())
            .map_err(|e| {
                display::show_error_msg("Failed to write config dump file.", Some(e));
                EX_IOERR
            })?;
    }

    let mut file = File::create(&args.output).map_err(|e| {
        display::show_error_msg("Failed to create the output file.", Some(e));
        EX_CANTCREAT
    })?;
    let bits_written = write_stream(&stream, &mut file).map_err(|e| {
        display::show_error_msg("Failed to write to the output file.", Some(e));
        EX_IOERR
    })?;

    let encode_time = encoder_start.elapsed();
    let _ = display::show_progress(
        &args,
        &Progress::Encoded {
            encode_time,
            bits_written,
            source_bytes,
            source_duration_secs,
        },
    );
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::unnecessary_wraps)]
fn main_dec_body(args: DecodeArgs) -> Result<(), NonZeroU8> {
    let _ = display::show_banner(&display::Banner::DecoderSmall);
    log_build_constants();

    let _ = display::show_progress(&args, &Progress::Started);
    let decoder_start = Instant::now();
    let bytes = std::fs::read(&args.source).map_err(|e| {
        display::show_error_msg("Failed to read the source FLAC file.", Some(e));
        EX_NOINPUT
    })?;
    let (remaining_input, stream) = parser::stream::<()>(&bytes).map_err(|e| {
        display::show_error_msg("Failed to parse the source FLAC file.", Some(e));
        EX_NOINPUT
    })?;

    if !remaining_input.is_empty() {
        display::show_error_msg::<std::io::Error>(
            "The input file contains extra trailing bytes.",
            None,
        );
        return Err(EX_NOINPUT);
    }

    let stream_info = stream.stream_info();
    let sample_count = stream_info.total_samples();
    let sample_rate = stream_info.sample_rate() as f32;
    let frame_count = stream.frame_count();

    if let Some(ref path) = args.dump_struct {
        let data = rmp_serde::to_vec_named(&stream).map_err(|e| {
            display::show_error_msg("Failed to serialize into msgpack format.", Some(e));
            EX_SOFTWARE
        })?;

        let mut file = File::create(path).map_err(|e| {
            display::show_error_msg("Failed to create the stream dump file.", Some(e));
            EX_CANTCREAT
        })?;

        file.write_all(&data).map_err(|e| {
            display::show_error_msg("Failed to write stream dump file.", Some(e));
            EX_IOERR
        })?;
    }

    let temp_wav_file = tempfile::NamedTempFile::new().map_err(|e| {
        display::show_error_msg("Unable to create temporary output file.", Some(e));
        EX_IOERR
    })?;

    let mut writer = hound::WavWriter::new(
        &temp_wav_file,
        hound::WavSpec {
            channels: stream_info.channels() as u16,
            sample_rate: stream_info.sample_rate() as u32,
            bits_per_sample: stream_info.bits_per_sample() as u16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .map_err(|e| {
        display::show_error_msg("Failed to create temporary output file.", Some(e));
        EX_CANTCREAT
    })?;

    let mut md5 = md5::Md5::new();
    let bytes_per_sample = (stream_info.bits_per_sample() + 7) / 8;
    for n in 0..frame_count {
        let frame = stream.frame(n).unwrap();
        let frame_signal = frame.decode();
        for v in &frame_signal {
            let v = *v;
            writer.write_sample(v).map_err(|e| {
                display::show_error_msg("Failed to write decoded samples.", Some(e));
                EX_CANTCREAT
            })?;
            md5.update(&v.to_le_bytes()[0..bytes_per_sample]);
        }
    }
    let computed_digest: [u8; 16] = md5.finalize().into();
    let stored_digest = stream_info.md5_digest();
    if stored_digest != &[0x00; 16] && stored_digest != &computed_digest {
        display::show_error_msg::<std::convert::Infallible>("MD5 digests did not match.", None);
        return Err(EX_CANTCREAT);
    }

    std::fs::rename(temp_wav_file.path(), &args.output).map_err(|e| {
        display::show_error_msg("Failed to write to the output path.", Some(e));
        EX_CANTCREAT
    })?;

    let decode_time = decoder_start.elapsed();
    let _ = display::show_progress(
        &args,
        &Progress::Decoded {
            decode_time,
            samples_written: sample_count,
            source_duration_secs: (sample_count as f32) / sample_rate,
            source_bytes: bytes.len(),
        },
    );

    Ok(())
}

#[cfg(feature = "pprof")]
fn run_with_profiler_if_requested<F>(args: EncodeArgs, body: F) -> Result<(), NonZeroU8>
where
    F: FnOnce(EncodeArgs) -> Result<(), NonZeroU8>,
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

/// Parses args with prepending the default command "encode" if no command is specified.
fn parse_args<I>(args: I) -> ProgramArgs
where
    I: Iterator<Item = std::ffi::OsString>,
{
    // If the first argument is not the known subcommand, this part of code inserts the default
    // subcommand "encode" into `args_raw`. This is a bit hacky, but "flacenc encode" is a bit
    // redundant as the command name already contains "enc".
    let mut args_raw: Vec<std::ffi::OsString> = args.collect();
    let empty = std::ffi::OsString::new();
    if !KNOWN_COMMAND_STRS.contains(&args_raw.get(1).unwrap_or(&empty).to_string_lossy().as_ref()) {
        args_raw.insert(1, DEFAULT_COMMAND_STR.to_owned().into());
    }
    ProgramArgs::parse_from(args_raw.iter())
}

#[cfg(not(feature = "pprof"))]
#[inline]
fn run_with_profiler_if_requested<F>(args: EncodeArgs, body: F) -> Result<(), NonZeroU8>
where
    F: FnOnce(EncodeArgs) -> Result<(), NonZeroU8>,
{
    body(args)
}

fn main() -> ExitCode {
    env_logger::Builder::from_env("FLACENC_LOG")
        .format_timestamp(None)
        .init();

    match parse_args(std::env::args_os()).command {
        Commands::Encode(args) => run_with_profiler_if_requested(args, main_enc_body),
        Commands::Decode(args) => main_dec_body(args),
    }
    .map_or_else(|e| ExitCode::from(e.get()), |()| ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    use flacenc::sigen::*;
    use rstest::rstest;

    #[test]
    fn arg_parser() {
        match parse_args(
            ["binary-name", "encode", "-o", "output", "source"]
                .iter()
                .map(|s| s.to_owned().into()),
        )
        .command
        {
            Commands::Encode(args) => {
                assert_eq!("output", args.output);
                assert_eq!("source", args.source);
            }
            x @ Commands::Decode(_) => {
                panic!("result should be encode args, but {x:?}.");
            }
        }

        match parse_args(
            ["binary-name", "decode", "-o", "output", "source"]
                .iter()
                .map(|s| s.to_owned().into()),
        )
        .command
        {
            Commands::Decode(args) => {
                assert_eq!("output", args.output);
                assert_eq!("source", args.source);
            }
            x @ Commands::Encode(_) => {
                panic!("result should be decode args, but {x:?}");
            }
        }

        match parse_args(
            ["binary-name", "-o", "output", "source"]
                .iter()
                .map(|s| s.to_owned().into()),
        )
        .command
        {
            Commands::Encode(args) => {
                assert_eq!("output", args.output);
                assert_eq!("source", args.source);
            }
            x @ Commands::Decode(_) => {
                panic!("result should be encode args, but {x:?}");
            }
        }
    }

    struct SourceConfig {
        duration_secs: usize,
        sample_rate: usize,
        channels: usize,
        bits_per_sample: usize,
    }

    fn generate_test_wav<P: AsRef<Path>>(output_path: P, src: &SourceConfig) {
        let period_440 = src.sample_rate / 440;
        let signal_len = src.sample_rate * src.duration_secs;
        let mut signals = vec![];
        for _ch in 0..src.channels {
            signals.push(
                Sine::new(period_440, 0.8)
                    .mix(Noise::new(0.2))
                    .to_vec_quantized(src.bits_per_sample, signal_len),
            );
        }

        let mut writer = hound::WavWriter::create(
            &output_path,
            hound::WavSpec {
                channels: src.channels as u16,
                sample_rate: src.sample_rate as u32,
                bits_per_sample: src.bits_per_sample as u16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("Should be able to create test wav.");

        for t in 0..signal_len {
            for s in &signals {
                let v = s[t];
                writer
                    .write_sample(v)
                    .expect("Should be able to write a sample");
            }
        }
    }

    fn check_wav_file_eq<P: AsRef<Path>, Q: AsRef<Path>>(expected: P, actual: Q) {
        let mut reader_expected =
            hound::WavReader::open(expected).expect("expected file should be openable.");
        let mut reader_actual =
            hound::WavReader::open(actual).expect("actual file should be openable.");

        assert_eq!(reader_expected.spec(), reader_actual.spec());
        assert_eq!(reader_expected.len(), reader_actual.len());

        for (n, (e, a)) in reader_expected
            .samples()
            .zip(reader_actual.samples())
            .enumerate()
        {
            let e: i32 = e.expect("sample from expected file  should be readable");
            let a: i32 = a.expect("sample from actual file  should be readable");
            assert_eq!(
                e, a,
                "expected and actual samples should be identical (sample-offset={n})."
            );
        }
    }

    #[rstest]
    #[case(
        "canonical", 
        SourceConfig { duration_secs: 3, sample_rate: 44100, channels: 2, bits_per_sample: 16 }
    )]
    #[case(
        "sr44097",
        SourceConfig { duration_secs: 3, sample_rate: 44097, channels: 2, bits_per_sample: 16 }
    )]
    #[case(
        "ch1",
        SourceConfig { duration_secs: 3, sample_rate: 44100, channels: 1, bits_per_sample: 16 }
    )]
    #[case(
        "ch3",
        SourceConfig { duration_secs: 3, sample_rate: 44100, channels: 3, bits_per_sample: 16 }
    )]
    #[case(
        "bps24",
        SourceConfig { duration_secs: 3, sample_rate: 44100, channels: 2, bits_per_sample: 24 }
    )]
    #[case(
        "bps8",
        SourceConfig { duration_secs: 3, sample_rate: 44100, channels: 1, bits_per_sample: 8 }
    )]
    fn integration_encoder_decoder(#[case] case_name: &str, #[case] source_config: SourceConfig) {
        let tmpdir = tempfile::tempdir().unwrap();
        let tmpdir_path = std::env::var_os("FLACENC_TEST_WORKDIR").map_or_else(
            || tmpdir.as_ref().to_path_buf(),
            |dir| {
                eprintln!("FLACENC_TEST_WORKDIR is set to {}", dir.to_string_lossy());
                let mut workdir = std::path::PathBuf::from(dir);
                workdir.push(case_name);
                std::fs::create_dir_all(&workdir).expect("Work dir creation should not fail.");
                workdir
            },
        );
        let mut config_dump_path = tmpdir_path.clone();
        config_dump_path.push("dumped.toml");
        let mut flac_path = tmpdir_path.clone();
        flac_path.push("compressed.flac");
        let mut source_path = tmpdir_path.clone();
        source_path.push("source.wav");
        let mut decoded_path = tmpdir_path;
        decoded_path.push("decoded.wav");

        generate_test_wav(&source_path, &source_config);

        main_enc_body(EncodeArgs {
            config: None,
            dump_config: Some(config_dump_path.to_string_lossy().to_string()),
            source: source_path.to_string_lossy().to_string(),
            output: flac_path.to_string_lossy().to_string(),
        })
        .expect("no error expected.");

        main_dec_body(DecodeArgs {
            dump_struct: None,
            source: flac_path.to_string_lossy().to_string(),
            output: decoded_path.to_string_lossy().to_string(),
        })
        .expect("no error expected.");

        check_wav_file_eq(source_path, decoded_path);
    }
}
