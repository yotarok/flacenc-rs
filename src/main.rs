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
    clippy::use_self,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::multiple_crate_versions,
    clippy::must_use_candidate
)]

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use bitvec::prelude::BitVec;
use bitvec::prelude::Msb0;
use clap::Parser;

use flacenc_rs::coding;
use flacenc_rs::component::BitRepr;
use flacenc_rs::component::Stream;
use flacenc_rs::config;
use flacenc_rs::source;

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
}

/// Serializes `Stream` to a file.
#[allow(clippy::expect_used)]
fn write_stream<F: Write>(stream: &Stream, file: &mut F) {
    eprintln!("{} bits to be written", stream.count_bits());
    let mut bv: BitVec<u8, Msb0> = BitVec::with_capacity(stream.count_bits());
    stream.write(&mut bv).expect("Bitstream formatting failed.");
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bv.as_raw_slice())
        .expect("File write failed.");
}

#[allow(clippy::expect_used)]
fn main() {
    let args = Args::parse();

    let encoder_config = args.config.map_or_else(config::Encoder::default, |path| {
        let conf_str = std::fs::read_to_string(path).expect("Config file read error.");
        toml::from_str(&conf_str).expect("Config file syntax error.")
    });

    let source =
        source::PreloadedSignal::from_path(&args.source).expect("Failed to load input source.");

    let block_size = encoder_config
        .fixed_block_size
        .expect("Variable block-size is not supported yet.");
    let stream = coding::encode_with_fixed_block_size(&encoder_config, source, block_size);

    if let Some(path) = args.dump_config {
        let mut file = File::create(path).expect("Failed to create a file.");
        file.write_all(toml::to_string(&encoder_config).unwrap().as_bytes())
            .expect("File write failed.");
    }

    let mut file = File::create(args.output).expect("Failed to create a file.");
    write_stream(&stream, &mut file);
}
