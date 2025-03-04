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

#![allow(dead_code)]

// mimic clippy. This file acts as a part of the crate when checked by clippy.
// but it is outside of the crate when it is actually used by doctests.
#[cfg(clippy)]
use crate as flacenc;

use flacenc::component::Frame;
use flacenc::component::FrameHeader;
use flacenc::component::Stream;
use flacenc::config;
use flacenc::error::Verify;
use flacenc::source::MemSource;

/// Makes a `Stream` for doctest.
pub fn make_example_stream(
    signal_len: usize,
    block_size: usize,
    channels: usize,
    sample_rate: usize,
) -> Stream {
    let signal = vec![0i32; signal_len * channels];
    let bits_per_sample = 16;
    let source = MemSource::from_samples(&signal, channels, bits_per_sample, sample_rate);
    flacenc::encode_with_fixed_block_size(
        &config::Encoder::default()
            .into_verified()
            .expect("config value error"),
        source,
        block_size,
    )
    .expect("encoder error")
}

/// Makes a `Frame` for doctest.
pub fn make_example_frame(
    signal_len: usize,
    block_size: usize,
    channels: usize,
    sample_rate: usize,
) -> Frame {
    make_example_stream(signal_len, block_size, channels, sample_rate)
        .frame(0)
        .unwrap()
        .clone()
}

/// Makes a `Frame` for doctest.
pub fn make_example_frame_header(
    signal_len: usize,
    block_size: usize,
    channels: usize,
    sample_rate: usize,
) -> FrameHeader {
    make_example_frame(signal_len, block_size, channels, sample_rate)
        .header()
        .clone()
}
