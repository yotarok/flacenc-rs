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

#![allow(clippy::missing_panics_doc)]

use std::collections::BTreeMap;
use std::io::Write;

use once_cell::sync::Lazy;
use tempfile::NamedTempFile;

use super::arrayutils::le_bytes_to_i32s;
use super::bitsink::ByteSink;
use super::component::BitRepr;
use super::component::Stream;
use super::error::Verify;
use super::source::MemSource;
use super::source::Seekable;
use super::source::Source;

#[macro_export]
macro_rules! assert_close {
    ($actual:expr, $expected:expr, rtol = $rtol:expr, atol = $atol:expr) => {{
        let err = ($actual - $expected).abs();
        #[allow(clippy::suboptimal_flops)]
        let tol = $rtol * ($expected).abs() + $atol;
        assert!(err < tol);
    }};
    ($actual:expr, $expected:expr) => {{
        assert_close!($actual, $expected, rtol = 0.00001, atol = 0.00001);
    }};
}

#[macro_export]
macro_rules! assert_finite {
    ($result:expr) => {{
        for (i, &value) in $result.iter().enumerate() {
            assert!(
                value.is_normal() || value == 0.0,
                "{}-th element in a vector is not finite ({}), x={:?}.",
                i,
                value,
                $result
            );
        }
    }};
}

#[allow(clippy::similar_names)]
fn read_le16(src: &[u8]) -> Vec<i32> {
    assert!(src.len() % 2 == 0);
    let mut ret = vec![0i32; src.len() / 2];
    le_bytes_to_i32s(src, &mut ret, 2);
    ret
}

static TEST_SIGNALS: Lazy<BTreeMap<(&str, usize), Vec<i32>>> = Lazy::new(|| {
    BTreeMap::from([
        (
            ("sus109", 0),
            read_le16(include_bytes!("resource/testsignal.sus109.ch0.bin")),
        ),
        (
            ("sus109", 1),
            read_le16(include_bytes!("resource/testsignal.sus109.ch1.bin")),
        ),
        (
            ("sus6", 0),
            read_le16(include_bytes!("resource/testsignal.sus6.ch0.bin")),
        ),
        (
            ("sus6", 1),
            read_le16(include_bytes!("resource/testsignal.sus6.ch1.bin")),
        ),
        (
            ("ras22", 0),
            read_le16(include_bytes!("resource/testsignal.ras22.ch0.bin")),
        ),
        (
            ("ras22", 1),
            read_le16(include_bytes!("resource/testsignal.ras22.ch1.bin")),
        ),
        (
            ("ras103", 0),
            read_le16(include_bytes!("resource/testsignal.ras103.ch0.bin")),
        ),
        (
            ("ras103", 1),
            read_le16(include_bytes!("resource/testsignal.ras103.ch1.bin")),
        ),
    ])
});

/// Loads monoral test signal by key and the channel specifier.
#[allow(dead_code)]
pub fn test_signal(src: &str, ch: usize) -> Vec<i32> {
    TEST_SIGNALS
        .get(&(src, ch))
        .expect("Specified test signal not found.")
        .clone()
}

/// Runs an integrity test over the given encoding function.
///
/// This runs `encoder` function followed by `claxon`-based FLAC decoding, and
/// compares the waveforms of the original signal and reconstructed signal.
pub fn integrity_test<Enc>(encoder: Enc, src: &MemSource) -> Stream
where
    Enc: Fn(MemSource) -> Stream,
{
    let stream = encoder(src.clone());

    assert!(stream.verify().is_ok());

    let mut file = NamedTempFile::new().expect("Failed to create temp file.");

    let mut bv: ByteSink = ByteSink::with_capacity(stream.count_bits());
    stream.write(&mut bv).expect("Bitstream formatting failed.");
    file.write_all(bv.as_slice()).expect("File write failed.");

    let flac_path = file.into_temp_path();

    let mut reader = claxon::FlacReader::open(&flac_path).unwrap();
    let streaminfo = reader.streaminfo();
    assert_eq!(streaminfo.channels as usize, src.channels());
    assert_eq!(streaminfo.sample_rate as usize, src.sample_rate());
    assert_eq!(streaminfo.bits_per_sample as usize, src.bits_per_sample());
    assert_eq!(streaminfo.samples, Some(src.len() as u64));

    let mut offsets: Vec<usize> = vec![];
    let mut offset = 0;
    for frame in stream.frames() {
        offsets.push(offset);
        offset += frame.header().block_size();
    }
    offsets.push(offset);

    let channels = streaminfo.channels as usize;
    let loaded = reader.samples().map(Result::unwrap).collect::<Vec<i32>>();
    let mut head = 0;
    for t in 0..src.len() {
        if t >= offsets[head] {
            head += 1;
        }
        let current_offset = offsets[head - 1];

        for ch in 0..channels {
            assert_eq!(
                loaded.as_slice()[t * channels + ch],
                src.as_slice()[t * channels + ch],
                "Failed at t={} of ch={} (block={}, in-block-t={})\n{:?}",
                t,
                ch,
                head,
                t - current_offset,
                stream.frame(head).unwrap().subframe(ch).unwrap()
            );
        }
    }
    stream
}
