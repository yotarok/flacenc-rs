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

#![feature(portable_simd)]
// Note that clippy attributes should be in sync with those declared in "main.rs"
#![warn(clippy::all, clippy::nursery, clippy::pedantic, clippy::cargo)]
// Some of clippy::pedantic rules are actually useful, so use it with a lot of
// ad-hoc exceptions.
#![allow(
    // Reactivate "use_self" once false-positive issue is gone.
    // https://github.com/rust-lang/rust-clippy/issues/6902
    clippy::use_self,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::multiple_crate_versions,
    clippy::must_use_candidate
)]

pub mod bitsink;
pub mod coding;
pub mod component;
pub mod config;
pub mod constant;
pub mod error;
pub mod lpc;
pub mod rice;
pub mod source;

#[cfg(test)]
mod test_helper;

#[cfg(test)]
mod test {
    // end-to-end, but transparent test.
    use super::*;
    use component::BitRepr;
    use source::Seekable;
    use source::Source;

    use std::io::Write;

    use bitvec::prelude::BitVec;
    use bitvec::prelude::Msb0;
    use rstest::rstest;
    use tempfile::NamedTempFile;

    #[rstest]
    fn e2e_with_generated_sinusoids(
        #[values(1, 2, 3, 5, 8)]
        channels: usize,
        #[values(512, 1024)]
        block_size: usize,
    ) {
        let signal_len = 16123;
        let bits_per_sample = 16;
        let sample_rate = 16000;

        let mut channel_signals = vec![];
        for _ch in 0..channels {
            channel_signals.push(test_helper::sinusoid_plus_noise(
                signal_len, 36, 10000.0, 123,
            ));
        }

        let mut signal = vec![];
        for t in 0..signal_len {
            for s in &channel_signals {
                signal.push(s[t]);
            }
        }
        let source =
            source::PreloadedSignal::from_samples(&signal, channels, bits_per_sample, sample_rate);

        let stream =
            coding::encode_with_fixed_block_size(&config::Encoder::default(), source, block_size)
                .expect("Source error");

        let bits = stream.count_bits();
        let comp_rate = bits as f64 / (signal.len() * bits_per_sample) as f64;
        eprintln!(
            "Compressed {} samples ({} bits) to {} bits. Compression Rate = {}",
            signal.len(),
            signal.len() * bits_per_sample,
            bits,
            comp_rate
        );

        // It should be okay to assume that compression rates for sinusoidal
        // signal are always lower than 0.8 :)
        assert!(comp_rate < 0.8);

        let mut file = NamedTempFile::new().expect("Failed to create temp file.");
        let mut bv: BitVec<u8, Msb0> = BitVec::with_capacity(stream.count_bits());
        stream.write(&mut bv).expect("Bitstream formatting failed.");
        file.write_all(bv.as_raw_slice())
            .expect("File write failed.");

        let flac_path = file.into_temp_path();

        // Use sndfile (i.e. libflac) decoder for integrity check.
        let loaded = source::PreloadedSignal::from_path(&flac_path).expect("Failed to decode.");
        assert_eq!(loaded.channels(), channels);
        assert_eq!(loaded.sample_rate(), sample_rate);
        assert_eq!(loaded.bits_per_sample(), bits_per_sample);
        assert_eq!(loaded.len(), signal_len);

        for t in 0..signal_len {
            for ch in 0..channels {
                assert_eq!(
                    loaded.as_raw_slice()[t * channels + ch],
                    signal[t * channels + ch],
                    "Failed at t={} of ch={} (block={}, in-block-t={})\n{:?}",
                    t,
                    ch,
                    t / block_size,
                    t % block_size,
                    stream.frame(t / block_size).subframe(ch)
                );
            }
        }
    }
}
