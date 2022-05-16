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

use std::collections::BTreeMap;
use std::f32::consts::PI;

use once_cell::sync::Lazy;
use rand::distributions::Distribution;
use rand::distributions::Uniform;

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

/// Generates a test signal with sinusoid and uniform-white noise.
#[allow(dead_code)]
pub fn sinusoid_plus_noise(
    block_size: usize,
    period: usize,
    amplitude: f32,
    noise_width: i32,
) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    let period = period as f32;
    let die = Uniform::from(-noise_width..=noise_width);
    let mut ret = Vec::new();
    for t in 0..block_size {
        let sin = (amplitude * (2.0 * (t as f32) * PI / period).sin()) as i32;
        ret.push(sin + die.sample(&mut rng));
    }
    ret
}

/// Generates DC signal with constant offset and random uniform-white noise.
#[allow(dead_code)]
pub fn constant_plus_noise(block_size: usize, dc_offset: i32, noise_width: i32) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(-noise_width..=noise_width);
    let mut ret = Vec::new();
    for _t in 0..block_size {
        ret.push(dc_offset + die.sample(&mut rng));
    }
    ret
}

#[allow(clippy::similar_names)]
fn read_le16(src: &[u8]) -> Vec<i16> {
    assert!(src.len() % 2 == 0);
    let mut ret = Vec::with_capacity(src.len() / 2);
    for bs in src.chunks(2) {
        let (lsbs, msbs) = (bs[0], bs[1]);
        let u: u16 = u16::from(lsbs) + u16::from(msbs) * 0x100u16;
        ret.push(u as i16);
    }
    ret
}

static TEST_SIGNALS: Lazy<BTreeMap<(&str, usize), Vec<i16>>> = Lazy::new(|| {
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

#[allow(dead_code)]
pub fn test_signal(src: &str, ch: usize) -> Vec<i32> {
    TEST_SIGNALS
        .get(&(src, ch))
        .expect("Specified test signal not found.")
        .iter()
        .copied()
        .map(i32::from)
        .collect()
}
