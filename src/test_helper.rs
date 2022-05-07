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

use std::f32::consts::PI;

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

pub fn constant_plus_noise(block_size: usize, dc_offset: i32, noise_width: i32) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(-noise_width..=noise_width);
    let mut ret = Vec::new();
    for _t in 0..block_size {
        ret.push(dc_offset + die.sample(&mut rng));
    }
    ret
}
