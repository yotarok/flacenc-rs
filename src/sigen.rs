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

//! Test signal generator module.
//!
//! This module is primarily intended to be used for tests. However, unlike
//! a module in `test_helper.rs`, this module is intended to be exposed to the
//! outside of the crate for external testing frameworks
//! (specifically for cargo-fuzz).

use std::rc::Rc;
use std::sync::Arc;

use rand::Rng;
use rand::SeedableRng;

/// Test signal generators.
pub trait Signal: std::fmt::Debug {
    /// Generates a signal from t=`sample_offset` and fills the buffer `dest`.
    fn fill_buffer(&self, sample_offset: usize, dest: &mut [f32]);

    /// Generates a signal and returns `Vec` containing quantized ints.
    fn to_vec_quantized(&self, bits_per_sample: usize, block_size: usize) -> Vec<i32> {
        assert!(bits_per_sample <= 24);
        assert!(bits_per_sample > 4);
        // note that scalefactor below can make samples exceed iXX::MAX by 1.
        let scalefactor = 1usize << (bits_per_sample - 1);
        let min_target = -((1usize << (bits_per_sample - 1)) as i32);
        let max_target = (1usize << (bits_per_sample - 1)) as i32 - 1i32;

        let mut ret = vec![0i32; block_size];
        let mut buffer = vec![0.0f32; block_size];
        self.fill_buffer(0, &mut buffer);

        for (p, x) in ret.iter_mut().zip(buffer.iter()) {
            *p = (scalefactor as f32 * x)
                .round()
                .clamp(min_target as f32, max_target as f32) as i32;
        }
        ret
    }

    /// Decorate the signal generator with output clipping.
    fn clip(self) -> Clip<Self>
    where
        Self: Sized,
    {
        Clip::new(self)
    }

    /// Mixes noise
    fn noise(self, amplitude: f32) -> Mix<Self, Noise>
    where
        Self: Sized,
    {
        self.mix(Noise::new(amplitude))
    }

    /// Mixes noise
    fn noise_with_seed(self, seed0: u64, amplitude: f32) -> Mix<Self, Noise>
    where
        Self: Sized,
    {
        self.mix(Noise::with_seed(seed0, amplitude))
    }

    /// Mixes signal from the other generator
    fn mix<T: Signal + Sized>(self, other: T) -> Mix<Self, T>
    where
        Self: Sized,
    {
        Mix::new(1.0, self, 1.0, other)
    }

    /// Concats `other` signal after `offset_time` samples are generated.
    fn concat<T: Signal + Sized>(self, offset_time: usize, other: T) -> Switch<Self, T>
    where
        Self: Sized,
    {
        Switch::new(self, offset_time, other)
    }
}

macro_rules! impl_signal_for_pointers {
    ($pointertype:ident) => {
        impl<T: Signal + ?Sized> Signal for $pointertype<T> {
            fn fill_buffer(&self, sample_offset: usize, dest: &mut [f32]) {
                <$pointertype<T> as AsRef<T>>::as_ref(self).fill_buffer(sample_offset, dest);
            }
        }
    };
}

impl_signal_for_pointers!(Box);
impl_signal_for_pointers!(Rc);
impl_signal_for_pointers!(Arc);

/// Generator for constant signals.
#[derive(Clone, Debug)]
pub struct Dc {
    offset: f32,
}

impl Dc {
    /// Constructs new `Dc` signal.
    pub fn new(offset: f32) -> Self {
        Self { offset }
    }
}

impl Signal for Dc {
    fn fill_buffer(&self, _offset: usize, dest: &mut [f32]) {
        for p in dest {
            *p = self.offset;
        }
    }
}

/// Generator for a sinusoidal wave.
#[derive(Clone, Debug)]
pub struct Sine {
    period: usize,
    amplitude: f32,
    initial_phase: f32,
}

impl Sine {
    /// Constructs new sine wave signal with `period` and `amplitude`.
    pub fn new(period: usize, amplitude: f32) -> Self {
        let initial_phase = 0.0;
        Self {
            period,
            amplitude,
            initial_phase,
        }
    }

    pub fn with_initial_phase(period: usize, amplitude: f32, initial_phase: f32) -> Self {
        Self {
            period,
            amplitude,
            initial_phase,
        }
    }
}

impl Signal for Sine {
    fn fill_buffer(&self, offset: usize, dest: &mut [f32]) {
        let period = self.period as f32;
        for (t, p) in dest.iter_mut().enumerate() {
            let t = (t + offset) as f32;
            *p = self.amplitude
                * f32::sin(self.initial_phase + 2.0 * std::f32::consts::PI * t / period);
        }
    }
}

/// Generator for a uniform random white noise.
#[derive(Clone, Debug)]
pub struct Noise {
    seed0: u64,
    amplitude: f32,
}

impl Noise {
    /// Constructs new noise generator.
    pub fn new(amplitude: f32) -> Self {
        let seed0: u64 = rand::thread_rng().gen();
        Self { seed0, amplitude }
    }

    /// Constructs new noise generator with specifying a seed.
    pub fn with_seed(seed0: u64, amplitude: f32) -> Self {
        Self { seed0, amplitude }
    }
}

impl Signal for Noise {
    /// Fills buffer with the uniform random values.
    ///
    /// # Note
    ///
    /// This method doesn't ensure reproducibility if it is called in an
    /// arbitraly order, e.g.
    /// `noise.fill_buffer(0, &mut dest[..])` generate different results from
    /// `noise.fill_buffer(0, &mut dest[0..10])` and
    /// `noise.fill_buffer(10, &mut dest[10..])`.
    fn fill_buffer(&self, offset: usize, dest: &mut [f32]) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed0.wrapping_add(offset as u64));
        for p in dest {
            *p = self.amplitude * 2.0 * (rng.sample::<f32, _>(rand::distributions::Open01) - 0.5);
        }
    }
}

/// Decorator that mixes outputs from the inner generators.
#[derive(Clone, Debug)]
pub struct Mix<T1: Signal + Sized, T2: Signal + Sized> {
    weight1: f32,
    weight2: f32,
    signal1: T1,
    signal2: T2,
}

impl<T1: Signal + Sized, T2: Signal + Sized> Mix<T1, T2> {
    /// Constructs new two-inputs mixer.
    pub fn new(weight1: f32, signal1: T1, weight2: f32, signal2: T2) -> Self {
        Self {
            weight1,
            weight2,
            signal1,
            signal2,
        }
    }
}

impl<T1: Signal + Sized, T2: Signal + Sized> Signal for Mix<T1, T2> {
    fn fill_buffer(&self, offset: usize, dest: &mut [f32]) {
        for p in &mut *dest {
            *p = 0.0f32;
        }

        let mut buf = vec![0.0f32; dest.len()];
        self.signal1.fill_buffer(offset, &mut buf);
        for (p, x) in dest.iter_mut().zip(buf.iter()) {
            *p += self.weight1 * *x;
        }
        self.signal2.fill_buffer(offset, &mut buf);
        for (p, x) in dest.iter_mut().zip(buf.iter()) {
            *p += self.weight2 * *x;
        }
    }
}

/// Decorator that clips the output of the inner generator.
#[derive(Clone, Debug)]
pub struct Clip<T: Signal + Sized> {
    inner: T,
    min: f32,
    max: f32,
}

impl<T: Signal + Sized> Clip<T> {
    /// Constructs a clipper.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            min: -1.0,
            max: 1.0,
        }
    }
}

impl<T: Signal + Sized> Signal for Clip<T> {
    fn fill_buffer(&self, offset: usize, dest: &mut [f32]) {
        self.inner.fill_buffer(offset, dest);
        for p in dest {
            *p = p.clamp(self.min, self.max);
        }
    }
}

/// Decorator that switches multiple generatros depending on the timestamp.
#[derive(Clone, Debug)]
pub struct Switch<T1: Signal + Sized, T2: Signal + Sized> {
    input1: T1,
    offset: usize,
    input2: T2,
}

impl<T1: Signal + Sized, T2: Signal + Sized> Switch<T1, T2> {
    /// Cosntructs a switcher.
    pub fn new(input1: T1, offset: usize, input2: T2) -> Self {
        Self {
            input1,
            offset,
            input2,
        }
    }
}

impl<T1: Signal + Sized, T2: Signal + Sized> Signal for Switch<T1, T2> {
    fn fill_buffer(&self, offset: usize, dest: &mut [f32]) {
        // not very efficient, but let's keep it simple.
        // use `input1` to fill the entire buffer
        self.input1.fill_buffer(offset, dest);
        // and then overwrite the later part of the signal.
        self.input2
            .fill_buffer(offset + self.offset, &mut dest[self.offset..]);
    }
}
