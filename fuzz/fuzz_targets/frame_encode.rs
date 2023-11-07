#![no_main]

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

use flacenc::component;
use flacenc::config;
use flacenc::constant;
use flacenc::encode_fixed_size_frame;
use flacenc::error::Verify;
use flacenc::sigen;
use flacenc::sigen::Signal;
use flacenc::source::Fill;
use flacenc::source::FrameBuf;

fn arbitrary_signal(
    u: &mut Unstructured,
    block_size: usize,
) -> Result<Box<dyn Signal>, arbitrary::Error> {
    let do_clip = bool::arbitrary(u)?;
    let unclipped: Box<dyn Signal> = match u.int_in_range(0..=4usize)? {
        0 => {
            // Dc
            let amplitude = u32::arbitrary(u)? as f32 / u32::MAX as f32;
            Box::new(sigen::Dc::new(amplitude))
        }
        1 => {
            // Noise
            let amplitude = u32::arbitrary(u)? as f32 / u32::MAX as f32;
            let seed = u64::arbitrary(u)?;
            Box::new(sigen::Noise::with_seed(seed, amplitude))
        }
        2 => {
            // Sin
            let amplitude = u32::arbitrary(u)? as f32 / u32::MAX as f32;
            let phase = (u32::arbitrary(u)? as f32 / u32::MAX as f32) * 2.0 * std::f32::consts::PI;
            let period_fraction = u32::arbitrary(u)? as f32 / u32::MAX as f32;
            let period: usize = (block_size as f32 * period_fraction) as usize;
            Box::new(sigen::Sine::with_initial_phase(period, amplitude, phase))
        }
        3 => {
            // Mix
            let mix_fraction = u32::arbitrary(u)? as f32 / u32::MAX as f32;
            let signal1 = arbitrary_signal(u, block_size)?;
            let signal2 = arbitrary_signal(u, block_size)?;
            Box::new(sigen::Mix::new(
                mix_fraction,
                signal1,
                1.0 - mix_fraction,
                signal2,
            ))
        }
        4 => {
            // Switch
            let time_fraction = u32::arbitrary(u)? as f32 / u32::MAX as f32;
            let time = (block_size as f32 * time_fraction) as usize;
            let signal1 = arbitrary_signal(u, time)?;
            let signal2 = arbitrary_signal(u, block_size - time)?;
            Box::new(sigen::Switch::new(signal1, time, signal2))
        }
        _ => {
            unreachable!();
        }
    };
    if do_clip {
        Ok(Box::new(unclipped.clip()))
    } else {
        Ok(unclipped)
    }
}

#[derive(Debug)]
struct Input {
    channel_count: usize,
    block_size: usize,
    sample_rate: usize,
    bits_per_sample: usize,
    signals: Vec<Box<dyn Signal>>,
}

impl Input {
    pub fn framebuf(&self) -> FrameBuf {
        let sample_count = self.channel_count * self.block_size;
        let mut buffer = vec![0i32; sample_count];
        for (ch, sig) in self.signals.iter().enumerate() {
            for (t, x) in sig
                .to_vec_quantized(self.bits_per_sample, self.block_size)
                .into_iter()
                .enumerate()
            {
                buffer[t * self.channel_count + ch] = x;
            }
        }

        let mut fb = FrameBuf::with_size(self.channel_count, self.block_size);
        fb.fill_interleaved(&buffer).unwrap();
        fb
    }

    pub fn stream_info(&self) -> component::StreamInfo {
        component::StreamInfo::new(self.sample_rate, self.channel_count, self.bits_per_sample)
    }
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self, arbitrary::Error> {
        let channel_count = u.int_in_range(1usize..=constant::MAX_CHANNELS)?;
        let block_size = u.int_in_range(constant::MIN_BLOCKSIZE..=constant::MAX_BLOCKSIZE)?;
        let sample_rate = u.int_in_range(1usize..=192_000)?;
        let bits_per_sample = (u.int_in_range(2u8..=6u8)? * 4) as usize;

        let mut signals = vec![];
        for _ch in 0..channel_count {
            signals.push(arbitrary_signal(u, block_size)?);
        }

        Ok(Self {
            channel_count,
            block_size,
            sample_rate,
            bits_per_sample,
            signals,
        })
    }
}

fuzz_target!(|input: Input| {
    let fb = input.framebuf();
    match encode_fixed_size_frame(&config::Encoder::default(), &fb, 0, &input.stream_info()) {
        Ok(frame) => {
            frame.verify().unwrap();
        }
        Err(_) => {
            // currently all errors are ignorable
            // (either a source error or a config error)
        }
    }
});
