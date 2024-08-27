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

//! Components to be written in the output file.

mod bitrepr;
mod datatype;
#[cfg(any(test, feature = "decode"))]
mod decode;
#[cfg(any(test, feature = "decode"))]
pub mod parser;
mod verify;

pub use bitrepr::BitRepr;
pub(crate) use datatype::BlockSizeSpec;
pub use datatype::ChannelAssignment;
pub use datatype::Constant;
pub use datatype::FixedLpc;
pub use datatype::Frame;
pub use datatype::FrameHeader;
pub use datatype::FrameOffset;
pub use datatype::Lpc;
pub use datatype::MetadataBlock;
pub use datatype::MetadataBlockData;
pub use datatype::QuantizedParameters;
pub use datatype::Residual;
pub(crate) use datatype::SampleRateSpec;
pub(crate) use datatype::SampleSizeSpec;
pub use datatype::Stream;
pub use datatype::StreamInfo;
pub use datatype::SubFrame;
pub use datatype::Verbatim;
#[cfg(any(test, feature = "decode"))]
pub use decode::Decode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitsink::MemSink;
    use crate::component::bitrepr::encode_to_utf8like;
    use crate::error::OutputError;
    use crate::error::RangeError;
    use crate::sigen;
    use crate::sigen::Signal;
    use crate::test_helper::make_verbatim_frame;

    #[test]
    fn utf8_encoding() -> Result<(), RangeError> {
        let v = 0x56;
        let bs = encode_to_utf8like(v)?;
        assert_eq!(bs, &[0x56]);

        let v = 0x1024;
        let bs = encode_to_utf8like(v)?;
        assert_eq!(bs, &[0xE1, 0x80, 0xA4]);

        let v = 0xF_FFFF_FFFFu64; // 36 bits of ones
        let bs = encode_to_utf8like(v)?;
        assert_eq!(bs, &[0xFE, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]);

        let v = 0x10_0000_0000u64; //  out of domain
        encode_to_utf8like(v).expect_err("Should be out of domain");

        Ok(())
    }

    #[test]
    fn stream_info_update() {
        let mut stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let framebuf = sigen::Dc::new(0.01)
            .noise(0.002)
            .to_vec_quantized(16, 256 * 2);
        let frame1 = make_verbatim_frame(&stream_info, &framebuf, 0);
        stream_info.update_frame_info(&frame1);
        let framebuf = sigen::Dc::new(0.02)
            .noise(0.1)
            .to_vec_quantized(16, 192 * 2);
        let frame2 = make_verbatim_frame(&stream_info, &framebuf, 256);
        stream_info.update_frame_info(&frame2);

        assert_eq!(stream_info.min_block_size(), 192);
        assert_eq!(stream_info.max_block_size(), 256);

        // header_size = 5 + sample_number + block_size + sample_rate
        // sample_rate = 0 (in this implementation)
        // block_size = 0 (since preset sizes (192 and 256) are used)
        // sample_number = 1 for the first frame, 2 for the second frame
        // footer_size = 2
        // verbatim_subframe_size = 2 * block_size * 2 + subframe_header_size
        // subframe_header_size = 1
        // so overall:
        // first_frame_size = 5 + 1 + 0 + 0 + 2 + 1 + 2 * 256 * 2 + 1 = 1034
        // second_frame_size = 5 + 2 + 0 + 0 + 2 + 1 + 2 * 192 * 2 + 1 = 779
        assert_eq!(stream_info.min_frame_size(), 779);
        assert_eq!(stream_info.max_frame_size(), 1034);
    }

    #[test]
    fn frame_bitstream_precomputataion() -> Result<(), OutputError<MemSink<u64>>> {
        let stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let samples = sigen::Sine::new(128, 0.2)
            .noise(0.1)
            .to_vec_quantized(12, 512);
        let mut frame = make_verbatim_frame(&stream_info, &samples, 0);
        let mut bv_ref: MemSink<u64> = MemSink::new();
        let frame_cloned = frame.clone();
        frame_cloned.write(&mut bv_ref)?;
        assert!(bv_ref.len() % 8 == 0); // frame must be byte-aligned.

        frame.precompute_bitstream();
        assert!(frame.is_bitstream_precomputed());
        assert!(!frame_cloned.is_bitstream_precomputed());

        let mut bv: MemSink<u64> = MemSink::new();
        frame.write(&mut bv)?;
        assert_eq!(bv.to_bitstring(), bv_ref.to_bitstring());

        // this makes `Frame` broken as the header says it has two channels.
        frame.add_subframe(frame.subframe(0).unwrap().clone());
        // anyway cache should be discarded.
        assert!(!frame.is_bitstream_precomputed());
        Ok(())
    }
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;

    use crate::bitsink::MemSink;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    fn residual_write_to_u64s(b: &mut Bencher) {
        let warmup_len = 13;
        let mut quotients = [2u32; 4096];
        let mut remainders = [0u32; 4096];
        for t in 0..warmup_len {
            quotients[t] = 0u32;
            remainders[t] = 0u32;
        }
        let residual = Residual::new(8, 4096, warmup_len, &[8u8; 256], &quotients, &remainders)
            .expect("Residual construction failed.");
        let mut sink = MemSink::<u64>::with_capacity(4096 * 2 * 8);

        b.iter(|| {
            sink.clear();
            residual.write(black_box(&mut sink))
        });
    }

    #[bench]
    fn residual_bit_counter(b: &mut Bencher) {
        let warmup_len = 13;
        let mut quotients = [2u32; 4096];
        let mut remainders = [0u32; 4096];
        for t in 0..warmup_len {
            quotients[t] = 0u32;
            remainders[t] = 0u32;
        }
        let residual = Residual::new(8, 4096, warmup_len, &[8u8; 256], &quotients, &remainders)
            .expect("Residual construction failed.");

        b.iter(|| black_box(&residual).count_bits());
    }
}
