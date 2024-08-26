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
mod verify;

pub use bitrepr::*;
pub use datatype::*;
#[cfg(any(test, feature = "decode"))]
pub use decode::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitsink::MemSink;
    use crate::error::OutputError;
    use crate::error::RangeError;
    use crate::error::Verify;
    use crate::sigen;
    use crate::sigen::Signal;
    use crate::test_helper::make_random_residual;
    use crate::test_helper::make_verbatim_frame;

    #[test]
    fn write_empty_stream() -> Result<(), OutputError<MemSink<u8>>> {
        let stream = Stream::new(44100, 2, 16).unwrap();
        let mut bv: MemSink<u8> = MemSink::new();
        stream.write(&mut bv)?;
        assert_eq!(
            bv.len(),
            32 // fLaC
      + 1 + 7 + 24 // METADATA_BLOCK_HEADER
      + 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128 // METADATA_BLOCK_STREAMINFO
        );
        assert_eq!(stream.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn write_stream_info() -> Result<(), OutputError<MemSink<u8>>> {
        let stream_info = StreamInfo::new(44100, 2, 16).unwrap();
        let mut bv: MemSink<u8> = MemSink::new();
        stream_info.write(&mut bv)?;
        assert_eq!(bv.len(), 16 + 16 + 24 + 24 + 20 + 3 + 5 + 36 + 128);
        assert_eq!(stream_info.count_bits(), bv.len());
        Ok(())
    }

    #[test]
    fn write_frame_header() -> Result<(), OutputError<MemSink<u8>>> {
        let header = FrameHeader::new(2304, ChannelAssignment::Independent(2), 192);
        let mut bv: MemSink<u8> = MemSink::new();
        header.write(&mut bv)?;

        // test with canonical frame
        let header = FrameHeader::new(192, ChannelAssignment::Independent(2), 0);
        let mut bv: MemSink<u8> = MemSink::new();
        header.write(&mut bv)?;

        assert_eq!(
            bv.to_bitstring(),
            concat!(
                "11111111_111110", // sync
                "01_",             // reserved/ blocking strategy (const in this impl)
                "00010000_",       // block size/ sample_rate (0=header)
                "00010000_",       // channel/ bps (0=header)/ reserved
                "00000000_",       // sample number
                "01101001",        // crc8
            )
        );

        assert_eq!(header.count_bits(), bv.len());

        Ok(())
    }

    #[test]
    fn write_verbatim_frame() -> Result<(), OutputError<MemSink<u64>>> {
        let nchannels: usize = 3;
        let nsamples: usize = 17;
        let bits_per_sample: usize = 16;
        let stream_info = StreamInfo::new(16000, nchannels, bits_per_sample).unwrap();
        let framebuf = vec![-1i32; nsamples * nchannels];
        let frame = make_verbatim_frame(&stream_info, &framebuf, 0);
        let mut bv: MemSink<u64> = MemSink::new();

        frame.header().write(&mut bv)?;
        assert_eq!(frame.header().count_bits(), bv.len());

        for ch in 0..3 {
            bv.clear();
            frame.subframe(ch).unwrap().write(&mut bv)?;
            assert_eq!(frame.subframe(ch).unwrap().count_bits(), bv.len());
        }

        bv.clear();
        frame.write(&mut bv)?;
        assert_eq!(frame.count_bits(), bv.len());
        Ok(())
    }

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
    fn block_size_encoding() {
        let (head, _foot, footsize) = block_size_spec(192);
        assert_eq!(head, 0x01);
        assert_eq!(footsize, 0);

        let (head, _foot, footsize) = block_size_spec(2048);
        assert_eq!(head, 0x0B);
        assert_eq!(footsize, 0);

        let (head, _foot, footsize) = block_size_spec(1152);
        assert_eq!(head, 0x03);
        assert_eq!(footsize, 0);

        let (head, foot, footsize) = block_size_spec(193);
        assert_eq!(head, 0x06);
        assert_eq!(footsize, 8);
        assert_eq!(foot, 0xC0);

        let (head, foot, footsize) = block_size_spec(1151);
        assert_eq!(head, 0x07);
        assert_eq!(footsize, 16);
        assert_eq!(foot, 0x047E);
    }

    #[test]
    fn channel_assignment_encoding() -> Result<(), OutputError<MemSink<u8>>> {
        let ch = ChannelAssignment::Independent(8);
        let mut bv: MemSink<u8> = MemSink::new();
        ch.write(&mut bv)?;
        assert_eq!(bv.to_bitstring(), "0111****");
        let ch = ChannelAssignment::RightSide;
        let mut bv: MemSink<u8> = MemSink::new();
        ch.write(&mut bv)?;
        assert_eq!(bv.to_bitstring(), "1001****");
        assert_eq!(ch.count_bits(), bv.len());
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
    #[allow(clippy::cast_lossless)]
    fn bit_count_residual() -> Result<(), OutputError<MemSink<u64>>> {
        let residual = make_random_residual(rand::thread_rng(), 0);
        residual
            .verify()
            .expect("should construct a valid Residual");

        let mut bv: MemSink<u64> = MemSink::new();
        residual.write(&mut bv)?;

        assert_eq!(residual.count_bits(), bv.len());
        Ok(())
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

    #[test]
    fn channel_assignment_is_small_enough() {
        let size = std::mem::size_of::<ChannelAssignment>();
        assert_eq!(size, 2);
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
