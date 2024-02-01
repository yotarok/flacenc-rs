// Copyright 2024 Google LLC
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

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::Error;
use crate::error::FormatError;

/// Trait for defining an input FLAC bitstream.
pub trait BitSource {
    /// Returns the offset of the seek head in the number of bits.
    fn current_bit_offset(&self) -> usize;
    /// Skips to the next byte boundary.
    fn skip_to_next_byte(&mut self);
    /// Read bytes from the next byte boundary.
    fn read_bytes_aligned(&mut self, bytes: usize, buf: &mut [u8]) -> Result<(), Error>;
    /// Read unsigned integer from the next `bits` bits.
    fn read_u64(&mut self, bits: usize) -> Result<u64, Error>;
    /// Read signed integer from the next `bits` bits.
    fn read_i64(&mut self, bits: usize) -> Result<i64, Error> {
        if bits > 63 || bits == 0 {
            return Err(FormatError::new(self.current_bit_offset(), "unsupported bps").into());
        }
        let u = self.read_u64(bits)?;
        let msb = 1u64 << (bits - 1);
        let offset = (1u64 << bits) as i64;
        let x = if u >= msb {
            u as i64 - offset
        } else {
            u as i64
        };
        Ok(x)
    }
    /// Utility function that calls `read_bytes_aligned` and returns results in `Vec`.
    fn read_bytevec_aligned(&mut self, bytes: usize) -> Result<Vec<u8>, Error> {
        let mut ret = vec![0; bytes];
        self.read_bytes_aligned(bytes, &mut ret)?;
        Ok(ret)
    }
    /// Utility function that read UTF-8-like encoded integer.
    fn read_utf8_aligned(&mut self) -> Result<u64, Error> {
        let mut head = [0u8; 1];
        self.read_bytes_aligned(1, &mut head)?;
        let head = u64::from(head[0]);

        let (tail_count, acc) = if head < 128 {
            (0, head & 0x7F)
        } else if head < 0xE0 {
            (1, head & 0x1F)
        } else if head < 0xF0 {
            (2, head & 0x0F)
        } else if head < 0xF8 {
            (3, head & 0x07)
        } else if head < 0xFC {
            (4, head & 0x03)
        } else if head < 0xFE {
            (5, head & 0x01)
        } else if head == 0xFE {
            (6, 0)
        } else {
            return Err(FormatError::new(self.current_bit_offset(), "invalid UTF-8").into());
        };

        let mut tails = [0u8; 6];
        self.read_bytes_aligned(tail_count, &mut tails)?;

        let mut acc: u64 = acc;
        for b in &tails[..tail_count] {
            acc = acc << 6 | u64::from(*b & 0x3F);
        }
        Ok(acc)
    }
    /// Reads unary code (i.e. counts the number of leading "1"s before "0").
    fn read_unary_code(&mut self) -> Result<u64, Error> {
        // TODO: This is very inefficient. However, since the decoder is
        // currently only for testing/ analysis purposes and not optimized,
        // it should be acceptable.
        let mut ret = 0;
        let mut b = self.read_u64(1)? != 0;
        while !b {
            ret += 1;
            b = self.read_u64(1)? != 0;
        }
        Ok(ret)
    }
}

/// `BitSource` that reads a bitstream from the preloaded `Vec<u8>`.
pub struct MemSource {
    values: Vec<u8>,
    head_bytes: usize,
    head_bits: usize,
}

impl MemSource {
    /// Reads `MemSource` from `path`.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let mut f = File::open(path)?;
        let mut ret = Self {
            values: vec![],
            head_bytes: 0,
            head_bits: 0,
        };
        f.read_to_end(&mut ret.values)?;
        Ok(ret)
    }
}

impl BitSource for MemSource {
    fn current_bit_offset(&self) -> usize {
        self.head_bytes * 8 + self.head_bits
    }

    fn skip_to_next_byte(&mut self) {
        if self.head_bits != 0 {
            self.head_bits = 0;
            self.head_bytes += 1;
        }
    }

    fn read_bytes_aligned(&mut self, bytes: usize, buf: &mut [u8]) -> Result<(), Error> {
        self.skip_to_next_byte();
        assert!(buf.len() >= bytes);
        if self.values.len() < self.head_bytes + bytes {
            return Err(Error::stream_ended());
        }
        buf[..bytes].copy_from_slice(&self.values[self.head_bytes..self.head_bytes + bytes]);
        self.head_bytes += bytes;
        Ok(())
    }

    fn read_u64(&mut self, bits: usize) -> Result<u64, Error> {
        let mut ret = 0u64;
        let mut nread = 0;

        if self.head_bits > 0 {
            let tail_bits = 8 - self.head_bits;
            ret = u64::from(self.values[self.head_bytes] & ((1 << tail_bits) - 1));
            nread = tail_bits;
            self.head_bits = 0;
            self.head_bytes += 1;
        }

        while nread < bits {
            if self.head_bytes >= self.values.len() {
                return Err(Error::stream_ended());
            }
            ret = (ret << 8) | u64::from(self.values[self.head_bytes]);
            nread += 8;
            self.head_bytes += 1;
        }

        if nread > bits {
            // unread the last byte
            let n = nread - bits;
            assert!(n < 8);
            ret >>= n;
            self.head_bytes -= 1;
            self.head_bits = 8 - n;
        }
        Ok(ret)
    }
}
