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

//! Abstract interface for bit-based output.

use num_traits::PrimInt;
use num_traits::ToBytes;

/// Alias trait for the bit-addressible integers.
pub trait PackedBits:
    ToBytes + From<u8> + Into<u64> + PrimInt + std::ops::ShlAssign<usize>
where
    Self::Bytes: AsRef<[u8]>,
{
    const PACKED_BITS: usize;
}

impl<T: ToBytes + From<u8> + Into<u64> + PrimInt + std::ops::ShlAssign<usize>> PackedBits for T
where
    <T as ToBytes>::Bytes: AsRef<[u8]>,
{
    /// The number of bits packed in this type. Synonym of `T::BITS`.
    const PACKED_BITS: usize = std::mem::size_of::<T>() * 8;
}

/// Storage-agnostic interface trait for bit-based output.
///
/// `BitSink` API is unstable and will be subjected to change in a minor
/// version up. Therefore, it is not recommended to derive this trait for
/// supporting new output types. Rather, it is recommended to use
/// `bitvec::BitVec` or `ByteVec` (in this module) and covert the contents to
/// a desired type.
pub trait BitSink: Sized {
    /// Puts zeros to `BitSink` until the length aligns to the byte boundaries.
    ///
    /// # Returns
    ///
    /// The number of zeros put.
    fn align_to_byte(&mut self) -> usize;

    /// Writes bytes after alignment, and returns padded bits.
    fn write_bytes_aligned(&mut self, bytes: &[u8]) -> usize {
        let ret = self.align_to_byte();
        for b in bytes {
            self.write_lsbs(*b, 8);
        }
        ret
    }

    /// Writes `n` LSBs to the sink.
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize);

    /// Writes `n` MSBs to the sink.
    fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize);

    /// Writes all bits in `val: PackedBits`.
    fn write<T: PackedBits>(&mut self, val: T);

    /// Writes `val` in two's coplement format.
    #[inline]
    fn write_twoc<T: Into<i64>>(&mut self, val: T, bits_per_sample: usize) {
        let val: i64 = val.into();
        let shifted = (val << (64 - bits_per_sample)) as u64;
        self.write_msbs(shifted, bits_per_sample);
    }
}

pub struct ByteVec {
    bytes: Vec<u8>,
    bitlength: usize,
}

impl Default for ByteVec {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteVec {
    /// Creates new `ByteVec` instance with the default capacity.
    pub fn new() -> Self {
        Self {
            bytes: vec![],
            bitlength: 0usize,
        }
    }

    /// Creates new `ByteVec` instance with the specified capacity (in bits).
    pub fn with_capacity(capacity_in_bits: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity_in_bits / 8 + 1),
            bitlength: 0usize,
        }
    }

    /// Clears the vector, removing all values.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.bitlength = 0;
    }

    /// Reseerves capacity for at least `additional_in_bits` more bits.
    pub fn reserve(&mut self, additional_in_bits: usize) {
        self.bytes.reserve(additional_in_bits / 8 + 1);
    }

    /// Returns the remaining number of bits in the last byte in `self.bytes`.
    #[inline]
    fn tail_len(&self) -> usize {
        let r = self.bitlength % 8;
        if r == 0 {
            0
        } else {
            8 - r
        }
    }

    /// Consumes `ByteVec` and returns the internal buffer.
    #[inline]
    pub fn bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns bits in a string for tests.
    #[cfg(test)]
    fn to_debug_bitstring(&self) -> String {
        let mut ret = String::new();
        for b in &self.bytes {
            ret.push_str(&format!("{b:08b}_"));
        }
        ret.pop();
        ret
    }

    pub fn as_byte_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl BitSink for ByteVec {
    #[inline]
    fn write<T: PackedBits>(&mut self, val: T) {
        let nbitlength = self.bitlength + 8 * std::mem::size_of::<T>();
        let tail = self.tail_len();
        if tail > 0 {
            self.write_msbs(val, tail);
        }
        let val = val << tail;
        let bytes: T::Bytes = val.to_be_bytes();
        self.bytes.extend_from_slice(bytes.as_ref());
        self.bitlength = nbitlength;
    }

    #[inline]
    fn align_to_byte(&mut self) -> usize {
        let r = self.tail_len();
        self.bitlength += r;
        r
    }

    #[inline]
    fn write_bytes_aligned(&mut self, bytes: &[u8]) -> usize {
        let ret = self.align_to_byte();
        self.bytes.extend_from_slice(bytes);
        self.bitlength += 8 * bytes.len();
        ret
    }

    #[inline]
    fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) {
        if n == 0 {
            return;
        }
        let mut val: T = val;
        let mut n = n;
        let nbitlength = self.bitlength + n;
        let r = self.tail_len();

        if r != 0 {
            let b = (val >> (T::PACKED_BITS - r)).to_u8().unwrap();
            let tail = self.bytes.len() - 1;
            self.bytes[tail] |= b;
            val <<= r;
            n = n.saturating_sub(r);
        }
        let bytes_to_write = n / 8;
        if bytes_to_write > 0 {
            let bytes = val.to_be_bytes();
            self.bytes
                .extend_from_slice(&bytes.as_ref()[0..bytes_to_write]);
            n %= 8;
        }
        if n > 0 {
            val <<= bytes_to_write * 8;
            let mask = !((1 << (8 - n)) - 1);
            let shifted = val >> (T::PACKED_BITS - 8) & mask.into();
            let tail_byte = shifted.to_u8().unwrap();
            self.bytes.push(tail_byte);
        }
        self.bitlength = nbitlength;
    }

    #[inline]
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) {
        if n == 0 {
            return;
        }
        self.write_msbs(val << (T::PACKED_BITS - n), n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bitvec::prelude::bits;
    use bitvec::prelude::BitOrder;
    use bitvec::prelude::BitStore;
    use bitvec::prelude::BitVec;
    use bitvec::prelude::Lsb0;
    use bitvec::prelude::Msb0;
    use bitvec::view::BitView;

    impl<T2, O2> BitSink for BitVec<T2, O2>
    where
        T2: BitStore,
        O2: BitOrder,
    {
        #[inline]
        fn align_to_byte(&mut self) -> usize {
            let npad = 8 - self.len() % 8;
            if npad == 8 {
                return 0;
            }
            self.write_lsbs(0u8, npad);
            npad
        }

        fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) {
            let val: u64 = val.into();
            self.extend_from_bitslice(&val.view_bits::<Msb0>()[64 - n..]);
        }

        fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) {
            let val: u64 = val.into();
            self.extend_from_bitslice(&val.view_bits::<Msb0>()[0..n]);
        }

        fn write<T: PackedBits>(&mut self, val: T) {
            self.write_lsbs(val, std::mem::size_of::<T>() * 8);
        }
    }

    #[test]
    fn align_to_byte_with_bitvec() {
        let mut sink: BitVec<u8> = BitVec::new();
        sink.write_lsbs(0x01u8, 1);
        sink.align_to_byte();
        assert_eq!(sink.len(), 8);
        sink.align_to_byte();
        assert_eq!(sink.len(), 8);
        sink.write_lsbs(0x01u8, 2);
        assert_eq!(sink.len(), 10);
        sink.align_to_byte();
        assert_eq!(sink.len(), 16);
    }

    #[test]
    fn twoc_writing() {
        let mut sink: BitVec<u8> = BitVec::new();
        sink.write_twoc(-7, 4);
        assert_eq!(sink, bits![1, 0, 0, 1]);
    }

    #[test]
    fn bytevec_write_msb() {
        let mut bv = ByteVec::new();
        bv.write_msbs(0xFFu8, 3);
        bv.write_msbs(0x0u64, 12);
        bv.write_msbs(0xFFFF_FFFFu32, 9);
        bv.write_msbs(0x0u16, 8);
        assert_eq!(
            bv.to_debug_bitstring(),
            "11100000_00000001_11111111_00000000"
        );
    }

    #[test]
    fn bytevec_write_lsb() {
        let mut bv = ByteVec::new();
        bv.write_lsbs(0xFFu8, 3);
        bv.write_lsbs(0x0u64, 12);
        bv.write_lsbs(0xFFFF_FFFFu32, 9);
        bv.write_lsbs(0x0u16, 8);
        assert_eq!(
            bv.to_debug_bitstring(),
            "11100000_00000001_11111111_00000000"
        );
    }
}
