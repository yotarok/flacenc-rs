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

use bitvec::prelude::BitOrder;
use bitvec::prelude::BitSlice;
use bitvec::prelude::BitStore;
use bitvec::prelude::BitVec;
use bitvec::prelude::Msb0;
use bitvec::view::BitView;

/// Storage-agnostic interface trait for bit-based output.
pub trait BitSink: Sized {
    fn write_bitslice<T: BitStore, O: BitOrder>(&mut self, other: &BitSlice<T, O>);

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

    // Type signature may change. Don't override.
    #[inline]
    fn write_lsbs<T: Into<u64>>(&mut self, val: T, nbits: usize) {
        let val: u64 = val.into();
        self.write_bitslice(&val.view_bits::<Msb0>()[64 - nbits..]);
    }

    #[inline]
    fn write_msbs<T: Into<u64>>(&mut self, val: T, nbits: usize) {
        let val: u64 = val.into();
        self.write_bitslice(&val.view_bits::<Msb0>()[0..nbits]);
    }

    #[inline]
    fn write<T: BitView>(&mut self, val: T) {
        self.write_bitslice(val.view_bits::<Msb0>());
    }

    #[inline]
    fn write_twoc<T: Into<i64>>(&mut self, val: T, bits_per_sample: usize) {
        let shifted = (val.into() << (64 - bits_per_sample)) as u64;
        self.write_msbs(shifted, bits_per_sample);
    }
}

impl<T2, O2> BitSink for BitVec<T2, O2>
where
    T2: BitStore,
    O2: BitOrder,
{
    #[inline]
    fn write_bitslice<T: BitStore, O: BitOrder>(&mut self, other: &BitSlice<T, O>) {
        self.extend_from_bitslice(other);
    }

    #[inline]
    fn align_to_byte(&mut self) -> usize {
        let npad = 8 - self.len() % 8;
        if npad == 8 {
            return 0;
        }
        self.write_lsbs(0u8, npad);
        npad
    }
}

pub struct Tee<'a, L: BitSink, R: BitSink> {
    primary: &'a mut L,
    secondary: &'a mut R,
}

impl<'a, L: BitSink, R: BitSink> Tee<'a, L, R> {
    #[allow(dead_code)]
    pub fn new(primary: &'a mut L, secondary: &'a mut R) -> Self {
        Tee { primary, secondary }
    }
}

impl<'a, L: BitSink, R: BitSink> BitSink for Tee<'a, L, R> {
    fn write_bitslice<T: BitStore, O: BitOrder>(&mut self, other: &BitSlice<T, O>) {
        self.primary.write_bitslice(other);
        self.secondary.write_bitslice(other);
    }

    fn align_to_byte(&mut self) -> usize {
        let padded = self.primary.align_to_byte();
        self.secondary.write_lsbs(0u8, padded);
        padded
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

    /// Appends first `nbits` bits (from MSB) to the `ByteVec`.
    #[inline]
    fn push_u64_msbs(&mut self, val: u64, nbits: usize) {
        let mut val: u64 = val;
        let mut nbits = nbits;
        let nbitlength = self.bitlength + nbits;
        let r = self.tail_len();

        if r != 0 {
            let b: u8 = ((val >> (64 - r)) & ((1 << r) - 1)) as u8;
            let tail = self.bytes.len() - 1;
            self.bytes[tail] |= b;
            val <<= r;
            nbits = if nbits > r { nbits - r } else { 0 };
        }
        while nbits >= 8 {
            let b: u8 = (val >> (64 - 8) & 0xFFu64) as u8;
            self.bytes.push(b);
            val <<= 8;
            nbits -= 8;
        }
        if nbits > 0 {
            let b: u8 = ((val >> (64 - nbits)) << (8 - nbits)) as u8;
            self.bytes.push(b);
        }
        self.bitlength = nbitlength;
    }

    pub fn as_byte_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl BitSink for ByteVec {
    #[inline]
    fn write_bitslice<T: BitStore, O: BitOrder>(&mut self, other: &BitSlice<T, O>) {
        let mut bv: BitVec<u64, Msb0> = BitVec::with_capacity(other.len());
        bv.extend_from_bitslice(other);
        let mut size = bv.len();
        for elem in bv.as_raw_slice() {
            self.write_msbs(*elem, std::cmp::min(64, size));
            size = if size > 64 { size - 64 } else { 0 };
        }
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
        self.align_to_byte();
        self.bytes.extend_from_slice(bytes);
        ret
    }

    #[inline]
    fn write_msbs<T: Into<u64>>(&mut self, val: T, nbits: usize) {
        if nbits == 0 {
            return;
        }
        let initial_shift = 64 - (std::mem::size_of::<T>() * 8);
        self.push_u64_msbs(val.into() << initial_shift, nbits);
    }

    #[inline]
    fn write_lsbs<T: Into<u64>>(&mut self, val: T, nbits: usize) {
        if nbits == 0 {
            return;
        }
        self.push_u64_msbs(val.into() << (64 - nbits), nbits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bitvec::prelude::bits;
    use bitvec::prelude::Lsb0;

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
    fn write_with_tee() {
        let mut sink: BitVec<u8> = BitVec::new();
        sink.write_lsbs(0x01u8, 1);
        let mut aux_sink: BitVec<u16> = BitVec::new();
        let mut tee = Tee::new(&mut sink, &mut aux_sink);
        tee.write_lsbs(0x01u8, 1);
        tee.align_to_byte();
        assert_eq!(sink.len(), 8);
        assert_eq!(aux_sink.len(), 7);
        assert_eq!(sink, bits![1, 1, 0, 0, 0, 0, 0, 0]);
        assert!(aux_sink[0]);
        assert!(!aux_sink[1]);
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
