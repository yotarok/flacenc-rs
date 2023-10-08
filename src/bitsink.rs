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

use super::error::Never;

/// Trait for the bit-addressible integers.
///
/// This trait is seaked so the user cannot implement it. Currently, this trait
/// covers: `u8`, `u16`, `u32`, and `u64`.
pub trait PackedBits: seal_packed_bits::Sealed {
    const PACKED_BITS: usize;
}

impl<T: seal_packed_bits::Sealed> PackedBits for T {
    /// The number of bits packed in this type. Synonym of `T::BITS`.
    const PACKED_BITS: usize = std::mem::size_of::<T>() * 8;
}

/// Trait for the signed integers that can be provided to bitsink.
///
/// This trait is seaked so the user cannot implement it. Currently, this trait
/// covers: `i8`, `i16`, `i32`, and `i64`.
pub trait SignedBits: seal_signed_bits::Sealed {}

impl<T: seal_signed_bits::Sealed> SignedBits for T {}

/// Storage-agnostic interface trait for bit-based output.
///
/// `BitSink` API is unstable and will be subjected to change in a minor
/// version up. Therefore, it is not recommended to derive this trait for
/// supporting new output types. Rather, it is recommended to use
/// `bitvec::BitVec` or `ByteSink` (in this module) and covert the contents to
/// a desired type.
pub trait BitSink: Sized {
    /// Error type that may happen while writing bits to `BitSink`.
    type Error: std::error::Error;

    /// Puts zeros to `BitSink` until the length aligns to the byte boundaries.
    ///
    /// # Returns
    ///
    /// The number of zeros put.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    fn align_to_byte(&mut self) -> Result<usize, Self::Error>;

    /// Writes bytes after alignment, and returns padded bits.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    #[inline]
    fn write_bytes_aligned(&mut self, bytes: &[u8]) -> Result<usize, Self::Error> {
        let ret = self.align_to_byte()?;
        for b in bytes {
            self.write(*b)?;
        }
        Ok(ret)
    }

    /// Writes `n` LSBs to the sink.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error>;

    /// Writes `n` MSBs to the sink.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error>;

    /// Writes all bits in `val: PackedBits`.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    fn write<T: PackedBits>(&mut self, val: T) -> Result<(), Self::Error>;

    /// Writes `val` in two's coplement format.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    #[inline]
    fn write_twoc<T: SignedBits>(
        &mut self,
        val: T,
        bits_per_sample: usize,
    ) -> Result<(), Self::Error> {
        let val: i64 = val.into();
        let shifted = (val << (64 - bits_per_sample)) as u64;
        self.write_msbs(shifted, bits_per_sample)
    }
}

#[derive(Clone, Debug)]
pub struct ByteSink {
    bytes: Vec<u8>,
    bitlength: usize,
}

impl Default for ByteSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteSink {
    /// Creates new `ByteSink` instance with the default capacity.
    #[allow(clippy::missing_const_for_fn)] // for API robustness.
    pub fn new() -> Self {
        Self {
            bytes: vec![],
            bitlength: 0usize,
        }
    }

    /// Creates new `ByteSink` instance with the specified capacity (in bits).
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
    const fn paddings(&self) -> usize {
        let r = self.bitlength % 8;
        if r == 0 {
            0
        } else {
            8 - r
        }
    }

    /// Consumes `ByteSink` and returns the internal buffer.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns bits in a string for tests.
    #[cfg(test)]
    fn to_bitstring(&self) -> String {
        let mut ret = String::new();
        for b in &self.bytes {
            ret.push_str(&format!("{b:08b}_"));
        }
        ret.pop();

        // Not very efficient but should be okay for non-performance critical
        // use cases.
        for _t in 0..self.paddings() {
            ret.pop();
        }
        for _t in 0..self.paddings() {
            ret.push('*');
        }
        ret
    }

    pub fn as_byte_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl BitSink for ByteSink {
    type Error = Never;

    #[inline]
    fn write<T: PackedBits>(&mut self, val: T) -> Result<(), Self::Error> {
        let nbitlength = self.bitlength + 8 * std::mem::size_of::<T>();
        let tail = self.paddings();
        if tail > 0 {
            self.write_msbs(val, tail)?;
        }
        let val = val << tail;
        let bytes: T::Bytes = val.to_be_bytes();
        self.bytes.extend_from_slice(bytes.as_ref());
        self.bitlength = nbitlength;
        Ok(())
    }

    #[inline]
    fn align_to_byte(&mut self) -> Result<usize, Self::Error> {
        let r = self.paddings();
        self.bitlength += r;
        Ok(r)
    }

    #[inline]
    fn write_bytes_aligned(&mut self, bytes: &[u8]) -> Result<usize, Self::Error> {
        let ret = self.align_to_byte()?;
        self.bytes.extend_from_slice(bytes);
        self.bitlength += 8 * bytes.len();
        Ok(ret)
    }

    #[inline]
    fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
        if n == 0 {
            return Ok(());
        }
        let mut val: T = val;
        let mut n = n;
        let nbitlength = self.bitlength + n;
        let r = self.paddings();

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
        Ok(())
    }

    #[inline]
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
        if n == 0 {
            return Ok(());
        }
        self.write_msbs(val << (T::PACKED_BITS - n), n)
    }
}

mod seal_packed_bits {
    use num_traits::PrimInt;
    use num_traits::ToBytes;
    pub trait Sealed:
        ToBytes + From<u8> + Into<u64> + PrimInt + std::ops::ShlAssign<usize>
    {
    }

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

mod seal_signed_bits {
    pub trait Sealed: Into<i64> {}

    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
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
        type Error = Never;

        #[inline]
        fn align_to_byte(&mut self) -> Result<usize, Self::Error> {
            let npad = 8 - self.len() % 8;
            if npad == 8 {
                return Ok(0);
            }
            self.write_lsbs(0u8, npad)?;
            Ok(npad)
        }

        fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
            let val: u64 = val.into();
            self.extend_from_bitslice(&val.view_bits::<Msb0>()[64 - n..]);
            Ok(())
        }

        fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
            let val: u64 = val.into();
            self.extend_from_bitslice(&val.view_bits::<Msb0>()[0..n]);
            Ok(())
        }

        fn write<T: PackedBits>(&mut self, val: T) -> Result<(), Self::Error> {
            self.write_lsbs(val, std::mem::size_of::<T>() * 8)?;
            Ok(())
        }
    }

    #[test]
    fn align_to_byte_with_bitvec() -> Result<(), Never> {
        let mut sink: BitVec<u8> = BitVec::new();
        sink.write_lsbs(0x01u8, 1)?;
        sink.align_to_byte()?;
        assert_eq!(sink.len(), 8);
        sink.align_to_byte()?;
        assert_eq!(sink.len(), 8);
        sink.write_lsbs(0x01u8, 2)?;
        assert_eq!(sink.len(), 10);
        sink.align_to_byte()?;
        assert_eq!(sink.len(), 16);
        Ok(())
    }

    #[test]
    fn twoc_writing() -> Result<(), Never> {
        let mut sink: BitVec<u8> = BitVec::new();
        sink.write_twoc(-7, 4)?;
        assert_eq!(sink, bits![1, 0, 0, 1]);
        Ok(())
    }

    #[test]
    fn bytevec_write_msb() -> Result<(), Never> {
        let mut bv = ByteSink::new();
        bv.write_msbs(0xFFu8, 3)?;
        bv.write_msbs(0x0u64, 12)?;
        bv.write_msbs(0xFFFF_FFFFu32, 9)?;
        bv.write_msbs(0x0u16, 8)?;
        assert_eq!(bv.to_bitstring(), "11100000_00000001_11111111_00000000");
        Ok(())
    }

    #[test]
    fn bytevec_write_lsb() -> Result<(), Never> {
        let mut bv = ByteSink::new();
        bv.write_lsbs(0xFFu8, 3)?;
        bv.write_lsbs(0x0u64, 12)?;
        bv.write_lsbs(0xFFFF_FFFFu32, 9)?;
        bv.write_lsbs(0x0u16, 8)?;
        assert_eq!(bv.to_bitstring(), "11100000_00000001_11111111_00000000");

        let mut bv = ByteSink::new();
        bv.write_lsbs(0xFFu8, 3)?;
        bv.write_lsbs(0x0u64, 12)?;
        bv.write_lsbs(0xFFFF_FFFFu32, 9)?;
        bv.write_lsbs(0x0u16, 5)?;
        assert_eq!(bv.to_bitstring(), "11100000_00000001_11111111_00000***");
        Ok(())
    }
}
