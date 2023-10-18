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

use std::convert::Infallible;

/// Trait for the bit-addressible integers.
///
/// This trait is sealed so a user cannot implement it. Currently, this trait
/// covers: [`u8`], [`u16`], [`u32`], and [`u64`].
pub trait PackedBits: seal_packed_bits::Sealed {
    const PACKED_BITS: usize = 1usize << Self::PACKED_BITS_LOG2;
    const PACKED_BYTES: usize = Self::PACKED_BITS / 8usize;
    const PACKED_BITS_LOG2: usize;
}

impl<T: seal_packed_bits::Sealed> PackedBits for T {
    /// The number of bits packed in this type. Synonym of `T::BITS`.
    #[rustversion::since(1.67)]
    const PACKED_BITS_LOG2: usize = (std::mem::size_of::<T>() * 8).ilog2() as usize;
    #[rustversion::before(1.67)]
    const PACKED_BITS_LOG2: usize = 3 + std::mem::size_of::<T>().trailing_zeros() as usize;
}

/// Trait for the signed integers that can be provided to bitsink.
///
/// This trait is sealed so a user cannot implement it. Currently, this trait
/// covers: [`i8`], [`i16`], [`i32`], and [`i64`].
pub trait SignedBits: seal_signed_bits::Sealed {}

impl<T: seal_signed_bits::Sealed> SignedBits for T {}

/// Storage-agnostic interface trait for bit-based output.
///
/// The encoder repeatedly generates arrays of code bits that are typically
/// smaller than a byte (8 bits).  Type implementing `BitSink` is used to
/// arrange those bits typically in bytes, and transfer them to the backend
/// storage. [`ByteSink`] is a standard implementation of `BitSink` that stores
/// code bits to a `Vec` of [`u8`]s.
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
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    /// let mut sink = ByteSink::new();
    ///
    /// sink.write_lsbs(0xFFu8, 3);
    /// assert_eq!(sink.len(), 3);
    ///
    /// let pads = sink.align_to_byte()?;
    /// assert_eq!(pads, 5);
    /// assert_eq!(sink.len(), 8);
    /// # Ok(())}
    /// ```
    fn align_to_byte(&mut self) -> Result<usize, Self::Error>;

    /// Writes bytes after alignment, and returns padded bits.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    /// let mut sink = ByteSink::new();
    ///
    /// sink.write_lsbs(0xFFu8, 3);
    /// assert_eq!(sink.len(), 3);
    ///
    /// sink.write_bytes_aligned(&[0xB7, 0x7D])?;
    ///
    /// assert_eq!(sink.to_bitstring(), "11100000_10110111_01111101");
    /// # Ok(())}
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    ///
    /// let mut sink = ByteSink::new();
    /// sink.write_lsbs(0x0Fu8, 3);
    ///
    /// assert_eq!(sink.len(), 3);
    /// assert_eq!(sink.to_bitstring(), "111*****");
    /// # Ok(())}
    /// ```
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error>;

    /// Writes `n` MSBs to the sink.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    ///
    /// let mut sink = ByteSink::new();
    /// sink.write_msbs(0xF0u8, 3);
    ///
    /// assert_eq!(sink.to_bitstring(), "111*****");
    /// # Ok(())}
    /// ```
    fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error>;

    /// Writes all bits in `val: PackedBits`.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    ///
    /// let mut sink = ByteSink::new();
    /// sink.write_msbs(0xF0u8, 3);
    ///
    /// sink.write(0x5555u16);
    ///
    /// assert_eq!(sink.to_bitstring(), "11101010_10101010_101*****");
    /// # Ok(())}
    /// ```
    fn write<T: PackedBits>(&mut self, val: T) -> Result<(), Self::Error>;

    /// Writes `val` in two's coplement format.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    ///
    /// let mut sink = ByteSink::new();
    /// sink.write_msbs(0xF0u8, 3);
    /// assert_eq!(sink.to_bitstring(), "111*****");
    ///
    /// // two's complement of 00011 in 11101
    /// sink.write_twoc(-3i32, 5);
    /// assert_eq!(sink.to_bitstring(), "11111101");
    /// # Ok(())}
    /// ```
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

    /// Writes `n`-bits of zeros.
    ///
    /// A default implementation using `write_msbs` is provided. An impl can
    /// provide a faster short-cut for writing zeros.
    ///
    /// # Errors
    ///
    /// It can emit errors describing backend issues.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// use flacenc::bitsink::{ByteSink, BitSink};
    ///
    /// let mut sink = ByteSink::new();
    /// sink.write_msbs(0xF0u8, 3);
    /// assert_eq!(sink.to_bitstring(), "111*****");
    ///
    /// sink.write_zeros(6);
    /// assert_eq!(sink.to_bitstring(), "11100000_0*******");
    /// # Ok(())}
    /// ```
    #[inline]
    fn write_zeros(&mut self, n: usize) -> Result<(), Self::Error> {
        let mut n = n;
        while n > 64 {
            self.write(0u64)?;
            n -= 64;
        }
        self.write_msbs(0u64, n)?;
        Ok(())
    }
}

/// `BitSink` implementation based on [`Vec`] of unsigned ints.
#[derive(Clone, Debug)]
pub struct MemSink<S> {
    storage: Vec<S>,
    bitlength: usize,
}

/// `BitSink` implementation based on [`Vec`] of [`u8`]s.
///
/// Since this type store code bits in [`u8`]s, the internal buffer can directly
/// be written to, e.g. [`std::io::Write`] via [`write_all`] method.
///
/// [`write_all`]: std::io::Write::write_all
pub type ByteSink = MemSink<u8>;

impl<S: PackedBits> Default for MemSink<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: PackedBits> MemSink<S> {
    /// Creates new `MemSink` instance with the default capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// let empty: [u8; 0] = [];
    /// // note thatc `as_byte_slice` is only implemented for <u8>.
    /// assert_eq!(&empty, sink.as_byte_slice());
    /// ```
    pub fn new() -> Self {
        Self {
            storage: vec![],
            bitlength: 0usize,
        }
    }

    /// Creates new `MemSink` instance with the specified capacity (in bits).
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::with_capacity(128);
    /// sink.write_lsbs(0x00FFu16, 10);
    /// assert!(sink.into_bytes().capacity() > 128 / 8);
    /// ```
    pub fn with_capacity(capacity_in_bits: usize) -> Self {
        Self {
            storage: Vec::with_capacity((capacity_in_bits >> S::PACKED_BITS_LOG2) + 1),
            bitlength: 0usize,
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// sink.write_lsbs(0xAAAAAAAAu32, 14);
    /// assert_eq!(sink.to_bitstring(), "10101010_101010**");
    /// sink.clear();
    /// assert_eq!(sink.to_bitstring(), "");
    /// ```
    pub fn clear(&mut self) {
        self.storage.clear();
        self.bitlength = 0;
    }

    /// Returns the number of bits stored in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// sink.write(0u64);
    /// sink.write_msbs(0u8, 6);
    /// assert_eq!(sink.len(), 70)
    /// ```
    pub fn len(&self) -> usize {
        self.bitlength
    }

    /// Checks if the buffer is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// assert!(sink.is_empty());
    /// sink.write_msbs(0u8, 6);
    /// assert!(!sink.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.bitlength == 0
    }

    /// Reseerves capacity for at least `additional_in_bits` more bits.
    ///
    /// This function reserves the 'Vec's capacity so that the allocated size is
    /// sufficient for storing `self.len() + additional_in_bits` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::with_capacity(1);
    /// sink.write_bytes_aligned(&[0u8; 128]);
    /// assert_eq!(sink.len(), 1024);
    /// sink.reserve(2048);
    /// assert!(sink.into_bytes().capacity() > (1024 + 2048) / 8);
    /// ```
    pub fn reserve(&mut self, additional_in_bits: usize) {
        self.storage
            .reserve((additional_in_bits >> S::PACKED_BITS_LOG2) + 1);
    }

    /// Returns the remaining number of bits in the last byte in `self.bytes`.
    #[inline]
    const fn paddings(&self) -> usize {
        ((!self.bitlength).wrapping_add(1)) & (S::PACKED_BITS - 1)
    }

    /// Returns the remaining number of bits in the last byte in `self.bytes`.
    #[inline]
    const fn paddings_to_byte(&self) -> usize {
        ((!self.bitlength).wrapping_add(1)) & 7
    }

    /// Returns bits in a string.
    ///
    /// This function formats an internal buffer state to a human-readable
    /// string. Each byte is shown in eight characters joined by `'_'`, and the
    /// last bits of the last byte that are not yet filled are shown as `'*'`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// sink.write_msbs(0x3456u16, 13);
    /// assert_eq!(sink.to_bitstring(), "00110100_01010***");
    /// ```
    pub fn to_bitstring(&self) -> String {
        let mut ret = String::new();
        for v in &self.storage {
            for b in v.to_be_bytes().as_ref() {
                ret.push_str(&format!("{b:08b}"));
            }
            ret.push('_');
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

    pub fn write_to_byte_slice(&self, dest: &mut [u8]) {
        let destlen = dest.len();
        let mut head = 0;
        'outer: for v in &self.storage {
            for b in v.to_be_bytes().as_ref() {
                if head >= destlen {
                    break 'outer;
                }
                dest[head] = *b;
                head += 1;
            }
        }
    }
}

impl MemSink<u8> {
    /// Consumes `ByteSink` and returns the internal buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// sink.write_bytes_aligned(&[0xABu8; 4]);
    /// let v: Vec<u8> = sink.into_bytes();
    /// assert_eq!(&v, &[0xAB; 4]);
    /// ```
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.storage
    }

    /// Returns a reference to the internal bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::bitsink::*;
    /// let mut sink = MemSink::<u8>::new();
    /// sink.write_msbs(0x3456u16, 13);
    /// assert_eq!(sink.to_bitstring(), "00110100_01010***");
    /// ```
    pub fn as_byte_slice(&self) -> &[u8] {
        &self.storage
    }
}

// Implemet it for only few special cases
impl BitSink for MemSink<u8> {
    type Error = Infallible;

    #[inline]
    fn write<T: PackedBits>(&mut self, val: T) -> Result<(), Self::Error> {
        let nbitlength = self.bitlength + 8 * std::mem::size_of::<T>();
        let tail = self.paddings();
        if tail > 0 {
            self.write_msbs(val, tail)?;
        }
        let val = val << tail;
        let bytes: T::Bytes = val.to_be_bytes();
        self.storage.extend_from_slice(bytes.as_ref());
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
        self.storage.extend_from_slice(bytes);
        self.bitlength += 8 * bytes.len();
        Ok(ret)
    }

    #[inline]
    fn write_msbs<T: PackedBits>(&mut self, mut val: T, mut n: usize) -> Result<(), Self::Error> {
        if n == 0 {
            return Ok(());
        }
        let r = self.paddings();
        self.bitlength += n;
        val = val & !((T::one() << (T::PACKED_BITS - n)) - T::one());

        if r != 0 {
            let b = (val >> (T::PACKED_BITS - r)).as_();
            *self.storage.last_mut().unwrap() |= b;
            val <<= r;
            if r >= n {
                return Ok(());
            }
            n -= r;
        }
        let bytes_to_write = n >> 3;
        if bytes_to_write > 0 {
            let bytes = val.to_ne_bytes();
            let bytes = bytes.as_ref();
            #[cfg(target_endian = "little")]
            {
                for i in 0..bytes_to_write {
                    self.storage.push(bytes[std::mem::size_of::<T>() - i - 1]);
                }
            }
            #[cfg(target_endian = "big")]
            {
                for i in 0..bytes_to_write {
                    self.storage.push(bytes[i]);
                }
            }
            n &= 7;
        }
        if n > 0 {
            val <<= bytes_to_write << 3;
            let tail_byte: u8 = (val >> (T::PACKED_BITS - 8)).as_();
            self.storage.push(tail_byte);
        }
        Ok(())
    }

    #[inline]
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
        if n == 0 {
            return Ok(());
        }
        self.write_msbs(val << (T::PACKED_BITS - n), n)
    }

    #[inline]
    fn write_zeros(&mut self, n: usize) -> Result<(), Self::Error> {
        let pad = self.paddings();
        if n <= pad {
            self.bitlength += n;
            return Ok(());
        }
        self.bitlength += pad;
        let n = n - pad;

        let bytes = (n + 7) >> 3;
        self.storage.resize(self.storage.len() + bytes, 0u8);
        self.bitlength += n;

        Ok(())
    }
}

impl BitSink for MemSink<u64> {
    type Error = Infallible;

    #[inline]
    fn write<T: PackedBits>(&mut self, val: T) -> Result<(), Self::Error> {
        self.write_msbs(val, T::PACKED_BITS)
    }

    #[inline]
    fn align_to_byte(&mut self) -> Result<usize, Self::Error> {
        let r = self.paddings_to_byte();
        self.bitlength += r;
        Ok(r)
    }

    #[inline]
    fn write_bytes_aligned(&mut self, bytes: &[u8]) -> Result<usize, Self::Error> {
        // this will not be called for u64. so the implementation is not
        // efficient.
        let r = self.align_to_byte()?;
        for b in bytes {
            self.write(*b)?;
        }
        Ok(r)
    }

    #[inline]
    fn write_msbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
        // this routine is optimized especially for `Residual::write`.
        // and is trying to maximize efficiency of the auto-vectorization by
        // explicitly deferring "if"-statement and actual storage access.
        let r = self.paddings();
        self.bitlength += n;
        let mut val: u64 = val.into();
        val <<= 64 - T::PACKED_BITS;

        // clear lsbs
        val &= !((1u64 << (64 - n)) - 1);

        let last_setter = val.wrapping_shr(64u32 - r as u32);
        val = val.wrapping_shl(r as u32);
        // u64 is the maximum size, so we only need to push remaining bits.
        if r != 0 && n > 0 {
            *self.storage.last_mut().unwrap() |= last_setter;
        }
        if r < n && n > 0 {
            self.storage.push(val);
        }
        Ok(())
    }

    #[inline]
    fn write_lsbs<T: PackedBits>(&mut self, val: T, n: usize) -> Result<(), Self::Error> {
        self.write_msbs(val << (T::PACKED_BITS - n), n)
    }

    #[inline]
    fn write_zeros(&mut self, n: usize) -> Result<(), Self::Error> {
        // this routine is optimized especially for `Residual::write`.
        // and is trying to maximize efficiency of the auto-vectorization by
        // explicitly deferring "if"-statement and actual storage access.
        debug_assert!(n < isize::MAX as usize);
        let pad = self.paddings() as isize;
        self.bitlength += n;
        let n = std::cmp::max(n as isize - pad, 0) as usize;
        let elems: usize = (n + u64::PACKED_BITS - 1) >> u64::PACKED_BITS_LOG2;
        if elems > 0 {
            self.storage.resize(self.storage.len() + elems, 0u64);
        }
        Ok(())
    }
}

mod seal_packed_bits {
    use num_traits::AsPrimitive;
    use num_traits::One;
    use num_traits::PrimInt;
    use num_traits::ToBytes;
    use num_traits::WrappingShl;
    pub trait Sealed:
        ToBytes
        + From<u8>
        + Into<u64>
        + PrimInt
        + std::ops::ShlAssign<usize>
        + AsPrimitive<u8>
        + One
        + WrappingShl
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
        type Error = Infallible;

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
            self.write_lsbs(val, std::mem::size_of::<T>() << 3)?;
            Ok(())
        }
    }

    #[test]
    fn align_to_byte_with_bitvec() -> Result<(), Infallible> {
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
    fn twoc_writing() -> Result<(), Infallible> {
        let mut sink: BitVec<u8> = BitVec::new();
        sink.write_twoc(-7, 4)?;
        assert_eq!(sink, bits![1, 0, 0, 1]);
        Ok(())
    }

    #[test]
    fn bytevec_write_msb() -> Result<(), Infallible> {
        let mut bv = ByteSink::new();
        bv.write_msbs(0xFFu8, 3)?;
        bv.write_msbs(0x0u64, 12)?;
        bv.write_msbs(0xFFFF_FFFFu32, 9)?;
        bv.write_msbs(0x0u16, 8)?;
        assert_eq!(bv.to_bitstring(), "11100000_00000001_11111111_00000000");

        let mut bv = ByteSink::new();
        bv.write_msbs(0xA0u8, 3)?;
        assert_eq!(bv.to_bitstring(), "101*****");

        let mut bv = ByteSink::new();
        bv.write_msbs(0x00u8, 2)?;
        bv.write_msbs(0xFFu8, 3)?;
        bv.write_msbs(0x00u8, 2)?;
        assert_eq!(bv.to_bitstring(), "0011100*");

        Ok(())
    }

    #[test]
    fn bytevec_write_lsb() -> Result<(), Infallible> {
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

    #[test]
    fn u64vec() -> Result<(), Infallible> {
        let mut sink = MemSink::<u64>::new();
        sink.write_msbs(0xFFFF_FFFFu32, 17)?;
        assert_eq!(
            sink.to_bitstring(),
            "11111111111111111***********************************************"
        );
        assert_eq!(sink.len(), 17);

        sink.write_bytes_aligned(&[0xCA, 0xFE])?;
        assert_eq!(
            sink.to_bitstring(),
            "1111111111111111100000001100101011111110************************"
        );
        assert_eq!(sink.len(), 40);

        sink.write_lsbs(1u16, 2)?;
        assert_eq!(
            sink.to_bitstring(),
            "111111111111111110000000110010101111111001**********************"
        );
        assert_eq!(sink.len(), 42);

        sink.write_lsbs(0xAAAA_AAAAu32, 31)?;
        assert_eq!(
            sink.to_bitstring(),
            concat!(
                "1111111111111111100000001100101011111110010101010101010101010101_",
                "010101010*******************************************************"
            )
        );
        assert_eq!(sink.len(), 73);
        Ok(())
    }
}
