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

    fn align_to_byte(&mut self) -> usize;

    // Type signature may change. Don't override.
    #[inline]
    fn write_lsbs<T: Into<u64>>(&mut self, val: T, nbits: usize) {
        let val: u64 = val.into();
        self.write_bitslice(&val.view_bits::<Msb0>()[64 - nbits..]);
    }

    #[inline]
    fn write_msbs<T: BitStore>(&mut self, val: T, nbits: usize) {
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
    fn write_bitslice<T: BitStore, O: BitOrder>(&mut self, other: &BitSlice<T, O>) {
        self.extend_from_bitslice(other);
    }

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
}
