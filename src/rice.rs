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

//! Functions for partitioned rice coding (PRC).

use std::cell::RefCell;

use std::simd::SimdUint;

use std::simd::SimdPartialEq;
use std::simd::SimdPartialOrd;

use seq_macro::seq;

use super::constant::MAX_RICE_PARAMETER;
use super::constant::MAX_RICE_PARTITIONS;
use super::constant::MAX_RICE_PARTITION_ORDER;
use super::constant::MIN_RICE_PARTITION_SIZE;

/// Table that contains the numbers of bits needed for a partition.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
struct PrcBitTable {
    p_to_bits: std::simd::u32x16,
    mask: std::simd::Mask<<u32 as std::simd::SimdElement>::Mask, 16>,
}

static ZEROS: std::simd::u32x16 = std::simd::u32x16::from_array([0u32; 16]);
static INDEX: std::simd::u32x16 =
    std::simd::u32x16::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
static MAXES: std::simd::u32x16 = std::simd::u32x16::from_array([u32::MAX; 16]);

impl PrcBitTable {
    pub fn zero(max_p: usize) -> Self {
        debug_assert!(max_p <= MAX_RICE_PARAMETER);
        Self {
            p_to_bits: ZEROS,
            mask: INDEX.simd_le(std::simd::u32x16::splat(max_p as u32)),
        }
    }

    pub fn from_signal(signal: &[i32], max_p: usize, offset: usize) -> Self {
        let mut ret = Self::zero(max_p);
        ret.init_with_signal(signal, offset);
        ret
    }

    #[allow(unused_assignments, clippy::identity_op)]
    fn init_with_signal(&mut self, signal: &[i32], offset: usize) {
        let mut p_to_bits: [u32; 16] = [0u32; 16];
        seq!(p in 0..15 {
            p_to_bits[p] = (offset + signal.len() * (p + 1)) as u32;
        });

        for v in signal.iter().map(|x| encode_signbit(*x)) {
            let mut v = v;
            seq!(p in 0..15 {
                p_to_bits[p] += v as u32;
                v >>= 1;
            });
        }
        self.p_to_bits = std::simd::u32x16::from_array(p_to_bits);
    }

    #[cfg(test)]
    pub fn bits(&self, p: usize) -> usize {
        self.p_to_bits[p] as usize
    }

    #[inline]
    pub fn minimizer(&self) -> (usize, usize) {
        let ret_bits = self.mask.select(self.p_to_bits, MAXES).reduce_min();
        let ret_p = self
            .p_to_bits
            .simd_eq(std::simd::u32x16::splat(ret_bits))
            .select(INDEX, ZEROS)
            .reduce_max();

        (ret_p as usize, ret_bits as usize)
    }

    #[allow(unused_comparisons)]
    #[inline]
    pub fn merge(&self, other: &Self, offset: usize) -> Self {
        let offset = std::simd::u32x16::splat(offset as u32);
        let offset = self.mask.select(offset, ZEROS);
        Self {
            p_to_bits: self.p_to_bits + other.p_to_bits - offset,
            mask: self.mask,
        }
    }
}

/// Finds the number of finest partitions.
#[inline]
fn finest_partition_order(size: usize, min_part_size: usize) -> usize {
    assert!(min_part_size >= 1);
    let max_splits: u32 = (size / min_part_size) as u32;
    let max_order_for_min_part = (32 - max_splits.leading_zeros() - 1) as usize;
    std::cmp::min(
        MAX_RICE_PARTITION_ORDER,
        std::cmp::min(max_order_for_min_part, size.trailing_zeros() as usize),
    )
}

/// Encodes the sign bit into its LSB (for Rice coding).
#[inline]
pub const fn encode_signbit(v: i32) -> u32 {
    let is_negative = if v < 0 { 1 } else { 0 };
    v.unsigned_abs() * 2 - is_negative
}

/// Recovers a sign bit from its LSB.
#[inline]
pub const fn decode_signbit(v: u32) -> i32 {
    let is_negative = v % 2 == 1;
    if is_negative {
        -(((v >> 1) + 1) as i32)
    } else {
        (v >> 1) as i32
    }
}

/// Computes the storage bits with given the slice of `RicePerformanceTable`s.
///
/// NOTE: API design of this function looks a bit strange but is intended to
/// minimize heap allocation.
fn eval_partitions(tables: &[PrcBitTable], ps: &mut [usize]) -> usize {
    assert!(ps.len() >= tables.len());
    let mut sum_bits = 0;
    for (dest, t) in ps.iter_mut().zip(tables) {
        let (p, bits) = t.minimizer();
        sum_bits += bits;
        *dest = p;
    }
    sum_bits
}

/// Merges `RicePerformanceTable`s and overwrites the table with merged values.
///
/// NOTE: API design of this function looks a bit strange but is intended to
/// minimize heap allocation.
fn merge_partitions(tables: &mut [PrcBitTable]) -> usize {
    assert!(tables.len() < MAX_RICE_PARTITIONS);
    let merged_len = tables.len() / 2;

    for part_id in 0..merged_len {
        tables[part_id] = tables[part_id * 2].merge(&tables[part_id * 2 + 1], 4);
    }
    merged_len
}

/// Parameter for PRC (partitioned Rice-coding).
pub struct PrcParameter {
    pub order: usize,
    pub ps: Vec<u8>,
    pub code_bits: usize,
}

impl PrcParameter {
    pub fn new(order: usize, ps: Vec<u8>, code_bits: usize) -> Self {
        Self {
            order,
            ps,
            code_bits,
        }
    }
}

/// Helper object that holds pre-allocated buffer for PRC optimization.
struct PrcParameterFinder {
    pub tables: Vec<PrcBitTable>,
    pub ps: Vec<usize>,
    pub min_ps: Vec<usize>,
}

impl PrcParameterFinder {
    pub const fn new() -> Self {
        Self {
            tables: Vec::new(),
            ps: Vec::new(),
            min_ps: Vec::new(),
        }
    }

    pub fn find(&mut self, signal: &[i32], warmup_length: usize, max_p: usize) -> PrcParameter {
        let mut partition_order = finest_partition_order(
            signal.len(),
            std::cmp::max(MIN_RICE_PARTITION_SIZE, warmup_length),
        );
        let mut nparts = 1 << (partition_order as i32);

        self.tables.clear();
        self.min_ps.resize(nparts, 0);

        let part_size = signal.len() / nparts;
        for p in 0..nparts {
            let start = std::cmp::max(p * part_size, warmup_length);
            let end = (p + 1) * part_size;
            let table = PrcBitTable::from_signal(&signal[start..end], max_p, 4);
            self.tables.push(table);
        }
        let mut min_bits = eval_partitions(&self.tables, &mut self.min_ps);
        let mut min_order = partition_order;

        while nparts > 1 {
            nparts = merge_partitions(&mut self.tables[0..nparts]);
            partition_order -= 1;
            self.ps.resize(nparts, 0);
            let next_bits = eval_partitions(&self.tables[0..nparts], &mut self.ps);
            if next_bits < min_bits {
                min_bits = next_bits;
                self.min_ps = self.ps.clone();
                min_order = partition_order;
            }
        }
        self.min_ps.truncate(1usize << min_order);
        PrcParameter::new(
            min_order as usize,
            self.min_ps.iter().map(|x| *x as u8).collect(),
            min_bits,
        )
    }
}

thread_local! {
    static RICE_PARAMETER_FINDER: RefCell<PrcParameterFinder> = RefCell::new(PrcParameterFinder::new());
}

pub fn find_partitioned_rice_parameter(
    signal: &[i32],
    warmup_length: usize,
    max_p: usize,
) -> PrcParameter {
    RICE_PARAMETER_FINDER.with(|finder| finder.borrow_mut().find(signal, warmup_length, max_p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

    #[test]
    fn bit_table_initialization() {
        let table = PrcBitTable::from_signal(&[3, 4, 5, 6], 2, 4);
        assert_eq!(table.bits(0), 3 * 2 + 4 * 2 + 5 * 2 + 6 * 2 + 8);
        assert_eq!(table.bits(1), 3 + 4 + 5 + 6 + 8 + 4);
    }

    #[test]
    fn prc_parameter_search() {
        let signal = test_helper::constant_plus_noise(64, 0, 4096);
        let table = PrcBitTable::from_signal(&signal, 14, 4);
        let (p, _bits) = table.minimizer();
        eprintln!("Table = {:?}", table);
        eprintln!("Found p = {}", p);
        // assert at least there's some parameter smaller than verbatim coding.
        assert!(p < 13);
        // Also, must be better than unary coding.
        assert!(p > 0);
    }

    #[test]
    fn finest_partition_order_search() {
        assert_eq!(finest_partition_order(64, 4), 4);
        assert_eq!(finest_partition_order(64, 3), 4);

        assert_eq!(finest_partition_order(192, 1), 6);
        assert_eq!(finest_partition_order(192, 3), 6);
        assert_eq!(finest_partition_order(192, 4), 5);
    }

    #[test]
    fn partitioned_rice_parameter_search() {
        let signal_left = test_helper::constant_plus_noise(64, 0, 2048);
        let signal_right = test_helper::constant_plus_noise(64, 0, 12);
        let signal = [signal_left, signal_right].concat();
        let (_single_param, single_bits) =
            PrcBitTable::from_signal(&signal[4..], 14, 4).minimizer();
        let prc_p = super::find_partitioned_rice_parameter(&signal, 4, 14);

        assert!(prc_p.code_bits <= single_bits);
        assert_eq!(prc_p.order, 1); // this only holds stochastically
    }

    #[test]
    fn partition_evaluation() {
        let mut part1 = PrcBitTable::zero(4);
        part1.p_to_bits[0..5].copy_from_slice(&[17, 19, 15, 11, 19]);
        let mut part2 = PrcBitTable::zero(4);
        part2.p_to_bits[0..5].copy_from_slice(&[12, 14, 16, 18, 20]);

        let mut params = [0, 0];
        let min_bits = eval_partitions(&[part1, part2], &mut params);
        assert_eq!(min_bits, 23);
        assert_eq!(params, [3, 0]);
    }

    #[test]
    fn partition_merging() {
        let mut part1 = PrcBitTable::zero(4);
        part1.p_to_bits[0..5].copy_from_slice(&[17, 19, 15, 11, 19]);
        let mut part2 = PrcBitTable::zero(4);
        part2.p_to_bits[0..5].copy_from_slice(&[12, 14, 16, 18, 20]);

        let mut table = [part1, part2];
        let table_size = merge_partitions(&mut table);
        assert_eq!(table_size, 1);
        assert_eq!(table[0].p_to_bits[0..5], [25, 29, 27, 25, 35]);
    }

    #[test]
    fn minimizer_search() {
        let mut bt = PrcBitTable::zero(4);
        bt.p_to_bits[0..8].copy_from_slice(&[6, 7, 4, 5, 9, 0, 0, 0]);
        assert_eq!(bt.minimizer(), (2, 4));

        let mut bt = PrcBitTable::zero(4);
        bt.p_to_bits[0..8].copy_from_slice(&[6, 7, 8, 5, 3, 0, 0, 0]);
        assert_eq!(bt.minimizer(), (4, 3));

        let mut bt = PrcBitTable::zero(4);
        bt.p_to_bits[0..8].copy_from_slice(&[1, 7, 8, 5, 3, 0, 0, 0]);
        assert_eq!(bt.minimizer(), (0, 1));

        let mut bt = PrcBitTable::zero(4);
        bt.p_to_bits[0..8].copy_from_slice(&[1, 7, 1, 1, 3, 0, 0, 0]);
        // Current implementation prefers the largest p when there're multiple
        // minimizers.
        assert_eq!(bt.minimizer(), (3, 1));
    }
}
