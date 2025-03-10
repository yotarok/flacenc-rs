// Copyright 2022-2024 Google LLC
// Copyright 2025- flacenc-rs developers
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

use super::arrayutils::unaligned_map_and_update;
use super::constant::rice::MAX_PARTITIONS as MAX_RICE_PARTITIONS;
use super::constant::rice::MAX_PARTITION_ORDER as MAX_RICE_PARTITION_ORDER;
use super::constant::rice::MAX_RICE_PARAMETER;
use super::constant::rice::MIN_PARTITION_SIZE as MIN_RICE_PARTITION_SIZE;
use super::repeat::repeat;

import_simd!(as simd);

/// Table that contains the numbers of bits needed for a partition.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
struct PrcBitTable {
    p_to_bits: simd::u32x16,
}

static ZEROS: simd::u32x16 = simd::u32x16::from_array([0u32; 16]);
static INDEX: simd::u32x16 =
    simd::u32x16::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
static INDEX1: simd::u32x16 =
    simd::u32x16::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
static MAXES: simd::u32x16 = simd::u32x16::from_array([u32::MAX; 16]);

// max value of p_to_bits is chosen so that an estimate doesn't overflow after
// being added 2^4 = 16 times each other at maximum.
static MAX_P_TO_BITS: u32 = (1 << 28) - 1;
static MAX_P_TO_BITS_VEC: simd::u32x16 = simd::u32x16::from_array([MAX_P_TO_BITS; 16]);

const PRC_BIT_TABLE_FROM_ERRORS_UNROLL_N: usize = 16; // must be up to 16.

impl PrcBitTable {
    #[cfg(test)]
    pub fn zero() -> Self {
        Self { p_to_bits: ZEROS }
    }

    pub fn from_errors(errors: &[u32], offset: usize) -> Self {
        // ensure that there's no overflow.
        debug_assert!(offset < (1 << 31));
        let offset =
            simd::u32x16::splat(offset as u32) + simd::u32x16::splat(errors.len() as u32) * INDEX1;
        let mut p_to_bits = ZEROS;

        // MAX_P_TO_BITS is designed not to overflow after 16 times of addition.
        //
        // TODO: there's still a risk of overflow when there's a consecutive 16
        // elements in `error` where all are larger than `1 << 28`. Since it's
        // very low probability and clamping inputs may degrade the performance,
        // this issue is ignored currently.
        //
        // In most of SIMD-capable CPUs, saturating ops can be done with a
        // single instruction. However, strangely the use of `saturating_add`
        // and removing `simd_min` from the loop actually slowed down the
        // computation by almost twice.
        for chunk in errors.chunks(PRC_BIT_TABLE_FROM_ERRORS_UNROLL_N) {
            if chunk.len() == PRC_BIT_TABLE_FROM_ERRORS_UNROLL_N {
                repeat!(n to PRC_BIT_TABLE_FROM_ERRORS_UNROLL_N => {
                    p_to_bits += simd::Simd::splat(chunk[n]) >> INDEX;
                });
            } else {
                repeat!(
                    n to PRC_BIT_TABLE_FROM_ERRORS_UNROLL_N;
                    while n < chunk.len() => {
                        p_to_bits += simd::Simd::splat(chunk[n]) >> INDEX;
                    }
                );
            }
            p_to_bits = p_to_bits.simd_min(MAX_P_TO_BITS_VEC);
        }
        p_to_bits += offset;
        p_to_bits = p_to_bits.simd_min(MAX_P_TO_BITS_VEC);
        Self { p_to_bits }
    }

    #[cfg(test)]
    pub fn bits(&self, p: usize) -> usize {
        self.p_to_bits[p] as usize
    }

    #[inline]
    pub fn minimizer(&self, max_p: usize) -> (usize, usize) {
        debug_assert!(max_p <= MAX_RICE_PARAMETER);
        // exploit the fact that `p_to_bits` only occupies 28-bits of u32.
        let mask = INDEX.simd_le(simd::u32x16::splat(max_p as u32));
        let four = simd::u32x16::splat(4);
        let packed_bits_and_idxs = (mask.select(self.p_to_bits, MAXES) << four) | INDEX;
        let minim = packed_bits_and_idxs.reduce_min();
        let ret_bits = minim >> 4;
        let ret_p = minim & 0x0F;

        (ret_p as usize, ret_bits as usize)
    }

    #[inline]
    pub fn merge(&self, other: &Self, offset: usize) -> Self {
        let offset = simd::u32x16::splat(offset as u32);
        Self {
            p_to_bits: self.p_to_bits + other.p_to_bits - offset,
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
    (v.unsigned_abs() << 1) - (v < 0) as u32
}

#[inline]
pub fn encode_signbit_simd<const N: usize>(v: simd::Simd<i32, N>) -> simd::Simd<u32, N>
where
    simd::LaneCount<N>: simd::SupportedLaneCount,
{
    (v.abs().cast() << simd::Simd::splat(1u32)) - (v.cast() >> simd::Simd::splat(31u32))
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
fn eval_partitions(tables: &[PrcBitTable], ps: &mut [usize], max_p: usize) -> usize {
    assert!(ps.len() >= tables.len());
    let mut sum_bits = 0;
    for (dest, t) in ps.iter_mut().zip(tables) {
        let (p, bits) = t.minimizer(max_p);
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
#[derive(Clone, Debug)]
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
#[derive(Default)]
struct PrcParameterFinder {
    pub errors: Vec<u32>,
    pub tables: Vec<PrcBitTable>,
    pub ps: Vec<usize>,
    pub min_ps: Vec<usize>,
}

impl PrcParameterFinder {
    pub fn find(&mut self, signal: &[i32], warmup_length: usize, max_p: usize) -> PrcParameter {
        let mut partition_order = finest_partition_order(
            signal.len(),
            std::cmp::max(MIN_RICE_PARTITION_SIZE, warmup_length),
        );
        let mut nparts = 1 << (partition_order as i32);

        self.tables.clear();
        self.min_ps.resize(nparts, 0);
        self.errors.clear();
        self.errors.resize(signal.len(), 0u32);
        unaligned_map_and_update::<u32, 64, _, _, _>(
            signal,
            &mut self.errors,
            #[inline]
            |p, x| {
                *p = encode_signbit(x);
            },
            #[inline]
            |pv, v| {
                *pv = encode_signbit_simd(v);
            },
        );

        let part_size = signal.len() / nparts;
        for p in 0..nparts {
            let start = std::cmp::max(p * part_size, warmup_length);
            let end = (p + 1) * part_size;
            let table = PrcBitTable::from_errors(&self.errors[start..end], 4);
            self.tables.push(table);
        }
        let mut min_bits = eval_partitions(&self.tables, &mut self.min_ps, max_p);
        let mut min_order: usize = partition_order;

        while nparts > 1 {
            nparts = merge_partitions(&mut self.tables[0..nparts]);
            partition_order -= 1;
            self.ps.resize(nparts, 0);
            let next_bits = eval_partitions(&self.tables[0..nparts], &mut self.ps, max_p);
            if next_bits < min_bits {
                min_bits = next_bits;
                self.min_ps.clear();
                self.min_ps.extend_from_slice(&self.ps);
                min_order = partition_order;
            }
        }
        self.min_ps.truncate(1usize << min_order);
        PrcParameter::new(
            min_order,
            self.min_ps.iter().map(|x| *x as u8).collect(),
            min_bits,
        )
    }
}

reusable!(PRC_FINDER: PrcParameterFinder);

pub fn find_partitioned_rice_parameter(
    signal: &[i32],
    warmup_length: usize,
    max_p: usize,
) -> PrcParameter {
    reuse!(PRC_FINDER, |finder: &mut PrcParameterFinder| {
        finder.find(signal, warmup_length, max_p)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigen;
    use crate::sigen::Signal;

    #[test]
    fn bit_table_initialization() {
        let table = PrcBitTable::from_errors(&[6, 8, 10, 12], 4);
        assert_eq!(table.bits(0), 3 * 2 + 4 * 2 + 5 * 2 + 6 * 2 + 8);
        assert_eq!(table.bits(1), 3 + 4 + 5 + 6 + 8 + 4);
    }

    #[test]
    fn prc_parameter_search() {
        let signal = sigen::Noise::new(0.25).to_vec_quantized(12, 64);
        let errors: Vec<u32> = signal.iter().map(|v| encode_signbit(*v)).collect();
        let max_p = 14;
        let table = PrcBitTable::from_errors(&errors, 4);
        let (p, _bits) = table.minimizer(max_p);
        eprintln!("Table = {table:?}");
        eprintln!("Found p = {p}");
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
        let signal = sigen::Noise::with_seed(0, 0.5)
            .concat(64, sigen::Noise::with_seed(1, 0.05))
            .to_vec_quantized(8, 128);
        let errors: Vec<u32> = signal.iter().map(|v| encode_signbit(*v)).collect();
        let max_p = 14;
        let (_single_param, single_bits) =
            PrcBitTable::from_errors(&errors[4..], 4).minimizer(max_p);
        let prc_p = super::find_partitioned_rice_parameter(&signal, 4, 14);

        assert!(prc_p.code_bits <= single_bits);
        // this only holds stochastically but the random seeds are fixed.
        assert_eq!(prc_p.order, 1);
    }

    #[test]
    fn partition_evaluation() {
        let mut part1 = PrcBitTable::zero();
        part1.p_to_bits[0..5].copy_from_slice(&[17, 19, 15, 11, 19]);
        let mut part2 = PrcBitTable::zero();
        part2.p_to_bits[0..5].copy_from_slice(&[12, 14, 16, 18, 20]);

        let mut params = [0, 0];
        let min_bits = eval_partitions(&[part1, part2], &mut params, 4);
        assert_eq!(min_bits, 23);
        assert_eq!(params, [3, 0]);
    }

    #[test]
    fn partition_merging() {
        let mut part1 = PrcBitTable::zero();
        part1.p_to_bits[0..5].copy_from_slice(&[17, 19, 15, 11, 19]);
        let mut part2 = PrcBitTable::zero();
        part2.p_to_bits[0..5].copy_from_slice(&[12, 14, 16, 18, 20]);

        let mut table = [part1, part2];
        let table_size = merge_partitions(&mut table);
        assert_eq!(table_size, 1);
        assert_eq!(table[0].p_to_bits[0..5], [25, 29, 27, 25, 35]);
    }

    #[test]
    fn minimizer_search() {
        let mut bt = PrcBitTable::zero();
        bt.p_to_bits[0..8].copy_from_slice(&[6, 7, 4, 5, 9, 0, 0, 0]);
        assert_eq!(bt.minimizer(4), (2, 4));

        let mut bt = PrcBitTable::zero();
        bt.p_to_bits[0..8].copy_from_slice(&[6, 7, 8, 5, 3, 0, 0, 0]);
        assert_eq!(bt.minimizer(4), (4, 3));

        let mut bt = PrcBitTable::zero();
        bt.p_to_bits[0..8].copy_from_slice(&[1, 7, 8, 5, 3, 0, 0, 0]);
        assert_eq!(bt.minimizer(4), (0, 1));

        let mut bt = PrcBitTable::zero();
        bt.p_to_bits[0..8].copy_from_slice(&[7, 1, 1, 1, 3, 0, 0, 0]);
        // Current implementation prefers the smallest p when there're multiple
        // minimizers.
        assert_eq!(bt.minimizer(4), (1, 1));
    }

    #[test]
    fn prc_max_bits() {
        // check if the numbers of bits are bounded.
        let table = PrcBitTable::from_errors(&[0x0FFF_FFFE, 0x0100_0000], 0);
        assert_eq!(table.bits(0), MAX_P_TO_BITS as usize);
    }
}

#[cfg(all(test, feature = "simd-nightly"))]
mod bench {
    use super::*;

    extern crate test;

    use test::bench::Bencher;
    use test::black_box;

    #[bench]
    fn rice_find_minimizer(b: &mut Bencher) {
        let mut bt = PrcBitTable::zero();
        bt.p_to_bits[0..16].copy_from_slice(&[6, 7, 4, 5, 9, 9, 2, 3, 8, 2, 4, 3, 0, 0, 0, 0]);
        // This should be almost zero-cost
        b.iter(|| black_box(&bt).minimizer(black_box(12)));
    }

    #[bench]
    fn find_prc_parameter(b: &mut Bencher) {
        let mut signal = vec![];
        signal.extend(0..4096);

        b.iter(|| {
            find_partitioned_rice_parameter(black_box(&signal), black_box(14), black_box(14))
        });
    }

    #[bench]
    fn bit_table_creation(b: &mut Bencher) {
        let mut errors = vec![];
        errors.extend(0u32..4096u32);

        b.iter(|| PrcBitTable::from_errors(&errors, 123usize));
    }
}
