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

use std::cmp::Ordering;
use std::ops::RangeInclusive;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use num_traits::PrimInt;
use num_traits::Signed;

use crate::arbitrary::Arb;
use crate::arbitrary::CondArbitrary;
use crate::component::ChannelAssignment;
use crate::component::Constant;
use crate::component::Decode;
use crate::component::FixedLpc;
use crate::component::Frame;
use crate::component::FrameHeader;
use crate::component::FrameOffset;
use crate::component::Lpc;
use crate::component::MetadataBlockData;
use crate::component::QuantizedParameters;
use crate::component::Residual;
use crate::component::Stream;
use crate::component::StreamInfo;
use crate::component::SubFrame;
use crate::component::Verbatim;
use crate::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use crate::constant::qlpc::MAX_PRECISION as MAX_LPC_PRECISION;
use crate::constant::qlpc::MAX_SHIFT as MAX_LPC_SHIFT;
use crate::constant::qlpc::MIN_SHIFT as MIN_LPC_SHIFT;
use crate::constant::rice::MAX_RICE_PARAMETER;
use crate::constant::MAX_BLOCK_SIZE;
use crate::error::Verify;

impl<'a> CondArbitrary<'a> for Stream {
    type Condition = (Option<BlockSize>, ChannelCount, BitsPerSample, SampleRate);

    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (
            opt_block_size,
            channel_count @ ChannelCount(channels),
            BitsPerSample(bits_per_sample),
            SampleRate(sample_rate),
        ) = *cond;
        let mut stream =
            Stream::new(sample_rate as usize, channels as usize, bits_per_sample).unwrap();

        let metadata_count = u.int_in_range(0usize..=5usize)?;
        for _i in 0..metadata_count {
            stream.add_metadata_block(MetadataBlockData::arbitrary(u)?);
        }

        let frame_count = u.int_in_range(3usize..=15usize)?;
        let mut sample_offset = 0u64; // only used when opt_block_size.is_none();
        for i in 0..frame_count {
            let channel_assignment = ChannelAssignment::cond_arbitrary(u, &(channel_count,))?;
            let frame_offset = if opt_block_size.is_none() {
                FrameOffset::StartSample(sample_offset)
            } else {
                FrameOffset::Frame(i as u32)
            };

            let frame = Frame::cond_arbitrary(
                u,
                &(
                    opt_block_size,
                    channel_assignment,
                    BitsPerSample(bits_per_sample),
                    SampleRate(sample_rate),
                    frame_offset,
                ),
            )?;
            stream.stream_info_mut().update_frame_info(&frame);
            sample_offset += frame.block_size() as u64;
        }

        Ok(stream)
    }
}

impl<'a> Arbitrary<'a> for MetadataBlockData {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        match u.int_in_range(0usize..=1usize)? {
            0 => StreamInfo::arbitrary(u).map(Into::into),
            1 => {
                let typetag = u.int_in_range(1u8..=126u8)?;
                // from specification, metadata size can be up to 4MiB.
                // However, since the logic behind `Unknown` is simple enough,
                // so I believe we don't need to test edge cases in terms of
                // sizes. Thus, here, the upper bound is limited to 64KiB.
                // Similarly, we just use a dummy zeroed buffer as contents.
                let datasize = u.int_in_range(1usize..=65535usize)?;
                Ok(MetadataBlockData::new_unknown(typetag, &vec![0u8; datasize]).unwrap())
            }
            _ => unreachable!(),
        }
    }
}

impl<'a> Arbitrary<'a> for StreamInfo {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let min_block_size = u.int_in_range(0u16..=65535u16)?;
        let max_block_size = u.int_in_range(min_block_size..=65535u16)?;
        let min_frame_size = u.int_in_range(1000u32..=0x00FFFFFFu32)?;
        let max_frame_size = u.int_in_range(min_frame_size..=0x00FFFFFFu32)?;
        let sample_rate = u.int_in_range(1u32..=655350u32)?;
        let channels = u.int_in_range(1u8..=8u8)?;
        let bits_per_sample: usize = BitsPerSample::arbitrary(u)?.into();
        let total_samples = 0;
        let md5 = [0u8; 16];

        let mut stream_info = StreamInfo::new(
            sample_rate as usize,
            channels as usize,
            bits_per_sample as usize,
        )
        .unwrap();
        let _ = stream_info.set_block_sizes(min_block_size as usize, max_block_size as usize);
        let _ = stream_info.set_frame_sizes(min_frame_size as usize, max_frame_size as usize);
        stream_info.set_total_samples(total_samples);
        stream_info.set_md5_digest(&md5);
        Ok(stream_info)
    }
}

impl<'a> CondArbitrary<'a> for Frame {
    type Condition = (
        Option<BlockSize>,
        ChannelAssignment,
        BitsPerSample,
        SampleRate,
        FrameOffset,
    );

    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let header = FrameHeader::cond_arbitrary(u, cond)?;
        let chs = header.channel_assignment();
        let mut subframes = Vec::with_capacity(chs.channels());
        let bps_base = header.bits_per_sample().unwrap();
        for ch in 0..chs.channels() {
            let bps = bps_base + chs.bits_per_sample_offset(ch);
            subframes.push(SubFrame::cond_arbitrary(
                u,
                &(
                    BlockSize(header.block_size()),
                    Arb(BitsPerSampleChLocal(bps)),
                ),
            )?);
        }
        Ok(Frame::from_parts(header, subframes))
    }
}

impl<'a> CondArbitrary<'a> for FrameHeader {
    type Condition = (
        Option<BlockSize>,
        ChannelAssignment,
        BitsPerSample,
        SampleRate,
        FrameOffset,
    );

    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (
            opt_block_size,
            chs,
            BitsPerSample(bits_per_sample),
            SampleRate(sample_rate),
            frame_offset,
        ) = cond;

        let block_size = match *opt_block_size {
            None => BlockSize::arbitrary(u)?.into(),
            Some(BlockSize(bs)) => bs,
        };

        Ok(FrameHeader::new(
            block_size,
            chs.clone(),
            *bits_per_sample,
            *sample_rate as usize,
            *frame_offset,
        )
        .unwrap())
    }
}

impl<'a> CondArbitrary<'a> for ChannelAssignment {
    type Condition = (ChannelCount,);
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (ChannelCount(channels),) = *cond;
        if channels == 2 {
            u.choose(&[
                ChannelAssignment::Independent(2),
                ChannelAssignment::LeftSide,
                ChannelAssignment::MidSide,
                ChannelAssignment::RightSide,
            ])
            .cloned()
        } else {
            Ok(ChannelAssignment::Independent(channels))
        }
    }
}

impl<'a> Arbitrary<'a> for FrameOffset {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(match u.choose(&[false, true])? {
            false => FrameOffset::Frame(u.int_in_range(0..=((1u32 << 31) - 1))?),
            true => FrameOffset::StartSample(u.int_in_range(0..=((1u64 << 36) - 1))?),
        })
    }
}

impl<'a> CondArbitrary<'a> for SubFrame {
    type Condition = (BlockSize, Arb<BitsPerSampleChLocal>);

    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (bs, arb_bps) = *cond;
        match u.int_in_range(0usize..=3usize)? {
            0 => Constant::cond_arbitrary(u, &(bs, arb_bps)).map(Into::into),
            1 => Verbatim::cond_arbitrary(u, &(bs, arb_bps)).map(Into::into),
            2 => {
                let order = FixedLpcOrder::arbitrary(u)?;
                FixedLpc::cond_arbitrary(u, &(order, bs, arb_bps)).map(Into::into)
            }
            3 => {
                let params = Arb::<QuantizedParameters>::arbitrary(u)?;
                Lpc::cond_arbitrary(u, &(params, bs, arb_bps)).map(Into::into)
            }
            _ => unreachable!(),
        }
    }
}

impl<'a> CondArbitrary<'a> for Constant {
    type Condition = (BlockSize, Arb<BitsPerSampleChLocal>);
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (BlockSize(block_size), Arb(BitsPerSampleChLocal(bits_per_sample))) = *cond;
        let dc_offset = u.int_in_range(signed_sample_range(bits_per_sample))?;
        Ok(Self::from_parts(
            block_size,
            dc_offset,
            bits_per_sample as u8,
        ))
    }
}

/// This is very slow as it allocates decoding buffer on-the-fly.
#[inline]
fn fit_in_bits<T: Decode>(component: &T, bits_per_sample: usize) -> bool {
    let range = signed_sample_range(bits_per_sample);
    let mut samples = vec![0i32; component.signal_len()];
    component.copy_signal(&mut samples);
    samples.iter().all(|x| range.contains(x))
}

impl<'a> CondArbitrary<'a> for FixedLpc {
    type Condition = (FixedLpcOrder, BlockSize, Arb<BitsPerSampleChLocal>);
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (
            FixedLpcOrder(order),
            BlockSize(block_size),
            Arb(BitsPerSampleChLocal(bits_per_sample)),
        ) = *cond;
        let mut warm_up = arbitrary_int_vec(u, order, signed_sample_range(bits_per_sample))?;

        let mut max_q = MaxQuotient::arbitrary(u)?;
        let mut max_p = MaxRiceP::arbitrary(u)?;
        let residual =
            Residual::cond_arbitrary(u, &(BlockSize(block_size), LpcOrder(order), max_q, max_p))?;

        let mut ret = Self::new(&warm_up, residual.clone(), bits_per_sample).unwrap();
        while fit_in_bits(&ret, bits_per_sample) {
            if max_q.into_inner() == 0 && max_p.into_inner() == 0 {
                warm_up.iter_mut().for_each(|p| *p /= 2);
            }
            max_q = MaxQuotient(max_q.into_inner() / 2);
            max_p = MaxRiceP(max_p.into_inner().saturating_sub(1));
            let residual = Residual::cond_arbitrary(
                u,
                &(BlockSize(block_size), LpcOrder(order), max_q, max_p),
            )?;
            ret = Self::new(&warm_up, residual.clone(), bits_per_sample).unwrap();
        }
        Ok(ret)
    }
}

impl<'a> CondArbitrary<'a> for Lpc {
    type Condition = (
        Arb<QuantizedParameters>,
        BlockSize,
        Arb<BitsPerSampleChLocal>,
    );
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (
            Arb(ref parameters),
            BlockSize(block_size),
            Arb(BitsPerSampleChLocal(bits_per_sample)),
        ) = *cond;
        let mut parameters = parameters.clone();
        let order = parameters.order();
        let mut warm_up = arbitrary_int_vec(u, order, signed_sample_range(bits_per_sample))?;

        let mut max_q = MaxQuotient::arbitrary(u)?;
        let mut max_p = MaxRiceP::arbitrary(u)?;
        let residual =
            Residual::cond_arbitrary(u, &(BlockSize(block_size), LpcOrder(order), max_q, max_p))?;

        let mut ret = Self::new(&warm_up, parameters.clone(), residual, bits_per_sample).unwrap();
        while fit_in_bits(&ret, bits_per_sample) {
            if max_q.into_inner() == 0 && max_p.into_inner() == 0 {
                warm_up.iter_mut().for_each(|p| *p /= 2);

                if warm_up.iter().all(|x| *x == 0) {
                    parameters.increment_shift();
                }
            }
            max_q = MaxQuotient(max_q.into_inner() / 2);
            max_p = MaxRiceP(max_p.into_inner().saturating_sub(1));
            let residual = Residual::cond_arbitrary(
                u,
                &(BlockSize(block_size), LpcOrder(order), max_q, max_p),
            )?;
            ret = Self::new(&warm_up, parameters.clone(), residual, bits_per_sample).unwrap();
        }
        Ok(ret)
    }
}

impl<'a> CondArbitrary<'a> for QuantizedParameters {
    type Condition = (LpcOrder, QlpcPrecision, QlpcShift);
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (LpcOrder(order), QlpcPrecision(precision), QlpcShift(shift)) = *cond;
        let coefs = arbitrary_int_vec(u, order, signed_sample_range(precision))?;
        Ok(Self::from_parts(&coefs, order, shift, precision))
    }
}

fn arbitrary_int_vec<T>(
    u: &mut Unstructured<'_>,
    size: usize,
    range: RangeInclusive<T>,
) -> Result<Vec<T>>
where
    T: arbitrary::unstructured::Int + PrimInt + Signed,
{
    let mut ret = Vec::with_capacity(size);
    for _i in 0..size {
        let x: T = u.int_in_range(range.clone())?;
        ret.push(x);
    }
    Ok(ret)
}

fn signed_sample_range<T>(bps: usize) -> RangeInclusive<T>
where
    T: PrimInt + Signed,
{
    let t_bits = std::mem::size_of::<T>() * 8;
    match t_bits.cmp(&bps) {
        Ordering::Equal => T::min_value()..=T::max_value(),
        Ordering::Less => {
            panic!("bps ({bps}) must be equal to or smaller than data type size ({t_bits})",);
        }
        Ordering::Greater => {
            let bps_minus_1 = bps - 1usize;
            let low = -(T::one() << bps_minus_1);
            let high = (T::one() << bps_minus_1) - T::one();
            low..=high
        }
    }
}

impl<'a> CondArbitrary<'a> for Verbatim {
    type Condition = (BlockSize, Arb<BitsPerSampleChLocal>);
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (BlockSize(block_size), Arb(BitsPerSampleChLocal(bits_per_sample))) = *cond;
        let samples = arbitrary_int_vec(u, block_size, signed_sample_range(bits_per_sample))?;
        Ok(Self::from_samples(&samples, bits_per_sample as u8))
    }
}

impl<'a> CondArbitrary<'a> for Residual {
    type Condition = (BlockSize, LpcOrder, MaxQuotient, MaxRiceP);
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
        let (BlockSize(block_size), LpcOrder(lpc_order), MaxQuotient(max_q), MaxRiceP(max_rice_p)) =
            *cond;

        let max_partition_order = block_size.trailing_zeros() as u8;
        let partition_order = u.int_in_range(0u8..=max_partition_order)?;
        let nparts = 2usize.pow(u32::from(partition_order));
        let part_len = block_size / nparts;

        let mut rice_params = vec![];
        let mut quotients: Vec<u32> = vec![];
        let mut remainders: Vec<u32> = vec![];
        for p in 0..nparts {
            let rice_p = u.int_in_range(0u8..=max_rice_p)?;
            rice_params.push(rice_p);
            for i in 0..part_len {
                let (q, r) = if p == 0 && i <= lpc_order {
                    (0, 0)
                } else {
                    (
                        u.int_in_range(0u8..=max_q)?.into(),
                        #[allow(clippy::range_minus_one)]
                        // `int_in_range` only supports inclusive range.
                        u.int_in_range(0u32..=(2u32.pow(rice_p.into()) - 1))?,
                    )
                };
                quotients.push(q);
                remainders.push(r);
            }
        }

        let ret = Self::from_parts(
            partition_order,
            block_size,
            lpc_order,
            rice_params,
            quotients,
            remainders,
        );
        ret.verify().unwrap();
        Ok(ret)
    }
}

fn assert_unary_fn_type<F, A1, R>(f: F) -> F
where
    F: Fn(A1) -> R,
{
    f
}
fn assert_binary_fn_type<F, A1, A2, R>(f: F) -> F
where
    F: Fn(A1, A2) -> R,
{
    f
}

macro_rules! common_impl_arbitrary_newtype {
    ($type_name:ty, $inner_type:ty) => {
        impl $type_name {
            pub fn new(inner: $inner_type) -> Self {
                Self(inner)
            }

            pub fn into_inner(self) -> $inner_type {
                self.0
            }
        }

        impl From<$inner_type> for $type_name {
            fn from(x: $inner_type) -> Self {
                Self(x)
            }
        }

        impl From<$type_name> for $inner_type {
            fn from(x: $type_name) -> Self {
                x.0
            }
        }
    };
}

macro_rules! def_arbitrary_newtype {
    ($type_name:ident, $inner_type:ty, $arb_body:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $type_name(pub $inner_type);
        impl<'a> Arbitrary<'a> for $type_name {
            fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
                let inner = assert_unary_fn_type::<_, &mut Unstructured<'_>, Result<$inner_type>>(
                    $arb_body,
                )(u)?;
                Ok(Self(inner))
            }
        }
        common_impl_arbitrary_newtype!($type_name, $inner_type);
    };
    ($type_name:ident, $inner_type:ty, $cond_type: ty, $arb_body:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $type_name(pub $inner_type);
        impl<'a> CondArbitrary<'a> for $type_name {
            type Condition = $cond_type;
            fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self> {
                let inner = assert_binary_fn_type::<
                    _,
                    &mut Unstructured<'_>,
                    &Self::Condition,
                    Result<$inner_type>,
                >($arb_body)(u, cond)?;
                Ok(Self(inner))
            }
        }
        common_impl_arbitrary_newtype!($type_name, $inner_type);
    };
}

def_arbitrary_newtype!(BitsPerSample, usize, |u| {
    u.choose(&[8usize, 16, 24]).copied()
});

def_arbitrary_newtype!(
    BitsPerSampleChLocal,
    usize,
    (BitsPerSample, bool),
    |_u, c| { Ok(c.0.into_inner() + usize::from(c.1)) }
);

def_arbitrary_newtype!(BlockSize, usize, |u| {
    u.int_in_range(1usize..=MAX_BLOCK_SIZE)
});

def_arbitrary_newtype!(ChannelCount, u8, |u| { u.int_in_range(1u8..=8u8) });

def_arbitrary_newtype!(FixedLpcOrder, usize, |u| {
    u.int_in_range(0usize..=4usize)
});

def_arbitrary_newtype!(LpcOrder, usize, |u| {
    u.int_in_range(0usize..=MAX_LPC_ORDER)
});

def_arbitrary_newtype!(MaxRiceP, u8, |u| {
    u.int_in_range(0u8..=(MAX_RICE_PARAMETER as u8))
});

def_arbitrary_newtype!(MaxQuotient, u8, |u| { u.int_in_range(0u8..=16u8) });

def_arbitrary_newtype!(QlpcPrecision, usize, |u| {
    u.int_in_range(1usize..=MAX_LPC_PRECISION)
});

def_arbitrary_newtype!(QlpcShift, i8, |u| {
    u.int_in_range(MIN_LPC_SHIFT..=MAX_LPC_SHIFT)
});

def_arbitrary_newtype!(SampleRate, u32, |u| { u.int_in_range(1u32..=655350u32) });
