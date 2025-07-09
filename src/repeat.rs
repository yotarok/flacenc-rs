// Copyright 2023-2024 Google LLC
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

//! Utility for loop-unrolling.

use std::convert::Infallible;
use std::error::Error;

use seq_macro::seq;

macro_rules! repeat {
    ($counter:ident to $upto:expr => $body:block) => {
        <crate::repeat::Count<$upto> as crate::repeat::Repeat>::repeat(
            #[inline(always)]
            |$counter| $body,
        );
    };
    ($counter:ident to $upto:expr ; while $cond:expr => $body:block) => {
        <crate::repeat::Count<$upto> as crate::repeat::Repeat>::repeat_while(
            #[inline(always)]
            |$counter| $cond,
            #[inline(always)]
            |$counter| $body,
        )
    };
}
pub(crate) use repeat;

macro_rules! try_repeat {
    ($counter:ident to $upto:expr ; while $cond:expr => $body:block) => {
        <crate::repeat::Count<$upto> as crate::repeat::Repeat>::try_repeat_while(
            #[inline(always)]
            |$counter| $cond,
            #[inline(always)]
            |$counter| $body,
        )
    };
}
pub(crate) use try_repeat;

/// Repeat trait.
pub trait Repeat {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn repeat<F: FnMut(usize)>(body_fn: F) {
        Self::repeat_while(
            #[inline(always)]
            |_n| true,
            body_fn,
        );
    }
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn repeat_while<F: FnMut(usize), C: FnMut(usize) -> bool>(cond_fn: C, mut body_fn: F) {
        Self::try_repeat_while::<Infallible, _, _>(
            cond_fn,
            #[inline(always)]
            |n| {
                body_fn(n);
                Ok(())
            },
        )
        .unwrap();
    }
    fn try_repeat_while<E: Error, F: FnMut(usize) -> Result<(), E>, C: FnMut(usize) -> bool>(
        cond_fn: C,
        body_fn: F,
    ) -> Result<(), E>;
}

pub struct Count<const N: usize>;

seq!(M in 1..=32 {

    impl Repeat for Count<M> {
        #[allow(clippy::inline_always)]
        #[inline(always)]
        fn try_repeat_while<E: Error, F: FnMut(usize) -> Result<(), E>, C: FnMut(usize) -> bool>(mut cond_fn: C, mut body_fn: F) -> Result<(), E> {
            seq!(T in 0..M {
                if !cond_fn(T) {
                    return Ok(())
                }
                body_fn(T)?;
            });
            Ok(())
        }
    }

});
