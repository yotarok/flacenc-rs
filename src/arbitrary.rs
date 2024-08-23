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

//! Implementation of `Arbitrary` trait for `components`.
//!
//! Arbitrary is implemented primarily for fuzz testing but also for unit tests. Some
//! implementations do not cover all possible realization of the component; however, it's still
//! useful for obtaining a single valid example of the data.

use std::cell::RefCell;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::Rng;

thread_local! {
    static RANDOM_BUFFER: RefCell<Vec<u8>> = RefCell::default();
}

/// Extends the global (thread-local) random buffer.
fn extend_random_buffer<R: Rng>(mut rng: R, target_len: usize) -> usize {
    assert!(target_len < 1_000_000_000);
    RANDOM_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        while buf.len() < target_len {
            buf.push(Uniform::from(0u8..=255u8).sample(&mut rng));
        }
        buf.len()
    })
}

/// Calls a function with `Unstructured` constructed from the global random buffer.
fn with_random_unstructured<T, F>(f: F) -> T
where
    F: for<'a> FnOnce(&mut Unstructured<'a>) -> T,
    T: 'static,
{
    RANDOM_BUFFER.with(|cell| {
        let b = cell.borrow_mut();
        let mut u = Unstructured::new(&b);
        f(&mut u)
    })
}

/// Performs random sampling of `Arbitrary` data-type.
///
/// # Errors
///
/// This function propagates [`arbitrary::Result`] reported while performing
/// [`Arbitrary::arbitrary`].
pub fn random_sample<T, C, R>(mut rng: R, cond: &C) -> Result<T>
where
    R: Rng,
    T: 'static + for<'a> CondArbitrary<'a, Condition = C>,
{
    let mut target_len = 1024;
    target_len = extend_random_buffer(&mut rng, target_len);
    let mut result = with_random_unstructured(|u| T::cond_arbitrary(u, cond));

    while matches!(result, Err(arbitrary::Error::NotEnoughData)) {
        target_len *= 2;
        extend_random_buffer(&mut rng, target_len);
        result = with_random_unstructured(|u| T::cond_arbitrary(u, cond));
    }
    result
}

#[allow(clippy::module_name_repetitions)]
pub trait CondArbitrary<'a>: Sized {
    /// Type for variables that needs to be fixed prior to generate `Self`.
    type Condition: 'static + Sized;

    /// Generates an arbitrary value as similar to [`Arbitrary::arbitrary`], but with a parameter.
    ///
    /// # Errors
    ///
    /// See [`Arbitrary::arbitrary`] for error details.
    fn cond_arbitrary(u: &mut Unstructured<'a>, cond: &Self::Condition) -> Result<Self>;
}

// `T: Arbitrary` is `CondArbitrary` with `Condition=()`.
//
// Theoretically, the reverse is also true, i.e. `T: CondArbitrary<Condition=()>` is `Arbitrary`.
// However, we don't implement it to ensure no method overlap, and one can explicitly use `T`
// as `Arbitrary` by wrapping it as `Arb<T>`.

impl<'a, T> CondArbitrary<'a> for T
where
    T: Arbitrary<'a>,
{
    type Condition = ();
    fn cond_arbitrary(u: &mut Unstructured<'a>, _cond: &Self::Condition) -> Result<Self> {
        Self::arbitrary(u)
    }
}

/// Internal wrapper struct for adding utilities to `T: CondArbitrary`.
#[derive(Clone, Debug)]
pub struct Arb<T>(pub T);

impl<T: Copy> Copy for Arb<T> {}

impl<T> Arb<T> {
    /// Returns inner type wrapped in `Arb`.
    pub fn into_inner(self) -> T {
        self.0
    }

    /// Generates a random sample for `T`.
    ///
    /// This function is for test and will panic when it failed to produce a sample.
    ///
    /// # Panics
    ///
    /// It panics when the underlying call of [`Arbitrary::arbitrary`] failed.
    pub fn random_test_sample<R: Rng>(rng: R) -> T
    where
        Self: for<'a> Arbitrary<'a>,
        T: 'static + Sized,
    {
        random_sample(rng, &()).map(Self::into_inner).unwrap()
    }
}

impl<'a, T> Arbitrary<'a> for Arb<T>
where
    T: CondArbitrary<'a, Condition: Arbitrary<'a>>,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let cond = T::Condition::arbitrary(u)?;
        Ok(Self(T::cond_arbitrary(u, &cond)?))
    }
}
