// Copyright 2022-2024 Google LLC
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

//! Error and verification traits

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::rc::Rc;

#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use super::bitsink::BitSink;

/// Enum of errors that can be returned while making an output bitstream.
#[derive(Clone, Eq, Hash, PartialEq)]
#[allow(clippy::module_name_repetitions)]
#[non_exhaustive]
pub enum OutputError<S>
where
    S: BitSink,
    S::Error: std::error::Error,
{
    /// A parameter in a component doesn't fit in a format.
    Range(RangeError),
    /// I/O error propagated from [`BitSink`].
    Sink(S::Error),
}

impl<S> OutputError<S>
where
    S: BitSink,
    S::Error: std::error::Error,
{
    #[inline]
    pub(crate) const fn from_sink(e: S::Error) -> Self {
        Self::Sink(e)
    }

    pub(crate) fn ignore_sink_error<U>(err: OutputError<U>) -> Self
    where
        U: BitSink<Error = Infallible>,
    {
        match err {
            OutputError::Range(e) => Self::Range(e),
            #[allow(unreachable_patterns)]
            // There's a subtle incompatibility between stable `rustc` and nightly `clippy`
            // (@2024-08-12), `rustc` doesn't allow incomplete match arms even if a enum variant
            // is uninhabitated, whereas `clippy` warns if a match arm is unreachable.
            OutputError::Sink(_) => unreachable!(),
        }
    }
}

impl<S> Error for OutputError<S>
where
    S: BitSink,
    S::Error: Error,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl<S> fmt::Display for OutputError<S>
where
    S: BitSink,
    S::Error: std::error::Error,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(err) => err.fmt(f),
            Self::Sink(err) => err.fmt(f),
        }
    }
}

impl<S> fmt::Debug for OutputError<S>
where
    S: BitSink,
    S::Error: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(err) => f
                .debug_tuple("OutputError::InvalidRange")
                .field(&err)
                .finish(),
            Self::Sink(err) => f.debug_tuple("OutputError::Sink").field(&err).finish(),
        }
    }
}

impl<S> From<RangeError> for OutputError<S>
where
    S: BitSink,
    S::Error: fmt::Debug,
{
    fn from(e: RangeError) -> Self {
        Self::Range(e)
    }
}

/// Error emitted when a parameter is out of the expected range.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct RangeError {
    var: String,
    reason: String,
    actual: String,
}

/// Error object returned when a variable is out of supported range.
impl RangeError {
    /// Makes range error from `actual: impl Display` that is out of range.
    pub(crate) fn from_display<T>(var: &str, reason: &str, actual: &T) -> Self
    where
        T: fmt::Display,
    {
        Self {
            var: var.to_owned(),
            reason: reason.to_owned(),
            actual: format!("{actual}"),
        }
    }
}

impl Error for RangeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is out of range: {} (actual={})",
            self.var, self.reason, self.actual
        )
    }
}

/// Error object returned when config integrity verification failed.
///
/// This error maintains a path to the component that is actually erroneous
/// in the nested components.
///
/// # Design Notes
///
/// Parameter verification should happen on the public API boundary. When a
/// component (e.g. `FixedLpc`) is constructed via `new` from the outside of
/// this crate, new primitive parameters for `FixedLpc::new` should be
/// verified. However, the `Residual` parameter for `new` will not be verified
/// because it can be assumed to be verified in `Residual::new`, and redundant
/// verification should be avoided for efficiency. A rule of thumb is that any
/// argument implements `Verify` given to the public API is assumed to be
/// verified, and the called function should verify the remaining parameters
/// and interaction between the provided parameters.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct VerifyError {
    components: Vec<String>,
    reason: String,
}

impl VerifyError {
    /// Makes verification error for an invalid variable `component`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = VerifyError::new("order", "must be non-negative");
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "verification error: `order` is not valid. reason: must be non-negative"
    /// );
    pub fn new(component: &str, reason: &str) -> Self {
        Self {
            components: vec![component.to_owned()],
            reason: reason.to_owned(),
        }
    }

    /// Prepends the name of an enclosing component to the error location.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = VerifyError::new("order", "must be non-negative");
    /// let err = err.within("encoder");
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "verification error: `encoder.order` is not valid. reason: must be non-negative"
    /// );
    /// ```
    #[must_use]
    pub fn within(self, component: &str) -> Self {
        let mut components = self.components;
        let reason = self.reason;
        components.push(component.to_owned());
        Self { components, reason }
    }

    /// Gets dot-separated path string for the error location.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = VerifyError::new("order", "must be non-negative");
    /// let err = err.within("encoder");
    /// assert_eq!(err.path(), "encoder.order");
    /// ```
    pub fn path(&self) -> String {
        let mut path = String::new();
        for (i, name) in self.components.iter().rev().enumerate() {
            if i != 0 {
                path.push('.');
            }
            path.push_str(name);
        }
        path
    }
}

impl Error for VerifyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "verification error: `{}` is not valid. reason: {}",
            self.path(),
            self.reason
        )
    }
}

/// A wrapper that ensures that the inner `T` is verified and unchanged.
///
/// `Verified<T>` can be obtained via [`Verify::into_verified`] or
/// [`Verify::assume_verified`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Verified<T>(T);

impl<T> std::ops::Deref for Verified<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Trait for verifiable structs.
pub trait Verify: Sized + seal_verify::Sealed {
    /// Verifies there's no internal data inconsistency.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if there's an invalid variable.
    ///
    /// # Examples
    ///
    /// [`config::Prc`] implements `Verify`.
    ///
    /// [`config::Prc`]: crate::config::Prc
    ///
    /// ```
    /// # use flacenc::error::*;
    /// # use flacenc::config::Prc;
    /// let mut prc = Prc::default();
    /// prc.max_parameter = 256;  // invalid setting
    /// assert!(prc.verify().is_err());
    ///
    /// prc.max_parameter = 10; // valid setting
    /// assert!(prc.verify().is_ok());
    /// ```
    fn verify(&self) -> Result<(), VerifyError>;

    /// Wraps into `Verified` to indicate that the data is already verified.
    ///
    /// # Errors
    ///
    /// Returns the original input and `VerifyError` if `verify` failed.
    ///
    /// # Examples
    ///
    /// [`config::Prc`] implements `Verify`.
    ///
    /// [`config::Prc`]: crate::config::Prc
    ///
    /// ```
    /// # use flacenc::error::*;
    /// # use flacenc::config::Prc;
    /// # use flacenc::constant::rice::MAX_RICE_PARAMETER;
    /// let mut prc = Prc::default();
    /// fn do_something_with_prc_config(config: &Verified<Prc>) -> usize {
    ///     if config.max_parameter > MAX_RICE_PARAMETER {
    ///         panic!();
    ///     }
    ///     // do something
    ///     config.max_parameter * 2
    /// }
    ///
    /// prc.max_parameter = 256;  // invalid setting
    /// assert!(
    ///     prc.clone()
    ///        .into_verified()
    ///        .map(|x| do_something_with_prc_config(&x))
    ///        .is_err()
    /// );  // at least this does not panic
    ///
    /// prc.max_parameter = 7;
    /// assert_eq!(
    ///     prc.into_verified()
    ///        .map(|x| do_something_with_prc_config(&x))
    ///        .unwrap(),
    ///     14,
    /// );
    /// ```
    fn into_verified(self) -> Result<Verified<Self>, (Self, VerifyError)> {
        let result = self.verify();
        if let Err(e) = result {
            Err((self, e))
        } else {
            Ok(Verified(self))
        }
    }

    /// Wraps into `Verified` without actual verification.
    ///
    /// # Safety
    ///
    /// The use of `Verified` data obtained this way may cause an unexpected
    /// behavior. It should be okay if the data are previously verified with
    /// `verify` function and have not been changed after that.
    ///
    /// # Examples
    ///
    /// [`config::Prc`] implements `Verify`.
    ///
    /// [`config::Prc`]: crate::config::Prc
    ///
    /// ```should_panic
    /// # use flacenc::error::*;
    /// # use flacenc::config::Prc;
    /// # use flacenc::constant::rice::MAX_RICE_PARAMETER;
    /// fn do_something_with_prc_config(config: &Verified<Prc>) -> usize {
    ///     if config.max_parameter > MAX_RICE_PARAMETER {
    ///         panic!();
    ///     }
    ///     // do something
    ///     config.max_parameter * 2
    /// }
    /// let mut prc = Prc::default();
    /// prc.max_parameter = 256;
    /// // Should compile but panic.
    /// unsafe {
    ///     do_something_with_prc_config(&prc.assume_verified());
    /// }
    /// ```
    unsafe fn assume_verified(self) -> Verified<Self> {
        Verified(self)
    }
}

/// A wrapping function to make it compatible with "?" operator.
pub(crate) fn verify_macro_impl(cond: bool, varname: &str, msg: &str) -> Result<(), VerifyError> {
    if !cond {
        return Err(VerifyError::new(varname, msg));
    }
    Ok(())
}

/// Checks if `$cond` is true and do `return Err(...)` if so.
///
/// An error object `VerifyErr` is constructed using `$varname` and
/// `$msg` that are formatted using the extra args (`$args`).
macro_rules! verify_true {
    ($varname:literal, $cond:expr, $msg:literal, $($args: expr),*) => {
        crate::error::verify_macro_impl(
            $cond,
            &format!($varname, $($args),*),
            &format!($msg, $($args),*),
        )
    };
    ($varname:literal, $cond:expr, $msg:literal) => {
        verify_true!($varname, $cond, $msg,)
    }
}
pub(crate) use verify_true;

/// Checks if `$actual` is in the range, and emits err with default msgs if not.
///
/// An error is constructed using the same way as [`verify_true`].
macro_rules! verify_range {
    ($varname: literal, $actual:expr, $lowlimit:tt .. $highlimit:tt) => {
        verify_range!($varname, $actual, ($lowlimit)..)
            .and_then(|()| verify_range!($varname, $actual, ..($highlimit)))
    };
    ($varname: literal, $actual:expr, $lowlimit:tt ..= $highlimit:tt) => {
        verify_range!($varname, $actual, ($lowlimit)..)
            .and_then(|()| verify_range!($varname, $actual, ..=($highlimit)))
    };
    ($varname: literal, $actual:expr, $lowlimit:tt ..) => {{
        #[allow(unused_parens)]
        let limit = $lowlimit;
        verify_true!(
            $varname,
            $actual >= limit,
            "must be greater than or equal to {limit}"
        )
    }};
    ($varname: literal, $actual:expr, ..= $highlimit:tt) => {{
        #[allow(unused_parens)]
        let limit = $highlimit;
        verify_true!(
            $varname,
            $actual <= limit,
            "must be less than or equal to {limit}"
        )
    }};
    ($varname: literal, $actual:expr, .. $highlimit:tt) => {{
        #[allow(unused_parens)]
        let limit = $highlimit;
        verify_true!($varname, $actual < limit, "must be less than {limit}")
    }};
}
pub(crate) use verify_range;

/// Enum for possible encoder errors.
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug)]
pub enum EncodeError {
    /// Encoder errors due to input sources.
    Source(SourceError),
    /// Encoder errors due to invalid configuration.
    Config(VerifyError),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::Source(e) => e.fmt(f),
            Self::Config(e) => e.fmt(f),
        }
    }
}

impl Error for EncodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(e) => e.source(),
            Self::Config(e) => e.source(),
        }
    }
}

impl From<SourceError> for EncodeError {
    fn from(e: SourceError) -> Self {
        Self::Source(e)
    }
}

impl From<VerifyError> for EncodeError {
    fn from(e: VerifyError) -> Self {
        Self::Config(e)
    }
}

/// Struct that wraps errors from [`Source`].
///
/// [`Source`]: crate::source::Source
#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct SourceError {
    source_name: Option<String>,
    reason: SourceErrorReason,
}

impl SourceError {
    /// Constructs `SourceError` by choosing a reason.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = SourceError::by_reason(SourceErrorReason::Open);
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occurred while reading <unknown>. reason: cannot open file."
    /// );
    /// ```
    pub const fn by_reason(reason: SourceErrorReason) -> Self {
        Self {
            source_name: None,
            reason,
        }
    }

    /// Constructs `SourceError` with unknown (hidden) reason.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = SourceError::from_unknown();
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occurred while reading <unknown>. reason: unknown I/O error."
    /// );
    /// ```
    pub const fn from_unknown() -> Self {
        Self {
            source_name: None,
            reason: SourceErrorReason::IO(None),
        }
    }

    /// Constructs `SourceError` from an [`io::Error`].
    ///
    /// [`io::Error`]: std::io::Error
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// # use std::io;
    /// let err = SourceError::from_io_error(io::Error::new(io::ErrorKind::Other, "oh no!"));
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occurred while reading <unknown>. reason: I/O error: oh no!."
    /// );
    /// ```
    pub fn from_io_error<E: Error + 'static>(e: E) -> Self {
        Self {
            source_name: None,
            reason: SourceErrorReason::IO(Some(Rc::new(e))),
        }
    }

    /// Set path as the source name (informative when [`Source`] is file-based.)
    ///
    /// [`Source`]: crate::source::Source
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = SourceError::by_reason(SourceErrorReason::Open);
    /// let err = err.set_path("missing.wav");
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occurred while reading missing.wav. reason: cannot open file."
    /// );
    /// ```
    #[must_use]
    pub fn set_path<P: AsRef<Path>>(self, path: P) -> Self {
        Self {
            source_name: Some(path.as_ref().to_string_lossy().to_string()),
            ..self
        }
    }
}

/// Enum covering possible error reasons from [`Source`].
///
/// [`Source`]: crate::source::Source
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SourceErrorReason {
    /// The source file cannot be opened.
    Open,
    /// [`FrameBuf`] is not properly prepared.
    ///
    /// [`FrameBuf`]: crate::source::FrameBuf
    InvalidBuffer,
    /// The content of file is not readable.
    InvalidFormat,
    /// Type of file is not supported.
    UnsupportedFormat,
    /// Other IO-related error.
    IO(Option<Rc<dyn Error + 'static>>),
}

impl Error for SourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error occurred while reading {}. reason: {}.",
            self.source_name
                .as_ref()
                .map_or("<unknown>", String::as_str),
            self.reason
        )
    }
}

impl fmt::Display for SourceErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => {
                write!(f, "cannot open file")
            }
            Self::InvalidBuffer => {
                write!(f, "buffer is invalid")
            }
            Self::InvalidFormat => {
                write!(f, "source format is invalid")
            }
            Self::UnsupportedFormat => {
                write!(f, "source format is not supported")
            }
            Self::IO(Some(cause)) => {
                write!(f, "I/O error: {cause}")
            }
            Self::IO(None) => {
                write!(f, "unknown I/O error")
            }
        }
    }
}

mod seal_verify {
    pub trait Sealed {}

    impl Sealed for crate::component::ChannelAssignment {}
    impl Sealed for crate::component::MetadataBlockData {}
    impl Sealed for crate::component::SubFrame {}
    impl Sealed for crate::component::Constant {}
    impl Sealed for crate::component::FixedLpc {}
    impl Sealed for crate::component::Frame {}
    impl Sealed for crate::component::FrameHeader {}
    impl Sealed for crate::component::Lpc {}
    impl Sealed for crate::component::MetadataBlock {}
    impl Sealed for crate::component::QuantizedParameters {}
    impl Sealed for crate::component::Residual {}
    impl Sealed for crate::component::Stream {}
    impl Sealed for crate::component::StreamInfo {}
    impl Sealed for crate::component::Verbatim {}
    impl Sealed for crate::config::Encoder {}
    impl Sealed for crate::config::Fixed {}
    impl Sealed for crate::config::Prc {}
    impl Sealed for crate::config::Qlpc {}
    impl Sealed for crate::config::StereoCoding {}
    impl Sealed for crate::config::SubFrameCoding {}
    impl Sealed for crate::config::OrderSel {}
    impl Sealed for crate::config::Window {}
}
