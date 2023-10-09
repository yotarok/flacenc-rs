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

//! Error and verification traits

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::rc::Rc;

use super::bitsink::BitSink;

/// Enum of errors that can be returned in the encoder.
#[derive(Clone, Eq, Hash, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub enum OutputError<S>
where
    S: BitSink,
    S::Error: std::error::Error,
{
    /// A parameter in a component doesn't fit in a format.
    Range(RangeError),
    /// I/O error propagated from `BitSink`.
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
/// This error maintains a path to the component that is actually errorneous
/// in the nested components.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct VerifyError {
    components: Vec<String>,
    reason: String,
}

impl VerifyError {
    /// Makes range error from `actual: impl Display` that is out of range.
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

/// Trait for verifiable structs.
pub trait Verify {
    /// Verifies there's no internal data inconsistency.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if there's an invalid variable.
    fn verify(&self) -> Result<(), VerifyError>;
}

/// Enum for possible encoder errors.
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug)]
pub enum EncodeError {
    Source(SourceError),
}

impl From<SourceError> for EncodeError {
    fn from(e: SourceError) -> Self {
        Self::Source(e)
    }
}

/// Struct that wraps errors from `Source`.
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
    ///     "error occured while reading <unknown>. reason: cannot open file."
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
    /// let err = SourceError::by_reason(SourceErrorReason::Open);
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occured while reading <unknown>. reason: cannot open file."
    /// );
    /// ```
    pub const fn from_unknown() -> Self {
        Self {
            source_name: None,
            reason: SourceErrorReason::IO(None),
        }
    }

    /// Constructs `SourceError` from an `io::Error`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// # use std::io;
    /// let err = SourceError::from_io_error(io::Error::new(io::ErrorKind::Other, "oh no!"));
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occured while reading <unknown>. reason: I/O error: oh no!."
    /// );
    /// ```
    pub fn from_io_error<E: Error + 'static>(e: E) -> Self {
        Self {
            source_name: None,
            reason: SourceErrorReason::IO(Some(Rc::new(e))),
        }
    }

    /// Set path as the source name (informative when `Source` is file-based.)
    ///
    /// # Examples
    ///
    /// ```
    /// # use flacenc::error::*;
    /// let err = SourceError::by_reason(SourceErrorReason::Open);
    /// let err = err.set_path("missing.wav");
    /// assert_eq!(
    ///     format!("{}", err),
    ///     "error occured while reading missing.wav. reason: cannot open file."
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

/// Enum covering possible error reasons from `Source`.
#[derive(Clone, Debug)]
pub enum SourceErrorReason {
    /// The source file cannot be opened.
    Open,
    /// `FrameBuf` is not properly prepared.
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
            "error occured while reading {}. reason: {}.",
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
