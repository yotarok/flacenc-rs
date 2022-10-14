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

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::rc::Rc;

/// Enum of errors that can be returned in the encoder.
#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub enum EncodeError {
    Range(RangeError),
}

impl Error for EncodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(err) => err.fmt(f),
        }
    }
}

impl From<RangeError> for EncodeError {
    fn from(e: RangeError) -> Self {
        Self::Range(e)
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct RangeError {
    var: String,
    reason: String,
    actual: String,
}

/// Error object returned when a variable is out of supported range.
impl RangeError {
    pub fn from_display<T>(var: &str, reason: &str, actual: &T) -> Self
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

/// Error object returned when data integrity verification failed.
#[derive(Debug, Hash)]
#[allow(clippy::module_name_repetitions)]
pub struct VerifyError {
    components: Vec<String>,
    reason: String,
}

impl VerifyError {
    pub fn new(component: &str, reason: &str) -> Self {
        Self {
            components: vec![component.to_owned()],
            reason: reason.to_owned(),
        }
    }

    #[must_use]
    pub fn within(self, component: &str) -> Self {
        let mut components = self.components;
        let reason = self.reason;
        components.push(component.to_owned());
        Self { components, reason }
    }

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

pub trait Verify {
    /// Verifies there's no internal data inconsistency.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if there's an invalid variable.
    fn verify(&self) -> Result<(), VerifyError>;
}

#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct SourceError {
    source_name: Option<String>,
    reason: SourceErrorReason,
}

impl SourceError {
    pub const fn by_reason(reason: SourceErrorReason) -> Self {
        Self {
            source_name: None,
            reason,
        }
    }

    pub const fn from_unknown() -> Self {
        Self {
            source_name: None,
            reason: SourceErrorReason::IO(None),
        }
    }
    pub fn from_io_error<E: Error + 'static>(e: E) -> Self {
        Self {
            source_name: None,
            reason: SourceErrorReason::IO(Some(Rc::new(e))),
        }
    }

    #[must_use]
    pub fn set_path<P: AsRef<Path>>(self, path: P) -> Self {
        Self {
            source_name: Some(path.as_ref().to_string_lossy().to_string()),
            ..self
        }
    }
}

#[derive(Clone, Debug)]
pub enum SourceErrorReason {
    Open,
    InvalidBuffer,
    InvalidFormat,
    UnsupportedFormat,
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
