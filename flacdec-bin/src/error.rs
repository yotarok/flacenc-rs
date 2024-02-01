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

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;

use flacenc::error::VerifyError;

/// Decoder error.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Error {
    /// A variant that indicates a format error in the input stream.
    Format(FormatError),
    /// A variant that indicates out-of-range of the parsed parameters.
    Verify(VerifyError),
    /// A variant that indicates a (premature) end of the input stream.
    StreamEnded,
}

impl Error {
    /// Constructs an error object that indicates the end of input stream.
    pub const fn stream_ended() -> Self {
        Self::StreamEnded
    }

    /// Returns true if error is due to the end of input.
    ///
    /// `StreamEnded` error can be ignored if it is happened on the frame
    /// boundary.
    pub const fn is_stream_ended(&self) -> bool {
        match self {
            Self::StreamEnded => true,
            Self::Format(_) | Self::Verify(_) => false,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

impl From<FormatError> for Error {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

impl From<VerifyError> for Error {
    fn from(e: VerifyError) -> Self {
        Self::Verify(e)
    }
}

/// An error type for input format error.
#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct FormatError {
    /// The location of error in bit offset.
    location_in_bits: usize,
    /// Message that described the reason.
    message: String,
}

impl FormatError {
    /// Constructs new `FormatError`.
    pub fn new(location_in_bits: usize, message: &str) -> Self {
        Self {
            location_in_bits,
            message: message.to_owned(),
        }
    }
}

impl Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "input format error detected at b={}. (reason={})",
            self.location_in_bits, self.message
        )
    }
}
impl std::error::Error for FormatError {}
