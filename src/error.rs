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
            "Verification error: `{}` is not valid. Reason: {}",
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
