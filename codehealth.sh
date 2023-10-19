#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

FEATURES_NIGHTLY="experimental,par,simd-nightly"
FEATURES_BIN_NIGHTLY="simd-nightly"
FEATURES_STABLE="experimental,par"

rustup run nightly cargo fmt --check
rustup run nightly cargo clippy --tests --features "${FEATURES_NIGHTLY}" -- -D warnings
RUSTFLAGS="-D warnings" rustup run nightly cargo test --features "${FEATURES_NIGHTLY}"
rustup run nightly cargo doc

pushd flacenc-bin
rustup run nightly cargo fmt --check
rustup run nightly cargo clippy --tests --features "${FEATURES_BIN_NIGHTLY}" -- -D warnings
RUSTFLAGS="-D warnings" rustup run nightly cargo test --features "${FEATURES_BIN_NIGHTLY}"
popd

rustup run stable cargo clippy --tests --features "${FEATURES_STABLE}" -- -D warnings
RUSTFLAGS="-D warnings" rustup run stable cargo test --features "${FEATURES_STABLE}"
