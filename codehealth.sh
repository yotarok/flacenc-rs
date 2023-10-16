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

rustup default nightly
cargo fmt --check
cargo rustc --release -- -D warnings
cargo build --release
cargo test --features "experimental,par,simd-nightly"
cargo clippy --tests -- -D warnings
cargo doc

pushd flacenc-bin
cargo fmt --check
cargo test
cargo clippy --tests -- -D warnings
popd

rustup default stable
cargo rustc --release --no-default-features --features "par,experimental" -- -D warnings
cargo build --release --no-default-features --features "par,experimental"
cargo clippy --tests --no-default-features --features "par,experimental"
cargo test --no-default-features --features "experimental,par"
rustup default nightly
