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
cargo rustc --release -- -D warnings
cargo build --release
cargo test --features "test_helper,experimental,par"
cargo fmt --check
cargo clippy --tests -- -D warnings
cargo doc

pushd flacenc-bin
cargo test
cargo fmt --check
cargo clippy --tests -- -D warnings
popd

rustup default stable
cargo rustc --release --features "fakesimd" -- -D warnings
cargo build --release --features "fakesimd"
cargo clippy --tests --features "fakesimd"
cargo test --features "fakesimd,test_helper,experimental,par"
rustup default nightly
