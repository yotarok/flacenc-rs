#!/bin/bash
# Copyright 2023 Google LLC
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

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install --upgrade black flake8
pip install --upgrade pytype --ignore-requires-python

black --check --line-length=80 --diff testtool
pytype testtool | true
flake8 testtool

rustup run nightly \
    cargo bench --features simd-nightly --benches -- --skip "tests::" \
    > report/bench_results.txt

pushd flacenc-bin
rustup run stable cargo build --release
popd
python ./testtool/reporter.py
mv report/report.md report/report.stable.md

pushd flacenc-bin
rustup run nightly cargo build --release --features "simd-nightly"
popd
python ./testtool/reporter.py
mv report/report.md report/report.nightly.md

pushd flacenc-bin
cargo build --release --features pprof,simd-nightly
./target/release/flacenc --output ./tmp.flac --config ../report/st.config.toml \
    --pprof-output ./profile.pb \
    ../testwav/wikimedia.winter_kiss.wav && \
    pprof -pdf -output ../report/profile.st.pdf profile.pb
rm ./tmp.flac
popd
