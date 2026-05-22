#!/bin/sh
set -eux

export SRC_PATH="/src.tar.gz"
export FLAC_VERSION="1.4.3"
export FLAC_RELEASE_DIR="https://ftp.osuosl.org/pub/xiph/releases/flac"
export PATH=$PATH:/root/.local/bin
export REPORT_DIR="/report.$(date +%Y%m%d_%H%M%S)"
export DEST_BUCKET="flacenc-rs-benchmark-results"
export GCS_KEY_PATH="/secrets/gcs.json"

mkdir "${REPORT_DIR}"

tar xvzf "${SRC_PATH}"
cd flacenc-rs

## Collect System Information
bash /collect_host_info.sh | tee "${REPORT_DIR}/system_info.md"

## Build FLAC reference encoder

rm -fr flac-${FLAC_VERSION}
curl "${FLAC_RELEASE_DIR}/flac-${FLAC_VERSION}.tar.xz" | tar -xJ
cd flac-${FLAC_VERSION}
./configure && make
cd ../

## Micro benchmarking (only for nightly)

rustup run nightly cargo bench \
    --features "experimental,par,simd-nightly" -- \
    --skip tests:: | tee "${REPORT_DIR}/bench_results.nightly.warmup.txt"

rustup run nightly cargo bench \
    --features "experimental,par,simd-nightly" -- \
    --skip tests:: | tee "${REPORT_DIR}/bench_results.nightly.txt"

cp -r ../testwav ./

## Integration benchmarking

cd flacenc-bin
rustup run stable cargo build --release
cd ..
uv run --project pytools python ./pytools/reporter.py \
       --flacbin flac-${FLAC_VERSION}/src/flac/flac \
       --testbin flacenc-bin/target/release/flacenc \
       --workdir report/out/nightly \
       --output ${REPORT_DIR}/report.stable.md

cd flacenc-bin
rustup run nightly cargo build --release --features "simd-nightly"
cd ..
uv run --project pytools python ./pytools/reporter.py \
       --flacbin flac-${FLAC_VERSION}/src/flac/flac \
       --testbin flacenc-bin/target/release/flacenc \
       --workdir report/out/nightly \
       --output ${REPORT_DIR}/report.nightly.md

## Upload

if [[ -f "${GCS_KEY_PATH}" ]] ; then
  gsutil \
    -o "Credentials:gs_service_key_file=${GCS_KEY_PATH}" \
    cp -r "${REPORT_DIR}" "gs://${DEST_BUCKET}/${REPO_DIGEST}/$(date +%Y%m%d_%H%M%S)"
else
  gsutil \
    cp -r "${REPORT_DIR}" "gs://${DEST_BUCKET}/${REPO_DIGEST}/$(date +%Y%m%d_%H%M%S)"
fi
