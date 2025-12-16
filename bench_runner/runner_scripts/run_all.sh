#!/bin/sh
set -eux

export FLAC_VERSION="1.4.3"
export FLAC_RELEASE_DIR="https://ftp.osuosl.org/pub/xiph/releases/flac"
export PATH=$PATH:/root/.local/bin
export WIKIMEDIA_ROOT="https://upload.wikimedia.org/wikipedia/commons"
export REPORT_DIR="/report.$(date +%Y%m%d_%H%M%S)"
export DEST_BUCKET="flacenc-rs-benchmark-results"
export GCS_KEY_PATH="/secrets/gcs.json"

mkdir "${REPORT_DIR}"

git clone https://github.com/yotarok/flacenc-rs.git
cd flacenc-rs

export HEAD_DIGEST="$(git rev-parse --short=7 HEAD)"

## Collect System Information
bash /collect_host_info.sh | tee "${REPORT_DIR}/system_info.md"

## Download testwavs

mkdir testwav
curl "${WIKIMEDIA_ROOT}/9/97/%22I_Love_You%2C_California%22%2C_performed_by_the_Prince%27s_Orchestra_in_1914_for_Columbia_Records.oga" \
  | ffmpeg -i - testwav/wikimedia.i_love_you_california.wav

curl "${WIKIMEDIA_ROOT}/b/bd/Drozerix_-_A_Winter_Kiss.wav" \
  > testwav/wikimedia.winter_kiss.wav

curl "${WIKIMEDIA_ROOT}/7/7f/Jazz_Funk_no1_%28saxophone%29.flac" \
  | ffmpeg -i - testwav/wikimedia.jazz_funk_no1_sax.wav

curl "${WIKIMEDIA_ROOT}/5/5e/Albert_Roussel_-_Suite_en_Fa%2C_op.33_-_I._Pr%C3%A9lude.flac" \
  | ffmpeg -i - testwav/wikimedia.suite_en_fa_op_33_1.wav

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

if [[ -f ${GCS_KEY_PATH} ]] ; then
  gsutil \
    -o "Credentials:gs_service_key_file=${GCS_KEY_PATH}" \
    cp -r "${REPORT_DIR}" "gs://${DEST_BUCKET}/${HEAD_DIGEST}/$(date +%Y%m%d_%H%M%S)"
else
  gsutil \
    cp -r "${REPORT_DIR}" "gs://${DEST_BUCKET}/${HEAD_DIGEST}/$(date +%Y%m%d_%H%M%S)"
fi
