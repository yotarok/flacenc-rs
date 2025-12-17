COMMIT_HASH="${1:-$(git rev-parse --short=7 HEAD)}"

source .env
LATEST_RESULT="$(gsutil ls -d "gs://flacenc-rs-benchmark-results/${COMMIT_HASH}/*/" | sort | tail -n 1)"

echo "Retrieve results from ${LATEST_RESULT}"
DEST="../report"

gsutil cp "${LATEST_RESULT}bench_results.nightly.txt" "${DEST}/bench_results.txt"
gsutil cp "${LATEST_RESULT}report.nightly.md" "${DEST}/report.nightly.md"
gsutil cp "${LATEST_RESULT}report.stable.md" "${DEST}/report.stable.md"
gsutil cp "${LATEST_RESULT}system_info.md" "${DEST}/system_info.md"
