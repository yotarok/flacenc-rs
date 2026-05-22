#!/bin/sh

set -eu

if [[ -f .env ]] ; then
  source .env
fi

if [[ ! -n "$GCP_PROJECT" ]] ; then
  cat <<EOF
Put ".env" file with the following content in the same directory with this script:

GCP_ZONE=asia-northeast1-a
GCP_SERVICE_ACCOUNT=[[SERVICE_ACCOUNT]]
GCP_PROJECT=[[GCP_PROJECT]]
GCP_REPOSITORY=asia-northeast1-docker.pkg.dev
EOF
  exit 1
fi
echo "Project: ${GCP_PROJECT}"

pushd $(git rev-parse --show-toplevel)
git archive --format=tgz --prefix=flacenc-rs/ -o bench_runner/src.tar.gz HEAD
popd

REPO_DIGEST="$(git rev-parse --short=7 HEAD)"

## Download testwavs
if [[ ! -d .testwav ]] ; then
  export WIKIMEDIA_ROOT="https://upload.wikimedia.org/wikipedia/commons"
  mkdir .testwav/
  curl -sSL "${WIKIMEDIA_ROOT}/9/97/%22I_Love_You%2C_California%22%2C_performed_by_the_Prince%27s_Orchestra_in_1914_for_Columbia_Records.oga" | ffmpeg -i - .testwav/wikimedia.i_love_you_california.wav
  curl -sSL "${WIKIMEDIA_ROOT}/b/bd/Drozerix_-_A_Winter_Kiss.wav" > .testwav/wikimedia.winter_kiss.wav 
  curl -sSL "${WIKIMEDIA_ROOT}/7/7f/Jazz_Funk_no1_%28saxophone%29.flac" | ffmpeg -i - .testwav/wikimedia.jazz_funk_no1_sax.wav
  curl -sSL "${WIKIMEDIA_ROOT}/5/5e/Albert_Roussel_-_Suite_en_Fa%2C_op.33_-_I._Pr%C3%A9lude.flac" | ffmpeg -i - .testwav/wikimedia.suite_en_fa_op_33_1.wav
fi

podman build --platform=linux/amd64 --build-arg "REPO_DIGEST=${REPO_DIGEST}" -t "${GCP_REPOSITORY}/${GCP_PROJECT}/benchrunner/benchrunner:${REPO_DIGEST}" .
podman push "${GCP_REPOSITORY}/${GCP_PROJECT}/benchrunner/benchrunner:${REPO_DIGEST}"

gcloud compute instances create-with-container benchrunner-vm \
  --zone "${GCP_ZONE}" \
  --machine-type c2-standard-8 \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --service-account "${GCP_SERVICE_ACCOUNT}" \
  --container-image asia-northeast1-docker.pkg.dev/${GCP_PROJECT}/benchrunner/benchrunner:${REPO_DIGEST} \
  --container-restart-policy never \
  --project ${GCP_PROJECT} \
  --metadata=google-logging-enabled=true \
  --scopes cloud-platform \
  --metadata-from-file=startup-script=runner_scripts/gce_auto_delete.sh
