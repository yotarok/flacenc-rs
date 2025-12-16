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

REPO_DIGEST="$(git rev-parse --short=7 HEAD)"
podman build --platform=linux/amd64 -t "${GCP_REPOSITORY}/${GCP_PROJECT}/benchrunner/benchrunner:${REPO_DIGEST}" .
podman push "${GCP_REPOSITORY}/${GCP_PROJECT}/benchrunner/benchrunner:${REPO_DIGEST}"

gcloud compute instances create-with-container benchrunner-vm \
  --zone "${GCP_ZONE}" \
  --machine-type e2-standard-8 \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --service-account "${GCP_SERVICE_ACCOUNT}" \
  --container-image asia-northeast1-docker.pkg.dev/${GCP_PROJECT}/benchrunner/benchrunner:${REPO_DIGEST} \
  --container-restart-policy never \
  --project ${GCP_PROJECT} \
  --metadata=google-logging-enabled=true \
  --scopes cloud-platform \
  --metadata-from-file=startup-script=runner_scripts/gce_auto_delete.sh
