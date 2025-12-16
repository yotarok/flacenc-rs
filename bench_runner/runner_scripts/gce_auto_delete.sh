#!/bin/bash

# --- logging ---
exec > /var/log/startup-script.log 2>&1
echo "[startup-script] begin"

# --- paths ---
WATCHER=/etc/systemd/system/container-watcher.sh
SERVICE=/etc/systemd/system/container-watcher.service

# --- install watcher script ---
cat <<'EOF' > "$WATCHER"
#!/bin/bash
set -u

LOG=/var/log/container-watcher.log
exec >> "$LOG" 2>&1
echo "[watcher] started"

# wait for docker socket (containerd/docker bridge)
for i in $(seq 1 120); do
  if docker info >/dev/null 2>&1; then
    echo "[watcher] docker is ready"
    break
  fi
  sleep 1
done

# wait for klt-* container to appear
while true; do
  CONTAINER=$(docker ps -a --format '{{.Names}}' | grep '^klt-' || true)
  if [ -n "$CONTAINER" ]; then
    echo "[watcher] found container: $CONTAINER"
    break
  fi
  sleep 1
done

# wait until container exits
docker wait "$CONTAINER"
STATUS=$?
echo "[watcher] container exited with status $STATUS"

# small delay for log flush
sleep 5

# metadata
META=http://metadata.google.internal/computeMetadata/v1
HDR="Metadata-Flavor: Google"

NAME=$(curl -sf -H "$HDR" $META/instance/name)
ZONE=$(curl -sf -H "$HDR" $META/instance/zone | awk -F/ '{print $NF}')
PROJECT=$(curl -sf -H "$HDR" $META/project/project-id)

echo "[watcher] deleting instance $NAME ($ZONE)"

# delete self without using `gcloud`
TOKEN=$(curl -sf \
  -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p')
curl -sf -X DELETE \
  -H "Authorization: Bearer $TOKEN" \
  "https://compute.googleapis.com/compute/v1/projects/$PROJECT/zones/$ZONE/instances/$NAME"
# above should be equivalent to below:
#   /usr/bin/gcloud compute instances delete "$NAME" \
#     --zone "$ZONE" \
#     --project "$PROJECT" \
#     --quiet
EOF

chmod +x "$WATCHER"

# --- install systemd unit ---
cat <<EOF > "$SERVICE"
[Unit]
Description=Container completion watcher
After=containerd.service docker.service
Wants=containerd.service docker.service

[Service]
Type=simple
ExecStart=$WATCHER
Restart=no

[Install]
WantedBy=multi-user.target
EOF

# --- activate service ---
systemctl daemon-reload
systemctl enable --now container-watcher.service

echo "[startup-script] finished successfully"
exit 0
