#!/bin/sh
set -eu

echo "# Host / Environment Information"
echo

section() {
  echo "## $1"
}

cmd() {
  echo "### $*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 || true
  else
    echo "(command not found)"
  fi
  echo
}

raw() {
  echo "### $1"
  cat "$1" 2>&1 || true
  echo 
}

# ------------------------------------------------------------

section "Timestamp"
date -u
echo

section "OS / Kernel"
cmd uname -a
cmd cat /etc/os-release

section "CPU"
cmd nproc
raw /proc/cpuinfo

section "Memory"
cmd free -h
raw /proc/meminfo

section "Disk"
cmd df -h
cmd mount

section "Cgroup"
raw /proc/self/cgroup
[ -f /sys/fs/cgroup/cpu.max ] && raw /sys/fs/cgroup/cpu.max
[ -f /sys/fs/cgroup/memory.max ] && raw /sys/fs/cgroup/memory.max

section "Container Detection"
if [ -f /.dockerenv ]; then
  echo "Running inside Docker"
fi
cmd cat /proc/1/cgroup

section "Environment Variables (filtered)"
cmd env | grep -Ev '(KEY|API|SECRET)'

section "Git"
cmd git rev-parse HEAD
cmd git status --porcelain

section "Rust"
cmd rustup run stable rustc --version --verbose
cmd rustup run stable cargo --version
cmd rustup run nightly rustc --version --verbose
cmd rustup run nightly cargo --version

section "Cloud Metadata"
if command -v curl >/dev/null 2>&1; then
  echo "### GCE metadata"
  curl -fs -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null \
    || echo "(not GCE)"
  echo
fi

section "Limits"
cmd ulimit -a

section "End"
