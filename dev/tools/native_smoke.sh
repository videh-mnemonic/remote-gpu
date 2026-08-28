#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
image_name=${REMOTE_GPU_IMAGE:-remote-gpu-pytorch:2.12.0-cu130}
result_name=${1:-native-smoke-local}
raw_dir="$project_root/dev/results/raw"

mkdir -p "$raw_dir"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is required" >&2
  exit 1
fi
if ! command -v docker >/dev/null; then
  echo "docker is required" >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null; then
  echo "GPU is not accessible; refusing to start a benchmark" >&2
  exit 1
fi

active_compute=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
if [[ -n "$active_compute" ]]; then
  echo "GPU already has compute processes; refusing to interfere:" >&2
  echo "$active_compute" >&2
  exit 2
fi

docker build \
  --target common \
  --tag "$image_name" \
  --file "$project_root/dev/containers/common-pytorch/Dockerfile" \
  "$project_root"

python3 "$project_root/dev/tools/run_timed.py" \
  --label "$result_name" \
  --mode native \
  --candidate none \
  --timeout 120 \
  --output "$raw_dir/$result_name.runner.json" \
  -- \
  docker run --rm --gpus all \
    --volume "$project_root:/workspace:ro" \
    --volume "$raw_dir:/results" \
    "$image_name" \
    python /workspace/dev/tests/smoke/pytorch_smoke.py \
      --compile \
      --graphs \
      --output "/dev/results/$result_name.smoke.json"
