#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config_file=${1:-$project_root/dev/config/hosts.json}
raw_dir="$project_root/dev/results/raw"
client_image=${LUPINE_CLIENT_IMAGE:-remote-gpu-lupine-client:2.12.0-cu130}
server_image=${LUPINE_SERVER_IMAGE:-ghcr.io/lupinemachines/lupine-server:cuda-13.1.0-ubuntu24.04}
server_name="remote-gpu-lupine-smoke"

if [[ ! -f "$config_file" ]]; then
  echo "missing config: $config_file" >&2
  exit 1
fi
python3 "$project_root/dev/tools/upstreams.py" audit lupine

readarray -t remote_values < <(python3 - "$config_file" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
print(config["remote"]["ssh_user"])
print(config["remote"]["address"])
PY
)
remote_user=${remote_values[0]}
remote_address=${remote_values[1]}
ssh_target="$remote_user@$remote_address"
ssh_options=(-F /dev/null -o BatchMode=yes -o ConnectTimeout=5)

mkdir -p "$raw_dir"

docker build \
  --target lupine-client \
  --tag "$client_image" \
  --file "$project_root/dev/containers/common-pytorch/Dockerfile" \
  "$project_root"

active_compute=$(ssh "${ssh_options[@]}" "$ssh_target" \
  "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" || true)
if [[ -n "$active_compute" ]]; then
  echo "Remote GPU already has compute processes; refusing to interfere:" >&2
  echo "$active_compute" >&2
  exit 2
fi

if ssh "${ssh_options[@]}" "$ssh_target" \
  "docker container inspect $server_name >/dev/null 2>&1"; then
  echo "Refusing to reuse or replace existing container: $server_name" >&2
  exit 2
fi

cleanup() {
  ssh "${ssh_options[@]}" "$ssh_target" \
    "docker stop --time 10 $server_name >/dev/null 2>&1" || true
}
trap cleanup EXIT INT TERM

ssh "${ssh_options[@]}" "$ssh_target" "docker pull $server_image"
ssh "${ssh_options[@]}" "$ssh_target" \
  "docker run --detach --rm --name $server_name --gpus all --network host $server_image"

ready=0
for _ in $(seq 1 20); do
  if python3 - "$remote_address" <<'PY'
import socket
import sys
with socket.create_connection((sys.argv[1], 14833), timeout=0.25):
    pass
PY
  then
    ready=1
    break
  fi
  sleep 0.25
done
if [[ "$ready" -ne 1 ]]; then
  ssh "${ssh_options[@]}" "$ssh_target" "docker logs $server_name" >&2 || true
  echo "LUPINE server did not become ready" >&2
  exit 1
fi

docker image inspect "$client_image" >"$raw_dir/lupine-client-image.json"
ssh "${ssh_options[@]}" "$ssh_target" \
  "docker image inspect $server_image" >"$raw_dir/lupine-server-image.json"

python3 "$project_root/dev/tools/run_timed.py" \
  --label lupine-remote-smoke \
  --mode remote-cuda \
  --candidate lupine \
  --timeout 120 \
  --output "$raw_dir/lupine-remote-smoke.runner.json" \
  -- \
  docker run --rm --network host \
    --env "LUPINE_SERVER=$remote_address:14833" \
    --volume "$project_root:/workspace:ro" \
    --volume "$raw_dir:/results" \
    "$client_image" \
    python3 /workspace/dev/tests/smoke/pytorch_smoke.py \
      --compile \
      --graphs \
      --output /dev/results/lupine-remote-smoke.smoke.json

ssh "${ssh_options[@]}" "$ssh_target" \
  "nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader" \
  >"$raw_dir/lupine-post-smoke-gpu.txt"

echo "LUPINE smoke test passed; results written beneath $raw_dir"
