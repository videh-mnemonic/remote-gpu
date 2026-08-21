#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config_file=${1:-$project_root/dev/config/hosts.json}
raw_dir="$project_root/dev/results/raw"

if [[ ! -f "$config_file" ]]; then
  echo "missing config: $config_file" >&2
  exit 1
fi

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

run_direction() {
  local direction=$1
  local reverse_flag=()
  if [[ "$direction" == "reverse" ]]; then
    reverse_flag=(-R)
  fi
  ssh "${ssh_options[@]}" "$ssh_target" \
    "iperf3 --server --daemon --one-off --port 5201"
  python3 "$project_root/dev/tools/run_timed.py" \
    --label "iperf3-$direction" \
    --mode remote-process \
    --candidate none \
    --timeout 30 \
    --output "$raw_dir/iperf3-$direction.runner.json" \
    -- \
    iperf3 --client "$remote_address" --port 5201 --time 10 --parallel 1 --json "${reverse_flag[@]}" \
    >"$raw_dir/iperf3-$direction.summary.json"
}

ping -c 20 -W 1 "$remote_address" >"$raw_dir/ping.txt"
run_direction forward
run_direction reverse

echo "network results written beneath $raw_dir"
