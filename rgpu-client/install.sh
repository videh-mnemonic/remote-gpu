#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
hosts=()
skip_build=0
release_wheel=""

usage() {
  echo "usage: $0 --host [user@]address[:port] [--host ...] [--skip-build] [--wheel FILE]" >&2
}

while (($#)); do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      hosts+=("$2")
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --wheel)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      release_wheel=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

((${#hosts[@]})) || { usage; exit 2; }

for command in docker ssh python3; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

if ((skip_build == 0)); then
  bash "$project_root/dev/tools/build_images.sh"
fi

install_source="$project_root/rgpu-client"
if [[ -n "$release_wheel" ]]; then
  [[ -f "$release_wheel" ]] || { echo "wheel not found: $release_wheel" >&2; exit 1; }
  install_source=$(realpath "$release_wheel")
fi

if command -v pipx >/dev/null; then
  # Keep the application isolated from distro-managed Python (PEP 668) while
  # exposing only the rgpu entry point in the user's normal local bin path.
  pipx install --force "$install_source"
  rgpu_bin=$(command -v rgpu || true)
  if [[ -z "$rgpu_bin" ]]; then
    bin_root=${PIPX_BIN_DIR:-${XDG_BIN_HOME:-"$HOME/.local/bin"}}
    rgpu_bin="$bin_root/rgpu"
  fi
else
  data_root=${XDG_DATA_HOME:-"$HOME/.local/share"}
  bin_root=${XDG_BIN_HOME:-"$HOME/.local/bin"}
  app_venv="$data_root/remote-gpu/venv"
  python3 -m venv "$app_venv"
  "$app_venv/bin/python" -m pip install --upgrade --force-reinstall \
    "$install_source"
  install -d "$bin_root"
  ln -sfn "$app_venv/bin/rgpu" "$bin_root/rgpu"
  rgpu_bin="$app_venv/bin/rgpu"
fi

[[ -x "$rgpu_bin" ]] || {
  echo "rgpu was installed but its executable was not found: $rgpu_bin" >&2
  exit 1
}

for host in "${hosts[@]}"; do
  "$rgpu_bin" deploy --host "$host"
done

echo "rgpu is installed and the validated server image is deployed."
echo "Example: rgpu run --host ${hosts[0]} -- nvidia-smi -L"
echo "Existing image: rgpu run --host ${hosts[0]} --image YOUR_PYTORCH_IMAGE -- python3 train.py"
