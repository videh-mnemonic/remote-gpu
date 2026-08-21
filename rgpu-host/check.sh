#!/usr/bin/env bash
set -euo pipefail

image=remote-gpu-host:0.1.0

usage() {
  echo "usage: $0 [--image IMAGE]" >&2
}

while (($#)); do
  case "$1" in
    --image)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      image=$2
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

for command in docker nvidia-smi; do
  command -v "$command" >/dev/null || {
    echo "rgpu host check: missing required command: $command" >&2
    exit 1
  }
done

nvidia-smi -L >/dev/null || {
  echo "rgpu host check: the NVIDIA driver is not healthy" >&2
  exit 1
}

docker info >/dev/null || {
  echo "rgpu host check: Docker is unavailable to this user" >&2
  exit 1
}

docker info --format '{{json .Runtimes}}' | grep -q 'nvidia' || {
  echo "rgpu host check: Docker's NVIDIA runtime is not installed" >&2
  exit 1
}

docker image inspect "$image" >/dev/null || {
  echo "rgpu host check: server image is not loaded: $image" >&2
  exit 1
}

echo "rgpu host check: ready ($image)"
