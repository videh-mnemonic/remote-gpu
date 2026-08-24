#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_tree=""
image="rgpu/ninfer-qwen38-pipeline:dev"

while (($#)); do
  case "$1" in
    --source)
      source_tree=${2:?missing value for --source}
      shift 2
      ;;
    --image)
      image=${2:?missing value for --image}
      shift 2
      ;;
    *)
      echo "usage: $0 --source PATH [--image NAME]" >&2
      exit 2
      ;;
  esac
done

[[ -n "$source_tree" && -f "$source_tree/Dockerfile" ]] || {
  echo "--source must name an NInfer checkout containing Dockerfile" >&2
  exit 2
}

build_tree=$(mktemp -d -t rgpu-ninfer-pipeline.XXXXXX)
cleanup() {
  rm -rf -- "$build_tree"
}
trap cleanup EXIT

# Work from a disposable copy: neither ali nor its nested NInfer checkout is
# modified, and any local Responses API compatibility changes are preserved.
rsync -a --exclude .git --exclude build -- "$source_tree/" "$build_tree/"
(
  cd "$build_tree"
  git apply "$project_root/dev/patches/ninfer-qwen38-1m-context.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-two-gpu-pipeline.patch"
)
docker build --tag "$image" "$build_tree"
echo "built $image from disposable source copy"
