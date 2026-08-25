#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_tree=""
image="rgpu/ninfer-qwen38-pipeline:dev"
codex_only=0

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
    --codex-only)
      codex_only=1
      shift
      ;;
    *)
      echo "usage: $0 --source PATH [--image NAME] [--codex-only]" >&2
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
  if ((!codex_only)); then
    git apply "$project_root/dev/patches/ninfer-qwen38-1m-context.patch"
    git apply "$project_root/dev/patches/ninfer-qwen38-two-gpu-pipeline.patch"
    git apply "$project_root/dev/patches/ninfer-qwen38-mixed-device-graphs.patch"
    git apply "$project_root/dev/patches/ninfer-qwen38-fused-gdn-split.patch"
    git apply "$project_root/dev/patches/ninfer-qwen38-pipelined-prefill.patch"
  fi
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-tool-calls.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-direct-shell.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-command-forms.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-continuation.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-command-array.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-progress-sentences.patch"
  git apply "$project_root/dev/patches/ninfer-qwen38-codex-instruction-order.patch"
)
docker build --tag "$image" "$build_tree"
echo "built $image from disposable source copy"
