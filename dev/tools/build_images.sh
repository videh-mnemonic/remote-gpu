#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
lupine_source="$project_root/lupine"
release_version=$(PYTHONPATH="$project_root/rgpu-client" python3 -c \
  'from remote_gpu.version import __version__; print(__version__)')

[[ -f "$lupine_source/CMakeLists.txt" ]] || {
  echo "missing canonical LUPINE source: $lupine_source" >&2
  exit 1
}

docker build \
  --target client \
  --tag remote-gpu-lupine-client:optimized \
  --file "$lupine_source/Dockerfile" \
  "$lupine_source"

# Build the runtime-injected client against glibc 2.35 so it can load in both
# Ubuntu 22.04 and newer PyTorch/CUDA workload images. The server remains on
# the primary Ubuntu 24.04 build because it is isolated in its own container.
docker build \
  --build-arg UBUNTU_VERSION=22.04 \
  --target client \
  --tag remote-gpu-lupine-client:portable \
  --file "$lupine_source/Dockerfile" \
  "$lupine_source"

docker build \
  --target server \
  --tag remote-gpu-lupine-server:optimized \
  --file "$lupine_source/Dockerfile" \
  "$lupine_source"

docker build \
  --target common \
  --tag remote-gpu-pytorch-native:2.12.0-cu130 \
  --file "$project_root/dev/containers/common-pytorch/Dockerfile" \
  "$project_root"

docker build \
  --build-arg LUPINE_SHIM_IMAGE=remote-gpu-lupine-client:optimized \
  --build-arg PYTORCH_IMAGE=remote-gpu-pytorch-native:2.12.0-cu130 \
  --tag remote-gpu-lupine-pytorch:optimized \
  --file "$project_root/rgpu-client/containers/lupine-overlay/Dockerfile" \
  "$project_root"

# Build the exact protocol-6 images consumed by the rgpu CLI defaults.  The
# CUDA/NCCL workload is Debian-based, so both injected client and server
# binaries are built on Ubuntu 22.04 to keep their glibc/libstdc++ ABI portable.
docker tag \
  remote-gpu-lupine-client:portable \
  remote-gpu-lupine-client:nccl-rpc-v6

docker build \
  --build-arg SHIM_IMAGE=remote-gpu-lupine-client:nccl-rpc-v6 \
  --tag remote-gpu-hostwide-sandbox:nccl-rpc-v6 \
  --file "$project_root/rgpu-client/containers/hostwide-sandbox/Dockerfile" \
  "$project_root"

docker build \
  --build-arg BASE_IMAGE=remote-gpu-hostwide-sandbox:nccl-rpc-v6 \
  --tag remote-gpu-lupine-pytorch:nccl-rpc-v6 \
  --file "$project_root/rgpu-client/containers/runtime-libraries/Dockerfile" \
  "$project_root"

docker build \
  --build-arg UBUNTU_VERSION=22.04 \
  --target server \
  --tag remote-gpu-lupine-server:nccl-rpc-v6 \
  --file "$lupine_source/Dockerfile" \
  "$lupine_source"

docker build \
  --build-arg SERVER_IMAGE=remote-gpu-lupine-server:nccl-rpc-v6 \
  --tag remote-gpu-lupine-server:nccl-rpc-pytorch-v6 \
  --file "$project_root/rgpu-host/containers/server-overlay/Dockerfile" \
  "$project_root"

docker tag \
  remote-gpu-lupine-server:nccl-rpc-pytorch-v6 \
  "remote-gpu-host:$release_version"

docker tag \
  remote-gpu-lupine-pytorch:nccl-rpc-v6 \
  "remote-gpu-workload:$release_version"

docker build \
  --build-arg SHIM_IMAGE=remote-gpu-lupine-client:nccl-rpc-v6 \
  --build-arg RUNTIME_IMAGE=remote-gpu-lupine-pytorch:nccl-rpc-v6 \
  --tag "remote-gpu-client:$release_version" \
  --file "$project_root/rgpu-client/containers/release-artifact/Dockerfile" \
  "$project_root"
