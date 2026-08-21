# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG LUPINE_IMAGE=ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04

FROM ${PYTHON_IMAGE} AS common

ARG TORCH_VERSION=2.12.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCHVISION_VERSION=0.27.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      --index-url ${TORCH_INDEX} \
      torch==${TORCH_VERSION}

RUN python -m pip install --no-cache-dir numpy pillow

RUN python -m pip install --no-cache-dir --no-deps \
      --index-url ${TORCH_INDEX} \
      torchvision==${TORCHVISION_VERSION}

WORKDIR /workspace

FROM ${LUPINE_IMAGE} AS lupine-source

FROM lupine-source AS lupine-client

ARG TORCH_VERSION=2.12.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCHVISION_VERSION=0.27.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/opt/lupine/lib

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
      --index-url ${TORCH_INDEX} \
      torch==${TORCH_VERSION}

RUN python3 -m pip install --break-system-packages --no-cache-dir numpy pillow

RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      --index-url ${TORCH_INDEX} \
      torchvision==${TORCHVISION_VERSION}

# Triton builds a small Python extension on first use. Keep this after the
# large PyTorch layer so correcting build prerequisites does not redownload
# the CUDA wheel stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
