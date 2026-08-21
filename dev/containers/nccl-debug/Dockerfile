ARG BASE_IMAGE=remote-gpu-lupine-pytorch:nccl-rpc-v5
FROM ${BASE_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gdb \
    && rm -rf /var/lib/apt/lists/*
