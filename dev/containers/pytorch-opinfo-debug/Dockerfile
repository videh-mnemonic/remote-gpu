ARG BASE_IMAGE=remote-gpu-pytorch-opinfo:nccl-rpc-v6
FROM ${BASE_IMAGE}
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gdb \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT []
CMD ["bash"]
