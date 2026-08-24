ARG SHIM_IMAGE=remote-gpu-lupine-client:nccl-rpc-v6
ARG RUNTIME_IMAGE=remote-gpu-lupine-pytorch:nccl-rpc-v6

FROM ${RUNTIME_IMAGE} AS runtime
FROM ${SHIM_IMAGE}

COPY --from=runtime /opt/rgpu/lib /opt/rgpu/lib
COPY --from=runtime \
  /usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
  /opt/rgpu/lib/libnccl.so.2
COPY rgpu-client/remote_gpu/mixed_python/sitecustomize.py /opt/rgpu/python/sitecustomize.py

LABEL org.opencontainers.image.title="rgpu-client-artifact"
LABEL org.opencontainers.image.description="Self-contained rgpu CUDA/NVML shim and opaque-library interposers"
