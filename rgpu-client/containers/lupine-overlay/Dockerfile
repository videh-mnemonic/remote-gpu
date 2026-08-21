ARG LUPINE_SHIM_IMAGE=remote-gpu-lupine-client:v1-cache
ARG PYTORCH_IMAGE=remote-gpu-lupine-pytorch:v1.0.0

FROM ${LUPINE_SHIM_IMAGE} AS shim
FROM ${PYTORCH_IMAGE}

# The bundled legacy MAGMA batched-LU path launches correctly but dereferences
# an invalid pointer through remote CUDA.  PyTorch's supported cuSOLVER backend
# preserves linalg semantics and serves as the synchronous compatibility path.
# A caller can still override this environment variable explicitly.
ENV TORCH_LINALG_PREFER_CUSOLVER=1
ENV LD_PRELOAD=/opt/lupine/lib/liblupine-cudart-compat.so

COPY --from=shim /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=shim /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so.1
COPY --from=shim /opt/lupine/lib/liblupine-cudart-compat.so /opt/lupine/lib/liblupine-cudart-compat.so
