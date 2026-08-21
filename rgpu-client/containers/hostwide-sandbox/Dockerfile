# Disposable root filesystem for validating host-wide loader integration.
# Nothing in this image is installed on the workstation host.
ARG SHIM_IMAGE=remote-gpu-lupine-client:portable
ARG WORKLOAD_IMAGE=remote-gpu-pytorch-native:2.12.0-cu130

FROM ${SHIM_IMAGE} AS shim
FROM ${WORKLOAD_IMAGE}

ARG RGPU_ENDPOINT=10.77.77.1:14833

USER root

COPY --from=shim /opt/lupine/lib/libcuda.so.1 /usr/local/lib/rgpu/libcuda.so.1
COPY --from=shim /opt/lupine/lib/libnvidia-ml.so.1 /usr/local/lib/rgpu/libnvidia-ml.so.1
COPY --from=shim /opt/lupine/lib/liblupine-cudart-compat.so /usr/local/lib/rgpu/liblupine-cudart-compat.so
COPY --from=shim /opt/lupine/lib/libcudart.so.13 /usr/local/lib/rgpu/libcudart.so.13
COPY --from=shim /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi

RUN set -eux; \
    ln -s libcuda.so.1 /usr/local/lib/rgpu/libcuda.so; \
    ln -s libnvidia-ml.so.1 /usr/local/lib/rgpu/libnvidia-ml.so; \
    mkdir -p /etc/rgpu; \
    printf '%s\n' "${RGPU_ENDPOINT}" > /etc/rgpu/endpoints; \
    printf '%s\n' '/usr/local/lib/rgpu' > /etc/ld.so.conf.d/00-rgpu.conf; \
    ldconfig; \
    test "$(ldconfig -p | awk '/libcuda\.so\.1/{print $NF; exit}')" = \
      /usr/local/lib/rgpu/libcuda.so.1; \
    test "$(ldconfig -p | awk '/libnvidia-ml\.so\.1/{print $NF; exit}')" = \
      /usr/local/lib/rgpu/libnvidia-ml.so.1

# Deliberately no LUPINE_SERVER, LD_LIBRARY_PATH, LD_PRELOAD, or wrapper
# entrypoint. Tests invoke nvidia-smi and Python as ordinary processes.
ENTRYPOINT []
CMD ["bash"]
