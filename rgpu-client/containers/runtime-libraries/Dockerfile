ARG BASE_IMAGE=remote-gpu-pytorch-native:2.12.0-cu130
FROM ${BASE_IMAGE}
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/*
COPY rgpu-client/remote_gpu/nccl_forward_interposer.c /tmp/nccl_forward_interposer.c
RUN mkdir -p /opt/rgpu/lib \
    && cp -L /usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
       /opt/rgpu/lib/libnccl_real.so.2 \
    && patchelf --set-soname libnccl_real.so.2 /opt/rgpu/lib/libnccl_real.so.2 \
    && gcc -shared -fPIC -O2 /tmp/nccl_forward_interposer.c \
       -L/opt/rgpu/lib -Wl,-rpath,/opt/rgpu/lib \
       -Wl,-soname,libnccl.so.2 -Wl,--no-as-needed -l:libnccl_real.so.2 \
       -o /tmp/libnccl.so.2 \
    && cp /tmp/libnccl.so.2 \
       /usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    && rm /tmp/nccl_forward_interposer.c

COPY rgpu-client/remote_gpu/cublas_rpc_interposer.c /tmp/cublas_rpc_interposer.c
COPY rgpu-client/remote_gpu/cublas_rpc.map /tmp/cublas_rpc.map
RUN gcc -shared -fPIC -O2 /tmp/cublas_rpc_interposer.c -ldl -lpthread \
       -Wl,--version-script=/tmp/cublas_rpc.map \
       -o /opt/rgpu/lib/libcublas_rpc.so \
    && rm /tmp/cublas_rpc_interposer.c /tmp/cublas_rpc.map

COPY rgpu-client/remote_gpu/cusolver_rpc_interposer.c /tmp/cusolver_rpc_interposer.c
RUN gcc -shared -fPIC -O2 /tmp/cusolver_rpc_interposer.c -ldl -lpthread \
       -o /opt/rgpu/lib/libcusolver_rpc.so \
    && rm /tmp/cusolver_rpc_interposer.c

COPY rgpu-client/remote_gpu/cufft_rpc_interposer.c /tmp/cufft_rpc_interposer.c
RUN gcc -shared -fPIC -O2 /tmp/cufft_rpc_interposer.c -ldl -lpthread \
       -o /opt/rgpu/lib/libcufft_rpc.so \
    && rm /tmp/cufft_rpc_interposer.c

COPY rgpu-client/remote_gpu/unsupported_rpc_guard.c /tmp/unsupported_rpc_guard.c
RUN gcc -shared -fPIC -O2 /tmp/unsupported_rpc_guard.c \
       -o /opt/rgpu/lib/libunsupported_rpc_guard.so \
    && rm /tmp/unsupported_rpc_guard.c
