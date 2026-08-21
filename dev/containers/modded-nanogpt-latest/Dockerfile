ARG BASE_IMAGE=remote-gpu-lupine-pytorch:optimized
FROM ${BASE_IMAGE}

# Keep the pinned PyTorch/CUDA profile supplied by the base image. The upstream
# requirements file pins a different torch build, so install only application
# dependencies here.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      datasets \
      huggingface-hub \
      kernels \
      numpy \
      setuptools \
      tiktoken \
      tqdm \
      typing-extensions==4.15.0

ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13

ENTRYPOINT []
CMD ["bash"]
