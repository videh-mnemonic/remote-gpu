ARG BASE_IMAGE=remote-gpu-lupine-pytorch:optimized
FROM ${BASE_IMAGE}

# Runtime dependencies for the exact locally mounted Transformers and Diffusers
# source commits. Model weights and datasets are intentionally not installed.
RUN python3 -m pip install --break-system-packages --no-cache-dir --upgrade \
      "huggingface-hub>=1.23.0,<2.0" \
      "tokenizers>=0.23.1,<0.24.0" \
      "safetensors>=0.8.0" \
      "regex>=2025.10.22" \
      "accelerate>=1.1.0" \
      "Pillow>=10.0.1" \
      requests \
      scipy

ENTRYPOINT []
CMD ["bash"]
