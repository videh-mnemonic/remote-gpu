ARG BASE_IMAGE=remote-gpu-huggingface:optimized
FROM ${BASE_IMAGE}

RUN python3 -m pip install --break-system-packages --no-cache-dir \
      "jsonargparse[signatures]>=4.37,<=4.41" \
      "lightning>=2.6.1" \
      "psutil==7.2.2" \
      "spmd_types==0.2.3" \
      tyro

ENTRYPOINT []
CMD ["bash"]
