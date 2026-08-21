ARG PYTORCH_IMAGE=remote-gpu-pytorch-native:2.12.0-cu130
ARG BASE_IMAGE=remote-gpu-lupine-pytorch:optimized
FROM ${PYTORCH_IMAGE} AS opinfo-deps

RUN python3 -m pip install --break-system-packages --no-cache-dir \
      --target /opt/rgpu-opinfo-deps \
      expecttest \
      hypothesis \
      psutil \
      pyyaml \
      sentencepiece \
      tabulate \
      transformers==4.57.3

FROM ${BASE_IMAGE}

COPY --from=opinfo-deps /opt/rgpu-opinfo-deps /opt/rgpu-opinfo-deps
ENV PYTHONPATH=/opt/rgpu-opinfo-deps:${PYTHONPATH}

ENTRYPOINT []
CMD ["bash"]
