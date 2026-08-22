ARG SERVER_IMAGE=remote-gpu-lupine-server:nccl-rpc-v6
ARG WORKLOAD_IMAGE=remote-gpu-pytorch-native:2.12.0-cu130

FROM ${SERVER_IMAGE} AS server
FROM ${WORKLOAD_IMAGE}

COPY --from=server /opt/lupine/bin/lupine_driver_server \
     /opt/lupine/bin/lupine_driver_server

ENV LUPINE_PORT=14836
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]
