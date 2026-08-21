# Deployment architecture

The repository contains three product components and one development area.

```text
requesting machine                         GPU-providing machine

rgpu-client/                              rgpu-host/
  rgpu CLI                                  server OCI image
  CUDA/NVML shim          CUDA RPC          LUPINE server
  library interposers  ---------------->    NVIDIA userspace + driver
         \                    /
          \---- lupine/ -----/
             shared protocol
```

`lupine/` owns CUDA Driver API/NVML transport, object virtualization, and the
server implementation. `rgpu-client/` owns installation, orchestration,
attachment, image injection, and opaque-library interposition. `rgpu-host/`
packages the server for an NVIDIA Container Toolkit host. `dev/` is excluded
from the intended release footprint.

The current source installer builds both sides and uses SSH to transfer the
host image. A packaged release can publish the host image and client shim,
reducing client installation to the Python package plus its validated image
artifacts; the development directory is not required at runtime.
