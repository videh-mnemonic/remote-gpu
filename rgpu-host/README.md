# rgpu host

This is the deployable server side for a machine that provides an NVIDIA GPU.
Its versioned product artifact is the `remote-gpu-host:0.2.1`
OCI image, containing the LUPINE server built from `../lupine/` plus the CUDA
userspace libraries needed by opaque-library RPC.

The host requires an NVIDIA driver, Docker, NVIDIA Container Toolkit, SSH
access for deployment and lifecycle control, and network reachability from the
client. It does not require the Python `rgpu` package. `rgpu deploy` transfers
and loads this image; workload sessions start isolated server containers only
after admission control succeeds.

After loading the image, validate the host without starting a GPU workload:

```bash
./check.sh --image remote-gpu-host:0.2.1
```
