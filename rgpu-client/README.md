# rgpu client

This is the software installed on a machine that wants to consume remote
GPUs. It provides:

- the `rgpu` launcher, deploy, status, attach, detach, and garbage-collection
  commands;
- `rgpu-rescue` for local recovery from an interrupted host-wide transaction;
- CUDA-library interposers for cuBLAS/cuBLASLt, cuSOLVER, cuFFT, and NCCL; and
- container definitions used to package and inject the LUPINE CUDA/NVML shim.

Release `0.2.0` installs the Python commands from a wheel and injects the
self-contained `remote-gpu-client:0.2.0` artifact. The artifact includes the
LUPINE CUDA/NVML libraries plus cuBLAS, cuSOLVER, cuFFT, NCCL, and strict
unsupported-call guards; a workload image does not need `/opt/rgpu` baked in.

For the current source-based installation from the repository root:

```bash
./rgpu-client/install.sh --host user@gpu-host
```

`--skip-build` reuses already-built client and host images. The client deploys
the exact host image over SSH and verifies its image identity before use.
Pass `--wheel FILE` to install a wheel produced by
`dev/tools/build_release.py` instead of installing from the source tree.
