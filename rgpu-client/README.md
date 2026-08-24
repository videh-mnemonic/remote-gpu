# rgpu client

This is the software installed on a machine that wants to consume remote
GPUs. It provides:

- the `rgpu` launcher, deploy, status, attach, detach, and garbage-collection
  commands;
- `rgpu-rescue` for local recovery from an interrupted host-wide transaction;
- CUDA-library interposers for cuBLAS/cuBLASLt, cuSOLVER, cuFFT, and NCCL; and
- container definitions used to package and inject the LUPINE CUDA/NVML shim.

Release `0.2.1` installs the Python commands from a wheel and injects the
self-contained `remote-gpu-client:0.2.1` artifact. The artifact includes the
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

For host-wide use, attach transactionally and then launch PyTorch from a new
terminal/login shell so it inherits `/etc/profile.d/rgpu.sh`:

```bash
sudo "$(command -v rgpu)" attach --host user@gpu-host
nvidia-smi
python3 -c 'import torch; print(torch.cuda.device_count())'
sudo "$(command -v rgpu)" detach
```

The profile changes only loader paths and rgpu feature flags. It does not use
`LD_PRELOAD`, replace the NVIDIA driver, or modify kernel modules.
