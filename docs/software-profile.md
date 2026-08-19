# Provisional software profile

The initial common profile is Python 3.12, PyTorch 2.12.0, and the official
CUDA 13.0 wheel index. It is provisional until the same native container passes
on both RTX 5090 workstations.

Reasons:

- Official PyTorch CUDA 13.0 wheels exist for Linux x86-64 and Python 3.12.
- CUDA 13 supports the RTX 5090's Blackwell compute capability.
- LUPINE publishes a CUDA 13.1 client/server image, close enough to evaluate
  driver-API compatibility with a CUDA 13.0 PyTorch application.
- The observed local NVIDIA 610.43.02 driver is newer than the driver shown in
  LUPINE's CUDA 13.1 example, but this must be tested rather than assumed.

The current modded-nanogpt main branch is optimized for multiple H100 GPUs and
uses kernels that may be Hopper-specific. Native bring-up will determine the
smallest Blackwell compatibility patch. That exact patch, if required, applies
to all local, remote-CUDA, and whole-process control runs; it must not contain a
backend-specific performance optimization.

Sources:

- [Official PyTorch CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/)
- [LUPINE CUDA 13.1 quick start](https://github.com/lupinemachines/lupine)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)

