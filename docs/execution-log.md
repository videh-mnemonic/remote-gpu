# Execution log

## 2026-08-19 — bake-off results

- Verified both RTX 5090s, driver 610.43.02, Docker GPU access, 0.17 ms ping,
  MTU 9000, and 9.90 Gb/s iperf3 in both directions with zero retransmits.
- Preserved exact upstream commits and disabled public pushes in every clone
  and initialized submodule.
- Confirmed native PyTorch 2.12.0+cu130 passes all eight smoke checks on
  `ws-5090-1`.
- Found a regression in current LUPINE main: even basic PyTorch tensor and
  matmul probes terminate the server child in the library-snapshot path.
- Built LUPINE v1.0.0 and passed all eight smoke checks, the bounded 124M
  nanoGPT workload, and torchvision ResNet-50. One-rank DDP construction and
  NCCL all-reduce crash and were skipped as semantic no-ops for the single-GPU
  nanoGPT measurement.
- Measured nanoGPT at 31.07 ms/step native and 399.07 ms/step through LUPINE
  v1. End-to-end cold times were 54.10 s and 61.40 s.
- Built Cricket against CUDA 12.1. Its remote CUDA matmul passes; unmodified
  PyTorch 2.12 crashes during initialization. Upstream's documented PyTorch
  route requires a patched source build of PyTorch 1.13.1.
- Built GVirtuS against CUDA 11.8 after replacing a dead dependency mirror
  with system packages. Its remote matmul passes. It does not build against
  CUDA 12 without porting removed texture APIs, and current CUDA 13 PyTorch
  does not see a device through it.
- Left both remote test servers stopped after each candidate run.

The local nanoGPT control remains pending. `ws-5090-2` continuously had an
unrelated `ninfer-serve` process using about 25.7 GiB, so no benchmark was
started on that GPU.

## 2026-08-19 — harness bring-up

Completed without GPU load:

- Confirmed ws-5090-2 has an RTX 5090 PCIe device, NVIDIA kernel modules, an
  active 10 GbE `eno2` link, and MTU 9,000 through read-only host interfaces.
- Added ignored local-clone storage and a manifest for LUPINE, Cricket,
  GVirtuS, modded-nanogpt, PyTorch, and TorchBench.
- Added a clone utility that disables the public origin push URL, installs a
  rejecting pre-push hook, and records exact provenance.
- Added a read-only local/remote doctor, a structured command runner with a
  timeout strictly below 5 min, and a short PyTorch CUDA smoke suite.
- Added unit tests for the no-fork/no-push manifest and timing policy.

Initial environment limitation (subsequently resolved by relaunching with
unrestricted host access):

- The managed Codex sandbox is launched with an isolated network namespace.
  Socket creation is denied, so SSH, ping, GitHub cloning, and direct-link
  benchmarks cannot run.
- `/dev/nvidia*` is not exposed inside the sandbox, so `nvidia-smi` and CUDA
  workloads cannot run.
- The Docker socket is visible but access is denied from the sandbox.

These were execution-environment limitations, not workstation failures. The
same checks later passed from the unrestricted session as recorded above.
