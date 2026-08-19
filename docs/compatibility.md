# Compatibility findings

Test date: 2026-08-19

The current conclusion is that transparent remote CUDA is already partly
solved, but transparent current PyTorch with low overhead is not solved by the
tested open-source projects on this hardware.

| Candidate | Own CUDA demo | PyTorch 2.12 + CUDA 13 | nanoGPT short run | Finding |
|---|---:|---:|---:|---|
| LUPINE main `af5af6e9` | Not separately required | Fail | Blocked by the same failure | PyTorch tensor creation and matmul crash the server in the new library-snapshot path. |
| LUPINE v1.0.0 `ebf4c278` | Pass | Pass with limits | Pass after removing redundant one-rank NCCL calls | Only viable modern-PyTorch candidate tested. Eager, cuDNN, autograd, streams/events, compile, CUDA Graphs, nanoGPT, and torchvision ResNet-50 pass. DDP/NCCL initialization and all-reduce crash. |
| Cricket `5d55bd58` | Pass remotely | Fail | Not eligible | Its CUDA 12.1 matmul passes on the remote RTX 5090. An unmodified current wheel crashes during initialization; upstream documents a patched PyTorch 1.13.1 source build. |
| GVirtuS `18f16dc3` | Pass remotely on CUDA 11.8 | CUDA unavailable | Not eligible | CUDA 12 build fails on removed texture-reference APIs. CUDA 11.8 works after dependency and shared-cudart fixes, but does not interpose the CUDA 13 Driver API used by current PyTorch. |

## LUPINE v1 performance

All native controls below ran directly on `ws-5090-1`, the same physical GPU
used by LUPINE. This removes GPU-to-GPU variation and measures the remoting
overhead itself. The true `ws-5090-2` local control remains pending because an
unrelated process occupied about 25.7 GiB of its GPU throughout testing.

| Workload | Native | LUPINE v1 | Relative result |
|---|---:|---:|---:|
| 124M ModernArch nanoGPT, 20 steps, complete process | 54.10 s | 61.40 s | 1.14× elapsed |
| Same nanoGPT, steady step average | 31.07 ms | 399.07 ms | 12.84× slower |
| torchvision ResNet-50, three eager train steps | 0.496 s total | 1.095 s total | 2.21× slower |
| Same ResNet, mean of steps two and three | 9.12 ms | 26.00 ms | 2.85× slower |

The nanoGPT process result hides the steady-state penalty because cold
`torch.compile` work dominates this deliberately short run. For longer
training, the per-step ratio is the relevant measure.

## Correctness and unsupported surface

The common smoke suite passed eight of eight checks both natively and through
LUPINE v1: identity, host/device copy, matrix multiplication, autograd, cuDNN
convolution, streams/events, `torch.compile`, and CUDA Graph capture/replay.
The torchvision workload produced the same three losses and peak allocation in
both modes.

LUPINE v1 is not evidence of all-PyTorch support. The observed NCCL failures
exclude DDP even with a single rank, and untested areas include multi-GPU
collectives, custom extensions with unusual Driver API use, profilers,
distributed checkpointing, IPC, peer access, and failure recovery.

## Decision implication

Do not build a remoting stack from scratch yet. Use LUPINE v1 as the reference
implementation and test target, report the main-branch regression upstream
only if separately authorized, and investigate why nanoGPT makes remoting much
more latency-sensitive than eager ResNet. Adoption would require either NCCL
support or an explicit single-GPU-only product boundary, plus performance work
around launch batching, CUDA Graph replay, and local simulation of synchronous
state queries.
