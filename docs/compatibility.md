# Compatibility findings

Last updated: 2026-08-21

Transparent CUDA-over-network is partially solved in open source, but none of
the tested upstream projects provided current PyTorch, broad API coverage,
mixed local/remote enumeration, and low launch overhead without substantial
work. LUPINE v1.0.0 remains the viable protocol/code-generation base; this
repository carries a private compatibility and performance patch series.

## Open-source screening

| Candidate | Own CUDA demo | PyTorch 2.12 / CUDA 13 | Finding |
|---|---:|---:|---|
| LUPINE main `af5af6e9` | Not required | Fail | Current-main library snapshot path crashes tensor creation/matmul. |
| LUPINE v1.0.0 `ebf4c278` | Pass | Pass with upstream gaps | Selected base; upstream lacks the compatibility, routing, and performance work below. |
| Cricket `5d55bd58` | Pass | Fail | Current wheel crashes; upstream targets a patched PyTorch 1.13.1 build. |
| GVirtuS `18f16dc3` | Pass on CUDA 11.8 | CUDA unavailable | Does not interpose the CUDA 13 Driver API used by current PyTorch. |

No public fork or push was created. Public sources are local, push-disabled
clones under `dev/external/`.

## What the private runtime adds

- Per-object local/remote routing for devices, contexts, allocations, streams,
  events, modules, functions, libraries, graphs, graph executions, graph nodes,
  and memory pools.
- Local-first plus remote NVML fan-out, virtual PCI domains, reverse lookup,
  and a container-provided `nvidia-smi`.
- CPU-addressable pinned memory and pageable/pinned bulk transfer pipelines.
- cuSOLVER allocation fallback, CUDA runtime compatibility symbols, expanded
  NVML surface, and the Driver API hooks required by PyTorch/NCCL.
- Predictive immutable metadata caches with lifetime-aware invalidation.
- Ordered asynchronous RPC coalescing for safe calls, immediate asynchronous
  graph dispatch, and an explicit synchronous graph-launch fallback.
- Correct CUDA Graph capture-mode virtualization and stable graph replay.
- Server/image lease checks, stale-image rejection, GPU-use refusal, signal
  cleanup, link-loss behavior, and local-driver identity checks.
- Portable userspace injection into existing Ubuntu 22.04-or-newer compatible
  PyTorch images; no workload image rebuild is required.
- Per-route cuBLAS/cuBLASLt, cuSOLVER, and cuFFT object virtualization for a
  process that uses local `cuda:0` and remote `cuda:1` concurrently. The
  expanded surface includes GEMM/GEMV/dot/triangular and batched solve paths,
  LU/QR/Cholesky/eigensolver paths, and FFT plan creation/execution.
- Driver-level tensor-map encoding for Triton/Inductor TMA kernels, plus
  complex GEMV and all scalar variants of symmetric-indefinite factorization
  exercised by the expanded upstream dtype matrix.
- Strict remote-only opaque-library routing no longer requires exposing the
  local GPU. It selects PyTorch's semantics-equivalent cuSOLVER backend by
  default because MAGMA embeds device addresses inside device-resident pointer
  arrays that cannot be generically relocated across CUDA processes; an
  explicit user backend choice is preserved.

The implementation never installs or replaces a host driver, kernel module,
NVML library, `nvidia-smi`, or device node. Interposition exists only inside
the launched client container.

## Current differential compatibility

The current default and portable-injection paths pass:

- 8/8 core PyTorch smoke checks, including compile and CUDA Graphs;
- 82/82 curated upstream OpInfo samples;
- 79/79 available second-sample curated OpInfos with the same three empty
  sample generators natively and remotely; the first alternate dtype passes
  82/82 on both sides, and the second alternate dtype passes 80/82 on both
  sides with the same two dtype-index exhaustion errors; the third alternate
  dtype passes all 79 available cases on both sides with the same three
  dtype-index exhaustion errors and no remote-specific mismatch;
- 17 complete 634-name upstream OpInfo differentials (10,778 selections):
  sample index 0 at dtype indexes 0 through 6, sample indexes 1 through 9 at
  dtype index 0, and sample index 1 at dtype index 1, with exact
  native/strict-remote status agreement and no remote-specific mismatch;
- pinned `readinto`, non-blocking transfer, and DataLoader semantics;
- single-rank NCCL initialization, all-reduce, all-gather, and destruction;
- world-size-one DTensor redistribution and distributed checkpoint save/load;
- world-size-one composable FSDP2 forward/backward/AdamW with exact native losses;
- CUDA IPC tensor sharing with a spawned PyTorch process;
- all 16 bounded Hugging Face/Transformers/Diffusers training families;
- a current-stack refresh of GPT-2, Mamba2, VideoMAE, and DETR, each completing
  strict-remote forward, backward, and optimizer steps;
- actual upstream nanochat GPT, LitGPT Pythia, and TorchTitan Qwen3 training,
  including GQA, fused SDPA, packed causal FlexAttention, fused QKV, and QK norm;
- TorchBench DCGAN and HF BERT, torchvision, nanoGPT, cuBLAS, cuDNN, cuFFT,
  cuSPARSE, cuSOLVER, SDPA, Triton/Inductor, NVRTC/JIT loading, and graph replay;
- PyTorch's CUDA virtual-memory allocator with
  `PYTORCH_ALLOC_CONF=expandable_segments:True` on local and remote ordinals in
  the same process, including clean allocator teardown;
- invalid-device and OOM recovery; and
- deliberate server loss and launcher cancellation without a leaked lease.

The six native-matched OpInfo exceptions are two empty sample generators,
two invalid resize-with-autograd cases, complex-eigenvector backward with an
ill-defined phase-sensitive loss, and a sparse CSR reduction without a CUDA
kernel in the installed PyTorch build. They are not counted as remote misses.

The distributed-checkpoint probe initially hung in NCCL all-gather. The
minimized failure found that generic `cuMemcpyAsync` had a client wrapper but
no server handler, and batched stream-memory operations serialized only their
first element. Both protocol defects are fixed and retained as regressions.

Every new HTTP/2 session now carries protocol revision 1 in both directions.
The client waits for a compatible response before starting its dispatch
thread; the server returns HTTP 426 and terminates a missing or mismatched
protocol before any CUDA RPC. The normal round-7 client/server pair passes the
full 8/8 compile-and-graph smoke suite. A retained round-6 portable client is
rejected against the round-7 server, and the launcher releases the server lease.

## PyTorch-image transparency

The launcher extracts `libcuda.so.1`, `libnvidia-ml.so.1`, the small CUDA
runtime compatibility library, and `nvidia-smi` from a content-addressed,
validated client artifact. It bind-mounts them into the requested workload
container and sets process-local loader variables. An otherwise untouched
`remote-gpu-pytorch-native:2.12.0-cu130` image passes strict enumeration and
the full smoke suite through this path.

The first injection attempt correctly exposed an ABI issue: a shim compiled on
Ubuntu 24.04 required glibc 2.38 and could not load in the Ubuntu 22.04
workload image. The production injection artifact is now built on Ubuntu 22.04
(glibc 2.35) and passes in both older and newer tested images.

## Distributed and composition status

- Single-rank NCCL all-reduce/all-gather, DTensor redistribution, FSDP2
  training, and distributed checkpoint save/load pass.
- One process can enumerate local `cuda:0` and remote `cuda:1` and execute an
  explicit workload on remote `cuda:1`.
- A real local-plus-remote two-rank probe now sees distinct virtual PCI
  identities, connects socket channels, and completes NCCL communicator
  initialization on both RTX 5090s.
- Packed `cuLaunchKernelEx` forwarding reaches an actual remote NCCL kernel.
  Relocating UVA host pointers embedded in NCCL control structures removes the
  original illegal access. An opt-in protocol-3 coherence prototype now passes
  constrained one-channel all-reduce/all-gather and a small matrix including
  broadcast across one local and one virtualized remote GPU.
- The prototype is not production-ready: its first tiny all-reduce takes about
  9 seconds and a 1 MiB matrix does not finish within 60 seconds. Correct queue
  publication requires mirroring multi-megabyte windows, making generic polling
  the wrong high-throughput data plane.
- A native two-host fallback passes a five-step Transformer DDP smoke and
  reaches 9.83 Gbit/s for a 64 MiB NCCL all-reduce over 10GbE. Remote-native
  collective execution is therefore the next implementation target.
- Multiple `--host` values are parsed, leased, ordered, and passed to the
  routing layer, but physical multi-host execution cannot be accepted with only
  one available remote workstation.

## Performance conclusion

Upstream LUPINE's nanoGPT steady step measured 399.07 ms versus 31.07 ms
native. Metadata caching, ordered asynchronous sends, and related fixes reduce
the current guardrail to 32.97 ms. More recent optimized profiles bring DETR
to 1.33× native GPU time, Mixtral batched experts to 1.00×, compiled TorchBench
DCGAN to 1.31×, and CUDA Graph replay to 1.11×.

Compiled-region execution is now validated on three additional real upstream
training stacks. Compiled model plus optimizer reaches 1.00× native GPU time
for TorchTitan Qwen3, 0.99× for LitGPT Pythia-14M, and 1.20× for nanochat's
sub-millisecond GPT step. Synchronized wall ratios are 1.06×, 1.09×, and 1.48×
respectively. This supports automatic region/graph execution as the primary
performance path while retaining eager CUDA API virtualization as fallback.

TorchTitan also exposed two native-matched upstream compatibility issues in
this dependency snapshot: its mask helper passes a newer optional
`create_block_mask` keyword, and PyTorch 2.12's default float32 FlexAttention
autotuner offers only tiles that exceed the RTX 5090 shared-memory limit. A
semantics-preserving mask adapter and explicit conservative kernel tiles are
applied identically to native and remote tests.

Remote whole-step graph capture now passes 15 of 16 Hugging Face/Diffusers
families and the actual nanochat, LitGPT, and TorchTitan sources. The server
must use relaxed capture mode because RPCs are handled by transport worker
threads; the client continues to enforce the application's requested capture
policy. Nanochat, LitGPT, TorchTitan, and Llama measure 1.10×, 1.02×, 1.03×,
and 1.04× native GPU time under whole-step replay. DETR remains the deliberate
fallback because its SciPy Hungarian matcher consumes GPU values on the CPU.

Bulk transfer remains hardware-bound: a 256 MiB H2D plus D2H round trip moves
512 MiB and achieves about 7.40 Gb/s on the measured 9.90 Gb/s link, but is far
slower than local PCIe. The product must favor GPU residency, bulk asynchronous
movement, and overlap.

Detailed matched measurements are in [timing-summary.md](timing-summary.md).

The conservative static audit currently interposes 200 of the 364 GPU-library
symbols referenced by the pinned PyTorch image. The remaining 164 are a
prioritized compatibility backlog; they are not observed failures, as the
10,778-selection runtime differential demonstrates. Coverage continues to
expand from minimized upstream failures instead of speculative ABI wrappers.

CUDA VMM required more than device-ordinal translation. A local and remote
allocator can otherwise reserve the same numeric virtual address in their
separate CUDA processes, making PyTorch mistake two allocations for one. The
client now gives each remote route a disjoint high-address reservation hint,
tracks the route owner, and translates allocation/access descriptors to the
server's physical ordinal. This changes no host driver or device mapping.

## Remaining blockers to the full north star

1. Continue the differential beyond the first two samples and first four
   supported dtype tiers, especially larger shapes and less common linked
   CUDA-library APIs. The reproducible pinned-image ABI audit currently finds
   184 referenced symbols covered out of 364 referenced by PyTorch, with 194
   total interposer exports; the 180-name remainder is a conservative static
   backlog, not 180 observed failures. Round 31's complex strided-batched GEMM
   closes the only remote-specific failure exposed by the fourth dtype tier.
   The complete second-sample sweep is now iterating from its minimized MAGMA
   batched-LU failure through PyTorch's equivalent routed cuSOLVER backend.
2. Validate two distinct remote hosts and deterministic ordering/routing.
3. Add authenticated transport, health reporting, and production admission
   control.
4. Expand failure injection to packet loss, server restart, and multi-rank
   cancellation.
5. Add remote CUPTI collection: native `torch.profiler` reports CUDA activity,
   while the strict remote client currently reports `CUPTI_ERROR_NOT_INITIALIZED`.
6. Continue upstream PyTorch coverage beyond first-sample OpInfo probes,
   including peer access, multi-rank DTensor/FSDP, and custom extensions.
7. Improve launch-heavy performance without changing observable synchronous
   semantics; retain the compatibility fallbacks.
