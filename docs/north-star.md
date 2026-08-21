# North star: remote GPUs behave like local PyTorch devices

## Current milestone status

- M1 execution is functionally complete for the pinned PyTorch/CUDA profile: core smoke,
  pinned memory, 82/82 curated OpInfos, 17 complete 634-name differentials
  totaling 10,778 exact native/remote status comparisons,
  single-rank NCCL all-gather, DTensor/FSDP2/checkpoint, CUDA IPC, model suites, link
  loss, and cancellation pass. CUDA activity profiling remains an observability
  gap because CUPTI is outside the Driver API protocol.
- M3 enumeration and independent execution on both ordinals work. PyTorch can
  select local `cuda:0` or remote `cuda:1`. The opt-in cuBLAS/cuBLASLt RPC path
  now passes same-process local-first linear forward/backward for float32,
  float16, and bfloat16 with exact local/remote loss agreement, plus strided
  batched and complex GEMM, GEMV/dot/triangular solves, FFT/STFT, modern
  cuSOLVER linalg, SDPA, 16 Hugging Face families, nanochat, LitGPT, and
  TorchTitan. Additional dtypes/shapes and unobserved linked-library APIs remain
  expansion gates.
- M2 is accepted for the pinned profile. Ordinary unmodified two-process
  `torchrun` completes exact mixed local/remote DDP and a nine-operation NCCL
  matrix through the remote-native NCCL RPC data plane. A 16 MiB collective
  reaches approximately 8.2–9.1 Gbit/s versus about 9.7 Gbit/s natively.
  Repeated teardown, server loss, and a subsequent clean session pass.
  M4 physical multi-host execution remains open; only one remote GPU
  workstation is currently available.

## Product contract

An unmodified PyTorch application started on `ws-5090-2` can use:

1. its local RTX 5090 only;
2. the RTX 5090 on `ws-5090-1` only;
3. one local and one remote RTX 5090 in the same job; and
4. eventually, any mixture of leased local and remote GPUs.

The application keeps its Python process, CPU preprocessing, filesystem view,
and control flow on `ws-5090-2`. Selecting a remote GPU must require launcher
configuration, not source edits. Unsupported behavior must fail explicitly;
the launcher must never silently fall back to the local GPU.

“Any PyTorch workload” is treated as a continuously expanding compatibility
contract, not a one-time claim. A release may make the claim only for its exact
pinned CUDA, driver, PyTorch, GPU, and tested feature profile.

## Execution milestones

### M1 — one process, one remote GPU

LUPINE's current model is the starting point: every CUDA object in one client
process belongs to one remote session. Required gaps before calling this mode
generally usable are:

- CPU-addressable pinned-host allocations and async transfers;
- correct `torch.linalg.lu_factor` behavior and recovery after CUDA errors;
- complete runtime/Driver API routing for JIT-loaded kernels;
- clear diagnostics for unsupported calls; and
- lifecycle cleanup after normal exit, exception, signal, or broken link.

### M2 — one job, one local worker plus one remote worker

The first mixed-GPU target is standard `torchrun`/DDP with two client workers on
`ws-5090-2`. Rank 0 uses the local GPU. Rank 1 loads the remoting shim and owns
one GPU context on `ws-5090-1`. Existing PyTorch DDP/FSDP code remains
unchanged.

This milestone requires remoting NCCL initialization and collectives, routing
NCCL calls and GPU pointers to the correct server session, allowing NCCL's data
plane to connect over the direct Ethernet link, and propagating failures to all
ranks without hanging. It is the shortest path to useful local-plus-remote
training and should precede same-process multi-device virtualization.

### M3 — one process sees local `cuda:0` and remote `cuda:1`

The shim presents a unified virtual device table. Every allocation, stream,
event, module, graph, library handle, pointer, and context carries an owning
route. Calls dispatch locally or remotely from that ownership; cross-device
copies and peer queries preserve CUDA semantics. Device enumeration,
`set_device`, guards, autocast, allocator state, and error state must be
per-device.

This is substantially more invasive than M2 because LUPINE currently chooses a
route for the process rather than for each CUDA object. Implement it only after
the distributed two-worker path is correct.

### M4 — multiple remote GPUs and production lifecycle

Add multiple servers, deterministic virtual ordering, leases, topology and
health reporting, admission control, authentication, observability, bounded
resource use, failure injection, and version negotiation. No GPU is shared or
preempted in the initial implementation.

## Correctness acceptance

Every required gate is differential: run the same pinned software and seed
natively and remotely, compare values and exceptions, and retain raw artifacts.
Performance never compensates for a semantic failure.

| Gate | Release requirement |
|---|---|
| Core smoke | All allocation, copy, math, autograd, cuDNN, streams/events, compile, and graph cases pass. |
| PyTorch OpInfo | All selected shards pass; expand toward the complete CUDA-supported database. No crash or poisoned session is accepted. |
| Memory | Pageable, pinned, registered, non-blocking, DataLoader, allocator, OOM, and cleanup cases pass. |
| Libraries | cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, SDPA, and JIT module loading pass. |
| Compilation | Inductor, Triton, inline NVRTC, custom operators, CUDA extensions, and graph capture/replay pass. |
| Models | Short CNN, transformer, generative, compiled, and dynamic-shape train/eval workloads pass. |
| Distributed | Single-rank NCCL all-reduce/all-gather and DTensor/checkpoint pass; two-rank local-only control, remote-only control, and mixed local/remote DDP pass without hang. |
| Failure semantics | Invalid calls, OOM, client death, server death, link loss, and cancellation fail deterministically and release the lease. |

Each test process is capped below five minutes. Large suites are split into
independently reproducible shards. A new failure is minimized, added as a
regression, fixed, rerun in its feature category, and followed by all earlier
gates. Iteration continues until every gate required by the current milestone
passes.

## Performance acceptance

Performance is classified by unavoidable data movement rather than summarized
by one misleading number.

- Compute-resident training: target median steady-state slowdown at or below
  1.20× and p95 at or below 1.30× relative to the same GPU on `ws-5090-1`.
- Launch/RPC-sensitive workloads: initially accept at most 1.50×, then use
  traces to remove repeated synchronous queries and safely batch asynchronous
  calls.
- Bulk host/device transfer: target at least 80% of measured application-level
  link throughput; do not compare 10 GbE transfer time to local PCIe as if it
  were software overhead.
- Mixed local/remote DDP: report per-rank compute, collective time, scaling
  efficiency, straggler time, and bytes on the wire. Establish empirical
  targets after NCCL works because this workstation has no two-local-GPU
  control.
- Cold connection, first CUDA call, compilation, and steady training are
  reported separately. Dependency download and one-time compilation are setup
  metrics, not hidden inside per-step timing.

The existing optimized LUPINE result already approaches the compute target for
several resident workloads. Transfer-heavy jobs cannot be near-native on the
current 10 GbE hardware unless tensors remain resident or transfers overlap
usefully with compute.

## Open-source workload ladder

The ladder grows from small regressions to realistic training while keeping
each invocation bounded:

| Project/workload | Feature value | Current status |
|---|---|---|
| PyTorch OpInfo shards | Broad upstream operator and autograd inputs | 82/82 first-sample curated pass; 17 complete 634-name sample/dtype differentials (10,778 selections) have exact native/remote status agreement, through nine additional sample shapes and a second-sample alternate-dtype tier. |
| TorchBench DCGAN train | Convolutional generator/discriminator, two optimizers | Native and remote pass; compiled remote is 1.31× native GPU time. |
| TorchBench Hugging Face BERT train | Transformer, embedding, optimizer, backward | Native and LUPINE pass. |
| latest modded-nanogpt | FP8, custom ops, NVRTC, Triton, compile, FA3 | Current kernel imports pass remotely, including the RTX 5090 target. A bounded two-step launch reaches upstream's approximately seven-minute first-run compile/warmup phase without a remoting failure, then stops at the short-test cap. |
| Expandable-segments allocator | CUDA VMM reserve/map/access and teardown | Native `cuda:0` and remote `cuda:1` both pass six changing allocation sizes in mixed mode with disjoint route-owned virtual addresses. |
| nanochat short pretrain | Modern compact LLM, GQA/SDPA, sliding attention, value residual | Actual upstream model trains remotely; compiled model + optimizer is 1.20× native GPU time. |
| LitGPT Pythia-14m debug pretrain | RoPE, fused SDPA, GELU MLP, configurable attention | Actual upstream model trains remotely; compiled model + optimizer is 0.99× native GPU time. |
| TorchTitan Qwen3 debug model | Fused QKV, QK norm, packed causal FlexAttention, compile | Actual 32M-parameter upstream model trains remotely; compiled model + optimizer is 1.00× native GPU time. Distributed features remain an expansion gate. |
| Whole-step CUDA Graph profiles | Forward, backward, optimizer, allocation capture | 15/16 Hugging Face families plus nanochat, LitGPT, and TorchTitan pass remotely; DETR retains the host-control fallback. |
| Transformers causal-LM example | Mainstream trainer/data pipeline and dynamic batches | Candidate expansion workload. |
| Diffusers tiny text-to-image/LoRA | Diffusion UNet, mixed precision, attention, checkpointing | Candidate expansion workload. |
| torchtune single-device LoRA | Fine-tuning, activation checkpointing, adapters | Candidate expansion workload. |

All public repositories remain local clones with disabled push URLs. No public
fork, issue, pull request, or upstream push is part of this plan.

## Immediate implementation order

1. Package the passing remote-native NCCL replacement with strict client/server
   NCCL-version validation and make `rgpu run --include-local` the reproducible
   standard-`torchrun` path.
2. Expand remaining NCCL ABI calls, fail closed for unsupported remote
   communicators, and add peer-rank abort during an active collective.
3. Expand the passing same-process cuBLAS/cuBLASLt/cuSOLVER/cuFFT RPC slice to
   additional samples, dtypes, shapes, and remaining linked APIs; retain local
   native forwarding and explicit failure for unsupported remote handles.
4. Validate two distinct physical remote hosts when one becomes available.
5. Deploy and validate a TLS-terminating proxy with a client-access policy.
   `rgpu run` and `attach` now accept ordered `--endpoint https://...` mappings;
   both CUDA and NVML verify the proxy certificate and hostname, while
   protocol-version negotiation fails closed before CUDA dispatch.
6. Design server-side CUPTI activity collection and expand peer access,
   multi-rank FSDP/DTensor, custom extensions, and multiple samples/dtypes.
7. Extend TorchTitan to DTensor/FSDP and distributed checkpoint paths after the
   mixed-NCCL gate, retaining its passing single-GPU Qwen3 regression.
8. Generalize automatic graph-region discovery for host-controlled programs
   such as DETR. Whole-step capture now brings static resident workloads near
   native; required synchronous results must never become stale.
