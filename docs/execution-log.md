# Execution log

## 2026-08-23 — transparent mixed host-wide release gate

- Upgraded the live attachment to release 0.2.0/wire protocol 6 without
  changing the local NVIDIA driver, kernel modules, packages, device nodes, or
  `nvidia-smi`. Ordinary host `nvidia-smi -L` enumerates the local RTX 5090 as
  GPU 0 and the remote RTX 5090 as GPU 1. The independent physical-stack
  fingerprint and vendor-only health snapshot remain unchanged.
- The live strict-remote PyTorch smoke passes all eight checks, including
  `torch.compile` and CUDA Graph capture/replay. A deterministic NInfer
  64-token request produced the expected output digest at 224.02 tok/s with
  80.96 ms time to first token; model startup was 23.798 seconds.
- Found that pip PyTorch's wheel RUNPATH can select its bundled CUDA libraries
  ahead of the loader cache. Release 0.2.1 adds a profile-scoped Python
  bootstrap that loads rgpu's CUDA and math-library routes only on the first
  `torch` import. It avoids system-wide `LD_PRELOAD`; non-Torch Python remains
  clean.
- Fixed context-wide CUDA Graph synchronization so graph-owned device-to-host
  copies are published after `torch.cuda.synchronize(device)`, not only after
  explicit stream synchronization. Pending graph copies are consumed by the
  first matching synchronization, preventing a later unrelated synchronize
  from rewriting application-owned host memory. Dynamic pinned-host replay now
  returns the expected changing values across repeated launches.
- The complete 0.2.1 client/server path passes a local-loopback host-wide gate:
  two devices enumerate, local and remote kernels agree, mixed cuBLAS training
  is exact for float32/float16/bfloat16, remote `torch.compile` passes, and
  dynamic CUDA Graph replay returns 22, 38, and 92. The bounded gate completes
  in 5.21 seconds.
- Corrected a diagnostic false alarm: the apparent 245-descriptor server was
  remote PID 1 (`systemd`) observed through the host PID namespace. The actual
  idle Lupine parent has four descriptors and no connection children. No
  lifecycle patch was needed.
- The live workstation remains attached to 0.2.0 while 0.2.1 is finalized. A
  final live detach/reattach and remote acceptance run are still required
  before calling 0.2.1 validated on the physical two-host path.

## 2026-08-22 — correct, near-native NInfer CUDA Graph replay

- Fixed NInfer's graph startup failure without changing NInfer. The client now
  materializes one registered kernel per CUDA fatbin before the application's
  memory query, so lazy remote module allocation is included in its normal
  budget. Reported graph memory fell from 96 MiB (over the 82 MiB limit) to
  2 MiB, and model initialization completes in 23.492 seconds remotely.
- Minimized a CUDA Graph correctness defect to a standalone native/remote test:
  graph replay reused the host-to-device staging bytes captured on the first
  launch. Pinned host allocations now have stable server mirrors and client
  dirty-page tracking; dirty ranges are propagated before each graph launch.
  The test covers changing inputs, two captures on one stream, graph-exec
  update, and device-to-host results.
- Isolated graph-owned copy resources per capture and rebind them after a
  successful graph-exec update. This removed NInfer's subsequent paged-KV trim
  failure and prevents one stream's captures from sharing stale resource
  metadata.
- Five matched 257-token prompt/512-token generation requests measure 197.43
  tok/s remote versus 204.26 tok/s native, with 2.648 versus 2.555 seconds
  median wall time. Remote graph decode is 3.26 times the original broken
  60.56 tok/s result and is now 3.3% below native throughput. A separate
  deterministic 64-token run produced the exact same streamed-output SHA-256
  digest natively and remotely.
- Advanced the client/server contract to release 0.2.0 and wire protocol 6 for
  the graph host-source packet extension. Matching packaged client/host images
  pass the graph regression and an eight-check PyTorch smoke; a protocol-6
  client rejects the live protocol-1 daemon with HTTP 426 before CUDA dispatch.
  The live persistent attachment was intentionally not restarted.
- The complete GPU-free suite passes 90/90. Versioned client, host, and default
  workload images build successfully, and the normal local `ali` service was
  restored healthy after every bounded benchmark.

## 2026-08-21 — Qwen3.8 27B NInfer serving optimization

- Ran the unmodified private `ali` Qwen3.8 27B NVFP4 serving image natively on
  `ws-5090-2` and through strict rgpu execution on `ws-5090-1`. The benchmark
  client uses bounded OpenAI-compatible streaming requests and records startup,
  time to first token, wall time, prefill throughput, and decode throughput.
- The first remote run exposed one compatibility issue: NInfer's CUDA Graph
  memory budget observes 96 MiB of lazy-loaded remote modules versus its 82 MiB
  allowance. The supported `--no-cuda-graph` mode runs successfully. A
  server-side eager-module experiment passed the memory check but failed a
  cooperative extended launch, so it was rejected and no eager-loading option
  remains in the CLI.
- RPC profiling found that device-only `cuMemcpy2DAsync` incorrectly waited for
  a response and synchronized the remote stream. It is now fire-and-forget;
  copies involving host memory retain synchronized staging-buffer lifetime.
  Long-prompt prefill improved from approximately 7,244 to 8,295 tok/s, or
  from 86.6% to 99.2% of matched native throughput.
- Added a bounded, route-aware successful-result cache for pure tensor-map
  encoding. It reduced profiled remote encodes from 6,720 to 228 calls. Tests
  cover both optimizations, the complete GPU-free suite passes 88/88, and a
  release-image remote transpose/copy checksum passes on the RTX 5090.
- Deterministic five-run results with prefix reuse disabled are 176.07 versus
  202.65 tok/s for decode and 8,294.75 versus 8,358.53 tok/s for long-prompt
  prefill. The remaining decode gap is host-control bound: MTP requires a host
  decision every speculative round over a direct link whose measured average
  RTT was 0.34 ms. Two extended-launch pipeline variants and fixed coalescing
  policies did not materially improve it and were removed.
- Promoted the optimized client and host images and deployed the host image
  without restarting the existing persistent attachment or modifying either
  NVIDIA driver stack. The normal `ali` server was restored and its health
  endpoint passes.

## 2026-08-21 — production-oriented source layout

- Promoted the fully integrated LUPINE-derived runtime from a disposable
  external worktree to canonical top-level `lupine/` source. Preserved the
  41-round patch trail under `dev/patches/` and the push-disabled upstream
  checkout under ignored `dev/external/` for provenance.
- Isolated the installable consumer package under `rgpu-client/`, including the
  Python CLI, host-wide lifecycle code, CUDA-library interposers, and client
  image definitions. Isolated the provider-side server image under
  `rgpu-host/`. Moved tests, tools, research images, manifests, and evidence
  under `dev/`.
- Replaced patch-replay builds with direct builds from canonical `lupine/`.
  Independent client packaging/import and CLI-help checks pass; every client,
  shim, interposer, and host-server image rebuilds from the new paths.
- The complete GPU-free suite passes 82/82, including two new layout-contract
  regressions. A newly built strict-remote image
  enumerates the RTX 5090, computes the elementwise checksum 140, and returns
  64.0 from a cuBLAS-routed matrix multiplication. A separate `nvidia-smi -L`
  launch reports the expected remote GPU. The existing persistent host-wide
  attachment was left running and the host NVIDIA stack was not modified.

## 2026-08-21 — deeper OpInfo samples and batched-library repairs

- Extended the complete 634-name native/strict-remote differential through
  sample indexes 4 through 9 at dtype index 0 and sample index 1 at dtype index
  1. Across 17 complete tiers and 10,778 status comparisons, native and remote
  agree exactly with zero remote-specific difference.
- Sample index 4 exposed device-resident FP4/FP8 scaling pointers in
  `cublasLtMatmul`. Descriptor pointer mode is now retained and determines
  whether scalar parameters are serialized host values or remote device
  addresses. The minimized `_scaled_mm_v2` regression and complete tier pass.
- Sample index 8 exposed missing float32 batched Cholesky routing. Added
  float32, complex64, and complex128 `potrfBatched` cuSOLVER routes; the
  minimized batched forward/backward case and complete tier pass. The protocol
  packet suite now exercises 71 calls, and all 80 GPU-free tests pass.
- The linked-library audit rises from 200/364 to 203/364 PyTorch-referenced
  symbols interposed. The remaining 161 symbols are a conservative static
  expansion backlog, not observed failures.
- Of the newly added tiers, sample 8/dtype 0 has the largest aggregate ratio:
  11.148 seconds remote versus 7.683 seconds native, or 1.451x. It is recorded
  as the next performance-triage target; testing is paused for user review.

## 2026-08-20 — complete ten-tier OpInfo differential and modern model refresh

- Added remote Driver API routing for `cuTensorMapEncodeTiled`, unblocking
  Triton/Inductor tensor-memory-accelerator descriptors used by modern scaled
  matrix multiplication. Added complex/double symmetric-indefinite cuSOLVER
  factorization and complex64/complex128 cuBLAS GEMV routing after broader
  dtype tiers exposed those locally resolved library calls.
- Completed ten full 634-name upstream OpInfo differentials: sample index 0 at
  dtype indexes 0 through 6, plus sample indexes 1 through 3 at dtype index 0.
  Native and strict remote status agree across all 6,340 selections with no
  remote-specific status difference. Sample index 3 exposed batched float32
  least squares: routing S/D/C/Z batched QR and least-squares cuBLAS families
  removes the local-library crash, and the minimized forward/backward sample
  plus its complete four-shard differential pass.
- Refreshed the conservative linked-library audit. The current interposers
  cover 200 of 364 symbols referenced by the pinned PyTorch image; the 164
  remaining symbols are an explicit ABI backlog, not 164 observed failures.
  The GPU-free protocol probe now exercises 63 RPC calls under ASan/UBSan.
  The unattended OpInfo launcher now takes a process lock so duplicate
  admission gates cannot later overlap; the complete unit suite passes 80/80.
- Rebuilt the latest modded-nanogpt image on the current stack. Its modern
  kernel imports pass remotely, including the RTX 5090 compute target. A
  bounded two-step training launch reaches upstream's first-run compile/warmup
  phase without a remoting error; upstream labels that phase as approximately
  seven minutes, so the run was deliberately stopped at the short-test cap.
- Revalidated four current Transformers families with synthetic inputs and no
  weights or dataset downloads. GPT-2, Mamba2, VideoMAE, and DETR each complete
  forward, backward, and optimizer steps through strict remoting. Their two-step
  median wall times were 13.522, 17.729, 14.744, and 30.907 ms respectively.

## 2026-08-20 — broad second-sample differential and MAGMA isolation

- Expanded the idle-gated runner to the complete 634-name OpInfo list, strict
  remote-only execution, arbitrary sample/dtype indexes, saved-native reuse,
  explicit deploy opt-in, and durable comparison artifacts.
- Native sample index 1 executes 586 cases and reports 48 native exceptions,
  of which 42 are exhausted sample generators. The initial remote sweep
  reached 102 passes before `lu_unpack` caused an illegal access; the remaining
  failures were poisoned-context fallout, not independent gaps.
- A fresh minimization traced the fault to MAGMA's batched LU kernels, whose
  device-resident arrays contain device addresses that cannot be generically
  relocated between CUDA processes. PyTorch's semantics-equivalent cuSOLVER
  backend passes, so `--cublas-rpc` now selects it by default while preserving
  an explicit user override.
- Added the linked but previously missing `cublasSgetrfBatched` route as round
  32 and verified its required `libcublas.so.13` ELF symbol version. The GPU-free
  packet regression now covers 47 calls under ASan/UBSan; 78 unit tests pass.
  Static linked
  coverage is 184/364, with 194 total exports and 180 conservative candidates
  remaining. The monolithic rerun reached its 240-second process cap deep in
  linalg; focused `linalg.lu` passes. Round 34 therefore uses four independently
  bounded shards against the saved native result. Shard 0 passes 148 cases with
  the same 11 native exceptions and no remote-specific status difference.

## 2026-08-20 — complex strided GEMM and bounded launcher cleanup

- The first idle-gated dtype-index-3 run completed natively with 79 passes and
  the same three exhausted-dtype exceptions expected from test selection. The
  remote run reached complex64 `baddbmm` and then exited before JSON because
  `cublasCgemmStridedBatched` was not interposed.
- Added route-owned host/device pointer-mode RPC for that API as round 31. The
  focused remote `baddbmm` regression now passes in 36 ms; the 46-call packet
  test passes under ASan/UBSan. Static linked-symbol coverage rises to 183/364,
  with cuBLAS at 59/176.
- Rebuilt the client, server, host-wide, PyTorch, and OpInfo images. The
  idle-gated round-31 differential reuses the valid native artifact and passes
  with no remote-specific status difference: 79 selected cases pass on both
  sides and the same three dtype selections are exhausted.
- Every SSH operation now has a 15-second subprocess bound and keepalives;
  image deployment has a 240-second bound and kills both pipeline children on
  timeout or interruption. Cleanup restores signal handlers and reports a
  failed exact-name lease removal instead of hanging silently. The GPU-free
  unit suite is 70/70.

## 2026-08-20 — attached-state physical stack fingerprint

- Corrected the safety snapshot so its `nvidia-smi` subprocess preloads the
  canonical vendor NVML and removes endpoint overrides, bypassing the active
  host-wide shim. The repository tool now imports correctly when invoked by
  path and fails closed if the physical identity query does not finish within
  15 seconds.
- The attached-state immutable fingerprint is exactly the baseline value,
  `0f44b820510ea845efe8d12af7e7629df2369dc5538234dc865cc97c18b5c9bd`.
  This covers vendor-library hashes and targets, kernel versions, device nodes,
  the `nvidia-smi` binary, and physical GPU identity. Raw evidence is retained
  in `dev/results/raw/host-nvidia-safety-attached-20260820.json`.
- The full GPU-free unit suite was 67/67.

## 2026-08-20 — host-wide NVML process-row diagnosis

- Read-only public NVML probing proves that the attached shim correctly returns
  the local process PID and memory. The live attachment predates round 10,
  however: its library does not export the R610 `_v3` running-process symbols,
  and its legacy remote server lacks the host PID namespace. This explains why
  current host-wide `nvidia-smi` shows memory/utilization but omits process rows.
- The current reproducible image exports all compute/graphics/MPS `_v3`
  variants, and newly created persistent leases use `--pid host`. A static
  regression locks those symbols in; the full GPU-free suite was 64/64.
- Replacing the live shim or server while applications may be using it would
  violate the no-disruption rule. End-to-end process-row validation is queued
  for the next deliberate detach/reattach after the user returns.

## 2026-08-20 — durable idle-gated differential artifacts

- The shared-GPU runner now retains stdout and stderr even when a native or
  remote workload times out or fails before emitting its final JSON payload.
  It stops cleanly instead of converting that failure into a parser traceback.
- Added CPU-only regressions for successful and incomplete artifact capture;
  the full GPU-free unit suite was 63/63, including fail-closed remote-inspection
  handling in the lease collector. The refreshed persistent runner still
  sees `ninfer-serve` holding the remote GPU and has not launched a workload.

## 2026-08-20 — conservative orphaned-run reclamation

- Per-command server leases now carry mode, creation-time, and session labels
  and use Docker's no-restart policy. Persistent host-wide attachments retain
  reboot persistence and are categorically excluded from garbage collection.
- Added `rgpu gc`, which touches only an expired, labeled ephemeral container
  after verifying that its server port has no established client. SSH and
  removal errors fail closed; unknown or old unlabeled containers are retained.
- The full CPU/GPU-free unit suite was 60/60.

## 2026-08-20 — server-authenticated HTTPS endpoint mapping

- Added ordered `--endpoint https://host[:port]` mappings to both `rgpu run`
  and host-wide `attach` while keeping SSH deployment and lease identity on
  `--host`. Invalid schemes, credentials, paths, counts, duplicate endpoints,
  and ports fail before any remote action.
- CUDA and NVML use the existing TLS 1.2-or-newer client path with certificate,
  hostname, and SNI verification through the system trust store. Actual proxy
  deployment and a client-access policy remain production gates.
- The full CPU/GPU-free unit suite was 57/57. The shared remote GPU remained
  occupied, so the idle-gated dtype differential was not started.

## 2026-08-20 — resumable detach and out-of-band rescue

- Added a durable `detaching` journal phase. A detach interrupted after files
  are restored—or after a loader-refresh failure—can now distinguish desired
  rgpu files, restored backups, and safely absent new files, then resume the
  same rollback idempotently. Atomic replacements/removals and final journal
  deletion now include parent-directory `fsync` for power-loss durability.
- Added the separately installed `rgpu-rescue` command. It imports no launcher
  or network path, audits its own process mappings, refuses if CUDA/NVML is
  loaded, restores locally without contacting a server, and reports lease
  records for later cleanup. The installed pipx application exposes both
  `rgpu` and `rgpu-rescue`.
- Failure injection after all seven install replacements, all seven detach
  removals, and the post-restore `ldconfig` boundary passes, as does the
  sandbox rescue flow. The full CPU/GPU-free unit suite was 48/48 at this point.

## 2026-08-20 — double-precision BLAS-1 coverage

- Added float64 AXPY, copy, scale, norm, absolute sum, swap, and extrema-index
  routing, including correct host/device pointer-mode result behavior.
- CUDA 13.1 client/server images build successfully; the 45-call packet probe
  passes ASan/UBSan and all 32 unit tests pass. Static linked-symbol coverage
  reaches 182/364, exactly half of the conservative linked ABI inventory.

## 2026-08-20 — batched double-precision solvers

- Added double-precision batched Cholesky factor/solve and batched Jacobi
  symmetric eigendecomposition RPCs, covering shape-dependent batched linalg
  dispatch in the next OpInfo dtype tier.
- CUDA 13.1 builds pass, the packet probe covers 36 calls under ASan/UBSan, and
  all 32 unit tests pass. Explicit static coverage is now 174/364 linked symbols
  with 190 conservative backlog entries; GPU validation remains lease-gated.

## 2026-08-20 — classic double-precision solver core

- Added direct legacy double-precision QR factorization, SVD, and symmetric
  eigendecomposition workspace/execution RPCs. Together with round 27, this
  covers both generic X APIs and legacy solver selections used across PyTorch
  versions and shape-dependent paths.
- Refreshed client, server, and OpInfo images compile against CUDA 13.1. The
  packet probe now exercises 32 calls and passes ASan/UBSan; all 32 unit tests
  pass. Static linked-symbol coverage increases to 170/364, with the remaining
  194 entries retained as a conservative—not observed—backlog.
- Added an unattended admission runner for the pending dtype differential. It
  requires stable idleness, rechecks after the native baseline, retains native
  and remote JSON/stderr artifacts, and leaves rgpu's final lease guard intact.

## 2026-08-20 — double-precision Cholesky expansion

- Added direct legacy double-precision Cholesky workspace, factorization, and
  solve RPCs. This complements the existing generic cuSOLVER X API and covers
  callers that still select `cusolverDnDpotrf*`/`cusolverDnDpotrs`.
- The authoritative CUDA 13.1 server build and refreshed client/OpInfo images
  complete successfully. The protocol regression now checks 26 RPC calls and
  passes both the full 32-test suite and AddressSanitizer/UndefinedBehaviorSanitizer.
- The rebuilt static audit now covers 164 of 364 PyTorch-linked symbols, leaving
  200 as a conservative unobserved ABI backlog. No shared GPU was consumed.

## 2026-08-20 — linked-library audit and double-precision expansion

- Added a reproducible ABI audit against the pinned PyTorch image. PyTorch
  references 364 cuBLAS/cuSOLVER/cuFFT/NCCL symbols; round 26 explicitly
  interposes 161 of those (plus 10 compatibility exports). cuFFT is complete
  for the linked surface, while unobserved
  cuBLAS/cuSOLVER entry points remain a quantified compatibility backlog rather
  than an implicit completeness claim.
- Added double-precision GEMM, pointer-batched and strided-batched GEMM, GEMV,
  dot, triangular and batched triangular solve, batched LU solve/factorization, cuSOLVER LU,
  Jacobi SVD, and QR
  generation/application routing. The new server and both client interposers
  compile cleanly; a GPU-free packet regression verifies all new opcodes and
  scalar metadata and passes AddressSanitizer/UndefinedBehaviorSanitizer. The
  exact external delta is preserved as round 26.
- Added `--describe` to the OpInfo harness so future dtype/sample tiers can be
  enumerated without initializing CUDA. Dtype index 3 selects 79/82 curated
  operators; `cdist`, `complex`, and `quantile` exhaust their dtype lists.
  Empirical dtype-index 3 validation is
  queued until the remotely shared GPU is idle; the lease guard continues to
  refuse interference with the unrelated `ninfer-serve` process.

## 2026-08-20 — alternate-dtype routing and crash elimination

- Extended the upstream OpInfo harness with deterministic sample and dtype
  indices plus a forward-only isolation mode. The second generated input for
  the curated set is an exact differential: 79/82 execute on both native and
  remote CUDA, while the same three operators define no second sample.
- The first alternate dtype passes 82/82 natively and initially exposed four
  unsafe remote-library fall-throughs. Added RPC paths for `cublasDotEx`,
  complex128 GEMM and batched triangular solve, complex128 Jacobi SVD, and
  complex128 QR application. The isolated BF16 dot-product `matmul`,
  complex128 Cholesky, eigendecomposition sample construction, and least
  squares now pass remotely instead of crashing.
- The first alternate dtype now passes 82/82 remotely as well. The second
  alternate dtype matches native exactly at 80 passes and the same two
  dtype-index exhaustion errors (`cdist` and `quantile`). It exposed and fixed
  complex scalar alignment, complex dot products, strided GEMM, Jacobi SVD,
  triangular solve, QR application/generation, and LU-family routing.
- Mixed-process LU now reapplies PyTorch's cuSOLVER preference after CUDA
  priming. This avoids MAGMA's process-global queue for a remote complex64
  non-square LU while preserving native local routing.
- Preserved the server/protocol changes as exact rounds 24 through 26; applying
  round 26 atop rounds 1–25 reconstructs the current external files
  byte-for-byte. The unit suite is 32/32. No occupied GPU was preempted.

## 2026-08-20 — full OpInfo parity, CUDA VMM, and final clean rebuild

- Expanded same-process CUDA-library routing across cuBLAS, cuBLASLt,
  cuSOLVER, and cuFFT, then ran all 634 unique first-sample PyTorch 2.12
  OpInfo names in eight bounded shards. Local `cuda:0` and remote `cuda:1`
  each pass 628; the same six native/sample-generation exceptions occur on
  both sides, with zero remote-specific failures or process crashes.
- Added route translation for CUDA virtual-memory allocation descriptors used
  by `PYTORCH_ALLOC_CONF=expandable_segments:True`. Remote reservations use a
  route-specific high virtual-address region, preventing numeric pointer
  collisions with the local allocator in a mixed process. Six changing
  allocation sizes pass and the process exits cleanly on both devices.
- Rebuilt the client, portable injection artifact, hostwide artifact, PyTorch
  image, server, and server overlay from the exact round-1 through round-23
  patch chain through round 23. A normal `rgpu run`, without test monkeypatches, passes remote
  expandable-segment allocation through the newly deployed server image.
- Revalidated actual upstream nanochat, LitGPT, and TorchTitan eager training.
  Their final remote/native GPU-time ratios are 10.73, 7.31, and 2.43 before
  region optimization; nanochat falls to 2.15 with `torch.compile` and 1.10
  with whole-step CUDA Graph replay.
- The latest modded-nanogpt kernel module imports on local and remote RTX 5090.
  Its bounded full training entry point reaches Inductor compilation on both
  devices and then fails identically because a generated Triton configuration
  requests 180,248 bytes of shared memory against the GPU's 101,376-byte
  limit. The pinned single-GPU/RTX-5090 test adaptation is reproducible as an
  exact patch against commit `ecbb5862`.

## 2026-08-20 — mixed model breadth, asynchronous cuBLAS, and whole-step graphs

- Added same-process routing for classic and extended strided-batched GEMM,
  deferred cuBLAS handle state, and a composable remote SDPA fallback. All 16
  bounded Hugging Face/Diffusers families pass in a local-first process on
  remote `cuda:1`; DETR uses FP32 because native `cdist_cuda` rejects BF16.
- Disabled PyTorch's descriptor-heavy cuBLASLt addmm path by default only when
  `--cublas-rpc` is active. Equivalent classic cuBLAS reduces synchronized
  Swin from about 129 ms to 28 ms without changing local execution.
- Added `--cublas-async`. Valid compute calls use the owning stream's ordered
  deferred lane; malformed dimensions remain synchronous. The invalid-leading-
  dimension regression returns cuBLAS status 7 and subsequent work passes.
- The asynchronous 16-family sweep passes. Representative eager results are
  Llama 7.985 ms remote versus 7.662 ms native and DETR 15.594 ms versus
  10.876 ms, down from synchronized remote times of 22.223 and 33.195 ms.
- Fixed remote CUDA Graph allocation capture by starting server-side capture in
  relaxed mode. Application capture policy remains enforced on the client,
  while legal allocator calls may arrive on different RPC worker threads.
- Whole eager forward/backward/AdamW capture now passes 15/16 model families,
  including VideoMAE and both Diffusers models. Mixtral selects its upstream
  graph-safe batched-MM implementation. DETR deliberately retains compiled/eager
  regions around its CPU SciPy Hungarian matcher.
- Actual upstream whole-step results pass with matching loss trajectories:
  nanochat 1.509 ms remote versus 1.369 ms native, LitGPT 3.022 versus 2.973
  ms, and TorchTitan/FlexAttention 6.539 versus 6.348 ms.
- Preserved external changes as reproducible patch rounds 18–20 and verified
  the full patch chain reconstructs the edited client, protocol, and server.

## 2026-08-20 — same-process local/remote cuBLAS and cuBLASLt

- Added an opt-in `--cublas-rpc` execution path. Local library calls remain
  native; remote cuBLAS handles, cuBLASLt descriptors, attributes, heuristic
  results, algorithms, streams, workspaces, and tensor pointers execute in the
  GPU-owning server process.
- Implemented `cublasSgemm_v2`, `cublasSgemmEx`, `cublasGemmEx`, and the
  descriptor-based cuBLASLt matmul lifecycle. Server-side calls receive the
  exact routed CUDA context token rather than relying on implicit lane state.
- Added a deterministic one-time remote training warm-up before local CUDA
  library initialization. It covers float32, float16, and bfloat16 without
  initializing CUDA RNG and has no steady-state cost.
- Passed one-process local-first `nn.Linear` forward/loss/backward on `cuda:0`
  and remote `cuda:1` for all three dtypes with exact matching losses:
  0.4355998933 for float32 and 0.4358062744 for float16/bfloat16.
- Added `dev/tests/workloads/mixed_cublas_training.py`; the bounded process finishes
  in roughly four seconds including connection and startup.
- Re-ran 27 unit tests, mixed `ncclCommInitAll`, the nine-operation collective
  matrix, and five-step ordinary `torchrun` DDP. All pass; DDP reports exact
  final loss sum 5.375 and a 0.662-second timed region.
- Preserved the external changes as
  `dev/patches/lupine-v1-round17-cublas-rpc.patch` and verified that applying rounds
  1–17 reconstructs the edited LUPINE files exactly.
- `torch.linalg.vector_norm` exposes an unimplemented cuBLAS vector-call path;
  vector, batched, device-pointer-mode, and broader cuBLASLt APIs remain the
  next compatibility expansion. The feature therefore remains explicit rather
  than becoming the default.

## 2026-08-20 — remote-native NCCL and ordinary mixed `torchrun`

- Added protocol-5 NCCL RPC execution in the GPU-owning server process. The
  client replacement library covers communicator lifecycle, group operations,
  reductions, gathers/scatters, all-to-all, point-to-point, registration,
  split, queries, and async error polling.
- Passed mixed local/remote 1 MiB and 16 MiB collective measurements. The
  16 MiB path reaches approximately 8.2–9.1 Gbit/s versus about 9.7 Gbit/s for
  matched native ranks.
- Fixed early virtual-device enumeration by initializing the real driver before
  counting local devices. Added rank-aware opaque CUDA table selection so
  cuBLAS works on local `cuda:0` and remote `cuda:1` without host changes.
- Fixed local NCCL VMM cleanup by routing `cuMemRetainAllocationHandle`; three
  consecutive DDP sessions now exit without communicator cleanup errors.
- Passed unmodified `torchrun --standalone --nproc-per-node=2` for five exact
  TinyTransformer DDP steps. Four runs measured 0.663–0.695 s for the timed
  region, versus 0.372 s for the identical native two-host control.
- Passed the standard-launcher matrix for reduce, reduce-scatter, all-gather,
  all-to-all, gather, scatter, batched send/receive, averaged all-reduce, and
  barrier. Diagnosed an all-to-all stall as NCCL advertising a VPN interface;
  `rgpu` now detects and propagates the interface carrying the SSH route.
- Stopping only the disposable server caused a remote CUDA loop to report
  device-unavailable within the bounded test. Restarting the server immediately
  allowed a fresh exact CUDA session.
- Generated and apply-checked
  `dev/patches/lupine-v1-round14-native-nccl-rpc.patch`. No public fork or upstream
  push was created.

## 2026-08-20 — mixed local/remote NCCL checkpoint

- Traced the first mixed-rank illegal access to canonical client UVA host
  pointers embedded in NCCL's `ncclKernelComm` control structure. Added
  shallow mapped-host-pointer relocation for packed kernel launch buffers and
  small host-to-device control copies.
- Added an opt-in protocol-3 mapped-host coherence experiment. The server
  snapshots live CUDA host mappings without waiting behind NCCL's persistent
  kernel; the client merges those snapshots with the client-side socket proxy.
- A constrained one-channel NCCL smoke now passes correctly across the local
  RTX 5090 and virtualized remote RTX 5090: all-reduce and all-gather return the
  expected values on both ranks and both processes exit cleanly.
- Small all-reduce, all-gather, and broadcast coverage also passes with a 4 MiB
  coherence window. The initial all-reduce takes roughly 9 seconds, so this is
  a correctness checkpoint rather than an acceptable data plane.
- A 1 MiB mixed payload advances through all-reduce but does not complete the
  following all-gather inside 60 seconds. Sparse 64 KiB through 3 MiB windows
  either stall or violate NCCL publication semantics; NCCL's queue slot state
  spans roughly 4 MiB of its mapped allocation in this configuration.
- Added a native two-host fallback that streams a self-contained workload to
  unmodified NVIDIA/PyTorch containers on both workstations. Two-rank NCCL,
  a five-step Transformer DDP smoke, and exact collective validation pass.
- Native cross-host NCCL reaches 8.44 Gbit/s for 1 MiB, 9.73 Gbit/s for 16 MiB,
  and 9.83 Gbit/s for 64 MiB on the measured 10GbE link. This proves the link,
  GPU, NCCL, and host topology are healthy and sets the data-plane target.
- Unit coverage is now 24/24. The production attachment on port 14833 and the
  workstation NVIDIA loader/driver state were not modified; all protocol-3
  work used the disposable server on port 14835.

Conclusion: generic polling and mirroring of NCCL's mapped memory is useful for
compatibility research but cannot meet the performance target. The next mixed
path should execute NCCL's proxy/data plane natively on the GPU-owning host (or
interpose collectives into an equivalent remote-native transport), while the
generic CUDA remoting path remains the single-device compatibility fallback.

## 2026-08-20 — mixed local/remote NCCL reduction

- Ran the first real two-device experiment with one rank on the local RTX 5090
  and one rank on the remote RTX 5090. A rank-isolated launcher gives each
  worker a one-device application view, while the local worker preloads the
  physical NVIDIA libraries and the remote worker uses the shim.
- Virtualized `cuDeviceGetPCIBusId` for remote routes. NCCL now sees distinct
  bus identities (`0000:01:00.0` and `00001000:01:00.0`) instead of rejecting
  the pair as a duplicate GPU.
- Forced socket transport for the mixed topology. Both ranks establish all
  channels and complete communicator initialization; the local-rank shim
  segfault is avoided by using the native local library path.
- Added protocol-v2 support for `cuLaunchKernelEx` packed argument buffers.
  This advances NCCL from `CUDA_ERROR_NOT_SUPPORTED`, through server argument
  validation, to a real remote collective-kernel launch.
- Isolated the remaining mixed-collective blocker: NCCL proxy transports need
  continuously coherent GPU-mapped host control memory. LUPINE currently uses
  separate client and server shadows, so the launched remote kernel faults
  when it follows control pointers that a client-side proxy expects to update.
  Shared-memory transport fails earlier at the same host-registration boundary.
- Preserved the PCI and packed-launch work as reproducible rounds 11 and 12.
  The live attachment and its port-14833 server were not replaced; protocol-v2
  work was deployed only to a disposable port-14834 test server.

## 2026-08-20 — live attachment, mixed routing, and exact process telemetry

- Enabled transactional live-root attach/detach with root enforcement, idle-GPU
  admission, invoking-user SSH identity, post-attach UUID verification, and
  automatic rollback. The user validated that ordinary host `nvidia-smi`
  enumerates local `cuda:0` plus remote `cuda:1` without an rgpu shell.
- A controlled native workload on `ws-5090-1` made the remote row in bare
  `nvidia-smi` on `ws-5090-2` jump from idle to 100% utilization and roughly
  1 GiB allocated, while direct telemetry on the server changed in lockstep.
- Fixed local-route `cuMemcpyDtoHAsync_v2` and local-to-local
  `cuMemcpyAsync`. Independent PyTorch compute and device-to-host copies now
  pass on both ordinals. A Driver API regression keeps allocations on both
  devices and repeatedly switches, clears, copies, and releases both primary
  contexts successfully.
- Isolated the remaining same-process local-first PyTorch failure to CUDA
  Runtime state: the equivalent mixed Driver API test passes, while a later
  Runtime `cudaGetDevice` reports `cudaErrorDevicesUnavailable`. Separate
  processes and remote-first execution remain viable; arbitrary same-process
  switching is not yet accepted.
- Fixed current `nvidia-smi` process reporting by preserving the physical R610
  NVML export table and redirecting its compute/graphics/MPS process slots
  through remote-handle translation. Idle queries are now empty instead of
  `[Uninitialized]`, and an active remote job reports its PID, remote UUID, and
  1054 MiB allocation exactly.
- Future server leases now use the host PID namespace for process visibility,
  survive remote reboot via `--restart unless-stopped`, and are removed by
  exact container name on detach. The currently attached legacy lease remains
  untouched until the user performs a normal detach/reattach.

## 2026-08-20 — host-wide sandbox attachment

- Added `/etc/rgpu/endpoints` discovery to the CUDA and NVML shims so ordinary
  processes no longer require `LUPINE_SERVER`.
- Added a transactional root installer with atomic writes, collision backups,
  content fingerprints, fail-closed detach, and a hard lock against `/`.
- In a disposable root filesystem, bare `nvidia-smi`, bare PyTorch, the 8/8
  compile/graph suite, NCCL, DTensor/checkpoint, FSDP2, and CUDA IPC pass with
  no wrapper or CUDA environment variables.
- Validated mixed physical-plus-remote enumeration as `cuda:0` and `cuda:1` and
  computed explicitly on remote `cuda:1`; the occupied local GPU received no
  test allocation.
- Found that NVIDIA's container hook refreshes the loader cache after image
  construction. The attach-time `ldconfig` refresh deterministically restores
  rgpu library precedence; no workstation loader state was changed.
- Preserved endpoint discovery as
  `dev/patches/lupine-v1-round8-hostwide-config.patch`, verified exact eight-patch
  reconstruction, and promoted the round-8 images.
- Completed sandbox attach/detach and interruption cleanup, then irreversibly
  removed the generated 82 MB root and superseded development image tags. All
  removed artifacts remain reproducible from source.
- Added an out-of-band physical NVIDIA snapshot covering GPU/driver identity,
  canonical library and binary hashes, kernel versions, and character-device
  metadata. Two consecutive immutable fingerprints match; the existing local
  `ninfer-serve` process remains the only local compute consumer.

## 2026-08-20 — fail-closed wire-protocol negotiation

- Added an explicit HTTP/2 protocol revision to every client request and server
  response. The client waits for a compatible response once at connection
  startup; steady-state CUDA RPC framing and latency are unchanged.
- The server rejects an absent or mismatched revision with HTTP 426 before CUDA
  dispatch. CPU-only transport tests cover normal, unknown-CUDA-metadata, and
  deliberate protocol-mismatch cases; all three pass.
- Preserved the implementation as
  `dev/patches/lupine-v1-round7-protocol-negotiation.patch` and verified that the
  complete seven-patch series exactly reconstructs the source worktree.
- Rebuilt and deployed the optimized server, client, portable injection shim,
  and PyTorch image. Strict remote `nvidia-smi` passes and the promoted
  compile-and-CUDA-Graph smoke suite passes 8/8.
- Verified the negative end-to-end path with the retained round-6 client: the
  round-7 server rejects it, the client exits nonzero, and no remote compute
  process or lease remains.

## 2026-08-19 — installable runtime and real-source compiled regions

- Added `rgpu` install/deploy/run/status commands with exact server-image
  verification, remote GPU admission checks, strict remote-only execution,
  mixed local/remote enumeration, signal cleanup, and link-loss handling.
- Added portable, content-addressed userspace shim injection into existing
  workload images. An untouched PyTorch 2.12/CUDA 13 image passes strict
  `nvidia-smi`, the 8/8 smoke suite, compilation, graphs, and explicit remote
  `cuda:1` computation without changing the host NVIDIA stack.
- Expanded the upstream OpInfo sweep to 634 unique names. 628 pass; all six
  exceptions reproduce natively, leaving zero remote-specific misses.
- Trained actual upstream nanochat, LitGPT Pythia-14M, and TorchTitan Qwen3
  models. Compiled model-plus-optimizer GPU ratios are 1.20×, 0.99×, and 1.00×
  respectively; synchronized wall ratios are 1.48×, 1.09×, and 1.06×.
- TorchTitan exposed two native-matched dependency issues: a newer optional
  mask-construction keyword and default FlexAttention tiles that exceed the
  RTX 5090 shared-memory limit. The bounded harness preserves the same mask
  semantics and uses conservative upstream-supported tiles in both modes.
- Compiling the Diffusers VAE model and optimizer reduced the remote result
  from 9.277 ms to 5.020 ms versus 4.640 ms native, improving its ratio from
  1.44× to 1.08×.
- Added differential DTensor, distributed-checkpoint, CUDA IPC, and profiler
  probes. DTensor and CUDA IPC pass. Distributed checkpoint initially hung in
  NCCL all-gather; fixing missing generic `cuMemcpyAsync` server dispatch and
  count-aware stream-memory-op framing makes the original save/load pass.
- Native CUPTI profiling reports nine CUDA activities. Strict remote execution
  completes but reports `CUPTI_ERROR_NOT_INITIALIZED`, establishing a separate
  server-side observability project rather than a CUDA execution failure.

## 2026-08-19 — expanded families and ratio-driven optimization loop

- Added Swin, SegFormer, Whisper, VideoMAE, and a Diffusers VAE, bringing the
  synthetic Hugging Face matrix to 16 families. All 16 native and all 16
  frozen-client remote runs pass with no weight or dataset downloads; complete
  processes remain about five to seven seconds.
- Changed performance ranking from noisy fresh-process wall time to CUDA-event
  medians and repeated the leading cases with 100–200 steps in three fresh
  processes per side. This exposed large GPU-clock variance in short sub-10 ms
  controls and cleared false alarms for Llama and Mamba2.
- Removed steady remote `cuGetErrorString`, `cuDriverGetVersion`, and
  `cuCtxGetLimit` round trips with invalidation-aware caches. The context-limit
  cache reduced SegFormer from 12.631 to 10.528 ms median GPU time.
- Made same-server `cuStreamWaitEvent` and cooperative kernel launches ordered
  fire-and-forget operations. Wav2Vec2 fell from 12.082 to 6.282 ms in the
  comparable short probe; cooperative launch reduced SegFormer again to
  9.658 ms.
- Served rich stream-capture info locally when the client has no active
  capture. The Diffusers VAE fell from 10.168 to 9.277 ms, while the complete
  CUDA Graph/compile smoke suite continued to pass.
- Tuned VideoMAE's high-density eager launch stream to a 10 µs coalescing
  window: 4.113 to 3.706 ms. Kept the global default at 5 µs because the
  nanoGPT guardrail is faster there.
- Rejected a deferred-sender wakeup reduction after it regressed nanoGPT from
  34.82 to 37.36 ms. Reverting it produced the final 32.97 ms guardrail.
- Profiled the two remaining ratio leaders. DETR performs 28 required
  host-visible synchronizations per step for Hungarian matching; Mixtral
  performs 20 for eager expert routing. These are architecture/RTT plateaus:
  suppressing the waits would return stale host-control values and break
  PyTorch semantics.
- Preserved the second optimization round as a separate private patch and made
  the build script produce a protocol-matched server image as well as client
  and PyTorch images.

## 2026-08-19 — Hugging Face model-family matrix

- Cloned current Transformers and Diffusers locally at recorded exact commits;
  both public origins have disabled push URLs and rejecting pre-push hooks.
- Added bounded, synthetic-data training for GPT-2, Llama, T5, Mixtral,
  Mamba2, ViT, ConvNeXT, Wav2Vec2, CLIP, DETR, and a Diffusers U-Net. No test
  downloads weights or data.
- All 11 native controls and all 11 optimized-LUPINE runs pass forward,
  backward, and AdamW updates. Complete processes took about five to six
  seconds each, far below the per-test limit.
- Median synchronized step slowdown across the matrix is 1.28×. ConvNeXT
  (3.09×), Wav2Vec2 (2.85×), Mixtral (1.87×), DETR (1.76×), and CLIP (1.68×)
  exceed the 1.5× performance-triage threshold.
- Isolated an event-lifecycle issue: LUPINE fails a later elapsed-time query if
  recorded warmup timing events are discarded unqueried. Querying every pair
  before destruction makes the full event-timing probe pass. Performance
  measurements use synchronized wall time and do not include event RPCs.

## 2026-08-19 — upstream breadth and modern training probes

- Cloned TorchBench, nanochat, LitGPT, and TorchTitan locally at recorded exact
  commits. Their public origin push URLs are disabled and rejecting pre-push
  hooks are installed; no fork or upstream write was created.
- Added four bounded PyTorch OpInfo shards selecting 81 distinct CUDA operators
  and using upstream-generated sample inputs. Native passes 81/81; optimized
  LUPINE passes 80/81.
- Minimized the failure to `torch.linalg.lu_factor`, which triggers an illegal
  memory access. `linalg.svd` and `masked.sum` pass in fresh sessions and were
  only false follow-on failures from the poisoned context.
- Ran official TorchBench DCGAN and Hugging Face BERT training. DCGAN measured
  15.721 ms/batch native versus 39.132 ms through LUPINE; BERT measured
  27.617 ms versus 33.160 ms. Both completed correctly.
- Adapted latest modded-nanogpt commit `ecbb5862` to a single process and made
  its H100-specific inline-kernel target configurable. Its runtime NVRTC/Triton
  module loads both natively and through LUPINE on the RTX 5090.
- Found that full latest modded-nanogpt is not natively runnable on RTX 5090 as
  shipped: its downloaded FlashAttention 3 bundle has no compatible kernel
  image. The first compiled attempt was stopped at the 270-second cap during
  the approximately seven-minute compilation documented upstream.
- Found and minimized an independent LUPINE defect: a CPU syscall cannot write
  into a PyTorch pinned-host tensor and returns `EFAULT`. Disabling pinned
  buffers diagnostically lets the remote run reach the same FlashAttention
  hardware blocker as native.
- Identified the next correctness priorities as pinned host allocation
  semantics, `lu_factor`, and NCCL/distributed composition before further
  single-GPU latency tuning.

## 2026-08-19 — extended workload and subsystem validation

- Ran 200 nanoGPT steps: 30.77 ms native versus 38.09 ms through optimized
  LUPINE (1.24×); complete-process time was 58.99 s versus 63.72 s (1.08×).
- Added a bounded PyTorch matrix covering ResNet-50 and ViT training, fused
  attention, cuFFT, cuSOLVER, cuSPARSE, CUDA Graph replay, and a 256 MiB
  bidirectional host-transfer stress case.
- All eight subsystem cases passed through LUPINE. The seven compute cases
  ranged from 1.12× to 1.29× native, with a 1.27× median slowdown.
- The 512 MiB aggregate transfer took 580.76 ms remotely versus 9.29 ms
  natively, reaching 7.40 Gb/s on the 10 GbE link and confirming a hardware
  bandwidth boundary for transfer-heavy workloads.
- Reconfirmed the major functionality gap: native single-rank NCCL passes,
  while optimized LUPINE exits 139 during NCCL initialization. A subsequent
  eight-check smoke run passed, showing the server survives the client fault.
- Recommendation: prioritize distributed/NCCL support and broader API
  compatibility before further single-GPU steady-state optimization.

## 2026-08-19 — LUPINE steady-state optimization

- Added per-operation RPC profiling and separated cold compilation traffic
  from four steady nanoGPT steps.
- Identified 10,246 pointer-attribute and 5,196 function-attribute round trips
  in that four-step window, accounting for about 1.22 seconds of wait time.
- Added lifetime-aware client caches for pointer/function metadata and a
  redundant kernel-attribute setter fast path.
- Preserved the changes as a private patch against exact tag v1.0.0 and added
  reproducible client/PyTorch overlay build tooling.
- Passed the eight-check PyTorch smoke suite after the optimization.
- Measured 42.43 ms/step over a 20-step steady window, versus 399.07 ms
  upstream and 31.07 ms native on the same remote RTX 5090.
- Tested and rejected client-side event pooling and TCP corking because they
  did not improve steady-state time and broadened semantic risk.
- Added an asynchronous ordered RPC coalescer for fire-and-forget CUDA calls.
  A focused transport test verifies ordering, synchronous draining, and
  progress of a lone asynchronous request; the full HTTP/2 test passes.
- Tuned the coalescing window across 5, 10, 25, and 100 µs and retained 10 µs.
  The final 20-step steady window is 36.48 ms/step: 10.94× faster than
  upstream LUPINE and 1.17× native.

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
## 2026-08-20 — complete mixed OpInfo differential and opaque-library routing

- Added per-route cuFFT plan virtualization and RPC execution while preserving
  native cuFFT forwarding on local `cuda:0`; FFT2, inverse FFT, and STFT now
  pass on both devices in one process.
- Expanded cuSOLVER routing across LU, QR, Cholesky, symmetric-indefinite,
  eigenvalue, and triangular-solve paths, including server-owned host
  workspaces for generic APIs.
- Expanded cuBLAS routing with GEMV, dot, complex GEMM, triangular batched
  solve, and real/complex batched getrs paths discovered by upstream backward
  tests.
- Ran all 634 unique first-sample PyTorch 2.12 OpInfo names in eight bounded
  shards on local `cuda:0` and remote `cuda:1`. Both sides pass 628 and report
  the same six upstream/sample-generation exceptions; remote-only failures are
  zero and no shard crashes.
- Preserved the LUPINE changes as the exact round-22 patch and kept all work in
  the private repository; no public fork or push was created.

## 2026-08-20 — fourth dtype differential and strict remote-only library RPC

- Added complex64 strided-batched GEMM as protocol call 46 after the fourth
  dtype tier isolated `baddbmm` as the only remote crash. The packet regression
  passes under ASan/UBSan and the rebuilt server image is pinned as round 31.
- Completed the idle-gated native/remote differential. Both sides pass 79
  selected operators and report the same three exhausted-dtype selections;
  remote-specific status differences are empty. Summed per-operator time is
  4.893 seconds native and 5.173 seconds remote (1.057x) in this short probe.
- Removed an unnecessary launcher constraint tying opaque-library RPC to
  mixed local/remote enumeration. A strict remote-only complex `baddbmm` now
  passes with no local NVIDIA devices mounted into the client, allowing remote
  work while the local RTX 5090 remains occupied by another user.
- Hardened detach so local physical-GPU restoration is always checked and all
  remote leases are attempted even if one remote cleanup fails. The installed
  `rgpu`/`rgpu-rescue` entry points include this change; 74 CPU-only tests pass.
