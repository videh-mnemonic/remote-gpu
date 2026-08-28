# Remote PyTorch on gpu-host: research and validation plan

Status: initial bake-off, optimization, and first upstream-breadth pass executed;
see `docs/compatibility.md` and `docs/north-star.md`

Research date: 2026-08-19

> Execution note: the initial managed-sandbox observations below were later
> superseded by an unrestricted run. Hardware, SSH, Docker GPU access, and the
> 10 GbE link all passed. The measured candidate results are maintained in the
> compatibility report and execution log.

Client/application host: `client-host`

GPU server: `gpu-host` (`192.0.2.10`)

## Executive recommendation (revised)

The controlled OSS bake-off selected LUPINE v1 as the only current-PyTorch
starting point, and a private latency patch brought resident compute close to
native. Functionality now takes priority: close pinned-host memory,
`torch.linalg.lu_factor`, and NCCL gaps, while continuing upstream PyTorch and
real-training coverage. The exact mixed local/remote target and acceptance
gates are maintained in `docs/north-star.md`.

The primary workload is a pinned, single-GPU adaptation of
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Each viable
backend must run the same training code locally and remotely. Compare both a
fixed-work trial and a fixed 3 min steady-state trial; keep data preparation,
`torch.compile` compilation, initialization, and teardown as separately
reported phases.

Only after the bake-off and broader PyTorch compatibility testing should we
decide whether to adopt an existing project, maintain private patches, combine
ideas, or build something new. If an OSS project already satisfies the target,
this repository should remain a thin reproducibility, testing, and operations
layer.

The eventual user interface remains tentatively:

```text
rgpu run --host gpu-host -- python train.py
```

The Python process and its CPU work remain on client-host. Calls that PyTorch
makes to `libcuda.so.1` are intercepted, serialized, sent to gpu-host, and
executed against its physical RTX 5090. The application should need no source
change. A second `rgpu exec` mode that runs the whole container on gpu-host
over SSH should be retained as a performance control and pragmatic fallback,
but it is not the requested transparent GPU-over-IP mode.

Do not write a CUDA remoting layer from scratch during the bake-off. Modern
PyTorch exercises a large and version-sensitive surface: CUDA driver APIs,
`cuGetProcAddress`, library loading, streams, events, CUDA Graphs, managed
memory, cuBLAS, cuDNN, NVML, and JIT-loaded kernels. LUPINE is Apache-2.0,
actively developed, generates shims from CUDA headers, publishes current CUDA
client/server images, and already has CUDA, cuBLAS, cuDNN, CUDA Graph, and
PyTorch tests. It is a credible bootstrap, not yet a proven production
dependency: its project has open work on quotas, topology/health reporting,
vLLM compatibility, virtual handle creation, and graph-parameter round trips.
That makes it the first candidate, not the predetermined winner.

The first release should be deliberately narrow:

- Trusted users on the dedicated point-to-point Ethernet link.
- One remote client lease per physical GPU; no simultaneous sharing.
- Linux containers, one pinned CUDA/PyTorch combination, one RTX 5090.
- PyTorch eager, `torch.compile`, mixed precision, streams/events, and CUDA
  Graphs within an explicit tested compatibility matrix.
- Fail closed if the remote connection dies; never silently fall back to the
  local GPU.

The eventual system would replicate Thunder Compute's core execution mechanism,
not its fleet-scale scheduler or every proprietary optimization. Thunder
describes its own design
as CUDA-layer translation of calls into network messages over TCP, with a
dedicated GPU for a process and workload-dependent slowdowns. That is the same
architectural family proposed here.

## What was observed locally

Read-only inspection of client-host found:

| Item | Observation | Consequence |
|---|---|---|
| OS | Ubuntu 26.04 LTS, kernel 7.0 | Prefer containers over host-library installation. |
| CPU/RAM | Threadripper 7970X, 32 cores / 64 threads, 123 GiB RAM | Enough CPU and memory for serialization and staging experiments. |
| GPU | RTX 5090, Blackwell, PCIe device `01:00.0` | Matches the intended peer GPU class. |
| NVIDIA kernel driver | Open kernel module 610.43.02 | Validate user-space/container compatibility in gate 0. |
| Direct NIC | Aquantia AQC113 10 GbE, `eno2`, full duplex | TCP is the initial transport; the NIC is not an RDMA target. |
| MTU | 9,000 on `eno2` | Jumbo frames are already configured locally; verify end-to-end. |
| NIC PCIe attachment | 16 GT/s ×2 | The NIC's PCIe attachment exceeds 10 GbE line rate. |
| Container engine | Docker present | Use pinned OCI images for both ends. |
| Host Python | 3.14.4; no importable PyTorch found | Do not depend on host Python. |

`nvidia-smi` could not access the GPU and socket creation was denied from the
managed execution environment, even though the NVIDIA kernel modules and PCIe
device are present. The
same socket restriction prevented `ping` and SSH, so no claims are made about
gpu-host, RTT, packet loss, or achieved throughput yet. Those are the first
tests to run from an unrestricted shell.

NVIDIA lists the RTX 5090 as PCIe Gen 5 with 32 GB GDDR7 and no NVLink. The
direct Ethernet line rate is only 10 Gb/s = 1.25 GB/s before headers and TCP
overheads. Consequently, a one-time 32 GB transfer has an absolute wire-time
floor of 25.6 s. API remoting can be near-native only when most state remains
on the server GPU and kernels are large enough to amortize network call
latency. Transfer-heavy or fine-grained synchronized programs cannot be made
near-native by software alone.

Hardware source: [NVIDIA RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).

## Research findings

### Thunder Compute's disclosed model

Thunder Compute says its virtualization is below the workload at the CUDA
layer, translates CUDA calls into network messages, communicates over TCP, and
assigns the whole GPU to an instance. It reports an initial connection cost in
the tens of milliseconds, says common workloads can have negligible impact,
and acknowledges roughly 2× slowdown in unfavorable edge cases. These are
vendor claims rather than reproducible public benchmarks, so they inform the
architecture but not our acceptance criteria.

Sources:

- [Thunder Compute: How GPU-over-TCP works](https://www.thundercompute.com/blog/how-thunder-compute-works-gpu-over-tcp)
- [Thunder Compute: GPU virtualization approaches](https://www.thundercompute.com/blog/why-network-based-gpu-virtualization-is-the-future)
- [Thunder Compute enterprise description](https://www.thundercompute.com/enterprise)

### Candidate implementations

| Candidate | Fit | Decision |
|---|---|---|
| LUPINE | Current Apache-2.0 CUDA driver/NVML shims, header-driven code generation, HTTP/2/TCP transport, published CUDA images, PyTorch tests, active development. The former `kevmo314/scuda` URL now redirects here, so SCUDA is not a separate candidate. | Test first: published image, source build, native smoke, remote smoke, then nanoGPT. |
| Cricket | MIT, academically documented RPC split, CUDA 12.1 example, TCP and InfiniBand transports; PyTorch support is explicitly experimental. | Test second from a pinned local clone; stop at a documented incompatibility if modern PyTorch cannot initialize. |
| GVirtuS | Open source but its public setup targets Ubuntu 18.04 and CUDA 10.2. | Attempt a bounded build/smoke screen. Run the common suite only if it reaches modern PyTorch without a substantial port. |
| rCUDA | Strong historical research results and InfiniBand support, but not a straightforward current open-source base. | Record as non-runnable unless current source and a usable license can be obtained. |
| Juice | Commercial GPU-over-IP with transparent CUDA support; public docs describe a narrower tested CUDA/PyTorch matrix. | Product comparator, not the repository foundation. |
| Whole-process SSH/container execution | Near-native GPU behavior because CUDA remains local to gpu-host; requires code/data/environment staging and moves CPU work. | Keep as control and fallback, not primary mode. |
| PyTorch RPC or application service | Can perform well but requires application changes and is not a virtual CUDA device. | Out of scope for transparent mode. |
| New remoting stack from scratch | Maximum control, but the API/ABI and semantic surface is large and changes with CUDA. | Decide only after the bake-off identifies shared gaps in the viable OSS candidates. |

Primary implementation sources:

- [LUPINE repository and usage](https://github.com/lupinemachines/lupine)
- [LUPINE open issues](https://github.com/lupinemachines/lupine/issues)
- [Cricket repository](https://github.com/RWTH-ACS/cricket)
- [GVirtuS archived setup](https://github.com/cjg/GVirtuS)
- [Juice architecture overview](https://juice-labs.github.io/juice-docs/docs/juice/intro)
- [Juice CUDA compatibility](https://docs.juicelabs.co/docs/juice/user-guide/cli-app/cuda-compatibility)

### External-source and publication policy

This is a public repository. Public dependencies and benchmark repositories
must be cloned locally; they must never be forked on GitHub or pushed to their
public upstreams.

- Clone public repositories beneath an ignored `dev/external/` directory.
- Immediately disable pushing in each clone by setting its push URL to an
  intentionally invalid local value. Never run `gh repo fork`.
- Record upstream URL, exact commit, retrieval date, license, build profile,
  and content checksum in `dev/manifests/upstreams.lock`.
- Keep local experiments in unpushed branches or detached worktrees. Store any
  changes we need as patch files under `dev/patches/<project>/` in this private
  repository, with the upstream commit they apply to.
- Do not add nested public `.git` directories, downloaded datasets, container
  layers, or bulky traces to this repository. Add ignore rules before cloning.
- Do not push anything anywhere unless the user separately authorizes the
  destination and exact content. In particular, never create a public fork or
  submit an issue/pull request as part of this work.
- Preserve all upstream licenses and notices when building containers or
  redistributing source-derived artifacts inside the private repository.

The public repositories currently expected are LUPINE, Cricket, GVirtuS,
modded-nanogpt, PyTorch (a shallow source checkout matching the tested wheel),
and the official PyTorch Benchmark/TorchBench repository.

### Optimization lessons

The new Gleam paper identifies the two dominant LAN remoting costs: repeated
bulk transfer of stationary model weights and accumulated latency from frequent
API calls. Its proposed remedies—server-side weight caching, asynchronous API
execution, locally simulated state, and prefetched resource handles—provide a
useful optimization roadmap if baseline LUPINE is correct but too slow. These
ideas must be introduced only with differential correctness tests because they
relax request/response behavior and complicate recovery.

NVIDIA documents two application-level mechanisms that matter here:

- Pinned host memory and non-default streams allow asynchronous transfer and
  compute overlap. Pinned memory is scarce and expensive to create, so pools
  should be measured and bounded rather than enabled indiscriminately.
- CUDA Graphs pay setup once and replay a workflow with much lower CPU launch
  overhead. Across a network, graph replay can also collapse many per-kernel
  submissions into fewer RPC interactions, provided LUPINE preserves graph
  semantics.

Sources:

- [Gleam: Adaptive Network-Efficient CUDA API Remoting](https://arxiv.org/abs/2607.23115)
- [NVIDIA CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## Bake-off architecture

```text
client-host                                      gpu-host
┌────────────────────────────┐                 ┌───────────────────────────┐
│ rgpu CLI                   │  lease/status   │ lease guard               │
│ pinned PyTorch container   │ ──────────────► │ candidate backend server  │
│ application CPU process    │                 │ real NVIDIA libcuda       │
│ candidate CUDA/NVML shims  │ ══ candidate ══►│ CUDA context + RTX 5090   │
└────────────────────────────┘   10 GbE/MTU 9000└───────────────────────────┘
```

### Data plane

For each candidate, use its intended driver/runtime interposition mechanism,
not a Python monkey patch. PyTorch and native extensions should continue to
discover CUDA through the normal dynamic-library ABI.
Use the direct `192.0.2.x` interface and host networking to avoid bridge/NAT
overhead. Bind the server only to the point-to-point address and firewall it to
client-host.

Plain TCP is acceptable only because this is a trusted private cable. TLS
termination adds operational complexity and some overhead without protecting
against a hostile local root user. If this ever traverses a switch shared by
untrusted systems, authentication and encryption become a release blocker.

### Control plane and coexistence

For two workstations, use an explicit lease rather than Kubernetes or Slurm.
The lease service should:

1. Report server reachability, GPU identity, memory use, active compute
   processes, current lease owner, and expiry.
2. Refuse a lease if a non-project process already uses the GPU.
3. Start or admit a candidate backend session only after a lease is granted.
4. Refresh the lease with heartbeats and drain the session on clean exit.
5. Never kill an unknown process and never reset the GPU automatically.
6. Reclaim only an expired lease whose process and connection are both gone.

The initial policy is exclusive tenancy. RTX 5090 has no MIG, and opportunistic
multi-process sharing would make benchmarks unpredictable and could interfere
with other users. A listening daemon is acceptable if it does not create a
CUDA context or reserve VRAM before a lease; verify that behavior.

### Reproducibility and user interface

Pin the candidate source commit and image digest, CUDA client/server version,
Python and PyTorch wheels, CUDA libraries, and minimum/tested NVIDIA drivers as
one compatibility profile.

The CLI should support:

```text
rgpu status gpu-host
rgpu run --host gpu-host -- python train.py
rgpu run --local -- python train.py
rgpu exec --host gpu-host -- python train.py
rgpu doctor
rgpu bench --suite smoke
```

`rgpu run` mounts the working tree read-only by default, uses an explicit
writable output directory, forwards exit status and signals, labels all logs
with a session ID, and prints the actual remote GPU UUID. It must detect that
the shim loaded; accidental use of client-host's local `libcuda` is a test
failure.

### Implemented repository shape

```text
README.md
docs/
lupine/                       # canonical integrated transport source
rgpu-client/                  # Python CLI, interposers, client packaging
rgpu-host/                    # GPU-host server packaging
dev/
  tools/
  tests/
  containers/
  patches/
  manifests/
  config/
  benchmarks/
  results/
  external/                   # gitignored public clones; never pushed
```

Keep orchestration, tests, patch files, manifests, and small result summaries
in this repository. Every tested candidate gets an exact lock entry. Source
changes remain private patch series rather than public forks or untraceable
binaries.

## Runtime budget and fast-funnel policy

The discovery pass is designed to finish in a few hours, with an 8 h stop-and-
report ceiling. No individual native baseline, remote benchmark, PyTorch test
shard, or TorchBench model trial may run for 5 min or longer.

| Activity | Budget |
|---|---|
| CUDA/PyTorch smoke test | Target ≤ 60 s; terminate at 2 min. |
| nanoGPT steady-state timed window | 3 min. |
| nanoGPT fixed-work baseline | Choose work that takes about 2 min natively; terminate at 4 min 30 s and record timeout. |
| Individual microbenchmark group | ≤ 2 min after warm-up. |
| PyTorch test shard or TorchBench model | ≤ 4 min; split or reduce selectors if it exceeds the cap. |
| Candidate build/setup | 45 min CPU-wall budget; legacy compatibility investigation gets 30 min before exclusion. |
| Full first-pass bake-off | Target 4–6 h; stop new work at 8 h and publish partial dev/results/blockers. |

Use a fast funnel:

1. One short native sanity run establishes that a test itself works.
2. One short remote screening run decides whether a candidate advances.
3. Only candidates that complete nanoGPT correctly receive the broader suite.
4. Only the leading one or two candidates receive repeat trials. Finalists get
   three total measurements; eliminated candidates do not.
5. Cache source clones, containers, compiled artifacts, and prepared data across
   candidates where doing so does not alter the tested software profile.

Compilation, image download, and dataset preparation are recorded but do not
extend GPU occupancy. If a task threatens the overall 8 h ceiling, preserve the
current evidence and report the deferred test rather than silently running into
an overnight experiment.

## Performance model

For one iteration, record:

- T_gpu: server GPU execution time.
- D_in and D_out: bytes crossing the Ethernet link.
- B_net: achieved one-way application payload bandwidth.
- N_sync: CUDA/API operations that force a serialized network response.
- RTT: network round-trip time.

A useful lower-bound model is:

```text
T_remote  ≥  max(T_gpu, (D_in + D_out) / B_net) + N_sync · RTT
```

Some transfers and calls can overlap, so this is a diagnostic model rather
than an exact predictor. It makes the two levers explicit: reduce bytes crossing
the wire and reduce serialized round trips. Faster serialization cannot defeat
the 10 GbE bound.

Always report both:

```text
slowdown       = T_remote / T_native
efficiency     = throughput_remote / throughput_native
```

The primary native comparator is the same workload run directly on gpu-host,
because it isolates remoting overhead. Run client-host local as a second native
comparator to answer the user's practical question and expose host variance.

## Iterative implementation plan

### Phase 0 — baseline, provenance, and common harness

- From an unrestricted shell, inventory gpu-host and confirm its GPU, driver,
  OS, Docker, CPU, RAM, NIC, MTU, and active users.
- Resolve `nvidia-smi` and a minimal CUDA container on both hosts before adding
  remoting.
- Measure direct-link RTT, loss, TCP throughput in both directions, and CPU
  cost. Confirm traffic routes over `eno2`, not Wi-Fi, Tailscale, or another
  interface.
- Create ignore rules, `dev/manifests/upstreams.lock`, result schema, and scripts
  that disable push URLs before cloning any public repository locally.
- Pin one common Python/PyTorch/CUDA profile supported natively by both RTX
  5090s. Do not change it per candidate unless incompatibility is itself the
  recorded result.
- Clone and pin modded-nanogpt locally. Prepare its dataset once on shared/local
  storage outside timed runs and preserve dataset checksums.
- Establish native baselines on gpu-host and client-host with identical
  containers, commits, data, environment, and single-GPU configuration.

Exit gate: reproducible native PyTorch and nanoGPT runs work on both hosts; the
network baseline is recorded; upstream locks and no-push safeguards are
verified; neither GPU has an unknown owner when tests begin.

### Phase 1 — bounded candidate viability screen

Evaluate candidates one at a time in this order: LUPINE, Cricket, GVirtuS.
For each candidate:

1. Clone locally without a public fork, pin the exact commit, preserve license,
   and disable the clone's push URL.
2. Build its own documented examples and run its upstream CPU-only/unit tests.
3. Run native loopback mode where supported, then A→B remote device discovery.
4. Run allocation, copy, kernel, cuBLAS, cuDNN, autograd, and `torch.compile`
   smoke tests.
5. Attempt the pinned nanoGPT script only if all prerequisite smoke tests pass.
6. Save build logs and classify the first hard blocker without immediately
   porting or optimizing the project.

Time-box legacy enablement work. A candidate that assumes an old CUDA ABI and
cannot initialize the common PyTorch profile receives a documented
`not viable on current stack` result; it does not consume repeated GPU trials.

Exit gate: every surveyed OSS project has either a reproducible remote smoke
result or a precise exclusion reason. Every viable project has completed at
least one short nanoGPT training segment with correct loss movement and cleanup.

### Phase 2 — common nanoGPT bake-off

Run each viable candidate in four configurations:

1. Native on gpu-host.
2. Remote CUDA calls from client-host to gpu-host.
3. Whole-process SSH/container execution on gpu-host.
4. Native on client-host.

Use two complementary comparisons:

- **Fixed work:** calibrate a step count whose native gpu-host steady-state
  training takes about 2 min, then run exactly those steps everywhere. Compare
  elapsed time, tokens/s, final loss, and loss trajectory.
- **Fixed time:** after compilation and warm-up, train for exactly 3 min in
  every configuration. Compare completed steps, tokens, tokens/s, and loss
  progress. Stop only at a safe optimizer-step boundary.

Record cold start, dependency/JIT compilation, model/optimizer initialization,
initial parameter transfer, steady-state training, evaluation, checkpointing,
and teardown separately. Perform one untimed warm-up before timed repeats, but
also retain a dedicated cold-run measurement. Use identical seeds and input
token order; numerical trajectories must match native within the tolerance of
the pinned mixed-precision configuration.

Exit gate: one fixed-work and one fixed-time screening trial are complete for
each viable backend, or a reproducible nanoGPT incompatibility is documented
with a minimized test. Repeat until there are three total trials only for the
leading one or two candidates and their native comparator.

### Phase 3 — broader PyTorch compatibility

- Shallow-clone the PyTorch tag matching the installed wheel and run selected
  upstream `test_cuda.py` and `test_ops.py`/OpInfo CUDA tests against native and
  remote devices. Expand subsets by feature area rather than immediately
  running the entire PyTorch suite. Keep each selector shard below 4 min and
  cap the first compatibility pass at 30 min per advancing candidate.
- Clone the official [PyTorch Benchmark/TorchBench](https://github.com/pytorch/benchmark)
  repository and run a curated training/evaluation matrix spanning convolution,
  transformers, eager mode, compilation, dynamic shapes, and common libraries.
  Begin with four representative models; each model has a 4 min cap.
- Run each candidate's own test suite first, then the common compatibility
  matrix below, then upstream PyTorch tests, then TorchBench models.
- Differentially compare values, exceptions, device properties, lifecycle,
  leaks, and failures. Minimize every mismatch into a project-local regression.
- Do not claim support for `all of PyTorch`; publish exact tested commits,
  features, pass counts, expected failures, and untested areas.

Exit gate: the selected required subset passes completely for at least one OSS
candidate; every omission or expected failure is explicit and reproducible.

### Phase 4 — evidence-based decision

Compare candidates on correctness breadth, nanoGPT steady-state efficiency,
cold-start behavior, maintainability, license, activity, observability,
failure cleanup, and patch size. Then choose among:

- Adopt the best candidate unchanged and build only private wrappers/tests.
- Adopt it with a small private patch series.
- Port a specific missing feature from another compatible project if licenses
  allow it.
- Use whole-process remote execution if transparent remoting is not viable.
- Design a new implementation only if the bake-off demonstrates a concrete,
  shared architectural limitation worth solving.

No transport tuning, caching, lease service, or production CLI work begins
before this decision unless it is required to make the comparison fair.

### Phase 5 — optimize and harden the chosen path

Only after selection, use traces to decide whether to optimize RPC round trips,
copy pipelining, bounded pinned-memory pools, CUDA Graph replay, query caching,
or persistent immutable weights. Add leases, graceful drain, health checks,
metrics, failure injection, and bounded soak tests around the chosen backend.

Exit gate: agreed compatibility and performance targets pass without leaks,
or a theoretical/network bound and the required next change are documented.

## Test plan

### Gate 0: network and host qualification (no GPU load)

Collect machine-readable JSON plus raw command output for:

- Host, kernel, CPU, memory, PCI topology, driver packages, and Docker runtime.
- `ethtool` link/driver/offloads/ring sizes and MTU at both ends.
- Route selection and neighbor state for `192.0.2.10`.
- 1,000 ping samples after warm-up; report min, median, p95, p99, maximum, loss.
- `iperf3` one-way A→B, B→A, and bidirectional trials, 1 and 4 streams, 30 s
  each; report throughput, retransmits, CPU, and variance.

Initial targets, to revise from observed hardware stability:

- 0 % packet loss.
- RTT p99 ≤ 0.5 ms on the direct idle link.
- Sustained one-way TCP throughput ≥ 9.0 Gb/s with low retransmits.
- No MTU black hole at 9,000 bytes.

### Gate 1: smoke and identity

- Remote `nvidia-smi -L` through the NVML shim.
- `torch.cuda.is_available()` and device properties.
- Allocate/free across sizes; host→device→host round trip with byte comparison.
- Vector add and matrix multiply with CPU-reference comparison.
- Intentional invalid device, invalid pointer, and out-of-memory cases.
- SIGINT during compute and client process death.
- Assert the server GPU UUID is gpu-host's and client-host utilization/VRAM do
  not increase beyond monitoring noise.

### Gate 2: required PyTorch compatibility matrix

| Area | Required cases |
|---|---|
| Tensor/math | Elementwise, reductions, GEMM, broadcasting, non-contiguous tensors, dtypes FP32/FP16/BF16/INT. |
| Autograd/training | Forward, backward, optimizer step, gradient accumulation, checkpoint save/load. |
| Libraries | cuBLAS, cuDNN convolution, SDPA paths available in the pinned profile. |
| Execution | Default and non-default streams, events, synchronization, CUDA Graph capture/replay. |
| Transfers | Pageable and pinned memory, blocking and non-blocking copies, varied sizes and directions. |
| Memory | Caching allocator, empty-cache behavior, OOM recovery, managed memory where exposed. |
| Compilation | `torch.compile`, Inductor/Triton JIT compile/load/run, a minimal custom extension. |
| Numerics | Fixed seeds, RNG state save/restore, native-vs-remote tolerances appropriate to dtype. |
| Processes | DataLoader workers; spawned CUDA child only if declared supported. Fork-after-CUDA should fail clearly. |
| Introspection | Device count/name/capability/memory and selected NVML queries. |

Correctness acceptance is strict: 100 % of required tests pass. Performance
never compensates for incorrect CUDA semantics.

### Gate 3: microbenchmarks

For every test, use warm-up, explicit accelerator synchronization, multiple
replicates, median/p95, and saved environment metadata. PyTorch's benchmark
utilities perform warm-up and accelerator synchronization and are the preferred
starting point: [PyTorch benchmark guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).

Measure:

- Cold connection, CUDA initialization, first allocation, first kernel, and
  clean shutdown.
- Kernel launch latency for empty/tiny kernels, with eager and CUDA Graph replay.
- Synchronous API/query latency and RPC count.
- H→D and D→H transfer bandwidth at 4 KiB through 1 GiB, pageable and pinned.
- Full-duplex transfer and transfer/compute overlap across stream counts and
  chunk sizes.
- Allocation/free and stream/event create/destroy rates.
- GEMM sizes from launch-bound to compute-bound.
- CPU utilization, context switches, NIC throughput/retransmits, GPU
  utilization, power, clocks, and thermals.

Run four configurations in randomized order:

1. Native on gpu-host.
2. Remote CUDA API from client-host to gpu-host.
3. Whole-process remote execution on gpu-host.
4. Native on client-host.

### Gate 4: modded-nanoGPT comparison

Use a pinned modded-nanogpt commit and one checked-in private harness patch that
makes the current training program run on one GPU without changing its model,
token order, optimizer semantics, or precision. The current public benchmark is
optimized for multiple H100s, so its published record time is context, not a
performance target for a single RTX 5090.

For each of the four configurations and every viable backend, report:

- Exact upstream and harness commits, image digests, driver/toolkit/library
  versions, seed, effective global batch, sequence length, and token order.
- Cold initialization and compile time.
- Initial model/optimizer transfer bytes and time.
- Fixed-work elapsed time and slowdown relative to native gpu-host.
- Fixed-time completed optimizer steps, tokens, tokens/s, loss start/end, and
  sampled loss trajectory.
- Evaluation/checkpoint time, RPC calls and bytes where exposed, client/server
  CPU, NIC throughput, GPU utilization, clocks, power, temperature, and peak
  VRAM.
- Cleanup status and whether client-host's local GPU remained idle during remote
  execution.

No backend-specific optimization is allowed during the first comparison beyond
the configuration necessary to run correctly. If a backend cannot run the
script, minimize the failing feature—likely compilation/JIT loading,
FlexAttention/SDPA, distributed initialization, a library call, or graph
behavior—and record that as a meaningful compatibility result.

### Gate 5: broader open-source PyTorch coverage

Use three layers so “supports PyTorch” is backed by evidence:

1. Matching upstream PyTorch tests: selected CUDA lifecycle tests from
   [`test_cuda.py`](https://github.com/pytorch/pytorch/blob/main/test/test_cuda.py)
   and operator/gradient cases from
   [`test_ops.py`](https://github.com/pytorch/pytorch/blob/main/test/test_ops.py).
2. The official [TorchBench](https://github.com/pytorch/benchmark) suite with a
   curated model set covering CNN training, transformer training/inference,
   eager execution, and `torch.compile`.
3. Focused project tests for streams/events, allocators, CUDA Graphs, pinned
   transfers, NVML, errors, cleanup, and connection loss.

Publish pass/fail/skip counts and exact test selectors. A passing nanoGPT run is
strong evidence for that workload but is not evidence of complete PyTorch API
coverage. Conversely, a legacy backend that fails initialization remains in the
comparison table with its exact blocker rather than disappearing from the
results.

The bake-off initially ranks rather than enforces speculative performance
thresholds. After native variance and candidate behavior are measured, agree on
adoption targets for steady-state slowdown, cold start, compatibility breadth,
and reliability before optimization begins.

### Statistical and resource discipline

- Check GPU ownership immediately before every GPU test and abort if occupied.
- Use the shortest trial that produces stable intervals; begin at 10–30 s per
  workload, not hours.
- Separate setup from measurement and cool down only when thermals require it.
- Use one screening trial for every candidate. Use three total independent
  trials only for the leading one or two candidates and native comparator;
  microbenchmarks may use many short inner replicates within their time cap.
- Randomize native/remote order to reduce drift; record clocks and temperature.
- Reject comparisons made across different software profiles.
- Store summaries in Git; keep bulky traces outside Git with checksums.
- Do not run a soak during the discovery bake-off. Reliability soak testing is
  deferred until after a candidate is selected.

## Iteration loop and decision rules

During the bake-off, for each failed gate:

1. Reproduce with the smallest CUDA or PyTorch case.
2. Compare native gpu-host, remote mode, and whole-process control.
3. Capture RPC count/bytes/blocking time and host/network/GPU counters.
4. Classify the bottleneck or semantic mismatch.
5. Minimize and document the failure before changing upstream source.
6. If a small compatibility patch is necessary to continue evaluation, store
   it only in this private repository and label patched and unpatched results.
7. Re-run the failed test, its category, then all earlier gates. Keep a patch
   only if correctness remains exact and its effect is outside run-to-run noise.

Decision points:

- If a candidate cannot pass basic PyTorch correctness on the common profile,
  record the blocker and continue to the next candidate. Do not silently change
  CUDA/PyTorch versions just to improve one project's score.
- If no OSS candidate reaches nanoGPT, compare their minimized blockers before
  deciding between a bounded port, a new shim, and whole-process execution; do
  not drift silently into a rewrite.
- If RTT dominates, prioritize asynchronous safe calls, local query caches,
  handle prefetch, and CUDA Graphs.
- If bandwidth dominates, prioritize residency, pinned pipelining, smaller
  inputs/results, and eventually immutable weight caching. A faster NIC is the
  only general cure for irreducible transfer volume.
- If serialization/CPU dominates, profile copies and encoders, remove redundant
  copies, use scatter/gather, and pin the data-plane threads before changing the
  protocol.
- If a test is below target for a theoretical reason, document the bound and
  propose the smallest hardware or workload change that can satisfy it.

## Risks and non-goals

- **Version fragility:** CUDA driver entry points and private export tables
  evolve. A tested-version matrix and generated stubs are mandatory.
- **Failure semantics:** losing a stateful TCP session can leave uncertain GPU
  state. Transparent retry is unsafe; terminate the application and clean the
  server session.
- **Unified/managed memory:** page ownership assumptions do not naturally span
  Ethernet and can cause large hidden transfers. Test it, but do not promise
  local behavior or performance initially.
- **Security:** a remote CUDA service accepts executable GPU code and complex
  serialized inputs. Restrict it to trusted hosts and users; add protocol fuzz
  tests before broader exposure.
- **Fairness:** no preemption, MIG, or simultaneous tenancy in the first release.
  The lease guard protects other users by refusing work, not by evicting them.
- **No single-device memory aggregation:** the two 5090 memories will not be
  fused into one larger CUDA allocation. The north star does include standard
  multi-device PyTorch jobs split across local and remote GPUs, first through
  DDP/FSDP workers and later through same-process virtual device routing.
- **No universal near-zero overhead claim:** large transfers and serialized
  fine-grained calls are physically constrained by 10 GbE and RTT.

## Discussion checkpoints

Before running the GPU bake-off, agree on:

1. The exact modded-nanogpt commit and the minimal one-GPU harness adaptation
   after native bring-up shows what the current script requires.
2. Whether the first candidate scope—LUPINE, Cricket, and a bounded GVirtuS
   attempt—is broad enough; SCUDA is already LUPINE, while non-source/commercial
   projects cannot enter a reproducible OSS bake-off.
3. The initial curated PyTorch/TorchBench selectors after their collection is
   enumerated on the pinned software profile.
4. Short windows when gpu-host and then both GPUs may be used exclusively for
   repeated native-vs-remote trials.

Then execute the bake-off gate by gate. Select or modify an implementation only
after the comparison report is complete.
