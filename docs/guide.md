# rgpu deployment and engineering guide

This document explains release `0.2.1`: how to deploy it, how the system is
constructed, which semantics are currently supported, where performance comes
from, and how to recover safely. For a first run, start with the concise
[README](../README.md).

`rgpu` makes one or more leased remote NVIDIA GPUs visible to an unmodified
PyTorch process without installing a kernel module or changing the local
NVIDIA driver. The application stays on the client workstation; CUDA Driver
API and NVML operations are routed over the direct network link.

The original per-command launcher is stable. Host-wide attach now lets
ordinary `nvidia-smi` and ordinary Python processes discover the remote GPU
through the system loader with no wrapper, container, or CUDA environment
variables. Attach is transaction-journaled and fail-closed; it does not replace
the NVIDIA kernel driver or kernel modules.

The original `rgpu run` launcher is intentionally container-scoped. The
host-wide mode installs separate transaction-journaled shim files and loader
configuration, but neither mode replaces the host's NVIDIA libraries, kernel
modules, device nodes, driver packages, or `nvidia-smi` binary. In strict run
mode the client container receives no local NVIDIA devices, so an unsupported
path cannot silently fall back to the local GPU.

## Repository layout

The deployable product is split into three explicit components, with all
development-only material isolated in a fourth directory:

| Directory | Purpose | Deployed where |
|---|---|---|
| `lupine/` | Canonical modified CUDA Driver API and NVML remoting engine | Built into both client shim and GPU-host server artifacts |
| `rgpu-client/` | `rgpu`/`rgpu-rescue`, attachment logic, client interposers, and injection images | Machine requesting remote GPUs |
| `rgpu-host/` | GPU server image packaging | Machine providing GPUs |
| `dev/` | Tests, benchmarks, research containers, historical patches, manifests, and result artifacts | Development only |

An end-user checkout does not need `dev/` once release images and client
packages are published. The public upstream clones under `dev/external/` are
push-disabled research inputs and are not part of either deployment.

## Architecture

The Python CLI is the control plane. It checks host capacity over SSH, verifies
that both ends have the identical server image, leases a TCP port, starts one
isolated server container per session, injects the client libraries, and
always attempts cleanup when the child exits. SSH carries lifecycle commands;
CUDA traffic uses the direct data connection.

```text
requesting machine                              GPU-providing machine

application / PyTorch
        |
CUDA, CUDA Runtime, NVML, math-library shims
        |
virtual handles + batched RPC  ===== 10 GbE ====>  LUPINE server
        |                                           |
local GPU (optional)                         native CUDA libraries
                                                    |
                                              physical NVIDIA GPU
```

The LUPINE-derived layer transports the CUDA Driver API and NVML and maps
remote handles, addresses, streams, and events into client-visible identities.
The rgpu interposers add CUDA Runtime compatibility and explicit routes for
cuBLAS/cuBLASLt, cuSOLVER, cuFFT, and NCCL. The server executes opaque CUDA
libraries beside the owning CUDA context, which avoids sending embedded device
pointers into a local vendor library.

There are two exposure modes:

- `rgpu run` bind-injects a content-addressed library bundle into one workload
  container. Strict mode does not pass local NVIDIA devices into that
  container; `--include-local` deliberately exposes both sets of devices.
- `rgpu attach` installs only rgpu-owned files under `/usr/local/lib/rgpu`, an
  endpoint file, an isolated loader configuration, and
  `/etc/profile.d/rgpu.sh`. Newly launched native processes then resolve the
  interposed CUDA/NVML libraries normally. Python defers loading the RPC math
  libraries until the first `torch` import, so ordinary non-PyTorch Python
  processes do not initialize CUDA. A durable transaction journal makes detach
  and recovery resumable.

This is userspace virtualization, not a virtual PCI device or replacement
kernel driver. It makes NVML and CUDA consumers such as `nvidia-smi` and
PyTorch see the device, but software that talks directly to PCI sysfs or a
physical `/dev/nvidia*` node can distinguish it from hardware installed in the
client.

## Release artifacts

Release `0.2.1` consists of artifacts that share one version and wire-protocol
contract:

| Artifact | Role |
|---|---|
| `remote_gpu-0.2.1-py3-none-any.whl` | `rgpu`, `rgpu-rescue`, orchestration, and safe attachment logic |
| `remote-gpu-client:0.2.1` | Self-contained injectable CUDA/NVML shim plus math-library and NCCL interposers |
| `remote-gpu-host:0.2.1` | GPU-side LUPINE server and compatible CUDA userspace |
| `remote-gpu-workload:0.2.1` | Validated default PyTorch workload; optional when supplying another image |

`python3 dev/tools/build_release.py` rebuilds those images, builds the wheel,
and writes a manifest and SHA-256 checksums under `dev/releases/0.2.1`.
`--export-images` also emits compressed OCI archives for an offline/private
release. The CLI refuses a client/server image mismatch and protocol revision
mismatch before dispatching GPU work.

## Install and run

Validated client prerequisites are Linux, Python 3.10 or newer, Docker, and
key-based SSH access to the GPU host. The host needs a healthy NVIDIA driver,
Docker, NVIDIA Container Toolkit, and an SSH user permitted to run Docker.
The raw data port must be reachable only over a trusted network. On the host,
`rgpu-host/check.sh` verifies these prerequisites and the loaded server image
without starting a GPU workload.

From this repository:

```bash
rgpu discover --candidate user@192.0.2.10
./rgpu-client/install.sh --host user@192.0.2.10
rgpu run --host user@192.0.2.10 -- nvidia-smi -L
rgpu run --host user@192.0.2.10 -- python3 train.py
```

`rgpu discover` identifies the local host, visible interfaces, local GPUs, and
Docker state, then probes explicit candidate SSH targets. Deployment and run
commands reject a target that resolves to the local machine, so checked-in
examples cannot accidentally reverse the client/server direction. The
installer builds the reproducible client, portable injection shim, server, and
default PyTorch image; installs the Python launcher; and copies the exact
server image to the selected host. A subsequent run verifies the remote image
ID before opening a session. The client and server also negotiate an explicit
wire-protocol revision and reject mismatches before dispatching a CUDA call.
Use `--deploy` after rebuilding an image.

To install a wheel produced by the release builder while reusing already
loaded images:

```bash
./rgpu-client/install.sh \
  --host user@192.0.2.10 \
  --skip-build \
  --wheel dev/releases/0.2.1/wheels/remote_gpu-0.2.1-py3-none-any.whl
```

Upgrade while detached: run `rgpu detach` if host-wide mode is active, install
the new wheel and matching images, run `rgpu deploy`, then attach again. The
image identity and protocol checks prevent a partial client/server upgrade
from being used accidentally. Reinstalling the same version is idempotent.

After a live attach, open a new terminal or start a new login shell before
launching PyTorch. `nvidia-smi` sees the remote device immediately through the
loader cache, but an existing shell cannot retroactively inherit the profile
environment. rgpu deliberately does not write `/etc/ld.so.preload`: that would
inject its libraries into every host process and create an unnecessary safety
boundary.

Installation also provides `rgpu-rescue`, a local-only recovery path for an
interrupted host-wide transaction. It does not contact remote hosts and refuses
to run if its own process maps CUDA or NVML:

```bash
sudo "$(command -v rgpu-rescue)"
```

Normal `rgpu run` servers are labeled as ephemeral and no longer restart after
a remote reboot. If the local launcher is killed uncatchably, a conservative
collector can reclaim only a labeled ephemeral lease that is old enough and
has no established client connection:

```bash
rgpu gc --host user@192.0.2.10 --min-age 3600
```

Persistent host-wide attachments are never eligible for this collector.

An existing compatible PyTorch/CUDA image does not need to be rebuilt. `rgpu`
extracts the validated userspace shim to a content-addressed local cache and
bind-injects it at process launch:

```bash
rgpu run \
  --host user@192.0.2.10 \
  --image YOUR_EXISTING_PYTORCH_IMAGE \
  -- python3 train.py
```

The portable shim is built on Ubuntu 22.04 (glibc 2.35) and is validated in an
otherwise untouched PyTorch 2.12/CUDA 13 image. Older userspaces require a
separately built compatible shim.

To append remote devices after local devices, opt in explicitly:

```bash
rgpu run \
  --host user@192.0.2.10 \
  --include-local \
  --image YOUR_EXISTING_PYTORCH_IMAGE \
  -- python3 train.py
```

In the validated two-workstation setup this exposes the local RTX 5090 as
`cuda:0` and the remote RTX 5090 as `cuda:1`. Multiple remote hosts can be
specified with repeated `--host` options or a comma-separated list. Actual
multi-host execution remains an acceptance gate because only one remote GPU
host is currently available.

For a server placed behind a TLS-terminating proxy, keep `--host` as its SSH
deployment/lease address and provide the corresponding HTTPS data endpoint:

```bash
rgpu run \
  --host user@192.0.2.10 \
  --endpoint https://gpu.example \
  -- python3 train.py
```

`attach` accepts the same option. CUDA and NVML traffic then verifies the
proxy certificate and hostname against the system trust store; invalid or
untrusted certificates fail closed. Endpoint count and order must exactly
match `--host`. Deploying and validating the proxy and its client-access
policy remain production gates, so the raw protocol should still be confined
to the trusted point-to-point network.

Same-process PyTorch linear algebra is available behind an explicit
compatibility flag while its ABI surface is expanded:

```bash
rgpu run \
  --host user@192.0.2.10 \
  --include-local \
  --cublas-rpc \
  --cublas-async \
  -- python3 train.py
```

This routes remote cuBLAS/cuBLASLt, cuSOLVER, and cuFFT operations to the
GPU-owning server while local operations continue through native libraries.
The same `--cublas-rpc` flag is valid in strict remote-only mode without
`--include-local`; in that mode the client receives no local GPU device and
the remote GPU remains `cuda:0`.
The current gate passes local-first forward/backward for float32, float16, and
bfloat16, strided-batched and complex GEMM, FFT/STFT, LU/QR/Cholesky/eigenvalue
families, transformer SDPA, all 16 bounded model families, and the complete
634-name first-sample OpInfo differential in one unmodified PyTorch process.
`--cublas-async` defers validated compute calls on their owning stream;
malformed dimensions retain immediate cuBLAS errors. Unimplemented linked APIs
remain fail-closed expansion work, so these flags are not yet the default.

For transparent distributed training, run ordinary `torchrun` inside an
include-local session:

```bash
rgpu run \
  --host user@192.0.2.10 \
  --include-local \
  -- torchrun --standalone --nproc-per-node=2 train.py
```

Both Python ranks stay on the initiating workstation. Rank 0 uses local
`cuda:0`; rank 1 uses remote `cuda:1`. Remote NCCL control executes beside the
remote CUDA context while tensor traffic uses NCCL's native Ethernet data
plane. `rgpu native-python` remains available as a differential control.

## Current validated surface

- Strict remote-only and mixed local-plus-remote `nvidia-smi`/PyTorch device
  enumeration, including virtual PCI identities and reverse lookup.
- Allocation, pageable and pinned transfers, kernel-syscall writes into pinned
  memory, DataLoader pinning, autograd, cuDNN, streams/events, compilation, and
  CUDA Graph capture/replay.
- 628 successful upstream PyTorch OpInfo samples across 634 unique names on
  both local `cuda:0` and remote `cuda:1` in the same process. The other six
  produce identical native and remote exceptions; there are no
  remoting-specific failures in this sweep.
- Single-rank NCCL all-reduce/all-gather, mixed two-rank DDP, the nine-family
  NCCL collective matrix, DTensor redistribution, FSDP2
  training, distributed checkpoint save/load, spawned-process CUDA IPC, 16 Hugging Face/Diffusers
  training families, nanoGPT,
  TorchBench DCGAN/BERT, torchvision, nanochat, LitGPT, TorchTitan Qwen3, and
  the bounded subsystem matrix.
- Invalid-device and OOM recovery, server link-loss detection, signal
  cancellation, deterministic lease cleanup, conservative expired-lease
  reclamation, and stale-image rejection.
- Fail-fast client/server wire-protocol negotiation; a mismatched client is
  rejected with HTTP 426 before CUDA RPC dispatch.
- Transparent asynchronous CUDA Graph launch with
  `LUPINE_SYNC_GRAPH_LAUNCH=1` as the immediate-result compatibility fallback.
- Dynamic pinned-host graph inputs remain live across replay through stable
  server mirrors and dirty-range propagation. Separate captures on one stream
  and successful graph-exec updates retain their own copy resources.
- Native two-host NCCL and five-step Transformer DDP through `rgpu
  native-python`; 64 MiB all-reduce reaches 9.83 Gbit/s over the 10GbE link.
- Transparent local-plus-remote NCCL reduce, reduce-scatter, all-gather,
  all-to-all, gather, scatter, point-to-point, all-reduce, broadcast, and
  barrier through ordinary `torchrun`.

## Performance profile

The latest optimized examples include whole-training-step replay at 1.469 ms
remote versus 1.408 ms native for Llama, 3.022 versus 2.973 ms for LitGPT,
and 6.539 versus 6.348 ms for TorchTitan/FlexAttention. See
[the timing summary](timing-summary.md) for scopes and caveats.

For Qwen3.8 27B serving, NInfer's default CUDA Graph path measures 197.43 tok/s
remote versus 204.26 tok/s native, with identical deterministic streamed
output. Model initialization remains transfer-bound because 19.73 GiB of
weights cross the 10 GbE link.

The experimental local-plus-remote Qwen3.8 path can instead reserve a full 1M
INT8 KV context across two 32 GiB RTX 5090s. Build a disposable patched NInfer
image and launch it without modifying `ali`:

```bash
dev/tools/build_ninfer_qwen38_pipeline.sh \
  --source /home/USER/Documents/ali/ninfer
dev/tools/run_ali_qwen38_pipeline.sh \
  --ali /home/USER/Documents/ali \
  --host USER@GPU_HOST
```

The launcher defaults to a balanced 32/32 layer split, 1,048,576-token cache,
the measured 1,536-token prefill chunk, pipelined prefill, and mixed-device CUDA
Graph decode. During non-terminal prompt chunks, the local GPU begins the next
chunk while the remote GPU completes the previous one; only the terminal
residual crosses back for final normalization and sampling.
The remote half of each decode profile is captured and uploaded during startup;
the local half and the host-to-host device boundary remain eager. Small control
tensors are packed into one asynchronous transfer. Pass `--no-cuda-graph` to
retain the exact eager compatibility path.

Compiled model-plus-optimizer execution is the strongest current performance
path for modern training code: TorchTitan Qwen3 measures 4.606 ms remote versus
4.622 ms native GPU time, and LitGPT measures 2.179 ms versus 2.211 ms. Eager
CUDA API virtualization remains the transparent compatibility fallback.

## Important boundaries

- The raw direct protocol has no authentication or encryption policy. The
  optional HTTPS path provides encryption and server identity verification,
  but its TLS proxy and client-access policy are not yet installed or validated
  by `rgpu` itself.
- Bulk host/device movement is bounded by the 10 GbE link. Good performance
  depends on keeping tensors resident and using bulk asynchronous transfers.
- The current PyTorch-linked-library audit has 161 callable symbols without a
  remote route: 106 cuBLAS, 48 cuSOLVER, and 7 NCCL symbols. With
  `--cublas-rpc`, generated guard exports make those calls return a
  library-specific not-supported status and name the exact missing symbol.
  `--allow-unsupported-library-fallback` disables this guard for debugging,
  but is unsafe when an API contains embedded device pointers.
- `torch.compile` and CUDA Graph acceleration are honored transparently when
  applications use them. Workload-specific semantic rewrites such as DETR
  matcher batching or Mixtral's batched-expert implementation are opt-in test
  profiles, not silently applied to arbitrary programs.
- CUDA-transparent mixed NCCL now uses a remote-native data plane and passes
  1 MiB/16 MiB payloads plus short DDP. The accepted model is currently one
  assigned GPU per process, matching standard PyTorch launchers. Experimental
  same-process cuBLAS/cuBLASLt alternation now passes the three primary training
  dtypes, strided-batched GEMM, and broad models behind `--cublas-rpc`; the
  asynchronous lane is opt-in with `--cublas-async`. The remaining CUDA-library
  ABI surface and multi-remote-host validation remain open gates.
- The latest modded-nanogpt kernel imports pass. Its bounded full training run
  reaches Inductor on both native and remote RTX 5090, where both hit the same
  generated Triton shared-memory configuration beyond the hardware limit;
  that is not a remoting-specific failure.
- CUDA execution inside `torch.profiler` works, but remote CUDA activity events
  are not yet collected because CUPTI is a separate driver-side profiling API.
  Native collection passes; remote CPU profiling remains available.

The exact release contract and remaining gates are in
[the north-star specification](north-star.md). Current evidence is in
[the compatibility findings](compatibility.md), with raw local artifacts
under `dev/results/raw/`.

The replacement mixed-collective architecture and its bounded acceptance
ladder are in [remote-native collectives](remote-native-collectives.md).

The host-wide design, completed sandbox gates, and live-install safety boundary
are documented in [host-wide attachment](hostwide-attachment.md).

## Safety and provenance

Public repositories are cloned only into the ignored `dev/external/` directory.
`dev/tools/upstreams.py` disables each clone's push URL and installs a rejecting
pre-push hook. Do not create public forks or push changes upstream.

The optimized runtime is reproduced from LUPINE v1.0.0 using the private patch
series in `dev/patches/`; the canonical integrated source now lives in `lupine/`
and is built by `dev/tools/build_images.sh`. No build, install,
run, or test step modifies the host NVIDIA software stack.
