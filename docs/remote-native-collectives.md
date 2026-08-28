# Remote-native collective data plane

## Decision

Do not optimize generic polling of NCCL's CUDA host mappings into the mixed
training data plane. Keep protocol-3 coherence as a compatibility experiment
and regression, but execute collectives natively on the workstation that owns
each GPU.

The native two-host control reaches 9.83 Gbit/s on the 10GbE link. The mirrored
path needs multi-megabyte coherence windows for a tiny collective and does not
complete the 1 MiB matrix inside 60 seconds. This difference is architectural:
NCCL's GPU kernel and CPU proxy assume cache-coherent mapped memory on one host.

## Current status — 2026-08-20

Phases A–C now pass for the pinned PyTorch 2.12.0/CUDA 13.0/NCCL 2.29.7
profile. A replacement `libnccl.so.2` sends remote-rank control calls to native
NCCL in the GPU-owning LUPINE server process; local ranks continue through the
renamed native library. Ordinary unmodified
`torchrun --standalone --nproc-per-node=2` now sees local `cuda:0` and remote
`cuda:1` and completes:

- five TinyTransformer DDP steps with exact final loss sum 5.375;
- reduce, reduce-scatter, all-gather, all-to-all, gather, scatter, batched
  send/receive, averaged all-reduce, and barrier;
- three consecutive clean DDP sessions without communicator-cleanup errors;
- bounded server-loss detection followed by a successful fresh session.

Matched large-payload all-reduce is close to native: the 16 MiB operation took
16.390 ms on the local rank through the mixed transparent path versus
15.144 ms natively. The five-step DDP timed region is 0.663–0.695 s remotely
versus 0.372 s natively, so control/compute overhead remains a performance
target even though correctness is accepted.

The remote server must bind NCCL to the interface carrying the client route.
Without this, NCCL can advertise an unrelated VPN address and all-to-all will
stall. `rgpu` now detects the interface from `SSH_CONNECTION` plus `ip route`
and injects `NCCL_SOCKET_IFNAME` into the server lease.

## Target flow

```text
unmodified PyTorch rank on client-host
        |
        | NCCL ABI interposer (only for a virtualized GPU route)
        v
LUPINE session RPC ====================================+
                                                       |
                                                       v
                                      native NCCL in the GPU-owning
                                      session on gpu-host
                                                       |
                                      native CUDA stream and pointers
                                                       |
                                                       v
                                               remote RTX 5090
```

The CUDA allocation, stream, event, and module handles already returned by
LUPINE are server-side handles represented as opaque client values. A
server-side NCCL call can therefore consume the same buffer and stream values
without copying tensor payloads through the client. NCCL creates its socket
proxy and mapped control memory on `gpu-host`, beside the GPU, and communicates
with the native rank on `client-host` over the direct Ethernet link.

## Why this preserves the product contract

- The application process, Python control flow, files, and data loading remain
  on `client-host`.
- PyTorch source code is unchanged. Installation supplies a version-matched
  NCCL interposer alongside the existing CUDA/NVML shim.
- Local ranks continue to bind directly to the workload's native NCCL.
- Remote ranks forward only the NCCL control ABI; tensor data remains resident
  in remote VRAM and follows native NCCL's data plane.
- `nvidia-smi`, CUDA enumeration, and non-collective PyTorch operations retain
  the existing transparent behavior.

This differs from `rgpu native-python`: that fallback moves the whole remote
rank process to `gpu-host`. The target flow moves only NCCL execution while
keeping the rank process on the initiating workstation.

## Prototype phases

### Phase A — ABI interception proof

1. Pin the exact NCCL library shipped by the workload image and record its
   exported ABI.
2. Inject a replacement `libnccl.so.2` for the remote rank. It forwards local
   queries and error strings to the original library and logs communicator and
   collective calls.
3. Verify that a native two-rank smoke is unchanged when every call is simply
   forwarded. This is the loader/interposition acceptance gate.
4. Retain probes for `ncclCommInitRankScalable`, config-based initialization,
   group start/end, asynchronous error polling, finalize, destroy, and abort;
   current PyTorch declares a materially broader symbol surface than a minimal
   all-reduce sample exercises.

Initial loader probing found that the pinned NVIDIA NCCL 2.29.7 wheel is built
with `PROFAPI`: public `nccl*` functions are weak and strong `pnccl*` aliases
name the implementation. A plain preload wrapper reliably observes bootstrap
but does not safely own the collective path; wrapping both alias families
destabilizes communicator startup. Phase A must therefore use either a true
replacement `libnccl.so.2` that explicitly delegates to a renamed original, or
the supported profiling-ABI mechanism with exact symbol/version handling.
Passing the forward-only two-host control is mandatory before adding RPCs.

### Phase B — server-side one-channel smoke

1. Add protocol-versioned RPCs for communicator initialization, group
   boundaries, all-reduce, all-gather, broadcast, async error, finalize,
   destroy, and abort.
2. Load the exact same NCCL build in the disposable LUPINE server image.
3. Reconstruct `ncclConfig_t` from explicit scalar wire fields; never transmit
   raw client pointers such as `netName` or `commName`.
4. Call native NCCL on the server connection lane that owns the CUDA context.
   Preserve ordering with the existing per-client-thread RPC lane.
5. Run the current mixed smoke with mapped-memory coherence disabled. Passing
   both values and teardown proves the architecture.

### Phase C — PyTorch distributed surface

Add all-reduce, all-gather, reduce-scatter, broadcast, reduce, send/receive,
all-to-all, communicator split, custom reduction operators, registration, and
graph-safe group semantics. Unsupported symbols must return a deterministic
NCCL error before launching work, never fall through to client-side NCCL for a
remote communicator.

### Phase D — lifecycle and scale

- Cache communicator and registration handles by server session and clean them
  on normal exit, abort, client death, and link loss.
- Match PyTorch's asynchronous stream semantics and watchdog polling.
- Support multiple remote servers and mixed communicator membership.
- Negotiate NCCL ABI/version with the server before communicator creation.
- Add authentication and admission control before leaving the trusted link.

Current Phase D checkpoint: normal teardown, server loss, transport error
surfacing, and subsequent clean-session recovery pass. Peer-rank abort during
an active collective, multiple remote servers, authenticated transport, and
version negotiation beyond the current protocol/NCCL pin remain open.

## Acceptance tests

Every invocation remains below five minutes.

| Gate | Required result |
|---|---|
| Forward-only ABI interposer | Native two-host smoke and DDP match the current control. |
| Remote-server scalar smoke | Correct all-reduce/all-gather/broadcast; clean teardown; coherence thread disabled. |
| Payload ladder | 1 MiB, 16 MiB, and 64 MiB exact results. |
| Throughput | Approach the native 8.44, 9.73, and 9.83 Gbit/s controls respectively. |
| DDP | Five-step Transformer exact loss/checksum differential. |
| Modern training | Bounded nanoGPT and one compiled Transformer workload. |
| Failure | Peer abort, server death, link loss, timeout, and subsequent clean session. |
| Compatibility | Single-remote regression matrix and 82 curated OpInfos remain passing. |

The first performance target is at least 80 percent of the matched native
cross-host throughput for 16 MiB and 64 MiB collectives. Optimization begins
only after exact result and lifecycle gates pass.

## Risks

- NCCL is not a permanently stable internal ABI. Client and server must use the
  exact workload NCCL build or reject the session.
- NCCL's `PROFAPI` weak/strong alias model makes generic ELF preloading
  insufficient. The packaged interposer must control library resolution and
  cover both public and profiling symbol families without recursion.
- Recent PyTorch/NCCL versions expose scalable initialization, device
  communicators, registration, window registration, and experimental APIs.
  The interposer must distinguish required runtime calls from merely linked
  symbols and fail closed for unsupported use.
- NCCL calls must run with the correct server CUDA context and preserve the
  application stream's order. A helper process would require CUDA IPC events;
  in-process server execution avoids that complexity.
- CUDA Graph capture and custom reduction operators require explicit wire
  contracts and cannot be treated as raw pointer serialization.

Primary implementation references are NVIDIA NCCL's `src/collectives.cc`,
`src/proxy.cc`, and public `nccl.h`, plus PyTorch's `ProcessGroupNCCL.cpp` and
`NCCLUtils.hpp` at the pinned workload version.
