# Host-wide attachment

## Product contract

The final interface is:

```bash
sudo rgpu attach --host user@192.0.2.10
nvidia-smi
python3 train.py
sudo rgpu detach
```

When the raw service is behind a TLS-terminating proxy, attachment can map the
same SSH lease host to a separately addressed, server-authenticated data path:

```bash
sudo rgpu attach \
  --host user@192.0.2.10 \
  --endpoint https://gpu.example
```

The endpoint is persisted in `/etc/rgpu/endpoints`; ordinary CUDA and NVML
clients verify its certificate and hostname using the system trust store.
Proxy deployment and client authorization are still explicit production
gates.

Newly started CUDA and NVML applications must see the local GPU followed by the
attached remote GPUs without a wrapper command, container boundary,
`LD_LIBRARY_PATH`, or `LD_PRELOAD`. Detach must restore the exact pre-attach
loader state and stop every remote lease.

## Current implementation

Round 8 adds file-based endpoint discovery to both CUDA and NVML shims.
`LUPINE_SERVER` remains an explicit override, but otherwise both libraries read
the first valid line from `/etc/rgpu/endpoints`. The host-wide installer places
the shim in `/usr/local/lib/rgpu`, adds an isolated loader configuration file,
runs `ldconfig`, and records fingerprints and backups in
`/var/lib/rgpu/state.json`.

Attach and detach are transactional for an explicitly supplied sandbox root:

```bash
rgpu attach --host user@192.0.2.10 --root /path/to/disposable/root
rgpu detach --root /path/to/disposable/root
```

Passing `/` enables the live transaction only when invoked as root. Live attach
refuses an already-attached root or active local compute processes, verifies the
union of local and remote UUIDs after refreshing the loader, and rolls back both
files and newly started leases on failure. Root network operations run as the
original `SUDO_USER`, preserving that user's SSH keys.

Detach removes only files whose fingerprints still match the attach manifest.
If an administrator or package manager changes a managed file, detach fails
closed rather than overwriting that change. Pre-existing files are backed up
and restored atomically.

Detach now writes a durable `detaching` phase before changing any managed file.
If interruption or `ldconfig` failure occurs after some originals have already
been restored, the same detach can resume from a desired file, restored backup,
or safely absent new file. A failure-injected loader-refresh regression proves
that the second attempt completes and removes the journal. Every atomic managed
replacement and removal now `fsync`s its parent directory, including final
journal deletion, so the filesystem ordering survives more than process-level
interruption.

The separately installed `rgpu-rescue` entry point imports no launcher or
network code, audits `/proc/self/maps`, and refuses recovery if CUDA or NVML is
already mapped. It restores the local transaction without contacting a remote
host and reports any lease records for later cleanup:

```bash
sudo "$(command -v rgpu-rescue)"
```

## Passing sandbox gates

The disposable root filesystem contains no per-process CUDA configuration and
has no wrapper entrypoint. The following pass with the system loader selecting
the installed rgpu libraries:

- bare `nvidia-smi -L` reports the remote RTX 5090;
- bare PyTorch reports CUDA available and executes a tensor operation;
- the full 8/8 compile-and-CUDA-Graph smoke suite;
- single-rank NCCL all-reduce and all-gather;
- DTensor redistribution and distributed checkpoint save/load;
- composable FSDP2 training;
- spawned-process CUDA IPC; and
- mixed enumeration with the physical GPU as `cuda:0` and the remote GPU as
  `cuda:1`, followed by explicit computation on only `cuda:1`.

The NVIDIA container hook initially regenerated its loader cache after image
construction and selected the physical libraries. Running the same `ldconfig`
refresh performed by attach restored deterministic rgpu precedence. This was
found entirely inside the disposable image; the workstation loader cache was
never changed.

The complete sandbox attach/detach flow also passes when the remote lease has
already disappeared: local files are restored and cleanup remains idempotent.

## Remaining production gates

1. Snapshot and compare the host's NVIDIA libraries, symlinks, modules, device
   nodes, loader cache, and `nvidia-smi` identity before and after every test.
2. Revalidate the immutable physical-stack fingerprint after the current live
   attachment is detached.
3. Re-run the passing interruption matrix on a live attachment. Sandbox tests
   inject after all seven install replacements, all seven detach removals, and
   post-restore loader refresh; every retry restores the original state.
4. Complete crash/expiry recovery around the reboot-persistent named lease.
5. Validate package upgrades and NVIDIA driver upgrades while detached and
   while attachment is requested.
6. Pass the local-plus-remote DDP gate when the local GPU is available. Until
   then, only enumeration and explicit remote-ordinal computation are allowed.
7. Validate the passing out-of-band `rgpu-rescue` flow once against a live
   attachment; its sandbox transaction and no-CUDA/NVML mapping checks pass.
8. After a deliberate detach, reattach with the current round-30 shim/server
   and verify local and remote process rows. The currently attached legacy shim
   predates R610 `_v3` process exports and the legacy server predates host-PID
   visibility; aggregate memory and utilization remain correct.

Host-wide leases are deliberately marked persistent and are excluded from
`rgpu gc`. Per-command leases are marked ephemeral, do not restart after a
remote reboot, and can be reclaimed only after their minimum age has elapsed
and the service has no established client connection. Automated persistent
lease expiry still remains open because an attached host may legitimately be
idle for long periods.

Live attachment adds only the transaction-journaled rgpu libraries, endpoint
file, and loader configuration. It does not replace an NVIDIA library, kernel
module, device node, driver package, or `/usr/bin/nvidia-smi`.

`dev/tools/host_nvidia_snapshot.py` records the physical GPU identity, NVIDIA
kernel versions, canonical driver-library targets and SHA-256 hashes,
`nvidia-smi` binary hash, and every NVIDIA character device's identity and
permissions without loading the rgpu shim. The initial and immediate repeat
snapshots match at immutable fingerprint
`0f44b820510ea845efe8d12af7e7629df2369dc5538234dc865cc97c18b5c9bd`;
the raw baseline is retained at
`dev/results/raw/host-nvidia-safety-baseline.json`.

The corrected out-of-band snapshot taken while host-wide attachment is active
has the same immutable fingerprint. It forces the canonical vendor NVML rather
than trusting loader precedence; its raw evidence is
`dev/results/raw/host-nvidia-safety-attached-20260820.json`. A final identical
post-detach snapshot is still required.
