# rgpu

`rgpu` lets Linux programs use an NVIDIA GPU in another machine as though it
were local. It supports a contained `rgpu run` mode and an experimental
host-wide mode in which ordinary `nvidia-smi` and PyTorch processes see the
remote GPU without a wrapper.

> **Status:** experimental release `0.2.1`, validated on two Ubuntu
> workstations with RTX 5090 GPUs, CUDA 13, PyTorch 2.12, Docker, NVIDIA
> Container Toolkit, and direct 10 GbE. Do not deploy it on an untrusted
> network.

The current Qwen3.8 27B CUDA Graph benchmark reaches 197.43 tok/s remotely
versus 204.26 tok/s natively on the same GPU, with identical deterministic
streamed output. Startup remains bounded by transferring model weights.

## Quick start

The GPU host needs a healthy NVIDIA driver, Docker with NVIDIA Container
Toolkit, SSH access, and a user allowed to run Docker. The client needs Docker,
Python 3.10 or newer, and key-based SSH access to the host.

From a source checkout on the client:

```bash
rgpu discover --candidate USER@GPU_HOST
./rgpu-client/install.sh --host USER@GPU_HOST
rgpu status --host USER@GPU_HOST
rgpu run --host USER@GPU_HOST -- nvidia-smi -L
rgpu run --host USER@GPU_HOST -- python3 your_training_script.py
```

`rgpu discover` reports local host identity, interfaces, local GPUs, Docker
state, and candidate remote GPU hosts without relying on checked-in machine
names. Runtime commands reject a `--host` target that resolves back to the
current machine, which catches accidentally reversed workstation commands
before deployment.

Use any compatible existing workload image without rebuilding it:

```bash
rgpu run --host USER@GPU_HOST \
  --image YOUR_PYTORCH_IMAGE \
  --cublas-rpc \
  -- python3 your_training_script.py
```

To expose local and remote GPUs in one process, add `--include-local`. Local
GPUs are listed first:

```bash
rgpu run --host USER@GPU_HOST --include-local --cublas-rpc -- \
  python3 -c 'import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

## Bidirectional two-host setup

For two workstations that should each be able to use the other's GPU, put the
direct 10 GbE interfaces on the same private point-to-point subnet. Use local
values for the placeholders below:

```text
HOST_A           first workstation
HOST_B           second workstation
USER_A           SSH user on HOST_A
USER_B           SSH user on HOST_B
DIRECT_IF        10 GbE interface, for example eno2
HOST_A_DIRECT_IP static direct-link address for HOST_A
HOST_B_DIRECT_IP static direct-link address for HOST_B
```

Temporary network setup until reboot:

```bash
# On HOST_A
sudo ip link set DIRECT_IF up
sudo ip addr replace HOST_A_DIRECT_IP/24 dev DIRECT_IF
sudo ip link set DIRECT_IF mtu 9000

# On HOST_B
sudo ip link set DIRECT_IF up
sudo ip addr replace HOST_B_DIRECT_IP/24 dev DIRECT_IF
sudo ip link set DIRECT_IF mtu 9000
```

Persistent NetworkManager setup:

```bash
# On HOST_A
sudo nmcli con add type ethernet ifname DIRECT_IF con-name rgpu-direct \
  ipv4.method manual ipv4.addresses HOST_A_DIRECT_IP/24 \
  ipv6.method ignore 802-3-ethernet.mtu 9000 autoconnect yes
sudo nmcli con up rgpu-direct

# On HOST_B
sudo nmcli con add type ethernet ifname DIRECT_IF con-name rgpu-direct \
  ipv4.method manual ipv4.addresses HOST_B_DIRECT_IP/24 \
  ipv6.method ignore 802-3-ethernet.mtu 9000 autoconnect yes
sudo nmcli con up rgpu-direct
```

Verify that each direction routes directly over the 10 GbE interface and not
through a gateway:

```bash
# On HOST_A
ip route get HOST_B_DIRECT_IP
ssh USER_B@HOST_B_DIRECT_IP 'hostname; nvidia-smi -L'

# On HOST_B
ip route get HOST_A_DIRECT_IP
ssh USER_A@HOST_A_DIRECT_IP 'hostname; nvidia-smi -L'
```

In contained mode, each side can append the other GPU after its local GPU:

```bash
# On HOST_A: local GPU is cuda:0, HOST_B's GPU is cuda:1.
rgpu run --host USER_B@HOST_B_DIRECT_IP --include-local --cublas-rpc -- \
  python3 -c 'import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'

# On HOST_B: local GPU is cuda:0, HOST_A's GPU is cuda:1.
rgpu run --host USER_A@HOST_A_DIRECT_IP --include-local --cublas-rpc -- \
  python3 -c 'import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

Validate one direction at a time first. `rgpu` uses exclusive remote GPU
leases and refuses to interfere with active compute work.

## Host-wide attachment

This changes only rgpu-owned loader configuration and libraries; it does not
replace the NVIDIA driver, kernel modules, packages, device nodes, or the
`nvidia-smi` executable. Attachment is journaled and rollback-safe, but it is
still experimental. Stop local CUDA workloads before attaching.

```bash
sudo "$(command -v rgpu)" attach --host USER@GPU_HOST
# Open a new terminal (or start a new login shell) before launching Python.
nvidia-smi
python3 -c 'import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
sudo "$(command -v rgpu)" detach
```

To make ordinary `nvidia-smi` on each workstation show both GPUs, attach each
machine to the other machine's direct-link SSH target:

```bash
# On HOST_A
sudo "$(command -v rgpu)" attach --host USER_B@HOST_B_DIRECT_IP
nvidia-smi

# On HOST_B
sudo "$(command -v rgpu)" attach --host USER_A@HOST_A_DIRECT_IP
nvidia-smi
```

Each host-wide view is local-first: the physical GPU in that workstation is
listed before the attached remote GPU. Validate one side first, then the other.
If the second side refuses because a GPU is already leased or busy, detach the
first side and use contained `rgpu run --include-local` for bidirectional
testing until the lease policy is relaxed.

`nvidia-smi` is immediately transparent. PyTorch must be launched from a new
login shell so it inherits rgpu's narrowly scoped CUDA-library path; attach
does not inject a global `LD_PRELOAD` or alter an already-running shell.

If an interrupted transaction cannot be resumed normally:

```bash
sudo "$(command -v rgpu-rescue)"
```

For architecture, artifact layout, security boundaries, compatibility,
performance, upgrades, and troubleshooting, read the
[deployment and engineering guide](docs/guide.md). Exact benchmark tables are
in the [timing summary](docs/timing-summary.md).

## Source layout

| Directory | Contents |
|---|---|
| `lupine/` | Modified CUDA Driver API and NVML remoting engine |
| `rgpu-client/` | Client package, CLI, attachment layer, and injected libraries |
| `rgpu-host/` | Versioned GPU-server image packaging and host check |
| `dev/` | Tests, benchmarks, reproducibility tools, and research artifacts |

Build the release artifacts with `dev/tools/build_release.py`. Run the fast
regression suite with `pytest -q`.
