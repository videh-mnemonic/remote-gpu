#!/usr/bin/env python3
"""Run commands against one or more remote GPUs without host driver changes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import pwd
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Sequence
from urllib.parse import urlsplit

from remote_gpu.hostwide import (
    HostwideError,
    install as hostwide_install,
    uninstall as hostwide_uninstall,
)
from remote_gpu.version import __version__


DEFAULT_PORT = 14833
DEFAULT_CLIENT_IMAGE = f"remote-gpu-workload:{__version__}"
DEFAULT_SERVER_IMAGE = f"remote-gpu-host:{__version__}"
DEFAULT_SHIM_IMAGE = f"remote-gpu-client:{__version__}"


class RgpuError(RuntimeError):
    """A user-facing launcher failure."""


class LaunchInterrupted(BaseException):
    """Break a blocking child wait while preserving the received signal."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


@dataclass(frozen=True)
class Host:
    ssh_target: str
    address: str
    port: int = DEFAULT_PORT
    transport_endpoint: str | None = None

    @property
    def endpoint(self) -> str:
        return self.transport_endpoint or f"{self.address}:{self.port}"


def parse_host(value: str) -> Host:
    """Parse [ssh-user@]address[:lupine-port]."""
    value = value.strip()
    if not value or any(char.isspace() for char in value):
        raise argparse.ArgumentTypeError("host must not be empty or contain whitespace")
    ssh_target = value
    address_part = value.rsplit("@", 1)[-1]
    address = address_part
    port = DEFAULT_PORT
    if address_part.count(":") == 1:
        candidate, raw_port = address_part.rsplit(":", 1)
        if raw_port.isdigit():
            address = candidate
            port = int(raw_port)
            ssh_target = value[: -len(raw_port) - 1]
    if not address or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("invalid host address or port")
    return Host(ssh_target=ssh_target, address=address, port=port)


def expand_hosts(values: Sequence[str]) -> list[Host]:
    hosts: list[Host] = []
    for value in values:
        for item in value.split(","):
            hosts.append(parse_host(item))
    if not hosts:
        raise RgpuError("at least one --host is required")
    endpoints = [host.endpoint for host in hosts]
    if len(endpoints) != len(set(endpoints)):
        raise RgpuError("duplicate remote GPU endpoint")
    return hosts


def apply_transport_endpoints(
    hosts: Sequence[Host], values: Sequence[str] | None
) -> list[Host]:
    """Map authenticated client endpoints to SSH/deployment hosts by order."""
    if not values:
        return list(hosts)
    endpoints = [item.strip() for value in values for item in value.split(",")]
    if len(endpoints) != len(hosts):
        raise RgpuError("--endpoint count must exactly match --host count")
    configured = []
    for host, endpoint in zip(hosts, endpoints, strict=True):
        parsed = urlsplit(endpoint)
        try:
            port = parsed.port
        except ValueError as exc:
            raise RgpuError(f"invalid authenticated endpoint: {endpoint!r}") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or (port is not None and not 1 <= port <= 65535)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise RgpuError(
                "authenticated endpoints must be https://host[:port] with no "
                f"credentials, path, query, or fragment: {endpoint!r}"
            )
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = f"[{hostname}]"
        canonical = f"https://{hostname}:{443 if port is None else port}"
        configured.append(replace(host, transport_endpoint=canonical))
    if len({host.endpoint for host in configured}) != len(configured):
        raise RgpuError("duplicate authenticated transport endpoint")
    return configured


def run_checked(
    command: Sequence[str],
    *,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            list(command),
            124,
            exc.stdout or "",
            exc.stderr or f"timed out after {timeout:g}s",
        )


def invoking_user_prefix() -> list[str]:
    """Run network operations as the user who invoked sudo, preserving SSH identity."""
    if os.geteuid() != 0:
        return []
    sudo_user = os.environ.get("SUDO_USER", "")
    if not sudo_user or sudo_user == "root":
        return []
    try:
        account = pwd.getpwnam(sudo_user)
    except KeyError as error:
        raise RgpuError(f"invalid SUDO_USER account: {sudo_user!r}") from error
    if account.pw_uid == 0:
        return []
    return ["sudo", "-u", sudo_user, "-H", "--"]


def ssh(
    host: Host,
    command: Sequence[str],
    *,
    capture: bool = True,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    return run_checked(
        [
            *invoking_user_prefix(),
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            host.ssh_target,
            shlex.join(command),
        ],
        capture=capture,
        timeout=timeout,
    )


def require_success(result: subprocess.CompletedProcess[str], context: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RgpuError(f"{context}: {detail}")
    return (result.stdout or "").strip()


def local_gpu_identity() -> str:
    result = run_checked(["nvidia-smi", "-L"], timeout=15)
    return require_success(result, "local nvidia-smi safety check")


def local_compute_processes() -> str:
    result = run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    return require_success(result, "local GPU process safety check")


def remote_compute_processes(host: Host) -> str:
    result = ssh(
        host,
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    )
    return require_success(result, f"remote GPU check on {host.ssh_target}")


def remote_gpu_identity(host: Host) -> str:
    result = ssh(host, ["nvidia-smi", "-L"])
    return require_success(result, f"remote GPU identity on {host.ssh_target}")


def remote_client_interface(host: Host) -> str | None:
    """Return the remote interface carrying this SSH connection."""
    result = ssh(
        host,
        [
            "sh",
            "-lc",
            'set -- $SSH_CONNECTION; ip -o route get "$1"',
        ],
    )
    if result.returncode != 0:
        return None
    fields = (result.stdout or "").split()
    try:
        interface = fields[fields.index("dev") + 1]
    except (ValueError, IndexError):
        return None
    safe = interface.replace("-", "").replace("_", "").isalnum()
    return interface if interface and safe else None


def local_server_interface(host: Host) -> str | None:
    """Return the local interface used to reach a remote GPU host."""
    result = run_checked(["ip", "-o", "route", "get", host.address])
    if result.returncode != 0:
        return None
    fields = (result.stdout or "").split()
    try:
        interface = fields[fields.index("dev") + 1]
    except (ValueError, IndexError):
        return None
    safe = interface.replace("-", "").replace("_", "").isalnum()
    return interface if interface and safe else None


def gpu_uuids(identity: str) -> set[str]:
    uuids: set[str] = set()
    for line in identity.splitlines():
        marker = "UUID: "
        if marker in line:
            uuids.add(line.split(marker, 1)[1].rstrip(")"))
    return uuids


def local_image_id(image: str) -> str:
    result = run_checked(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    return require_success(result, f"inspect local image {image}")


def remote_image_id(host: Host, image: str) -> str:
    result = ssh(host, ["docker", "image", "inspect", "--format", "{{.Id}}", image])
    return require_success(result, f"inspect {image} on {host.ssh_target}")


def require_matching_server_image(host: Host, image: str) -> None:
    local_id = local_image_id(image)
    try:
        remote_id = remote_image_id(host, image)
    except RgpuError as error:
        raise RgpuError(
            f"server image {image} is not deployed on {host.ssh_target}; "
            "rerun with --deploy"
        ) from error
    if remote_id != local_id:
        raise RgpuError(
            f"server image {image} differs on {host.ssh_target} "
            f"(local {local_id[:19]}, remote {remote_id[:19]}); rerun with --deploy"
        )


def shim_cache_root() -> Path:
    override = os.environ.get("RGPU_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve() / "shims"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache.resolve() / "remote-gpu" / "shims"


def prepare_shim_bundle(image: str) -> Path:
    """Atomically cache userspace interposition files from a validated image."""
    image_id = local_image_id(image)
    cache_root = shim_cache_root()
    bundle = cache_root / image_id.replace(":", "-")
    required = (
        bundle / "lib" / "libcuda.so.1",
        bundle / "lib" / "libnvidia-ml.so.1",
        bundle / "lib" / "liblupine-cudart-compat.so",
        bundle / "lib" / "libcudart.so.13",
        bundle / "bin" / "nvidia-smi",
        bundle / "lib" / "libcublas_rpc.so",
        bundle / "lib" / "libcusolver_rpc.so",
        bundle / "lib" / "libcufft_rpc.so",
        bundle / "lib" / "libunsupported_rpc_guard.so",
        bundle / "lib" / "libnccl.so.2",
        bundle / "lib" / "libnccl_real.so.2",
        bundle / "python" / "sitecustomize.py",
    )
    if all(path.is_file() for path in required):
        return bundle

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="shim-", dir=cache_root))
    container_id = ""
    try:
        (temporary / "lib").mkdir()
        (temporary / "bin").mkdir()
        (temporary / "python").mkdir()
        container_id = require_success(
            run_checked(["docker", "create", image, "true"]),
            f"create shim extraction container from {image}",
        )
        require_success(
            run_checked(
                [
                    "docker",
                    "cp",
                    f"{container_id}:/opt/lupine/lib/.",
                    str(temporary / "lib"),
                ]
            ),
            f"extract CUDA shim libraries from {image}",
        )
        require_success(
            run_checked(
                [
                    "docker",
                    "cp",
                    f"{container_id}:/opt/rgpu/lib/.",
                    str(temporary / "lib"),
                ]
            ),
            f"extract CUDA-library interposers from {image}",
        )
        require_success(
            run_checked(
                [
                    "docker",
                    "cp",
                    f"{container_id}:/opt/rgpu/python/.",
                    str(temporary / "python"),
                ]
            ),
            f"extract Python bootstrap from {image}",
        )
        require_success(
            run_checked(
                [
                    "docker",
                    "cp",
                    f"{container_id}:/usr/bin/nvidia-smi",
                    str(temporary / "bin" / "nvidia-smi"),
                ]
            ),
            f"extract nvidia-smi from {image}",
        )
        extracted = (
            temporary / "lib" / "libcuda.so.1",
            temporary / "lib" / "libnvidia-ml.so.1",
            temporary / "lib" / "liblupine-cudart-compat.so",
            temporary / "lib" / "libcudart.so.13",
            temporary / "bin" / "nvidia-smi",
            temporary / "lib" / "libcublas_rpc.so",
            temporary / "lib" / "libcusolver_rpc.so",
            temporary / "lib" / "libcufft_rpc.so",
            temporary / "lib" / "libunsupported_rpc_guard.so",
            temporary / "lib" / "libnccl.so.2",
            temporary / "lib" / "libnccl_real.so.2",
            temporary / "python" / "sitecustomize.py",
        )
        if not all(path.is_file() for path in extracted):
            raise RgpuError(f"shim image {image} is missing required runtime files")
        try:
            os.replace(temporary, bundle)
        except OSError:
            if not all(path.is_file() for path in required):
                raise
        return bundle
    finally:
        if container_id:
            run_checked(["docker", "rm", "--force", container_id])
        if temporary.exists():
            shutil.rmtree(temporary)


def server_name(host: Host) -> str:
    """A stable Docker name acts as an atomic per-host/port lease."""
    return f"rgpu-server-{host.port}"


def remote_port_listener(host: Host) -> str:
    result = ssh(host, ["ss", "-H", "-ltn", f"sport = :{host.port}"])
    return require_success(result, f"check port {host.port} on {host.ssh_target}")


def remote_active_clients(host: Host) -> str:
    result = ssh(
        host,
        ["ss", "-H", "-tn", "state", "established", f"sport = :{host.port}"],
    )
    return require_success(result, f"check clients on {host.endpoint}")


def wait_ready(host: Host, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = run_checked(
            [
                sys.executable,
                "-c",
                (
                    "import socket,sys; "
                    "s=socket.create_connection((sys.argv[1],int(sys.argv[2])),.5); s.close()"
                ),
                host.address,
                str(host.port),
            ]
        )
        if probe.returncode == 0:
            return
        time.sleep(0.2)
    raise RgpuError(f"server {host.endpoint} did not become ready")


def start_server(
    host: Host,
    name: str,
    image: str,
    *,
    lease_mode: str = "ephemeral",
    session: str | None = None,
) -> None:
    active = remote_compute_processes(host)
    if active:
        raise RgpuError(
            f"remote GPU {host.ssh_target} is already in use; refusing to interfere:\n{active}"
        )
    existing = ssh(host, ["docker", "container", "inspect", name])
    if existing.returncode == 0:
        raise RgpuError(
            f"remote endpoint {host.endpoint} is leased by container {name}; "
            "refusing to share another user's server"
        )
    listener = remote_port_listener(host)
    if listener:
        raise RgpuError(
            f"remote endpoint {host.endpoint} already has a listener; refusing to attach:\n"
            f"{listener}"
        )
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--restart",
        "unless-stopped" if lease_mode == "persistent" else "no",
        "--label",
        "io.rgpu.managed=true",
        "--label",
        f"io.rgpu.endpoint={host.endpoint}",
        "--label",
        f"io.rgpu.lease-mode={lease_mode}",
        "--label",
        f"io.rgpu.created-unix={int(time.time())}",
    ]
    if session is not None:
        command.extend(["--label", f"io.rgpu.session={session}"])
    command.extend(
        [
            "--gpus",
            "all",
            "--pid",
            "host",
            "--network",
            "host",
            "-e",
            f"LUPINE_PORT={host.port}",
        ]
    )
    nccl_interface = remote_client_interface(host)
    if nccl_interface is not None:
        command.extend(["-e", f"NCCL_SOCKET_IFNAME={nccl_interface}"])
    command.append(image)
    require_success(ssh(host, command), f"start server on {host.ssh_target}")
    wait_ready(host)
    running = require_success(
        ssh(host, ["docker", "inspect", "--format", "{{.State.Running}}", name]),
        f"verify server on {host.ssh_target}",
    )
    if running != "true":
        raise RgpuError(f"remote server container {name} exited during startup")


def stop_server(host: Host, name: str) -> None:
    # Keep cleanup inside common orchestrator SIGTERM→SIGKILL grace periods.
    # The server owns only this launcher's leased CUDA contexts and does not
    # persist state, so a short Docker grace period is sufficient.
    require_success(
        ssh(host, ["docker", "rm", "--force", name], timeout=15),
        f"remove lease {name} on {host.ssh_target}",
    )


def deploy_image(host: Host, image: str) -> None:
    save = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    assert save.stdout is not None
    load = subprocess.Popen(
        [
            *invoking_user_prefix(),
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            host.ssh_target,
            "docker load",
        ],
        stdin=save.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    save.stdout.close()
    try:
        _, load_error = load.communicate(timeout=240)
        save_result = save.wait(timeout=15)
    except subprocess.TimeoutExpired as exc:
        load.kill()
        save.kill()
        load.wait()
        save.wait()
        raise RgpuError(
            f"deploy {image} to {host.ssh_target} exceeded 240s"
        ) from exc
    except BaseException:
        load.kill()
        save.kill()
        load.wait()
        save.wait()
        raise
    if save_result != 0 or load.returncode != 0:
        raise RgpuError(f"deploy {image} to {host.ssh_target}: {load_error.strip()}")


def client_command(
    args: argparse.Namespace,
    hosts: Sequence[Host],
    session: str,
    shim_bundle: Path | None = None,
    nccl_interfaces: Sequence[str] = (),
) -> list[str]:
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        raise RgpuError(f"workspace is not a directory: {workspace}")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RgpuError("missing command after --")
    mount_mode = "rw" if args.workspace_write else "ro"
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"rgpu-client-{session}",
        "--network",
        "host",
        "-e",
        f"LUPINE_SERVER={','.join(host.endpoint for host in hosts)}",
        "-e",
        f"LUPINE_SESSION={session}",
        "-v",
        f"{workspace}:{workspace}:{mount_mode}",
        "-w",
        str(workspace),
    ]
    if args.include_local:
        # The NVIDIA container runtime mounts the host driver/device nodes;
        # LUPINE's routing layer then appends remote ordinals after them. This
        # is opt-in so the default remains a strict remote-only safety boundary.
        docker_command[3:3] = ["--gpus", "all"]
        user_selected_nccl_interface = any(
            value == "NCCL_SOCKET_IFNAME"
            or value.startswith("NCCL_SOCKET_IFNAME=")
            for value in args.env
        )
        if nccl_interfaces and not user_selected_nccl_interface:
            docker_command.extend(
                ["-e", f"NCCL_SOCKET_IFNAME={','.join(dict.fromkeys(nccl_interfaces))}"]
            )
        mixed_python = Path(__file__).resolve().parent / "mixed_python"
        if mixed_python.is_dir():
            docker_command.extend(
                [
                    "-v",
                    f"{mixed_python}:/run/rgpu/python:ro",
                    "-e",
                    "PYTHONPATH=/run/rgpu/python",
                    "-e",
                    "RGPU_MIXED_PYTORCH_PRIME=1",
                ]
            )
    else:
        docker_command.extend(["-e", "LUPINE_DISABLE_LOCAL=1"])
    if shim_bundle is not None:
        # Keep the injected CUDA/NVML shim separate from workload-owned files.
        # The default NCCL overlay stores its real-library companion under
        # /opt/rgpu/lib, so mounting the shim bundle there would hide it.
        container_lib = "/run/rgpu/lib"
        preload = f"{container_lib}/liblupine-cudart-compat.so"
        if getattr(args, "cublas_rpc", False):
            preload += (
                f":{container_lib}/libcublas_rpc.so"
                f":{container_lib}/libcusolver_rpc.so"
                f":{container_lib}/libcufft_rpc.so"
            )
            if not getattr(args, "allow_unsupported_library_fallback", False):
                preload += f":{container_lib}/libunsupported_rpc_guard.so"
            docker_command.extend(["-e", "RGPU_CUFFT_RPC=1"])
            user_selected_addmm_backend = any(
                value == "DISABLE_ADDMM_CUDA_LT"
                or value.startswith("DISABLE_ADDMM_CUDA_LT=")
                for value in args.env
            )
            if not user_selected_addmm_backend:
                # PyTorch's cuBLASLt addmm path creates and destroys a large
                # descriptor graph for every matrix operation. Across a LAN,
                # its synchronous control RPCs cost far more than the GEMM.
                # The classic cuBLAS path has equivalent PyTorch semantics and
                # lets handle state ride on the GEMM request itself.
                docker_command.extend(["-e", "DISABLE_ADDMM_CUDA_LT=1"])
            user_selected_linalg_backend = any(
                value == "TORCH_LINALG_PREFER_CUSOLVER"
                or value.startswith("TORCH_LINALG_PREFER_CUSOLVER=")
                for value in args.env
            )
            if not user_selected_linalg_backend:
                # MAGMA launches kernels over device-resident arrays of device
                # pointers. Those embedded addresses are not generally
                # relocatable across CUDA-remoting address spaces. PyTorch's
                # equivalent cuSOLVER backend uses the explicitly virtualized
                # opaque-library path and preserves the same public semantics.
                docker_command.extend(["-e", "TORCH_LINALG_PREFER_CUSOLVER=1"])
            if getattr(args, "cublas_async", False):
                user_selected_async = any(
                    value == "RGPU_CUBLAS_ASYNC"
                    or value.startswith("RGPU_CUBLAS_ASYNC=")
                    for value in args.env
                )
                if not user_selected_async:
                    docker_command.extend(["-e", "RGPU_CUBLAS_ASYNC=1"])
        docker_command.extend(
            [
                "-v",
                f"{shim_bundle / 'lib'}:{container_lib}:ro",
                "-v",
                f"{shim_bundle / 'bin' / 'nvidia-smi'}:/usr/local/bin/nvidia-smi:ro",
                "-e",
                f"LUPINE_LIBCUDA={container_lib}/libcuda.so.1",
                "-e",
                f"LUPINE_LIB={container_lib}/libcuda.so.1",
                "-e",
                f"LD_LIBRARY_PATH={container_lib}",
                "-e",
                f"LD_PRELOAD={preload}",
            ]
        )
    if args.output:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        docker_command.extend(["-v", f"{output}:{output}:rw"])
    for value in args.env:
        if "=" not in value and value not in os.environ:
            raise RgpuError(f"environment variable is not set: {value}")
        docker_command.extend(["-e", value])
    docker_command.extend([args.image, *command])
    return docker_command


def command_status(args: argparse.Namespace) -> int:
    hosts = expand_hosts(args.host)
    payload = []
    for host in hosts:
        gpu = require_success(
            ssh(host, ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used", "--format=csv,noheader"]),
            f"query {host.ssh_target}",
        )
        payload.append(
            {
                "ssh_target": host.ssh_target,
                "endpoint": host.endpoint,
                "gpu": gpu.splitlines(),
                "compute_processes": remote_compute_processes(host).splitlines(),
            }
        )
    print(json.dumps(payload, indent=2))
    return 0


def command_deploy(args: argparse.Namespace) -> int:
    for host in expand_hosts(args.host):
        deploy_image(host, args.server_image)
        print(f"deployed {args.server_image} to {host.ssh_target}")
    return 0


def command_attach(args: argparse.Namespace) -> int:
    """Transactionally attach into a disposable root or the idle live host."""
    requested_root = Path(args.root).resolve()
    live = requested_root == Path("/")
    if live and os.geteuid() != 0:
        raise HostwideError("live attachment requires sudo")
    state_path = requested_root / "var/lib/rgpu/state.json"
    if state_path.exists():
        raise HostwideError(f"rgpu is already attached in {requested_root}")
    if live:
        active = local_compute_processes()
        if active:
            raise HostwideError(
                "local GPU has active compute processes; refusing live attachment:\n"
                + active
            )
    hosts = apply_transport_endpoints(expand_hosts(args.host), args.endpoint)
    before = local_gpu_identity()
    expected_uuids = gpu_uuids(before)
    shim_bundle = prepare_shim_bundle(args.shim_image)
    started: list[tuple[Host, str]] = []
    try:
        if args.deploy:
            for host in hosts:
                deploy_image(host, args.server_image)
        for host in hosts:
            require_matching_server_image(host, args.server_image)
            expected_uuids.update(gpu_uuids(remote_gpu_identity(host)))
            name = server_name(host)
            start_server(host, name, args.server_image, lease_mode="persistent")
            started.append((host, name))
        state = hostwide_install(
            requested_root,
            shim_bundle,
            [host.endpoint for host in hosts],
            metadata={
                "physical_gpu_identity": before,
                "leases": [
                    {
                        "ssh_target": host.ssh_target,
                        "address": host.address,
                        "port": host.port,
                        "name": name,
                        "server_image": args.server_image,
                    }
                    for host, name in started
                ]
            },
            allow_live=live,
        )
        if live:
            attached_identity = local_gpu_identity()
            attached_uuids = gpu_uuids(attached_identity)
            if not expected_uuids.issubset(attached_uuids):
                raise RgpuError(
                    "post-attach nvidia-smi did not enumerate every local and remote GPU"
                )
    except Exception as attach_error:
        rollback_errors: list[str] = []
        if state_path.exists():
            try:
                hostwide_uninstall(requested_root, allow_live=live)
            except Exception as rollback_error:
                rollback_errors.append(f"local transaction: {rollback_error}")
        for host, name in reversed(started):
            try:
                stop_server(host, name)
            except Exception as cleanup_error:
                rollback_errors.append(
                    f"remote lease {name} on {host.ssh_target}: {cleanup_error}"
                )
        if live:
            try:
                restored_identity = local_gpu_identity()
            except Exception as identity_error:
                rollback_errors.append(f"physical identity check: {identity_error}")
            else:
                if restored_identity != before:
                    rollback_errors.append("physical GPU identity was not restored")
        if rollback_errors:
            rescue = "; run sudo rgpu-rescue immediately" if state_path.exists() else ""
            raise RgpuError(
                f"attach failed: {attach_error}; automatic rollback incomplete: "
                + "; ".join(rollback_errors)
                + rescue
            ) from attach_error
        raise
    finally:
        if not live:
            after = local_gpu_identity()
            if after != before:
                raise RgpuError("local GPU identity changed during sandbox attach")
    print(
        json.dumps(
            {
                "status": "attached",
                "root": state["root"],
                "endpoints": state["endpoints"],
                "live_root": state["live_root"],
            },
            indent=2,
        )
    )
    if live:
        print(
            "nvidia-smi is ready now; open a new terminal or start a new "
            "login shell before launching transparent host-wide PyTorch."
        )
    return 0


def command_detach(args: argparse.Namespace) -> int:
    requested_root = Path(args.root).resolve()
    live = requested_root == Path("/")
    if live and os.geteuid() != 0:
        raise HostwideError("live detach requires sudo")
    before = "" if live else local_gpu_identity()
    state = hostwide_uninstall(requested_root, allow_live=live)
    cleanup_errors: list[str] = []
    for lease in reversed(state.get("leases", [])):
        host = Host(
            ssh_target=lease["ssh_target"],
            address=lease["address"],
            port=int(lease["port"]),
        )
        try:
            stop_server(host, lease["name"])
        except Exception as exc:
            # The local driver restoration has already happened and must be
            # verified even when a remote machine is unreachable. Also keep
            # attempting later leases so one failed host cannot strand the
            # remaining servers.
            cleanup_errors.append(str(exc))
    after = local_gpu_identity()
    if live and after != state.get("physical_gpu_identity"):
        detail = f"; remote cleanup errors: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
        raise RgpuError(f"physical GPU identity was not restored after live detach{detail}")
    if not live and after != before:
        detail = f"; remote cleanup errors: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
        raise RgpuError(f"local GPU identity changed during sandbox detach{detail}")
    if cleanup_errors:
        raise RgpuError(
            "local GPU restoration verified, but remote lease cleanup failed: "
            + "; ".join(cleanup_errors)
        )
    print(json.dumps({"status": "detached", "root": state["root"]}, indent=2))
    return 0


def command_gc(args: argparse.Namespace) -> int:
    """Reclaim only disconnected, expired ephemeral rgpu server leases."""
    if args.min_age < 60:
        raise RgpuError("--min-age must be at least 60 seconds")
    report = []
    now = int(time.time())
    for host in expand_hosts(args.host):
        name = server_name(host)
        result = ssh(
            host,
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", name],
        )
        if result.returncode != 0:
            if result.returncode != 1:
                require_success(result, f"inspect lease on {host.ssh_target}")
            report.append({"host": host.ssh_target, "status": "no-lease"})
            continue
        try:
            labels = json.loads(result.stdout)
            created = int(labels.get("io.rgpu.created-unix", "0"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RgpuError(f"invalid lease metadata on {host.ssh_target}") from exc
        if labels.get("io.rgpu.managed") != "true":
            raise RgpuError(f"refusing unmanaged container {name} on {host.ssh_target}")
        if labels.get("io.rgpu.lease-mode") != "ephemeral":
            report.append({"host": host.ssh_target, "status": "persistent"})
            continue
        age = max(0, now - created)
        if created <= 0 or age < args.min_age:
            report.append({"host": host.ssh_target, "status": "not-expired", "age": age})
            continue
        if clients := remote_active_clients(host):
            report.append(
                {
                    "host": host.ssh_target,
                    "status": "active-client",
                    "clients": clients.splitlines(),
                }
            )
            continue
        require_success(
            ssh(host, ["docker", "rm", "--force", name]),
            f"reclaim lease on {host.ssh_target}",
        )
        report.append({"host": host.ssh_target, "status": "reclaimed", "age": age})
    print(json.dumps(report, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    if getattr(args, "cublas_async", False) and not args.cublas_rpc:
        raise RgpuError("--cublas-async requires --cublas-rpc")
    if (
        getattr(args, "allow_unsupported_library_fallback", False)
        and not args.cublas_rpc
    ):
        raise RgpuError(
            "--allow-unsupported-library-fallback requires --cublas-rpc"
        )
    hosts = apply_transport_endpoints(expand_hosts(args.host), args.endpoint)
    nccl_interfaces = tuple(
        interface
        for host in hosts
        if (interface := local_server_interface(host)) is not None
    )
    before = local_gpu_identity()
    shim_bundle = prepare_shim_bundle(args.shim_image)
    session = f"{os.getpid():x}{int(time.time()):x}"
    client_name = f"rgpu-client-{session}"
    started: list[tuple[Host, str]] = []
    child: subprocess.Popen[bytes] | None = None

    def forward(signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        # Docker's attached client can keep waiting after SIGTERM even when
        # the process inside the container exits. Unwind immediately so the
        # launcher's finally block releases every remote lease.
        raise LaunchInterrupted(signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.deploy:
            for host in hosts:
                deploy_image(host, args.server_image)
        for host in hosts:
            require_matching_server_image(host, args.server_image)
            name = server_name(host)
            start_server(
                host,
                name,
                args.server_image,
                lease_mode="ephemeral",
                session=session,
            )
            started.append((host, name))
        child = subprocess.Popen(
            client_command(
                args,
                hosts,
                session,
                shim_bundle,
                nccl_interfaces=nccl_interfaces,
            )
        )
        return_code = child.wait()
    except LaunchInterrupted as interruption:
        return_code = 128 + interruption.signum
    finally:
        cleanup_errors = []
        if child is not None and child.poll() is None:
            # A signal sent to the attached Docker CLI is not guaranteed to
            # stop the container. Target only this launcher's unique name.
            run_checked(["docker", "stop", "--timeout", "1", client_name])
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for host, name in reversed(started):
            try:
                stop_server(host, name)
            except RgpuError as error:
                cleanup_errors.append(str(error))
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        after = local_gpu_identity()
        if after != before:
            raise RgpuError(
                "local GPU identity changed during rgpu run; no host files were modified, "
                "but the safety invariant requires investigation"
            )
        if cleanup_errors:
            raise RgpuError("remote lease cleanup failed: " + "; ".join(cleanup_errors))
    return return_code


def native_python_rank_command(
    image: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    environment: Sequence[str],
    script_args: Sequence[str],
) -> list[str]:
    """Build a native-driver container command for one distributed rank."""
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "--network",
        "host",
        "--shm-size=1g",
        "-e",
        f"MASTER_ADDR={master_addr}",
        "-e",
        f"MASTER_PORT={master_port}",
        "-e",
        f"RANK={rank}",
        "-e",
        f"WORLD_SIZE={world_size}",
        "-e",
        "LOCAL_RANK=0",
        "-e",
        "NCCL_SOCKET_IFNAME=eno2",
    ]
    for value in environment:
        if "=" not in value and value not in os.environ:
            raise RgpuError(f"environment variable is not set: {value}")
        command.extend(["-e", value])
    command.extend([image, "python3", "-", *script_args])
    return command


def command_native_python(args: argparse.Namespace) -> int:
    """Run native NCCL ranks locally and remotely from one Python script."""
    hosts = expand_hosts(args.host)
    script = Path(args.script).resolve()
    if not script.is_file():
        raise RgpuError(f"Python workload does not exist: {script}")
    source = script.read_bytes()
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    local_id = local_image_id(args.image)
    for host in hosts:
        try:
            remote_id = remote_image_id(host, args.image)
        except RgpuError as error:
            raise RgpuError(
                f"workload image {args.image} is missing on {host.ssh_target}"
            ) from error
        if remote_id != local_id:
            raise RgpuError(
                f"workload image {args.image} differs on {host.ssh_target}"
            )

    world_size = len(hosts) + 1
    children: list[subprocess.Popen[bytes]] = []

    def stop_children() -> None:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 3
        for child in children:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    try:
        for rank, host in enumerate(hosts, start=1):
            docker_command = native_python_rank_command(
                args.image,
                rank,
                world_size,
                args.master_addr,
                args.master_port,
                args.env,
                script_args,
            )
            remote_command = [
                *invoking_user_prefix(),
                "ssh",
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                host.ssh_target,
                shlex.join(docker_command),
            ]
            child = subprocess.Popen(remote_command, stdin=subprocess.PIPE)
            children.append(child)
            assert child.stdin is not None
            child.stdin.write(source)
            child.stdin.close()

        local_command = native_python_rank_command(
            args.image,
            0,
            world_size,
            args.master_addr,
            args.master_port,
            args.env,
            script_args,
        )
        local = subprocess.Popen(local_command, stdin=subprocess.PIPE)
        children.append(local)
        assert local.stdin is not None
        local.stdin.write(source)
        local.stdin.close()

        deadline = time.monotonic() + args.timeout
        while True:
            codes = [child.poll() for child in children]
            failed = next((code for code in codes if code not in (None, 0)), None)
            if failed is not None:
                stop_children()
                return failed if failed > 0 else 128 - failed
            if all(code == 0 for code in codes):
                return 0
            if time.monotonic() >= deadline:
                stop_children()
                raise RgpuError(f"native distributed run exceeded {args.timeout}s")
            time.sleep(0.05)
    except BaseException:
        stop_children()
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rgpu")
    root.add_argument("--version", action="version", version=f"rgpu {__version__}")
    commands = root.add_subparsers(dest="subcommand", required=True)

    status = commands.add_parser("status", help="inspect remote GPUs without starting a server")
    status.add_argument("--host", action="append", required=True)
    status.set_defaults(func=command_status)

    deploy = commands.add_parser("deploy", help="copy the pinned server image to GPU hosts")
    deploy.add_argument("--host", action="append", required=True)
    deploy.add_argument("--server-image", default=DEFAULT_SERVER_IMAGE)
    deploy.set_defaults(func=command_deploy)

    attach = commands.add_parser(
        "attach",
        help="attach remote GPUs host-wide, or into an explicit disposable root",
    )
    attach.add_argument("--host", action="append", required=True)
    attach.add_argument(
        "--endpoint",
        action="append",
        help=("verified HTTPS transport endpoint corresponding by order to each "
              "--host; repeat or use comma-separated values"),
    )
    attach.add_argument(
        "--root",
        default="/",
        help="installation root; defaults to the live host and requires sudo",
    )
    attach.add_argument("--server-image", default=DEFAULT_SERVER_IMAGE)
    attach.add_argument("--shim-image", default=DEFAULT_SHIM_IMAGE)
    attach.add_argument("--deploy", action="store_true")
    attach.set_defaults(func=command_attach)

    detach = commands.add_parser(
        "detach", help="detach remote GPUs and restore the prior loader state"
    )
    detach.add_argument(
        "--root",
        default="/",
        help="installation root; defaults to the live host and requires sudo",
    )
    detach.set_defaults(func=command_detach)

    gc = commands.add_parser(
        "gc", help="reclaim disconnected, expired ephemeral server leases"
    )
    gc.add_argument("--host", action="append", required=True)
    gc.add_argument("--min-age", type=int, default=3600)
    gc.set_defaults(func=command_gc)

    run = commands.add_parser("run", help="run a command with the selected remote GPUs")
    run.add_argument("--host", action="append", required=True)
    run.add_argument(
        "--endpoint",
        action="append",
        help=("verified HTTPS transport endpoint corresponding by order to each "
              "--host; repeat or use comma-separated values"),
    )
    run.add_argument("--image", default=DEFAULT_CLIENT_IMAGE)
    run.add_argument("--server-image", default=DEFAULT_SERVER_IMAGE)
    run.add_argument("--shim-image", default=DEFAULT_SHIM_IMAGE)
    run.add_argument("--deploy", action="store_true")
    run.add_argument("--workspace", default=os.getcwd())
    run.add_argument("--workspace-write", action="store_true")
    run.add_argument(
        "--include-local",
        action="store_true",
        help="append selected remote GPUs after this machine's local GPUs",
    )
    run.add_argument(
        "--cublas-rpc",
        action="store_true",
        help=("enable experimental same-process local/remote CUDA math-library "
              "routing (cuBLAS, cuBLASLt, cuSOLVER, and cuFFT)"),
    )
    run.add_argument(
        "--cublas-async",
        action="store_true",
        help="defer valid remote cuBLAS compute calls on their CUDA stream",
    )
    run.add_argument(
        "--allow-unsupported-library-fallback",
        action="store_true",
        help=("permit unimplemented opaque-library APIs to fall through to "
              "vendor userspace; unsafe for calls containing embedded device "
              "pointers and disabled by default"),
    )
    run.add_argument("--output")
    run.add_argument("--env", action="append", default=[])
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=command_run)

    native_python = commands.add_parser(
        "native-python",
        help="run one native NCCL Python rank locally and on each remote host",
    )
    native_python.add_argument("--host", action="append", required=True)
    native_python.add_argument(
        "--image", default="remote-gpu-pytorch-native:2.12.0-cu130"
    )
    native_python.add_argument("--master-addr", default="10.77.77.2")
    native_python.add_argument("--master-port", type=int, default=29681)
    native_python.add_argument("--timeout", type=float, default=300.0)
    native_python.add_argument("--env", action="append", default=[])
    native_python.add_argument("script")
    native_python.add_argument("script_args", nargs=argparse.REMAINDER)
    native_python.set_defaults(func=command_native_python)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (RgpuError, HostwideError) as error:
        print(f"rgpu: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
