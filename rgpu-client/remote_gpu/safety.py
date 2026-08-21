"""Read-only, out-of-band fingerprints for the physical NVIDIA stack."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


PHYSICAL_LIBRARY_CANDIDATES = (
    Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    Path("/usr/lib/aarch64-linux-gnu/libcuda.so.1"),
    Path("/usr/lib64/libcuda.so.1"),
    Path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
    Path("/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1"),
    Path("/usr/lib64/libnvidia-ml.so.1"),
)


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timed out after {timeout:g}s",
        }
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_libraries() -> list[dict[str, object]]:
    records = []
    seen: set[Path] = set()
    for candidate in PHYSICAL_LIBRARY_CANDIDATES:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(
            {
                "link": str(candidate),
                "target": str(resolved),
                "size": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    return records


def _device_nodes() -> list[dict[str, object]]:
    records = []
    for path in sorted(Path("/dev").glob("nvidia*")):
        info = path.stat()
        if not stat.S_ISCHR(info.st_mode):
            continue
        records.append(
            {
                "path": str(path),
                "major": os.major(info.st_rdev),
                "minor": os.minor(info.st_rdev),
                "mode": stat.S_IMODE(info.st_mode),
                "uid": info.st_uid,
                "gid": info.st_gid,
            }
        )
    return records


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _physical_nvml_environment() -> dict[str, str]:
    """Force nvidia-smi to map the vendor NVML, bypassing an attached shim."""
    physical = next(
        (
            candidate.resolve()
            for candidate in PHYSICAL_LIBRARY_CANDIDATES
            if candidate.name == "libnvidia-ml.so.1" and candidate.exists()
        ),
        None,
    )
    if physical is None:
        raise RuntimeError("physical NVIDIA management library was not found")
    environment = os.environ.copy()
    environment["LD_PRELOAD"] = str(physical)
    environment.pop("LUPINE_SERVER", None)
    return environment


def collect() -> dict[str, object]:
    physical_environment = _physical_nvml_environment()
    gpu_identity = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        environment=physical_environment,
    )
    if gpu_identity["exit_code"] != 0:
        raise RuntimeError(
            "physical nvidia-smi identity query failed: "
            + str(gpu_identity["stderr"]).strip()
        )
    record: dict[str, object] = {
        "schema": 1,
        "gpu_identity": gpu_identity,
        "compute_processes": _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            environment=physical_environment,
        ),
        "kernel_module_version": _read_optional(Path("/sys/module/nvidia/version")),
        "proc_driver_version": _read_optional(Path("/proc/driver/nvidia/version")),
        "physical_libraries": _physical_libraries(),
        "device_nodes": _device_nodes(),
        "nvidia_smi_binary": None,
    }
    nvidia_smi = Path("/usr/bin/nvidia-smi")
    if nvidia_smi.is_file():
        resolved = nvidia_smi.resolve()
        record["nvidia_smi_binary"] = {
            "target": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }

    immutable = {
        key: record[key]
        for key in (
            "gpu_identity",
            "kernel_module_version",
            "proc_driver_version",
            "physical_libraries",
            "device_nodes",
            "nvidia_smi_binary",
        )
    }
    record["immutable_fingerprint"] = hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record
