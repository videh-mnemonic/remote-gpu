#!/usr/bin/env python3
"""Run the bounded release gate against an attached host-wide rgpu install."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str], timeout: float = 240) -> dict[str, object]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "exit_code": 124,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": error.stdout or "",
            "stderr": error.stderr or f"timed out after {timeout:g} seconds",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="remote-gpu-workload:0.2.1")
    parser.add_argument("--expected-devices", type=int, default=2)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    required = (
        Path("/usr/local/lib/rgpu/libcuda.so.1"),
        Path("/usr/local/lib/rgpu/libnvidia-ml.so.1"),
        Path("/usr/local/lib/rgpu/libcublas_rpc.so"),
        Path("/usr/local/lib/rgpu/libcusolver_rpc.so"),
        Path("/usr/local/lib/rgpu/libcufft_rpc.so"),
        Path("/usr/local/lib/rgpu/libunsupported_rpc_guard.so"),
        Path("/usr/local/lib/rgpu/python/sitecustomize.py"),
        Path("/etc/rgpu/endpoints"),
        Path("/etc/profile.d/rgpu.sh"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    record: dict[str, object] = {
        "schema": 1,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "image": args.image,
        "expected_devices": args.expected_devices,
        "missing_install_files": missing,
        "checks": {},
    }

    checks: dict[str, object] = record["checks"]  # type: ignore[assignment]
    checks["nvidia_smi"] = run(["nvidia-smi", "-L"], timeout=15)
    nvidia_output = str(checks["nvidia_smi"].get("stdout", ""))  # type: ignore[union-attr]
    enumerated = sum(line.startswith("GPU ") for line in nvidia_output.splitlines())
    checks["nvidia_smi"]["enumerated_devices"] = enumerated  # type: ignore[index]

    checks["non_torch_python"] = run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "/usr/local/lib/rgpu:/usr/local/lib/rgpu:ro",
            "-e",
            "LD_LIBRARY_PATH=/usr/local/lib/rgpu",
            "-e",
            "PYTHONPATH=/usr/local/lib/rgpu/python",
            "-e",
            "RGPU_HOSTWIDE_PYTHON=1",
            args.image,
            "python3",
            "-c",
            (
                "from pathlib import Path; "
                "maps=Path('/proc/self/maps').read_text(); "
                "assert 'libcuda.so' not in maps and 'libcublas_rpc.so' not in maps"
            ),
        ],
        timeout=30,
    )

    mixed_output = args.output.parent / "hostwide-mixed.json"
    checks["mixed_pytorch"] = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "host",
            "-v",
            f"{ROOT}:/workspace:ro",
            "-v",
            f"{args.output.parent}:{args.output.parent}:rw",
            "-v",
            "/usr/local/lib/rgpu:/usr/local/lib/rgpu:ro",
            "-v",
            "/etc/rgpu:/etc/rgpu:ro",
            "-w",
            "/workspace",
            "-e",
            "LD_LIBRARY_PATH=/usr/local/lib/rgpu",
            "-e",
            "PYTHONPATH=/usr/local/lib/rgpu/python",
            "-e",
            "RGPU_HOSTWIDE_PYTHON=1",
            "-e",
            "RGPU_MIXED_PYTORCH_PRIME=1",
            "-e",
            "RGPU_CUFFT_RPC=1",
            "-e",
            "DISABLE_ADDMM_CUDA_LT=1",
            "-e",
            "TORCH_LINALG_PREFER_CUSOLVER=1",
            args.image,
            "python3",
            "dev/tests/workloads/hostwide_mixed_acceptance.py",
            "--output",
            str(mixed_output),
        ]
    )
    if mixed_output.is_file():
        checks["mixed_pytorch"]["result"] = json.loads(  # type: ignore[index]
            mixed_output.read_text(encoding="utf-8")
        )

    passed = (
        not missing
        and checks["nvidia_smi"]["exit_code"] == 0  # type: ignore[index]
        and enumerated == args.expected_devices
        and "(via lupine " in nvidia_output
        and checks["non_torch_python"]["exit_code"] == 0  # type: ignore[index]
        and checks["mixed_pytorch"]["exit_code"] == 0  # type: ignore[index]
    )
    record["status"] = "pass" if passed else "fail"
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": record["status"], "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
