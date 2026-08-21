#!/usr/bin/env python3
"""Bounded CUDA error-recovery and link-loss probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch


def local_errors() -> dict[str, object]:
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA is unavailable")

    invalid_device_error = None
    try:
        torch.cuda.set_device(device_count)
    except RuntimeError as error:
        invalid_device_error = str(error)
    if invalid_device_error is None:
        raise AssertionError("an out-of-range CUDA device was accepted")

    torch.cuda.set_device(0)
    probe = torch.arange(16, device="cuda")
    torch.cuda.synchronize()
    if int(probe.sum().cpu()) != 120:
        raise AssertionError("valid work failed after the invalid-device error")

    properties = torch.cuda.get_device_properties(0)
    oom_error = None
    try:
        torch.empty(properties.total_memory // 4 + 1, dtype=torch.float32, device="cuda")
    except torch.OutOfMemoryError as error:
        oom_error = str(error)
    if oom_error is None:
        raise AssertionError("an allocation larger than device memory was accepted")

    torch.cuda.empty_cache()
    recovery = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    if float(recovery.sum().cpu()) != 1024.0:
        raise AssertionError("valid work failed after CUDA OOM")

    return {
        "device_count": device_count,
        "invalid_device": "pass",
        "oom": "pass",
        "post_error_recovery": "pass",
    }


def wait_for_link_loss(ready: Path, timeout_seconds: float) -> dict[str, object]:
    value = torch.ones(4096, device="cuda")
    torch.cuda.synchronize()
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + timeout_seconds
    iterations = 0
    while time.monotonic() < deadline:
        try:
            value.add_(1)
            torch.cuda.synchronize()
            iterations += 1
        except BaseException as error:
            return {
                "link_loss": "detected",
                "iterations": iterations,
                "error": f"{type(error).__name__}: {error}",
            }
    raise TimeoutError("the CUDA client did not detect link loss before the deadline")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-link-loss", action="store_true")
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.wait_for_link_loss:
        if args.ready is None:
            parser.error("--wait-for-link-loss requires --ready")
        result = wait_for_link_loss(args.ready, args.timeout_seconds)
    else:
        result = local_errors()
    result["status"] = "pass"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
