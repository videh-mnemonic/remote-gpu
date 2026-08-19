#!/usr/bin/env python3
"""Run one benchmark command with a hard timeout and structured raw output."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import time


MAX_CAPTURE_BYTES = 10 * 1024 * 1024


def clipped(value: bytes) -> tuple[str, bool]:
    was_clipped = len(value) > MAX_CAPTURE_BYTES
    if was_clipped:
        value = value[-MAX_CAPTURE_BYTES:]
    return value.decode("utf-8", errors="replace"), was_clipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("native", "remote-cuda", "remote-process"), required=True)
    parser.add_argument("--candidate", default="none")
    parser.add_argument("--timeout", type=float, default=270.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command after --")
    if not 0 < args.timeout < 300:
        parser.error("timeout must be greater than zero and strictly less than 300 seconds")

    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        elapsed = time.monotonic() - started
        record = {
            "schema_version": 1,
            "label": args.label,
            "mode": args.mode,
            "candidate": args.candidate,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "started_at": started_at.isoformat(),
            "elapsed_seconds": elapsed,
            "timeout_seconds": args.timeout,
            "timed_out": False,
            "exit_code": 127,
            "command": command,
            "stdout": "",
            "stderr": str(error),
            "stdout_clipped": False,
            "stderr_clipped": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"label": args.label, "elapsed_seconds": elapsed, "timed_out": False, "exit_code": 127}))
        return 127
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    stdout_text, stdout_clipped = clipped(stdout)
    stderr_text, stderr_clipped = clipped(stderr)
    record = {
        "schema_version": 1,
        "label": args.label,
        "mode": args.mode,
        "candidate": args.candidate,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": elapsed,
        "timeout_seconds": args.timeout,
        "timed_out": timed_out,
        "exit_code": process.returncode,
        "command": command,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_clipped": stdout_clipped,
        "stderr_clipped": stderr_clipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("label", "elapsed_seconds", "timed_out", "exit_code")}))
    return 124 if timed_out else int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
