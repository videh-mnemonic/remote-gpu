#!/usr/bin/env python3
"""Launch one local-visible rank and one remote-only rank for mixed NCCL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workload", nargs="?", type=Path,
        default=Path(__file__).with_name("nccl_smoke.py")
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    workload = args.workload.resolve()
    children: list[subprocess.Popen[bytes]] = []
    for rank in range(2):
        env = os.environ.copy()
        env.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29603",
                "RANK": str(rank),
                "WORLD_SIZE": "2",
                # Each rank intentionally receives a one-device application
                # view even though their physical routes differ.
                "LOCAL_RANK": "0",
            }
        )
        if rank == 1:
            env["LUPINE_DISABLE_LOCAL"] = "1"
            # NCCL must execute beside the virtualized CUDA context so its
            # proxy threads and registrations use server-native addresses.
            if env.get("RGPU_NCCL_AUTO_ROUTE") == "1":
                env.pop("RGPU_NCCL_REMOTE", None)
            else:
                env["RGPU_NCCL_REMOTE"] = "1"
            remote_preload = env.get("RGPU_REMOTE_PRELOAD")
            if remote_preload:
                existing = env.get("LD_PRELOAD")
                env["LD_PRELOAD"] = (
                    f"{remote_preload}:{existing}" if existing else remote_preload
                )
        else:
            env.pop("LUPINE_DISABLE_LOCAL", None)
            env.pop("RGPU_NCCL_REMOTE", None)
            local_preload = env.get("RGPU_LOCAL_PRELOAD")
            if local_preload:
                existing = env.get("LD_PRELOAD")
                env["LD_PRELOAD"] = (
                    f"{local_preload}:{existing}" if existing else local_preload
                )
        children.append(subprocess.Popen([sys.executable, str(workload)], env=env))

    result = 0
    try:
        deadline = time.monotonic() + args.timeout
        pending = set(range(len(children)))
        while pending and time.monotonic() < deadline:
            for rank in tuple(pending):
                returncode = children[rank].poll()
                if returncode is None:
                    continue
                pending.remove(rank)
                print(f"launcher rank={rank} returncode={returncode}", flush=True)
                result = max(
                    result, 128 - returncode if returncode < 0 else returncode
                )
                if returncode != 0:
                    return result
            time.sleep(0.01)
        if pending:
            result = 124
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
