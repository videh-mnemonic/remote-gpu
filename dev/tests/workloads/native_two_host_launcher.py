#!/usr/bin/env python3
"""Run one native NCCL rank on each of two Ethernet-connected workstations.

This is a correctness/performance fallback for mixed local+remote jobs while
same-process remote NCCL proxy memory is still under development.  The caller
sees one launch; each rank uses its workstation's unmodified NVIDIA stack.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def docker_rank_command(
    image: str, rank: int, master_addr: str, master_port: int
) -> list[str]:
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
        "WORLD_SIZE=2",
        "-e",
        "LOCAL_RANK=0",
        "-e",
        "NCCL_SOCKET_IFNAME=eno2",
    ]
    matrix_elements = os.environ.get("RGPU_NCCL_MATRIX_ELEMENTS")
    if matrix_elements:
        command.extend(["-e", f"RGPU_NCCL_MATRIX_ELEMENTS={matrix_elements}"])
    command.extend([
        image,
        "python3",
        "-",
    ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="SSH target for rank 1")
    parser.add_argument(
        "--image", default="remote-gpu-pytorch-native:2.12.0-cu130"
    )
    parser.add_argument("--master-addr", default="10.77.77.2")
    parser.add_argument("--master-port", type=int, default=29681)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("workload", type=Path)
    args = parser.parse_args()

    source = args.workload.read_bytes()
    local_command = docker_rank_command(
        args.image, 0, args.master_addr, args.master_port
    )
    remote_docker = docker_rank_command(
        args.image, 1, args.master_addr, args.master_port
    )
    remote_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        args.host,
        shlex.join(remote_docker),
    ]

    children: list[subprocess.Popen[bytes]] = []

    def stop_children() -> None:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 3
        for child in children:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    try:
        remote = subprocess.Popen(remote_command, stdin=subprocess.PIPE)
        children.append(remote)
        assert remote.stdin is not None
        remote.stdin.write(source)
        remote.stdin.close()

        local = subprocess.Popen(local_command, stdin=subprocess.PIPE)
        children.append(local)
        assert local.stdin is not None
        local.stdin.write(source)
        local.stdin.close()

        deadline = time.monotonic() + args.timeout
        while True:
            return_codes = [child.poll() for child in children]
            failed = next(
                (code for code in return_codes if code not in (None, 0)), None
            )
            if failed is not None:
                stop_children()
                return failed if failed > 0 else 128 - failed
            if all(code == 0 for code in return_codes):
                return 0
            if time.monotonic() >= deadline:
                print("native two-host launch timed out", file=sys.stderr)
                stop_children()
                return 124
            time.sleep(0.05)
    except BaseException:
        stop_children()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
