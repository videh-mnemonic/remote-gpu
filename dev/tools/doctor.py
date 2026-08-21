#!/usr/bin/env python3
"""Collect read-only host, GPU, container, and direct-link diagnostics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import socket
import subprocess


CHECKS = {
    "hostname": ["hostname"],
    "kernel": ["uname", "-a"],
    "os_release": ["sed", "-n", "1,12p", "/etc/os-release"],
    "cpu": ["lscpu"],
    "memory": ["free", "-h"],
    "pci": ["lspci", "-nn"],
    "nvidia_smi": [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,pstate,temperature.gpu",
        "--format=csv,noheader",
    ],
    "nvidia_processes": [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader",
    ],
    "docker": ["docker", "info", "--format", "{{json .}}"],
    "network_addresses": ["ip", "-details", "-statistics", "address"],
}


def execute(command: list[str], timeout: float = 15.0) -> dict:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as error:
        return {"command": command, "exit_code": 127, "stdout": "", "stderr": str(error)}
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "exit_code": 124,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "timeout",
        }


def remote_execute(ssh_target: str, command: list[str]) -> dict:
    ssh_command = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        ssh_target,
        shlex.join(command),
    ]
    return execute(ssh_command, timeout=20)


def collect_host(interface: str, remote: str | None = None) -> dict:
    checks = dict(CHECKS)
    checks["interface"] = ["ethtool", interface]
    checks["interface_driver"] = ["ethtool", "-i", interface]
    checks["interface_features"] = ["ethtool", "-k", interface]
    checks["interface_mtu"] = ["cat", f"/sys/class/net/{interface}/mtu"]
    results = {}
    for name, command in checks.items():
        results[name] = remote_execute(remote, command) if remote else execute(command)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    remote_config = config["remote"]
    ssh_target = f"{remote_config['ssh_user']}@{remote_config['address']}"
    record = {
        "schema_version": 1,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collector_host": socket.gethostname(),
        "config": config,
        "local": collect_host(config["local"]["direct_interface"]),
        "remote": collect_host(remote_config["direct_interface"], ssh_target),
        "connectivity": {
            "route": execute(["ip", "route", "get", remote_config["address"]]),
            "ping": execute(["ping", "-c", "10", "-W", "1", remote_config["address"]], timeout=15),
            "ssh": remote_execute(ssh_target, ["true"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = []
    for location in ("local", "remote"):
        for name in ("nvidia_smi", "docker", "interface"):
            if record[location][name]["exit_code"] != 0:
                failures.append(f"{location}.{name}")
    for name in ("route", "ping", "ssh"):
        if record["connectivity"][name]["exit_code"] != 0:
            failures.append(f"connectivity.{name}")
    print(f"wrote {args.output}; failed checks: {', '.join(failures) if failures else 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

