#!/usr/bin/env python3
"""Run a native/remote OpInfo tier only after the shared GPU is stably idle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = PROJECT_ROOT / "dev/tests/workloads/pytorch_opinfo_smoke.py"
RESULTS = PROJECT_ROOT / "dev/results/raw"
LOCK_PATH = Path("/tmp/rgpu-opinfo-runner.lock")


def acquire_runner_lock(path: Path = LOCK_PATH):
    """Hold a process-scoped lock so two unattended GPU gates cannot overlap."""
    lock = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()}\n")
    lock.flush()
    return lock


def remote_compute_processes(host: str) -> list[str]:
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            host,
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def last_json_object(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "records" in value:
            return value
    raise RuntimeError("workload did not emit a final JSON result")


def save_result(
    name: str, completed: subprocess.CompletedProcess[str]
) -> dict | None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.stdout.log").write_text(completed.stdout)
    (RESULTS / f"{name}.stderr.log").write_text(completed.stderr)
    try:
        payload = last_json_object(completed.stdout)
    except RuntimeError:
        partial = []
        for line in completed.stderr.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("completed"), dict):
                partial.append(value["completed"])
        if partial:
            (RESULTS / f"{name}.partial.json").write_text(
                json.dumps({"complete": False, "records": partial}, indent=2, sort_keys=True)
                + "\n"
            )
        return None
    (RESULTS / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def workload_arguments(args: argparse.Namespace, device: int) -> list[str]:
    values = [
        "--shard", str(args.shard), "--shards", str(args.shards),
        "--device", str(device),
        "--sample-index", str(args.sample_index),
        "--dtype-index", str(args.dtype_index),
    ]
    if args.all:
        values.append("--all")
    if args.forward_only:
        values.append("--forward-only")
    if args.max_ops is not None:
        values.extend(["--max-ops", str(args.max_ops)])
    return values


def artifact_name(args: argparse.Namespace, side: str, round_number: int) -> str:
    scope = "all" if args.all else "curated"
    shard = f"-shard{args.shard}of{args.shards}" if args.shards > 1 else ""
    return (
        f"opinfo-{scope}-sample{args.sample_index}-dtype{args.dtype_index}-"
        f"{side}{shard}-round{round_number}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="mnemonic-1@10.77.77.1")
    parser.add_argument("--port", type=int, default=14837)
    parser.add_argument("--dtype-index", type=int, default=3)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--max-ops", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--round", type=int, default=30)
    parser.add_argument(
        "--reuse-native-round",
        type=int,
        help="reuse an existing native JSON artifact instead of rerunning it",
    )
    parser.add_argument(
        "--reuse-native-whole-round",
        type=int,
        help="derive this shard from a saved one-shard native artifact",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--wait-hours", type=float, default=10.0)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="deploy the server image before the run (normally pre-deployed)",
    )
    args = parser.parse_args()
    runner_lock = acquire_runner_lock()
    if runner_lock is None:
        print(f"runner-already-active: {LOCK_PATH}", flush=True)
        return 76
    if not 0 < args.poll_seconds or not 0 < args.stable_polls:
        parser.error("polling values must be positive")
    if args.sample_index < 0 or args.dtype_index < 0:
        parser.error("sample and dtype indexes must be nonnegative")
    if args.max_ops is not None and args.max_ops < 1:
        parser.error("--max-ops must be positive")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if args.reuse_native_round is not None and args.reuse_native_whole_round is not None:
        parser.error("choose only one native reuse mode")

    deadline = time.monotonic() + args.wait_hours * 3600
    idle_polls = 0
    while time.monotonic() < deadline:
        try:
            processes = remote_compute_processes(args.host)
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"admission-check-error: {exc}", flush=True)
            idle_polls = 0
        else:
            if processes:
                print(f"busy: {'; '.join(processes)}", flush=True)
                idle_polls = 0
            else:
                idle_polls += 1
                print(
                    f"idle-observation: {idle_polls}/{args.stable_polls}",
                    flush=True,
                )
                if idle_polls >= args.stable_polls:
                    break
        time.sleep(args.poll_seconds)
    else:
        print("idle-wait-expired", flush=True)
        return 75

    if args.reuse_native_whole_round is not None:
        whole_args = argparse.Namespace(**vars(args))
        whole_args.shard = 0
        whole_args.shards = 1
        native_path = RESULTS / (
            artifact_name(whole_args, "native", args.reuse_native_whole_round)
            + ".json"
        )
        native_payload = json.loads(native_path.read_text())
        native_payload["records"] = native_payload["records"][args.shard::args.shards]
        native_payload["shard"] = args.shard
        native_payload["shards"] = args.shards
        native_payload["passed"] = sum(
            item["status"] == "pass" for item in native_payload["records"]
        )
        native_payload["failed"] = sum(
            item["status"] == "fail" for item in native_payload["records"]
        )
        print(f"native-whole-reused: {native_path.name}", flush=True)
    elif args.reuse_native_round is not None:
        native_path = RESULTS / (
            artifact_name(args, "native", args.reuse_native_round) + ".json"
        )
        native_payload = json.loads(native_path.read_text())
        print(f"native-reused: {native_path.name}", flush=True)
    else:
        script = WORKLOAD.read_text()
        native = subprocess.run(
            [
                "timeout",
                "240",
                "ssh",
                args.host,
                "docker run --rm -i --gpus device=0 "
                "remote-gpu-pytorch-opinfo:native python3 - "
                + " ".join(workload_arguments(args, 0)),
            ],
            input=script,
            capture_output=True,
            text=True,
            timeout=255,
        )
        native_payload = save_result(
            artifact_name(args, "native", args.round), native
        )
        if native_payload is None:
            print(f"native-no-result: exit={native.returncode}", flush=True)
            return 2
        print(
            f"native-complete: passed={native_payload['passed']} "
            f"failed={native_payload['failed']} exit={native.returncode}",
            flush=True,
        )

    # Recheck immediately. rgpu performs its own admission check as well, so a
    # workload arriving in this gap causes a safe refusal.
    if remote_compute_processes(args.host):
        print("remote-became-busy-after-native", flush=True)
        return 75

    remote_command = [
            "timeout",
            "240",
            "rgpu",
            "run",
            "--host",
            f"{args.host}:{args.port}",
            "--image",
            "remote-gpu-pytorch-opinfo:nccl-rpc-v6",
            "--server-image",
            "remote-gpu-lupine-server:nccl-rpc-pytorch-v6",
            "--cublas-rpc",
            "--workspace",
            str(PROJECT_ROOT),
            "--",
            "python3",
            str(PROJECT_ROOT / "dev/tests/workloads/pytorch_opinfo_smoke.py"),
            *workload_arguments(args, 0),
        ]
    if args.deploy:
        remote_command.insert(remote_command.index("--cublas-rpc"), "--deploy")
    remote = subprocess.run(
        remote_command,
        capture_output=True,
        text=True,
        timeout=255,
    )
    remote_payload = save_result(
        artifact_name(args, "remote", args.round), remote
    )
    if remote_payload is None:
        print(f"remote-no-result: exit={remote.returncode}", flush=True)
        return 2
    print(
        f"remote-complete: passed={remote_payload['passed']} "
        f"failed={remote_payload['failed']} exit={remote.returncode}",
        flush=True,
    )
    native_status = {
        item["op"]: item["status"] for item in native_payload["records"]
    }
    remote_specific = [
        item["op"]
        for item in remote_payload["records"]
        if item["status"] != native_status.get(item["op"])
    ]
    native_seconds = sum(item.get("seconds", 0.0) for item in native_payload["records"])
    remote_seconds = sum(item.get("seconds", 0.0) for item in remote_payload["records"])
    comparison = {
        "dtype_index": args.dtype_index,
        "native_failed": native_payload["failed"],
        "native_passed": native_payload["passed"],
        "native_seconds_sum": native_seconds,
        "native_source_round": (
            args.reuse_native_whole_round or args.reuse_native_round or args.round
        ),
        "remote_failed": remote_payload["failed"],
        "remote_over_native": remote_seconds / native_seconds if native_seconds else None,
        "remote_passed": remote_payload["passed"],
        "remote_seconds_sum": remote_seconds,
        "remote_specific_status_differences": remote_specific,
        "round": args.round,
    }
    (RESULTS / (artifact_name(args, "comparison", args.round) + ".json")).write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(f"remote-specific-status-differences: {remote_specific}", flush=True)
    return 0 if not remote_specific else 1


if __name__ == "__main__":
    raise SystemExit(main())
