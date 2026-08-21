#!/usr/bin/env python3
"""Bounded probes for advanced PyTorch CUDA integration surfaces."""

from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile


def ipc_child(tensor_queue, result_queue) -> None:
    import torch

    tensor = tensor_queue.get(timeout=20)
    result_queue.put(float((tensor * 3).sum().cpu()))


def test_cuda_ipc() -> dict:
    import torch

    context = mp.get_context("spawn")
    tensor_queue = context.Queue()
    result_queue = context.Queue()
    tensor = torch.arange(1024, dtype=torch.float32, device="cuda")
    child = context.Process(target=ipc_child, args=(tensor_queue, result_queue))
    child.start()
    tensor_queue.put(tensor)
    value = result_queue.get(timeout=30)
    child.join(timeout=10)
    if child.is_alive():
        child.terminate()
        child.join(timeout=5)
        raise RuntimeError("CUDA IPC child did not exit")
    if child.exitcode != 0:
        raise RuntimeError(f"CUDA IPC child exited with {child.exitcode}")
    expected = float(3 * sum(range(1024)))
    if value != expected:
        raise AssertionError(f"CUDA IPC value {value} != {expected}")
    return {"child_exitcode": child.exitcode, "value": value}


def test_cuda_profiler() -> dict:
    import torch

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    left = torch.randn((512, 512), device="cuda")
    right = torch.randn((512, 512), device="cuda")
    with torch.profiler.profile(activities=activities) as profile:
        for _ in range(3):
            left = torch.relu(left @ right)
        torch.cuda.synchronize()
    events = profile.events()
    cuda_events = [event for event in events if event.device_type == torch.autograd.DeviceType.CUDA]
    if not cuda_events:
        raise AssertionError("profiler returned no CUDA events")
    return {
        "events": len(events),
        "cuda_events": len(cuda_events),
        "cuda_time_us": sum(event.self_device_time_total for event in cuda_events),
    }


def test_dtensor_checkpoint() -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed._tensor import distribute_tensor, init_device_mesh, Shard
    import torch.distributed.checkpoint as dcp

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29651")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    def stage(name: str) -> None:
        print(f"stage={name}", file=sys.stderr, flush=True)

    stage("init_process_group")
    dist.init_process_group("nccl")
    try:
        torch.cuda.set_device(0)
        stage("init_device_mesh")
        mesh = init_device_mesh("cuda", (1,))
        stage("create_dtensor")
        local = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
        distributed = distribute_tensor(local, mesh, [Shard(0)])
        stage("redistribute")
        transformed = (distributed.sin() + distributed.cos()).redistribute(
            mesh, placements=[Shard(1)]
        )
        stage("validate_dtensor")
        expected = local.sin() + local.cos()
        torch.testing.assert_close(transformed.to_local(), expected)

        state = {"model": {"weight": transformed}}
        with tempfile.TemporaryDirectory(prefix="rgpu-dcp-") as checkpoint_dir:
            stage("checkpoint_save")
            faulthandler.dump_traceback_later(15)
            try:
                dcp.save(state, checkpoint_id=checkpoint_dir)
            finally:
                faulthandler.cancel_dump_traceback_later()
            restored_local = torch.zeros_like(local)
            restored = distribute_tensor(restored_local, mesh, [Shard(1)])
            restored_state = {"model": {"weight": restored}}
            stage("checkpoint_load")
            dcp.load(restored_state, checkpoint_id=checkpoint_dir)
            stage("validate_checkpoint")
            torch.testing.assert_close(restored.to_local(), expected)
            files = sorted(path.name for path in Path(checkpoint_dir).iterdir())
        return {
            "backend": dist.get_backend(),
            "checkpoint_files": files,
            "mesh_size": mesh.size(),
            "sum": float(restored.to_local().sum().cpu()),
        }
    finally:
        dist.destroy_process_group()


def test_fsdp2() -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29652")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(1234)
        mesh = init_device_mesh("cuda", (1,))
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 32),
        ).cuda()
        fully_shard(model[0], mesh=mesh)
        fully_shard(model[2], mesh=mesh)
        fully_shard(model, mesh=mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        inputs = torch.randn((16, 64), device="cuda")
        targets = torch.randn((16, 32), device="cuda")
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not all(torch.isfinite(torch.tensor(losses))):
            raise AssertionError(f"non-finite FSDP2 losses: {losses}")
        return {
            "backend": dist.get_backend(),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "mesh_size": mesh.size(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    finally:
        dist.destroy_process_group()


CASES = {
    "cuda_ipc": test_cuda_ipc,
    "cuda_profiler": test_cuda_profiler,
    "dtensor_checkpoint": test_dtensor_checkpoint,
    "fsdp2": test_fsdp2,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    payload = {"case": args.case, "status": "pass", **CASES[args.case]()}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
