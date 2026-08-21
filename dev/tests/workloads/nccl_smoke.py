#!/usr/bin/env python3
"""Isolated single-rank NCCL compatibility probe."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> int:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([3.25], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    torch.cuda.synchronize()
    result = {
        "backend": dist.get_backend(),
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "device": local_rank,
        "value": value.item(),
        "all_gather": [tensor.item() for tensor in gathered],
    }
    dist.destroy_process_group()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
