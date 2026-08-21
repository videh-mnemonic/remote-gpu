#!/usr/bin/env python3
"""Short native cross-host NCCL all-reduce throughput probe."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    try:
        results = []
        for size_bytes in (1 << 20, 16 << 20, 64 << 20):
            tensor = torch.ones(size_bytes // 4, device="cuda")
            for _ in range(3):
                dist.all_reduce(tensor)
                tensor.mul_(0.5)
            torch.cuda.synchronize()
            iterations = 10
            started = time.perf_counter()
            for _ in range(iterations):
                dist.all_reduce(tensor)
                tensor.mul_(0.5)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "bytes": size_bytes,
                    "iterations": iterations,
                    "mean_seconds": elapsed / iterations,
                    "payload_gbps": (size_bytes * 8 * iterations) / elapsed / 1e9,
                }
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": dist.get_backend(),
                        "collective": "all_reduce",
                        "results": results,
                        "world_size": dist.get_world_size(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
