#!/usr/bin/env python3
"""Focused, bounded NCCL all-to-all correctness probe."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    iterations = int(os.environ.get("RGPU_ITERATIONS", "3"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        for iteration in range(iterations):
            source = torch.tensor(
                [rank * 100.0 + iteration] * 4
                + [rank * 100.0 + iteration + 10.0] * 4,
                device="cuda",
            )
            output = torch.empty_like(source)
            dist.all_to_all_single(output, source)
            expected = torch.tensor(
                [iteration + rank * 10.0] * 4
                + [100.0 + iteration + rank * 10.0] * 4,
                device="cuda",
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        print(
            json.dumps(
                {"collective": "all_to_all_single", "iterations": iterations, "rank": rank}
            ),
            flush=True,
        )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
