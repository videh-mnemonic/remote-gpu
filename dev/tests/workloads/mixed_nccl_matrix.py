#!/usr/bin/env python3
"""Correctness and timing matrix for mixed local/virtualized-remote NCCL."""

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
    results = []
    try:
        configured_elements = os.environ.get("RGPU_NCCL_MATRIX_ELEMENTS")
        element_counts = (
            tuple(int(value) for value in configured_elements.split(","))
            if configured_elements
            else (1, 256 * 1024, 1024 * 1024)
        )
        for elements in element_counts:
            print(
                json.dumps({"rank": rank, "stage": "all_reduce", "elements": elements}),
                flush=True,
            )
            tensor = torch.full((elements,), float(rank + 1), device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            all_reduce_seconds = time.perf_counter() - started
            torch.testing.assert_close(
                tensor, torch.full_like(tensor, 3.0), rtol=0, atol=0
            )

            gathered = [torch.empty_like(tensor) for _ in range(2)]
            print(
                json.dumps({"rank": rank, "stage": "all_gather", "elements": elements}),
                flush=True,
            )
            started = time.perf_counter()
            dist.all_gather(gathered, tensor)
            torch.cuda.synchronize()
            all_gather_seconds = time.perf_counter() - started
            for output in gathered:
                torch.testing.assert_close(
                    output, torch.full_like(output, 3.0), rtol=0, atol=0
                )

            broadcast = torch.full(
                (elements,), 7.0 if rank == 0 else -1.0, device="cuda"
            )
            print(
                json.dumps({"rank": rank, "stage": "broadcast", "elements": elements}),
                flush=True,
            )
            dist.broadcast(broadcast, src=0)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                broadcast, torch.full_like(broadcast, 7.0), rtol=0, atol=0
            )
            results.append(
                {
                    "bytes": elements * tensor.element_size(),
                    "all_gather_seconds": all_gather_seconds,
                    "all_reduce_seconds": all_reduce_seconds,
                }
            )
        print(
            json.dumps(
                {"backend": dist.get_backend(), "rank": rank, "results": results},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
