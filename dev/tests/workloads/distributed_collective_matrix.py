#!/usr/bin/env python3
"""Bounded two-rank coverage for the PyTorch NCCL collective surface."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main() -> int:
    rank = int(os.environ["RANK"])
    peer = 1 - rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    completed: list[str] = []
    try:
        reduced = torch.full((8,), float(rank + 1), device="cuda")
        dist.reduce(reduced, dst=0)
        if rank == 0:
            exact(reduced, torch.full_like(reduced, 3.0))
        completed.append("reduce")

        reduce_scatter_input = torch.full((8,), float(rank + 1), device="cuda")
        reduce_scatter_output = torch.empty(4, device="cuda")
        dist.reduce_scatter_tensor(reduce_scatter_output, reduce_scatter_input)
        exact(reduce_scatter_output, torch.full_like(reduce_scatter_output, 3.0))
        completed.append("reduce_scatter_tensor")

        all_gather_input = torch.full((4,), float(rank + 5), device="cuda")
        all_gather_output = torch.empty(8, device="cuda")
        dist.all_gather_into_tensor(all_gather_output, all_gather_input)
        exact(
            all_gather_output,
            torch.tensor([5.0] * 4 + [6.0] * 4, device="cuda"),
        )
        completed.append("all_gather_into_tensor")

        all_to_all_input = torch.tensor(
            [rank * 10.0] * 4 + [rank * 10.0 + 1.0] * 4, device="cuda"
        )
        all_to_all_output = torch.empty_like(all_to_all_input)
        dist.all_to_all_single(all_to_all_output, all_to_all_input)
        exact(
            all_to_all_output,
            torch.tensor([float(rank)] * 4 + [10.0 + rank] * 4, device="cuda"),
        )
        completed.append("all_to_all_single")

        gathered = [torch.empty(2, device="cuda") for _ in range(2)] if rank == 0 else None
        dist.gather(torch.full((2,), float(rank), device="cuda"), gathered, dst=0)
        if gathered is not None:
            exact(gathered[0], torch.zeros_like(gathered[0]))
            exact(gathered[1], torch.ones_like(gathered[1]))
        completed.append("gather")

        scattered = torch.empty(2, device="cuda")
        scatter_list = (
            [torch.full((2,), 20.0, device="cuda"), torch.full((2,), 21.0, device="cuda")]
            if rank == 0
            else None
        )
        dist.scatter(scattered, scatter_list, src=0)
        exact(scattered, torch.full_like(scattered, 20.0 + rank))
        completed.append("scatter")

        sent = torch.full((4,), 30.0 + rank, device="cuda")
        received = torch.empty_like(sent)
        operations = [
            dist.P2POp(dist.isend, sent, peer),
            dist.P2POp(dist.irecv, received, peer),
        ]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        exact(received, torch.full_like(received, 30.0 + peer))
        completed.append("batch_isend_irecv")

        averaged = torch.full((4,), float(rank + 1), device="cuda")
        dist.all_reduce(averaged, op=dist.ReduceOp.AVG)
        exact(averaged, torch.full_like(averaged, 1.5))
        completed.append("all_reduce_avg")

        dist.barrier()
        torch.cuda.synchronize()
        completed.append("barrier")
        print(json.dumps({"completed": completed, "rank": rank}), flush=True)
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
