#!/usr/bin/env python3
"""Bounded two-rank DDP training probe for native remote-rank execution."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


class TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4096, 512)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=4)
        self.projection = torch.nn.Linear(512, 4096, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(self.embedding(tokens)))


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(1234)
        model = TinyTransformer().cuda().to(torch.bfloat16)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        tokens = torch.randint(0, 4096, (8, 256), device="cuda")
        targets = torch.randint(0, 4096, (8, 256), device="cuda")

        iterations = 5
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses = []
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten()
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        checksum = torch.tensor(losses[-1], device="cuda")
        dist.all_reduce(checksum)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": dist.get_backend(),
                        "elapsed_seconds": elapsed,
                        "final_loss_sum": float(checksum.cpu()),
                        "iterations": iterations,
                        "model": "tiny_transformer_ddp",
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
