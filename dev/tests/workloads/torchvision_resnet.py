#!/usr/bin/env python3
"""Bounded torchvision ResNet training compatibility workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torchvision.models import resnet50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(1234)
    device = torch.device("cuda:0")
    model = resnet50(weights=None).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(args.batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 1000, (args.batch_size,), device=device)

    step_seconds = []
    losses = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.monotonic() - started)
        losses.append(loss.detach().item())

    record = {
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "step_seconds": step_seconds,
        "losses": losses,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
