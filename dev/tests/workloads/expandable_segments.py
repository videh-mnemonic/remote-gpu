#!/usr/bin/env python3
"""Exercise PyTorch's CUDA virtual-memory allocator on one visible ordinal."""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if "expandable_segments:True" not in os.environ.get("PYTORCH_ALLOC_CONF", ""):
        parser.error("set PYTORCH_ALLOC_CONF=expandable_segments:True before Python starts")

    torch.cuda.set_device(args.device)
    values = []
    for elements in (1, 1024, 1 << 20, 3 << 20, 512, 2 << 20):
        tensor = torch.ones(elements, device=f"cuda:{args.device}")
        values.append(float(tensor.sum()))
        del tensor
    torch.cuda.synchronize()

    expected = [1.0, 1024.0, float(1 << 20), float(3 << 20), 512.0, float(2 << 20)]
    if values != expected:
        raise RuntimeError(f"allocator result mismatch: {values!r}")
    print(
        json.dumps(
            {
                "status": "pass",
                "device": args.device,
                "device_name": torch.cuda.get_device_name(args.device),
                "allocations": len(values),
                "values": values,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
