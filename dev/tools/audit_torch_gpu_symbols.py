#!/usr/bin/env python3
"""Compare PyTorch's linked GPU-library ABI with rgpu's interposed surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PREFIXES = ("cublas", "cuBLASLt", "cusolver", "cufft", "nccl")


def symbols(image: str, mode: str) -> set[str]:
    if mode == "needed":
        command = r'''
for library in /usr/local/lib/python*/site-packages/torch/lib/*.so; do
  nm -D --undefined-only "$library" 2>/dev/null
done
'''
        marker = " U "
    else:
        command = r'''
for library in /opt/rgpu/lib/libcublas_rpc.so \
               /opt/rgpu/lib/libcusolver_rpc.so \
               /opt/rgpu/lib/libcufft_rpc.so \
               /usr/local/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2; do
  nm -D --defined-only "$library" 2>/dev/null
done
'''
        marker = None
    completed = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", image, "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    found: set[str] = set()
    for line in completed.stdout.splitlines():
        if marker is not None and marker not in line:
            continue
        name = line.rsplit(maxsplit=1)[-1].split("@", maxsplit=1)[0]
        if name.startswith(PREFIXES):
            found.add(name)
    return found


def family(name: str) -> str:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return "cublasLt" if prefix == "cuBLASLt" else prefix
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", default="remote-gpu-pytorch-opinfo:nccl-rpc-v5"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    needed = symbols(args.image, "needed")
    interposed = symbols(args.image, "interposed")
    missing = needed - interposed
    families = {}
    for name in sorted(set(map(family, needed)) | set(map(family, interposed))):
        required = sorted(item for item in needed if family(item) == name)
        exported = sorted(item for item in interposed if family(item) == name)
        absent = sorted(set(required) - set(exported))
        families[name] = {
            "torch_referenced": len(required),
            "interposed": len(exported),
            "torch_referenced_interposed": len(set(required) & set(exported)),
            "not_interposed": len(absent),
            "not_interposed_symbols": absent,
        }
    report = json.dumps(
        {
            "image": args.image,
            "torch_referenced": len(needed),
            "interposed": len(interposed),
            "torch_referenced_interposed": len(needed & interposed),
            "not_interposed": len(missing),
            "families": families,
        },
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    else:
        print(report)


if __name__ == "__main__":
    main()
