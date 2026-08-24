#!/usr/bin/env python3
"""Plan a two-stage NInfer Qwen3.8 pipeline without loading model payloads.

The NInfer v2 directory is JSON immediately after its 16-byte prefix.  Reading
only that directory keeps this admission/memory probe CPU-only and independent
of PyTorch or NInfer's Python conversion environment.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


PREFIX = struct.Struct("<8sQ")
MAGIC = b"NINFER\x00\x02"
LAYER = re.compile(r"^text/layers/(\d+)/")


def read_directory(path: Path) -> dict[str, object]:
    with path.open("rb") as artifact:
        prefix = artifact.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ValueError("artifact is shorter than its v2 prefix")
        magic, directory_bytes = PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise ValueError(f"unsupported artifact magic: {magic!r}")
        raw = artifact.read(directory_bytes)
        if len(raw) != directory_bytes:
            raise ValueError("artifact directory is truncated")
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("objects"), list):
        raise ValueError("artifact directory has no object list")
    return value


def plan(path: Path, context: int, kv_group: int) -> dict[str, object]:
    directory = read_directory(path)
    objects = directory["objects"]
    layer_bytes: dict[int, int] = {}
    full_layers: set[int] = set()
    embedding_bytes = 0
    tail_bytes = 0

    for entry in objects:
        if not isinstance(entry, dict) or entry.get("kind") != "tensor":
            continue
        name = str(entry["name"])
        size = int(entry["bytes"])
        match = LAYER.match(name)
        if match:
            layer = int(match.group(1))
            layer_bytes[layer] = layer_bytes.get(layer, 0) + size
            if "/attention/" in name:
                full_layers.add(layer)
        elif name == "text/token_embedding":
            embedding_bytes += size
        elif name in ("text/final_norm", "text/output_head"):
            tail_bytes += size

    if not layer_bytes or sorted(layer_bytes) != list(range(max(layer_bytes) + 1)):
        raise ValueError("artifact text layers are missing or non-contiguous")
    if not full_layers:
        raise ValueError("artifact contains no full-attention layers")

    # Qwen3.8-27B target geometry. INT8 stores one code byte per K/V element
    # and one FP16 scale per quantization group.
    kv_heads = 4
    head_dim = 256
    kv_bytes_per_layer_token = 2 * (
        kv_heads * head_dim + kv_heads * (head_dim // kv_group) * 2
    )

    candidates = []
    total_layers = len(layer_bytes)
    for split in range(1, total_layers):
        stage0_weights = embedding_bytes + sum(layer_bytes[i] for i in range(split))
        stage1_weights = tail_bytes + sum(layer_bytes[i] for i in range(split, total_layers))
        stage0_full = sum(i < split for i in full_layers)
        stage1_full = len(full_layers) - stage0_full
        stage0_kv = context * kv_bytes_per_layer_token * stage0_full
        stage1_kv = context * kv_bytes_per_layer_token * stage1_full
        stage0_total = stage0_weights + stage0_kv
        stage1_total = stage1_weights + stage1_kv
        candidates.append(
            {
                "split_after_layer": split - 1,
                "stage0_layers": split,
                "stage1_layers": total_layers - split,
                "stage0_full_attention_layers": stage0_full,
                "stage1_full_attention_layers": stage1_full,
                "stage0_weight_bytes": stage0_weights,
                "stage1_weight_bytes": stage1_weights,
                "stage0_kv_bytes": stage0_kv,
                "stage1_kv_bytes": stage1_kv,
                "stage0_modeled_bytes": stage0_total,
                "stage1_modeled_bytes": stage1_total,
                "peak_modeled_bytes": max(stage0_total, stage1_total),
                "imbalance_bytes": abs(stage0_total - stage1_total),
            }
        )

    best = min(candidates, key=lambda item: (item["peak_modeled_bytes"], item["imbalance_bytes"]))
    performance_default = next(
        item for item in candidates if item["stage0_layers"] == total_layers // 2
    )
    return {
        "artifact": str(path),
        "identity": directory.get("identity"),
        "context_tokens": context,
        "kv_dtype": "int8",
        "kv_quant_group": kv_group,
        "kv_bytes_per_full_attention_layer_token": kv_bytes_per_layer_token,
        "full_attention_layers": sorted(full_layers),
        "modeled_features": ["text", "no-speculation", "no-vision"],
        "excluded_from_modeled_bytes": [
            "allocator alignment",
            "workspaces",
            "round state",
            "CUDA graphs",
            "MTP/draft weights and KV",
            "vision weights",
        ],
        "recommended": best,
        "runtime_performance_default": performance_default,
        "candidates": sorted(
            candidates, key=lambda item: (item["peak_modeled_bytes"], item["imbalance_bytes"])
        )[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--context", type=int, default=1_048_576)
    parser.add_argument("--kv-group", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.context <= 0 or args.kv_group <= 0 or 256 % args.kv_group:
        parser.error("context must be positive and kv-group must divide the 256-wide KV head")
    result = plan(args.artifact, args.context, args.kv_group)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    chosen = result["recommended"]
    runtime = result["runtime_performance_default"]
    gib = 1024**3
    print(f"model: {result['identity']}")
    print(f"context: {result['context_tokens']:,} tokens")
    print(
        "recommended: "
        f"layers {chosen['stage0_layers']} + {chosen['stage1_layers']} "
        f"(split after layer {chosen['split_after_layer']})"
    )
    print(
        "modeled weights + KV: "
        f"{chosen['stage0_modeled_bytes'] / gib:.2f} GiB + "
        f"{chosen['stage1_modeled_bytes'] / gib:.2f} GiB"
    )
    print(
        "runtime default: "
        f"layers {runtime['stage0_layers']} + {runtime['stage1_layers']} "
        f"for balanced compute ({runtime['stage0_modeled_bytes'] / gib:.2f} GiB + "
        f"{runtime['stage1_modeled_bytes'] / gib:.2f} GiB modeled)"
    )
    print("warning: modeled totals exclude runtime/workspace/graph overhead")


if __name__ == "__main__":
    main()
