#!/usr/bin/env python3
"""Bounded CUDA subsystem and model performance matrix."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
from torchvision import models


def timed_iterations(fn, warmup: int, iterations: int) -> tuple[list[float], float]:
    for _ in range(warmup):
        value = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    if isinstance(value, torch.Tensor):
        checksum = float(value.detach().float().mean().cpu())
    else:
        checksum = float(value)
    return samples, checksum


def model_train(model_factory, batch_size: int, warmup: int, iterations: int):
    torch.manual_seed(1234)
    model = model_factory(weights=None).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    inputs = torch.randn(batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(0, 1000, (batch_size,), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    return timed_iterations(step, warmup, iterations)


def resnet50_train(warmup: int, iterations: int):
    return model_train(models.resnet50, 16, warmup, iterations)


def vit_b16_train(warmup: int, iterations: int):
    return model_train(models.vit_b_16, 8, warmup, iterations)


def sdpa_train(warmup: int, iterations: int):
    torch.manual_seed(1234)
    tensors = [
        torch.randn(8, 16, 1024, 64, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
        for _ in range(3)
    ]

    def step():
        for tensor in tensors:
            tensor.grad = None
        output = F.scaled_dot_product_attention(*tensors, is_causal=True)
        loss = output.float().square().mean()
        loss.backward()
        return loss

    return timed_iterations(step, warmup, iterations)


def fft_roundtrip(warmup: int, iterations: int):
    torch.manual_seed(1234)
    source = torch.randn(32, 1024, 1024, device="cuda")

    def step():
        spectrum = torch.fft.rfft2(source)
        return torch.fft.irfft2(spectrum, source.shape[-2:])

    return timed_iterations(step, warmup, iterations)


def linalg_cholesky(warmup: int, iterations: int):
    torch.manual_seed(1234)
    source = torch.randn(2048, 2048, device="cuda")
    matrix = source @ source.T
    matrix.diagonal().add_(2048.0)

    def step():
        return torch.linalg.cholesky(matrix)

    return timed_iterations(step, warmup, iterations)


def sparse_mm(warmup: int, iterations: int):
    torch.manual_seed(1234)
    size = 8192
    nonzero = 1_000_000
    indices = torch.randint(0, size, (2, nonzero), device="cuda")
    values = torch.randn(nonzero, device="cuda")
    sparse = torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
    dense = torch.randn(size, 256, device="cuda")

    def step():
        return torch.sparse.mm(sparse, dense)

    return timed_iterations(step, warmup, iterations)


def cuda_graph_replay(warmup: int, iterations: int):
    torch.manual_seed(1234)
    left = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    static = left
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(8):
            static = torch.relu(static @ right)

    def step():
        graph.replay()
        return static

    return timed_iterations(step, warmup, iterations)


def transfer_256mb(warmup: int, iterations: int):
    elements = 256 * 1024 * 1024 // 4
    host_source = torch.randn(elements, pin_memory=True)
    host_target = torch.empty(elements, pin_memory=True)
    device = torch.empty(elements, device="cuda")

    def roundtrip():
        device.copy_(host_source, non_blocking=True)
        host_target.copy_(device, non_blocking=True)
        return host_target[0]

    return timed_iterations(roundtrip, warmup, iterations)


CASES = {
    "resnet50_train": resnet50_train,
    "vit_b16_train": vit_b16_train,
    "sdpa_train": sdpa_train,
    "fft_roundtrip": fft_roundtrip,
    "linalg_cholesky": linalg_cholesky,
    "sparse_mm": sparse_mm,
    "cuda_graph_replay": cuda_graph_replay,
    "transfer_256mb": transfer_256mb,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    selected = args.cases.split(",")
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")

    record = {
        "schema_version": 1,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cases": {},
    }
    for name in selected:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            samples, checksum = CASES[name](args.warmup, args.iterations)
            ordered = sorted(samples)
            result = {
                "status": "pass",
                "samples_seconds": samples,
                "mean_seconds": sum(samples) / len(samples),
                "median_seconds": ordered[len(ordered) // 2],
                "checksum": checksum,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        except Exception as error:  # Preserve later cases after one unsupported op.
            result = {
                "status": "fail",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        result["elapsed_seconds"] = time.perf_counter() - started
        record["cases"][name] = result
        print(json.dumps({"case": name, **result}, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    failed = [name for name, result in record["cases"].items()
              if result["status"] != "pass"]
    print(json.dumps({"failed": failed, "output": str(args.output)}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
