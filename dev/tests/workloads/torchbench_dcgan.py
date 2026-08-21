#!/usr/bin/env python3
"""Bounded TorchBench DCGAN training with optional model/optimizer AOT regions."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torchbenchmark.models.dcgan import Model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-optimizer", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    args = parser.parse_args()

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    benchmark = Model("train", "cuda", batch_size=args.batch_size)
    compile_options = {
        "dynamic": False,
        "mode": None if args.compile_mode == "default" else args.compile_mode,
    }
    if args.compile:
        benchmark.model = torch.compile(benchmark.model, **compile_options)
        benchmark.netG = torch.compile(benchmark.netG, **compile_options)
        if args.compile_mode == "reduce-overhead":
            def mark_graph_step(_module, _args):
                torch.compiler.cudagraph_mark_step_begin()

            benchmark.model.register_forward_pre_hook(mark_graph_step)
            benchmark.netG.register_forward_pre_hook(mark_graph_step)
    if args.compile_optimizer:
        if not args.compile:
            raise ValueError("--compile-optimizer requires --compile")
        benchmark.optimizerD.step = torch.compile(
            benchmark.optimizerD.step, **compile_options
        )
        benchmark.optimizerG.step = torch.compile(
            benchmark.optimizerG.step, **compile_options
        )

    for _ in range(args.warmup):
        benchmark.train()
    torch.cuda.synchronize()

    wall_samples = []
    gpu_samples = []
    for _ in range(args.iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        benchmark.train()
        end_event.record()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - started) * 1000 / 5)
        gpu_samples.append(start_event.elapsed_time(end_event) / 5)

    with torch.no_grad():
        checksum = sum(
            parameter.float().mean().item()
            for module in (benchmark.model, benchmark.netG)
            for parameter in module.parameters()
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "project": "torchbench",
                "model": "dcgan",
                "compiled": args.compile,
                "compiled_optimizer": args.compile_optimizer,
                "compile_mode": args.compile_mode if args.compile else None,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "inner_batches_per_iteration": 5,
                "median_gpu_ms_per_batch": statistics.median(gpu_samples),
                "median_wall_ms_per_batch": statistics.median(wall_samples),
                "checksum": checksum,
                "gpu_samples_ms_per_batch": gpu_samples,
                "wall_samples_ms_per_batch": wall_samples,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
