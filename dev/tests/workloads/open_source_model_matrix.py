#!/usr/bin/env python3
"""Short synthetic training steps through real nanochat and LitGPT models."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for source_tree in ("nanochat", "litgpt", "torchtitan"):
    source_path = str(REPOSITORY_ROOT / "dev" / "external" / source_tree)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def build_nanochat():
    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=128,
        vocab_size=2048,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=256,
        window_pattern="SL",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    inputs = torch.randint(0, config.vocab_size, (4, config.sequence_len), device="cuda")
    targets = torch.randint(0, config.vocab_size, inputs.shape, device="cuda")

    def loss_fn(module):
        return module(inputs, targets=targets)

    return model, loss_fn, "nanochat GPT; GQA/SDPA; sliding attention; value residual; smear/backout"


def build_litgpt():
    from litgpt.model import GPT

    model = GPT.from_name("pythia-14m", block_size=128).cuda()
    model.apply(model._init_weights)
    inputs = torch.randint(0, model.config.vocab_size, (4, 128), device="cuda")
    targets = torch.randint(0, model.config.vocab_size, inputs.shape, device="cuda")

    def loss_fn(module):
        logits = module(inputs)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    return model, loss_fn, "LitGPT Pythia-14m; RoPE; fused SDPA; GELU MLP"


def build_torchtitan():
    from torch.nn.attention.flex_attention import and_masks, create_block_mask
    from torchtitan.models.common.attention import (
        get_causal_mask_mod,
        get_efficient_causal_mask_mod_for_packed_document,
    )
    from torchtitan.models.qwen3 import qwen3_configs

    config = qwen3_configs["debugmodel"]("flex")
    # PyTorch 2.12's generic float32 FlexAttention autotuner offers no kernel
    # that fits Blackwell's per-SM shared-memory limit for this head shape.
    # Pin the conservative forward/backward tiling supported by the same
    # upstream implementation. The identical settings are used natively.
    config.first_attention.inner_attention.kernel_options.update(
        {
            "fwd_BLOCK_M": 32,
            "fwd_BLOCK_N": 16,
            "fwd_num_stages": 1,
            "fwd_num_warps": 4,
            "bwd_BLOCK_M1": 16,
            "bwd_BLOCK_N1": 16,
            "bwd_BLOCK_M2": 16,
            "bwd_BLOCK_N2": 16,
            "bwd_num_stages": 1,
            "bwd_num_warps": 4,
        }
    )
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cuda")
    model.init_states(buffer_device=torch.device("cuda"))
    sequence_length = 128
    inputs = torch.randint(0, config.vocab_size, (2, sequence_length), device="cuda")
    targets = torch.randint(0, config.vocab_size, inputs.shape, device="cuda")
    positions = torch.arange(sequence_length, device="cuda").expand_as(inputs)
    mask_adapter = False
    try:
        attention_masks = model.get_attention_masks(positions)
    except TypeError as error:
        # TorchTitan currently passes a create_block_mask keyword introduced
        # after the PyTorch build used by this test image. Preserve its exact
        # causal + packed-document mask semantics while omitting only that
        # optional kernel scheduling hint. The native baseline uses this same
        # path, so this is not a remote-runtime workaround.
        if "separate_full_blocks" not in str(error):
            raise
        mask_adapter = True
        mask_mod = and_masks(
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        )
        attention_masks = create_block_mask(
            mask_mod,
            inputs.shape[0],
            None,
            sequence_length,
            sequence_length,
            device=positions.device,
            BLOCK_SIZE=config.first_attention.inner_attention.block_size,
        )

    def loss_fn(module):
        logits = module(inputs, positions=positions, attention_masks=attention_masks)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    feature = "TorchTitan Qwen3 debug model; FlexAttention; fused QKV; QK norm"
    if mask_adapter:
        feature += "; PyTorch create_block_mask compatibility adapter"
    return model, loss_fn, feature


BUILDERS = {
    "nanochat": build_nanochat,
    "litgpt": build_litgpt,
    "torchtitan": build_torchtitan,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(BUILDERS), required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA ordinal to train on (use 1 for the first remote GPU in mixed mode)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-optimizer", action="store_true")
    parser.add_argument(
        "--whole-step-graph",
        action="store_true",
        help="capture forward, backward, and optimizer into one CUDA Graph",
    )
    parser.add_argument("--reset-rpc-stats-after-warmup", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)
    model, loss_fn, feature = BUILDERS[args.model]()
    model.train()
    executed_model = (
        torch.compile(model, dynamic=False, mode="reduce-overhead")
        if args.compile
        else model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        foreach=True,
        capturable=args.whole_step_graph,
    )
    if args.compile_optimizer and not args.compile:
        parser.error("--compile-optimizer requires --compile")
    optimizer_step = (
        torch.compile(optimizer.step, dynamic=False, mode="reduce-overhead")
        if args.compile_optimizer
        else optimizer.step
    )

    samples = []
    losses = []
    whole_step_graph = None
    graph_loss = None
    process_started = time.perf_counter()
    for iteration in range(args.warmup + args.iterations):
        if args.whole_step_graph and iteration < args.warmup:
            continue
        if iteration == args.warmup and args.reset_rpc_stats_after_warmup:
            try:
                ctypes.CDLL("libcuda.so.1").lupine_rpc_stats_reset()
            except (OSError, AttributeError) as error:
                raise RuntimeError(
                    "the active CUDA library does not expose lupine_rpc_stats_reset"
                ) from error
        if iteration == args.warmup and args.whole_step_graph:
            optimizer.zero_grad(set_to_none=True)
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=False)
                    capture_loss = loss_fn(executed_model)
                    capture_loss.backward()
                    optimizer_step()
                    del capture_loss
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            gc.collect()
            optimizer.zero_grad(set_to_none=False)
            whole_step_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(whole_step_graph, stream=capture_stream):
                graph_loss = loss_fn(executed_model)
                graph_loss.backward()
                optimizer_step()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
        if whole_step_graph is None:
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        if whole_step_graph is not None:
            whole_step_graph.replay()
            loss = graph_loss
        else:
            loss = loss_fn(executed_model)
            loss.backward()
            optimizer_step()
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000
        gpu_ms = start.elapsed_time(end)
        if iteration >= args.warmup:
            samples.append({"gpu_ms": gpu_ms, "wall_ms": wall_ms})
            losses.append(float(loss.detach().cpu()))

    payload = {
        "project": args.model,
        "feature": feature,
        "status": "pass",
        "compiled": args.compile,
        "compiled_optimizer": args.compile_optimizer,
        "whole_step_graph": args.whole_step_graph,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "median_gpu_ms": statistics.median(sample["gpu_ms"] for sample in samples),
        "median_wall_ms": statistics.median(sample["wall_ms"] for sample in samples),
        "process_training_seconds": time.perf_counter() - process_started,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
