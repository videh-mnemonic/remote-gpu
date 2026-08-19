#!/usr/bin/env python3
"""Short CUDA/PyTorch compatibility smoke suite for native or remoted devices."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import socket
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true", dest="test_compile")
    parser.add_argument("--graphs", action="store_true", dest="test_graphs")
    args = parser.parse_args()

    record = {
        "schema_version": 1,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "tests": {},
    }
    try:
        import torch
        import torch.nn.functional as functional
    except BaseException:
        record["import_error"] = traceback.format_exc()
        return finish(args.output, record, 1)

    record["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        record["fatal_error"] = "torch.cuda.is_available() returned false"
        return finish(args.output, record, 1)

    device = torch.device("cuda:0")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    def run_test(name, function) -> None:
        started = time.monotonic()
        try:
            details = function()
            torch.cuda.synchronize(device)
            record["tests"][name] = {
                "status": "pass",
                "elapsed_seconds": time.monotonic() - started,
                "details": details,
            }
        except BaseException:
            record["tests"][name] = {
                "status": "fail",
                "elapsed_seconds": time.monotonic() - started,
                "traceback": traceback.format_exc(),
            }

    def identity():
        properties = torch.cuda.get_device_properties(device)
        return {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
            "device_count": torch.cuda.device_count(),
        }

    def copy_roundtrip():
        host = torch.arange(4096, dtype=torch.int64)
        returned = host.to(device).cpu()
        assert torch.equal(host, returned)
        return {"bytes": host.numel() * host.element_size()}

    def matmul():
        left = torch.randn(128, 128, dtype=torch.float32)
        right = torch.randn(128, 128, dtype=torch.float32)
        expected = left @ right
        actual = (left.to(device) @ right.to(device)).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        return {"shape": [128, 128]}

    def autograd():
        value = torch.randn(1024, device=device, requires_grad=True)
        loss = value.sin().square().mean()
        loss.backward()
        assert value.grad is not None
        assert torch.isfinite(value.grad).all().item()
        return {"loss": loss.detach().item()}

    def convolution():
        inputs = torch.randn(2, 8, 16, 16, device=device)
        weights = torch.randn(16, 8, 3, 3, device=device)
        output = functional.conv2d(inputs, weights, padding=1)
        assert output.shape == (2, 16, 16, 16)
        assert torch.isfinite(output).all().item()
        return {"shape": list(output.shape), "cudnn_available": torch.backends.cudnn.is_available()}

    def streams_and_events():
        stream = torch.cuda.Stream(device=device)
        event = torch.cuda.Event()
        value = torch.zeros(1024, device=device)
        with torch.cuda.stream(stream):
            value.add_(7)
            event.record(stream)
        torch.cuda.current_stream(device).wait_event(event)
        assert value.sum().item() == 7168
        return {"event_query": event.query()}

    def compiled():
        def operation(value):
            return torch.relu(value @ value.T + 0.5)

        compiled_operation = torch.compile(operation)
        value = torch.randn(64, 64, device=device)
        expected = operation(value)
        actual = compiled_operation(value)
        actual = compiled_operation(value)
        torch.testing.assert_close(actual, expected)
        return {"backend": "default", "shape": list(actual.shape)}

    def cuda_graph():
        static_input = torch.randn(1024, device=device)
        static_output = torch.empty_like(static_input)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                static_output.copy_(static_input * 2 + 1)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output.copy_(static_input * 2 + 1)
        static_input.copy_(torch.arange(1024, device=device, dtype=static_input.dtype))
        graph.replay()
        expected = static_input * 2 + 1
        torch.testing.assert_close(static_output, expected)
        return {"elements": static_input.numel()}

    run_test("identity", identity)
    run_test("copy_roundtrip", copy_roundtrip)
    run_test("matmul", matmul)
    run_test("autograd", autograd)
    run_test("convolution", convolution)
    run_test("streams_and_events", streams_and_events)
    if args.test_compile:
        run_test("torch_compile", compiled)
    if args.test_graphs:
        run_test("cuda_graph", cuda_graph)

    torch.cuda.empty_cache()
    failed = [name for name, result in record["tests"].items() if result["status"] != "pass"]
    record["summary"] = {
        "passed": len(record["tests"]) - len(failed),
        "failed": len(failed),
        "failed_tests": failed,
    }
    return finish(args.output, record, int(bool(failed)))


def finish(output: Path, record: dict, exit_code: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = record.get("summary", {})
    print(json.dumps({"output": str(output), "exit_code": exit_code, **summary}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

