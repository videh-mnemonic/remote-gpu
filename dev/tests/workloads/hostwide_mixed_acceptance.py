#!/usr/bin/env python3
"""Bounded ordinary-PyTorch acceptance test for host-wide local + remote GPUs."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record: dict[str, object] = {"tests": {}}
    started = time.monotonic()
    try:
        import torch

        library = Path(
            __import__("os").environ.get(
                "RGPU_HOSTWIDE_LIB", "/usr/local/lib/rgpu"
            )
        ) / "libcuda.so.1"
        route = ctypes.CDLL(str(library)).lupine_cuda_device_route_id
        route.argtypes = [ctypes.c_int]
        route.restype = ctypes.c_int
        routes = [route(index) for index in range(torch.cuda.device_count())]
        local = next(index for index, value in enumerate(routes) if value == -1)
        remote = next(index for index, value in enumerate(routes) if value >= 0)
        local_device = f"cuda:{local}"
        remote_device = f"cuda:{remote}"
        record["devices"] = {
            "count": torch.cuda.device_count(),
            "names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "routes": routes,
            "local": local,
            "remote": remote,
        }

        local_value = torch.arange(64, device=local_device).square().cpu()
        remote_value = torch.arange(64, device=remote_device).square().cpu()
        torch.testing.assert_close(remote_value, local_value)
        record["tests"]["local_remote_kernel"] = "pass"

        losses = {}
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = []
            for device in (local_device, remote_device):
                layer = torch.nn.Linear(64, 96, device="cpu", dtype=dtype).to(device)
                layer.weight.data.fill_(0.01)
                layer.bias.data.fill_(0.02)
                inputs = torch.ones((32, 64), device=device, dtype=dtype)
                loss = layer(inputs).float().square().mean()
                loss.backward()
                torch.cuda.synchronize(device)
                values.append(float(loss.detach()))
            if abs(values[0] - values[1]) > 1e-6:
                raise AssertionError(f"{dtype}: local={values[0]} remote={values[1]}")
            losses[str(dtype).removeprefix("torch.")] = values[0]
        record["tests"]["mixed_cublas_training"] = {
            "status": "pass",
            "losses": losses,
        }

        def operation(value):
            return torch.relu(value @ value.T + 0.5)

        compiled = torch.compile(operation)
        compiled_input = torch.randn(64, 64, device=remote_device)
        expected = operation(compiled_input)
        actual = compiled(compiled_input)
        actual = compiled(compiled_input)
        torch.testing.assert_close(actual, expected)
        record["tests"]["remote_torch_compile"] = "pass"

        host_input = torch.empty(1, dtype=torch.int32, pin_memory=True)
        host_output = torch.empty(1, dtype=torch.int32, pin_memory=True)
        device_input = torch.empty(1, dtype=torch.int32, device=remote_device)
        device_output = torch.empty_like(device_input)
        stream = torch.cuda.Stream(device=remote)
        stream.wait_stream(torch.cuda.current_stream(remote))
        with torch.cuda.stream(stream):
            device_input.copy_(host_input, non_blocking=True)
            device_output.copy_(device_input + 1)
            host_output.copy_(device_output, non_blocking=True)
        torch.cuda.current_stream(remote).wait_stream(stream)
        with torch.cuda.device(remote):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                device_input.copy_(host_input, non_blocking=True)
                device_output.copy_(device_input + 1)
                host_output.copy_(device_output, non_blocking=True)
        outputs = []
        for value in (21, 37, 91):
            host_input.fill_(value)
            host_output.fill_(-1)
            graph.replay()
            torch.cuda.synchronize(remote)
            outputs.append(int(host_output.item()))
        if outputs != [22, 38, 92]:
            raise AssertionError(f"dynamic graph outputs are stale: {outputs}")
        # A synchronization consumes pending graph-owned D2H copies. A later
        # unrelated synchronize must not replay the old copy into host memory.
        host_output.fill_(-123)
        torch.cuda.synchronize(remote)
        post_sync_host_value = int(host_output.item())
        if post_sync_host_value != -123:
            raise AssertionError(
                "unrelated synchronize rewrote completed graph output: "
                f"{post_sync_host_value}"
            )
        record["tests"]["dynamic_host_cuda_graph"] = {
            "status": "pass",
            "outputs": outputs,
            "post_sync_host_value": post_sync_host_value,
        }
        record["status"] = "pass"
        return finish(args.output, record, started, 0)
    except BaseException:
        record["status"] = "fail"
        record["traceback"] = traceback.format_exc()
        return finish(args.output, record, started, 1)


def finish(output: Path, record: dict, started: float, code: int) -> int:
    record["elapsed_seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": record["status"], "output": str(output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
