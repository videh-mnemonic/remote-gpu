"""Exercise local and remote cuBLAS/cuBLASLt from one PyTorch process."""

from __future__ import annotations

import json

import torch


def train_step(device: str, dtype: torch.dtype) -> float:
    layer = torch.nn.Linear(64, 96, device="cpu", dtype=dtype).to(device)
    layer.weight.data.fill_(0.01)
    layer.bias.data.fill_(0.02)
    inputs = torch.ones(
        (32, 64), device=device, dtype=dtype, requires_grad=True
    )
    loss = layer(inputs).float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    return float(loss.detach())


def main() -> None:
    results: dict[str, dict[str, float]] = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        name = str(dtype).removeprefix("torch.")
        local = train_step("cuda:0", dtype)
        remote = train_step("cuda:1", dtype)
        if abs(local - remote) > 1e-6:
            raise AssertionError(f"{name}: local={local} remote={remote}")
        results[name] = {"local_loss": local, "remote_loss": remote}
    print(json.dumps({"mixed_cublas_training": "pass", "results": results}))


if __name__ == "__main__":
    main()
