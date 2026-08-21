"""Exercise lazy PyTorch CUDA kernels across local and remote GPU routes.

The local-first order is intentional: it catches route translation bugs where
CUDA library or module handles created on cuda:0 are incorrectly reused on the
remote cuda:1 route.
"""

import argparse
import time

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 64)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(64, 128)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        return self.output(self.encoder(hidden))


def train_steps(device: str, warmup: int, steps: int) -> tuple[float, float]:
    torch.manual_seed(20260820)
    model = TinyTransformer().to(device=device, dtype=torch.bfloat16)
    tokens = torch.arange(64, dtype=torch.long).reshape(4, 16).to(device)
    targets = torch.arange(1, 65, dtype=torch.long).remainder(128).reshape(4, 16)
    targets = targets.to(device)
    started = None
    for iteration in range(warmup + steps):
        if iteration == warmup:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten()
        )
        loss.backward()
    torch.cuda.synchronize(device)
    assert started is not None
    return loss.item(), (time.perf_counter() - started) / steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    assert torch.cuda.device_count() >= 2
    local_loss, local_seconds = train_steps("cuda:0", args.warmup, args.steps)
    remote_loss, remote_seconds = train_steps("cuda:1", args.warmup, args.steps)
    print(
        {
            "local_loss": local_loss,
            "remote_loss": remote_loss,
            "local_seconds_per_step": local_seconds,
            "remote_seconds_per_step": remote_seconds,
            "remote_over_local": remote_seconds / local_seconds,
        }
    )
    assert abs(local_loss - remote_loss) < 1e-3
