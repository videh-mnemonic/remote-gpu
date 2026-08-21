"""Diagnose local-first to remote CUDA behavior by PyTorch component."""

import argparse

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class EmbeddingCase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Embedding(128, 64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.module(tokens).float().square().mean()


class LayerNormCase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.LayerNorm(64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        inputs = self.module.weight.new_ones((4, 16, 64), requires_grad=True)
        return self.module(inputs).float().square().mean()


class FeedForwardCase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        inputs = self.module[0].weight.new_ones((4, 16, 64), requires_grad=True)
        return self.module(inputs).float().square().mean()


class AttentionCase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        inputs = self.module.in_proj_weight.new_ones((4, 16, 64), requires_grad=True)
        return self.module(inputs, inputs, inputs, need_weights=False)[0].float().square().mean()


class SdpaCase(nn.Module):
    backend = SDPBackend.MATH

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        inputs = self.anchor.new_ones((4, 4, 16, 16), requires_grad=True)
        with sdpa_kernel(self.backend):
            output = nn.functional.scaled_dot_product_attention(inputs, inputs, inputs)
        return output.float().square().mean()


class SdpaFlashCase(SdpaCase):
    backend = SDPBackend.FLASH_ATTENTION


class BmmCase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        left = self.anchor.new_ones((16, 16, 16), requires_grad=True)
        right = self.anchor.new_ones((16, 16, 16), requires_grad=True)
        return torch.bmm(left, right).float().square().mean()


class SdpaDecomposedCase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        inputs = self.anchor.new_ones((4, 4, 16, 16), requires_grad=True)
        scores = torch.matmul(inputs, inputs.transpose(-2, -1)) / 4.0
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, inputs).float().square().mean()


CASES = {
    "embedding": EmbeddingCase,
    "layernorm": LayerNormCase,
    "feedforward": FeedForwardCase,
    "attention": AttentionCase,
    "sdpa_math": SdpaCase,
    "sdpa_flash": SdpaFlashCase,
    "bmm": BmmCase,
    "sdpa_decomposed": SdpaDecomposedCase,
}


def run(case: str, device: str) -> float:
    torch.manual_seed(20260820)
    model = CASES[case]().to(device=device, dtype=torch.bfloat16)
    tokens = torch.arange(64, dtype=torch.long).reshape(4, 16).to(device)
    loss = model(tokens)
    loss.backward()
    torch.cuda.synchronize(device)
    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--single-thread-autograd", action="store_true")
    args = parser.parse_args()
    if args.single_thread_autograd:
        torch.autograd.set_multithreading_enabled(False)
    results = {device: run(args.case, device) for device in args.devices}
    print({"case": args.case, "results": results})
