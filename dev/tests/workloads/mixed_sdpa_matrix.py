"""Validate the mixed-mode SDPA fallback across common API variants."""

import torch
from torch.nn import functional as F


def execute(device: str, case: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.cuda.manual_seed_all(2026)
    query_heads = 4
    key_heads = 2 if case == "gqa" else 4
    base = torch.arange(query_heads * 8 * 16, dtype=torch.float32)
    query = (base.reshape(1, query_heads, 8, 16) / 1000).to(
        device=device, dtype=torch.bfloat16
    )
    key_base = torch.arange(key_heads * 8 * 16, dtype=torch.float32)
    key = (key_base.reshape(1, key_heads, 8, 16) / 900).to(
        device=device, dtype=torch.bfloat16
    )
    value = (key_base.flip(0).reshape(1, key_heads, 8, 16) / 800).to(
        device=device, dtype=torch.bfloat16
    )
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    kwargs = {}
    if case == "causal":
        kwargs["is_causal"] = True
    elif case == "bool_mask":
        kwargs["attn_mask"] = torch.ones((8, 8), dtype=torch.bool, device=device).tril()
    elif case == "additive_mask":
        mask = torch.zeros((8, 8), dtype=torch.bfloat16, device=device)
        mask[:, -1] = -10.0
        kwargs["attn_mask"] = mask
    elif case == "scale":
        kwargs["scale"] = 0.125
    elif case == "gqa":
        kwargs["enable_gqa"] = True
    elif case == "dropout":
        kwargs["dropout_p"] = 0.25
    output = F.scaled_dot_product_attention(query, key, value, **kwargs)
    output.float().square().mean().backward()
    torch.cuda.synchronize(device)
    return output.detach().cpu(), query.grad.detach().cpu()


if __name__ == "__main__":
    for name in (
        "plain", "causal", "bool_mask", "additive_mask", "scale", "gqa", "dropout"
    ):
        local_output, local_grad = execute("cuda:0", name)
        remote_output, remote_grad = execute("cuda:1", name)
        assert torch.isfinite(remote_output).all()
        assert torch.isfinite(remote_grad).all()
        if name != "dropout":
            # Native local SDPA may select FlashAttention while the remote
            # compatibility path uses composable matmuls. BF16 accumulation
            # order therefore differs by a few units in the last place.
            torch.testing.assert_close(
                remote_output, local_output, rtol=0.01, atol=0.004
            )
            torch.testing.assert_close(
                remote_grad, local_grad, rtol=0.02, atol=0.004
            )
        print({"case": name, "output_checksum": remote_output.float().sum().item()})
