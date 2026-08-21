"""Verify that CPU syscalls can write into a PyTorch pinned-host tensor."""

import json

import torch


tensor = torch.empty(1024 * 1024, dtype=torch.uint8, pin_memory=True)
with open("/dev/zero", "rb", buffering=0) as source:
    bytes_read = source.readinto(tensor.numpy())

assert bytes_read == tensor.numel()
assert tensor.sum().item() == 0
print(json.dumps({"bytes_read": bytes_read, "status": "pass"}, sort_keys=True))
