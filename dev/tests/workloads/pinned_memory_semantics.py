#!/usr/bin/env python3
"""Pinned staging-memory semantics, including kernel syscall writes."""

import json

import torch
from torch.utils.data import DataLoader, TensorDataset


elements = 1024 * 1024
source = torch.empty(elements, dtype=torch.uint8, pin_memory=True)
with open("/dev/zero", "rb", buffering=0) as zero:
    bytes_read = zero.readinto(source.numpy())
assert bytes_read == elements
source[::4096] = 7

device = source.to("cuda", non_blocking=True)
device.add_(3)
target = torch.empty_like(source, pin_memory=True)
target.copy_(device, non_blocking=True)
torch.cuda.synchronize()
assert target[0].item() == 10
assert target[1].item() == 3

dataset = TensorDataset(torch.arange(32, dtype=torch.float32).reshape(8, 4))
batch = next(iter(DataLoader(dataset, batch_size=4, pin_memory=True)))
assert batch[0].is_pinned()
assert float(batch[0].to("cuda", non_blocking=True).sum().cpu()) == 120.0

print(json.dumps({
    "bytes_read": bytes_read,
    "dataloader_pinned": batch[0].is_pinned(),
    "status": "pass",
}, sort_keys=True))
