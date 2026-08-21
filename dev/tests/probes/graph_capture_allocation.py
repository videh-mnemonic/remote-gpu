#!/usr/bin/env python3
"""Probe PyTorch caching-allocator behavior during CUDA Graph capture."""

import json

import torch


torch.manual_seed(7)
device = torch.device("cuda")
capture_stream = torch.cuda.Stream()
capture_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(capture_stream):
    for _ in range(3):
        warm = torch.empty((256, 768), dtype=torch.bfloat16, device=device)
        warm.fill_(2)
torch.cuda.current_stream().wait_stream(capture_stream)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=capture_stream):
    output = torch.empty((256, 768), dtype=torch.bfloat16, device=device)
    output.fill_(3)
torch.cuda.current_stream().wait_stream(capture_stream)
graph.replay()
torch.cuda.synchronize()
print(json.dumps({"status": "pass", "checksum": output.float().sum().item()}))
