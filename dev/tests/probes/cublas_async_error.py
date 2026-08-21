#!/usr/bin/env python3
"""Ensure deferred cuBLAS preserves immediate validation and queue health."""

import argparse
import ctypes
import json

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=1)
args = parser.parse_args()
torch.cuda.set_device(args.device)

process = ctypes.CDLL(None)
create = process.cublasCreate_v2
create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
create.restype = ctypes.c_int
destroy = process.cublasDestroy_v2
destroy.argtypes = [ctypes.c_void_p]
destroy.restype = ctypes.c_int
sgemm = process.cublasSgemm_v2
sgemm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
]
sgemm.restype = ctypes.c_int

handle = ctypes.c_void_p()
assert create(ctypes.byref(handle)) == 0
a = torch.eye(2, device=f"cuda:{args.device}", dtype=torch.float32)
b = torch.eye(2, device=f"cuda:{args.device}", dtype=torch.float32)
c = torch.zeros((2, 2), device=f"cuda:{args.device}", dtype=torch.float32)
alpha = ctypes.c_float(1)
beta = ctypes.c_float(0)
status = sgemm(
    handle,
    0,
    0,
    2,
    2,
    2,
    ctypes.byref(alpha),
    ctypes.c_void_p(a.data_ptr()),
    0,  # Deliberately invalid: lda must be at least 2.
    ctypes.c_void_p(b.data_ptr()),
    2,
    ctypes.byref(beta),
    ctypes.c_void_p(c.data_ptr()),
    2,
)
assert status == 7, f"expected CUBLAS_STATUS_INVALID_VALUE (7), got {status}"
assert destroy(handle) == 0

# A rejected call must not poison or reorder the stream's deferred queue.
result = a @ b
torch.cuda.synchronize(args.device)
assert torch.equal(result, a)
print(json.dumps({"status": "pass", "invalid_status": status}))
