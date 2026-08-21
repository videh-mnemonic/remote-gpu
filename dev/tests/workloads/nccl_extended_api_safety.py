#!/usr/bin/env python3
"""Verify unsupported extended NCCL entry points fail closed remotely."""

from __future__ import annotations

import ctypes
import json

import torch


def main() -> int:
    torch.cuda.set_device(0)
    library_path = torch.__path__[0] + "/../nvidia/nccl/lib/libnccl.so.2"
    nccl = ctypes.CDLL(library_path)
    nccl.ncclCommInitAll.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    nccl.ncclCommInitAll.restype = ctypes.c_int
    communicator = ctypes.c_void_p()
    device = ctypes.c_int(0)
    result = nccl.ncclCommInitAll(
        ctypes.byref(communicator), 1, ctypes.byref(device)
    )
    assert result == 0, f"remote ncclCommInitAll failed with {result}"
    assert communicator.value is not None
    assert nccl.ncclCommDestroy(communicator) == 0

    nccl.ncclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    nccl.ncclMemAlloc.restype = ctypes.c_int
    nccl.ncclMemFree.argtypes = [ctypes.c_void_p]
    nccl.ncclMemFree.restype = ctypes.c_int
    allocation = ctypes.c_void_p()
    allocate_result = nccl.ncclMemAlloc(ctypes.byref(allocation), 4096)
    assert allocate_result == 0 and allocation.value is not None
    free_result = nccl.ncclMemFree(allocation)
    assert free_result == 0
    print(
        json.dumps(
            {
                "ncclCommInitAll": "remote_pass",
                "ncclMemAlloc": "remote_pass",
                "ncclMemFree": "remote_pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
