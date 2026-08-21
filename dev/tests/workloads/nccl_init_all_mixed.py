#!/usr/bin/env python3
"""Validate single-process NCCL initialization across local and remote GPUs."""

from __future__ import annotations

import ctypes
import json
import threading

import torch


def main() -> int:
    assert torch.cuda.device_count() == 2
    torch.cuda.set_device(0)
    library_path = torch.__path__[0] + "/../nvidia/nccl/lib/libnccl.so.2"
    nccl = ctypes.CDLL(library_path)
    inputs = []
    outputs = []
    for rank in range(2):
        with torch.cuda.device(rank):
            inputs.append(torch.full((8,), float(rank + 1), device="cuda"))
            outputs.append(torch.empty_like(inputs[-1]))
    communicators = (ctypes.c_void_p * 2)()
    devices = (ctypes.c_int * 2)(0, 1)
    result = nccl.ncclCommInitAll(communicators, 2, devices)
    assert result == 0, f"mixed ncclCommInitAll failed with {result}"
    records = []
    try:
        for expected_rank, communicator in enumerate(communicators):
            handle = ctypes.c_void_p(communicator)
            count = ctypes.c_int()
            rank = ctypes.c_int()
            device = ctypes.c_int()
            assert nccl.ncclCommCount(handle, ctypes.byref(count)) == 0
            assert nccl.ncclCommUserRank(handle, ctypes.byref(rank)) == 0
            assert nccl.ncclCommCuDevice(handle, ctypes.byref(device)) == 0
            assert count.value == 2
            assert rank.value == expected_rank
            records.append(
                {"comm_device": device.value, "rank": rank.value, "world_size": count.value}
            )

        nccl.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        errors = []

        def all_reduce(rank: int) -> None:
            result = nccl.ncclAllReduce(
                ctypes.c_void_p(inputs[rank].data_ptr()),
                ctypes.c_void_p(outputs[rank].data_ptr()),
                inputs[rank].numel(),
                7,  # ncclFloat32
                0,  # ncclSum
                ctypes.c_void_p(communicators[rank]),
                None,
            )
            if result != 0:
                errors.append((rank, result))

        threads = [threading.Thread(target=all_reduce, args=(rank,)) for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        for rank in range(2):
            torch.cuda.synchronize(rank)
            torch.testing.assert_close(
                outputs[rank].cpu(), torch.full((8,), 3.0), rtol=0, atol=0
            )
    finally:
        for communicator in communicators:
            assert nccl.ncclCommDestroy(ctypes.c_void_p(communicator)) == 0
    print(
        json.dumps(
            {
                "all_reduce": "mixed_pass",
                "ncclCommInitAll": "mixed_pass",
                "ranks": records,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
