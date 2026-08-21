#!/usr/bin/env python3
"""Exercise one-rank remote communicator management through NCCL RPC."""

from __future__ import annotations

import ctypes
import json

import torch


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


def main() -> int:
    torch.cuda.set_device(0)
    library_path = torch.__path__[0] + "/../nvidia/nccl/lib/libnccl.so.2"
    nccl = ctypes.CDLL(library_path)

    identifier = UniqueId()
    communicator = ctypes.c_void_p()
    assert nccl.ncclGetUniqueId(ctypes.byref(identifier)) == 0
    assert nccl.ncclCommInitRank(ctypes.byref(communicator), 1, identifier, 0) == 0
    assert communicator.value is not None
    try:
        ranks = ctypes.c_int()
        assert nccl.ncclCommCount(communicator, ctypes.byref(ranks)) == 0
        assert ranks.value == 1

        total_bytes = ctypes.c_uint64()
        assert nccl.ncclCommMemStats(communicator, 3, ctypes.byref(total_bytes)) == 0

        assert nccl.ncclCommSuspend(communicator, 1) == 0
        assert nccl.ncclCommResume(communicator) == 0

        allocation = ctypes.c_void_p()
        window = ctypes.c_void_p()
        user_pointer = ctypes.c_void_p()
        assert nccl.ncclMemAlloc(ctypes.byref(allocation), 4096) == 0
        assert nccl.ncclCommWindowRegister(
            communicator, allocation, 4096, ctypes.byref(window), 0
        ) == 0
        assert nccl.ncclWinGetUserPtr(
            communicator, window, ctypes.byref(user_pointer)
        ) == 0
        assert user_pointer.value == allocation.value
        assert nccl.ncclCommWindowDeregister(communicator, window) == 0
        assert nccl.ncclMemFree(allocation) == 0

        grow_identifier = UniqueId()
        assert nccl.ncclCommGetUniqueId(
            communicator, ctypes.byref(grow_identifier)
        ) == 0

        print(
            json.dumps(
                {
                    "comm_get_unique_id": "pass",
                    "comm_mem_stats": "pass",
                    "comm_suspend_resume": "pass",
                    "comm_window": "pass",
                    "comm_total_bytes": total_bytes.value,
                    "world_size": ranks.value,
                },
                sort_keys=True,
            )
        )
    finally:
        assert nccl.ncclCommFinalize(communicator) == 0
        assert nccl.ncclCommDestroy(communicator) == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
