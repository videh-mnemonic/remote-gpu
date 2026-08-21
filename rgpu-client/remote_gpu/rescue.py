"""Minimal local-only recovery entry point for a host-wide rgpu transaction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .hostwide import HostwideError, uninstall


GPU_LIBRARY_MARKERS = ("libcuda.so", "libnvidia-ml.so")


def loaded_gpu_libraries() -> list[str]:
    """Return CUDA/NVML mappings without loading either library."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise HostwideError("cannot audit rescue process library mappings")
    paths = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        if candidate.startswith("/") and any(
            marker in Path(candidate).name for marker in GPU_LIBRARY_MARKERS
        ):
            paths.add(candidate)
    return sorted(paths)


def recover(root: Path, *, refresh_loader: bool = True) -> dict[str, object]:
    root = root.resolve()
    if root == Path("/") and os.geteuid() != 0:
        raise HostwideError("live rescue requires sudo")
    loaded = loaded_gpu_libraries()
    if loaded:
        raise HostwideError(
            "rescue process already maps a GPU library; refusing recovery: "
            + ", ".join(loaded)
        )
    state = uninstall(
        root,
        allow_live=root == Path("/"),
        refresh_loader=refresh_loader,
    )
    loaded_after = loaded_gpu_libraries()
    if loaded_after:
        raise HostwideError(
            "GPU library unexpectedly loaded during recovery: "
            + ", ".join(loaded_after)
        )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rgpu-rescue",
        description=(
            "restore a host-wide rgpu journal locally without contacting a "
            "remote host or loading CUDA/NVML"
        ),
    )
    parser.add_argument("--root", default="/")
    args = parser.parse_args()
    try:
        state = recover(Path(args.root))
    except HostwideError as exc:
        parser.exit(1, f"rgpu-rescue: {exc}\n")
    print(
        json.dumps(
            {
                "status": "recovered",
                "root": state["root"],
                "remote_leases_not_contacted": state.get("leases", []),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
