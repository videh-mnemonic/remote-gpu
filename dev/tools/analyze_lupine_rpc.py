#!/usr/bin/env python3
"""Resolve LUPINE RPC statistics to CUDA operation names."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFINE = re.compile(r"^#define\s+((?:LUPINE_)?RPC_\w+)\s+(\d+)\s*$")


def operation_names(header: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in header.read_text(encoding="utf-8").splitlines():
        match = DEFINE.match(line)
        if match:
            names[int(match.group(2))] = match.group(1)
    return names


def statistics(path: Path) -> dict[int, tuple[int, int]]:
    values: dict[int, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        operation, count, wait_ns = (int(value) for value in line.split("\t"))
        values[operation] = (count, wait_ns)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path)
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("lupine/codegen/gen_api.h"),
    )
    parser.add_argument("--subtract", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    names = operation_names(args.header)
    current = statistics(args.stats)
    baseline = statistics(args.subtract) if args.subtract else {}
    rows = []
    for operation, (count, wait_ns) in current.items():
        base_count, base_wait = baseline.get(operation, (0, 0))
        rows.append((wait_ns - base_wait, count - base_count, operation))

    print(f"{'operation':48} {'count':>10} {'wait ms':>12} {'mean us':>12}")
    for wait_ns, count, operation in sorted(rows, reverse=True)[: args.limit]:
        mean_us = wait_ns / count / 1_000 if count else 0.0
        print(
            f"{names.get(operation, str(operation)):48} {count:10d} "
            f"{wait_ns / 1_000_000:12.3f} {mean_us:12.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
