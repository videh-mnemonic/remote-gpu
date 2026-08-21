# rgpu integration

This directory is the canonical LUPINE-derived transport source used by rgpu.
It began from LUPINE v1.0.0 and includes the compatibility, routing,
performance, NVML, graph, NCCL, and opaque CUDA-library extensions developed
for this project.

The original license and notices remain in this directory. Historical changes
are retained as a sequential patch series under `../dev/patches/`, but builds
consume this integrated source directly and do not recreate a patched external
worktree.
