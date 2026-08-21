# Development

Nothing in this directory is required in a packaged client or GPU-host
deployment.

- `tests/`: GPU-free regressions and bounded native/remote workloads.
- `tools/`: build, audit, benchmark, safety-snapshot, and upstream-management
  commands.
- `containers/`: model, PyTorch, debugger, and compatibility test images.
- `patches/`: historical development patches, including the 41-round LUPINE
  integration trail. Canonical build input is now top-level `lupine/`.
- `external/`: local push-disabled public-source checkouts.
- `manifests/`, `config/`, `benchmarks/`, and `results/`: reproducibility and
  evidence.

Build all current product and validation images from the repository root with:

```bash
./dev/tools/build_images.sh
```
