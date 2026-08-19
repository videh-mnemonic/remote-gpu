# remote-gpu

Private test harness for comparing open-source CUDA-over-network projects on
`ws-5090-2` and `ws-5090-1`.

The initial OSS bake-off is complete. See
[the compatibility findings](docs/compatibility.md) for measured results.
LUPINE v1.0.0 is the only tested candidate that runs current PyTorch; it has a
large nanoGPT steady-state penalty and does not support the observed NCCL/DDP
path. Current LUPINE main has a PyTorch regression.

The repository objective remains evidence gathering before a new
implementation:

1. Establish short native PyTorch and nanoGPT baselines.
2. Screen LUPINE, Cricket, and GVirtuS with the same compatibility tests.
3. Run fixed-work and fixed-time nanoGPT comparisons for viable candidates.
4. Use selected upstream PyTorch tests and TorchBench models to measure
   compatibility breadth.
5. Choose what, if anything, needs to be built afterward.

See [the research and validation plan](docs/research-and-validation-plan.md).

## Safety and provenance

Public repositories are cloned only into the ignored `external/` directory.
Use `tools/upstreams.py`; it disables the clone's push URL and installs a
rejecting pre-push hook. Do not create public forks or push changes upstream.

Raw results and datasets are ignored. Small summaries, manifests, tests, and
private patch files belong in this repository.

## Fast start

From an unrestricted shell with GPU, Docker, and network access:

```bash
cp config/hosts.example.json config/hosts.json
python3 tools/doctor.py --config config/hosts.json --output results/raw/doctor.json
python3 tools/upstreams.py clone lupine
python3 tools/upstreams.py clone modded-nanogpt
python3 tests/smoke/pytorch_smoke.py --output results/raw/native-smoke.json
```

Every benchmark command is wrapped with `tools/run_timed.py`. The hard per-run
ceiling is 270 seconds, and the normal nanoGPT steady-state window is 180
seconds.
