# Results

Commit small, reviewed JSON summaries here. Raw command output, traces, model
data, and temporary artifacts belong under `dev/results/raw/` or `dev/results/tmp/` and
are intentionally ignored.

Every result must identify the host, candidate, upstream commit, software
profile, command, timeout, start/end timestamps, exit status, and whether the
run was native, remote CUDA, or whole-process remote execution.

`LUPINE_RPC_STATS=/path/to/file.tsv` enables LUPINE v1's per-operation trace.
Resolve operation IDs with `dev/tools/analyze_lupine_rpc.py`; pass `--subtract` to
isolate a longer run from an otherwise identical shorter run.

The extended comparison uses `dev/tests/workloads/pytorch_matrix.py` for bounded
model and CUDA-subsystem coverage and `dev/tests/workloads/nccl_smoke.py` for the
isolated distributed compatibility gate. Aggregated values are recorded in
`dev/results/summary.json`; raw per-iteration samples remain ignored.

The broader upstream pass uses `dev/tests/workloads/pytorch_opinfo_smoke.py` to
sample PyTorch's own OpInfo-generated CUDA inputs in eight bounded shards on
both local and remote devices in one process. Official
TorchBench DCGAN and Hugging Face BERT runs use the locally cloned, push-guarded
commit recorded in `dev/manifests/resolved/torchbench.json`.

The same harness accepts `--sample-index` and `--dtype-index` for deterministic
coverage expansion, and `--describe` enumerates the selected upstream dtypes
without initializing CUDA. `opinfo-sample1-curated-*` records the exact 79-pass,
three-empty-generator second-sample differential; `opinfo-dtype1-*` contains
the first alternate-dtype failure-isolation and 82/82 final evidence.
`opinfo-dtype2-curated-remote-round25-final.json` records the second alternate
dtype at 80 passes with the same two harness dtype-exhaustion errors as native.
The fourth curated dtype tier has an exact native/remote differential in
`opinfo-dtype3-comparison-round31.json`: both sides pass 79 available cases and
report the same three exhausted-dtype selections, with no remote-specific
status difference. Complete 634-name differentials for sample 0/dtype indexes
0 through 6 are stored in four shards under rounds 36, 38/41, 42, 43, 44, and
45; sample indexes 1 through 9 at dtype 0 are under rounds 34/35, 46, 47/48,
49/50, 51, 52, 53, 54/55, and 56; sample 1/dtype 1 is round 57. Together these
are 10,778 exact native/remote status comparisons with no remote-specific
mismatch. `torch-gpu-symbol-coverage-round57.json` is the latest static
linked-library audit and records 203 of 364 PyTorch-referenced
symbols as interposed;
regenerate it with `dev/tools/audit_torch_gpu_symbols.py` after rebuilding images.
`dev/tools/run_idle_opinfo_tier.py` performs native/remote dtype and sample-tier
differentials only after three idle observations, records raw streams and both
JSON payloads, and reports remote-specific status differences without
bypassing rgpu's lease guard. It can cover the curated or complete upstream
operator list and deploys images only when explicitly requested.

`opinfo-all-sample1-dtype0-*-round32` records the complete second-sample
baseline and the first remote attempt. The native side executes 586 of 634
names; the initial remote process is intentionally retained as failure evidence
because one MAGMA batched-LU illegal access poisons its later CUDA context. The
focused round-32 cuSOLVER-backend regression passes, and round 33 is the clean
full rerun against the saved native artifact.
Because the monolithic remote second-sample process reached its 240-second
cap, round 34 splits it into four deterministic shards. Each invocation remains
below the five-minute policy while `--reuse-native-whole-round` derives exact
per-shard controls from the saved 634-record native result.

`dev/tests/workloads/pinned_memory_cpu_access.py` is the minimized regression for
CPU addressability of CUDA pinned-host allocations. The latest
modded-nanogpt single-process and RTX 5090 diagnostic changes are preserved in
`dev/patches/modded-nanogpt-latest-single-gpu.patch`; they are test adaptations,
not changes to a public fork.

`dev/tests/workloads/expandable_segments.py` is the mixed-process CUDA VMM
regression. Run it with `PYTORCH_ALLOC_CONF=expandable_segments:True`; the
local and remote JSON artifacts verify allocation, access, resizing patterns,
and clean teardown without a cross-route virtual-address collision.

The Hugging Face breadth pass uses
`dev/tests/workloads/huggingface_model_matrix.py` with exact locally cloned
Transformers and Diffusers commits. It creates random models and synthetic
inputs, downloads no weights or datasets, and now covers 16 model families.
The broad frozen-client sweep records two warmups plus 30 optimizer steps;
ranked performance cases use three longer 100–200-step processes per side.
`current-round36-hf-refresh.json` records the current-stack strict-remote
GPT-2, Mamba2, VideoMAE, and DETR refresh with one warmup and two measured
training steps per family.

`dev/tests/workloads/open_source_model_matrix.py` runs actual source from the
push-disabled nanochat, LitGPT, and TorchTitan clones with synthetic tokens and
no model or dataset download. It covers modern GQA/SDPA, RoPE, fused QKV, QK
normalization, packed causal FlexAttention, `torch.compile`, and compiled
optimizer steps. TorchTitan's two dependency-version adapters are applied
identically to its native and remote baselines.

`dev/results/raw/release-round7-smoke.json` is the promoted protocol-negotiated
client/server check: all eight core compile-and-CUDA-Graph cases pass. The
negative compatibility control uses the retained round-6 portable shim against
the round-7 server and must exit nonzero without leaving a remote lease.
