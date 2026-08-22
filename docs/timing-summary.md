# Native and remote GPU timing summary

Last updated: 2026-08-21

“Native” is direct execution on the RTX 5090 in `ws-5090-1`. “Remote” is
execution from `ws-5090-2` on that same GPU through the current optimized
runtime. Dependency installation and image build time are excluded. Every
individual test is bounded below five minutes.

Sub-millisecond and low-single-millisecond tests are sensitive to GPU clocks,
so close results should be interpreted as ranges rather than absolute rankings.

## Qwen3.8 27B NInfer serving

This comparison uses the private `ali` serving setup without modifying it:
Qwen3.8 27B NVFP4 weights, INT8 KV cache, and three-token MTP. Native ran on
the local RTX 5090 in `ws-5090-2`; remote ran unchanged from that machine on
the RTX 5090 in `ws-5090-1` over direct 10 GbE. Both measured servers disabled
CUDA Graphs and prefix reuse. Five deterministic requests generated identical
token sequences, eliminating speculative-acceptance and cache noise.

| Profile | Metric | Native | Remote | Remote/native | Result |
|---|---|---:|---:|---:|---|
| 257-token prompt, 512-token generation | Decode throughput | 202.65 tok/s | 176.07 tok/s | 0.869× | Pass; host-control/RTT bound |
| Same | Time to first token | 53.90 ms | 58.80 ms | 1.091× | Pass |
| Same | Request wall time | 2.577 s | 2.960 s | 1.149× | Pass |
| 15,613-token prompt, 16-token generation | Prefill throughput to first token | 8,358.53 tok/s | 8,294.75 tok/s | 0.992× | Pass; near native |
| Same | Time to first token | 1.868 s | 1.882 s | 1.008× | Pass |
| Same | Request wall time | 1.934 s | 1.960 s | 1.013× | Pass |
| Model initialization | Model loaded | 3.614 s | 21.679 s | 5.999× | Pass; 19.73 GiB weights cross 10 GbE |

## Current performance-focused results

| Workload / profile | Timing scope | Native | Remote | Remote/native | Result |
|---|---|---:|---:|---:|---|
| nanochat GPT 4.5M, final eager rebuild | CUDA event | 9.128 ms | 97.979 ms | 10.73× | Pass; launch/control bound |
| nanochat GPT 4.5M, final compiled model | CUDA event | 1.022 ms | 2.195 ms | 2.15× | Pass |
| nanochat GPT 4.5M, final whole-step graph | CUDA event | 1.375 ms | 1.516 ms | 1.10× | Pass |
| LitGPT Pythia-14M, final eager rebuild | CUDA event | 9.269 ms | 67.778 ms | 7.31× | Pass; deferred for later performance triage |
| TorchTitan Qwen3 32M, final eager rebuild | CUDA event | 13.272 ms | 32.232 ms | 2.43× | Pass; FlexAttention |
| Llama 5.1M, whole eager training-step CUDA Graph | CUDA event | 1.408 ms | 1.469 ms | 1.04× | Pass; mixed `cuda:1` |
| nanochat GPT 4.5M, whole eager training-step CUDA Graph | CUDA event | 1.369 ms | 1.509 ms | 1.10× | Pass |
| nanochat GPT 4.5M, same profile | Synchronized wall | 1.378 ms | 1.701 ms | 1.23× | Pass; fixed sync cost dominates |
| LitGPT Pythia-14M, whole eager training-step CUDA Graph | CUDA event | 2.973 ms | 3.022 ms | 1.02× | Pass |
| TorchTitan Qwen3 32M, whole eager training-step CUDA Graph | CUDA event | 6.348 ms | 6.539 ms | 1.03× | Pass; FlexAttention |
| Mixed Llama eager, asynchronous cuBLAS | CUDA event | 7.662 ms | 7.985 ms | 1.04× | Pass; before whole-step capture |
| Mixed DETR eager FP32, asynchronous cuBLAS | CUDA event | 10.876 ms | 15.594 ms | 1.43× | Pass; CPU Hungarian fallback |
| Same-process cuBLASLt BF16 GEMM, 8192² | 10-operation synchronized mean | 4.486 ms | 4.671 ms | 1.04× | Pass; `--cublas-rpc` |
| Same-process cuBLASLt BF16 GEMM, 1024² | 50-operation synchronized mean | 0.019 ms | 0.572 ms | 30.70× | Pass; per-call RPC latency dominates |
| CUDA Graph replay, portable injected shim | Synchronized iteration | 0.811 ms | 0.902 ms | 1.11× | Pass |
| nanoGPT, optimized 30-step guardrail | Steady training step | 31.07 ms | 32.97 ms | 1.06× | Pass |
| TorchTitan Qwen3 32M, compiled model + optimizer | CUDA event | 4.622 ms | 4.606 ms | 1.00× | Pass; FlexAttention |
| TorchTitan Qwen3 32M, same profile | Synchronized wall | 4.634 ms | 4.908 ms | 1.06× | Pass |
| LitGPT Pythia-14M, compiled model + optimizer | CUDA event | 2.211 ms | 2.179 ms | 0.99× | Pass; clock-noisy sample |
| LitGPT Pythia-14M, same profile | Synchronized wall | 2.223 ms | 2.431 ms | 1.09× | Pass |
| nanochat GPT 4.5M, compiled model + optimizer | CUDA event | 0.825 ms | 0.990 ms | 1.20× | Pass |
| nanochat GPT 4.5M, same profile | Synchronized wall | 0.837 ms | 1.236 ms | 1.48× | Pass; sub-millisecond native step |
| Diffusers VAE, compiled model + optimizer | CUDA event | 4.640 ms | 5.020 ms | 1.08× | Pass |
| Diffusers VAE, same profile | Synchronized wall | 4.658 ms | 5.482 ms | 1.18× | Pass |
| DETR, compiled core + exact host-control regions | CUDA event | 4.449 ms | 5.924 ms | 1.33× | Pass |
| DETR, same profile | Synchronized wall | 4.477 ms | 6.361 ms | 1.42× | Pass |
| Mixtral BF16, compiled batched experts | CUDA event | 5.900 ms | 5.921 ms | 1.00× | Pass |
| Mixtral BF16, same profile | Synchronized wall | 5.916 ms | 6.363 ms | 1.08× | Pass |
| VideoMAE, compiled model + optimizer | CUDA event | 1.000 ms | 1.405 ms | 1.41× | Pass |
| VideoMAE, same profile | Per-step synchronized wall | 1.015 ms | 1.851 ms | 1.82× | Pass; synchronization dominated |
| TorchBench DCGAN, compile default | GPU time per batch | 3.669 ms | 4.792 ms | 1.31× | Pass |
| TorchBench DCGAN, same profile | Wall time per batch | 3.674 ms | 4.847 ms | 1.32× | Pass |
| Wav2Vec2, compiled model + optimizer | CUDA event | 3.108 ms | 2.568 ms | 0.83× | Pass; clock-noisy sample |
| Wav2Vec2, same profile | Synchronized wall | 3.146 ms | 2.994 ms | 0.95× | Pass; clock-noisy sample |
| FFT round trip | Synchronized iteration | 1.064 ms | 1.203 ms | 1.13× | Pass |
| ViT-B/16 training | Synchronized step | 11.235 ms | 12.928 ms | 1.15× | Pass |
| Cholesky | Synchronized operation | 0.846 ms | 1.073 ms | 1.27× | Pass |
| Sparse matrix multiply | Synchronized operation | 0.735 ms | 0.934 ms | 1.27× | Pass |
| Fused SDPA training | Synchronized step | 0.618 ms | 0.795 ms | 1.29× | Pass |
| ResNet-50 training | Synchronized step | 10.002 ms | 12.922 ms | 1.29× | Pass |
| 256 MiB H2D + D2H | Synchronized round trip | 9.286 ms | 580.758 ms | 62.54× | Pass; 10 GbE-bound |

## Distributed data-plane checkpoint

| Path / payload | Scope | Result | Effective link rate / time |
|---|---|---|---:|
| Native rank on each host, 1 MiB | NCCL all-reduce | Pass | 8.44 Gbit/s |
| Native rank on each host, 16 MiB | NCCL all-reduce | Pass | 9.73 Gbit/s |
| Native rank on each host, 64 MiB | NCCL all-reduce | Pass | 9.83 Gbit/s |
| Native rank on each host, tiny Transformer | Five DDP steps | Pass | 1.70 s measured rerun |
| Remote-native mixed rank, 1 MiB | NCCL all-reduce | Pass | 3.209 ms local rank; 1.638 ms remote rank |
| Native rank on each host, 1 MiB | Matched NCCL all-reduce | Pass | 1.899 ms local rank; 1.443 ms remote rank |
| Remote-native mixed rank, 16 MiB | NCCL all-reduce | Pass | 16.390 ms local rank; 14.682 ms remote rank |
| Native rank on each host, 16 MiB | Matched NCCL all-reduce | Pass | 15.144 ms local rank; 13.891 ms remote rank |
| Remote-native mixed rank, tiny Transformer | Five DDP steps, standard `torchrun` | Pass | 0.663–0.695 s across four clean runs |
| Native rank on each host, identical tiny Transformer | Five DDP steps | Pass | 0.372 s |
| Remote-native mixed rank | Nine-operation PyTorch collective matrix | Pass | 6.4 s complete process |

The remote-native rows replace the protocol-3 mapped-memory coherence path.
For 16 MiB collectives, effective throughput is approximately 8.2–9.1 Gbit/s
through the mixed path versus about 9.7 Gbit/s in matched native runs. Small payloads
still expose control latency, and the short Transformer timed region is 1.78×
to 1.87× native, so distributed compute/control overhead remains open.

The transfer test moves 512 MiB total and measured about 7.40 Gb/s over the
9.90 Gb/s application-level link. It is a residency constraint, not a launch
latency result.

## Functionality and breadth

| Suite / project | Current remote result | Native differential | Notes |
|---|---:|---:|---|
| Core PyTorch smoke | 8/8 pass | 8/8 pass | Allocation, copies, math, autograd, cuDNN, streams/events, compile, graphs |
| Curated upstream OpInfo | 82/82 pass | 82/82 pass | Includes `linalg.lu_factor`, SVD, masked ops, FFT, sparse, SDPA |
| Curated OpInfo, sample index 1 | 79 pass, 3 empty generators | Same | Exact native/remote status and error-category match |
| Curated OpInfo, dtype index 1 | 82/82 pass | In progress | BF16/complex128 gaps found and fixed; full rerun deferred while remote GPU is occupied |
| Broad upstream OpInfo, sample 0/dtype 0 | 628 pass, 6 native-matched exceptions | Same | 634 names; 25.350 s remote / 21.387 s native = 1.185x |
| Broad upstream OpInfo, sample 0/dtype 1 | 622 pass, 12 native-matched exceptions | Same | 23.965 s remote / 19.841 s native = 1.208x |
| Broad upstream OpInfo, sample 0/dtype 2 | 603 pass, 31 native-matched exceptions | Same | 30.816 s remote / 45.144 s native = 0.683x; JIT/cache/order noise, not an intrinsic speedup |
| Broad upstream OpInfo, sample 0/dtype 3 | 596 pass, 38 native-matched exceptions | Same | 34.888 s remote / 33.875 s native = 1.030x |
| Broad upstream OpInfo, sample 0/dtype 4 | 456 pass, 178 native-matched exceptions | Same | 32.558 s remote / 27.820 s native = 1.170x |
| Broad upstream OpInfo, sample 0/dtype 5 | 446 pass, 188 native-matched exceptions | Same | 26.178 s remote / 20.643 s native = 1.268x |
| Broad upstream OpInfo, sample 0/dtype 6 | 400 pass, 234 native-matched exceptions | Same | 19.185 s remote / 14.872 s native = 1.290x |
| Broad upstream OpInfo, sample 1/dtype 0 | 586 pass, 48 native-matched exceptions | Same | Four shards; no remote-specific failure |
| Broad upstream OpInfo, sample 2/dtype 0 | 556 pass, 78 native-matched exceptions | Same | 30.274 s remote / 24.513 s native = 1.235x |
| Broad upstream OpInfo, sample 3/dtype 0 | 470 pass, 164 native-matched exceptions | Same | 20.095 s remote / 15.847 s native = 1.268x; batched QR gap fixed |
| Broad upstream OpInfo, sample 4/dtype 0 | 441 pass, 193 native-matched exceptions | Same | 20.235 s remote / 15.828 s native = 1.278x; FP4/FP8 cuBLASLt pointer-mode gap fixed |
| Broad upstream OpInfo, sample 5/dtype 0 | 410 pass, 224 native-matched exceptions | Same | 16.656 s remote / 12.596 s native = 1.322x |
| Broad upstream OpInfo, sample 6/dtype 0 | 379 pass, 255 native-matched exceptions | Same | 19.829 s remote / 15.618 s native = 1.270x |
| Broad upstream OpInfo, sample 7/dtype 0 | 352 pass, 282 native-matched exceptions | Same | 19.729 s remote / 15.573 s native = 1.267x |
| Broad upstream OpInfo, sample 8/dtype 0 | 309 pass, 325 native-matched exceptions | Same | 11.148 s remote / 7.683 s native = 1.451x; batched Cholesky gap fixed; current broad-tier ratio target |
| Broad upstream OpInfo, sample 9/dtype 0 | 210 pass, 424 native-matched exceptions | Same | 10.859 s remote / 7.486 s native = 1.451x |
| Broad upstream OpInfo, sample 1/dtype 1 | 580 pass, 54 native-matched exceptions | Same | 24.947 s remote / 20.575 s native = 1.212x |
| Pinned memory semantics | Pass | Pass | Kernel `readinto`, async round trip, DataLoader pinning |
| NCCL single rank | Pass | Pass | Initialization, all-reduce, all-gather, destroy |
| NCCL mixed local/remote | Pass | Native two-host pass | Standard `torchrun`; nine collective families, DDP, clean teardown |
| NCCL mixed payload ladder | Pass | Native two-host pass | 1 MiB and 16 MiB exact; large payload near native link rate |
| Native two-host DDP fallback | Pass | Pass | Five-step Transformer; one unmodified native rank per workstation |
| DTensor + distributed checkpoint | Pass | Pass | Shard redistribution, filesystem save/load, NCCL metadata collective |
| FSDP2 training | Pass | Pass | World size one; exact matched losses across three AdamW steps |
| Spawned-process CUDA IPC | Pass | Pass | Shared CUDA tensor consumed correctly in child process |
| `torch.profiler` CUDA activities | No CUDA activities | 9 activities | CUPTI is not yet transported; execution itself succeeds |
| Hugging Face/Diffusers families | 16/16 pass | 16/16 pass | Synthetic forward/backward/AdamW; zero model/data downloads |
| Whole-step graph model matrix | 15/16 pass | Same graph-safe set | DETR intentionally falls back around CPU Hungarian matching |
| nanochat GPT | Pass | Pass | Actual upstream GQA/SDPA, sliding attention, value residual, smear/backout |
| LitGPT Pythia-14M | Pass | Pass | Actual upstream RoPE, fused SDPA, GELU MLP |
| TorchTitan Qwen3 32M | Pass | Pass | Actual upstream fused QKV, QK norm, packed causal FlexAttention |
| TorchBench DCGAN | Pass | Pass | Real upstream model and training method |
| TorchBench HF BERT | Pass | Pass | Real upstream model and training method |
| Latest modded-nanogpt kernel import | Pass | Pass | Current image reaches upstream first-run compile/warmup remotely; bounded run stops before the documented approximately seven-minute phase completes |
| Expandable-segments CUDA VMM | Pass | Pass | Six allocation sizes; mixed ordinals and clean process teardown |
| Unmodified PyTorch image injection | 8/8 pass | N/A | No workload-image rebuild; strict `nvidia-smi`, compile, graphs |
| Mixed local + remote enumeration | 2 devices | N/A | Local `cuda:0`, remote `cuda:1`; remote ordinal computes |
| Same-process mixed cuBLAS training | 3/3 dtypes pass | Exact local/remote loss match | `nn.Linear` forward/backward; float32, float16, bfloat16; opt-in |
| Error recovery | Pass | Native-compatible | Invalid device, OOM, post-error CUDA work |
| Link loss and cancellation | Pass | N/A | Bounded error, local/remote containers removed, lease released |

All 16 release-candidate model-family runs are in `dev/results/raw/` as
`release-hf-<model>.json`. They are deliberately short functionality checks;
the performance rows above use matched optimized profiles and longer measured
windows.

## Optimization outcomes

- Device-only `cuMemcpy2DAsync` now remains asynchronous across the RPC
  boundary instead of synchronizing the remote stream for every copy. This
  raised the long-prompt NInfer prefill result from approximately 7,244 to
  8,295 tok/s and brought it from 86.6% to 99.2% of native throughput.
- Repeated tensor-map descriptors are cached by their complete request and GPU
  route. In the profiled NInfer run this reduced server tensor-map encodes from
  6,720 calls to 228, while retaining bounded storage and successful-result-only
  caching.
- Forced RPC coalescing windows of 0 and 20 microseconds both regressed NInfer;
  the adaptive default remains selected. Deferred and immediate pipelining of
  validated cooperative extended launches also produced no material decode
  gain, so that added protocol complexity was rejected.
- Immutable metadata caches and correct invalidation removed the original
  high-rate pointer/function query bottleneck.
- Deferred asynchronous command coalescing remains enabled for small ordered
  calls. CUDA Graph launches dispatch immediately and do not wait for a return;
  `LUPINE_SYNC_GRAPH_LAUNCH=1` restores immediate-result semantics.
- Delaying graph launch by 100, 250, or 1000 microseconds worsened VideoMAE
  from 1.405 ms to 1.686, 2.002, and 2.224 ms respectively, so it was rejected.
- Bulk matcher-index transfer and tensorized finite/GIoU checks reduced DETR's
  remote GPU time from 11.262 ms to 5.924 ms.
- Transformers' supported batched Mixtral expert implementation removed 19
  GPU-to-host synchronizations per step and reached native GPU time.
- Compiling the actual TorchBench DCGAN model in default mode reduced remote
  GPU time from 37.656 ms to 4.792 ms.
- Compiling both the real TorchTitan Qwen3 model and AdamW reduced its remote
  step from 15.718 ms to 4.606 ms and reached matched native GPU time. The same
  treatment brought LitGPT to matched native GPU time and nanochat to 1.20×.
- PyTorch 2.12's default float32 FlexAttention autotuner failed identically
  natively and remotely on the RTX 5090 because every offered tile exceeded
  the per-SM shared-memory limit. Conservative upstream-supported forward and
  backward tiles make the workload pass in both modes.
- Correcting the server's capture mode from global to relaxed across RPC worker
  threads unlocks whole eager training-step capture. VideoMAE, both Diffusers
  models, Mixtral batched-MM, and 11 other Hugging Face families pass; DETR's
  CPU Hungarian matcher remains the only whole-step exclusion.

## Raw evidence

Key current artifacts include:

- `dev/results/raw/default-optimized-smoke.json`
- `dev/results/raw/portable-default-smoke.json`
- `dev/results/raw/portable-default-graph.json`
- `dev/results/raw/current-opinfo-all-shard-0.json` through shard 7
- `dev/results/raw/current-opinfo-six-native.json`
- `dev/results/raw/current-pinned-memory.json`
- `dev/results/raw/current-failure-semantics.json`
- `dev/results/raw/current-link-loss.json`
- `dev/results/raw/injected-native-image-smoke-v2.json`
- `dev/results/raw/injected-native-image-mixed-remote-compute.log`
- `dev/results/raw/detr-host-regions-v2.runner.json`
- `dev/results/raw/mixtral-batched-graph.runner.json`
- `dev/results/raw/videomae-async-graph-immediate.runner.json`
- `dev/results/raw/torchbench-dcgan-compile-default-current.runner.json`
- `dev/results/raw/release-nanochat-compiled-optimizer-v2.json`
- `dev/results/raw/release-litgpt-compiled-optimizer.json`
- `dev/results/raw/release-litgpt-compiled-optimizer-native.json`
- `dev/results/raw/release-torchtitan-qwen3-compiled-optimizer.json`
- `dev/results/raw/release-torchtitan-qwen3-compiled-optimizer-native.json`
- `dev/results/raw/release-dtensor-checkpoint.json`
- `dev/results/raw/release-cuda-ipc.json`
- `dev/results/raw/release-cuda-profiler-gap.json`
- `dev/results/raw/release-fsdp2.json`
