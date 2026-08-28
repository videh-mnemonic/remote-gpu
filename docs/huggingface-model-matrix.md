# Hugging Face model training matrix

Test date: 2026-08-19

This matrix broadens application coverage beyond nanoGPT and PyTorch's own
tests. It constructs small random models directly from pinned upstream source,
then runs real forward, backward, loss, and AdamW steps on synthetic inputs.
It downloads no weights or datasets: every test uses 0 MB of data downloads.

- Transformers commit: `94f09cfec149050b5355bab7f207ac69e21f1a02`
- Diffusers commit: `4e0466f3e5260f0d78b5e2b68ffbf27d819cc6db`
- PyTorch: `2.12.0+cu130`
- Native: direct execution on the RTX 5090 in `gpu-host`
- Remote: execution from `client-host` through optimized LUPINE on that GPU
- Short sweep: two warmups and 30 measured steps, CUDA-event median
- Stable rerank: 10–20 warmups and 100–200 measured steps, three fresh
  processes per side; used to choose optimization targets

## Functionality and frozen-client sweep

All 16 native controls and all 16 frozen-client remote runs pass. The table is
the final short sweep. It is useful as a broad regression check, but sub-10 ms
models show large between-process GPU-clock variance; use the repeated table
below, not this single pass, to rank close performance cases.

| Model | PyTorch/CUDA surface | Parameters | Native GPU | Remote GPU | Ratio | Result |
|---|---|---:|---:|---:|---:|---|
| GPT-2 | Causal attention/loss, Conv1D projections | 4,240,896 | 4.807 ms | 5.160 ms | 1.07× | Pass |
| Llama | RoPE, grouped-query SDPA, RMSNorm, gated MLP | 5,114,112 | 5.469 ms | 8.429 ms | 1.54× | Pass |
| T5 | Encoder-decoder, cross-attention, relative bias | 4,985,344 | 10.052 ms | 12.711 ms | 1.26× | Pass |
| Mixtral | Sparse MoE routing, grouped-query attention | 7,312,128 | 7.982 ms | 13.511 ms | 1.69× | Pass; architectural triage |
| Mamba2 | State-space scan, causal convolution | 3,959,136 | 8.010 ms | 12.034 ms | 1.50× | Pass |
| ViT | Patch embedding, vision attention | 2,200,932 | 4.100 ms | 4.369 ms | 1.07× | Pass |
| ConvNeXT | Depthwise convolution, channels-last norm | 1,622,308 | 3.545 ms | 3.740 ms | 1.06× | Pass |
| Swin | Shifted-window attention, patch merging | 493,344 | 8.019 ms | 9.493 ms | 1.18× | Pass |
| SegFormer | Spatial-reduction attention, segmentation decoder | 1,060,108 | 7.109 ms | 10.164 ms | 1.43× | Pass |
| Wav2Vec2 | Raw-audio convolution, transformer encoder | 490,476 | 4.790 ms | 6.962 ms | 1.45× | Pass |
| Whisper | Log-Mel convolution, audio encoder-decoder | 3,207,360 | 6.438 ms | 10.054 ms | 1.56× | Pass |
| CLIP | Dual text/vision encoders, contrastive loss | 4,360,705 | 6.014 ms | 7.865 ms | 1.31× | Pass |
| DETR | CNN, encoder-decoder, Hungarian/GIoU losses | 862,639 | 7.686 ms | 16.611 ms | 2.16× | Pass; architectural triage |
| VideoMAE | Video tubelets, spatiotemporal attention | 969,236 | 2.518 ms | 4.722 ms | 1.88× | Pass; noisy short denominator |
| Diffusers U-Net | Residual convolution, spatial attention, resampling | 2,795,587 | 13.474 ms | 15.035 ms | 1.12× | Pass |
| Diffusers VAE | Encode, reparameterize, decode, spatial attention | 2,606,887 | 7.368 ms | 10.402 ms | 1.41× | Pass |

## Repeated performance rerank and optimization rounds

The optimization loop ranked by remote/native ratio, not absolute time. Each
round either removed a semantics-preserving RPC cost until that model was no
longer the slowest optimizable case, or recorded why the remaining cost cannot
be removed by the current CUDA-driver shim.

| Case / final profile | Metric | Native | Remote before | Remote after | Final ratio | Outcome |
|---|---|---:|---:|---:|---:|---|
| DETR, compiled core + host-control regions | CUDA event | 4.449 ms | 11.262 ms | 5.924 ms | 1.33× | Bulk matcher indices; cached criterion; exact tensorized finite/GIoU checks |
| Mixtral BF16, compiled batched experts | CUDA event | 5.900 ms | 7.334 ms grouped | 5.921 ms | 1.00× | Supported batched expert path removes 19 host synchronizations/step |
| VideoMAE, compiled model + optimizer | CUDA event | 1.000 ms | 1.953 ms | 1.405 ms | 1.41× | Immediate asynchronous CUDA Graph dispatch; three graph launches/step |
| VideoMAE, same profile | Synchronized wall | 1.015 ms | 2.387 ms | 1.851 ms | 1.82× | Per-step synchronization exposes irreducible RTT in this harness |
| Wav2Vec2, compiled model + optimizer | CUDA event | 3.108 ms | 6.282 ms | 2.568 ms | 0.83× | Pass; very small/noisy matched sample |
| ConvNeXT | CUDA event | 2.740 ms | 11.739 ms short | 3.589 ms | 1.31× | Driver/error metadata caches plus deferred RPC batching |
| SegFormer | CUDA event | 7.109 ms | 12.631 ms | 9.658 ms | 1.36× | Cached context limits; cooperative launch made ordered fire-and-forget |
| Mamba2 | CUDA event | 8.620 ms | 12.034 ms short | 10.537 ms | 1.22× | Long rerank cleared false short-sweep alarm |
| Diffusers VAE, compiled model + optimizer | CUDA event | 4.640 ms | 10.168 ms | 5.020 ms | 1.08× | Capture metadata cache plus compiled optimizer |
| Llama | CUDA event | 6.146 ms | 9.013 ms short | 7.688 ms | 1.25× | Long rerank cleared false short-sweep alarm |
| Whisper | CUDA event | 6.973 ms | 9.634 ms short | 9.323 ms | 1.34× | No new synchronous steady-state metadata hotspot |

The runtime now dispatches graph launches immediately without waiting for a
response. `LUPINE_SYNC_GRAPH_LAUNCH=1` is the compatibility fallback. Deferring
entire graph launches by 100, 250, or 1000 microseconds worsened VideoMAE to
1.686, 2.002, and 2.224 ms, so those policies were rejected. Full manual
VideoMAE training-step capture was also rejected because compiled convolution
backward is not capture-safe in this configuration; compiled subgraphs remain
the correct fallback.

## Correctness regressions

The final client/server pair passes the eight-part smoke suite, including
autograd, convolution, streams/events, `torch.compile`, and CUDA Graphs. It
also passes single-rank NCCL, pinned-memory semantics, and all 82 curated
OpInfos. A
temporary context-limit cache implementation exposed an exit-time destructor
ordering bug; it was fixed with process-lifetime cache storage before results
were accepted. A sender-wakeup optimization was rejected after nanoGPT
regressed from 34.82 to 37.36 ms. Reverting it produced a 32.97 ms final
nanoGPT guardrail.

## Reproduction

The workload is in
[`dev/tests/workloads/huggingface_model_matrix.py`](../dev/tests/workloads/huggingface_model_matrix.py).
Set `PYTHONPATH` to the mounted Transformers and Diffusers `src` directories,
then select a model with `--model`. `--cuda-events` records both synchronized
wall time and CUDA-event time. Network access is disabled for native runs and
Hugging Face offline mode is enabled for both paths.

Raw records and samples are under `dev/results/raw/` as
`huggingface-<model>-*.runner.json`. Frozen-client regressions end in
`-lupine-frozen-events.runner.json`; repeated long runs contain
`-long-events-` or the named optimization round.
