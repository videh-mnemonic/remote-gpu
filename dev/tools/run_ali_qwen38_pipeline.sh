#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ali_tree=""
host=""
image="rgpu/ninfer-qwen38-pipeline:dev"
shim_image="remote-gpu-client:0.2.1"
port=8112
context=1048576
prefill_chunk=1536
disable_cuda_graph=0
codex_mode=0
codex_greedy=0

while (($#)); do
  case "$1" in
    --ali)
      ali_tree=${2:?missing value for --ali}
      shift 2
      ;;
    --host)
      host=${2:?missing value for --host}
      shift 2
      ;;
    --image)
      image=${2:?missing value for --image}
      shift 2
      ;;
    --shim-image)
      shim_image=${2:?missing value for --shim-image}
      shift 2
      ;;
    --port)
      port=${2:?missing value for --port}
      shift 2
      ;;
    --context)
      context=${2:?missing value for --context}
      shift 2
      ;;
    --prefill-chunk)
      prefill_chunk=${2:?missing value for --prefill-chunk}
      shift 2
      ;;
    --no-cuda-graph)
      disable_cuda_graph=1
      shift
      ;;
    --codex)
      codex_mode=1
      shift
      ;;
    --codex-greedy)
      codex_mode=1
      codex_greedy=1
      shift
      ;;
    *)
      echo "usage: $0 --ali PATH --host USER@HOST [--image NAME] [--shim-image NAME] [--port N] [--no-cuda-graph] [--codex|--codex-greedy]" >&2
      exit 2
      ;;
  esac
done

model="$ali_tree/models/qwen3_8_27b_nvfp4.ninfer"
[[ -n "$host" && -f "$model" ]] || {
  echo "--ali must contain models/qwen3_8_27b_nvfp4.ninfer and --host is required" >&2
  exit 2
}

graph_args=()
if ((disable_cuda_graph)); then
  graph_args+=(--no-cuda-graph)
fi

generation_args=(--no-thinking --greedy)
default_max_tokens=32768
if ((codex_mode)); then
  # Keep thinking enabled while balancing strict tool syntax against repetitive
  # decoding. Very low temperatures repeated malformed calls; the registered
  # temperature-1 preset occasionally wandered or produced very long turns.
  generation_args=(--temperature 0.6 --top-p 0.95 --top-k 20)
  default_max_tokens=4096
fi
if ((codex_greedy)); then
  generation_args=(--greedy)
fi

exec env PYTHONPATH="$project_root/rgpu-client" python3 -m remote_gpu.cli run \
  --host "$host" \
  --reuse-attached \
  --include-local \
  --image "$image" \
  --shim-image "$shim_image" \
  --workspace "$ali_tree" \
  --env NINFER_PIPELINE_DEVICE=1 \
  --env LUPINE_PRELOAD_REGISTERED_KERNELS=0 \
  -- \
  ninfer-serve "$model" \
  --host 0.0.0.0 \
  --port "$port" \
  --model-id qwen3.8-27b-rgpu-pipeline \
  --max-context "$context" \
  --kv-capacity "$context" \
  --max-concurrency 1 \
  --prefill-chunk "$prefill_chunk" \
  --kv-dtype int8 \
  --default-max-tokens "$default_max_tokens" \
  "${graph_args[@]}" \
  --no-prefix-reuse \
  "${generation_args[@]}"
