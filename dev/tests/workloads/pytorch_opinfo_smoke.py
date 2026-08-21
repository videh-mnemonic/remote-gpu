"""Sample a broad cross-section of PyTorch's upstream OpInfo database on CUDA."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import sys
import time

import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_flatten


# Deliberately span pointwise, BLAS, reductions, indexing, NN, attention, FFT,
# linalg, masked, and special-function families. Each case uses upstream's own
# first CUDA SampleInput rather than a locally invented input.
WANTED = (
    "abs", "acos", "add", "addmm", "baddbmm", "bmm", "cdist", "clamp",
    "complex", "corrcoef", "cross", "cumprod", "diff", "einsum", "erf",
    "exp", "fft.fft2", "fft.irfft", "gather", "gradient", "index_add",
    "index_reduce", "logsumexp", "masked.logsumexp", "masked.softmax",
    "masked.sum", "matmul", "matrix_exp", "nan_to_num", "nn.functional.adaptive_avg_pool2d",
    "nn.functional.avg_pool2d", "nn.functional.binary_cross_entropy_with_logits",
    "nn.functional.conv1d", "nn.functional.conv2d", "nn.functional.conv3d",
    "nn.functional.cross_entropy", "nn.functional.dropout", "nn.functional.embedding",
    "nn.functional.gelu", "nn.functional.grid_sample", "nn.functional.group_norm",
    "nn.functional.interpolate", "nn.functional.layer_norm", "nn.functional.linear",
    "nn.functional.max_pool2d", "nn.functional.mse_loss", "nn.functional.normalize",
    "nn.functional.pad", "nn.functional.rms_norm", "nn.functional.scaled_dot_product_attention",
    "nn.functional.silu", "nn.functional.softmax", "nn.functional.unfold",
    "nonzero", "quantile", "scatter_add", "scatter_reduce", "sort", "special.bessel_j0",
    "special.erfcx", "special.ndtr", "split", "std_mean", "stft", "take_along_dim",
    "topk", "to_sparse", "unique", "var_mean", "where", "xlogy",
    "linalg.cholesky", "linalg.det", "linalg.eigh", "linalg.inv", "linalg.lstsq",
    "linalg.lu_factor", "linalg.matrix_norm", "linalg.pinv", "linalg.qr",
    "linalg.solve", "linalg.svd", "linalg.vector_norm",
)


def key(op) -> str:
    return op.name + (("." + op.variant_test_name) if op.variant_test_name else "")


def ordered_cuda_dtypes(op) -> list[torch.dtype]:
    dtypes = sorted(op.supported_dtypes("cuda"), key=str)
    if torch.float32 in dtypes:
        dtypes.remove(torch.float32)
        dtypes.insert(0, torch.float32)
    return dtypes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA ordinal (use 1 for the first remote GPU in mixed mode)",
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--all",
        action="store_true",
        help="sample every unique upstream OpInfo name instead of WANTED",
    )
    parser.add_argument(
        "--max-ops",
        type=int,
        help="cap the deterministic candidate list before sharding",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="zero-based upstream SampleInput index to execute",
    )
    parser.add_argument(
        "--dtype-index",
        type=int,
        default=0,
        help="zero-based supported dtype index (float32 is preferred at index 0)",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="skip autograd to isolate forward-only compatibility",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print candidate names and CUDA dtypes without initializing CUDA",
    )
    args = parser.parse_args()

    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if args.max_ops is not None and args.max_ops < 1:
        parser.error("--max-ops must be positive")
    if args.sample_index < 0 or args.dtype_index < 0:
        parser.error("--sample-index and --dtype-index must be nonnegative")
    candidates = []
    seen = set()
    for op in op_db:
        if (args.all or op.name in WANTED) and op.name not in seen:
            seen.add(op.name)
            candidates.append(op)
    if args.only:
        candidates = [op for op in candidates if op.name in args.only]
    if args.max_ops is not None:
        candidates = candidates[:args.max_ops]
    candidates = [op for i, op in enumerate(candidates) if i % args.shards == args.shard]

    if args.describe:
        print(
            json.dumps(
                {
                    "scope": "all" if args.all else "curated",
                    "shard": args.shard,
                    "shards": args.shards,
                    "dtype_index": args.dtype_index,
                    "sample_index": args.sample_index,
                    "operators": [
                        {
                            "op": key(op),
                            "dtypes": [str(dtype) for dtype in ordered_cuda_dtypes(op)],
                            "selected_dtype": (
                                str(ordered_cuda_dtypes(op)[args.dtype_index])
                                if args.dtype_index < len(ordered_cuda_dtypes(op))
                                else None
                            ),
                        }
                        for op in candidates
                    ],
                },
                sort_keys=True,
            )
        )
        return 0

    torch.cuda.set_device(args.device)

    records = []
    for op in candidates:
        started = time.perf_counter()
        record = {"op": key(op), "status": "pass", "backward": False}
        print(json.dumps({"starting": key(op)}), file=sys.stderr, flush=True)
        try:
            dtype = ordered_cuda_dtypes(op)[args.dtype_index]
            samples = op.sample_inputs(
                "cuda", dtype, requires_grad=dtype.is_floating_point
            )
            sample = next(islice(samples, args.sample_index, None))
            record["dtype"] = str(dtype)
            record["sample_index"] = args.sample_index
            sample_tensors, _ = tree_flatten(
                (sample.input, sample.args, sample.kwargs)
            )
            print(
                json.dumps(
                    {
                        "sample": key(op),
                        "dtype": str(dtype),
                        "tensors": [
                            {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for value in sample_tensors
                            if isinstance(value, torch.Tensor)
                        ],
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            result = op(sample.input, *sample.args, **sample.kwargs)
            tensors, _ = tree_flatten(result)
            differentiable = [x for x in tensors if isinstance(x, torch.Tensor) and x.requires_grad]
            if not args.forward_only and op.supports_autograd and differentiable:
                loss = sum((x.real if x.is_complex() else x).sum() for x in differentiable)
                loss.backward()
                record["backward"] = True
            torch.cuda.synchronize()
        except Exception as exc:  # Keep collecting compatibility gaps in one bounded process.
            record["status"] = "fail"
            record["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        print(json.dumps({"completed": record}), file=sys.stderr, flush=True)

    payload = {
        "shard": args.shard,
        "shards": args.shards,
        "scope": "all" if args.all else "curated",
        "max_ops": args.max_ops,
        "sample_index": args.sample_index,
        "dtype_index": args.dtype_index,
        "forward_only": args.forward_only,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "passed": sum(r["status"] == "pass" for r in records),
        "failed": sum(r["status"] == "fail" for r in records),
        "records": records,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
