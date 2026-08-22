#!/usr/bin/env python3
"""Bounded streaming benchmark for an OpenAI-compatible Responses server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid


def request_once(
    endpoint: str,
    model: str,
    input_words: int,
    output_tokens: int,
    timeout: float,
    fixed_prompt: bool,
) -> dict[str, float | int]:
    nonce = "fixed" if fixed_prompt else uuid.uuid4().hex
    filler = " ".join(f"word{i % 997}" for i in range(input_words))
    prompt = (
        f"Unique request {nonce}. Read the following benchmark text, then emit "
        "an unbroken numbered list of short facts. Continue until the output "
        f"limit; do not conclude early. Benchmark text: {filler}"
    )
    payload = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": "low"},
        "max_output_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "store": False,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    final = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if first_token is None and event.get("type") in {
                "response.reasoning_text.delta",
                "response.output_text.delta",
            }:
                first_token = time.perf_counter()
            if event.get("type") in {"response.completed", "response.incomplete"}:
                final = event.get("response")
    finished = time.perf_counter()
    if first_token is None or final is None or not final.get("usage"):
        raise RuntimeError("server stream did not contain first-token and usage events")
    usage = final["usage"]
    input_count = int(usage["input_tokens"])
    output_count = int(usage["output_tokens"])
    ttft = first_token - started
    wall = finished - started
    decode_window = max(finished - first_token, 1e-9)
    return {
        "input_tokens": input_count,
        "output_tokens": output_count,
        "ttft_s": ttft,
        "wall_s": wall,
        "input_tokens_per_s_to_first": input_count / ttft,
        "output_tokens_per_s_after_first": max(output_count - 1, 0) / decode_window,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8111")
    parser.add_argument("--model", default="qwen3.8-27b-local")
    parser.add_argument("--input-words", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--fixed-prompt",
        action="store_true",
        help="reuse deterministic prompt text (pair with server prefix reuse disabled)",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.input_words < 0 or args.output_tokens < 1 or args.repeats < 1:
        parser.error("input words must be nonnegative; tokens and repeats must be positive")

    runs = []
    for index in range(args.repeats):
        result = request_once(
            args.endpoint,
            args.model,
            args.input_words,
            args.output_tokens,
            args.timeout,
            args.fixed_prompt,
        )
        result["run"] = index + 1
        runs.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    metric_names = (
        "ttft_s",
        "wall_s",
        "input_tokens_per_s_to_first",
        "output_tokens_per_s_after_first",
    )
    summary = {
        "endpoint": args.endpoint,
        "model": args.model,
        "input_words": args.input_words,
        "requested_output_tokens": args.output_tokens,
        "fixed_prompt": args.fixed_prompt,
        "runs": runs,
        "median": {
            metric: statistics.median(float(run[metric]) for run in runs)
            for metric in metric_names
        },
        "mean": {
            metric: statistics.fmean(float(run[metric]) for run in runs)
            for metric in metric_names
        },
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
