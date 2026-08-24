"""Initialize PyTorch's remote CUDA bookkeeping in mixed processes.

CUDA Runtime contexts can be switched between LUPINE's local and remote routes,
but PyTorch 2.12 initializes some process-global allocator/device state on the
route of its first tensor operation.  A one-element remote reduction before the
first local tensor makes that state route-complete.  Delay the operation until
the application imports torch so ordinary Python programs pay no import or GPU
cost.
"""

from __future__ import annotations

import builtins
import atexit
import ctypes
import gc
import os
from pathlib import Path
import sys


_anchor = None
_anchor_device = None
_anchor_cleanup_registered = False
_original_import = builtins.__import__
_torch_import_depth = 0
_priming = False
_primed = False
_hostwide_preloaded = False
_sdpa_original = None


def _preload_hostwide_runtime() -> None:
    """Load rgpu before a wheel-bundled CUDA RUNPATH can select libcuda.

    A loader-cache entry is sufficient for ordinary ELF consumers such as
    nvidia-smi, but pip PyTorch searches its bundled CUDA directory before the
    cache.  Host-wide attachment opts Python into this narrowly scoped
    bootstrap instead of using /etc/ld.so.preload, which would inject rgpu into
    every process on the workstation.
    """
    if os.environ.get("RGPU_HOSTWIDE_PYTHON") != "1":
        return
    library_root = Path(
        os.environ.get("RGPU_HOSTWIDE_LIB", "/usr/local/lib/rgpu")
    )
    libraries = (
        "libcuda.so.1",
        "liblupine-cudart-compat.so",
        "libcublas_rpc.so",
        "libcusolver_rpc.so",
        "libcufft_rpc.so",
        "libunsupported_rpc_guard.so",
        "libnccl.so.2",
    )
    missing = [name for name in libraries if not (library_root / name).is_file()]
    if missing:
        raise RuntimeError(
            "rgpu host-wide Python runtime is incomplete: " + ", ".join(missing)
        )
    for name in libraries:
        ctypes.CDLL(str(library_root / name), mode=ctypes.RTLD_GLOBAL)


def _cublas_rpc_enabled() -> bool:
    return os.environ.get("RGPU_HOSTWIDE_PYTHON") == "1" or (
        "libcublas_rpc.so" in os.environ.get("LD_PRELOAD", "")
    )


def _enabled() -> bool:
    if os.environ.get("RGPU_MIXED_PYTORCH_PRIME") != "1":
        return False
    if os.environ.get("RGPU_DISABLE_MIXED_PYTORCH_PRIME") == "1":
        return False
    if os.environ.get("LOCAL_RANK") not in (None, ""):
        return False
    launcher = Path(sys.argv[0]).name
    return launcher not in {"torchrun", "torchrun.exe"}


def _route_function():
    default_library = (
        str(
            Path(os.environ.get("RGPU_HOSTWIDE_LIB", "/usr/local/lib/rgpu"))
            / "libcuda.so.1"
        )
        if os.environ.get("RGPU_HOSTWIDE_PYTHON") == "1"
        else "/run/rgpu/lib/libcuda.so.1"
    )
    library = os.environ.get("LUPINE_LIBCUDA", default_library)
    driver = ctypes.CDLL(library)
    route = driver.lupine_cuda_device_route_id
    route.argtypes = [ctypes.c_int]
    route.restype = ctypes.c_int
    return route


def _install_sdpa_fallback(torch) -> None:
    """Use composable CUDA ops when native SDPA cannot cross driver lanes."""
    global _sdpa_original
    if os.environ.get("RGPU_DISABLE_SDPA_FALLBACK") == "1":
        return
    functional = torch.nn.functional
    if _sdpa_original is not None:
        return
    _sdpa_original = functional.scaled_dot_product_attention
    route = _route_function()
    # Device routing cannot change during an rgpu process. Capture it once so
    # the hot wrapper contains no ctypes call (which is also a Dynamo graph
    # break) and route lookup becomes a constant membership test.
    remote_devices = frozenset(
        ordinal
        for ordinal in range(torch.cuda.device_count())
        if route(ordinal) >= 0
    )

    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        device_index = query.device.index
        if query.device.type != "cuda" or device_index not in remote_devices:
            return _sdpa_original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        if os.environ.get("RGPU_SDPA_FALLBACK") == "math":
            # PyTorch's native math SDPA keeps autograd and mask semantics in
            # upstream code while avoiding the CUDA Runtime flash-attention
            # context lane that cannot yet be routed independently.
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                return _sdpa_original(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
        if is_causal and attn_mask is not None:
            raise RuntimeError("Explicit attn_mask should not be set when is_causal=True")
        if enable_gqa:
            heads = query.size(-3)
            key_heads = key.size(-3)
            value_heads = value.size(-3)
            if heads % key_heads or heads % value_heads:
                raise RuntimeError(
                    "Number of query heads must be divisible by key and value heads"
                )
            key = key.repeat_interleave(heads // key_heads, dim=-3)
            value = value.repeat_interleave(heads // value_heads, dim=-3)
        scale_factor = query.size(-1) ** -0.5 if scale is None else scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if is_causal:
            causal = torch.ones(
                scores.size(-2),
                scores.size(-1),
                dtype=torch.bool,
                device=scores.device,
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        if dropout_p:
            weights = torch.dropout(weights, dropout_p, train=True)
        return torch.matmul(weights.to(value.dtype), value)

    scaled_dot_product_attention.__name__ = _sdpa_original.__name__
    scaled_dot_product_attention.__doc__ = _sdpa_original.__doc__
    functional.scaled_dot_product_attention = scaled_dot_product_attention


def _prime(torch) -> None:
    global _anchor, _anchor_cleanup_registered, _anchor_device
    route = _route_function()
    remote = next(
        (ordinal for ordinal in range(torch.cuda.device_count()) if route(ordinal) >= 0),
        None,
    )
    if remote is None:
        return
    _anchor_device = remote
    local = next(
        (ordinal for ordinal in range(torch.cuda.device_count()) if route(ordinal) == -1),
        None,
    )
    # Exercise the allocator, kernel launcher, and reduction path while the
    # remote route is the only initialized PyTorch device.
    _anchor = torch.ones(1, device=f"cuda:{remote}")
    _anchor.sum().item()
    # The opt-in cuBLAS RPC layer also needs its remote handle/descriptor path
    # initialized before libcublas creates process-global state for a local
    # device. Use deterministic tensors so this does not initialize CUDA RNG.
    if _cublas_rpc_enabled():
        # MAGMA is another process-global CUDA library and cannot distinguish
        # local from remote contexts in one process. Keep mixed linalg on the
        # cuSOLVER surface that the RPC layer routes by opaque handle.
        linalg_backend = os.environ.get("RGPU_LINALG_BACKEND", "cusolver")
        if linalg_backend:
            torch.backends.cuda.preferred_linalg_library(linalg_backend)
        anchors = [_anchor]
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            layer = torch.nn.Linear(
                64, 96, device="cpu", dtype=dtype
            ).to(f"cuda:{remote}")
            layer.weight.data.fill_(0.01)
            layer.bias.data.fill_(0.02)
            inputs = torch.ones(
                (32, 64),
                device=f"cuda:{remote}",
                dtype=dtype,
                requires_grad=True,
            )
            output = layer(inputs)
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize(remote)
            loss.detach().item()
            anchors.extend((layer, inputs, output, loss))
        _anchor = tuple(anchors)
        if local is not None:
            _install_sdpa_fallback(torch)
    if local is not None:
        torch.cuda.set_device(local)
    # Some first-use CUDA library initialization performed by the priming
    # operations restores PyTorch's default backend. Apply the remote-safe
    # preference after priming so small/non-square LU does not fall back to
    # MAGMA's process-global queue on a mixed local/remote process.
    if _cublas_rpc_enabled():
        linalg_backend = os.environ.get("RGPU_LINALG_BACKEND", "cusolver")
        if linalg_backend:
            torch.backends.cuda.preferred_linalg_library(linalg_backend)
    if not _anchor_cleanup_registered:
        atexit.register(_cleanup_anchor)
        _anchor_cleanup_registered = True


def _cleanup_anchor() -> None:
    """Release priming tensors on their owning route before allocator teardown."""
    global _anchor
    if _anchor is None or _anchor_device is None:
        return
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        torch.cuda.set_device(_anchor_device)
        _anchor = None
        gc.collect()
        torch.cuda.synchronize(_anchor_device)
    except Exception:
        # Interpreter shutdown must remain best-effort; the workload's own
        # result and exception, if any, take precedence over priming cleanup.
        pass


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    global _torch_import_depth, _primed, _priming, _hostwide_preloaded
    is_torch = name == "torch" or name.startswith("torch.")
    if (
        is_torch
        and os.environ.get("RGPU_HOSTWIDE_PYTHON") == "1"
        and not _hostwide_preloaded
    ):
        _preload_hostwide_runtime()
        _hostwide_preloaded = True
    if is_torch:
        _torch_import_depth += 1
    try:
        module = _original_import(name, globals, locals, fromlist, level)
    finally:
        if is_torch:
            _torch_import_depth -= 1
    if is_torch and _torch_import_depth == 0 and not _primed and not _priming:
        _priming = True
        try:
            _prime(sys.modules["torch"])
            _primed = True
            builtins.__import__ = _original_import
        finally:
            _priming = False
    return module


if _enabled():
    if _cublas_rpc_enabled() and os.environ.get("RGPU_HOSTWIDE_PYTHON") != "1":
        # The SDPA wrapper must be installed after torch has finished assigning
        # its C-extension symbols. The general lazy import hook can fire while
        # torch.__init__ is still completing, at which point the builtin would
        # overwrite the compatibility wrapper again.
        import torch

        _prime(torch)
        _primed = True
    else:
        builtins.__import__ = _import
