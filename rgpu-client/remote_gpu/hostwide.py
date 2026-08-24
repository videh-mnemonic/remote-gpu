"""Transactional host-wide shim installation, initially restricted to sandboxes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


class HostwideError(RuntimeError):
    """A host-wide installation safety failure."""


STATE_PATH = Path("var/lib/rgpu/state.json")
BACKUP_ROOT = Path("var/lib/rgpu/backups")
LIB_ROOT = Path("usr/local/lib/rgpu")
PYTHON_ROOT = LIB_ROOT / "python"
ENDPOINT_PATH = Path("etc/rgpu/endpoints")
LOADER_PATH = Path("etc/ld.so.conf.d/00-rgpu.conf")
PROFILE_PATH = Path("etc/profile.d/rgpu.sh")


def _profile_text() -> str:
    return """# Managed by rgpu; removed transactionally by `rgpu detach`.
if [ -d /usr/local/lib/rgpu/python ]; then
  case ":${LD_LIBRARY_PATH-}:" in
    *:/usr/local/lib/rgpu:*) ;;
    *) LD_LIBRARY_PATH="/usr/local/lib/rgpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  export LD_LIBRARY_PATH
  case ":${PYTHONPATH-}:" in
    *:/usr/local/lib/rgpu/python:*) ;;
    *) PYTHONPATH="/usr/local/lib/rgpu/python${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
  export PYTHONPATH
  RGPU_HOSTWIDE_PYTHON=${RGPU_HOSTWIDE_PYTHON:-1}
  RGPU_MIXED_PYTORCH_PRIME=${RGPU_MIXED_PYTORCH_PRIME:-1}
  RGPU_CUFFT_RPC=${RGPU_CUFFT_RPC:-1}
  DISABLE_ADDMM_CUDA_LT=${DISABLE_ADDMM_CUDA_LT:-1}
  TORCH_LINALG_PREFER_CUSOLVER=${TORCH_LINALG_PREFER_CUSOLVER:-1}
  export RGPU_HOSTWIDE_PYTHON RGPU_MIXED_PYTORCH_PRIME RGPU_CUFFT_RPC
  export DISABLE_ADDMM_CUDA_LT TORCH_LINALG_PREFER_CUSOLVER
fi
"""


def _inside(root: Path, relative: Path) -> Path:
    return root / relative


def _fingerprint(path: Path) -> str:
    if path.is_symlink():
        return f"symlink:{os.readlink(path)}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _fingerprint_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_copy(source: Path, target: Path, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(text: str, target: Path, mode: int = 0o644) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_symlink(link_target: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.link"
    temporary.unlink(missing_ok=True)
    os.symlink(link_target, temporary)
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def _remove_target(target: Path) -> None:
    existed = target.exists() or target.is_symlink()
    target.unlink(missing_ok=True)
    if existed:
        _fsync_directory(target.parent)


def _backup(root: Path, relative: Path) -> str | None:
    target = _inside(root, relative)
    if not target.exists() and not target.is_symlink():
        return None
    backup = _inside(root, BACKUP_ROOT / relative)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        _atomic_symlink(os.readlink(target), backup)
    else:
        _atomic_copy(target, backup, target.stat().st_mode & 0o777)
    return str(BACKUP_ROOT / relative)


def _validate_root(root: Path, allow_live: bool) -> Path:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise HostwideError(f"installation root is not a directory: {resolved}")
    if resolved == Path("/") and not allow_live:
        raise HostwideError(
            "live-root attachment is locked until sandbox rollback and coexistence gates pass"
        )
    return resolved


def _validate_endpoints(endpoints: Sequence[str]) -> str:
    if not endpoints:
        raise HostwideError("at least one endpoint is required")
    for endpoint in endpoints:
        if not endpoint or any(character.isspace() for character in endpoint):
            raise HostwideError(f"invalid endpoint: {endpoint!r}")
    return ",".join(endpoints)


def _refresh_loader(root: Path) -> None:
    command = ["ldconfig"] if root == Path("/") else ["ldconfig", "-r", str(root)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown ldconfig error").strip()
        raise HostwideError(f"ldconfig failed for sandbox root {root}: {detail}")


def install(
    root: Path,
    shim_bundle: Path,
    endpoints: Sequence[str],
    *,
    allow_live: bool = False,
    refresh_loader: bool = True,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Install into root transactionally; never permits / unless explicitly unlocked."""
    root = _validate_root(root, allow_live)
    state_path = _inside(root, STATE_PATH)
    if state_path.exists():
        raise HostwideError(f"rgpu is already attached in {root}")

    endpoint_text = _validate_endpoints(endpoints)
    sources = {
        LIB_ROOT / "libcuda.so.1": shim_bundle / "lib" / "libcuda.so.1",
        LIB_ROOT / "libnvidia-ml.so.1": shim_bundle / "lib" / "libnvidia-ml.so.1",
        LIB_ROOT / "liblupine-cudart-compat.so":
            shim_bundle / "lib" / "liblupine-cudart-compat.so",
        LIB_ROOT / "libcudart.so.13": shim_bundle / "lib" / "libcudart.so.13",
        LIB_ROOT / "libcublas_rpc.so": shim_bundle / "lib" / "libcublas_rpc.so",
        LIB_ROOT / "libcusolver_rpc.so": shim_bundle / "lib" / "libcusolver_rpc.so",
        LIB_ROOT / "libcufft_rpc.so": shim_bundle / "lib" / "libcufft_rpc.so",
        LIB_ROOT / "libunsupported_rpc_guard.so":
            shim_bundle / "lib" / "libunsupported_rpc_guard.so",
        LIB_ROOT / "libnccl.so.2": shim_bundle / "lib" / "libnccl.so.2",
        LIB_ROOT / "libnccl_real.so.2": shim_bundle / "lib" / "libnccl_real.so.2",
        PYTHON_ROOT / "sitecustomize.py":
            shim_bundle / "python" / "sitecustomize.py",
    }
    missing = [str(source) for source in sources.values() if not source.is_file()]
    if missing:
        raise HostwideError(f"shim bundle is incomplete: {', '.join(missing)}")

    managed = [
        *sources,
        LIB_ROOT / "libcuda.so",
        LIB_ROOT / "libnvidia-ml.so",
        ENDPOINT_PATH,
        LOADER_PATH,
        PROFILE_PATH,
    ]
    backups = {str(relative): _backup(root, relative) for relative in managed}

    desired = {
        str(relative): _fingerprint(source) for relative, source in sources.items()
    }
    desired[str(LIB_ROOT / "libcuda.so")] = "symlink:libcuda.so.1"
    desired[str(LIB_ROOT / "libnvidia-ml.so")] = "symlink:libnvidia-ml.so.1"
    desired[str(ENDPOINT_PATH)] = _fingerprint_bytes(
        f"{endpoint_text}\n".encode("utf-8")
    )
    desired[str(LOADER_PATH)] = _fingerprint_bytes(b"/usr/local/lib/rgpu\n")
    desired[str(PROFILE_PATH)] = _fingerprint_bytes(_profile_text().encode("utf-8"))
    state: dict[str, object] = {
        "schema": 1,
        "phase": "installing",
        "root": str(root),
        "endpoints": list(endpoints),
        "managed": desired,
        "backups": backups,
        "live_root": root == Path("/"),
    }
    if metadata:
        state.update(metadata)
    # Write-ahead recovery record: after this fsync, every subsequent atomic
    # replacement is either the desired fingerprint or recoverable backup.
    _atomic_text(json.dumps(state, indent=2, sort_keys=True) + "\n", state_path, 0o600)

    try:
        for relative, source in sources.items():
            mode = 0o644 if relative.suffix == ".py" else 0o755
            _atomic_copy(source, _inside(root, relative), mode)
        _atomic_symlink("libcuda.so.1", _inside(root, LIB_ROOT / "libcuda.so"))
        _atomic_symlink(
            "libnvidia-ml.so.1", _inside(root, LIB_ROOT / "libnvidia-ml.so")
        )
        _atomic_text(f"{endpoint_text}\n", _inside(root, ENDPOINT_PATH))
        _atomic_text("/usr/local/lib/rgpu\n", _inside(root, LOADER_PATH))
        _atomic_text(_profile_text(), _inside(root, PROFILE_PATH))
        if refresh_loader:
            _refresh_loader(root)

        fingerprints = {
            str(relative): _fingerprint(_inside(root, relative)) for relative in managed
        }
        if fingerprints != desired:
            raise HostwideError("installed file fingerprints do not match the transaction")
        state["phase"] = "attached"
        _atomic_text(json.dumps(state, indent=2, sort_keys=True) + "\n", state_path, 0o600)
        return state
    except Exception:
        # Leave the durable "installing" journal in place for detach/recovery.
        raise


def uninstall(
    root: Path,
    *,
    allow_live: bool = False,
    refresh_loader: bool = True,
) -> dict[str, object]:
    """Remove only unchanged managed files and restore pre-attach contents."""
    root = _validate_root(root, allow_live)
    state_path = _inside(root, STATE_PATH)
    if not state_path.is_file():
        raise HostwideError(f"rgpu is not attached in {root}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    managed: dict[str, str] = state.get("managed", {})
    backups: dict[str, str | None] = state.get("backups", {})
    phase = state.get("phase")
    if phase not in {"installing", "attached", "detaching"}:
        raise HostwideError(f"unsupported transaction phase: {phase!r}")

    changed = []
    for relative_text, expected in managed.items():
        target = _inside(root, Path(relative_text))
        if not target.exists() and not target.is_symlink():
            if phase == "attached":
                changed.append(f"{relative_text} (missing)")
            continue
        actual = _fingerprint(target)
        if actual == expected:
            continue
        backup_text = backups.get(relative_text)
        backup = _inside(root, Path(backup_text)) if backup_text else None
        if (
            phase in {"installing", "detaching"}
            and backup is not None
            and (backup.exists() or backup.is_symlink())
            and actual == _fingerprint(backup)
        ):
            continue
        changed.append(f"{relative_text} (modified)")
    if changed:
        raise HostwideError(
            "refusing detach because managed files changed: " + ", ".join(changed)
        )

    # Persist the rollback intent before changing any managed target. A retry
    # after interruption may therefore encounter either the rgpu fingerprint,
    # the restored backup fingerprint, or a missing target that had no backup.
    # All three are unambiguous in the journal and remain safely idempotent.
    if phase != "detaching":
        state["phase"] = "detaching"
        _atomic_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            state_path,
            0o600,
        )

    for relative_text in managed:
        target = _inside(root, Path(relative_text))
        _remove_target(target)
    for relative_text, backup_text in backups.items():
        if backup_text is None:
            continue
        backup = _inside(root, Path(backup_text))
        target = _inside(root, Path(relative_text))
        if backup.is_symlink():
            _atomic_symlink(os.readlink(backup), target)
        else:
            _atomic_copy(backup, target, backup.stat().st_mode & 0o777)

    if refresh_loader:
        _refresh_loader(root)
    _remove_target(state_path)
    shutil.rmtree(_inside(root, BACKUP_ROOT), ignore_errors=True)
    return state
