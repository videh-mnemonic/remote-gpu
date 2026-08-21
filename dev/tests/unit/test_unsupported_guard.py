import ctypes
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
CLIENT = ROOT / "rgpu-client" / "remote_gpu"


def test_generated_unsupported_surface_is_current():
    subprocess.run(
        ["python3", str(ROOT / "dev/tools/generate_unsupported_guard.py"), "--check"],
        check=True,
    )


def test_unsupported_surface_is_nonempty_and_consistent():
    surface = json.loads(
        (CLIENT / "unsupported_library_symbols.json").read_text(encoding="utf-8")
    )
    assert surface["total"] == sum(map(len, surface["families"].values()))
    assert surface["total"] == 161
    assert set(surface["families"]) == {"cublas", "cusolver", "nccl"}


def test_guard_returns_not_supported_and_reports_symbol_once(tmp_path, capfd):
    library = tmp_path / "libunsupported_rpc_guard.so"
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            str(library),
            str(CLIENT / "unsupported_rpc_guard.c"),
        ],
        check=True,
    )
    guard = ctypes.CDLL(str(library))
    call = guard.cublasSaxpy_v2
    assert call() == 13
    assert call() == 13
    stderr = capfd.readouterr().err
    assert stderr.count("unsupported remote cublas call: cublasSaxpy_v2") == 1
    assert "--allow-unsupported-library-fallback" in stderr
