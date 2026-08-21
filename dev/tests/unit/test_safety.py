from pathlib import Path
import subprocess

import remote_gpu.safety as safety


def test_read_only_command_timeout_is_bounded(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["nvidia-smi"], 15, output="partial")

    monkeypatch.setattr(safety.subprocess, "run", timeout)
    result = safety._run(["nvidia-smi"])
    assert result["exit_code"] == 124
    assert result["stdout"] == "partial"


def test_physical_nvml_environment_bypasses_attachment(tmp_path, monkeypatch):
    vendor = tmp_path / "libnvidia-ml.so.1"
    vendor.write_bytes(b"vendor")
    monkeypatch.setattr(safety, "PHYSICAL_LIBRARY_CANDIDATES", (vendor,))
    monkeypatch.setenv("LUPINE_SERVER", "remote:14833")
    monkeypatch.setenv("UNCHANGED", "yes")

    environment = safety._physical_nvml_environment()

    assert environment["LD_PRELOAD"] == str(vendor.resolve())
    assert "LUPINE_SERVER" not in environment
    assert environment["UNCHANGED"] == "yes"


def test_physical_nvml_environment_fails_without_vendor_library(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        safety,
        "PHYSICAL_LIBRARY_CANDIDATES",
        (Path(tmp_path / "missing-libnvidia-ml.so.1"),),
    )
    try:
        safety._physical_nvml_environment()
    except RuntimeError as exc:
        assert "was not found" in str(exc)
    else:
        raise AssertionError("missing vendor NVML must fail closed")
