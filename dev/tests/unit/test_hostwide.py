from pathlib import Path
import json

import pytest

import remote_gpu.hostwide as hostwide
from remote_gpu.hostwide import HostwideError, install, uninstall
from remote_gpu.rescue import loaded_gpu_libraries, recover


def shim_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    library = bundle / "lib"
    library.mkdir(parents=True)
    for name in (
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        "liblupine-cudart-compat.so",
        "libcudart.so.13",
        "libcublas_rpc.so",
        "libcusolver_rpc.so",
        "libcufft_rpc.so",
        "libunsupported_rpc_guard.so",
        "libnccl.so.2",
        "libnccl_real.so.2",
    ):
        (library / name).write_bytes(f"test {name}".encode())
    python = bundle / "python"
    python.mkdir()
    (python / "sitecustomize.py").write_text("# test bootstrap\n")
    return bundle


def test_live_root_is_locked():
    with pytest.raises(HostwideError, match="live-root attachment is locked"):
        install(Path("/"), Path("/missing"), ["10.77.77.1:14833"])


def test_sandbox_install_and_detach_are_reversible(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    original = root / "etc/rgpu/endpoints"
    original.parent.mkdir(parents=True)
    original.write_text("previous:14833\n", encoding="utf-8")

    state = install(
        root,
        shim_bundle(tmp_path),
        ["10.77.77.1:14833", "10.77.77.2:14834"],
        refresh_loader=False,
    )
    assert state["live_root"] is False
    assert state["phase"] == "attached"
    assert original.read_text(encoding="utf-8") == (
        "10.77.77.1:14833,10.77.77.2:14834\n"
    )
    assert (root / "usr/local/lib/rgpu/libcuda.so").readlink() == Path(
        "libcuda.so.1"
    )
    assert (root / "etc/ld.so.conf.d/00-rgpu.conf").read_text() == (
        "/usr/local/lib/rgpu\n"
    )
    profile = (root / "etc/profile.d/rgpu.sh").read_text()
    assert "RGPU_HOSTWIDE_PYTHON" in profile
    assert "LD_PRELOAD" not in profile
    assert (root / "usr/local/lib/rgpu/python/sitecustomize.py").is_file()

    uninstall(root, refresh_loader=False)
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not (root / "usr/local/lib/rgpu/libcuda.so.1").exists()
    assert not (root / "var/lib/rgpu/state.json").exists()


def test_detach_refuses_modified_managed_file(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    install(
        root,
        shim_bundle(tmp_path),
        ["10.77.77.1:14833"],
        refresh_loader=False,
    )
    (root / "etc/rgpu/endpoints").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(HostwideError, match="managed files changed.*modified"):
        uninstall(root, refresh_loader=False)


def test_invalid_endpoint_is_rejected_before_writes(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(HostwideError, match="invalid endpoint"):
        install(root, shim_bundle(tmp_path), ["bad endpoint"], refresh_loader=False)
    assert not (root / "var/lib/rgpu/state.json").exists()


def test_installing_journal_recovers_partial_replacements(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    original = root / "etc/rgpu/endpoints"
    original.parent.mkdir(parents=True)
    original.write_text("previous:14833\n", encoding="utf-8")
    install(
        root,
        shim_bundle(tmp_path),
        ["10.77.77.1:14833"],
        refresh_loader=False,
    )

    state_path = root / "var/lib/rgpu/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase"] = "installing"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    # Model SIGKILL before one file was replaced and after another atomic
    # replacement: an original target and a missing new target are both safe.
    original.write_text("previous:14833\n", encoding="utf-8")
    (root / "usr/local/lib/rgpu/libcuda.so.1").unlink()

    uninstall(root, refresh_loader=False)
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not state_path.exists()


def test_detach_retries_after_loader_refresh_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    original = root / "etc/rgpu/endpoints"
    original.parent.mkdir(parents=True)
    original.write_text("previous:14833\n", encoding="utf-8")
    install(
        root,
        shim_bundle(tmp_path),
        ["10.77.77.1:14833"],
        refresh_loader=False,
    )

    def fail_refresh(_root: Path) -> None:
        raise HostwideError("injected loader refresh failure")

    monkeypatch.setattr(hostwide, "_refresh_loader", fail_refresh)
    with pytest.raises(HostwideError, match="injected loader refresh failure"):
        uninstall(root, refresh_loader=True)

    state_path = root / "var/lib/rgpu/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase"] == "detaching"
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not (root / "usr/local/lib/rgpu/libcuda.so.1").exists()

    # The restored backup is valid evidence in the detaching journal, so the
    # exact same transaction can finish after the external fault clears.
    uninstall(root, refresh_loader=False)
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not state_path.exists()


def test_out_of_band_rescue_restores_without_gpu_libraries(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    original = root / "etc/rgpu/endpoints"
    original.parent.mkdir(parents=True)
    original.write_text("previous:14833\n", encoding="utf-8")
    install(
        root,
        shim_bundle(tmp_path),
        ["10.77.77.1:14833"],
        refresh_loader=False,
        metadata={"leases": [{"name": "test-lease"}]},
    )

    assert loaded_gpu_libraries() == []
    state = recover(root, refresh_loader=False)
    assert state["leases"] == [{"name": "test-lease"}]
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not (root / "var/lib/rgpu/state.json").exists()
    assert loaded_gpu_libraries() == []


@pytest.mark.parametrize(
    "boundary",
    [
        "usr/local/lib/rgpu/libcuda.so.1",
        "usr/local/lib/rgpu/libnvidia-ml.so.1",
        "usr/local/lib/rgpu/liblupine-cudart-compat.so",
        "usr/local/lib/rgpu/libcudart.so.13",
        "usr/local/lib/rgpu/libcublas_rpc.so",
        "usr/local/lib/rgpu/libcusolver_rpc.so",
        "usr/local/lib/rgpu/libcufft_rpc.so",
        "usr/local/lib/rgpu/libunsupported_rpc_guard.so",
        "usr/local/lib/rgpu/libnccl.so.2",
        "usr/local/lib/rgpu/libnccl_real.so.2",
        "usr/local/lib/rgpu/python/sitecustomize.py",
        "usr/local/lib/rgpu/libcuda.so",
        "usr/local/lib/rgpu/libnvidia-ml.so",
        "etc/rgpu/endpoints",
        "etc/ld.so.conf.d/00-rgpu.conf",
        "etc/profile.d/rgpu.sh",
    ],
)
def test_install_recovers_after_each_managed_replacement_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
):
    root = tmp_path / "root"
    root.mkdir()
    original = root / "etc/rgpu/endpoints"
    original.parent.mkdir(parents=True)
    original.write_text("previous:14833\n", encoding="utf-8")
    real_copy = hostwide._atomic_copy
    real_text = hostwide._atomic_text
    real_symlink = hostwide._atomic_symlink
    injected = False

    def should_fail(target: Path) -> bool:
        nonlocal injected
        try:
            relative = target.relative_to(root).as_posix()
        except ValueError:
            return False
        if not injected and relative == boundary and (root / hostwide.STATE_PATH).exists():
            injected = True
            return True
        return False

    def copy(source: Path, target: Path, mode: int) -> None:
        if should_fail(target):
            real_copy(source, target, mode)
            raise HostwideError(f"injected boundary: {boundary}")
        real_copy(source, target, mode)

    def text(value: str, target: Path, mode: int = 0o644) -> None:
        if should_fail(target):
            real_text(value, target, mode)
            raise HostwideError(f"injected boundary: {boundary}")
        real_text(value, target, mode)

    def symlink(link_target: str, target: Path) -> None:
        if should_fail(target):
            real_symlink(link_target, target)
            raise HostwideError(f"injected boundary: {boundary}")
        real_symlink(link_target, target)

    monkeypatch.setattr(hostwide, "_atomic_copy", copy)
    monkeypatch.setattr(hostwide, "_atomic_text", text)
    monkeypatch.setattr(hostwide, "_atomic_symlink", symlink)
    with pytest.raises(HostwideError, match="injected boundary"):
        install(
            root,
            shim_bundle(tmp_path),
            ["10.77.77.1:14833"],
            refresh_loader=False,
        )
    assert injected

    uninstall(root, refresh_loader=False)
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not (root / "var/lib/rgpu/state.json").exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "usr/local/lib/rgpu/libcuda.so.1",
        "usr/local/lib/rgpu/libnvidia-ml.so.1",
        "usr/local/lib/rgpu/liblupine-cudart-compat.so",
        "usr/local/lib/rgpu/libcudart.so.13",
        "usr/local/lib/rgpu/libcublas_rpc.so",
        "usr/local/lib/rgpu/libcusolver_rpc.so",
        "usr/local/lib/rgpu/libcufft_rpc.so",
        "usr/local/lib/rgpu/libunsupported_rpc_guard.so",
        "usr/local/lib/rgpu/libnccl.so.2",
        "usr/local/lib/rgpu/libnccl_real.so.2",
        "usr/local/lib/rgpu/python/sitecustomize.py",
        "usr/local/lib/rgpu/libcuda.so",
        "usr/local/lib/rgpu/libnvidia-ml.so",
        "etc/rgpu/endpoints",
        "etc/ld.so.conf.d/00-rgpu.conf",
        "etc/profile.d/rgpu.sh",
    ],
)
def test_detach_resumes_after_each_managed_removal_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
):
    root = tmp_path / "root"
    root.mkdir()
    original = root / "etc/rgpu/endpoints"
    original.parent.mkdir(parents=True)
    original.write_text("previous:14833\n", encoding="utf-8")
    install(
        root,
        shim_bundle(tmp_path),
        ["10.77.77.1:14833"],
        refresh_loader=False,
    )
    real_remove = hostwide._remove_target
    injected = False

    def remove(target: Path) -> None:
        nonlocal injected
        real_remove(target)
        if not injected and target.relative_to(root).as_posix() == boundary:
            injected = True
            raise HostwideError(f"injected removal: {boundary}")

    monkeypatch.setattr(hostwide, "_remove_target", remove)
    with pytest.raises(HostwideError, match="injected removal"):
        uninstall(root, refresh_loader=False)
    assert injected

    uninstall(root, refresh_loader=False)
    assert original.read_text(encoding="utf-8") == "previous:14833\n"
    assert not (root / "var/lib/rgpu/state.json").exists()
