from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_product_and_development_boundaries_are_explicit() -> None:
    for relative in ("lupine", "rgpu-client", "rgpu-host", "dev"):
        assert (ROOT / relative).is_dir()

    for retired in ("remote_gpu", "containers", "tools", "tests", "patches"):
        assert not (ROOT / retired).exists()

    assert (ROOT / "rgpu-client/pyproject.toml").is_file()
    assert (ROOT / "rgpu-client/containers/release-artifact/Dockerfile").is_file()
    assert (ROOT / "rgpu-host/containers/server-overlay/Dockerfile").is_file()
    assert (ROOT / "rgpu-host/check.sh").is_file()
    assert (ROOT / "lupine/CMakeLists.txt").is_file()


def test_build_uses_canonical_lupine_without_patch_replay() -> None:
    build = (ROOT / "dev/tools/build_images.sh").read_text(encoding="utf-8")
    assert 'lupine_source="$project_root/lupine"' in build
    assert "git apply" not in build
    assert "external/lupine-v1" not in build

    installer = (ROOT / "rgpu-client/install.sh").read_text(encoding="utf-8")
    assert 'pipx install --force "$install_source"' in installer
    assert 'bash "$project_root/dev/tools/build_images.sh"' in installer


def test_release_artifacts_share_one_version() -> None:
    version = (ROOT / "rgpu-client/remote_gpu/version.py").read_text(encoding="utf-8")
    assert '__version__ = "0.2.0"' in version
    assert 'WIRE_PROTOCOL_VERSION = "6"' in version

    transport = (ROOT / "lupine/h2.cpp").read_text(encoding="utf-8")
    assert 'kLupineProtocolVersion[] = "6"' in transport
    assert 'kLupineClientProtocolVersion[] = "6"' in transport

    cli = (ROOT / "rgpu-client/remote_gpu/cli.py").read_text(encoding="utf-8")
    assert 'f"remote-gpu-client:{__version__}"' in cli
    assert 'f"remote-gpu-host:{__version__}"' in cli
    assert 'f"remote-gpu-workload:{__version__}"' in cli

    build = (ROOT / "dev/tools/build_images.sh").read_text(encoding="utf-8")
    assert '"remote-gpu-client:$release_version"' in build
    assert '"remote-gpu-host:$release_version"' in build
    assert '"remote-gpu-workload:$release_version"' in build
