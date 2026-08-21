#!/usr/bin/env python3
"""Build a versioned rgpu client wheel and OCI release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "rgpu-client"
sys.path.insert(0, str(CLIENT))
from remote_gpu.version import WIRE_PROTOCOL_VERSION, __version__  # noqa: E402


def checked(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=True, text=True, capture_output=True, **kwargs
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_metadata(image: str) -> dict[str, object]:
    payload = json.loads(checked(["docker", "image", "inspect", image]).stdout)[0]
    return {
        "name": image,
        "id": payload["Id"],
        "size_bytes": payload["Size"],
        "repo_digests": payload.get("RepoDigests") or [],
    }


def export_image(image: str, target: Path) -> None:
    with target.open("wb") as output:
        save = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
        assert save.stdout is not None
        compress = subprocess.run(
            ["zstd", "-T0", "-6", "-c"],
            stdin=save.stdout,
            stdout=output,
            check=True,
        )
        save.stdout.close()
        if save.wait() != 0 or compress.returncode != 0:
            raise RuntimeError(f"failed to export {image}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-image-build", action="store_true")
    parser.add_argument("--export-images", action="store_true")
    args = parser.parse_args()

    release_root = (ROOT / "dev/releases").resolve()
    output = (args.output or release_root / __version__).resolve()
    if output == release_root or release_root not in output.parents:
        raise SystemExit(f"release output must be below {release_root}")
    if output.exists():
        shutil.rmtree(output)
    wheels = output / "wheels"
    wheels.mkdir(parents=True)

    if not args.skip_image_build:
        subprocess.run([str(ROOT / "dev/tools/build_images.sh")], check=True)

    with tempfile.TemporaryDirectory(prefix="rgpu-wheel-") as temporary:
        source = Path(temporary) / "rgpu-client"
        shutil.copytree(CLIENT, source, ignore=shutil.ignore_patterns("build", "*.egg-info"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                str(source),
            ],
            check=True,
        )

    wheel = next(wheels.glob("remote_gpu-*.whl"))
    images = {
        "client": image_metadata(f"remote-gpu-client:{__version__}"),
        "host": image_metadata(f"remote-gpu-host:{__version__}"),
        "default_workload": image_metadata(f"remote-gpu-workload:{__version__}"),
    }
    commit = checked(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()
    dirty = bool(checked(["git", "status", "--porcelain"], cwd=ROOT).stdout)
    manifest = {
        "schema": 1,
        "version": __version__,
        "wire_protocol": WIRE_PROTOCOL_VERSION,
        "git_commit": commit,
        "git_dirty": dirty,
        "client_wheel": {
            "path": str(wheel.relative_to(output)),
            "sha256": sha256(wheel),
            "size_bytes": wheel.stat().st_size,
        },
        "images": images,
    }
    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.export_images:
        image_dir = output / "images"
        image_dir.mkdir()
        for role, metadata in images.items():
            target = image_dir / f"{role}.tar.zst"
            export_image(str(metadata["name"]), target)
            metadata["archive"] = {
                "path": str(target.relative_to(output)),
                "sha256": sha256(target),
                "size_bytes": target.stat().st_size,
            }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    checksums = [
        f"{sha256(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
