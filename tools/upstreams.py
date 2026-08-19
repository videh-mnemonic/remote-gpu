#!/usr/bin/env python3
"""Clone public upstreams locally while making accidental pushes fail closed."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "upstreams.json"
CLONE_ROOT = ROOT / "external"
RESOLVED_ROOT = ROOT / "manifests" / "resolved"
DISABLED_PUSH_URL = "disabled://public-push-prohibited"


def run(args: list[str], cwd: Path | None = None, timeout: int = 900) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def project(name: str) -> dict:
    projects = load_manifest()["projects"]
    if name not in projects:
        choices = ", ".join(sorted(projects))
        raise SystemExit(f"unknown project {name!r}; choose one of: {choices}")
    return projects[name]


def install_push_guard(repo: Path) -> None:
    run(["git", "remote", "set-url", "--push", "origin", DISABLED_PUSH_URL], repo)
    git_dir = Path(run(["git", "rev-parse", "--git-dir"], repo))
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    hook = git_dir / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        "#!/bin/sh\n"
        "echo 'Push blocked: this is a local clone of a public upstream.' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)


def license_digest(repo: Path) -> tuple[str | None, str | None]:
    names = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")
    for name in names:
        candidate = repo / name
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            return candidate.name, digest
    return None, None


def write_resolved(name: str, repo: Path, spec: dict) -> None:
    license_file, license_sha256 = license_digest(repo)
    record = {
        "name": name,
        "url": spec["url"],
        "requested_ref": spec.get("ref"),
        "resolved_commit": run(["git", "rev-parse", "HEAD"], repo),
        "commit_time": run(["git", "show", "-s", "--format=%cI", "HEAD"], repo),
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "declared_license": spec.get("license"),
        "license_file": license_file,
        "license_sha256": license_sha256,
        "push_url": run(["git", "remote", "get-url", "--push", "origin"], repo),
    }
    RESOLVED_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESOLVED_ROOT / f"{name}.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {output.relative_to(ROOT)}")


def clone(name: str) -> None:
    spec = project(name)
    target = CLONE_ROOT / name
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing path: {target}")
    CLONE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--", spec["url"], str(target)],
            timeout=900,
        )
        install_push_guard(target)
        requested_ref = spec.get("ref")
        if requested_ref:
            run(["git", "checkout", "--detach", f"origin/{requested_ref}"], target)
        else:
            run(["git", "checkout", "--detach", "origin/HEAD"], target)
        write_resolved(name, target, spec)
        audit(name)
    except BaseException:
        if target.exists():
            print(f"clone failed; partial directory retained for inspection: {target}", file=sys.stderr)
        raise


def audit(name: str) -> None:
    spec = project(name)
    target = CLONE_ROOT / name
    if not target.is_dir():
        raise SystemExit(f"not cloned: {target}")
    actual_url = run(["git", "remote", "get-url", "origin"], target)
    push_url = run(["git", "remote", "get-url", "--push", "origin"], target)
    git_dir = Path(run(["git", "rev-parse", "--git-dir"], target))
    if not git_dir.is_absolute():
        git_dir = target / git_dir
    hook = git_dir / "hooks" / "pre-push"
    problems = []
    if actual_url != spec["url"]:
        problems.append(f"origin mismatch: {actual_url}")
    if push_url != DISABLED_PUSH_URL:
        problems.append(f"push URL is not disabled: {push_url}")
    if not hook.is_file() or not os.access(hook, os.X_OK):
        problems.append("rejecting pre-push hook is missing or not executable")
    if problems:
        raise SystemExit("\n".join(problems))
    commit = run(["git", "rev-parse", "HEAD"], target)
    status = run(["git", "status", "--short"], target)
    print(f"{name}: {commit} push=BLOCKED dirty={bool(status)}")


def list_projects() -> None:
    manifest = load_manifest()
    for name, spec in sorted(manifest["projects"].items()):
        target = CLONE_ROOT / name
        state = "cloned" if target.is_dir() else "absent"
        print(f"{name:18} {state:7} {spec['url']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for command in ("clone", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("name")
    args = parser.parse_args()
    if args.command == "list":
        list_projects()
    elif args.command == "clone":
        clone(args.name)
    elif args.command == "audit":
        audit(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
