from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_upstream_manifest_policy_and_projects():
    manifest = json.loads((ROOT / "manifests" / "upstreams.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["public_forks_allowed"] is False
    assert manifest["policy"]["push_allowed"] is False
    assert {"lupine", "cricket", "gvirtus", "modded-nanogpt", "pytorch", "torchbench"} <= set(
        manifest["projects"]
    )


def test_upstreams_list_runs():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "upstreams.py"), "list"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "lupine" in completed.stdout


def test_timed_runner_rejects_five_minute_timeout(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_timed.py"),
            "--label",
            "invalid-timeout",
            "--mode",
            "native",
            "--timeout",
            "300",
            "--output",
            str(tmp_path / "result.json"),
            "--",
            "true",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode != 0
    assert "strictly less than 300 seconds" in completed.stderr

