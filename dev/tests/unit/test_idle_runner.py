import json
import os
from pathlib import Path
import subprocess
from argparse import Namespace

from dev.tools import run_idle_opinfo_tier as runner


def test_save_result_retains_raw_streams_and_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    payload = {"records": [], "passed": 0, "failed": 0}
    completed = subprocess.CompletedProcess(
        [], 0, "setup noise\n" + json.dumps(payload) + "\n", "diagnostic\n"
    )

    assert runner.save_result("case", completed) == payload
    assert (tmp_path / "case.stdout.log").read_text() == completed.stdout
    assert (tmp_path / "case.stderr.log").read_text() == completed.stderr
    assert json.loads((tmp_path / "case.json").read_text()) == payload


def test_save_result_retains_failure_without_inventing_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    completed = subprocess.CompletedProcess([], 124, "partial output\n", "timeout\n")

    assert runner.save_result("case", completed) is None
    assert (tmp_path / "case.stdout.log").read_text() == "partial output\n"
    assert (tmp_path / "case.stderr.log").read_text() == "timeout\n"
    assert not (tmp_path / "case.json").exists()


def test_save_result_recovers_completed_records_from_interrupted_stderr(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    record = {"op": "add", "status": "pass", "seconds": 0.1}
    completed = subprocess.CompletedProcess(
        [], 124, "", json.dumps({"completed": record}) + "\ninterrupted\n"
    )
    assert runner.save_result("case", completed) is None
    assert json.loads((tmp_path / "case.partial.json").read_text()) == {
        "complete": False,
        "records": [record],
    }


def test_runner_deploy_is_explicit_opt_in():
    source = Path(runner.__file__).read_text()
    assert 'parser.add_argument(\n        "--deploy"' in source
    assert "if args.deploy:" in source


def test_runner_lock_prevents_overlapping_unattended_gates(tmp_path):
    lock_path = tmp_path / "runner.lock"
    first = runner.acquire_runner_lock(lock_path)
    assert first is not None
    assert first.readable()

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        first.close()
        second = runner.acquire_runner_lock(lock_path)
        os.write(write_fd, b"locked" if second is None else b"overlap")
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    assert os.read(read_fd, 16) == b"locked"
    os.close(read_fd)
    os.waitpid(pid, 0)
    first.close()


def test_workload_arguments_and_artifact_name_cover_broad_sample():
    args = Namespace(
        all=True,
        dtype_index=0,
        forward_only=True,
        max_ops=100,
        sample_index=1,
        shard=2,
        shards=4,
    )
    values = runner.workload_arguments(args, 1)
    assert values[-3:] == ["--forward-only", "--max-ops", "100"]
    assert "--all" in values
    assert runner.artifact_name(args, "remote", 32) == (
        "opinfo-all-sample1-dtype0-remote-shard2of4-round32"
    )
