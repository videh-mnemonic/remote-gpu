from argparse import Namespace
from pathlib import Path
import json
import subprocess

import pytest

from remote_gpu.cli import (
    Host,
    RgpuError,
    apply_transport_endpoints,
    client_command,
    command_attach,
    command_detach,
    command_gc,
    deploy_image,
    expand_hosts,
    expand_remote_hosts,
    host_points_to_local,
    invoking_user_prefix,
    local_server_interface,
    native_python_rank_command,
    parse_host,
    remote_client_interface,
    require_matching_server_image,
    run_checked,
    server_name,
    start_server,
    reuse_persistent_server,
    stop_server,
)


def test_native_python_rank_uses_physical_gpu_and_distributed_identity():
    command = native_python_rank_command(
        "torch:test", 2, 4, "192.0.2.1", 29681, "eth0", ["TOKEN=abc"], ["--steps", "3"]
    )
    assert command[command.index("--gpus") + 1] == "all"
    assert "RANK=2" in command
    assert "WORLD_SIZE=4" in command
    assert "MASTER_ADDR=192.0.2.1" in command
    assert "NCCL_SOCKET_IFNAME=eth0" in command
    assert command[-5:] == ["torch:test", "python3", "-", "--steps", "3"]


def test_parse_host_with_ssh_user_and_port():
    host = parse_host("user@192.0.2.10:14834")
    assert host == Host("user@192.0.2.10", "192.0.2.10", 14834)
    assert host.endpoint == "192.0.2.10:14834"


def test_expand_hosts_supports_repeat_and_comma():
    hosts = expand_hosts(["u@10.0.0.1,10.0.0.2:15000", "10.0.0.3"])
    assert [host.endpoint for host in hosts] == [
        "10.0.0.1:14833",
        "10.0.0.2:15000",
        "10.0.0.3:14833",
    ]


def test_expand_hosts_rejects_duplicate_endpoint():
    with pytest.raises(RgpuError, match="duplicate"):
        expand_hosts(["a@10.0.0.1", "b@10.0.0.1"])


def test_expand_remote_hosts_rejects_local_machine(monkeypatch):
    monkeypatch.setattr("remote_gpu.cli.local_hostnames", lambda: {"client-host"})
    monkeypatch.setattr("remote_gpu.cli.local_ip_addresses", lambda: {"192.0.2.20"})
    assert host_points_to_local(Host("user@client-host", "client-host"))
    with pytest.raises(RgpuError, match="this machine"):
        expand_remote_hosts(["user@192.0.2.20"])


def test_authenticated_transport_endpoint_preserves_ssh_lease_identity():
    hosts = apply_transport_endpoints(
        expand_hosts(["u@gpu.internal:14834"]), ["https://gpu.example"]
    )
    assert hosts == [
        Host(
            "u@gpu.internal",
            "gpu.internal",
            14834,
            "https://gpu.example:443",
        )
    ]
    assert hosts[0].endpoint == "https://gpu.example:443"
    assert server_name(hosts[0]) == "rgpu-server-14834"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gpu.example",
        "https://user@gpu.example",
        "https://gpu.example/rpc",
        "https://gpu.example?debug=true",
        "https://gpu.example#fragment",
        "https://gpu.example:0",
    ],
)
def test_authenticated_transport_endpoint_rejects_unsafe_forms(endpoint):
    with pytest.raises(RgpuError, match="https://host"):
        apply_transport_endpoints(expand_hosts(["u@gpu.internal"]), [endpoint])


def test_authenticated_transport_endpoint_count_must_match_hosts():
    with pytest.raises(RgpuError, match="count"):
        apply_transport_endpoints(
            expand_hosts(["u@gpu-a.internal,u@gpu-b.internal"]),
            ["https://gpu.example"],
        )


def test_server_name_is_a_stable_per_port_lease():
    assert server_name(Host("u@remote", "10.0.0.1", 15000)) == "rgpu-server-15000"


def test_remote_client_interface_uses_ssh_route(monkeypatch):
    result = subprocess.CompletedProcess(
        [], 0, "192.0.2.20 dev eth0 src 192.0.2.10 uid 1000\n", ""
    )
    monkeypatch.setattr("remote_gpu.cli.ssh", lambda *_args, **_kwargs: result)
    assert remote_client_interface(Host("u@remote", "192.0.2.20")) == "eth0"


def test_local_server_interface_uses_ip_route(monkeypatch):
    result = subprocess.CompletedProcess(
        [], 0, "192.0.2.10 dev eth0 src 192.0.2.20 uid 1000\n", ""
    )
    monkeypatch.setattr("remote_gpu.cli.run_checked", lambda *_args, **_kwargs: result)
    assert local_server_interface(Host("u@remote", "192.0.2.10")) == "eth0"


def test_server_lease_survives_remote_reboot(monkeypatch):
    commands = []

    def fake_ssh(_host, command, *, capture=True):
        commands.append(list(command))
        if command[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "missing")
        if command[:3] == ["docker", "inspect", "--format"]:
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("remote_gpu.cli.ssh", fake_ssh)
    monkeypatch.setattr("remote_gpu.cli.remote_compute_processes", lambda _host: "")
    monkeypatch.setattr("remote_gpu.cli.remote_port_listener", lambda _host: "")
    monkeypatch.setattr("remote_gpu.cli.remote_client_interface", lambda _host: "eno2")
    monkeypatch.setattr("remote_gpu.cli.wait_ready", lambda _host: None)
    host = Host("u@remote", "10.0.0.1")
    start_server(
        host, "rgpu-server-14833", "server:test", lease_mode="persistent"
    )
    run = next(command for command in commands if command[:2] == ["docker", "run"])
    assert run[run.index("--restart") + 1] == "unless-stopped"
    assert run[run.index("--pid") + 1] == "host"
    assert "--rm" not in run
    assert "io.rgpu.managed=true" in run
    assert "NCCL_SOCKET_IFNAME=eno2" in run


def test_ephemeral_server_lease_does_not_restart_after_reboot(monkeypatch):
    commands = []

    def fake_ssh(_host, command, *, capture=True):
        commands.append(list(command))
        if command[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "missing")
        if command[:3] == ["docker", "inspect", "--format"]:
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("remote_gpu.cli.ssh", fake_ssh)
    monkeypatch.setattr("remote_gpu.cli.remote_compute_processes", lambda _host: "")
    monkeypatch.setattr("remote_gpu.cli.remote_port_listener", lambda _host: "")
    monkeypatch.setattr("remote_gpu.cli.remote_client_interface", lambda _host: None)
    monkeypatch.setattr("remote_gpu.cli.wait_ready", lambda _host: None)
    start_server(
        Host("u@remote", "10.0.0.1"),
        "rgpu-server-14833",
        "server:test",
        session="abc",
    )
    run = next(command for command in commands if command[:2] == ["docker", "run"])
    assert run[run.index("--restart") + 1] == "no"
    assert "io.rgpu.lease-mode=ephemeral" in run
    assert "io.rgpu.session=abc" in run


def test_reuse_persistent_server_validates_managed_idle_lease(monkeypatch):
    commands = []
    labels = {
        "io.rgpu.managed": "true",
        "io.rgpu.lease-mode": "persistent",
        "io.rgpu.endpoint": "10.0.0.1:14833",
    }

    def fake_ssh(_host, command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, json.dumps(labels) + " true\n", ""
        )

    monkeypatch.setattr("remote_gpu.cli.ssh", fake_ssh)
    monkeypatch.setattr("remote_gpu.cli.remote_compute_processes", lambda _host: "")
    monkeypatch.setattr("remote_gpu.cli.wait_ready", lambda _host: None)
    reuse_persistent_server(Host("u@remote", "10.0.0.1"), "rgpu-server-14833")
    assert commands[0][:3] == ["docker", "inspect", "--format"]


def test_reuse_persistent_server_refuses_active_gpu(monkeypatch):
    monkeypatch.setattr(
        "remote_gpu.cli.remote_compute_processes", lambda _host: "123, python, 1 GiB"
    )
    with pytest.raises(RgpuError, match="already in use"):
        reuse_persistent_server(Host("u@remote", "10.0.0.1"), "rgpu-server-14833")


def test_gc_reclaims_only_expired_disconnected_ephemeral_lease(monkeypatch):
    commands = []
    labels = {
        "io.rgpu.managed": "true",
        "io.rgpu.lease-mode": "ephemeral",
        "io.rgpu.created-unix": "100",
    }
    monkeypatch.setattr("remote_gpu.cli.time.time", lambda: 1000)
    def fake_ssh(_host, command, **_kwargs):
        commands.append(list(command))
        output = json.dumps(labels) + "\n" if command[1] == "inspect" else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("remote_gpu.cli.ssh", fake_ssh)
    monkeypatch.setattr("remote_gpu.cli.remote_active_clients", lambda _host: "")
    assert command_gc(Namespace(host=["u@remote"], min_age=600)) == 0
    assert ["docker", "rm", "--force", "rgpu-server-14833"] in commands


def test_gc_never_reclaims_persistent_attachment(monkeypatch):
    labels = {
        "io.rgpu.managed": "true",
        "io.rgpu.lease-mode": "persistent",
        "io.rgpu.created-unix": "100",
    }
    monkeypatch.setattr("remote_gpu.cli.time.time", lambda: 1000)
    monkeypatch.setattr(
        "remote_gpu.cli.ssh",
        lambda _host, command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(labels) + "\n", ""
        ),
    )
    monkeypatch.setattr(
        "remote_gpu.cli.stop_server",
        lambda *_args: pytest.fail("persistent lease must not be removed"),
    )
    assert command_gc(Namespace(host=["u@remote"], min_age=600)) == 0


def test_gc_fails_closed_when_remote_inspection_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "remote_gpu.cli.ssh",
        lambda _host, command, **_kwargs: subprocess.CompletedProcess(
            command, 255, "", "ssh unavailable"
        ),
    )
    with pytest.raises(RgpuError, match="inspect lease"):
        command_gc(Namespace(host=["u@remote"], min_age=600))


def test_stop_server_removes_only_named_lease(monkeypatch):
    commands = []

    def fake_ssh(_host, command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("remote_gpu.cli.ssh", fake_ssh)
    stop_server(Host("u@remote", "10.0.0.1"), "rgpu-server-14833")
    assert commands == [["docker", "rm", "--force", "rgpu-server-14833"]]


def test_run_checked_converts_timeout_to_bounded_failure(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ssh"], 15, output="partial")

    monkeypatch.setattr("remote_gpu.cli.subprocess.run", timeout)
    result = run_checked(["ssh"], timeout=15)
    assert result.returncode == 124
    assert result.stdout == "partial"


def test_stop_server_fails_closed_after_bounded_ssh_timeout(monkeypatch):
    monkeypatch.setattr(
        "remote_gpu.cli.ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 124, "", "timed out after 15s"
        ),
    )
    with pytest.raises(RgpuError, match="remove lease.*timed out"):
        stop_server(Host("u@remote", "10.0.0.1"), "rgpu-server-14833")


def test_detach_verifies_local_identity_and_attempts_every_lease(monkeypatch, tmp_path):
    state = {
        "root": str(tmp_path),
        "leases": [
            {
                "ssh_target": "u@gpu-a",
                "address": "gpu-a",
                "port": 14833,
                "name": "rgpu-server-14833",
            },
            {
                "ssh_target": "u@gpu-b",
                "address": "gpu-b",
                "port": 14834,
                "name": "rgpu-server-14834",
            },
        ],
    }
    identities = iter(("physical", "physical"))
    stopped = []

    monkeypatch.setattr("remote_gpu.cli.local_gpu_identity", lambda: next(identities))
    monkeypatch.setattr("remote_gpu.cli.hostwide_uninstall", lambda *_args, **_kwargs: state)

    def fail_cleanup(host, name):
        stopped.append((host.ssh_target, name))
        raise RgpuError(f"unreachable {host.ssh_target}")

    monkeypatch.setattr("remote_gpu.cli.stop_server", fail_cleanup)
    with pytest.raises(RgpuError, match="local GPU restoration verified"):
        command_detach(Namespace(root=str(tmp_path)))
    assert stopped == [
        ("u@gpu-b", "rgpu-server-14834"),
        ("u@gpu-a", "rgpu-server-14833"),
    ]


def test_attach_failure_attempts_cleanup_for_every_started_lease(monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr("remote_gpu.cli.local_gpu_identity", lambda: "physical")
    monkeypatch.setattr("remote_gpu.cli.prepare_shim_bundle", lambda _image: tmp_path)
    monkeypatch.setattr("remote_gpu.cli.require_matching_server_image", lambda *_args: None)
    monkeypatch.setattr("remote_gpu.cli.remote_gpu_identity", lambda _host: "remote")
    monkeypatch.setattr("remote_gpu.cli.gpu_uuids", lambda _identity: set())
    monkeypatch.setattr("remote_gpu.cli.start_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "remote_gpu.cli.hostwide_install",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RgpuError("install failed")),
    )

    def fail_cleanup(host, name):
        stopped.append((host.ssh_target, name))
        raise RgpuError("cleanup unavailable")

    monkeypatch.setattr("remote_gpu.cli.stop_server", fail_cleanup)
    args = Namespace(
        root=str(tmp_path),
        host=["u@gpu-a:14833,u@gpu-b:14834"],
        endpoint=[],
        shim_image="shim:test",
        deploy=False,
        server_image="server:test",
    )
    with pytest.raises(RgpuError, match="automatic rollback incomplete"):
        command_attach(args)
    assert stopped == [
        ("u@gpu-b", "rgpu-server-14834"),
        ("u@gpu-a", "rgpu-server-14833"),
    ]


def test_deploy_timeout_kills_both_pipeline_children(monkeypatch):
    class Stream:
        def close(self):
            pass

    class Child:
        def __init__(self, *, times_out=False):
            self.stdout = Stream()
            self.returncode = None
            self.times_out = times_out
            self.killed = False

        def communicate(self, timeout=None):
            assert timeout == 240
            if self.times_out:
                raise subprocess.TimeoutExpired(["ssh"], timeout)
            return "", ""

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.returncode = -9 if self.killed else 0
            return self.returncode

    save = Child()
    load = Child(times_out=True)
    children = iter((save, load))
    monkeypatch.setattr(
        "remote_gpu.cli.subprocess.Popen", lambda *_args, **_kwargs: next(children)
    )

    with pytest.raises(RgpuError, match="exceeded 240s"):
        deploy_image(Host("u@remote", "10.0.0.1"), "server:test")
    assert save.killed is True
    assert load.killed is True


def test_root_network_operations_drop_to_invoking_user(monkeypatch):
    monkeypatch.setattr("remote_gpu.cli.os.geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_USER", "developer")
    monkeypatch.setattr(
        "remote_gpu.cli.pwd.getpwnam",
        lambda _name: type("Account", (), {"pw_uid": 1000})(),
    )
    assert invoking_user_prefix() == ["sudo", "-u", "developer", "-H", "--"]


def test_unprivileged_network_operations_keep_current_identity(monkeypatch):
    monkeypatch.setattr("remote_gpu.cli.os.geteuid", lambda: 1000)
    assert invoking_user_prefix() == []


def test_matching_server_image_is_accepted(monkeypatch):
    monkeypatch.setattr("remote_gpu.cli.local_image_id", lambda _image: "sha256:same")
    monkeypatch.setattr(
        "remote_gpu.cli.remote_image_id", lambda _host, _image: "sha256:same"
    )
    require_matching_server_image(Host("u@remote", "10.0.0.1"), "server:test")


def test_stale_server_image_requires_deploy(monkeypatch):
    monkeypatch.setattr("remote_gpu.cli.local_image_id", lambda _image: "sha256:new")
    monkeypatch.setattr(
        "remote_gpu.cli.remote_image_id", lambda _host, _image: "sha256:old"
    )
    with pytest.raises(RgpuError, match="differs.*--deploy"):
        require_matching_server_image(Host("u@remote", "10.0.0.1"), "server:test")


def test_missing_server_image_requires_deploy(monkeypatch):
    monkeypatch.setattr("remote_gpu.cli.local_image_id", lambda _image: "sha256:new")

    def missing(_host, _image):
        raise RgpuError("not found")

    monkeypatch.setattr("remote_gpu.cli.remote_image_id", missing)
    with pytest.raises(RgpuError, match="not deployed.*--deploy"):
        require_matching_server_image(Host("u@remote", "10.0.0.1"), "server:test")


def test_client_command_never_exposes_local_gpu(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="client:test",
        include_local=False,
        command=["--", "python3", "train.py"],
    )
    command = client_command(args, [Host("u@remote", "10.0.0.1")], "abc")
    assert "--gpus" not in command
    assert "LUPINE_SERVER=10.0.0.1:14833" in command
    assert "LUPINE_DISABLE_LOCAL=1" in command
    assert f"{tmp_path}:{tmp_path}:ro" in command


def test_client_command_uses_verified_https_transport(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="client:test",
        include_local=False,
        command=["--", "python3", "train.py"],
    )
    host = Host(
        "u@gpu.internal", "gpu.internal", 14833, "https://gpu.example:443"
    )
    command = client_command(args, [host], "abc")
    assert "LUPINE_SERVER=https://gpu.example:443" in command


def test_client_command_can_opt_in_to_local_plus_remote(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="client:test",
        include_local=True,
        command=["--", "python3", "train.py"],
    )
    command = client_command(
        args,
        [Host("u@remote", "10.0.0.1")],
        "abc",
        nccl_interfaces=("eno2",),
    )
    assert command[command.index("--gpus") + 1] == "all"
    assert "LUPINE_DISABLE_LOCAL=1" not in command
    assert "NCCL_SOCKET_IFNAME=eno2" in command


def test_client_command_injects_portable_shim_bundle(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="pytorch:unmodified",
        include_local=False,
        command=["--", "python3", "train.py"],
    )
    bundle = tmp_path / "shim"
    command = client_command(
        args, [Host("u@remote", "10.0.0.1")], "abc", shim_bundle=bundle
    )
    assert f"{bundle / 'lib'}:/run/rgpu/lib:ro" in command
    assert (
        f"{bundle / 'bin' / 'nvidia-smi'}:/usr/local/bin/nvidia-smi:ro"
        in command
    )
    assert "LUPINE_LIBCUDA=/run/rgpu/lib/libcuda.so.1" in command
    assert "LD_PRELOAD=/run/rgpu/lib/liblupine-cudart-compat.so" in command


def test_client_command_enables_cublas_rpc_for_mixed_run(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="pytorch:unmodified",
        include_local=True,
        cublas_rpc=True,
        cublas_async=False,
        command=["--", "python3", "train.py"],
    )
    bundle = tmp_path / "shim"
    command = client_command(
        args, [Host("u@remote", "10.0.0.1")], "abc", shim_bundle=bundle
    )
    assert (
        "LD_PRELOAD=/run/rgpu/lib/liblupine-cudart-compat.so:"
        "/run/rgpu/lib/libcublas_rpc.so:"
        "/run/rgpu/lib/libcusolver_rpc.so"
        ":/run/rgpu/lib/libcufft_rpc.so:"
        "/run/rgpu/lib/libunsupported_rpc_guard.so"
    ) in command
    assert "DISABLE_ADDMM_CUDA_LT=1" in command
    assert "TORCH_LINALG_PREFER_CUSOLVER=1" in command
    assert "RGPU_CUBLAS_ASYNC=1" not in command


def test_client_command_enables_cublas_rpc_without_exposing_local_gpu(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="pytorch:unmodified",
        include_local=False,
        cublas_rpc=True,
        cublas_async=False,
        command=["--", "python3", "train.py"],
    )
    command = client_command(
        args,
        [Host("u@remote", "10.0.0.1")],
        "abc",
        shim_bundle=tmp_path / "shim",
    )
    assert "--gpus" not in command
    assert "LUPINE_DISABLE_LOCAL=1" in command
    assert "/run/rgpu/lib/libcublas_rpc.so" in next(
        value for value in command if value.startswith("LD_PRELOAD=")
    )


def test_client_command_enables_explicit_async_cublas(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=[],
        image="pytorch:unmodified",
        include_local=True,
        cublas_rpc=True,
        cublas_async=True,
        command=["--", "python3", "train.py"],
    )
    bundle = tmp_path / "shim"
    command = client_command(
        args, [Host("u@remote", "10.0.0.1")], "abc", shim_bundle=bundle
    )
    assert "RGPU_CUBLAS_ASYNC=1" in command


def test_client_command_preserves_explicit_disabled_async_cublas(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=["RGPU_CUBLAS_ASYNC=0"],
        image="pytorch:unmodified",
        include_local=True,
        cublas_rpc=True,
        cublas_async=True,
        command=["--", "python3", "train.py"],
    )
    bundle = tmp_path / "shim"
    command = client_command(
        args, [Host("u@remote", "10.0.0.1")], "abc", shim_bundle=bundle
    )
    assert command.count("RGPU_CUBLAS_ASYNC=0") == 1
    assert "RGPU_CUBLAS_ASYNC=1" not in command


def test_client_command_preserves_explicit_addmm_backend(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=["DISABLE_ADDMM_CUDA_LT=0"],
        image="pytorch:unmodified",
        include_local=True,
        cublas_rpc=True,
        command=["--", "python3", "train.py"],
    )
    bundle = tmp_path / "shim"
    command = client_command(
        args, [Host("u@remote", "10.0.0.1")], "abc", shim_bundle=bundle
    )
    assert command.count("DISABLE_ADDMM_CUDA_LT=0") == 1
    assert "DISABLE_ADDMM_CUDA_LT=1" not in command


def test_client_command_preserves_explicit_linalg_backend(tmp_path: Path):
    args = Namespace(
        workspace=str(tmp_path),
        workspace_write=False,
        output=None,
        env=["TORCH_LINALG_PREFER_CUSOLVER=0"],
        image="pytorch:unmodified",
        include_local=False,
        cublas_rpc=True,
        cublas_async=False,
        command=["--", "python3", "train.py"],
    )
    command = client_command(
        args,
        [Host("u@remote", "10.0.0.1")],
        "abc",
        shim_bundle=tmp_path / "shim",
    )
    assert command.count("TORCH_LINALG_PREFER_CUSOLVER=0") == 1
    assert "TORCH_LINALG_PREFER_CUSOLVER=1" not in command
