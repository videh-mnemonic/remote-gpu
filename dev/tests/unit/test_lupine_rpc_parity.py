from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LUPINE_ROOT = PROJECT_ROOT / "lupine"


def test_every_client_rpc_has_server_dispatch() -> None:
    if not LUPINE_ROOT.exists():
        pytest.skip("LUPINE worktree has not been prepared")

    client_sources = [
        "client.cpp",
        "memcpy.cpp",
        "routing.cpp",
        "codegen/gen_client.cpp",
    ]
    server_sources = ["server.cpp", "codegen/gen_server.cpp"]
    client_text = "\n".join(
        (LUPINE_ROOT / source).read_text() for source in client_sources
    )
    server_text = "\n".join(
        (LUPINE_ROOT / source).read_text() for source in server_sources
    )
    client_rpcs = set(
        re.findall(r"\b(?:LUPINE_)?RPC_[A-Za-z0-9_]+\b", client_text)
    )
    missing = sorted(rpc for rpc in client_rpcs if rpc not in server_text)
    assert missing == [], f"client RPCs without server dispatch: {missing}"


def test_current_nvml_process_query_abi_is_exported() -> None:
    if not LUPINE_ROOT.exists():
        pytest.skip("LUPINE worktree has not been prepared")
    client = (LUPINE_ROOT / "nvml_client.cpp").read_text()
    for family in ("Compute", "Graphics", "MPSCompute"):
        assert f"nvmlDeviceGet{family}RunningProcesses_v3(" in client


def test_tiled_tensor_map_driver_api_has_manual_rpc_on_both_sides() -> None:
    client = (LUPINE_ROOT / "client.cpp").read_text()
    server = (LUPINE_ROOT / "server.cpp").read_text()
    handler = (LUPINE_ROOT / "manual_server.cpp").read_text()
    wire = (LUPINE_ROOT / "lupine_tensormap.h").read_text()

    assert 'extern "C" CUresult cuTensorMapEncodeTiled(' in client
    assert '"cuTensorMapEncodeTiled", (void *)cuTensorMapEncodeTiled' in client
    assert "rpc_write_start_request(conn, RPC_cuTensorMapEncodeTiled)" in client
    assert "RPC_cuTensorMapEncodeTiled" in server
    assert "handle_manual_cuTensorMapEncodeTiled" in server
    assert "int handle_manual_cuTensorMapEncodeTiled(conn_t *conn)" in handler
    assert "LUPINE_TENSOR_MAP_MAX_RANK = 5" in wire


def test_repeated_tensor_maps_are_cached_by_complete_request_and_route() -> None:
    client = (LUPINE_ROOT / "client.cpp").read_text()

    assert "kMaxCachedTensorMaps" in client
    assert "sizeof(int) + sizeof(request)" in client
    assert "lupine_route_identity(route)" in client
    assert "cache->insert_or_assign(std::move(cache_key), encoded)" in client


def test_device_only_async_2d_copy_is_fire_and_forget_on_both_sides() -> None:
    client = (LUPINE_ROOT / "client.cpp").read_text()
    handler = (LUPINE_ROOT / "manual_server.cpp").read_text()

    condition = "async && src_host_size == 0 && dst_host_size == 0"
    assert condition in client
    assert "return rpc_write_end_deferred(conn)" in client
    assert "if (src_host_size == 0 && dst_host_size == 0)" in handler
    assert "result = cuMemcpy2DAsync_v2(&copy, stream)" in handler


def test_runtime_memory_queries_preload_one_symbol_per_fatbin() -> None:
    compat = (LUPINE_ROOT / "cudart_compat.cpp").read_text()

    assert 'extern "C" void __cudaRegisterFunction(' in compat
    assert 'extern "C" void __cudaUnregisterFatBinary(' in compat
    assert 'extern "C" int cudaMemGetInfo(' in compat
    assert "return item.fatbin == entry.fatbin" in compat
    assert "if (entry.fatbin == item.fatbin) entry.warmed = true" in compat
    assert "preload == nullptr || preload[0] != '0'" in compat
    assert 'dlsym(RTLD_NEXT, "cudaFuncGetAttributes")' in compat


def test_graph_replay_refreshes_host_inputs_and_tracks_updated_resources() -> None:
    client = (LUPINE_ROOT / "client.cpp").read_text()
    copies = (LUPINE_ROOT / "copy_pipeline.cpp").read_text()
    memory = (LUPINE_ROOT / "memcpy.cpp").read_text()
    server = (LUPINE_ROOT / "manual_server.cpp").read_text()
    dispatch = (LUPINE_ROOT / "server.cpp").read_text()

    assert "lupine_flush_dirty_host_pages_to_server();" in client
    assert "lupine_prepare_graph_host_source(" in copies
    assert "rpc_write(conn, &graph_host_source" in copies
    assert "lupine_enable_dirty_tracking_locked(it->first" in memory
    assert "lupine_begin_stream_capture_resources(stream)" in server
    assert "lupine_graph_exec_resource_map().insert_or_assign(exec, resources)" in server
    assert "handle_manual_cuGraphExecUpdate" in dispatch


def test_extended_dot_rpc_is_defined_on_both_sides() -> None:
    protocol = (LUPINE_ROOT / "lupine_cublas.h").read_text()
    server = (LUPINE_ROOT / "manual_server.cpp").read_text()
    interposer = (PROJECT_ROOT / "rgpu-client/remote_gpu/cublas_rpc_interposer.c").read_text()

    assert "LUPINE_CUBLAS_DOT_EX = 71" in protocol
    assert "case LUPINE_CUBLAS_DOT_EX" in server
    assert "enum { DOT_EX = 71 }" in interposer
    assert "int cublasDotEx(" in interposer
    assert "LUPINE_CUBLAS_ZGEMM = 72" in protocol
    assert "case LUPINE_CUBLAS_ZGEMM" in server
    assert "enum { ZGEMM = 72 }" in interposer
    assert "int cublasZgemm_v2(" in interposer
    solver_interposer = (
        PROJECT_ROOT / "rgpu-client/remote_gpu/cusolver_rpc_interposer.c"
    ).read_text()
    assert "LUPINE_CUSOLVER_DN_ZGESVDJ = 74" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZGESVDJ" in server
    assert "int cusolverDnZgesvdj(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_ZGESVDJ_BATCHED = 76" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZGESVDJ_BATCHED" in server
    assert "int cusolverDnZgesvdjBatched(" in solver_interposer
    assert "LUPINE_CUBLAS_ZTRSM_BATCHED = 77" in protocol
    assert "case LUPINE_CUBLAS_ZTRSM_BATCHED" in server
    assert "int cublasZtrsmBatched(" in interposer
    assert "LUPINE_CUSOLVER_DN_ZUNMQR = 79" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZUNMQR" in server
    assert "int cusolverDnZunmqr(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_ZGETRS = 82" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZGETRS" in server
    assert "int cusolverDnZgetrs(" in solver_interposer
    assert "LUPINE_CUBLAS_ZGETRS_BATCHED = 83" in protocol
    assert "case LUPINE_CUBLAS_ZGETRS_BATCHED" in server
    assert "int cublasZgetrsBatched(" in interposer
    assert "LUPINE_CUSOLVER_DN_ZUNGQR = 85" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZUNGQR" in server
    assert "int cusolverDnZungqr(" in solver_interposer
    assert "LUPINE_CUBLAS_ZGEMM_STRIDED_BATCHED = 86" in protocol
    assert "case LUPINE_CUBLAS_ZGEMM_STRIDED_BATCHED" in server
    assert "int cublasZgemmStridedBatched(" in interposer
    assert "LUPINE_CUBLAS_ZDOTU = 89" in protocol
    assert "case LUPINE_CUBLAS_ZDOTU" in server
    assert "int cublasZdotu_v2(" in interposer
    assert "LUPINE_CUSOLVER_DN_CGESVDJ_BATCHED = 94" in protocol
    assert "case LUPINE_CUSOLVER_DN_CGESVDJ_BATCHED" in server
    assert "int cusolverDnCgesvdjBatched(" in solver_interposer
    assert "LUPINE_CUBLAS_CTRSM_BATCHED = 95" in protocol
    assert "case LUPINE_CUBLAS_CTRSM_BATCHED" in server
    assert "int cublasCtrsmBatched(" in interposer
    assert "LUPINE_CUSOLVER_DN_CUNMQR = 97" in protocol
    assert "case LUPINE_CUSOLVER_DN_CUNMQR" in server
    assert "int cusolverDnCunmqr(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_CUNGQR = 99" in protocol
    assert "case LUPINE_CUSOLVER_DN_CUNGQR" in server
    assert "int cusolverDnCungqr(" in solver_interposer
    assert "LUPINE_CUBLAS_CGETRF_BATCHED = 100" in protocol
    assert "case LUPINE_CUBLAS_CGETRF_BATCHED" in server
    assert "int cublasCgetrfBatched(" in interposer
    assert "alignas(16) unsigned char aligned_alpha[16]" in server
    assert "int cublasDgemm_v2(" in interposer
    assert "int cublasDgemmStridedBatched(" in interposer
    assert "int cublasDdot_v2(" in interposer
    assert "LUPINE_CUSOLVER_DN_DGETRF = 102" in protocol
    assert "case LUPINE_CUSOLVER_DN_DGETRF" in server
    assert "int cusolverDnDgetrf(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DGESVDJ_BATCHED = 107" in protocol
    assert "case LUPINE_CUSOLVER_DN_DGESVDJ_BATCHED" in server
    assert "int cusolverDnDgesvdjBatched(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DORGQR = 109" in protocol
    assert "case LUPINE_CUSOLVER_DN_DORGQR" in server
    assert "int cusolverDnDorgqr(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DORMQR = 111" in protocol
    assert "case LUPINE_CUSOLVER_DN_DORMQR" in server
    assert "int cusolverDnDormqr(" in solver_interposer
    assert "LUPINE_CUBLAS_DTRSM_BATCHED = 112" in protocol
    assert "case LUPINE_CUBLAS_DTRSM_BATCHED" in server
    assert "int cublasDtrsmBatched(" in interposer
    assert "LUPINE_CUBLAS_DGETRS_BATCHED = 113" in protocol
    assert "case LUPINE_CUBLAS_DGETRS_BATCHED" in server
    assert "int cublasDgetrsBatched(" in interposer
    assert "LUPINE_CUBLAS_DGETRF_BATCHED = 114" in protocol
    assert "case LUPINE_CUBLAS_DGETRF_BATCHED" in server
    assert "int cublasDgetrfBatched(" in interposer
    assert "LUPINE_CUBLAS_DGEMM_BATCHED = 115" in protocol
    assert "case LUPINE_CUBLAS_DGEMM_BATCHED" in server
    assert "int cublasDgemmBatched(" in interposer
    assert "LUPINE_CUBLAS_DGEMV = 116" in protocol
    assert "case LUPINE_CUBLAS_DGEMV" in server
    assert "int cublasDgemv_v2(" in interposer
    assert "LUPINE_CUBLAS_DTRSM = 117" in protocol
    assert "case LUPINE_CUBLAS_DTRSM" in server
    assert "int cublasDtrsm_v2(" in interposer
    assert "LUPINE_CUSOLVER_DN_DPOTRF_BUFFER_SIZE = 118" in protocol
    assert "case LUPINE_CUSOLVER_DN_DPOTRF_BUFFER_SIZE" in server
    assert "int cusolverDnDpotrf_bufferSize(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DPOTRF = 119" in protocol
    assert "case LUPINE_CUSOLVER_DN_DPOTRF" in server
    assert "int cusolverDnDpotrf(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DPOTRS = 120" in protocol
    assert "case LUPINE_CUSOLVER_DN_DPOTRS" in server
    assert "int cusolverDnDpotrs(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DGEQRF_BUFFER_SIZE = 121" in protocol
    assert "case LUPINE_CUSOLVER_DN_DGEQRF_BUFFER_SIZE" in server
    assert "int cusolverDnDgeqrf_bufferSize(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DGESVD = 124" in protocol
    assert "case LUPINE_CUSOLVER_DN_DGESVD" in server
    assert "int cusolverDnDgesvd(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DSYEVD = 126" in protocol
    assert "case LUPINE_CUSOLVER_DN_DSYEVD" in server
    assert "int cusolverDnDsyevd(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DPOTRF_BATCHED = 127" in protocol
    assert "case LUPINE_CUSOLVER_DN_DPOTRF_BATCHED" in server
    assert "int cusolverDnDpotrfBatched(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DSYEVJ_BATCHED = 130" in protocol
    assert "case LUPINE_CUSOLVER_DN_DSYEVJ_BATCHED" in server
    assert "int cusolverDnDsyevjBatched(" in solver_interposer
    assert "LUPINE_CUBLAS_DAXPY = 131" in protocol
    assert "case LUPINE_CUBLAS_DAXPY" in server
    assert "int cublasDaxpy_v2(" in interposer
    assert "LUPINE_CUBLAS_DNRM2 = 134" in protocol
    assert "case LUPINE_CUBLAS_DNRM2" in server
    assert "int cublasDnrm2_v2(" in interposer
    assert "LUPINE_CUBLAS_IDAMIN = 138" in protocol
    assert "case LUPINE_CUBLAS_IDAMIN" in server
    assert "int cublasIdamin_v2(" in interposer
    assert "LUPINE_CUBLAS_CGEMM_STRIDED_BATCHED = 139" in protocol
    assert "case LUPINE_CUBLAS_CGEMM_STRIDED_BATCHED" in server
    assert "int cublasCgemmStridedBatched(" in interposer
    assert "LUPINE_CUBLAS_SGETRF_BATCHED = 140" in protocol
    assert "case LUPINE_CUBLAS_SGETRF_BATCHED" in server
    assert "int cublasSgetrfBatched(" in interposer
    assert "LUPINE_CUSOLVER_DN_ZSYTRF_BUFFER_SIZE = 141" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZSYTRF_BUFFER_SIZE" in server
    assert "int cusolverDnZsytrf_bufferSize(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_ZSYTRF = 142" in protocol
    assert "case LUPINE_CUSOLVER_DN_ZSYTRF" in server
    assert "int cusolverDnZsytrf(" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_CSYTRF_BUFFER_SIZE = 143" in protocol
    assert "case LUPINE_CUSOLVER_DN_CSYTRF_BUFFER_SIZE" in server
    assert "DEFINE_SYTRF_RPC(C, void" in solver_interposer
    assert "LUPINE_CUSOLVER_DN_DSYTRF_BUFFER_SIZE = 145" in protocol
    assert "case LUPINE_CUSOLVER_DN_DSYTRF_BUFFER_SIZE" in server
    assert "DEFINE_SYTRF_RPC(D, double" in solver_interposer
    assert "LUPINE_CUBLAS_CGEMV = 147" in protocol
    assert "case LUPINE_CUBLAS_CGEMV" in server
    assert "DEFINE_COMPLEX_GEMV_RPC(C, CGEMV, 8)" in interposer
    assert "LUPINE_CUBLAS_ZGEMV = 148" in protocol
    assert "DEFINE_COMPLEX_GEMV_RPC(Z, ZGEMV, 16)" in interposer
    assert "LUPINE_CUBLAS_SGELS_BATCHED = 149" in protocol
    assert "DEFINE_GELS_BATCHED_RPC(S, float, SGELS_BATCHED)" in interposer
    assert "DEFINE_GELS_BATCHED_RPC(Z, void, ZGELS_BATCHED)" in interposer
    assert "case LUPINE_CUBLAS_SGELS_BATCHED" in server
    assert "LUPINE_CUBLAS_SGEQRF_BATCHED = 153" in protocol
    assert "DEFINE_GEQRF_BATCHED_RPC(S, float, SGEQRF_BATCHED)" in interposer
    assert "DEFINE_GEQRF_BATCHED_RPC(Z, void, ZGEQRF_BATCHED)" in interposer
    assert "case LUPINE_CUBLAS_SGEQRF_BATCHED" in server
    assert "LUPINE_CUSOLVER_DN_SPOTRF_BATCHED = 157" in protocol
    assert "LUPINE_CUSOLVER_DN_CPOTRF_BATCHED = 158" in protocol
    assert "LUPINE_CUSOLVER_DN_ZPOTRF_BATCHED = 159" in protocol
    assert "case LUPINE_CUSOLVER_DN_SPOTRF_BATCHED" in server
    assert "DEFINE_POTRF_BATCHED(cusolverDnSpotrfBatched" in solver_interposer
    assert "DEFINE_POTRF_BATCHED(cusolverDnCpotrfBatched" in solver_interposer
    assert "DEFINE_POTRF_BATCHED(cusolverDnZpotrfBatched" in solver_interposer
    assert 'q.value=0x100; /* Presence marker' in interposer
    assert "explicit_scalar_presence" in server
    assert "found->pointer_mode=pointer_mode" in interposer
    assert "entry.pointer_mode==0||entry.pointer_mode==4" in interposer
    assert "request.pointer_mode == 3" in server


def test_double_precision_interposers_emit_expected_packets(tmp_path: Path) -> None:
    cublas = tmp_path / "libcublas_rpc_test.so"
    cusolver = tmp_path / "libcusolver_rpc_test.so"
    executable = tmp_path / "double_library_rpc_packets"
    common = ["gcc", "-shared", "-fPIC", "-O2"]
    subprocess.run(
        common
        + [
            str(PROJECT_ROOT / "rgpu-client/remote_gpu/cublas_rpc_interposer.c"),
            "-ldl",
            "-lpthread",
            "-o",
            str(cublas),
        ],
        check=True,
    )
    subprocess.run(
        common
        + [
            str(PROJECT_ROOT / "rgpu-client/remote_gpu/cusolver_rpc_interposer.c"),
            "-ldl",
            "-lpthread",
            "-o",
            str(cusolver),
        ],
        check=True,
    )
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-rdynamic",
            str(PROJECT_ROOT / "dev/tests/probes/double_library_rpc_packets.c"),
            "-L",
            str(tmp_path),
            "-Wl,--no-as-needed",
            "-l:libcublas_rpc_test.so",
            "-l:libcusolver_rpc_test.so",
            f"-Wl,-rpath,{tmp_path}",
            "-o",
            str(executable),
        ],
        check=True,
    )
    completed = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    assert completed.stdout == "double-library RPC packet checks passed\n"


def test_supplemental_cublas_exports_have_torch_symbol_version(tmp_path: Path) -> None:
    library = tmp_path / "libcublas_rpc_test.so"
    subprocess.run(
        [
            "gcc", "-shared", "-fPIC", "-O2",
            str(PROJECT_ROOT / "rgpu-client/remote_gpu/cublas_rpc_interposer.c"),
            "-ldl", "-lpthread",
            f"-Wl,--version-script={PROJECT_ROOT / 'rgpu-client/remote_gpu/cublas_rpc.map'}",
            "-o", str(library),
        ],
        check=True,
    )
    exported = subprocess.run(
        ["objdump", "-T", str(library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for symbol in (
        "cublasSgetrfBatched",
        "cublasSgelsBatched",
        "cublasDgelsBatched",
        "cublasCgelsBatched",
        "cublasZgelsBatched",
        "cublasSgeqrfBatched",
        "cublasDgeqrfBatched",
        "cublasCgeqrfBatched",
        "cublasZgeqrfBatched",
    ):
        line = next(line for line in exported.splitlines() if symbol in line)
        assert "libcublas.so.13" in line
