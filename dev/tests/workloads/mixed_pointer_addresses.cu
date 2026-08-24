#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 1;
  void* local = nullptr;
  void* remote = nullptr;
  constexpr std::size_t bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
  if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&local, bytes) != cudaSuccess) return 2;
  if (cudaSetDevice(1) != cudaSuccess || cudaMalloc(&remote, bytes) != cudaSuccess) return 3;
  const auto local_begin = reinterpret_cast<std::uintptr_t>(local);
  const auto remote_begin = reinterpret_cast<std::uintptr_t>(remote);
  const bool overlap = local_begin < remote_begin + bytes && remote_begin < local_begin + bytes;
  std::printf("local=%p remote=%p bytes=%zu overlap=%s\n", local, remote, bytes,
              overlap ? "yes" : "no");
  cudaSetDevice(1);
  cudaFree(remote);
  cudaSetDevice(0);
  cudaFree(local);
  return overlap ? 4 : 0;
}
