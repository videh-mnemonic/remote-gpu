#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUdeviceptr = std::uint64_t;

static constexpr CUresult CUDA_SUCCESS = 0;

template <typename Function>
Function symbol(void *library, const char *name) {
  auto function = reinterpret_cast<Function>(dlsym(library, name));
  if (function == nullptr) {
    std::fprintf(stderr, "missing symbol: %s\n", name);
    std::exit(2);
  }
  return function;
}

static void check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed with CUresult %d\n", operation, result);
    std::exit(3);
  }
}

int main() {
  void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (cuda == nullptr) {
    std::fprintf(stderr, "dlopen libcuda failed: %s\n", dlerror());
    return 2;
  }
  auto init = symbol<CUresult (*)(unsigned int)>(cuda, "cuInit");
  auto get_count = symbol<CUresult (*)(int *)>(cuda, "cuDeviceGetCount");
  auto get_device = symbol<CUresult (*)(CUdevice *, int)>(cuda, "cuDeviceGet");
  auto retain = symbol<CUresult (*)(CUcontext *, CUdevice)>(
      cuda, "cuDevicePrimaryCtxRetain");
  auto release_primary = symbol<CUresult (*)(CUdevice)>(
      cuda, "cuDevicePrimaryCtxRelease_v2");
  auto set_current =
      symbol<CUresult (*)(CUcontext)>(cuda, "cuCtxSetCurrent");
  auto allocate =
      symbol<CUresult (*)(CUdeviceptr *, std::size_t)>(cuda, "cuMemAlloc_v2");
  auto memset_d32 = symbol<CUresult (*)(CUdeviceptr, unsigned int, std::size_t)>(
      cuda, "cuMemsetD32_v2");
  auto copy_to_host = symbol<CUresult (*)(void *, CUdeviceptr, std::size_t)>(
      cuda, "cuMemcpyDtoH_v2");
  auto synchronize = symbol<CUresult (*)()>(cuda, "cuCtxSynchronize");
  auto release = symbol<CUresult (*)(CUdeviceptr)>(cuda, "cuMemFree_v2");

  check(init(0), "cuInit");
  int count = 0;
  check(get_count(&count), "cuDeviceGetCount");
  if (count < 2) {
    std::fprintf(stderr, "expected at least two devices, got %d\n", count);
    return 4;
  }

  CUdevice devices[2] = {};
  CUcontext contexts[2] = {};
  CUdeviceptr allocations[2] = {};
  for (int ordinal = 0; ordinal < 2; ++ordinal) {
    check(get_device(&devices[ordinal], ordinal), "cuDeviceGet");
    check(retain(&contexts[ordinal], devices[ordinal]),
          "cuDevicePrimaryCtxRetain");
    check(set_current(contexts[ordinal]), "cuCtxSetCurrent");
    check(allocate(&allocations[ordinal], 4096), "cuMemAlloc_v2");
    check(memset_d32(allocations[ordinal], 0x11111111u * (ordinal + 1), 1024),
          "cuMemsetD32_v2");
  }

  for (int pass = 0; pass < 4; ++pass) {
    int ordinal = pass % 2;
    std::uint32_t value = 0;
    check(set_current(contexts[ordinal]), "cuCtxSetCurrent switch");
    check(synchronize(), "cuCtxSynchronize");
    check(copy_to_host(&value, allocations[ordinal], sizeof(value)),
          "cuMemcpyDtoH_v2");
    std::uint32_t expected = 0x11111111u * (ordinal + 1);
    if (value != expected) {
      std::fprintf(stderr, "device %d returned %#x, expected %#x\n", ordinal,
                   value, expected);
      return 5;
    }
  }

  for (int ordinal = 0; ordinal < 2; ++ordinal) {
    check(set_current(contexts[ordinal]), "cuCtxSetCurrent cleanup");
    check(release(allocations[ordinal]), "cuMemFree_v2");
    check(release_primary(devices[ordinal]), "cuDevicePrimaryCtxRelease_v2");
  }
  std::printf("PASS mixed Driver API routes: local cuda:0 + remote cuda:1\n");
  return 0;
}
