#include <dlfcn.h>
#include <cstdio>
#include <cstdint>

using nvmlReturn_t = int;
using nvmlDevice_t = void *;

struct nvmlProcessInfo_t {
  unsigned int pid;
  std::uint64_t usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
};

using process_fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *,
                                    nvmlProcessInfo_t *);

static nvmlReturn_t trace(const char *name, nvmlDevice_t device,
                          unsigned int *count, nvmlProcessInfo_t *infos) {
  auto fn = reinterpret_cast<process_fn>(dlsym(RTLD_NEXT, name));
  unsigned int before = count == nullptr ? 0 : *count;
  auto result = fn == nullptr ? 13 : fn(device, count, infos);
  std::fprintf(stderr, "TRACE %s count=%u infos=%p => result=%d count=%u\n",
               name, before, static_cast<void *>(infos), result,
               count == nullptr ? 0 : *count);
  return result;
}

#define TRACE_PROCESS_FN(name)                                                \
  extern "C" nvmlReturn_t name(nvmlDevice_t device, unsigned int *count,     \
                                nvmlProcessInfo_t *infos) {                   \
    return trace(#name, device, count, infos);                               \
  }

TRACE_PROCESS_FN(nvmlDeviceGetComputeRunningProcesses_v2)
TRACE_PROCESS_FN(nvmlDeviceGetGraphicsRunningProcesses_v2)
TRACE_PROCESS_FN(nvmlDeviceGetMPSComputeRunningProcesses_v2)
