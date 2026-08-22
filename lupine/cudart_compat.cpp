#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <vector>

extern "C" int lupine_launch_runtime_anchor_kernel();
extern "C" int lupine_restore_default_context_hint();
extern "C" int lupine_repair_current_context_device(int device);

static thread_local int lupine_runtime_device = -1;

struct lupine_registered_runtime_function {
  void **fatbin = nullptr;
  const void *host_function = nullptr;
  bool warmed = false;
};

static std::mutex &lupine_registered_runtime_function_mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::vector<lupine_registered_runtime_function> &
lupine_registered_runtime_functions() {
  static std::vector<lupine_registered_runtime_function> functions;
  return functions;
}

// Remember compiler-registered host symbols so runtime memory queries can
// materialize each fatbin before reporting free memory while the remote driver
// itself remains in lazy mode. Set LUPINE_PRELOAD_REGISTERED_KERNELS=0 to opt
// out. This keeps allocators and graph builders from charging later module
// initialization to their own memory budget.
extern "C" void __cudaRegisterFunction(
    void **fatbin, const char *host_function, char *device_function,
    const char *device_name, int thread_limit, void *thread_id,
    void *block_id, void *block_dim, void *grid_dim, int *warp_size) {
  using real_fn_t = void (*)(void **, const char *, char *, const char *, int,
                             void *, void *, void *, void *, int *);
  static real_fn_t real = reinterpret_cast<real_fn_t>(
      dlsym(RTLD_NEXT, "__cudaRegisterFunction"));
  if (real != nullptr) {
    real(fatbin, host_function, device_function, device_name, thread_limit,
         thread_id, block_id, block_dim, grid_dim, warp_size);
  }
  if (fatbin != nullptr && host_function != nullptr) {
    std::lock_guard<std::mutex> lock(
        lupine_registered_runtime_function_mutex());
    lupine_registered_runtime_functions().push_back(
        {fatbin, host_function, false});
  }
}

extern "C" void __cudaUnregisterFatBinary(void **fatbin) {
  using real_fn_t = void (*)(void **);
  static real_fn_t real = reinterpret_cast<real_fn_t>(
      dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary"));
  {
    std::lock_guard<std::mutex> lock(
        lupine_registered_runtime_function_mutex());
    auto &functions = lupine_registered_runtime_functions();
    functions.erase(
        std::remove_if(functions.begin(), functions.end(),
                       [fatbin](const auto &entry) {
                         return entry.fatbin == fatbin;
                       }),
        functions.end());
  }
  if (real != nullptr) real(fatbin);
}

extern "C" int cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
  using mem_info_fn_t = int (*)(size_t *, size_t *);
  using attributes_fn_t = int (*)(void *, const void *);
  static mem_info_fn_t real_mem_info = reinterpret_cast<mem_info_fn_t>(
      dlsym(RTLD_NEXT, "cudaMemGetInfo"));
  static attributes_fn_t real_attributes = reinterpret_cast<attributes_fn_t>(
      dlsym(RTLD_NEXT, "cudaFuncGetAttributes"));
  static thread_local bool warming = false;

  const char *preload = std::getenv("LUPINE_PRELOAD_REGISTERED_KERNELS");
  if (!warming && (preload == nullptr || preload[0] != '0') &&
      real_attributes != nullptr) {
    std::vector<lupine_registered_runtime_function> pending;
    {
      std::lock_guard<std::mutex> lock(
          lupine_registered_runtime_function_mutex());
      for (const auto &entry : lupine_registered_runtime_functions()) {
        if (entry.warmed) continue;
        const bool fatbin_pending =
            std::any_of(pending.begin(), pending.end(), [&](const auto &item) {
              return item.fatbin == entry.fatbin;
            });
        if (!fatbin_pending) pending.push_back(entry);
      }
    }
    warming = true;
    size_t warmed_count = 0;
    for (const auto &item : pending) {
      alignas(std::max_align_t) unsigned char attributes[1024] = {};
      if (real_attributes(attributes, item.host_function) == 0) {
        ++warmed_count;
        std::lock_guard<std::mutex> lock(
            lupine_registered_runtime_function_mutex());
        for (auto &entry : lupine_registered_runtime_functions()) {
          if (entry.fatbin == item.fatbin) entry.warmed = true;
        }
      }
    }
    warming = false;
    if (std::getenv("LUPINE_CUDART_COMPAT_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "lupine-cudart: preloaded %zu/%zu registered fatbins\n",
                   warmed_count, pending.size());
    }
  }
  return real_mem_info == nullptr ? 999
                                  : real_mem_info(free_bytes, total_bytes);
}

// CUDA 13 NCCL requires the query-status output from this Runtime API. With an
// interposed libcuda, libcudart falls back to the legacy four-argument Driver
// entry point: it obtains a valid pointer but leaves status as "not found".
// This preloadable bridge calls the five-argument Driver ABI explicitly.
extern "C" int cudaGetDriverEntryPointByVersion(
    const char *symbol, void **pfn, unsigned int cudaVersion,
    unsigned long long flags, int *driverStatus) {
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult result = cuGetProcAddress_v2(
      symbol, pfn, static_cast<int>(cudaVersion),
      static_cast<cuuint64_t>(flags), &status);
  if (driverStatus != nullptr) {
    *driverStatus = static_cast<int>(status);
  }
  return result == CUDA_SUCCESS ? 0 : 999; // cudaSuccess / cudaErrorUnknown
}

extern "C" int cudaGetDriverEntryPointByVersion_ptsz(
    const char *symbol, void **pfn, unsigned int cudaVersion,
    unsigned long long flags, int *driverStatus) {
  return cudaGetDriverEntryPointByVersion(symbol, pfn, cudaVersion, flags,
                                          driverStatus);
}

// libcudart's process-global current-device query can return
// cudaErrorDevicesUnavailable after alternating between primary contexts that
// live in local and remote driver processes. The interposed Driver API already
// owns the authoritative routed current context and maps its CUdevice back to
// the virtual ordinal, so use that whenever a context is current. Preserve the
// real Runtime behavior (including its default-device initialization) when no
// context exists yet.
extern "C" int cudaGetDevice(int *device) {
  if (device == nullptr) return 1; // cudaErrorInvalidValue
  bool debug = std::getenv("LUPINE_CUDART_COMPAT_DEBUG") != nullptr;
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr) {
    CUdevice current = -1;
    if (cuCtxGetDevice(&current) == CUDA_SUCCESS) {
      *device = static_cast<int>(current);
      if (debug)
        std::fprintf(stderr, "lupine-cudart: cudaGetDevice driver=%d\n",
                     *device);
      return 0; // cudaSuccess
    }
  }
  if (lupine_runtime_device >= 0) {
    if (context != nullptr &&
        lupine_repair_current_context_device(lupine_runtime_device) == 0) {
      *device = lupine_runtime_device;
      if (debug)
        std::fprintf(stderr,
                     "lupine-cudart: cudaGetDevice repaired-owner=%d\n",
                     *device);
      return 0;
    }
    if (lupine_restore_default_context_hint() == 0 &&
        cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr) {
      CUdevice current = -1;
      if (cuCtxGetDevice(&current) == CUDA_SUCCESS) {
        *device = static_cast<int>(current);
        if (debug)
          std::fprintf(stderr,
                       "lupine-cudart: cudaGetDevice restored-hint=%d\n",
                       *device);
        return 0;
      }
    }
    using set_device_fn = int (*)(int);
    using route_fn = int (*)(int);
    static set_device_fn real_set =
        reinterpret_cast<set_device_fn>(dlsym(RTLD_NEXT, "cudaSetDevice"));
    static route_fn route = reinterpret_cast<route_fn>(
        dlsym(RTLD_DEFAULT, "lupine_cuda_device_route_id"));
    int repair_result = 999;
    if (real_set != nullptr) {
      // libcudart can reject re-entering a remote primary context directly
      // after a CUDA library clears it. Passing through a local ordinal first
      // resets its thread-local primary-context bookkeeping.
      if (route != nullptr && route(lupine_runtime_device) >= 0) {
        for (int ordinal = 0; ordinal < 64; ++ordinal) {
          if (route(ordinal) == -1) {
            (void)real_set(ordinal);
            break;
          }
          if (route(ordinal) < -1) break;
        }
      }
      repair_result = real_set(lupine_runtime_device);
    }
    *device = lupine_runtime_device;
    if (debug)
      std::fprintf(stderr,
                   "lupine-cudart: cudaGetDevice repaired=%d result=%d\n",
                   *device, repair_result);
    return repair_result;
  }
  using get_device_fn = int (*)(int *);
  static get_device_fn real_get =
      reinterpret_cast<get_device_fn>(dlsym(RTLD_NEXT, "cudaGetDevice"));
  int result = real_get == nullptr ? 999 : real_get(device);
  if (debug)
    std::fprintf(stderr, "lupine-cudart: cudaGetDevice runtime=%d device=%d\n",
                 result, result == 0 ? *device : -1);
  return result;
}

// CUDA's private runtime state is process-wide, while a LUPINE mixed process
// combines contexts created by two driver processes. CUDA 13 accepts that
// arrangement when the remote runtime context is initialized first, but a
// local-first allocation leaves later remote allocation with
// cudaErrorDevicesUnavailable. Prime one remote context before the first local
// cudaSetDevice in a non-torchrun mixed process. Keep a one-byte allocation
// alive for the process lifetime: releasing the final remote Runtime allocation
// lets libcudart discard the state that makes the later mixed transition safe.
extern "C" int cudaSetDevice(int device) {
  using set_device_fn = int (*)(int);
  using malloc_fn = int (*)(void **, size_t);
  using memset_fn = int (*)(void *, int, size_t);
  using synchronize_fn = int (*)();
  using route_fn = int (*)(int);
  static set_device_fn real_set =
      reinterpret_cast<set_device_fn>(dlsym(RTLD_NEXT, "cudaSetDevice"));
  static malloc_fn real_malloc =
      reinterpret_cast<malloc_fn>(dlsym(RTLD_NEXT, "cudaMalloc"));
  static memset_fn real_memset =
      reinterpret_cast<memset_fn>(dlsym(RTLD_NEXT, "cudaMemset"));
  static synchronize_fn real_synchronize = reinterpret_cast<synchronize_fn>(
      dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
  static route_fn route = reinterpret_cast<route_fn>(
      dlsym(RTLD_DEFAULT, "lupine_cuda_device_route_id"));
  static std::once_flag prime_once;
  static void *remote_runtime_anchor = nullptr;
  static thread_local bool priming = false;

  if (real_set == nullptr) return 999; // cudaErrorUnknown
  if (priming) return real_set(device);
  const char *local_rank = std::getenv("LOCAL_RANK");
  bool rank_process = local_rank != nullptr && local_rank[0] != '\0';
  bool debug = std::getenv("LUPINE_CUDART_COMPAT_DEBUG") != nullptr;
  if (debug) {
    std::fprintf(stderr, "lupine-cudart: cudaSetDevice(%d) route=%d rank=%d\n",
                 device, route == nullptr ? -99 : route(device), rank_process);
  }
  if (!rank_process && route != nullptr && route(device) == -1) {
    std::call_once(prime_once, [&] {
      int remote_device = -1;
      for (int ordinal = 0; ordinal < 64; ++ordinal) {
        int route_id = route(ordinal);
        if (route_id >= 0) {
          remote_device = ordinal;
          break;
        }
        if (route_id < -1) break;
      }
      if (remote_device < 0 || real_malloc == nullptr)
        return;
      priming = true;
      int remote_set_result = real_set(remote_device);
      if (debug) {
        std::fprintf(stderr,
                     "lupine-cudart: prime remote=%d set_result=%d\n",
                     remote_device, remote_set_result);
      }
      if (remote_set_result == 0) {
        void *pointer = nullptr;
        int malloc_result = real_malloc(&pointer, 1);
        if (debug) {
          std::fprintf(stderr,
                       "lupine-cudart: prime malloc_result=%d pointer=%p\n",
                       malloc_result, pointer);
        }
        if (malloc_result == 0 && pointer != nullptr) {
          remote_runtime_anchor = pointer;
          int memset_result = real_memset == nullptr
                                  ? 999
                                  : real_memset(pointer, 0, 1);
          int launch_result = lupine_launch_runtime_anchor_kernel();
          int synchronize_result = real_synchronize == nullptr
                                       ? 999
                                       : real_synchronize();
          if (debug) {
            std::fprintf(stderr,
                         "lupine-cudart: prime memset=%d launch=%d "
                         "synchronize=%d\n",
                         memset_result, launch_result, synchronize_result);
          }
        }
      }
      int restore_result = real_set(device);
      if (debug) {
        std::fprintf(stderr, "lupine-cudart: prime restore=%d result=%d\n",
                     device, restore_result);
      }
      priming = false;
    });
  }
  int result = real_set(device);
  if (result == 0) lupine_runtime_device = device;
  if (debug) {
    std::fprintf(stderr, "lupine-cudart: cudaSetDevice(%d) result=%d\n",
                 device, result);
  }
  return result;
}
