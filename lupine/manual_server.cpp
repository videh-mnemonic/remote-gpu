#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <errno.h>
#if defined(__linux__)
#include <dlfcn.h>
#include <sys/mman.h> // memfd_create
#endif
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <vector>

#include <list>
#include <map>

#include "cuda_compat.h"

#include "cache.h"
#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "copy_pipeline.h"
#include "ipc.h"
#include "lupine_attr_sizes.h"
#include "lupine_fatbin.h"
#include "lupine_log.h"
#include "lupine_cublas.h"
#include "lupine_nccl.h"
#include "lupine_tensormap.h"
#include "manual_server.h"
#include "rpc.h"

#ifdef _WIN32
#include <io.h>
#define lupine_ipc_close_fd _close
#else
#include <unistd.h>
#define lupine_ipc_close_fd close
#endif
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

#if CUDA_VERSION < 12020
#ifdef CU_MEM_LOCATION_TYPE_HOST
static constexpr CUmemLocationType LUPINE_CU_MEM_LOCATION_TYPE_HOST =
    CU_MEM_LOCATION_TYPE_HOST;
#else
static constexpr CUmemLocationType LUPINE_CU_MEM_LOCATION_TYPE_HOST =
    static_cast<CUmemLocationType>(2);
#endif
#endif

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;

// The CUDA linker requires output option buffers to remain valid for the
// lifetime of CUlinkState. The mutex serializes cuLinkDestroy against a
// concurrent cuLinkComplete on another RPC lane: the cubin buffer returned by
// cuLinkComplete is driver-owned and freed by cuLinkDestroy, so it must stay
// locked out until the response holding that buffer has been flushed.
struct lupine_link_state {
  CUlinkState cuda_state = nullptr;
  rpc_jit_server_state jit;
  std::mutex mutex;
};

static lupine_link_state *lupine_link_state_from_handle(CUlinkState state) {
  return reinterpret_cast<lupine_link_state *>(state);
}

static CUlinkState lupine_link_state_to_handle(lupine_link_state *state) {
  return reinterpret_cast<CUlinkState>(state);
}

#ifdef _WIN32
static constexpr size_t LUPINE_HTOD_CHUNK_BYTES = 64 * 1024 * 1024;

struct lupine_deferred_host_free {
  void *ptr = nullptr;
  lupine_deferred_host_free *next = nullptr;
};

struct lupine_host_free_queue {
  std::mutex mutex;
  std::condition_variable condition;
  lupine_deferred_host_free *head = nullptr;
  lupine_deferred_host_free *tail = nullptr;
};

static lupine_host_free_queue &lupine_host_frees() {
  // The server process owns this detached worker until exit. Keep its queue
  // alive for the same lifetime so static destruction cannot race the worker.
  static auto *queue = new lupine_host_free_queue();
  return *queue;
}

static void lupine_reap_host_frees() {
  auto &queue = lupine_host_frees();
  for (;;) {
    lupine_deferred_host_free *item = nullptr;
    {
      std::unique_lock<std::mutex> lock(queue.mutex);
      queue.condition.wait(lock, [&queue] { return queue.head != nullptr; });
      item = queue.head;
      queue.head = item->next;
      if (queue.head == nullptr) {
        queue.tail = nullptr;
      }
    }

    CUresult result = cuMemFreeHost(item->ptr);
    if (result != CUDA_SUCCESS) {
      LUPINE_LOG_ERROR("Deferred cuMemFreeHost failed for " << item->ptr << ": "
                                                            << result);
    }
    delete item;
  }
}

static bool lupine_start_host_free_reaper() {
  static std::once_flag once;
  try {
    (void)lupine_host_frees();
    std::call_once(once, [] { std::thread(lupine_reap_host_frees).detach(); });
    return true;
  } catch (...) {
    return false;
  }
}

static void CUDA_CB lupine_queue_host_free(void *userData) {
  // CUDA host functions cannot call CUDA APIs. Hand the completed allocation
  // to a normal host thread before calling cuMemFreeHost.
  auto *item = static_cast<lupine_deferred_host_free *>(userData);
  auto &queue = lupine_host_frees();
  {
    std::lock_guard<std::mutex> lock(queue.mutex);
    if (queue.tail == nullptr) {
      queue.head = item;
    } else {
      queue.tail->next = item;
    }
    queue.tail = item;
  }
  queue.condition.notify_one();
}

static CUresult lupine_defer_host_free(CUstream stream, void *ptr) {
  auto *item = new (std::nothrow) lupine_deferred_host_free{ptr, nullptr};
  if (item == nullptr || !lupine_start_host_free_reaper()) {
    delete item;
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  CUresult result = cuLaunchHostFunc(stream, lupine_queue_host_free, item);
  if (result != CUDA_SUCCESS) {
    delete item;
  }
  return result;
}
#endif

struct lupine_captured_stdout {
  int saved_stdout = -1;
  bool active = false;
  std::string output;
};

static pthread_mutex_t lupine_stdout_capture_mutex = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> lupine_stdout_capture_required{false};

static bool lupine_image_contains(const unsigned char *image, size_t image_size,
                                  const char *needle, size_t needle_size) {
  return image != nullptr && needle != nullptr && needle_size != 0 &&
         image_size >= needle_size &&
         std::search(image, image + image_size, needle, needle + needle_size) !=
             image + image_size;
}

static bool lupine_image_may_use_device_stdout(const unsigned char *image,
                                               size_t image_size) {
  static constexpr char vprintf_symbol[] = "vprintf";
  if (lupine_image_contains(image, image_size, vprintf_symbol,
                            sizeof(vprintf_symbol) - 1)) {
    return true;
  }

  // PTX names vprintf directly and cubins retain it in their symbol data. A
  // fatbin whose members are all compressed exposes neither representation,
  // so keep capture enabled for that unknown case rather than dropping output.
  uint32_t magic = 0;
  if (image_size >= sizeof(magic)) {
    memcpy(&magic, image, sizeof(magic));
  }
  static constexpr char elf_magic[] = "\177ELF";
  static constexpr char ptx_version[] = ".version";
  return magic == LUPINE_FATBIN_MAGIC &&
         !lupine_image_contains(image, image_size, elf_magic,
                                sizeof(elf_magic) - 1) &&
         !lupine_image_contains(image, image_size, ptx_version,
                                sizeof(ptx_version) - 1);
}

static void lupine_note_device_stdout_image(const unsigned char *image,
                                            size_t image_size) {
  if (lupine_image_may_use_device_stdout(image, image_size)) {
    lupine_stdout_capture_required.store(true, std::memory_order_release);
  }
}

// Device printf output is drained by the CUDA driver as a write to fd 1
// (process stdout) during synchronization (see issue #294). We capture it by
// temporarily redirecting fd 1 to a backing file we can read back. The lupine
// server writes all of its own diagnostics to stderr, so fd 1 is exclusively
// the device-printf channel and nothing else can contaminate the capture.
//
// This returns a single process-global, reusable backing file: created once
// on first use and kept open for the process lifetime, so the per-synchronize
// hot path performs no filesystem open/close. On Linux it is an anonymous
// in-memory file from memfd_create() (no path, no /tmp, no inode, no page
// cache of a real file); other platforms (and old kernels without memfd)
// fall back to a single tmpfile() created once. The file is reset (truncated
// to empty) at the start of each capture.
static FILE *lupine_stdout_capture_file() {
  static FILE *file = []() -> FILE * {
#if defined(__linux__)
    int fd = memfd_create("lupine-stdout-capture", MFD_CLOEXEC);
    if (fd >= 0) {
      FILE *f = fdopen(fd, "w+");
      if (f != nullptr) {
        return f;
      }
      // fdopen failed; reclaim the fd and fall through to tmpfile().
      close(fd);
    }
#endif
    return tmpfile();
  }();
  return file;
}

static bool lupine_start_stdout_capture(lupine_captured_stdout *capture) {
  if (capture == nullptr) {
    return false;
  }
  capture->saved_stdout = -1;
  capture->active = false;
  capture->output.clear();

  // Redirecting fd 1 is process-global, so only pay the serialization and
  // syscall cost after a loaded image has shown that device stdout may be used.
  if (!lupine_stdout_capture_required.load(std::memory_order_acquire)) {
    return false;
  }

  FILE *capture_file = lupine_stdout_capture_file();
  if (capture_file == nullptr) {
    return false;
  }
  int capture_fd = lupine_fd_fileno(capture_file);
  if (capture_fd < 0) {
    return false;
  }

  if (pthread_mutex_lock(&lupine_stdout_capture_mutex) != 0) {
    return false;
  }

  fflush(stdout);
  std::cout.flush();

  // Reset the reused backing file to empty so this capture only contains
  // output produced during the synchronization below.
  if (lupine_fd_truncate(capture_fd, 0) != 0 ||
      lupine_fd_seek(capture_fd, 0, SEEK_SET) < 0) {
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  capture->saved_stdout = lupine_fd_dup(LUPINE_STDOUT_FD);
  if (capture->saved_stdout < 0) {
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  if (lupine_fd_dup2(capture_fd, LUPINE_STDOUT_FD) < 0) {
    lupine_fd_close(capture->saved_stdout);
    capture->saved_stdout = -1;
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  capture->active = true;
  return true;
}

static void lupine_finish_stdout_capture(lupine_captured_stdout *capture) {
  if (capture == nullptr || !capture->active) {
    return;
  }

  fflush(stdout);
  std::cout.flush();
  lupine_fd_dup2(capture->saved_stdout, LUPINE_STDOUT_FD);
  lupine_fd_close(capture->saved_stdout);
  capture->saved_stdout = -1;

  // The backing file is process-global and reused, so read it back without
  // closing it. Its extent is exactly the bytes written during this capture
  // (it was truncated to empty on entry).
  FILE *capture_file = lupine_stdout_capture_file();
  if (capture_file != nullptr) {
    int capture_fd = lupine_fd_fileno(capture_file);
    if (capture_fd >= 0 && lupine_fd_seek(capture_fd, 0, SEEK_SET) >= 0) {
      char buffer[4096];
      for (;;) {
        ssize_t bytes = lupine_fd_read(capture_fd, buffer, sizeof(buffer));
        if (bytes > 0) {
          capture->output.append(buffer, static_cast<size_t>(bytes));
          continue;
        }
        if (bytes == 0) {
          break;
        }
        if (errno == EINTR) {
          continue;
        }
        break;
      }
    }
  }
  capture->active = false;
  pthread_mutex_unlock(&lupine_stdout_capture_mutex);
}

static int lupine_write_captured_stdout(conn_t *conn,
                                        const lupine_captured_stdout &capture,
                                        uint64_t *output_size) {
  if (output_size == nullptr) {
    return -1;
  }
  *output_size = capture.output.size();
  if (rpc_write(conn, output_size, sizeof(*output_size)) < 0) {
    return -1;
  }
  if (*output_size != 0 &&
      rpc_write(conn, capture.output.data(), capture.output.size()) < 0) {
    return -1;
  }
  return 0;
}

struct lupine_private_module_node_capture {
  void *node = nullptr;
  uint64_t owner = 0;
  uint64_t count = 0;
};

static libcuckoo::cuckoohash_map<CUmodule, CUlibrary> &
lupine_module_libraries() {
  static auto *libraries = new libcuckoo::cuckoohash_map<CUmodule, CUlibrary>();
  return *libraries;
}

struct lupine_graph_host_copy {
  void *client_dst = nullptr;
  void *server_src = nullptr;
  size_t bytes = 0;
};

struct lupine_pending_dtoh_copy {
  CUstream stream = nullptr;
  void *client_dst = nullptr;
  void *server_src = nullptr;
  size_t bytes = 0;
  bool pinned = false;
};

struct lupine_graph_resources;

struct lupine_host_callback_data {
  conn_t *conn = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  lupine_graph_resources *resources = nullptr;
};

struct lupine_stream_callback_data {
  conn_t *conn = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
};

struct lupine_graph_host_copy_node {
  explicit lupine_graph_host_copy_node(lupine_graph_host_copy node_copy)
      : copy(node_copy) {}

  lupine_graph_host_copy copy;
  lupine_graph_host_copy_node *next = nullptr;
};

struct lupine_graph_capture_scratch {
  lupine_graph_capture_scratch(void *scratch_ptr, size_t scratch_size)
      : ptr(scratch_ptr), size(scratch_size) {}

  void *ptr;
  size_t size;
  std::atomic<size_t> offset{0};
};

struct lupine_graph_resources {
  void add_dtoh_copy(lupine_graph_host_copy copy) {
    auto *node = new lupine_graph_host_copy_node(copy);
    node->next = dtoh_copies.load(std::memory_order_relaxed);
    while (!dtoh_copies.compare_exchange_weak(node->next, node,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
    }
  }

  std::vector<lupine_graph_host_copy> dtoh_copy_snapshot() const {
    std::vector<lupine_graph_host_copy> copies;
    for (auto *node = dtoh_copies.load(std::memory_order_acquire);
         node != nullptr; node = node->next) {
      copies.push_back(node->copy);
    }
    std::reverse(copies.begin(), copies.end());
    return copies;
  }

  bool has_capture_scratch() const {
    return capture_scratch.load(std::memory_order_acquire) != nullptr;
  }

  bool install_capture_scratch(void *scratch, size_t size) {
    if (scratch == nullptr) {
      return false;
    }
    auto *candidate = new lupine_graph_capture_scratch(scratch, size);
    lupine_graph_capture_scratch *expected = nullptr;
    if (!capture_scratch.compare_exchange_strong(expected, candidate,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
      delete candidate;
      return false;
    }
    return true;
  }

  void *allocate_capture_scratch(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    auto *scratch = capture_scratch.load(std::memory_order_acquire);
    if (scratch == nullptr) {
      return nullptr;
    }
    size_t current = scratch->offset.load(std::memory_order_relaxed);
    for (;;) {
      if (current > scratch->size || current > SIZE_MAX - 255) {
        return nullptr;
      }
      size_t aligned = (current + 255) & ~size_t(255);
      if (aligned > scratch->size || bytes > scratch->size - aligned) {
        return nullptr;
      }
      if (scratch->offset.compare_exchange_weak(current, aligned + bytes,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
        return static_cast<unsigned char *>(scratch->ptr) + aligned;
      }
    }
  }

  std::atomic<lupine_graph_host_copy_node *> dtoh_copies{nullptr};
  std::atomic<lupine_graph_capture_scratch *> capture_scratch{nullptr};
};

// Graph host buffers and callback metadata must remain valid for any queued
// launch or callback. The per-connection process already owned them until exit;
// stable raw pointers make that lifetime explicit and avoid reference-count
// synchronization on every map access.
static libcuckoo::cuckoohash_map<CUgraph, lupine_graph_resources *> &
lupine_graph_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraph, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUgraphExec, lupine_graph_resources *> &
lupine_graph_exec_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraphExec, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *> &
lupine_stream_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *> &
lupine_event_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *>();
  return *resources;
}

using lupine_pending_dtoh_streams =
    std::unordered_map<CUstream, std::vector<lupine_pending_dtoh_copy>>;

static libcuckoo::cuckoohash_map<conn_t *, lupine_pending_dtoh_streams> &
lupine_pending_dtoh_copies() {
  static libcuckoo::cuckoohash_map<conn_t *, lupine_pending_dtoh_streams>
      copies;
  return copies;
}

static lupine_graph_resources *lupine_get_graph_resources(CUgraph graph) {
  auto *candidate = new lupine_graph_resources();
  auto *resources = candidate;
  lupine_graph_resource_map().upsert(
      graph,
      [&resources](lupine_graph_resources *&existing,
                   libcuckoo::UpsertContext) { resources = existing; },
      candidate);
  if (resources != candidate) {
    delete candidate;
  }
  return resources;
}

static lupine_graph_resources *lupine_get_stream_resources(CUstream stream) {
  auto *candidate = new lupine_graph_resources();
  auto *resources = candidate;
  lupine_stream_capture_resource_map().upsert(
      stream,
      [&resources](lupine_graph_resources *&existing,
                   libcuckoo::UpsertContext) { resources = existing; },
      candidate);
  if (resources != candidate) {
    delete candidate;
  }
  return resources;
}

static lupine_graph_resources *
lupine_begin_stream_capture_resources(CUstream stream) {
  auto *resources = new lupine_graph_resources();
  // Each capture owns distinct host-copy scratch and metadata. Reusing the
  // previous capture's object makes a synchronize publish D2H outputs from
  // every graph ever captured on this stream, including graphs not launched.
  lupine_stream_capture_resource_map().insert_or_assign(stream, resources);
  return resources;
}

static uint64_t lupine_fnv1a64(const void *data, size_t size) {
  static constexpr uint64_t kOffset = 14695981039346656037ull;
  static constexpr uint64_t kPrime = 1099511628211ull;
  const auto *bytes = static_cast<const unsigned char *>(data);
  uint64_t hash = kOffset;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

static uint64_t lupine_export_slot_hash(const void *fn) {
  if (fn == nullptr) {
    return 0;
  }
#ifdef _WIN32
  MEMORY_BASIC_INFORMATION info = {};
  if (VirtualQuery(fn, &info, sizeof(info)) == 0 ||
      info.AllocationBase == nullptr) {
    return 0;
  }
#else
  Dl_info info = {};
  if (dladdr(fn, &info) == 0 || info.dli_fname == nullptr) {
    return 0;
  }
#endif
  return lupine_fnv1a64(fn, 32);
}

int handle_manual_cuGetExportTableMetadata(conn_t *conn) {
  CUuuid uuid = {};
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  uint64_t byte_size = 0;
  uint32_t slot_count = 0;
  uint32_t trusted = 0;
  uint64_t hashes[LUPINE_PRIVATE_EXPORT_MAX_SLOTS] = {};

  if (rpc_read(conn, uuid.bytes, sizeof(uuid.bytes)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  const void *export_table = nullptr;
  cuInit(0);
  result = cuGetExportTable(&export_table, &uuid);
  if (result == CUDA_SUCCESS && export_table != nullptr) {
    const auto *slots = static_cast<const void *const *>(export_table);
    byte_size = reinterpret_cast<uintptr_t>(slots[0]);
    if (byte_size >= sizeof(void *) && byte_size % sizeof(void *) == 0 &&
        byte_size / sizeof(void *) <= LUPINE_PRIVATE_EXPORT_MAX_SLOTS) {
      trusted = 1;
      slot_count = static_cast<uint32_t>(byte_size / sizeof(void *));
      for (uint32_t i = 1; i < slot_count; ++i) {
        hashes[i] = lupine_export_slot_hash(slots[i]);
      }
    }
  }

  LUPINE_TRACE_LOG("LUPINE server cuGetExportTable metadata result="
                   << result << " bytes=" << byte_size
                   << " slots=" << slot_count << " trusted=" << trusted);

  size_t hash_bytes = static_cast<size_t>(slot_count) * sizeof(uint64_t);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &byte_size, sizeof(byte_size)) < 0 ||
      rpc_write(conn, &slot_count, sizeof(slot_count)) < 0 ||
      rpc_write(conn, &trusted, sizeof(trusted)) < 0 ||
      (hash_bytes != 0 && rpc_write(conn, hashes, hash_bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static void lupine_private_module_node_callback(void *opaque, void *node,
                                                uint64_t owner) {
  auto *capture = static_cast<lupine_private_module_node_capture *>(opaque);
  if (capture == nullptr || capture->node != nullptr) {
    return;
  }
  capture->node = node;
  capture->owner = owner;
  capture->count = 1;
}

int handle_manual_cuPrivateGetModuleNode(conn_t *conn) {
  static constexpr unsigned char PRIVATE_MODULE_ITERATOR_UUID[16] = {
      0x6e, 0x16, 0x3f, 0xbe, 0xb9, 0x58, 0x44, 0x4d,
      0x83, 0x5c, 0xe1, 0x82, 0xaf, 0xf1, 0x99, 0x1e};

  CUcontext context = nullptr;
  CUmodule module = nullptr;
  int request_id;
  CUfunction node = nullptr;
  uint64_t owner = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &context, sizeof(context)) < 0 ||
      rpc_read(conn, &module, sizeof(module)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUuuid uuid = {};
  memcpy(uuid.bytes, PRIVATE_MODULE_ITERATOR_UUID, sizeof(uuid.bytes));
  const void *export_table = nullptr;
  result = cuGetExportTable(&export_table, &uuid);
  if (result == CUDA_SUCCESS && export_table != nullptr) {
    const auto *slots = static_cast<const void *const *>(export_table);
    size_t byte_size = reinterpret_cast<uintptr_t>(slots[0]);
    if (byte_size <= 7 * sizeof(void *) || slots[7] == nullptr) {
      result = CUDA_ERROR_NOT_FOUND;
    } else {
      using private_module_iterator = uint64_t (*)(
          uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
      auto iterator = reinterpret_cast<private_module_iterator>(
          const_cast<void *>(slots[7]));
      lupine_private_module_node_capture capture;
      CUcontext previous = nullptr;
      cuCtxGetCurrent(&previous);
      if (context != nullptr) {
        cuCtxSetCurrent(context);
      }
      uint64_t count = iterator(
          reinterpret_cast<uint64_t>(context),
          reinterpret_cast<uint64_t>(module),
          reinterpret_cast<uint64_t>(&lupine_private_module_node_callback),
          reinterpret_cast<uint64_t>(&capture),
          reinterpret_cast<uint64_t>(module), 0);
      if (previous != context) {
        cuCtxSetCurrent(previous);
      }
      if (capture.node != nullptr) {
        node = reinterpret_cast<CUfunction>(capture.node);
        owner = capture.owner;
        result = CUDA_SUCCESS;
      } else {
        result = count == 0 ? CUDA_ERROR_NOT_FOUND : CUDA_ERROR_UNKNOWN;
      }
      LUPINE_TRACE_LOG("LUPINE server private module node module="
                       << module << " context=" << context << " count=" << count
                       << " node=" << node
                       << " owner=" << reinterpret_cast<void *>(owner)
                       << " result=" << static_cast<int>(result));
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &node, sizeof(node)) < 0 ||
      rpc_write(conn, &owner, sizeof(owner)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static size_t lupine_memcpy3d_host_span_bytes(const CUDA_MEMCPY3D &params,
                                              bool source) {
  size_t width = params.WidthInBytes;
  size_t height = params.Height == 0 ? 1 : params.Height;
  size_t depth = params.Depth == 0 ? 1 : params.Depth;
  size_t pitch = source ? params.srcPitch : params.dstPitch;
  if (pitch == 0) {
    pitch = width;
  }
  return pitch * height * depth;
}

static int lupine_read_graph_dependencies(conn_t *conn,
                                          std::vector<CUgraphNode> *deps) {
  size_t count = 0;
  if (deps == nullptr || rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  deps->resize(count);
  if (count != 0 &&
      rpc_read(conn, deps->data(), count * sizeof(CUgraphNode)) < 0) {
    return -1;
  }
  return 0;
}

static void *lupine_alloc_process_host_buffer(size_t bytes) {
  void *ptr = nullptr;
  if (bytes == 0) {
    return nullptr;
  }
  if (cuMemAllocHost(&ptr, bytes) != CUDA_SUCCESS) {
    return malloc(bytes);
  }
  return ptr;
}

static std::vector<lupine_pending_dtoh_copy>
lupine_detach_pending_dtoh_copies(conn_t *conn, CUstream stream,
                                  bool all_streams) {
  std::vector<lupine_pending_dtoh_copy> copies;
  lupine_pending_dtoh_copies().erase_fn(
      conn, [&](lupine_pending_dtoh_streams &streams) {
        if (all_streams) {
          for (auto &entry : streams) {
            auto &stream_copies = entry.second;
            copies.insert(copies.end(), stream_copies.begin(),
                          stream_copies.end());
          }
          return true;
        }

        auto stream_it = streams.find(stream);
        if (stream_it == streams.end()) {
          return false;
        }
        copies.swap(stream_it->second);
        streams.erase(stream_it);
        return streams.empty();
      });
  return copies;
}

static int lupine_write_pending_dtoh_copies(
    uint32_t *copy_count, conn_t *conn,
    const std::vector<lupine_pending_dtoh_copy> &pending) {
  if (copy_count != nullptr) {
    *copy_count = static_cast<uint32_t>(pending.size());
    if (rpc_write(conn, copy_count, sizeof(*copy_count)) < 0) {
      return -1;
    }
  }
  for (const auto &copy : pending) {
    if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
        rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
        (copy.bytes != 0 &&
         rpc_write_payload(conn, copy.server_src, copy.bytes) < 0)) {
      return -1;
    }
  }
  return 0;
}

static void lupine_cleanup_pending_dtoh_copies(
    std::vector<lupine_pending_dtoh_copy> *pending) {
  if (pending == nullptr) {
    return;
  }
  for (auto &copy : *pending) {
    if (copy.server_src != nullptr) {
      if (copy.pinned) {
        cuMemFreeHost(copy.server_src);
      } else {
        free(copy.server_src);
      }
      copy.server_src = nullptr;
    }
  }
  pending->clear();
}

static void *lupine_alloc_capture_scratch(lupine_graph_resources *resources,
                                          size_t bytes) {
  if (resources == nullptr || bytes == 0) {
    return nullptr;
  }
  return resources->allocate_capture_scratch(bytes);
}

int handle_manual_cuModuleLoad(conn_t *conn) {
  CUmodule module = nullptr;
  size_t image_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size + 1, 0);
  if (image_size == 0 || rpc_read_payload(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuModuleLoadData(&module, image.data());
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image.data(), image_size);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuModuleLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUmodule module = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 || rpc_read_payload(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (kind == LUPINE_MODULE_IMAGE_FATBINC_V1 ||
      kind == LUPINE_MODULE_IMAGE_FATBINC_V2) {
    result = cuModuleLoadFatBinary(&module, image.data());
  } else if (kind == LUPINE_MODULE_IMAGE_FATBIN_RAW) {
    result = cuModuleLoadData(&module, image.data());
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image.data(), image.size());
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_lupineModuleGetFunctionWithLayout(conn_t *conn) {
  CUmodule module = nullptr;
  size_t name_len = 0;
  if (rpc_read(conn, &module, sizeof(module)) < 0 ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0 || name_len == 0) {
    return -1;
  }
  std::vector<char> name(name_len);
  if (rpc_read(conn, name.data(), name.size()) < 0 || name.back() != '\0') {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUfunction function = nullptr;
  lupine_kernel_param_layout layout;
  CUresult result = cuModuleGetFunction(&function, module, name.data());
  if (result == CUDA_SUCCESS) {
    result = lupine_get_kernel_param_layout(function, &layout);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &function, sizeof(function)) < 0 ||
      rpc_write_kernel_param_layout(conn, &layout) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUlibrary library = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  rpc_jit_server_state jit_state;
  bool has_library_option_values = false;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 || rpc_read_payload(conn, image.data(), image_size) < 0) {
    return -1;
  }
  if (rpc_read_jit_options(conn, &jit_state) < 0) {
    return -1;
  }
  std::vector<CUlibraryOption> library_options;
  std::vector<uintptr_t> library_raw_values;
  if (rpc_read_library_options(conn, &library_options, &library_raw_values,
                               &has_library_option_values) < 0) {
    return -1;
  }
  unsigned int num_library_options =
      static_cast<unsigned int>(library_options.size());
  std::vector<void *> library_option_values(num_library_options);
  for (unsigned int i = 0; i < num_library_options; ++i) {
    library_option_values[i] = reinterpret_cast<void *>(library_raw_values[i]);
    // The client-side image is not the buffer passed to CUDA on this process.
    // Clear the preservation hint so the driver retains its own copy rather
    // than requiring a global library-to-image lifetime table.
    if (library_options[i] == CU_LIBRARY_BINARY_IS_PRESERVED) {
      library_option_values[i] = nullptr;
    }
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUjit_option *jit_opts =
      jit_state.options.empty() ? nullptr : jit_state.options.data();
  void **jit_vals = jit_state.option_values.empty()
                        ? nullptr
                        : jit_state.option_values.data();
  CUlibraryOption *lib_opts =
      library_options.empty() ? nullptr : library_options.data();
  void **lib_vals = !has_library_option_values || library_option_values.empty()
                        ? nullptr
                        : library_option_values.data();
  unsigned int num_jit_options =
      static_cast<unsigned int>(jit_state.options.size());

  if (kind == LUPINE_MODULE_IMAGE_FATBINC_V1 ||
      kind == LUPINE_MODULE_IMAGE_FATBINC_V2) {
    lupine_fatbin_wrapper wrapper = {
        LUPINE_FATBINC_MAGIC,
        kind == LUPINE_MODULE_IMAGE_FATBINC_V2 ? 2U : 1U,
        image.data(),
        nullptr,
    };
    result = cuLibraryLoadData(&library, &wrapper, jit_opts, jit_vals,
                               num_jit_options, lib_opts, lib_vals,
                               num_library_options);
  } else if (kind == LUPINE_MODULE_IMAGE_FATBIN_RAW) {
    result = cuLibraryLoadData(&library, image.data(), jit_opts, jit_vals,
                               num_jit_options, lib_opts, lib_vals,
                               num_library_options);
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image.data(), image.size());
  }

  // The response carries every kernel in the library with its name and full
  // parameter layout, so the client can serve cuLibraryGetKernel and the
  // per-kernel param-info walk from cache instead of one round trip per
  // query. Enumeration failures degrade to an empty table; the client then
  // falls back to the per-call RPCs. Records are built before any rpc_write
  // because queued iovecs are only transmitted at rpc_write_end.
  struct kernel_record {
    std::string name;
    CUkernel kernel = nullptr;
    uint32_t name_len = 0;
    uint32_t param_count = 0;
    std::vector<uint64_t> params;
  };
  std::vector<kernel_record> records;
#if CUDA_VERSION >= 12040
  if (result == CUDA_SUCCESS) {
    unsigned int kernel_count = 0;
    std::vector<CUkernel> kernels;
    if (cuLibraryGetKernelCount(&kernel_count, library) == CUDA_SUCCESS &&
        kernel_count != 0) {
      kernels.resize(kernel_count);
      if (cuLibraryEnumerateKernels(kernels.data(), kernel_count, library) !=
          CUDA_SUCCESS) {
        kernels.clear();
      }
    }
    for (CUkernel kernel : kernels) {
      const char *name = nullptr;
      if (kernel == nullptr || cuKernelGetName(&name, kernel) != CUDA_SUCCESS ||
          name == nullptr) {
        continue;
      }
      kernel_record record;
      record.name = name;
      record.name_len = static_cast<uint32_t>(record.name.size() + 1);
      record.kernel = kernel;
      for (size_t index = 0;; ++index) {
        size_t offset = 0;
        size_t size = 0;
        if (cuKernelGetParamInfo(kernel, index, &offset, &size) !=
            CUDA_SUCCESS) {
          break;
        }
        record.params.push_back(static_cast<uint64_t>(offset));
        record.params.push_back(static_cast<uint64_t>(size));
      }
      record.param_count = static_cast<uint32_t>(record.params.size() / 2);
      records.push_back(std::move(record));
    }
  }
#endif

  uint32_t table_count = static_cast<uint32_t>(records.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      rpc_write_jit_outputs(conn, &jit_state) < 0 ||
      rpc_write(conn, &table_count, sizeof(table_count)) < 0) {
    return -1;
  }
  for (const auto &record : records) {
    if (rpc_write(conn, &record.name_len, sizeof(record.name_len)) < 0 ||
        rpc_write(conn, record.name.c_str(), record.name_len) < 0 ||
        rpc_write(conn, &record.kernel, sizeof(record.kernel)) < 0 ||
        rpc_write(conn, &record.param_count, sizeof(record.param_count)) < 0 ||
        (record.param_count != 0 &&
         rpc_write(conn, record.params.data(),
                   record.params.size() * sizeof(uint64_t)) < 0)) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolSetAttribute(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemPool_attribute attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_mem_pool_attribute_size(attr, &expected_size) ||
      value_size != expected_size) {
    return -1;
  }

  std::vector<unsigned char> value(value_size);
  if (rpc_read(conn, value.data(), value_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuMemPoolSetAttribute(pool, attr, value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolGetAttribute(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemPool_attribute attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_mem_pool_attribute_size(attr, &expected_size) ||
      value_size != expected_size) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> value(value_size);
  result = cuMemPoolGetAttribute(pool, attr, value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, value.data(), value_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// POSIX-fd shareable handles cannot cross the wire as raw fd numbers. Export
// parks the real fd with the parent-process broker under a random token and
// returns the token; import redeems a token for the real fd (see ipc.h).

int handle_manual_cuMemExportToShareableHandle(conn_t *conn) {
  CUmemGenericAllocationHandle handle = 0;
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  lupine_ipc_token token = {};
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;

  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    int shareable_fd = -1;
    result =
        cuMemExportToShareableHandle(&shareable_fd, handle, handleType, flags);
    if (result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION,
                                        &token, shareable_fd) < 0) {
        result = CUDA_ERROR_UNKNOWN;
      }
      lupine_ipc_close_fd(shareable_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemImportFromShareableHandle(conn_t *conn) {
  lupine_ipc_token token = {};
  CUmemAllocationHandleType handleType;
  CUmemGenericAllocationHandle handle = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token);
    if (import_fd >= 0) {
      result = cuMemImportFromShareableHandle(
          &handle, reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd)),
          handleType);
      lupine_ipc_close_fd(import_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolExportToShareableHandle(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  lupine_ipc_token token = {};
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    int shareable_fd = -1;
    result = cuMemPoolExportToShareableHandle(&shareable_fd, pool, handleType,
                                              flags);
    if (result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token,
                                        shareable_fd) < 0) {
        result = CUDA_ERROR_UNKNOWN;
      }
      lupine_ipc_close_fd(shareable_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolImportFromShareableHandle(conn_t *conn) {
  lupine_ipc_token token = {};
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  CUmemoryPool pool = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token);
    if (import_fd >= 0) {
      result = cuMemPoolImportFromShareableHandle(
          &pool, reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd)),
          handleType, flags);
      lupine_ipc_close_fd(import_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerGetAttribute(conn_t *conn) {
  CUpointer_attribute attribute;
  CUdeviceptr ptr = 0;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  unsigned char value[64] = {};

  if (rpc_read(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_read(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_pointer_attribute_size(attribute, &expected_size) ||
      value_size != expected_size) {
    result = CUDA_ERROR_INVALID_VALUE;
    value_size = 0;
  } else if (value_size > sizeof(value)) {
    result = CUDA_ERROR_NOT_SUPPORTED;
    value_size = 0;
  } else {
    result = cuPointerGetAttribute(value, attribute, ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, value, value_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerSetAttribute(conn_t *conn) {
  CUpointer_attribute attribute;
  CUdeviceptr ptr = 0;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  unsigned char value[64] = {};

  if (rpc_read(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_read(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }
  // A payload larger than the largest pointer attribute cannot be a request
  // this server understands; drop the connection rather than leave unread
  // bytes desynchronizing the stream.
  if (value_size > sizeof(value)) {
    return -1;
  }
  if (value_size != 0 && rpc_read(conn, value, value_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_settable_pointer_attribute_size(attribute, &expected_size) ||
      value_size != expected_size) {
    result = CUDA_ERROR_INVALID_VALUE;
  } else {
    result = cuPointerSetAttribute(value, attribute, ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerGetAttributes(conn_t *conn) {
  unsigned int num_attributes = 0;
  CUdeviceptr ptr = 0;
  int request_id;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &num_attributes, sizeof(num_attributes)) < 0) {
    return -1;
  }
  std::vector<CUpointer_attribute> attributes(num_attributes);
  if (num_attributes != 0 &&
      rpc_read(conn, attributes.data(),
               num_attributes * sizeof(CUpointer_attribute)) < 0) {
    return -1;
  }
  if (rpc_read(conn, &ptr, sizeof(ptr)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<size_t> value_sizes(num_attributes, 0);
  std::vector<std::vector<unsigned char>> values(num_attributes);
  std::vector<void *> data(num_attributes, nullptr);
  for (unsigned int i = 0; i < num_attributes; ++i) {
    size_t value_size = 0;
    if (!lupine_pointer_attribute_size(attributes[i], &value_size)) {
      result = CUDA_ERROR_INVALID_VALUE;
      break;
    }
    value_sizes[i] = value_size;
    values[i].resize(value_size);
    data[i] = values[i].data();
  }

  if (result == CUDA_SUCCESS) {
    result = cuPointerGetAttributes(num_attributes, attributes.data(),
                                    data.data(), ptr);
  }
  if (result != CUDA_SUCCESS) {
    std::fill(value_sizes.begin(), value_sizes.end(), 0);
  }

  if (rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_attributes; ++i) {
    if (rpc_write(conn, &value_sizes[i], sizeof(value_sizes[i])) < 0 ||
        (value_sizes[i] != 0 &&
         rpc_write(conn, values[i].data(), value_sizes[i]) < 0)) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPrefetchAsync(conn_t *conn) {
  CUdeviceptr devPtr;
  size_t count;
  int location_type;
  int location_id;
  unsigned int flags;
  CUstream hStream;
  int request_id;
  CUresult result;
  if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      rpc_read(conn, &location_type, sizeof(location_type)) < 0 ||
      rpc_read(conn, &location_id, sizeof(location_id)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUmemLocation location = {};
  location.type = static_cast<CUmemLocationType>(location_type);
  location.id = location_id;
#if CUDA_VERSION >= 12020
  result = cuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream);
#else
  if (flags != 0 || (location.type != CU_MEM_LOCATION_TYPE_DEVICE &&
                     location.type != LUPINE_CU_MEM_LOCATION_TYPE_HOST)) {
    result = CUDA_ERROR_INVALID_VALUE;
  } else {
    CUdevice dstDevice = location.type == CU_MEM_LOCATION_TYPE_DEVICE
                             ? location.id
                             : CU_DEVICE_CPU;
    result = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
  }
#endif

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkCreate_v2(conn_t *conn) {
  auto link_state = std::make_unique<lupine_link_state>();
  CUlinkState client_state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read_jit_options(conn, &link_state->jit) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkCreate_v2(
      static_cast<unsigned int>(link_state->jit.options.size()),
      link_state->jit.options.empty() ? nullptr
                                      : link_state->jit.options.data(),
      link_state->jit.option_values.empty()
          ? nullptr
          : link_state->jit.option_values.data(),
      &link_state->cuda_state);
  if (result == CUDA_SUCCESS) {
    client_state = lupine_link_state_to_handle(link_state.release());
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &client_state, sizeof(client_state)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkAddData_v2(conn_t *conn) {
  CUlinkState state = nullptr;
  CUjitInputType type = CU_JIT_INPUT_PTX;
  size_t size = 0;
  size_t name_len = 0;
  rpc_jit_server_state jit_state;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> data(size);
  if ((size != 0 && rpc_read(conn, data.data(), size) < 0) ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0) {
    return -1;
  }
  std::vector<char> name(name_len == 0 ? 1 : name_len, '\0');
  if ((name_len != 0 && rpc_read(conn, name.data(), name_len) < 0) ||
      rpc_read_jit_options(conn, &jit_state) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state != nullptr) {
    result = cuLinkAddData_v2(
        link_state->cuda_state, type, data.data(), data.size(),
        name_len == 0 ? nullptr : name.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_jit_outputs(conn, &jit_state) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkAddFile_v2(conn_t *conn) {
  CUlinkState state = nullptr;
  CUjitInputType type = CU_JIT_INPUT_LIBRARY;
  size_t path_len = 0;
  uint8_t has_file_data = 0;
  uint64_t file_size = 0;
  rpc_jit_server_state jit_state;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      rpc_read(conn, &path_len, sizeof(path_len)) < 0) {
    return -1;
  }
  std::vector<char> path(path_len == 0 ? 1 : path_len, '\0');
  if ((path_len != 0 && rpc_read(conn, path.data(), path_len) < 0) ||
      rpc_read(conn, &has_file_data, sizeof(has_file_data)) < 0 ||
      rpc_read(conn, &file_size, sizeof(file_size)) < 0 ||
      file_size > (1ull << 32) || (file_size != 0 && has_file_data == 0)) {
    return -1;
  }
  std::vector<char> file_data;
  if (has_file_data != 0) {
    file_data.resize(static_cast<size_t>(file_size));
    if (!file_data.empty() &&
        rpc_read(conn, file_data.data(), file_data.size()) < 0) {
      return -1;
    }
  }
  if (rpc_read_jit_options(conn, &jit_state) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state == nullptr) {
    result = CUDA_ERROR_INVALID_HANDLE;
  } else if (!file_data.empty()) {
    result = cuLinkAddData_v2(
        link_state->cuda_state, type, file_data.data(), file_data.size(),
        path_len == 0 ? nullptr : path.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  } else {
    result = cuLinkAddFile_v2(
        link_state->cuda_state, type, path_len == 0 ? nullptr : path.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_jit_outputs(conn, &jit_state) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkComplete(conn_t *conn) {
  CUlinkState state = nullptr;
  void *cubin = nullptr;
  size_t size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  rpc_jit_server_state empty_jit_state;
  rpc_jit_server_state *jit_state = &empty_jit_state;
  // Held until the response is flushed: the cubin buffer belongs to the
  // driver's link state, and a concurrent cuLinkDestroy would free it while
  // the queued iovec still points at it.
  std::unique_lock<std::mutex> lock;
  if (link_state != nullptr) {
    lock = std::unique_lock<std::mutex>(link_state->mutex);
    result = cuLinkComplete(link_state->cuda_state, &cubin, &size);
    jit_state = &link_state->jit;
  }
  size_t returned_size = result == CUDA_SUCCESS ? size : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_size, sizeof(returned_size)) < 0 ||
      (returned_size != 0 && rpc_write(conn, cubin, returned_size) < 0) ||
      rpc_write_jit_outputs(conn, jit_state) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkDestroy(conn_t *conn) {
  CUlinkState state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state != nullptr) {
    // Wait for any cuLinkComplete on another lane to finish flushing the
    // driver-owned cubin buffer before cuLinkDestroy frees it.
    std::lock_guard<std::mutex> lock(link_state->mutex);
    result = cuLinkDestroy(link_state->cuda_state);
  }
  // Retain the opaque handle wrapper until process exit so stale client
  // handles never dereference freed memory.
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpy3D_v2(conn_t *conn) {
  CUDA_MEMCPY3D copy = {};
  size_t src_host_size = 0;
  size_t dst_host_size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &src_host_size, sizeof(src_host_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> src_host(src_host_size);
  if (src_host_size != 0 &&
      rpc_read(conn, src_host.data(), src_host_size) < 0) {
    return -1;
  }
  if (rpc_read(conn, &dst_host_size, sizeof(dst_host_size)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> dst_host(dst_host_size);
  if (copy.srcMemoryType == CU_MEMORYTYPE_HOST) {
    copy.srcHost = src_host.empty() ? nullptr : src_host.data();
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = dst_host.empty() ? nullptr : dst_host.data();
  }

  result = cuMemcpy3D_v2(&copy);
  size_t returned_dst_size = result == CUDA_SUCCESS ? dst_host.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0 ||
      (returned_dst_size != 0 &&
       rpc_write(conn, dst_host.data(), returned_dst_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int handle_manual_cuMemcpy2D_common(conn_t *conn, bool async,
                                           bool unaligned) {
  CUDA_MEMCPY2D copy = {};
  size_t src_host_size = 0;
  size_t dst_host_size = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &src_host_size, sizeof(src_host_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> src_host(src_host_size);
  if (src_host_size != 0 &&
      rpc_read(conn, src_host.data(), src_host_size) < 0) {
    return -1;
  }
  if (rpc_read(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> dst_host(dst_host_size);
  if (copy.srcMemoryType == CU_MEMORYTYPE_HOST) {
    copy.srcHost = src_host.empty() ? nullptr : src_host.data();
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = dst_host.empty() ? nullptr : dst_host.data();
  }

  if (async) {
    result = cuMemcpy2DAsync_v2(&copy, stream);
    if (src_host_size == 0 && dst_host_size == 0) {
      // The matching client call is fire-and-forget, just like generated
      // device-only async copies. CUDA stream ordering reports any deferred
      // execution error at the application's next synchronization boundary.
      return 0;
    }
    if (result == CUDA_SUCCESS) {
      result = cuStreamSynchronize(stream);
    }
  } else if (unaligned) {
    result = cuMemcpy2DUnaligned_v2(&copy);
  } else {
    result = cuMemcpy2D_v2(&copy);
  }

  size_t returned_dst_size = result == CUDA_SUCCESS ? dst_host.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0 ||
      (returned_dst_size != 0 &&
       rpc_write(conn, dst_host.data(), returned_dst_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpy2D_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, false, false);
}

int handle_manual_cuMemcpy2DUnaligned_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, false, true);
}

int handle_manual_cuMemcpy2DAsync_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, true, false);
}

int handle_manual_cuGraphAddMemAllocNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEM_ALLOC_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemAllocNode(&graphNode, hGraph,
                                  deps.empty() ? nullptr : deps.data(),
                                  deps.size(), &nodeParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemFreeNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUdeviceptr dptr = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &dptr, sizeof(dptr)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemFreeNode(&graphNode, hGraph,
                                 deps.empty() ? nullptr : deps.data(),
                                 deps.size(), dptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDeviceGetGraphMemAttribute(conn_t *conn) {
  CUdevice device = 0;
  CUgraphMem_attribute attr = CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
  cuuint64_t value = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuDeviceGetGraphMemAttribute(device, attr, &value);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDeviceSetGraphMemAttribute(conn_t *conn) {
  CUdevice device = 0;
  CUgraphMem_attribute attr = CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
  cuuint64_t value = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value, sizeof(value)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuDeviceSetGraphMemAttribute(device, attr, &value);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryGetModule(conn_t *conn) {
  CUlibrary library = nullptr;
  CUmodule module = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuLibraryGetModule(&module, library);
  if (result == CUDA_SUCCESS) {
    lupine_module_libraries().insert_or_assign(module, library);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// The library stays loaded: the client mirrors libraries onto other routes and
// caches the CUkernel handles this one owns, but it only ever sends the unload
// to the route that loaded it, so freeing here would dangle those handles.
int handle_manual_cuLibraryUnload(conn_t *conn) {
  CUlibrary library = nullptr;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  if (rpc_read_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuModuleGetGlobal_v2(conn_t *conn) {
  CUdeviceptr *dptr_null_check = nullptr;
  size_t *bytes_null_check = nullptr;
  CUdeviceptr dptr = 0;
  size_t bytes = 0;
  CUmodule module = nullptr;
  std::size_t name_len = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dptr_null_check, sizeof(dptr_null_check)) < 0 ||
      rpc_read(conn, &bytes_null_check, sizeof(bytes_null_check)) < 0 ||
      rpc_read(conn, &module, sizeof(module)) < 0 ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0) {
    return -1;
  }
  std::vector<char> name(name_len);
  if (name_len == 0 || rpc_read(conn, name.data(), name_len) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuModuleGetGlobal_v2(dptr_null_check ? &dptr : nullptr,
                                bytes_null_check ? &bytes : nullptr, module,
                                name.data());
  if (result != CUDA_SUCCESS) {
    CUlibrary library = nullptr;
    if (lupine_module_libraries().find(module, library)) {
      CUdeviceptr library_dptr = 0;
      size_t library_bytes = 0;
      CUresult library_result = cuLibraryGetGlobal(
          &library_dptr, &library_bytes, library, name.data());
      if (library_result == CUDA_SUCCESS) {
        dptr = library_dptr;
        bytes = library_bytes;
        result = library_result;
      }
    }
  }
  LUPINE_TRACE_LOG("LUPINE cuModuleGetGlobal name=" << name.data() << " result="
                                                    << static_cast<int>(result)
                                                    << " bytes=" << bytes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr_null_check, sizeof(dptr_null_check)) < 0 ||
      (dptr_null_check && rpc_write(conn, &dptr, sizeof(dptr)) < 0) ||
      rpc_write(conn, &bytes_null_check, sizeof(bytes_null_check)) < 0 ||
      (bytes_null_check && rpc_write(conn, &bytes, sizeof(bytes)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

CUresult lupine_get_kernel_param_layout(CUfunction f,
                                        lupine_kernel_param_layout *layout) {
  if (layout == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *layout = {};
  bool use_kernel_info = false;
  for (uint32_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = use_kernel_info
                          ? cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f),
                                                 i, &offset, &size)
                          : cuFuncGetParamInfo(f, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      return CUDA_SUCCESS;
    }
    if (i == 0 && result == CUDA_ERROR_INVALID_HANDLE) {
      use_kernel_info = true;
      result = cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f), i, &offset,
                                    &size);
      if (result == CUDA_ERROR_INVALID_VALUE) {
        return CUDA_SUCCESS;
      }
    }
    if (result != CUDA_SUCCESS) {
      return i == 0 ? result : CUDA_SUCCESS;
    }
    layout->offsets.push_back(offset);
    layout->sizes.push_back(size);
    layout->count = i + 1;
  }
  return CUDA_SUCCESS;
}

int handle_manual_cuLaunchKernel(conn_t *conn) {
  CUfunction f = nullptr;
  CUcontext ctx = nullptr;
  unsigned int gridDimX = 0;
  unsigned int gridDimY = 0;
  unsigned int gridDimZ = 0;
  unsigned int blockDimX = 0;
  unsigned int blockDimY = 0;
  unsigned int blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  uint32_t param_count = 0;
  size_t payload_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }

  if (ctx != nullptr) {
    CUcontext previous = nullptr;
    result = cuCtxGetCurrent(&previous);
    if (result == CUDA_SUCCESS && previous != ctx) {
      result = cuCtxSetCurrent(ctx);
    }
  } else {
    result = CUDA_SUCCESS;
  }

  lupine_kernel_param_layout layout;
  if (result == CUDA_SUCCESS) {
    result = lupine_get_kernel_param_layout(f, &layout);
  }
  if (result == CUDA_SUCCESS && layout.count != param_count) {
    result = CUDA_ERROR_INVALID_VALUE;
  }

  size_t storage_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    storage_size = std::max(storage_size, layout.offsets[i] + layout.sizes[i]);
  }
  std::vector<unsigned char> storage(storage_size);
  std::vector<void *> params(param_count);
  if (result == CUDA_SUCCESS &&
      rpc_read_kernel_param_values(
          conn, layout.count, layout.offsets.data(), layout.sizes.data(),
          payload_size, storage.data(), storage.size(), params.data()) < 0) {
    return -1;
  }
  if (result != CUDA_SUCCESS && rpc_drain(conn, payload_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    result =
        cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream,
                       param_count == 0 ? nullptr : params.data(), nullptr);
  }

  (void)request_id;
  (void)result;
  return 0;
}

int handle_manual_cuLaunchKernelEx(conn_t *conn) {
  CUlaunchConfig config = {};
  CUfunction f = nullptr;
  CUcontext ctx = nullptr;
  bool packed_launch = false;
  uint32_t param_count = 0;
  size_t payload_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  std::vector<CUlaunchAttribute> attributes;
  if (rpc_read_launch_config(conn, &config, &attributes) < 0 ||
      rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &packed_launch, sizeof(packed_launch)) < 0) {
    return -1;
  }

  if (packed_launch) {
    size_t packed_size = 0;
    if (rpc_read(conn, &packed_size, sizeof(packed_size)) < 0 ||
        packed_size > (1ULL << 30)) {
      return -1;
    }
    std::vector<unsigned char> packed(packed_size);
    if (rpc_read(conn, packed.data(), packed.size()) < 0) {
      return -1;
    }
    request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
#if CUDA_VERSION < 11080
    result = CUDA_ERROR_NOT_SUPPORTED;
#else
    if (ctx != nullptr) {
      CUcontext previous = nullptr;
      result = cuCtxGetCurrent(&previous);
      if (result == CUDA_SUCCESS && previous != ctx) {
        result = cuCtxSetCurrent(ctx);
      }
    } else {
      result = CUDA_SUCCESS;
    }
    if (result == CUDA_SUCCESS) {
      void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                       CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                       CU_LAUNCH_PARAM_END};
      result = cuLaunchKernelEx(&config, f, nullptr, extra);
      LUPINE_TRACE_LOG("LUPINE server raw packed cuLaunchKernelEx bytes="
                       << packed_size << " attrs=" << config.numAttrs
                       << " result=" << result);
    }
#endif
    if (config.numAttrs != 0) {
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &result, sizeof(result)) < 0 ||
          rpc_write_end(conn) < 0) {
        return -1;
      }
    }
    return 0;
  }

  if (rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }

  lupine_kernel_param_layout layout;
#if CUDA_VERSION < 11080
  result = CUDA_ERROR_NOT_SUPPORTED;
#else
  if (ctx != nullptr) {
    CUcontext previous = nullptr;
    result = cuCtxGetCurrent(&previous);
    if (result == CUDA_SUCCESS && previous != ctx) {
      result = cuCtxSetCurrent(ctx);
    }
  } else {
    result = CUDA_SUCCESS;
  }

  if (result == CUDA_SUCCESS) {
    result = lupine_get_kernel_param_layout(f, &layout);
  }
  if (result == CUDA_SUCCESS && layout.count != param_count) {
    result = CUDA_ERROR_INVALID_VALUE;
  }
#endif

  size_t storage_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    storage_size = std::max(storage_size, layout.offsets[i] + layout.sizes[i]);
  }
  std::vector<unsigned char> storage(storage_size);
  std::vector<void *> params(param_count);
  if (result == CUDA_SUCCESS &&
      rpc_read_kernel_param_values(
          conn, layout.count, layout.offsets.data(), layout.sizes.data(),
          payload_size, storage.data(), storage.size(), params.data()) < 0) {
    return -1;
  }
  if (result != CUDA_SUCCESS && rpc_drain(conn, payload_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

#if CUDA_VERSION >= 11080
  if (result == CUDA_SUCCESS) {
    result = cuLaunchKernelEx(
        &config, f, param_count == 0 ? nullptr : params.data(), nullptr);
  }
#endif

  // Mirror the client: attribute-free launches are fire-and-forget, launches
  // carrying attributes expect a synchronous result.
  if (config.numAttrs != 0) {
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }

  (void)request_id;
  (void)result;
  return 0;
}

int handle_manual_cuLaunchCooperativeKernel(conn_t *conn) {
  CUfunction f = nullptr;
  unsigned int gridDimX = 0;
  unsigned int gridDimY = 0;
  unsigned int gridDimZ = 0;
  unsigned int blockDimX = 0;
  unsigned int blockDimY = 0;
  unsigned int blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  uint32_t param_count = 0;
  size_t payload_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }

  lupine_kernel_param_layout layout;
  result = lupine_get_kernel_param_layout(f, &layout);
  if (result == CUDA_SUCCESS && layout.count != param_count) {
    result = CUDA_ERROR_INVALID_VALUE;
  }

  size_t storage_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    storage_size = std::max(storage_size, layout.offsets[i] + layout.sizes[i]);
  }
  std::vector<unsigned char> storage(storage_size);
  std::vector<void *> params(param_count);
  if (result == CUDA_SUCCESS &&
      rpc_read_kernel_param_values(
          conn, layout.count, layout.offsets.data(), layout.sizes.data(),
          payload_size, storage.data(), storage.size(), params.data()) < 0) {
    return -1;
  }
  if (result != CUDA_SUCCESS && rpc_drain(conn, payload_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    result = cuLaunchCooperativeKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, param_count == 0 ? nullptr : params.data());
  }

  (void)request_id;
  (void)result;
  return 0;
}

static void CUDA_CB lupine_graph_host_callback(void *userData) {
  auto *callback = static_cast<lupine_host_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr) {
    return;
  }

  std::vector<lupine_graph_host_copy> copies =
      callback->resources ? callback->resources->dtoh_copy_snapshot()
                          : std::vector<lupine_graph_host_copy>();
  int transfer_count = static_cast<int>(copies.size());

  conn_t *conn = callback->conn;
  if (rpc_write_start_request(conn, 1) < 0 ||
      rpc_write(conn, &transfer_count, sizeof(transfer_count)) < 0) {
    return;
  }
  for (const auto &copy : copies) {
    if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
        rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
        rpc_write_payload(conn, copy.server_src, copy.bytes) < 0) {
      return;
    }
  }
  CUhostFn fn = callback->fn;
  void *client_user_data = callback->userData;
  void *response = nullptr;
  if (rpc_write(conn, &fn, sizeof(fn)) < 0 ||
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &response, sizeof(response)) < 0 ||
      rpc_read_end(conn) < 0) {
    return;
  }
}

static void CUDA_CB lupine_stream_callback(CUstream stream, CUresult status,
                                           void *userData) {
  auto *callback = static_cast<lupine_stream_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr ||
      callback->callback == nullptr) {
    delete callback;
    return;
  }

  conn_t *conn = callback->conn;
  void *fn = reinterpret_cast<void *>(callback->callback);
  void *client_user_data = callback->userData;
  void *response = nullptr;
  uint32_t copy_count = 0;
  auto pending = lupine_detach_pending_dtoh_copies(conn, stream, false);
  if (rpc_write_start_request(conn, 2) >= 0 &&
      lupine_write_pending_dtoh_copies(&copy_count, conn, pending) >= 0 &&
      rpc_write(conn, &stream, sizeof(stream)) >= 0 &&
      rpc_write(conn, &status, sizeof(status)) >= 0 &&
      rpc_write(conn, &fn, sizeof(fn)) >= 0 &&
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) >= 0 &&
      rpc_wait_for_response(conn) >= 0) {
    rpc_read(conn, &response, sizeof(response));
    rpc_read_end(conn);
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
  delete callback;
}

struct lupine_kernel_param_payload {
  std::vector<unsigned char> storage;
  std::vector<void *> params;
};

static CUresult
lupine_read_kernel_param_values(conn_t *conn,
                                const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                uint32_t paramCount, size_t payloadSize,
                                lupine_kernel_param_payload *payload);

int handle_manual_cuGraphAddKernelNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read_kernel_node_params(conn, &nodeParams) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  result = lupine_read_kernel_param_values(conn, nodeParams, param_count,
                                           payload_size, &payload);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    nodeParams.kernelParams =
        payload.params.empty() ? nullptr : payload.params.data();
    nodeParams.extra = nullptr;
    result = cuGraphAddKernelNode_v2(&graphNode, hGraph,
                                     deps.empty() ? nullptr : deps.data(),
                                     deps.size(), &nodeParams);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static CUfunction
lupine_kernel_node_function(const CUDA_KERNEL_NODE_PARAMS &params) {
  CUfunction func = params.func;
#if CUDA_VERSION >= 12000
  if (func == nullptr) {
    func = reinterpret_cast<CUfunction>(params.kern);
  }
#endif
  return func;
}

static CUresult
lupine_prepare_server_kernel_params(const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                    CUDA_KERNEL_NODE_PARAMS *serialParams,
                                    lupine_kernel_param_layout *layout,
                                    size_t *payloadSize) {
  if (serialParams == nullptr || layout == nullptr || payloadSize == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *layout = {};
  *payloadSize = 0;
  if (nodeParams.extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUfunction func = lupine_kernel_node_function(nodeParams);
  CUresult result = lupine_get_kernel_param_layout(func, layout);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (layout->count != 0 && nodeParams.kernelParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  for (uint32_t i = 0; i < layout->count; ++i) {
    if (nodeParams.kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  *serialParams = nodeParams;
  serialParams->kernelParams = nullptr;
  serialParams->extra = nullptr;
  for (uint32_t i = 0; i < layout->count; ++i) {
    *payloadSize += layout->sizes[i];
  }
  return CUDA_SUCCESS;
}

static CUresult
lupine_read_kernel_param_values(conn_t *conn,
                                const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                uint32_t paramCount, size_t payloadSize,
                                lupine_kernel_param_payload *payload) {
  if (conn == nullptr || payload == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  payload->storage.clear();
  payload->params.clear();

  CUfunction func = lupine_kernel_node_function(nodeParams);
  lupine_kernel_param_layout layout;
  CUresult result = lupine_get_kernel_param_layout(func, &layout);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (layout.count != paramCount) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t expected_payload_size = 0;
  size_t storage_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    expected_payload_size += layout.sizes[i];
    storage_size = std::max(storage_size, layout.offsets[i] + layout.sizes[i]);
  }
  if (payloadSize != expected_payload_size) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  payload->storage.assign(storage_size, 0);
  payload->params.resize(paramCount);
  if (rpc_read_kernel_param_values(
          conn, layout.count, layout.offsets.data(), layout.sizes.data(),
          payloadSize, payload->storage.data(), payload->storage.size(),
          payload->params.data()) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

int handle_manual_cuGraphKernelNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  int request_id;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  CUDA_KERNEL_NODE_PARAMS serialParams = {};
  lupine_kernel_param_layout layout = {};
  size_t payloadSize = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphKernelNodeGetParams_v2(hNode, &nodeParams);
  if (result == CUDA_SUCCESS) {
    serialParams = nodeParams;
    serialParams.kernelParams = nullptr;
    serialParams.extra = nullptr;
    result = lupine_prepare_server_kernel_params(nodeParams, &serialParams,
                                                 &layout, &payloadSize);
  }

  // Copy the param values out of the driver's node storage before queueing:
  // rpc_write iovecs are only flushed at rpc_write_end, and a concurrent
  // cuGraphKernelNodeSetParams or node/graph destroy on another lane can
  // rewrite or free that storage before the response goes out.
  std::vector<unsigned char> value_storage;
  std::vector<void *> value_ptrs;
  if (result == CUDA_SUCCESS) {
    try {
      value_storage.resize(payloadSize);
      value_ptrs.resize(layout.count);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      layout = {};
      payloadSize = 0;
    }
  }
  if (result == CUDA_SUCCESS) {
    size_t offset = 0;
    for (uint32_t i = 0; i < layout.count; ++i) {
      memcpy(value_storage.data() + offset, nodeParams.kernelParams[i],
             layout.sizes[i]);
      value_ptrs[i] = value_storage.data() + offset;
      offset += layout.sizes[i];
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_kernel_node_params(conn, &serialParams) < 0 ||
      rpc_write_kernel_param_layout(conn, &layout) < 0 ||
      rpc_write(conn, &payloadSize, sizeof(payloadSize)) < 0 ||
      (result == CUDA_SUCCESS &&
       rpc_write_kernel_param_values(conn, layout.count, layout.sizes.data(),
                                     value_ptrs.data()) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphKernelNodeSetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read_kernel_node_params(conn, &nodeParams) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  result = lupine_read_kernel_param_values(conn, nodeParams, param_count,
                                           payload_size, &payload);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    nodeParams.kernelParams =
        payload.params.empty() ? nullptr : payload.params.data();
    nodeParams.extra = nullptr;
    result = cuGraphKernelNodeSetParams_v2(hNode, &nodeParams);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecKernelNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec = nullptr;
  CUgraphNode hNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read_kernel_node_params(conn, &nodeParams) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  result = lupine_read_kernel_param_values(conn, nodeParams, param_count,
                                           payload_size, &payload);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    nodeParams.kernelParams =
        payload.params.empty() ? nullptr : payload.params.data();
    nodeParams.extra = nullptr;
    result = cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, &nodeParams);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemcpyNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEMCPY3D copyParams = {};
  CUcontext ctx = nullptr;
  size_t host_src_bytes = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &copyParams, sizeof(copyParams)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &host_src_bytes, sizeof(host_src_bytes)) < 0) {
    return -1;
  }

  auto resources = lupine_get_graph_resources(hGraph);
  if (host_src_bytes != 0) {
    void *host = lupine_alloc_process_host_buffer(host_src_bytes);
    if (host == nullptr || rpc_read(conn, host, host_src_bytes) < 0) {
      return -1;
    }
    copyParams.srcHost = host;
  }

  if (copyParams.dstMemoryType == CU_MEMORYTYPE_HOST) {
    size_t host_dst_bytes = lupine_memcpy3d_host_span_bytes(copyParams, false);
    void *host = lupine_alloc_process_host_buffer(host_dst_bytes);
    if (host == nullptr && host_dst_bytes != 0) {
      return -1;
    }
    resources->add_dtoh_copy({copyParams.dstHost, host, host_dst_bytes});
    copyParams.dstHost = host;
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemcpyNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size(), &copyParams, ctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemsetNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEMSET_NODE_PARAMS memsetParams = {};
  CUcontext ctx = nullptr;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &memsetParams, sizeof(memsetParams)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemsetNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size(), &memsetParams, ctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddHostNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_HOST_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback =
      new lupine_host_callback_data{conn, nodeParams.fn, nodeParams.userData,
                                    lupine_get_graph_resources(hGraph)};
  CUDA_HOST_NODE_PARAMS serverParams = {};
  serverParams.fn = lupine_graph_host_callback;
  serverParams.userData = callback;

  result = cuGraphAddHostNode(&graphNode, hGraph,
                              deps.empty() ? nullptr : deps.data(), deps.size(),
                              &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphConditionalHandleCreate(conn_t *conn) {
  CUgraph hGraph = nullptr;
  CUcontext ctx = nullptr;
  unsigned int defaultLaunchValue = 0;
  unsigned int flags = 0;
  CUgraphConditionalHandle handle = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &defaultLaunchValue, sizeof(defaultLaunchValue)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphConditionalHandleCreate(&handle, hGraph, ctx,
                                          defaultLaunchValue, flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static CUresult lupine_server_cuGraphAddNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    CUgraphNodeParams *nodeParams) {
#if CUDA_VERSION >= 12060
  return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData,
                           numDependencies, nodeParams);
#else
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuGraphAddNode(phGraphNode, hGraph, dependencies, numDependencies,
                        nodeParams);
#endif
}

int handle_manual_cuGraphAddNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUgraphNodeParams nodeParams = {};
  uint32_t param_count = 0;
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    result = lupine_read_kernel_param_values(
        conn, *reinterpret_cast<CUDA_KERNEL_NODE_PARAMS *>(&nodeParams.kernel),
        param_count, payload_size, &payload);
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraph> child_graphs;
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    if (result == CUDA_SUCCESS) {
      nodeParams.kernel.kernelParams =
          payload.params.empty() ? nullptr : payload.params.data();
      nodeParams.kernel.extra = nullptr;
      result = lupine_server_cuGraphAddNode(
          &graphNode, hGraph, deps.empty() ? nullptr : deps.data(), nullptr,
          deps.size(), &nodeParams);
    }
  } else if (nodeParams.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    child_graphs.resize(nodeParams.conditional.size);
    nodeParams.conditional.phGraph_out = nullptr;
    result = lupine_server_cuGraphAddNode(&graphNode, hGraph,
                                          deps.empty() ? nullptr : deps.data(),
                                          nullptr, deps.size(), &nodeParams);
    if (result == CUDA_SUCCESS &&
        nodeParams.conditional.phGraph_out != nullptr) {
      for (size_t i = 0; i < child_graphs.size(); ++i) {
        child_graphs[i] = nodeParams.conditional.phGraph_out[i];
      }
    }
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  LUPINE_TRACE_LOG("LUPINE cuGraphAddNode type="
                   << nodeParams.type << " param_count=" << param_count
                   << " payload_size=" << payload_size << " graph=" << hGraph
                   << " node=" << graphNode << " result=" << result);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      (!child_graphs.empty() &&
       rpc_write(conn, child_graphs.data(),
                 child_graphs.size() * sizeof(CUgraph)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int lupine_handle_node_dependency_query(conn_t *conn, bool dependent) {
  CUgraphNode hNode = nullptr;
  size_t requested = 0;
  uint8_t want_edge = 0;
  size_t count = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &requested, sizeof(requested)) < 0 ||
      rpc_read(conn, &want_edge, sizeof(want_edge)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraphNode> nodes;
#if CUDA_VERSION >= 12030
  std::vector<CUgraphEdgeData> edges;
  auto call = [&](CUgraphNode *out, CUgraphEdgeData *edge,
                  size_t *n) -> CUresult {
    return dependent ? cuGraphNodeGetDependentNodes_v2(hNode, out, edge, n)
                     : cuGraphNodeGetDependencies_v2(hNode, out, edge, n);
  };
  if (requested == 0) {
    result = call(nullptr, nullptr, &count);
  } else {
    count = requested;
    nodes.resize(count);
    if (want_edge) {
      edges.resize(count);
    }
    result = call(nodes.data(), want_edge ? edges.data() : nullptr, &count);
  }
#else
  auto call = [&](CUgraphNode *out, size_t *n) -> CUresult {
    return dependent ? cuGraphNodeGetDependentNodes(hNode, out, n)
                     : cuGraphNodeGetDependencies(hNode, out, n);
  };
  if (requested == 0) {
    result = call(nullptr, &count);
  } else {
    count = requested;
    nodes.resize(count);
    result = call(nodes.data(), &count);
  }
#endif

  bool send_arrays = requested != 0 && count != 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (send_arrays &&
       rpc_write(conn, nodes.data(), count * sizeof(CUgraphNode)) < 0)) {
    return -1;
  }
#if CUDA_VERSION >= 12030
  if (send_arrays && want_edge &&
      rpc_write(conn, edges.data(), count * sizeof(CUgraphEdgeData)) < 0) {
    return -1;
  }
#endif
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphNodeGetDependencies(conn_t *conn) {
  return lupine_handle_node_dependency_query(conn, /*dependent=*/false);
}

int handle_manual_cuGraphNodeGetDependentNodes(conn_t *conn) {
  return lupine_handle_node_dependency_query(conn, /*dependent=*/true);
}

// cuGraphGetEdges: two parallel out node arrays (from/to) plus an optional
// CUgraphEdgeData array, all sized by an in/out count.
int handle_manual_cuGraphGetEdges(conn_t *conn) {
  CUgraph hGraph = nullptr;
  size_t requested = 0;
  uint8_t want_edge = 0;
  size_t count = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &requested, sizeof(requested)) < 0 ||
      rpc_read(conn, &want_edge, sizeof(want_edge)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraphNode> from;
  std::vector<CUgraphNode> to;
#if CUDA_VERSION >= 12030
  std::vector<CUgraphEdgeData> edges;
  if (requested == 0) {
    result = cuGraphGetEdges_v2(hGraph, nullptr, nullptr, nullptr, &count);
  } else {
    count = requested;
    from.resize(count);
    to.resize(count);
    if (want_edge) {
      edges.resize(count);
    }
    result = cuGraphGetEdges_v2(hGraph, from.data(), to.data(),
                                want_edge ? edges.data() : nullptr, &count);
  }
#else
  if (requested == 0) {
    result = cuGraphGetEdges(hGraph, nullptr, nullptr, &count);
  } else {
    count = requested;
    from.resize(count);
    to.resize(count);
    result = cuGraphGetEdges(hGraph, from.data(), to.data(), &count);
  }
#endif

  bool send_arrays = requested != 0 && count != 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (send_arrays &&
       rpc_write(conn, from.data(), count * sizeof(CUgraphNode)) < 0) ||
      (send_arrays &&
       rpc_write(conn, to.data(), count * sizeof(CUgraphNode)) < 0)) {
    return -1;
  }
#if CUDA_VERSION >= 12030
  if (send_arrays && want_edge &&
      rpc_write(conn, edges.data(), count * sizeof(CUgraphEdgeData)) < 0) {
    return -1;
  }
#endif
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// Host-node callbacks set after node creation have no graph handle to own their
// trampoline data. The server process is their lifetime boundary, so allocate
// them directly without a synchronized container that never reclaimed them.
static lupine_host_callback_data *
lupine_make_host_setparams_callback(conn_t *conn,
                                    const CUDA_HOST_NODE_PARAMS &params) {
  return new lupine_host_callback_data{conn, params.fn, params.userData,
                                       nullptr};
}

int handle_manual_cuGraphHostNodeSetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback = lupine_make_host_setparams_callback(conn, params);
  CUDA_HOST_NODE_PARAMS serverParams{};
  serverParams.fn = lupine_graph_host_callback;
  serverParams.userData = callback;
  result = cuGraphHostNodeSetParams(hNode, &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecHostNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec = nullptr;
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback = lupine_make_host_setparams_callback(conn, params);
  CUDA_HOST_NODE_PARAMS serverParams{};
  serverParams.fn = lupine_graph_host_callback;
  serverParams.userData = callback;
  result = cuGraphExecHostNodeSetParams(hGraphExec, hNode, &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphHostNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphHostNodeGetParams(hNode, &params);
  // Unwrap the trampoline so the client sees the fn/userData it registered.
  if (result == CUDA_SUCCESS && params.fn == lupine_graph_host_callback &&
      params.userData != nullptr) {
    auto *callback = static_cast<lupine_host_callback_data *>(params.userData);
    params.fn = callback->fn;
    params.userData = callback->userData;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &params, sizeof(params)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchHostFunc(conn_t *conn) {
  CUstream stream = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &fn, sizeof(fn)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *resources = lupine_get_stream_resources(stream);
  auto *callback = new lupine_host_callback_data{conn, fn, userData, resources};
  result = cuLaunchHostFunc(stream, lupine_graph_host_callback, callback);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamAddCallback(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *data = new lupine_stream_callback_data{conn, callback, userData};
  result = cuStreamAddCallback(stream, lupine_stream_callback, data, flags);
  if (result != CUDA_SUCCESS) {
    delete data;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventRecord(conn_t *conn, bool with_flags) {
  CUevent event = nullptr;
  CUstream stream = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_graph_resources *resources = nullptr;
  if (lupine_stream_capture_resource_map().find(stream, resources)) {
    lupine_event_capture_resource_map().insert_or_assign(event, resources);
  }

  result = with_flags ? cuEventRecordWithFlags(event, stream, flags)
                      : cuEventRecord(event, stream);
  (void)result;
  (void)request_id;
  return 0;
}

// The client batches every event it still has outstanding into one query so a
// poller walking several events pays one round trip; events[0] is the one the
// caller asked about and the only one that governs the deferred copy drain.
int handle_manual_cuEventQuery(conn_t *conn) {
  uint32_t count = 0;
  if (rpc_read(conn, &count, sizeof(count)) < 0 || count == 0 ||
      count > LUPINE_EVENT_QUERY_BATCH_MAX) {
    return -1;
  }
  CUevent events[LUPINE_EVENT_QUERY_BATCH_MAX];
  if (rpc_read(conn, events, count * sizeof(events[0])) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult results[LUPINE_EVENT_QUERY_BATCH_MAX];
  for (uint32_t i = 0; i < count; ++i) {
    results[i] = cuEventQuery(events[i]);
  }
  CUresult result = results[0];

  if (rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  uint32_t copy_count = 0;
  std::vector<lupine_pending_dtoh_copy> pending;
  if (result == CUDA_SUCCESS) {
    pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
    if (lupine_write_pending_dtoh_copies(&copy_count, conn, pending) < 0) {
      lupine_cleanup_pending_dtoh_copies(&pending);
      return -1;
    }
  } else {
    if (rpc_write(conn, &copy_count, sizeof(copy_count)) < 0) {
      return -1;
    }
  }
  if (rpc_write(conn, results, count * sizeof(results[0])) < 0 ||
      rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
  return 0;
}

int handle_manual_cuStreamWaitEvent(conn_t *conn) {
  CUstream stream = nullptr;
  CUevent event = nullptr;
  unsigned int flags = 0;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_graph_resources *event_resources = nullptr;
  if (lupine_event_capture_resource_map().find(event, event_resources)) {
    lupine_stream_capture_resource_map().insert(stream, event_resources);
  }

  cuStreamWaitEvent(stream, event, flags);
  return 0;
}

int handle_manual_cuStreamBeginCaptureToGraph(conn_t *conn) {
  CUstream stream = nullptr;
  CUgraph graph = nullptr;
  std::vector<CUgraphNode> deps;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamBeginCaptureToGraph(stream, graph,
                                       deps.empty() ? nullptr : deps.data(),
                                       nullptr, deps.size(), mode);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static CUresult lupine_server_cuStreamUpdateCaptureDependencies(
    CUstream stream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
#if CUDA_VERSION >= 12060
  return cuStreamUpdateCaptureDependencies_v2(
      stream, dependencies, dependencyData, numDependencies, flags);
#else
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuStreamUpdateCaptureDependencies(stream, dependencies,
                                           numDependencies, flags);
#endif
}

int handle_manual_cuStreamUpdateCaptureDependencies(conn_t *conn) {
  CUstream stream = nullptr;
  std::vector<CUgraphNode> deps;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = lupine_server_cuStreamUpdateCaptureDependencies(
      stream, deps.empty() ? nullptr : deps.data(), nullptr, deps.size(),
      flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamGetCaptureInfo(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  const CUgraphNode *deps_ptr = nullptr;
  const CUgraphEdgeData *edge_ptr = nullptr;
  size_t dep_count = 0;
  bool has_edge_data = false;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamGetCaptureInfo_v3(stream, &status, &id, &graph, &deps_ptr,
                                     &edge_ptr, &dep_count);
  has_edge_data = edge_ptr != nullptr && dep_count != 0;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &id, sizeof(id)) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &dep_count, sizeof(dep_count)) < 0 ||
      rpc_write(conn, &has_edge_data, sizeof(has_edge_data)) < 0 ||
      (dep_count != 0 && deps_ptr != nullptr &&
       rpc_write(conn, deps_ptr, dep_count * sizeof(CUgraphNode)) < 0) ||
      (has_edge_data &&
       rpc_write(conn, edge_ptr, dep_count * sizeof(CUgraphEdgeData)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamBeginCapture(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUstreamCaptureMode thread_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0 ||
      rpc_read(conn, &thread_mode, sizeof(thread_mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *resources = lupine_begin_stream_capture_resources(stream);
  if (!resources->has_capture_scratch()) {
    static constexpr size_t scratch_size = 128ull * 1024ull * 1024ull;
    void *scratch = nullptr;
    if (cuMemAllocHost(&scratch, scratch_size) == CUDA_SUCCESS) {
      if (!resources->install_capture_scratch(scratch, scratch_size)) {
        cuMemFreeHost(scratch);
      }
    }
  }

  // CUDA's capture policy is defined in terms of application host threads.
  // RPC handling deliberately decouples those from server worker threads, so
  // enforcing GLOBAL/THREAD_LOCAL again on the server rejects legal follow-up
  // calls (PyTorch synchronizes its generator-state stream immediately after
  // capture begins). The client preserves and validates the requested policy;
  // the transport side must remain relaxed across handler threads.
  (void)thread_mode;
  CUstreamCaptureMode server_previous = CU_STREAM_CAPTURE_MODE_RELAXED;
  result = cuThreadExchangeStreamCaptureMode(&server_previous);
  if (result == CUDA_SUCCESS) {
    result = !resources->has_capture_scratch()
                 ? CUDA_ERROR_OUT_OF_MEMORY
                 : cuStreamBeginCapture_v2(
                       stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamEndCapture(conn_t *conn) {
  CUstream stream = nullptr;
  CUgraph *graph_out = nullptr;
  CUgraph graph = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph_out, sizeof(graph_out)) < 0 ||
      (graph_out != nullptr && rpc_read(conn, &graph, sizeof(graph)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamEndCapture(stream, &graph);
  if (result == CUDA_SUCCESS) {
    lupine_graph_resources *resources = nullptr;
    if (lupine_stream_capture_resource_map().erase_fn(
            stream, [&resources](lupine_graph_resources *&stored) {
              resources = stored;
              return true;
            })) {
      lupine_graph_resource_map().insert_or_assign(graph, resources);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graph_out, sizeof(graph_out)) < 0 ||
      (graph_out != nullptr && rpc_write(conn, &graph, sizeof(graph)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphClone(conn_t *conn) {
  CUgraph clone = nullptr;
  CUgraph original = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &clone, sizeof(clone)) < 0 ||
      rpc_read(conn, &original, sizeof(original)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphClone(&clone, original);
  if (result == CUDA_SUCCESS) {
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(original, resources)) {
      lupine_graph_resource_map().insert_or_assign(clone, resources);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &clone, sizeof(clone)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphInstantiateWithFlags(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  unsigned long long flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphInstantiateWithFlags(&exec, graph, flags);
  if (result == CUDA_SUCCESS) {
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(graph, resources)) {
      lupine_graph_exec_resource_map().insert_or_assign(exec, resources);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphInstantiateWithParams(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  CUDA_GRAPH_INSTANTIATE_PARAMS params = {};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphInstantiateWithParams(&exec, graph, &params);
  if (result == CUDA_SUCCESS) {
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(graph, resources)) {
      lupine_graph_exec_resource_map().insert_or_assign(exec, resources);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &params, sizeof(params)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecUpdate(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  CUgraphExecUpdateResultInfo result_info = {};
  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &result_info, sizeof(result_info)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult result = cuGraphExecUpdate_v2(exec, graph, &result_info);
  if (result == CUDA_SUCCESS &&
      result_info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS) {
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(graph, resources)) {
      lupine_graph_exec_resource_map().insert_or_assign(exec, resources);
    }
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result_info, sizeof(result_info)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecDestroy(conn_t *conn) {
  CUgraphExec exec = nullptr;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphExecDestroy(exec);
  lupine_graph_exec_resource_map().erase(exec);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphDestroy(conn_t *conn) {
  CUgraph graph = nullptr;
  if (rpc_read(conn, &graph, sizeof(graph)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphDestroy(graph);
  lupine_graph_resource_map().erase(graph);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyAsync(conn_t *conn) {
  CUdeviceptr dst = 0;
  CUdeviceptr src = 0;
  size_t byte_count = 0;
  CUstream stream = nullptr;
  if (rpc_read(conn, &dst, sizeof(dst)) < 0 ||
      rpc_read(conn, &src, sizeof(src)) < 0 ||
      rpc_read(conn, &byte_count, sizeof(byte_count)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  // The client classifies pageable/pinned host addresses before choosing this
  // RPC, so this remaining path is a same-server device-to-device copy.
  CUresult result = cuMemcpyAsync(dst, src, byte_count, stream);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyHtoDAsync_v2(conn_t *conn) {
  CUdeviceptr dstDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  void *capture_host = nullptr;
  CUdeviceptr graph_host_source = 0;

  if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph_host_source, sizeof(graph_host_source)) < 0) {
    return -1;
  }

  int framed = lupine_payload_framed(conn, byteCount);
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CUresult capture_query_result = CUDA_SUCCESS;
  if (stream != nullptr) {
    capture_query_result = cuStreamIsCapturing(stream, &capture_status);
  }
  if (capture_query_result != CUDA_SUCCESS) {
    result = capture_query_result;
    if (rpc_drain_payload(conn, framed, byteCount) < 0) {
      return -1;
    }
  } else if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto *resources = lupine_get_stream_resources(stream);
    capture_host = graph_host_source == 0
                       ? lupine_alloc_capture_scratch(resources, byteCount)
                       : reinterpret_cast<void *>(graph_host_source);
    if (capture_host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_drain_payload(conn, framed, byteCount) < 0) {
        return -1;
      }
    } else {
      if (byteCount != 0 &&
          rpc_read_payload_part(conn, framed, capture_host, byteCount) < 0) {
        return -1;
      }
    }
  } else {
#ifdef _WIN32
    result = CUDA_SUCCESS;
    void *host = nullptr;
    if (byteCount != 0) {
      result = cuMemAllocHost(&host, byteCount);
    }
    if (result != CUDA_SUCCESS) {
      if (rpc_drain_payload(conn, framed, byteCount) < 0) {
        return -1;
      }
    }
    size_t offset = 0;
    while (result == CUDA_SUCCESS && offset < byteCount) {
      size_t chunk = std::min(LUPINE_HTOD_CHUNK_BYTES, byteCount - offset);
      auto *chunk_host = static_cast<unsigned char *>(host) + offset;
      if (rpc_read_payload_part(conn, framed, chunk_host, chunk) < 0) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
        return -1;
      }

      CUresult copy_result =
          cuMemcpyHtoDAsync_v2(dstDevice + offset, chunk_host, chunk, stream);
      if (copy_result != CUDA_SUCCESS) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
        result = copy_result;
        offset += chunk;
        if (rpc_drain_payload(conn, framed, byteCount - offset) < 0) {
          return -1;
        }
        host = nullptr;
        break;
      }
      offset += chunk;
    }
    if (host != nullptr && result == CUDA_SUCCESS) {
      result = lupine_defer_host_free(stream, host);
      if (result != CUDA_SUCCESS) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
      }
    }
#else
    if (lupine_server_copy_htod_async(conn, framed, dstDevice, byteCount,
                                      stream, result) < 0) {
      return -1;
    }
#endif
  }

  if (rpc_read_end(conn) < 0) {
    return -1;
  }

  if (capture_query_result == CUDA_SUCCESS &&
      capture_status != CU_STREAM_CAPTURE_STATUS_NONE &&
      result != CUDA_ERROR_OUT_OF_MEMORY) {
    result = cuMemcpyHtoDAsync_v2(dstDevice, capture_host, byteCount, stream);
  }

  return 0;
}

// Fire-and-forget: connection ordering already guarantees the flush is
// applied before any later request, so no response is sent.
int handle_manual_lupineManagedHostFlush(conn_t *conn) {
  uint32_t count = 0;

  if (rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }

  for (uint32_t i = 0; i < count; ++i) {
    void *server_host_ptr = nullptr;
    size_t bytes = 0;
    if (rpc_read(conn, &server_host_ptr, sizeof(server_host_ptr)) < 0 ||
        rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
        rpc_read(conn, server_host_ptr, bytes) < 0) {
      return -1;
    }
  }

  return rpc_read_end(conn) < 0 ? -1 : 0;
}

// Snapshot a server-side CUDA host mapping without synchronizing a stream.
// Persistent kernels and CPU proxy threads use this memory concurrently, so a
// CUDA D2H copy would deadlock behind the kernel it is intended to service.
int handle_manual_lupineMappedHostSnapshot(conn_t *conn) {
  static thread_local std::unordered_map<void *, std::vector<unsigned char>>
      previous_snapshots;
  void *server_host_ptr = nullptr;
  size_t bytes = 0;
  if (rpc_read(conn, &server_host_ptr, sizeof(server_host_ptr)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0 || bytes > (1ULL << 30)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  auto *source = static_cast<unsigned char *>(server_host_ptr);
  auto found = previous_snapshots.find(server_host_ptr);
  bool full = found == previous_snapshots.end() || found->second.size() != bytes;
  std::vector<std::pair<uint64_t, uint32_t>> changed;
  constexpr size_t block_bytes = 64;
  if (full) {
    previous_snapshots[server_host_ptr] =
        std::vector<unsigned char>(source, source + bytes);
  } else {
    auto &previous = found->second;
    for (size_t offset = 0; offset < bytes; offset += block_bytes) {
      size_t block = std::min(block_bytes, bytes - offset);
      if (memcmp(previous.data() + offset, source + offset, block) != 0) {
        changed.emplace_back(offset, static_cast<uint32_t>(block));
        memcpy(previous.data() + offset, source + offset, block);
      }
    }
  }
  uint32_t count = static_cast<uint32_t>(changed.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &full, sizeof(full)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (full && bytes != 0 && rpc_write(conn, server_host_ptr, bytes) < 0)) {
    return -1;
  }
  if (!full) {
    for (const auto &range : changed) {
      if (rpc_write(conn, &range.first, sizeof(range.first)) < 0 ||
          rpc_write(conn, &range.second, sizeof(range.second)) < 0 ||
          rpc_write(conn, source + range.first, range.second) < 0) {
        return -1;
      }
    }
  }
  if (rpc_write_end(conn) < 0) return -1;
  return 0;
}

int handle_manual_lupineCublasCall(conn_t *conn) {
  lupine_cublas_request request;
  if (rpc_read(conn, &request, sizeof(request)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  CUcontext saved_context = nullptr;
  CUresult saved_context_result = cuCtxGetCurrent(&saved_context);
  CUcontext requested_context =
      reinterpret_cast<CUcontext>(request.context);
  if (requested_context != nullptr)
    (void)cuCtxSetCurrent(requested_context);

  static void *library = [] {
    void *handle = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      handle = dlopen(
          "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/"
          "libcublas.so.13",
          RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr)
      LUPINE_LOG_ERROR("Unable to load server cuBLAS: " << dlerror());
    return handle;
  }();
  auto symbol = [](const char *name) -> void * {
    return library == nullptr ? nullptr : dlsym(library, name);
  };
  static void *lt_library = [] {
    void *lt = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_LOCAL);
    if (lt == nullptr) {
      lt = dlopen(
          "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/"
          "libcublasLt.so.13",
          RTLD_NOW | RTLD_LOCAL);
    }
    if (lt == nullptr)
      LUPINE_LOG_ERROR("Unable to load server cuBLASLt: " << dlerror());
    return lt;
  }();
  auto lt_symbol = [](const char *name) -> void * {
    return lt_library == nullptr ? nullptr : dlsym(lt_library, name);
  };
  static void *solver_library = [] {
    void *solver = dlopen("libcusolver.so.12", RTLD_NOW | RTLD_LOCAL);
    if (solver == nullptr) {
      solver = dlopen(
          "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/"
          "libcusolver.so.12",
          RTLD_NOW | RTLD_LOCAL);
    }
    if (solver == nullptr)
      LUPINE_LOG_ERROR("Unable to load server cuSOLVER: " << dlerror());
    return solver;
  }();
  auto solver_symbol = [](const char *name) -> void * {
    return solver_library == nullptr ? nullptr : dlsym(solver_library, name);
  };
  static void *fft_library = [] {
    void *fft = dlopen("libcufft.so.12", RTLD_NOW | RTLD_LOCAL);
    if (fft == nullptr) {
      fft = dlopen(
          "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/"
          "libcufft.so.12",
          RTLD_NOW | RTLD_LOCAL);
    }
    if (fft == nullptr)
      LUPINE_LOG_ERROR("Unable to load server cuFFT: " << dlerror());
    return fft;
  }();
  auto fft_symbol = [](const char *name) -> void * {
    return fft_library == nullptr ? nullptr : dlsym(fft_library, name);
  };

  lupine_cublas_response response;
  void *handle = reinterpret_cast<void *>(request.handle);
  int handle_state_status = 0;
  const bool applies_handle_state =
      request.opcode == LUPINE_CUBLAS_SGEMM ||
      request.opcode == LUPINE_CUBLAS_GEMM_EX ||
      request.opcode == LUPINE_CUBLAS_SGEMM_EX ||
      request.opcode == LUPINE_CUBLAS_GEMM_STRIDED_BATCHED_EX ||
      request.opcode == LUPINE_CUBLAS_CGEMM_STRIDED_BATCHED ||
      request.opcode == LUPINE_CUBLAS_ZGEMM_STRIDED_BATCHED ||
      request.opcode == LUPINE_CUBLAS_SGETRS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_SDOT ||
      request.opcode == LUPINE_CUBLAS_DOT_EX ||
      request.opcode == LUPINE_CUBLAS_STRSM_BATCHED ||
      request.opcode == LUPINE_CUBLAS_ZTRSM_BATCHED ||
      request.opcode == LUPINE_CUBLAS_CTRSM_BATCHED ||
      request.opcode == LUPINE_CUBLAS_SGEMV ||
      request.opcode == LUPINE_CUBLAS_CGEMM ||
      request.opcode == LUPINE_CUBLAS_ZGEMM ||
      request.opcode == LUPINE_CUBLAS_CDOTU ||
      request.opcode == LUPINE_CUBLAS_CDOTC ||
      request.opcode == LUPINE_CUBLAS_ZDOTU ||
      request.opcode == LUPINE_CUBLAS_ZDOTC ||
      request.opcode == LUPINE_CUBLAS_CGETRS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_CGETRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_SGETRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DGETRS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DGETRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DTRSM_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DTRSM ||
      request.opcode == LUPINE_CUBLAS_DGEMM_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DGEMV ||
      request.opcode == LUPINE_CUBLAS_CGEMV ||
      request.opcode == LUPINE_CUBLAS_ZGEMV ||
      request.opcode == LUPINE_CUBLAS_SGELS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DGELS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_CGELS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_ZGELS_BATCHED ||
      request.opcode == LUPINE_CUBLAS_SGEQRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DGEQRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_CGEQRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_ZGEQRF_BATCHED ||
      request.opcode == LUPINE_CUBLAS_DAXPY ||
      request.opcode == LUPINE_CUBLAS_DCOPY ||
      request.opcode == LUPINE_CUBLAS_DSCAL ||
      request.opcode == LUPINE_CUBLAS_DNRM2 ||
      request.opcode == LUPINE_CUBLAS_DASUM ||
      request.opcode == LUPINE_CUBLAS_DSWAP ||
      request.opcode == LUPINE_CUBLAS_IDAMAX ||
      request.opcode == LUPINE_CUBLAS_IDAMIN ||
      request.opcode == LUPINE_CUBLAS_ZGETRS_BATCHED;
  if (applies_handle_state &&
      (request.handle_state_mask & 1u) != 0) {
    using function = int (*)(void *, void *);
    auto call = reinterpret_cast<function>(symbol("cublasSetStream_v2"));
    handle_state_status =
        call == nullptr
            ? 13
            : call(handle, reinterpret_cast<void *>(request.stream));
  }
  if (applies_handle_state && handle_state_status == 0 &&
      (request.handle_state_mask & 2u) != 0) {
    using function = int (*)(void *, void *, size_t);
    auto call = reinterpret_cast<function>(symbol("cublasSetWorkspace_v2"));
    handle_state_status =
        call == nullptr
            ? 13
            : call(handle, reinterpret_cast<void *>(request.workspace),
                   static_cast<size_t>(request.workspace_size));
  }
  if (applies_handle_state && handle_state_status == 0 &&
      (request.handle_state_mask & 4u) != 0) {
    using function = int (*)(void *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSetMathMode"));
    handle_state_status =
        call == nullptr ? 13 : call(handle, request.math_mode);
  }
  if (applies_handle_state && handle_state_status == 0 &&
      (request.handle_state_mask & 8u) != 0) {
    using function = int (*)(void *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSetPointerMode_v2"));
    handle_state_status =
        call == nullptr ? 13 : call(handle, request.pointer_mode);
  }
  if (handle_state_status != 0) {
    response.status = handle_state_status;
  } else {
  switch (request.opcode) {
  case LUPINE_CUFFT_CREATE: {
    using function = int (*)(int *);
    auto call = reinterpret_cast<function>(fft_symbol("cufftCreate"));
    int plan = 0;
    response.status = call == nullptr ? 5 : call(&plan);
    response.value = plan;
    break;
  }
  case LUPINE_CUFFT_DESTROY: {
    using function = int (*)(int);
    auto call = reinterpret_cast<function>(fft_symbol("cufftDestroy"));
    response.status = call == nullptr ? 5 : call(static_cast<int>(request.handle));
    break;
  }
  case LUPINE_CUFFT_SET_AUTO_ALLOCATION: {
    using function = int (*)(int, int);
    auto call = reinterpret_cast<function>(fft_symbol("cufftSetAutoAllocation"));
    response.status = call == nullptr
                          ? 5
                          : call(static_cast<int>(request.handle), request.value);
    break;
  }
  case LUPINE_CUFFT_SET_STREAM: {
    using function = int (*)(int, void *);
    auto call = reinterpret_cast<function>(fft_symbol("cufftSetStream"));
    response.status = call == nullptr
                          ? 5
                          : call(static_cast<int>(request.handle),
                                 reinterpret_cast<void *>(request.d));
    break;
  }
  case LUPINE_CUFFT_SET_WORK_AREA: {
    using function = int (*)(int, void *);
    auto call = reinterpret_cast<function>(fft_symbol("cufftSetWorkArea"));
    response.status = call == nullptr
                          ? 5
                          : call(static_cast<int>(request.handle),
                                 reinterpret_cast<void *>(request.workspace));
    break;
  }
  case LUPINE_CUFFT_XT_MAKE_PLAN_MANY: {
    using function = int (*)(int, int, int64_t *, int64_t *, int64_t, int64_t,
                             int, int64_t *, int64_t, int64_t, int, int64_t,
                             size_t *, int);
    auto call = reinterpret_cast<function>(fft_symbol("cufftXtMakePlanMany"));
    const int rank = request.m;
    int64_t *dimensions = reinterpret_cast<int64_t *>(request.payload);
    int64_t *input_embed =
        (request.attribute & 1) == 0 ? nullptr : dimensions + rank;
    int64_t *output_embed =
        (request.attribute & 2) == 0 ? nullptr : dimensions + 2 * rank;
    size_t work_size = 0;
    response.status =
        call == nullptr || rank < 1 || rank > 5 ||
                request.payload_size < 3u * rank * sizeof(int64_t)
            ? 5
            : call(static_cast<int>(request.handle), rank, dimensions,
                   input_embed, request.stride_a, request.stride_b,
                   request.a_type, output_embed, request.stride_c,
                   request.leading_dimension, request.b_type,
                   static_cast<int64_t>(request.rows), &work_size,
                   request.compute_type);
    response.handle = static_cast<uint64_t>(work_size);
    break;
  }
  case LUPINE_CUFFT_XT_EXEC: {
    using function = int (*)(int, void *, void *, int);
    auto call = reinterpret_cast<function>(fft_symbol("cufftXtExec"));
    response.status =
        call == nullptr
            ? 5
            : call(static_cast<int>(request.handle),
                   reinterpret_cast<void *>(request.a),
                   reinterpret_cast<void *>(request.b), request.value);
    break;
  }
  case LUPINE_CUSOLVER_DN_CREATE: {
    using function = int (*)(void **);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCreate"));
    response.status = call == nullptr ? 7 : call(&handle);
    response.handle = reinterpret_cast<uint64_t>(handle);
    break;
  }
  case LUPINE_CUSOLVER_DN_DESTROY: {
    using function = int (*)(void *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDestroy"));
    response.status = call == nullptr ? 7 : call(handle);
    break;
  }
  case LUPINE_CUSOLVER_DN_SET_STREAM: {
    using function = int (*)(void *, void *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSetStream"));
    response.status = call == nullptr
                          ? 7
                          : call(handle, reinterpret_cast<void *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_SGETRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, float *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnSgetrf_bufferSize"));
    int workspace_elements = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<float *>(request.a), request.lda,
                   &workspace_elements);
    response.value = workspace_elements;
    break;
  }
  case LUPINE_CUSOLVER_DN_SGETRF: {
    using function = int (*)(void *, int, int, float *, int, float *, int *,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSgetrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<float *>(request.a), request.lda,
                   reinterpret_cast<float *>(request.workspace),
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_SGETRS: {
    using function = int (*)(void *, int, int, int, const float *, int,
                             const int *, float *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSgetrs"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const float *>(request.a), request.lda,
                   reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<float *>(request.c), request.ldb,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_CGETRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, void *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCgetrf_bufferSize"));
    int lwork = 0;
    response.status = call == nullptr
                          ? 7
                          : call(handle, request.m, request.n,
                                 reinterpret_cast<void *>(request.a),
                                 request.lda, &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_CGETRF: {
    using function = int (*)(void *, int, int, void *, int, void *, int *, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCgetrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<void *>(request.workspace),
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_CGETRS: {
    using function = int (*)(void *, int, int, int, const void *, int,
                             const int *, void *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCgetrs"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldb,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGETRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, void *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZgetrf_bufferSize"));
    int lwork = 0;
    response.status = call == nullptr
                          ? 7
                          : call(handle, request.m, request.n,
                                 reinterpret_cast<void *>(request.a),
                                 request.lda, &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGETRF: {
    using function = int (*)(void *, int, int, void *, int, void *, int *,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnZgetrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<void *>(request.workspace),
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGETRS: {
    using function = int (*)(void *, int, int, int, const void *, int,
                             const int *, void *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnZgetrs"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldb,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_DGETRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, double *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDgetrf_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DGETRF: {
    using function = int (*)(void *, int, int, double *, int, double *, int *,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDgetrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.workspace),
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_DGETRS: {
    using function = int (*)(void *, int, int, int, const double *, int,
                             const int *, double *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDgetrs"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<double *>(request.c), request.ldb,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_CREATE_GESVDJ_INFO: {
    using function = int (*)(void **);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnCreateGesvdjInfo"));
    response.status = call == nullptr ? 7 : call(&handle);
    response.handle = reinterpret_cast<uint64_t>(handle);
    break;
  }
  case LUPINE_CUSOLVER_DN_DESTROY_GESVDJ_INFO: {
    using function = int (*)(void *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDestroyGesvdjInfo"));
    response.status = call == nullptr ? 7 : call(handle);
    break;
  }
  case LUPINE_CUSOLVER_DN_GESVDJ_SET_TOLERANCE: {
    using function = int (*)(void *, double);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXgesvdjSetTolerance"));
    double tolerance = 0.0;
    if (request.payload_size >= sizeof(tolerance))
      memcpy(&tolerance, request.payload, sizeof(tolerance));
    response.status = call == nullptr ? 7 : call(handle, tolerance);
    break;
  }
  case LUPINE_CUSOLVER_DN_GESVDJ_SET_MAX_SWEEPS: {
    using function = int (*)(void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXgesvdjSetMaxSweeps"));
    response.status = call == nullptr ? 7 : call(handle, request.value);
    break;
  }
  case LUPINE_CUSOLVER_DN_GESVDJ_SET_SORT_EIG: {
    using function = int (*)(void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXgesvdjSetSortEig"));
    response.status = call == nullptr ? 7 : call(handle, request.value);
    break;
  }
  case LUPINE_CUSOLVER_DN_SGESVDJ_BATCHED_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const float *, int,
                             const float *, const float *, int, const float *,
                             int, int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnSgesvdjBatched_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<const float *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b),
                   reinterpret_cast<const float *>(request.c), request.ldb,
                   reinterpret_cast<const float *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_SGESVDJ_BATCHED: {
    using function = int (*)(void *, int, int, int, float *, int, float *,
                             float *, int, float *, int, float *, int, int *,
                             void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnSgesvdjBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<float *>(request.a), request.lda,
                   reinterpret_cast<float *>(request.b),
                   reinterpret_cast<float *>(request.c), request.ldb,
                   reinterpret_cast<float *>(request.d), request.ldc,
                   reinterpret_cast<float *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_DGESVDJ_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, const double *, int,
                             const double *, const double *, int,
                             const double *, int, int *, void *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDgesvdj_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.m, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<const double *>(request.c), request.ldb,
                   reinterpret_cast<const double *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor));
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DGESVDJ: {
    using function = int (*)(void *, int, int, int, int, double *, int,
                             double *, double *, int, double *, int, double *,
                             int, int *, void *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDgesvdj"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.m, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<double *>(request.c), request.ldb,
                   reinterpret_cast<double *>(request.d), request.ldc,
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor));
    break;
  }
  case LUPINE_CUSOLVER_DN_DGESVDJ_BATCHED_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const double *, int,
                             const double *, const double *, int,
                             const double *, int, int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDgesvdjBatched_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<const double *>(request.c), request.ldb,
                   reinterpret_cast<const double *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DGESVDJ_BATCHED: {
    using function = int (*)(void *, int, int, int, double *, int, double *,
                             double *, int, double *, int, double *, int,
                             int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDgesvdjBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<double *>(request.c), request.ldb,
                   reinterpret_cast<double *>(request.d), request.ldc,
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_CGESVDJ_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, const void *, int,
                             const float *, const void *, int, const void *,
                             int, int *, void *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnCgesvdj_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.m, request.n,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b),
                   reinterpret_cast<const void *>(request.c), request.ldb,
                   reinterpret_cast<const void *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor));
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_CGESVDJ: {
    using function = int (*)(void *, int, int, int, int, void *, int, float *,
                             void *, int, void *, int, void *, int, int *,
                             void *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCgesvdj"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.m, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<float *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldb,
                   reinterpret_cast<void *>(request.d), request.ldc,
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor));
    break;
  }
  case LUPINE_CUSOLVER_DN_CGESVDJ_BATCHED_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const void *, int,
                             const float *, const void *, int, const void *,
                             int, int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnCgesvdjBatched_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b),
                   reinterpret_cast<const void *>(request.c), request.ldb,
                   reinterpret_cast<const void *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_CGESVDJ_BATCHED: {
    using function = int (*)(void *, int, int, int, void *, int, float *,
                             void *, int, void *, int, void *, int, int *,
                             void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnCgesvdjBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<float *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldb,
                   reinterpret_cast<void *>(request.d), request.ldc,
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGESVDJ_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, const void *, int,
                             const double *, const void *, int, const void *,
                             int, int *, void *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZgesvdj_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.m, request.n,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<const void *>(request.c), request.ldb,
                   reinterpret_cast<const void *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor));
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGESVDJ: {
    using function = int (*)(void *, int, int, int, int, void *, int, double *,
                             void *, int, void *, int, void *, int, int *,
                             void *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnZgesvdj"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.m, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldb,
                   reinterpret_cast<void *>(request.d), request.ldc,
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor));
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGESVDJ_BATCHED_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const void *, int,
                             const double *, const void *, int, const void *,
                             int, int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZgesvdjBatched_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<const void *>(request.c), request.ldb,
                   reinterpret_cast<const void *>(request.d), request.ldc,
                   &lwork, reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_ZGESVDJ_BATCHED: {
    using function = int (*)(void *, int, int, int, void *, int, double *,
                             void *, int, void *, int, void *, int, int *,
                             void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZgesvdjBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.m, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldb,
                   reinterpret_cast<void *>(request.d), request.ldc,
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.preference),
                   reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_CREATE_PARAMS: {
    using function = int (*)(void **);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCreateParams"));
    response.status = call == nullptr ? 7 : call(&handle);
    response.handle = reinterpret_cast<uint64_t>(handle);
    break;
  }
  case LUPINE_CUSOLVER_DN_DESTROY_PARAMS: {
    using function = int (*)(void *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDestroyParams"));
    response.status = call == nullptr ? 7 : call(handle);
    break;
  }
  case LUPINE_CUSOLVER_DN_SET_ADV_OPTIONS: {
    using function = int (*)(void *, int, int);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSetAdvOptions"));
    response.status = call == nullptr
                          ? 7
                          : call(handle, request.attribute, request.algorithm);
    break;
  }
  case LUPINE_CUSOLVER_DN_XPOTRF_BUFFER_SIZE: {
    using function = int (*)(void *, void *, int, int64_t, int, const void *,
                             int64_t, int, size_t *, size_t *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXpotrf_bufferSize"));
    size_t device_bytes = 0, host_bytes = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, static_cast<int64_t>(request.rows),
                   request.a_type, reinterpret_cast<const void *>(request.a),
                   request.leading_dimension, request.compute_type,
                   &device_bytes, &host_bytes);
    uint64_t sizes[2] = {static_cast<uint64_t>(device_bytes),
                         static_cast<uint64_t>(host_bytes)};
    memcpy(response.payload, sizes, sizeof(sizes));
    response.payload_size = sizeof(sizes);
    break;
  }
  case LUPINE_CUSOLVER_DN_XPOTRF: {
    using function = int (*)(void *, void *, int, int64_t, int, void *, int64_t,
                             int, void *, size_t, void *, size_t, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXpotrf"));
    void *host_buffer = request.columns == 0
                            ? nullptr
                            : malloc(static_cast<size_t>(request.columns));
    response.status =
        call == nullptr || (request.columns != 0 && host_buffer == nullptr)
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, static_cast<int64_t>(request.rows),
                   request.a_type, reinterpret_cast<void *>(request.a),
                   request.leading_dimension, request.compute_type,
                   reinterpret_cast<void *>(request.workspace),
                   static_cast<size_t>(request.workspace_size), host_buffer,
                   static_cast<size_t>(request.columns),
                   reinterpret_cast<int *>(request.b));
    free(host_buffer);
    break;
  }
  case LUPINE_CUSOLVER_DN_XPOTRS: {
    using function = int (*)(void *, void *, int, int64_t, int64_t, int,
                             const void *, int64_t, int, void *, int64_t,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXpotrs"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, (int64_t)request.rows,
                   (int64_t)request.columns, request.a_type,
                   reinterpret_cast<const void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<void *>(request.b), request.stride_a,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_XGEQRF_BUFFER_SIZE: {
    using function = int (*)(void *, void *, int64_t, int64_t, int,
                             const void *, int64_t, int, const void *, int,
                             size_t *, size_t *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXgeqrf_bufferSize"));
    size_t device_bytes = 0, host_bytes = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   static_cast<int64_t>(request.rows),
                   static_cast<int64_t>(request.columns), request.a_type,
                   reinterpret_cast<const void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<const void *>(request.b),
                   request.compute_type, &device_bytes, &host_bytes);
    uint64_t sizes[2] = {static_cast<uint64_t>(device_bytes),
                         static_cast<uint64_t>(host_bytes)};
    memcpy(response.payload, sizes, sizeof(sizes));
    response.payload_size = sizeof(sizes);
    break;
  }
  case LUPINE_CUSOLVER_DN_XGEQRF: {
    using function = int (*)(void *, void *, int64_t, int64_t, int, void *,
                             int64_t, int, void *, int, void *, size_t, void *,
                             size_t, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXgeqrf"));
    size_t host_bytes = static_cast<size_t>(request.stride_a);
    void *host_buffer = host_bytes == 0 ? nullptr : malloc(host_bytes);
    response.status =
        call == nullptr || (host_bytes != 0 && host_buffer == nullptr)
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   static_cast<int64_t>(request.rows),
                   static_cast<int64_t>(request.columns), request.a_type,
                   reinterpret_cast<void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<void *>(request.b), request.compute_type,
                   reinterpret_cast<void *>(request.workspace),
                   static_cast<size_t>(request.workspace_size), host_buffer,
                   host_bytes, reinterpret_cast<int *>(request.c));
    free(host_buffer);
    break;
  }
  case LUPINE_CUSOLVER_DN_DORGQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const double *, int,
                             const double *, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDorgqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b), &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DORGQR: {
    using function = int (*)(void *, int, int, int, double *, int,
                             const double *, double *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDorgqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_SORGQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const float *, int,
                             const float *, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnSorgqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<const float *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b), &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_SORGQR: {
    using function = int (*)(void *, int, int, int, float *, int,
                             const float *, float *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSorgqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<float *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b),
                   reinterpret_cast<float *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_CUNGQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const void *, int,
                             const void *, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnCungqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const void *>(request.b), &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_CUNGQR: {
    using function = int (*)(void *, int, int, int, void *, int,
                             const void *, void *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCungqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_ZUNGQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const void *, int,
                             const void *, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZungqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const void *>(request.b), &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_ZUNGQR: {
    using function = int (*)(void *, int, int, int, void *, int,
                             const void *, void *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnZungqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n, request.k,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_XSYEV_BATCHED_BUFFER_SIZE: {
    using function = int (*)(void *, void *, int, int, int64_t, int,
                             const void *, int64_t, int, const void *, int,
                             size_t *, size_t *, int64_t);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXsyevBatched_bufferSize"));
    size_t device_bytes = 0, host_bytes = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, request.transb,
                   static_cast<int64_t>(request.rows), request.a_type,
                   reinterpret_cast<const void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<const void *>(request.b),
                   request.compute_type, &device_bytes, &host_bytes,
                   request.stride_b);
    uint64_t sizes[2] = {static_cast<uint64_t>(device_bytes),
                         static_cast<uint64_t>(host_bytes)};
    memcpy(response.payload, sizes, sizeof(sizes));
    response.payload_size = sizeof(sizes);
    break;
  }
  case LUPINE_CUSOLVER_DN_XSYEV_BATCHED: {
    using function = int (*)(void *, void *, int, int, int64_t, int, void *,
                             int64_t, int, void *, int, void *, size_t, void *,
                             size_t, int *, int64_t);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnXsyevBatched"));
    size_t host_bytes = static_cast<size_t>(request.stride_a);
    void *host_buffer = host_bytes == 0 ? nullptr : malloc(host_bytes);
    response.status =
        call == nullptr || (host_bytes != 0 && host_buffer == nullptr)
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, request.transb,
                   static_cast<int64_t>(request.rows), request.a_type,
                   reinterpret_cast<void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<void *>(request.b), request.compute_type,
                   reinterpret_cast<void *>(request.workspace),
                   static_cast<size_t>(request.workspace_size), host_buffer,
                   host_bytes, reinterpret_cast<int *>(request.c),
                   request.stride_b);
    free(host_buffer);
    break;
  }
  case LUPINE_CUSOLVER_DN_DORMQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, int, const double *,
                             int, const double *, const double *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDormqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const double *>(request.a),
                   request.lda, reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<const double *>(request.c), request.ldc,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DORMQR: {
    using function = int (*)(void *, int, int, int, int, int, const double *,
                             int, const double *, double *, int, double *, int,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDormqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const double *>(request.a),
                   request.lda, reinterpret_cast<const double *>(request.b),
                   reinterpret_cast<double *>(request.c), request.ldc,
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_DPOTRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, double *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDpotrf_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DPOTRF: {
    using function = int (*)(void *, int, int, double *, int, double *, int,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDpotrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.b));
    break;
  }
  case LUPINE_CUSOLVER_DN_DPOTRS: {
    using function = int (*)(void *, int, int, int, const double *, int,
                             double *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDpotrs"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b), request.ldb,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_DGEQRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, double *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDgeqrf_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DGEQRF: {
    using function = int (*)(void *, int, int, double *, int, double *,
                             double *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDgeqrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.m, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_DGESVD_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDgesvd_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr ? 7 : call(handle, request.m, request.n, &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DGESVD: {
    using function = int (*)(void *, signed char, signed char, int, int,
                             double *, int, double *, double *, int, double *,
                             int, double *, int, double *, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDgesvd"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, static_cast<signed char>(request.transa),
                   static_cast<signed char>(request.transb), request.m,
                   request.n, reinterpret_cast<double *>(request.a),
                   request.lda, reinterpret_cast<double *>(request.b),
                   reinterpret_cast<double *>(request.c), request.ldb,
                   reinterpret_cast<double *>(request.d), request.ldc,
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<double *>(request.a_descriptor),
                   reinterpret_cast<int *>(request.b_descriptor));
    break;
  }
  case LUPINE_CUSOLVER_DN_DSYEVD_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const double *, int,
                             const double *, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDsyevd_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b), &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DSYEVD: {
    using function = int (*)(void *, int, int, int, double *, int, double *,
                             double *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnDsyevd"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c));
    break;
  }
  case LUPINE_CUSOLVER_DN_DPOTRF_BATCHED: {
    using function = int (*)(void *, int, int, double **, int, int *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDpotrfBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<double **>(request.a), request.lda,
                   reinterpret_cast<int *>(request.b), request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_SPOTRF_BATCHED:
  case LUPINE_CUSOLVER_DN_CPOTRF_BATCHED:
  case LUPINE_CUSOLVER_DN_ZPOTRF_BATCHED: {
    using function = int (*)(void *, int, int, void **, int, int *, int);
    const char *symbol =
        request.opcode == LUPINE_CUSOLVER_DN_SPOTRF_BATCHED
            ? "cusolverDnSpotrfBatched"
            : request.opcode == LUPINE_CUSOLVER_DN_CPOTRF_BATCHED
                  ? "cusolverDnCpotrfBatched"
                  : "cusolverDnZpotrfBatched";
    auto call = reinterpret_cast<function>(solver_symbol(symbol));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<void **>(request.a), request.lda,
                   reinterpret_cast<int *>(request.b), request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_DPOTRS_BATCHED: {
    using function = int (*)(void *, int, int, int, const double **, int,
                             double **, int, int *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDpotrsBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const double **>(request.a), request.lda,
                   reinterpret_cast<double **>(request.b), request.ldb,
                   reinterpret_cast<int *>(request.c), request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_DSYEVJ_BATCHED_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, const double *, int,
                             const double *, int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDsyevjBatched_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b), &lwork,
                   reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_DSYEVJ_BATCHED: {
    using function = int (*)(void *, int, int, int, double *, int, double *,
                             double *, int, int *, void *, int);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnDsyevjBatched"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.transb, request.n,
                   reinterpret_cast<double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b),
                   reinterpret_cast<double *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.c),
                   reinterpret_cast<void *>(request.descriptor),
                   request.batch_count);
    break;
  }
  case LUPINE_CUSOLVER_DN_SORMQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, int, const float *,
                             int, const float *, const float *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnSormqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const float *>(request.a),
                   request.lda, reinterpret_cast<const float *>(request.b),
                   reinterpret_cast<const float *>(request.c), request.ldc,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_SORMQR: {
    using function = int (*)(void *, int, int, int, int, int, const float *,
                             int, const float *, float *, int, float *, int,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSormqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const float *>(request.a),
                   request.lda, reinterpret_cast<const float *>(request.b),
                   reinterpret_cast<float *>(request.c), request.ldc,
                   reinterpret_cast<float *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_CUNMQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             int, const void *, const void *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnCunmqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const void *>(request.a),
                   request.lda, reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<const void *>(request.c), request.ldc,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_CUNMQR: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             int, const void *, void *, int, void *, int,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnCunmqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const void *>(request.a),
                   request.lda, reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldc,
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_ZUNMQR_BUFFER_SIZE: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             int, const void *, const void *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZunmqr_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const void *>(request.a),
                   request.lda, reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<const void *>(request.c), request.ldc,
                   &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_ZUNMQR: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             int, const void *, void *, int, void *, int,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnZunmqr"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transb, request.transa, request.m, request.n,
                   request.k, reinterpret_cast<const void *>(request.a),
                   request.lda, reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<void *>(request.c), request.ldc,
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_SSYTRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, float *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSsytrf_bufferSize"));
    int lwork = 0;
    response.status = call == nullptr
                          ? 7
                          : call(handle, request.n,
                                 reinterpret_cast<float *>(request.a),
                                 request.lda, &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_SSYTRF: {
    using function = int (*)(void *, int, int, float *, int, int *, float *,
                             int, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnSsytrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<float *>(request.a), request.lda,
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<float *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_ZSYTRF_BUFFER_SIZE: {
    using function = int (*)(void *, int, void *, int, int *);
    auto call = reinterpret_cast<function>(
        solver_symbol("cusolverDnZsytrf_bufferSize"));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.n, reinterpret_cast<void *>(request.a),
                   request.lda, &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_ZSYTRF: {
    using function = int (*)(void *, int, int, void *, int, int *, void *, int,
                             int *);
    auto call =
        reinterpret_cast<function>(solver_symbol("cusolverDnZsytrf"));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_CSYTRF_BUFFER_SIZE:
  case LUPINE_CUSOLVER_DN_DSYTRF_BUFFER_SIZE: {
    const char *name =
        request.opcode == LUPINE_CUSOLVER_DN_CSYTRF_BUFFER_SIZE
            ? "cusolverDnCsytrf_bufferSize"
            : "cusolverDnDsytrf_bufferSize";
    using function = int (*)(void *, int, void *, int, int *);
    auto call = reinterpret_cast<function>(solver_symbol(name));
    int lwork = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.n, reinterpret_cast<void *>(request.a),
                   request.lda, &lwork);
    response.value = lwork;
    break;
  }
  case LUPINE_CUSOLVER_DN_CSYTRF:
  case LUPINE_CUSOLVER_DN_DSYTRF: {
    const char *name = request.opcode == LUPINE_CUSOLVER_DN_CSYTRF
                           ? "cusolverDnCsytrf"
                           : "cusolverDnDsytrf";
    using function = int (*)(void *, int, int, void *, int, int *, void *, int,
                             int *);
    auto call = reinterpret_cast<function>(solver_symbol(name));
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, request.n,
                   reinterpret_cast<void *>(request.a), request.lda,
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<void *>(request.workspace), request.value,
                   reinterpret_cast<int *>(request.d));
    break;
  }
  case LUPINE_CUSOLVER_DN_XSYTRS_BUFFER_SIZE: {
    using function = int (*)(void *, int, int64_t, int64_t, int, const void *,
                             int64_t, const int64_t *, int, void *, int64_t,
                             size_t *, size_t *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXsytrs_bufferSize"));
    size_t device_bytes = 0, host_bytes = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, (int64_t)request.rows,
                   (int64_t)request.columns, request.a_type,
                   reinterpret_cast<const void *>(request.a),
                   request.leading_dimension,
                   reinterpret_cast<const int64_t *>(request.b), request.b_type,
                   reinterpret_cast<void *>(request.c), request.stride_a,
                   &device_bytes, &host_bytes);
    uint64_t sizes[2] = {(uint64_t)device_bytes, (uint64_t)host_bytes};
    memcpy(response.payload, sizes, sizeof(sizes));
    response.payload_size = sizeof(sizes);
    break;
  }
  case LUPINE_CUSOLVER_DN_XSYTRS: {
    using function = int (*)(void *, int, int64_t, int64_t, int, const void *,
                             int64_t, const int64_t *, int, void *, int64_t,
                             void *, size_t, void *, size_t, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXsytrs"));
    std::vector<unsigned char> host_workspace((size_t)request.descriptor);
    response.status =
        call == nullptr
            ? 7
            : call(handle, request.transa, (int64_t)request.rows,
                   (int64_t)request.columns, request.a_type,
                   reinterpret_cast<const void *>(request.a),
                   request.leading_dimension,
                   reinterpret_cast<const int64_t *>(request.b), request.b_type,
                   reinterpret_cast<void *>(request.c), request.stride_a,
                   reinterpret_cast<void *>(request.workspace),
                   (size_t)request.workspace_size,
                   host_workspace.empty() ? nullptr : host_workspace.data(),
                   host_workspace.size(),
                   reinterpret_cast<int *>(request.preference));
    break;
  }
  case LUPINE_CUSOLVER_DN_XGEEV_BUFFER_SIZE: {
    using function = int (*)(void *, void *, int, int, int64_t, int,
                             const void *, int64_t, int, const void *, int,
                             const void *, int64_t, int, const void *, int64_t,
                             int, size_t *, size_t *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXgeev_bufferSize"));
    size_t device_bytes = 0, host_bytes = 0;
    response.status =
        call == nullptr
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, request.transb, (int64_t)request.rows,
                   request.a_type, reinterpret_cast<const void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<const void *>(request.b), request.c_type,
                   reinterpret_cast<const void *>(request.c), request.stride_a,
                   request.attribute, reinterpret_cast<const void *>(request.d),
                   request.stride_b, request.compute_type, &device_bytes,
                   &host_bytes);
    uint64_t sizes[2] = {(uint64_t)device_bytes, (uint64_t)host_bytes};
    memcpy(response.payload, sizes, sizeof(sizes));
    response.payload_size = sizeof(sizes);
    break;
  }
  case LUPINE_CUSOLVER_DN_XGEEV: {
    using function = int (*)(void *, void *, int, int, int64_t, int, void *,
                             int64_t, int, void *, int, void *, int64_t, int,
                             void *, int64_t, int, void *, size_t, void *,
                             size_t, int *);
    auto call = reinterpret_cast<function>(solver_symbol("cusolverDnXgeev"));
    std::vector<unsigned char> host_workspace((size_t)request.columns);
    response.status =
        call == nullptr
            ? 7
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   request.transa, request.transb, (int64_t)request.rows,
                   request.a_type, reinterpret_cast<void *>(request.a),
                   request.leading_dimension, request.b_type,
                   reinterpret_cast<void *>(request.b), request.c_type,
                   reinterpret_cast<void *>(request.c), request.stride_a,
                   request.attribute, reinterpret_cast<void *>(request.d),
                   request.stride_b, request.compute_type,
                   reinterpret_cast<void *>(request.workspace),
                   (size_t)request.workspace_size,
                   host_workspace.empty() ? nullptr : host_workspace.data(),
                   host_workspace.size(),
                   reinterpret_cast<int *>(request.preference));
    break;
  }
  case LUPINE_CUBLAS_SGETRS_BATCHED: {
    using function = int (*)(void *, int, int, int, const float *const *, int,
                             const int *, float *const *, int, int *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSgetrsBatched"));
    int info = 0;
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const float *const *>(request.a),
                   request.lda, reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<float *const *>(request.c), request.ldb,
                   &info, request.batch_count);
    response.value = info;
    break;
  }
  case LUPINE_CUBLAS_CGETRS_BATCHED: {
    using function = int (*)(void *, int, int, int, const void *const *, int,
                             const int *, void *const *, int, int *, int);
    auto call = reinterpret_cast<function>(symbol("cublasCgetrsBatched"));
    int info = 0;
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const void *const *>(request.a),
                   request.lda, reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<void *const *>(request.c), request.ldb,
                   &info, request.batch_count);
    response.value = info;
    break;
  }
  case LUPINE_CUBLAS_ZGETRS_BATCHED: {
    using function = int (*)(void *, int, int, int, const void *const *, int,
                             const int *, void *const *, int, int *, int);
    auto call = reinterpret_cast<function>(symbol("cublasZgetrsBatched"));
    int info = 0;
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const void *const *>(request.a),
                   request.lda, reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<void *const *>(request.c), request.ldb,
                   &info, request.batch_count);
    response.value = info;
    break;
  }
  case LUPINE_CUBLAS_DGETRS_BATCHED: {
    using function = int (*)(void *, int, int, int, const double *const *, int,
                             const int *, double *const *, int, int *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDgetrsBatched"));
    int info = 0;
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.n, request.k,
                   reinterpret_cast<const double *const *>(request.a),
                   request.lda, reinterpret_cast<const int *>(request.b),
                   reinterpret_cast<double *const *>(request.c), request.ldb,
                   &info, request.batch_count);
    response.value = info;
    break;
  }
  case LUPINE_CUBLAS_SDOT: {
    using function = int (*)(void *, int, const float *, int, const float *,
                             int, float *);
    auto call = reinterpret_cast<function>(symbol("cublasSdot_v2"));
    float host_result = 0.0f;
    float *result = request.pointer_mode == 0
                        ? &host_result
                        : reinterpret_cast<float *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<const float *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b), request.ldb,
                   result);
    memcpy(response.payload, &host_result, sizeof(host_result));
    response.payload_size = sizeof(host_result);
    break;
  }
  case LUPINE_CUBLAS_DOT_EX: {
    using function = int (*)(void *, int, const void *, int, int,
                             const void *, int, int, void *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasDotEx"));
    alignas(16) uint8_t host_result[16] = {};
    void *result = request.pointer_mode == 0
                       ? static_cast<void *>(host_result)
                       : reinterpret_cast<void *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<const void *>(request.a), request.a_type,
                   request.lda, reinterpret_cast<const void *>(request.b),
                   request.b_type, request.ldb, result, request.c_type,
                   request.compute_type);
    if (request.pointer_mode == 0 && response.status == 0) {
      response.payload_size =
          std::min<uint32_t>(request.scalar_size, sizeof(host_result));
      memcpy(response.payload, host_result, response.payload_size);
    }
    break;
  }
  case LUPINE_CUBLAS_STRSM_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, int,
                             const float *, const float *const *, int,
                             float *const *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasStrsmBatched"));
    const float *alpha = request.pointer_mode == 0
                             ? reinterpret_cast<const float *>(request.alpha_data)
                             : reinterpret_cast<const float *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transb, request.attribute, request.transa,
                   request.value, request.m, request.n, alpha,
                   reinterpret_cast<const float *const *>(request.a),
                   request.lda, reinterpret_cast<float *const *>(request.b),
                   request.ldb, request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_ZTRSM_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, int,
                             const void *, const void *const *, int,
                             void *const *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasZtrsmBatched"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transb, request.attribute, request.transa,
                   request.value, request.m, request.n, alpha,
                   reinterpret_cast<const void *const *>(request.a),
                   request.lda, reinterpret_cast<void *const *>(request.b),
                   request.ldb, request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_CTRSM_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, int,
                             const void *, const void *const *, int,
                             void *const *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasCtrsmBatched"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transb, request.attribute, request.transa,
                   request.value, request.m, request.n, alpha,
                   reinterpret_cast<const void *const *>(request.a),
                   request.lda, reinterpret_cast<void *const *>(request.b),
                   request.ldb, request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_DTRSM_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, int,
                             const double *, const double *const *, int,
                             double *const *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasDtrsmBatched"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    const double *alpha = request.pointer_mode == 0
                              ? reinterpret_cast<const double *>(aligned_alpha)
                              : reinterpret_cast<const double *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transb, request.attribute, request.transa,
                   request.value, request.m, request.n, alpha,
                   reinterpret_cast<const double *const *>(request.a),
                   request.lda, reinterpret_cast<double *const *>(request.b),
                   request.ldb, request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_DTRSM: {
    using function = int (*)(void *, int, int, int, int, int, int,
                             const double *, const double *, int, double *,
                             int);
    auto call = reinterpret_cast<function>(symbol("cublasDtrsm_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    const double *alpha = request.pointer_mode == 0
                              ? reinterpret_cast<const double *>(aligned_alpha)
                              : reinterpret_cast<const double *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transb, request.attribute, request.transa,
                   request.value, request.m, request.n, alpha,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b), request.ldb);
    break;
  }
  case LUPINE_CUBLAS_CGETRF_BATCHED: {
    using function = int (*)(void *, int, void *const *, int, int *, int *,
                             int);
    auto call = reinterpret_cast<function>(symbol("cublasCgetrfBatched"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<void *const *>(request.a), request.lda,
                   reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c), request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_DGETRF_BATCHED: {
    using function = int (*)(void *, int, double **, int, int *, int *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDgetrfBatched"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n, reinterpret_cast<double **>(request.a),
                   request.lda, reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c), request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_SGETRF_BATCHED: {
    using function = int (*)(void *, int, float **, int, int *, int *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSgetrfBatched"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n, reinterpret_cast<float **>(request.a),
                   request.lda, reinterpret_cast<int *>(request.b),
                   reinterpret_cast<int *>(request.c), request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_SGELS_BATCHED:
  case LUPINE_CUBLAS_DGELS_BATCHED:
  case LUPINE_CUBLAS_CGELS_BATCHED:
  case LUPINE_CUBLAS_ZGELS_BATCHED: {
    using function = int (*)(void *, int, int, int, int, void *const *, int,
                             void *const *, int, int *, int *, int);
    const char *name = request.opcode == LUPINE_CUBLAS_SGELS_BATCHED
                           ? "cublasSgelsBatched"
                       : request.opcode == LUPINE_CUBLAS_DGELS_BATCHED
                           ? "cublasDgelsBatched"
                       : request.opcode == LUPINE_CUBLAS_CGELS_BATCHED
                           ? "cublasCgelsBatched"
                           : "cublasZgelsBatched";
    auto call = reinterpret_cast<function>(symbol(name));
    int info = 0;
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.m, request.n, request.k,
                   reinterpret_cast<void *const *>(request.a), request.lda,
                   reinterpret_cast<void *const *>(request.b), request.ldb,
                   &info, reinterpret_cast<int *>(request.c),
                   request.batch_count);
    response.value = info;
    break;
  }
  case LUPINE_CUBLAS_SGEQRF_BATCHED:
  case LUPINE_CUBLAS_DGEQRF_BATCHED:
  case LUPINE_CUBLAS_CGEQRF_BATCHED:
  case LUPINE_CUBLAS_ZGEQRF_BATCHED: {
    using function = int (*)(void *, int, int, void *const *, int,
                             void *const *, int *, int);
    const char *name = request.opcode == LUPINE_CUBLAS_SGEQRF_BATCHED
                           ? "cublasSgeqrfBatched"
                       : request.opcode == LUPINE_CUBLAS_DGEQRF_BATCHED
                           ? "cublasDgeqrfBatched"
                       : request.opcode == LUPINE_CUBLAS_CGEQRF_BATCHED
                           ? "cublasCgeqrfBatched"
                           : "cublasZgeqrfBatched";
    auto call = reinterpret_cast<function>(symbol(name));
    int info = 0;
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.m, request.n,
                   reinterpret_cast<void *const *>(request.a), request.lda,
                   reinterpret_cast<void *const *>(request.b), &info,
                   request.batch_count);
    response.value = info;
    break;
  }
  case LUPINE_CUBLAS_DAXPY: {
    using function = int (*)(void *, int, const double *, const double *, int,
                             double *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDaxpy_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    const double *alpha = request.pointer_mode == 0
                              ? reinterpret_cast<const double *>(aligned_alpha)
                              : reinterpret_cast<const double *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n, alpha,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b), request.ldb);
    break;
  }
  case LUPINE_CUBLAS_DCOPY: {
    using function = int (*)(void *, int, const double *, int, double *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDcopy_v2"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<double *>(request.b), request.ldb);
    break;
  }
  case LUPINE_CUBLAS_DSCAL: {
    using function = int (*)(void *, int, const double *, double *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDscal_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    const double *alpha = request.pointer_mode == 0
                              ? reinterpret_cast<const double *>(aligned_alpha)
                              : reinterpret_cast<const double *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n, alpha,
                   reinterpret_cast<double *>(request.a), request.lda);
    break;
  }
  case LUPINE_CUBLAS_DNRM2:
  case LUPINE_CUBLAS_DASUM: {
    using function = int (*)(void *, int, const double *, int, double *);
    const char *name = request.opcode == LUPINE_CUBLAS_DNRM2
                           ? "cublasDnrm2_v2"
                           : "cublasDasum_v2";
    auto call = reinterpret_cast<function>(symbol(name));
    double host_result = 0.0;
    double *result = request.pointer_mode == 0
                         ? &host_result
                         : reinterpret_cast<double *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   result);
    if (request.pointer_mode == 0 && response.status == 0) {
      memcpy(response.payload, &host_result, sizeof(host_result));
      response.payload_size = sizeof(host_result);
    }
    break;
  }
  case LUPINE_CUBLAS_DSWAP: {
    using function = int (*)(void *, int, double *, int, double *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDswap_v2"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n, reinterpret_cast<double *>(request.a),
                   request.lda, reinterpret_cast<double *>(request.b),
                   request.ldb);
    break;
  }
  case LUPINE_CUBLAS_IDAMAX:
  case LUPINE_CUBLAS_IDAMIN: {
    using function = int (*)(void *, int, const double *, int, int *);
    const char *name = request.opcode == LUPINE_CUBLAS_IDAMAX
                           ? "cublasIdamax_v2"
                           : "cublasIdamin_v2";
    auto call = reinterpret_cast<function>(symbol(name));
    int host_result = 0;
    int *result = request.pointer_mode == 0
                      ? &host_result
                      : reinterpret_cast<int *>(request.c);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   result);
    if (request.pointer_mode == 0 && response.status == 0) {
      memcpy(response.payload, &host_result, sizeof(host_result));
      response.payload_size = sizeof(host_result);
    }
    break;
  }
  case LUPINE_CUBLAS_CREATE: {
    using function = int (*)(void **);
    auto call = reinterpret_cast<function>(symbol("cublasCreate_v2"));
    response.status = call == nullptr ? 13 : call(&handle);
    response.handle = reinterpret_cast<uint64_t>(handle);
    break;
  }
  case LUPINE_CUBLAS_DESTROY: {
    using function = int (*)(void *);
    auto call = reinterpret_cast<function>(symbol("cublasDestroy_v2"));
    response.status = call == nullptr ? 13 : call(handle);
    break;
  }
  case LUPINE_CUBLAS_SET_STREAM: {
    using function = int (*)(void *, void *);
    auto call = reinterpret_cast<function>(symbol("cublasSetStream_v2"));
    response.status = call == nullptr
                          ? 13
                          : call(handle, reinterpret_cast<void *>(request.stream));
    break;
  }
  case LUPINE_CUBLAS_SET_WORKSPACE: {
    using function = int (*)(void *, void *, size_t);
    auto call = reinterpret_cast<function>(symbol("cublasSetWorkspace_v2"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, reinterpret_cast<void *>(request.workspace),
                   static_cast<size_t>(request.workspace_size));
    break;
  }
  case LUPINE_CUBLAS_SET_MATH_MODE: {
    using function = int (*)(void *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSetMathMode"));
    response.status = call == nullptr ? 13 : call(handle, request.value);
    break;
  }
  case LUPINE_CUBLAS_SET_POINTER_MODE: {
    using function = int (*)(void *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSetPointerMode_v2"));
    response.status = call == nullptr ? 13 : call(handle, request.value);
    break;
  }
  case LUPINE_CUBLAS_SGEMM: {
    using function = int (*)(void *, int, int, int, int, int, const float *,
                             const float *, int, const float *, int,
                             const float *, float *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSgemm_v2"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, &request.alpha,
                   reinterpret_cast<const float *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b), request.ldb,
                   &request.beta, reinterpret_cast<float *>(request.c),
                   request.ldc);
    break;
  }
  case LUPINE_CUBLAS_SGEMV: {
    using function = int (*)(void *, int, int, int, const float *,
                             const float *, int, const float *, int,
                             const float *, float *, int);
    auto call = reinterpret_cast<function>(symbol("cublasSgemv_v2"));
    const float *alpha = request.pointer_mode == 0
                             ? reinterpret_cast<const float *>(request.alpha_data)
                             : reinterpret_cast<const float *>(request.d);
    const float *beta = request.pointer_mode == 0
                            ? reinterpret_cast<const float *>(request.beta_data)
                            : reinterpret_cast<const float *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.m, request.n, alpha,
                   reinterpret_cast<const float *>(request.a), request.lda,
                   reinterpret_cast<const float *>(request.b), request.ldb,
                   beta, reinterpret_cast<float *>(request.c), request.ldc);
    break;
  }
  case LUPINE_CUBLAS_DGEMV: {
    using function = int (*)(void *, int, int, int, const double *,
                             const double *, int, const double *, int,
                             const double *, double *, int);
    auto call = reinterpret_cast<function>(symbol("cublasDgemv_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const double *alpha = request.pointer_mode == 0
                              ? reinterpret_cast<const double *>(aligned_alpha)
                              : reinterpret_cast<const double *>(request.d);
    const double *beta = request.pointer_mode == 0
                             ? reinterpret_cast<const double *>(aligned_beta)
                             : reinterpret_cast<const double *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.m, request.n, alpha,
                   reinterpret_cast<const double *>(request.a), request.lda,
                   reinterpret_cast<const double *>(request.b), request.ldb,
                   beta, reinterpret_cast<double *>(request.c), request.ldc);
    break;
  }
  case LUPINE_CUBLAS_CGEMV:
  case LUPINE_CUBLAS_ZGEMV: {
    using function = int (*)(void *, int, int, int, const void *, const void *,
                             int, const void *, int, const void *, void *, int);
    const bool is_double = request.opcode == LUPINE_CUBLAS_ZGEMV;
    auto call = reinterpret_cast<function>(
        symbol(is_double ? "cublasZgemv_v2" : "cublasCgemv_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.d);
    const void *beta = request.pointer_mode == 0
                           ? static_cast<const void *>(aligned_beta)
                           : reinterpret_cast<const void *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.m, request.n, alpha,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const void *>(request.b), request.ldb,
                   beta, reinterpret_cast<void *>(request.c), request.ldc);
    break;
  }
  case LUPINE_CUBLAS_CDOTU:
  case LUPINE_CUBLAS_CDOTC:
  case LUPINE_CUBLAS_ZDOTU:
  case LUPINE_CUBLAS_ZDOTC: {
    using function = int (*)(void *, int, const void *, int, const void *, int,
                             void *);
    const char *name =
        request.opcode == LUPINE_CUBLAS_CDOTU ? "cublasCdotu_v2"
        : request.opcode == LUPINE_CUBLAS_CDOTC ? "cublasCdotc_v2"
        : request.opcode == LUPINE_CUBLAS_ZDOTU ? "cublasZdotu_v2"
                                                : "cublasZdotc_v2";
    auto call = reinterpret_cast<function>(symbol(name));
    alignas(16) unsigned char host_result[16] = {};
    void *result = request.pointer_mode == 0
                       ? static_cast<void *>(host_result)
                       : reinterpret_cast<void *>(request.c);
    response.status =
        call == nullptr || request.scalar_size > sizeof(host_result)
            ? 13
            : call(handle, request.n,
                   reinterpret_cast<const void *>(request.a), request.lda,
                   reinterpret_cast<const void *>(request.b), request.ldb,
                   result);
    if (request.pointer_mode == 0 && response.status == 0) {
      response.payload_size = request.scalar_size;
      memcpy(response.payload, host_result, request.scalar_size);
    }
    break;
  }
  case LUPINE_CUBLAS_CGEMM: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             const void *, int, const void *, int,
                             const void *, void *, int);
    auto call = reinterpret_cast<function>(symbol("cublasCgemm_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.d);
    const void *beta = request.pointer_mode == 0
                           ? static_cast<const void *>(aligned_beta)
                           : reinterpret_cast<const void *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, alpha, reinterpret_cast<const void *>(request.a),
                   request.lda, reinterpret_cast<const void *>(request.b),
                   request.ldb, beta, reinterpret_cast<void *>(request.c),
                   request.ldc);
    break;
  }
  case LUPINE_CUBLAS_ZGEMM: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             const void *, int, const void *, int,
                             const void *, void *, int);
    auto call = reinterpret_cast<function>(symbol("cublasZgemm_v2"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.d);
    const void *beta = request.pointer_mode == 0
                           ? static_cast<const void *>(aligned_beta)
                           : reinterpret_cast<const void *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, alpha, reinterpret_cast<const void *>(request.a),
                   request.lda, reinterpret_cast<const void *>(request.b),
                   request.ldb, beta, reinterpret_cast<void *>(request.c),
                   request.ldc);
    if (response.status != 0) {
      CUdeviceptr a_base = 0, b_base = 0, c_base = 0;
      size_t a_size = 0, b_size = 0, c_size = 0;
      CUresult a_result = cuMemGetAddressRange(
          &a_base, &a_size, static_cast<CUdeviceptr>(request.a));
      CUresult b_result = cuMemGetAddressRange(
          &b_base, &b_size, static_cast<CUdeviceptr>(request.b));
      CUresult c_result = cuMemGetAddressRange(
          &c_base, &c_size, static_cast<CUdeviceptr>(request.c));
      LUPINE_LOG_ERROR(
          "cublasZgemm failed status=" << response.status
          << " shape=" << request.m << "x" << request.n << "x" << request.k
          << " pointers A=" << reinterpret_cast<void *>(request.a)
          << " (lookup=" << a_result << " base="
          << reinterpret_cast<void *>(a_base) << " size=" << a_size << ")"
          << " B=" << reinterpret_cast<void *>(request.b)
          << " (lookup=" << b_result << " base="
          << reinterpret_cast<void *>(b_base) << " size=" << b_size << ")"
          << " C=" << reinterpret_cast<void *>(request.c)
          << " (lookup=" << c_result << " base="
          << reinterpret_cast<void *>(c_base) << " size=" << c_size << ")");
    }
    break;
  }
  case LUPINE_CUBLAS_DGEMM_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, const double *,
                             const double *const *, int,
                             const double *const *, int, const double *,
                             double *const *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasDgemmBatched"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const double *alpha = request.pointer_mode == 0
                              ? reinterpret_cast<const double *>(aligned_alpha)
                              : reinterpret_cast<const double *>(request.d);
    const double *beta = request.pointer_mode == 0
                             ? reinterpret_cast<const double *>(aligned_beta)
                             : reinterpret_cast<const double *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, alpha,
                   reinterpret_cast<const double *const *>(request.a),
                   request.lda,
                   reinterpret_cast<const double *const *>(request.b),
                   request.ldb, beta,
                   reinterpret_cast<double *const *>(request.c), request.ldc,
                   request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_GEMM_EX: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             const void *, int, int, const void *, int, int,
                             const void *, void *, int, int, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasGemmEx"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    response.status =
        call == nullptr || request.scalar_size > sizeof(request.alpha_data)
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, aligned_alpha,
                   reinterpret_cast<const void *>(request.a), request.a_type,
                   request.lda, reinterpret_cast<const void *>(request.b),
                   request.b_type, request.ldb, aligned_beta,
                   reinterpret_cast<void *>(request.c), request.c_type,
                   request.ldc, request.compute_type, request.algorithm);
    if (response.status != 0) {
      CUdeviceptr a_base = 0, b_base = 0, c_base = 0;
      size_t a_size = 0, b_size = 0, c_size = 0;
      CUresult a_result = cuMemGetAddressRange(
          &a_base, &a_size, static_cast<CUdeviceptr>(request.a));
      CUresult b_result = cuMemGetAddressRange(
          &b_base, &b_size, static_cast<CUdeviceptr>(request.b));
      CUresult c_result = cuMemGetAddressRange(
          &c_base, &c_size, static_cast<CUdeviceptr>(request.c));
      LUPINE_LOG_ERROR(
          "cublasGemmEx failed status=" << response.status
          << " types=" << request.a_type << "/" << request.b_type << "/"
          << request.c_type << " compute=" << request.compute_type
          << " shape=" << request.m << "x" << request.n << "x" << request.k
          << " pointers A=" << reinterpret_cast<void *>(request.a)
          << " (lookup=" << a_result << " base="
          << reinterpret_cast<void *>(a_base) << " size=" << a_size << ")"
          << " B=" << reinterpret_cast<void *>(request.b)
          << " (lookup=" << b_result << " base="
          << reinterpret_cast<void *>(b_base) << " size=" << b_size << ")"
          << " C=" << reinterpret_cast<void *>(request.c)
          << " (lookup=" << c_result << " base="
          << reinterpret_cast<void *>(c_base) << " size=" << c_size << ")");
    }
    break;
  }
  case LUPINE_CUBLAS_SGEMM_EX: {
    using function = int (*)(void *, int, int, int, int, int, const float *,
                             const void *, int, int, const void *, int, int,
                             const float *, void *, int, int);
    auto call = reinterpret_cast<function>(symbol("cublasSgemmEx"));
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, &request.alpha,
                   reinterpret_cast<const void *>(request.a), request.a_type,
                   request.lda, reinterpret_cast<const void *>(request.b),
                   request.b_type, request.ldb, &request.beta,
                   reinterpret_cast<void *>(request.c), request.c_type,
                   request.ldc);
    break;
  }
  case LUPINE_CUBLAS_GEMM_STRIDED_BATCHED_EX: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             const void *, int, int, int64_t, const void *, int,
                             int, int64_t, const void *, void *, int, int,
                             int64_t, int, int, int);
    auto call = reinterpret_cast<function>(
        symbol("cublasGemmStridedBatchedEx"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    response.status =
        call == nullptr || request.scalar_size > sizeof(request.alpha_data)
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, aligned_alpha,
                   reinterpret_cast<const void *>(request.a), request.a_type,
                   request.lda, request.stride_a,
                   reinterpret_cast<const void *>(request.b), request.b_type,
                   request.ldb, request.stride_b, aligned_beta,
                   reinterpret_cast<void *>(request.c), request.c_type,
                   request.ldc, request.stride_c, request.batch_count,
                   request.compute_type, request.algorithm);
    break;
  }
  case LUPINE_CUBLAS_ZGEMM_STRIDED_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             const void *, int, int64_t, const void *, int,
                             int64_t, const void *, void *, int, int64_t, int);
    auto call = reinterpret_cast<function>(
        symbol("cublasZgemmStridedBatched"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.d);
    const void *beta = request.pointer_mode == 0
                           ? static_cast<const void *>(aligned_beta)
                           : reinterpret_cast<const void *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, alpha, reinterpret_cast<const void *>(request.a),
                   request.lda, request.stride_a,
                   reinterpret_cast<const void *>(request.b), request.ldb,
                   request.stride_b, beta, reinterpret_cast<void *>(request.c),
                   request.ldc, request.stride_c, request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_CGEMM_STRIDED_BATCHED: {
    using function = int (*)(void *, int, int, int, int, int, const void *,
                             const void *, int, int64_t, const void *, int,
                             int64_t, const void *, void *, int, int64_t, int);
    auto call = reinterpret_cast<function>(
        symbol("cublasCgemmStridedBatched"));
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const void *alpha = request.pointer_mode == 0
                            ? static_cast<const void *>(aligned_alpha)
                            : reinterpret_cast<const void *>(request.d);
    const void *beta = request.pointer_mode == 0
                           ? static_cast<const void *>(aligned_beta)
                           : reinterpret_cast<const void *>(request.workspace);
    response.status =
        call == nullptr
            ? 13
            : call(handle, request.transa, request.transb, request.m, request.n,
                   request.k, alpha, reinterpret_cast<const void *>(request.a),
                   request.lda, request.stride_a,
                   reinterpret_cast<const void *>(request.b), request.ldb,
                   request.stride_b, beta, reinterpret_cast<void *>(request.c),
                   request.ldc, request.stride_c, request.batch_count);
    break;
  }
  case LUPINE_CUBLAS_LT_MATMUL_DESC_CREATE: {
    using function = int (*)(void **, int, int);
    auto call = reinterpret_cast<function>(lt_symbol("cublasLtMatmulDescCreate"));
    void *descriptor = nullptr;
    response.status = call == nullptr
                          ? 13
                          : call(&descriptor, request.compute_type,
                                 request.c_type);
    response.handle = reinterpret_cast<uint64_t>(descriptor);
    break;
  }
  case LUPINE_CUBLAS_LT_MATMUL_DESC_DESTROY:
  case LUPINE_CUBLAS_LT_MATRIX_LAYOUT_DESTROY:
  case LUPINE_CUBLAS_LT_PREFERENCE_DESTROY: {
    using function = int (*)(void *);
    const char *name = request.opcode == LUPINE_CUBLAS_LT_MATMUL_DESC_DESTROY
                           ? "cublasLtMatmulDescDestroy"
                       : request.opcode == LUPINE_CUBLAS_LT_MATRIX_LAYOUT_DESTROY
                           ? "cublasLtMatrixLayoutDestroy"
                           : "cublasLtMatmulPreferenceDestroy";
    auto call = reinterpret_cast<function>(lt_symbol(name));
    response.status = call == nullptr
                          ? 13
                          : call(reinterpret_cast<void *>(request.descriptor));
    break;
  }
  case LUPINE_CUBLAS_LT_MATMUL_DESC_SET_ATTRIBUTE:
  case LUPINE_CUBLAS_LT_MATRIX_LAYOUT_SET_ATTRIBUTE:
  case LUPINE_CUBLAS_LT_PREFERENCE_SET_ATTRIBUTE: {
    using function = int (*)(void *, int, const void *, size_t);
    const char *name =
        request.opcode == LUPINE_CUBLAS_LT_MATMUL_DESC_SET_ATTRIBUTE
            ? "cublasLtMatmulDescSetAttribute"
        : request.opcode == LUPINE_CUBLAS_LT_MATRIX_LAYOUT_SET_ATTRIBUTE
            ? "cublasLtMatrixLayoutSetAttribute"
            : "cublasLtMatmulPreferenceSetAttribute";
    auto call = reinterpret_cast<function>(lt_symbol(name));
    response.status =
        call == nullptr || request.payload_size > sizeof(request.payload)
            ? 13
            : call(reinterpret_cast<void *>(request.descriptor),
                   request.attribute, request.payload, request.payload_size);
    break;
  }
  case LUPINE_CUBLAS_LT_MATRIX_LAYOUT_CREATE: {
    using function = int (*)(void **, int, uint64_t, uint64_t, int64_t);
    auto call = reinterpret_cast<function>(lt_symbol("cublasLtMatrixLayoutCreate"));
    void *descriptor = nullptr;
    response.status =
        call == nullptr
            ? 13
            : call(&descriptor, request.a_type, request.rows, request.columns,
                   request.leading_dimension);
    response.handle = reinterpret_cast<uint64_t>(descriptor);
    break;
  }
  case LUPINE_CUBLAS_LT_PREFERENCE_CREATE: {
    using function = int (*)(void **);
    auto call = reinterpret_cast<function>(lt_symbol("cublasLtMatmulPreferenceCreate"));
    void *preference = nullptr;
    response.status = call == nullptr ? 13 : call(&preference);
    response.handle = reinterpret_cast<uint64_t>(preference);
    break;
  }
  case LUPINE_CUBLAS_LT_HEURISTIC: {
    using function = int (*)(void *, void *, void *, void *, void *, void *,
                             void *, int, void *, int *);
    auto call = reinterpret_cast<function>(
        lt_symbol("cublasLtMatmulAlgoGetHeuristic"));
    constexpr size_t result_size = 96;
    alignas(16) unsigned char results[8 * result_size] = {};
    int returned = 0;
    response.status =
        call == nullptr || request.requested_algorithms <= 0 ||
                request.requested_algorithms > 8
            ? 13
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   reinterpret_cast<void *>(request.a_descriptor),
                   reinterpret_cast<void *>(request.b_descriptor),
                   reinterpret_cast<void *>(request.c_descriptor),
                   reinterpret_cast<void *>(request.d_descriptor),
                   reinterpret_cast<void *>(request.preference),
                   request.requested_algorithms, results, &returned);
    if (response.status == 0) {
      response.returned_algorithms = returned;
      response.payload_size = static_cast<uint32_t>(returned * result_size);
      memcpy(response.payload, results, response.payload_size);
    }
    break;
  }
  case LUPINE_CUBLAS_LT_MATMUL: {
    using function = int (*)(void *, void *, const void *, const void *, void *,
                             const void *, void *, const void *, const void *,
                             void *, void *, void *, const void *, void *,
                             size_t, void *);
    auto call = reinterpret_cast<function>(lt_symbol("cublasLtMatmul"));
    const void *algorithm = request.payload_size == 64 ? request.payload : nullptr;
    alignas(16) unsigned char aligned_alpha[16] = {};
    alignas(16) unsigned char aligned_beta[16] = {};
    memcpy(aligned_alpha, request.alpha_data, sizeof(aligned_alpha));
    memcpy(aligned_beta, request.beta_data, sizeof(aligned_beta));
    const bool explicit_scalar_presence = (request.value & 0x100) != 0;
    const bool host_alpha = request.pointer_mode == 0;
    const bool host_beta = request.pointer_mode == 0 || request.pointer_mode == 4;
    const void *alpha = host_alpha
                            ? (explicit_scalar_presence &&
                                       (request.value & 1) == 0
                                   ? nullptr
                                   : static_cast<const void *>(aligned_alpha))
                            : reinterpret_cast<const void *>(request.preference);
    const void *beta = host_beta
                           ? (explicit_scalar_presence &&
                                      (request.value & 2) == 0
                                  ? nullptr
                                  : static_cast<const void *>(aligned_beta))
                           : request.pointer_mode == 3
                                 ? nullptr
                                 : reinterpret_cast<const void *>(request.rows);
    response.status =
        call == nullptr || request.scalar_size > sizeof(request.alpha_data)
            ? 13
            : call(handle, reinterpret_cast<void *>(request.descriptor),
                   alpha,
                   reinterpret_cast<const void *>(request.a),
                   reinterpret_cast<void *>(request.a_descriptor),
                   reinterpret_cast<const void *>(request.b),
                   reinterpret_cast<void *>(request.b_descriptor),
                   beta,
                   reinterpret_cast<const void *>(request.c),
                   reinterpret_cast<void *>(request.c_descriptor),
                   reinterpret_cast<void *>(request.d),
                   reinterpret_cast<void *>(request.d_descriptor), algorithm,
                   reinterpret_cast<void *>(request.workspace),
                   request.workspace_size,
                   reinterpret_cast<void *>(request.stream));
    break;
  }
  default:
    response.status = 15;
    break;
  }
  }
  if (requested_context != nullptr) {
    (void)cuCtxSetCurrent(requested_context);
  } else if (saved_context_result == CUDA_SUCCESS) {
    (void)cuCtxSetCurrent(saved_context);
  }
  if (request.asynchronous != 0) return 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &response, sizeof(response)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_lupineNcclCall(conn_t *conn) {
  lupine_nccl_request request;
  if (rpc_read(conn, &request, sizeof(request)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  struct nccl_unique_id {
    unsigned char bytes[128];
  };
  using nccl_comm = void *;
  using nccl_stream = void *;
  static void *library = [] {
    void *handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      handle = dlopen(
          "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/"
          "libnccl.so.2",
          RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr) {
      LUPINE_LOG_ERROR("Unable to load server NCCL: " << dlerror());
    }
    return handle;
  }();
  auto symbol = [](const char *name) -> void * {
    return library == nullptr ? nullptr : dlsym(library, name);
  };

  lupine_nccl_response response;
  nccl_comm communicator = reinterpret_cast<nccl_comm>(request.communicator);
  nccl_stream stream = reinterpret_cast<nccl_stream>(request.stream);
  switch (request.opcode) {
  case LUPINE_NCCL_GET_UNIQUE_ID: {
    using function = int (*)(nccl_unique_id *);
    auto call = reinterpret_cast<function>(symbol("ncclGetUniqueId"));
    nccl_unique_id id;
    response.result = call == nullptr ? 2 : call(&id);
    if (response.result == 0) {
      memcpy(response.unique_id, id.bytes, sizeof(id.bytes));
    }
    break;
  }
  case LUPINE_NCCL_INIT_RANK: {
    using function = int (*)(nccl_comm *, int, nccl_unique_id, int);
    auto call = reinterpret_cast<function>(symbol("ncclCommInitRank"));
    nccl_unique_id id;
    memcpy(id.bytes, request.unique_id, sizeof(id.bytes));
    response.result =
        call == nullptr ? 2 : call(&communicator, request.ranks, id, request.rank);
    response.communicator = reinterpret_cast<uint64_t>(communicator);
    break;
  }
  case LUPINE_NCCL_GROUP_START:
  case LUPINE_NCCL_GROUP_END: {
    using function = int (*)(void);
    const char *name = request.opcode == LUPINE_NCCL_GROUP_START
                           ? "ncclGroupStart"
                           : "ncclGroupEnd";
    auto call = reinterpret_cast<function>(symbol(name));
    response.result = call == nullptr ? 2 : call();
    break;
  }
  case LUPINE_NCCL_ALL_REDUCE: {
    using function = int (*)(const void *, void *, size_t, int, int, nccl_comm,
                             nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclAllReduce"));
    response.result = call == nullptr
                          ? 2
                          : call(reinterpret_cast<const void *>(request.send_buffer),
                                 reinterpret_cast<void *>(request.receive_buffer),
                                 request.count, request.datatype,
                                 request.reduction, communicator, stream);
    break;
  }
  case LUPINE_NCCL_ALL_GATHER: {
    using function = int (*)(const void *, void *, size_t, int, nccl_comm,
                             nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclAllGather"));
    response.result = call == nullptr
                          ? 2
                          : call(reinterpret_cast<const void *>(request.send_buffer),
                                 reinterpret_cast<void *>(request.receive_buffer),
                                 request.count, request.datatype, communicator,
                                 stream);
    break;
  }
  case LUPINE_NCCL_REDUCE_SCATTER: {
    using function = int (*)(const void *, void *, size_t, int, int, nccl_comm,
                             nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclReduceScatter"));
    response.result = call == nullptr
                          ? 2
                          : call(reinterpret_cast<const void *>(request.send_buffer),
                                 reinterpret_cast<void *>(request.receive_buffer),
                                 request.count, request.datatype,
                                 request.reduction, communicator, stream);
    break;
  }
  case LUPINE_NCCL_BROADCAST: {
    using function = int (*)(const void *, void *, size_t, int, int, nccl_comm,
                             nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclBroadcast"));
    response.result = call == nullptr
                          ? 2
                          : call(reinterpret_cast<const void *>(request.send_buffer),
                                 reinterpret_cast<void *>(request.receive_buffer),
                                 request.count, request.datatype, request.root,
                                 communicator, stream);
    break;
  }
  case LUPINE_NCCL_REDUCE: {
    using function = int (*)(const void *, void *, size_t, int, int, int,
                             nccl_comm, nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclReduce"));
    response.result = call == nullptr ? 2 : call(
        reinterpret_cast<const void *>(request.send_buffer),
        reinterpret_cast<void *>(request.receive_buffer), request.count,
        request.datatype, request.reduction, request.root, communicator, stream);
    break;
  }
  case LUPINE_NCCL_BCAST: {
    using function = int (*)(void *, size_t, int, int, nccl_comm, nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclBcast"));
    response.result = call == nullptr ? 2 : call(
        reinterpret_cast<void *>(request.send_buffer), request.count,
        request.datatype, request.root, communicator, stream);
    break;
  }
  case LUPINE_NCCL_ALL_TO_ALL: {
    using function = int (*)(const void *, void *, size_t, int, nccl_comm,
                             nccl_stream);
    auto call = reinterpret_cast<function>(symbol("ncclAlltoAll"));
    response.result = call == nullptr ? 2 : call(
        reinterpret_cast<const void *>(request.send_buffer),
        reinterpret_cast<void *>(request.receive_buffer), request.count,
        request.datatype, communicator, stream);
    break;
  }
  case LUPINE_NCCL_GATHER:
  case LUPINE_NCCL_SCATTER: {
    using function = int (*)(const void *, void *, size_t, int, int, nccl_comm,
                             nccl_stream);
    const char *name = request.opcode == LUPINE_NCCL_GATHER ? "ncclGather"
                                                            : "ncclScatter";
    auto call = reinterpret_cast<function>(symbol(name));
    response.result = call == nullptr ? 2 : call(
        reinterpret_cast<const void *>(request.send_buffer),
        reinterpret_cast<void *>(request.receive_buffer), request.count,
        request.datatype, request.root, communicator, stream);
    break;
  }
  case LUPINE_NCCL_SEND:
  case LUPINE_NCCL_RECV: {
    if (request.opcode == LUPINE_NCCL_SEND) {
      using function = int (*)(const void *, size_t, int, int, nccl_comm,
                               nccl_stream);
      auto call = reinterpret_cast<function>(symbol("ncclSend"));
      response.result = call == nullptr ? 2 : call(
          reinterpret_cast<const void *>(request.send_buffer), request.count,
          request.datatype, request.root, communicator, stream);
    } else {
      using function = int (*)(void *, size_t, int, int, nccl_comm,
                               nccl_stream);
      auto call = reinterpret_cast<function>(symbol("ncclRecv"));
      response.result = call == nullptr ? 2 : call(
          reinterpret_cast<void *>(request.receive_buffer), request.count,
          request.datatype, request.root, communicator, stream);
    }
    break;
  }
  case LUPINE_NCCL_COMM_COUNT:
  case LUPINE_NCCL_COMM_CUDA_DEVICE:
  case LUPINE_NCCL_COMM_USER_RANK: {
    using function = int (*)(const nccl_comm, int *);
    const char *name = request.opcode == LUPINE_NCCL_COMM_COUNT
                           ? "ncclCommCount"
                           : (request.opcode == LUPINE_NCCL_COMM_CUDA_DEVICE
                                  ? "ncclCommCuDevice"
                                  : "ncclCommUserRank");
    auto call = reinterpret_cast<function>(symbol(name));
    response.result = call == nullptr ? 2 : call(communicator, &response.value);
    break;
  }
  case LUPINE_NCCL_COMM_REGISTER: {
    using function = int (*)(const nccl_comm, void *, size_t, void **);
    auto call = reinterpret_cast<function>(symbol("ncclCommRegister"));
    void *handle = nullptr;
    response.result = call == nullptr ? 2 : call(
        communicator, reinterpret_cast<void *>(request.send_buffer),
        request.count, &handle);
    response.handle = reinterpret_cast<uint64_t>(handle);
    break;
  }
  case LUPINE_NCCL_COMM_DEREGISTER: {
    using function = int (*)(const nccl_comm, void *);
    auto call = reinterpret_cast<function>(symbol("ncclCommDeregister"));
    response.result = call == nullptr ? 2 : call(
        communicator, reinterpret_cast<void *>(request.send_buffer));
    break;
  }
  case LUPINE_NCCL_COMM_SPLIT: {
    using function = int (*)(nccl_comm, int, int, nccl_comm *, void *);
    auto call = reinterpret_cast<function>(symbol("ncclCommSplit"));
    nccl_comm new_communicator = nullptr;
    response.result = call == nullptr ? 2 : call(
        communicator, request.rank, request.root, &new_communicator, nullptr);
    response.communicator = reinterpret_cast<uint64_t>(new_communicator);
    break;
  }
  case LUPINE_NCCL_LAST_ERROR: {
    using function = const char *(*)(const nccl_comm);
    auto call = reinterpret_cast<function>(symbol("ncclGetLastError"));
    const char *message = call == nullptr ? "server NCCL symbol unavailable"
                                          : call(communicator);
    response.result = 0;
    if (message != nullptr) {
      strncpy(response.error_string, message, sizeof(response.error_string) - 1);
    }
    break;
  }
  case LUPINE_NCCL_ASYNC_ERROR: {
    using function = int (*)(nccl_comm, int *);
    auto call = reinterpret_cast<function>(symbol("ncclCommGetAsyncError"));
    response.result =
        call == nullptr ? 2 : call(communicator, &response.async_error);
    break;
  }
  case LUPINE_NCCL_MEM_ALLOC: {
    using function = int (*)(void **, size_t);
    auto call = reinterpret_cast<function>(symbol("ncclMemAlloc"));
    void *pointer = nullptr;
    response.result = call == nullptr ? 2 : call(&pointer, request.count);
    response.handle = reinterpret_cast<uint64_t>(pointer);
    break;
  }
  case LUPINE_NCCL_MEM_FREE: {
    using function = int (*)(void *);
    auto call = reinterpret_cast<function>(symbol("ncclMemFree"));
    response.result = call == nullptr
                          ? 2
                          : call(reinterpret_cast<void *>(request.send_buffer));
    break;
  }
  case LUPINE_NCCL_COMM_SUSPEND: {
    using function = int (*)(nccl_comm, int);
    auto call = reinterpret_cast<function>(symbol("ncclCommSuspend"));
    response.result = call == nullptr ? 2 : call(communicator, request.root);
    break;
  }
  case LUPINE_NCCL_COMM_RESUME: {
    using function = int (*)(nccl_comm);
    auto call = reinterpret_cast<function>(symbol("ncclCommResume"));
    response.result = call == nullptr ? 2 : call(communicator);
    break;
  }
  case LUPINE_NCCL_COMM_MEM_STATS: {
    using function = int (*)(nccl_comm, int, uint64_t *);
    auto call = reinterpret_cast<function>(symbol("ncclCommMemStats"));
    response.result =
        call == nullptr ? 2 : call(communicator, request.root, &response.handle);
    break;
  }
  case LUPINE_NCCL_WINDOW_REGISTER: {
    using function = int (*)(nccl_comm, void *, size_t, void **, int);
    auto call = reinterpret_cast<function>(symbol("ncclCommWindowRegister"));
    void *window = nullptr;
    response.result = call == nullptr ? 2 : call(
        communicator, reinterpret_cast<void *>(request.send_buffer),
        request.count, &window, request.root);
    response.handle = reinterpret_cast<uint64_t>(window);
    break;
  }
  case LUPINE_NCCL_WINDOW_DEREGISTER: {
    using function = int (*)(nccl_comm, void *);
    auto call = reinterpret_cast<function>(symbol("ncclCommWindowDeregister"));
    response.result = call == nullptr
                          ? 2
                          : call(communicator,
                                 reinterpret_cast<void *>(request.send_buffer));
    break;
  }
  case LUPINE_NCCL_WINDOW_USER_POINTER: {
    using function = int (*)(nccl_comm, void *, void **);
    auto call = reinterpret_cast<function>(symbol("ncclWinGetUserPtr"));
    void *pointer = nullptr;
    response.result = call == nullptr ? 2 : call(
        communicator, reinterpret_cast<void *>(request.send_buffer), &pointer);
    response.handle = reinterpret_cast<uint64_t>(pointer);
    break;
  }
  case LUPINE_NCCL_COMM_GET_UNIQUE_ID: {
    using function = int (*)(nccl_comm, nccl_unique_id *);
    auto call = reinterpret_cast<function>(symbol("ncclCommGetUniqueId"));
    nccl_unique_id id;
    response.result = call == nullptr ? 2 : call(communicator, &id);
    if (response.result == 0) {
      memcpy(response.unique_id, id.bytes, sizeof(id.bytes));
    }
    break;
  }
  case LUPINE_NCCL_FINALIZE:
  case LUPINE_NCCL_DESTROY:
  case LUPINE_NCCL_ABORT: {
    using function = int (*)(nccl_comm);
    const char *name = request.opcode == LUPINE_NCCL_FINALIZE
                           ? "ncclCommFinalize"
                           : (request.opcode == LUPINE_NCCL_DESTROY
                                  ? "ncclCommDestroy"
                                  : "ncclCommAbort");
    auto call = reinterpret_cast<function>(symbol(name));
    response.result = call == nullptr ? 2 : call(communicator);
    break;
  }
  default:
    response.result = 2;
    break;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &response, sizeof(response)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
// Serves LUPINE_RPC_lupineDeviceSnapshot: every immutable per-device value the
// client caches, for every device, in one response. Mutable state (primary
// context state, context limits) is deliberately excluded. The response is all
// or nothing: any query failure fails the whole RPC and the client falls back
// to the per-call paths. Individual attributes the driver rejects are simply
// absent from the pair list; that is expected, not an error.
struct lupine_device_snapshot_record {
  char name[LUPINE_DEVICE_SNAPSHOT_NAME_BYTES] = {};
  CUuuid uuid = {};
  uint64_t total_mem = 0;
  uint32_t pair_count = 0;
  std::vector<int32_t> pairs;
};

static CUresult
lupine_build_device_snapshot_record(size_t ordinal,
                                    lupine_device_snapshot_record *record) {
  CUdevice device = 0;
  size_t bytes = 0;
  CUresult result = cuDeviceGet(&device, static_cast<int>(ordinal));
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetName(record->name, sizeof(record->name), device);
    record->name[sizeof(record->name) - 1] = '\0';
  }
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetUuid_v2(&record->uuid, device);
  }
  if (result == CUDA_SUCCESS) {
    result = cuDeviceTotalMem_v2(&bytes, device);
    record->total_mem = bytes;
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }

  try {
    record->pairs.reserve(static_cast<size_t>(CU_DEVICE_ATTRIBUTE_MAX - 1) * 2);
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  for (int attrib = 1; attrib < CU_DEVICE_ATTRIBUTE_MAX; ++attrib) {
    int value = 0;
    if (cuDeviceGetAttribute(&value, static_cast<CUdevice_attribute>(attrib),
                             device) == CUDA_SUCCESS) {
      record->pairs.push_back(static_cast<int32_t>(attrib));
      record->pairs.push_back(static_cast<int32_t>(value));
    }
  }
  record->pair_count = static_cast<uint32_t>(record->pairs.size() / 2);
  return CUDA_SUCCESS;
}

int handle_manual_lupineDeviceSnapshot(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  int device_count = 0;
  CUresult result = cuDeviceGetCount(&device_count);
  if (result == CUDA_SUCCESS && device_count < 0) {
    result = CUDA_ERROR_UNKNOWN;
  }

  // rpc_write queues iovecs that are only sent at rpc_write_end, so all
  // records are built first in storage that stays stable until then.
  std::vector<lupine_device_snapshot_record> records;
  std::vector<CUresult> record_results;
  if (result == CUDA_SUCCESS) {
    try {
      records.resize(static_cast<size_t>(device_count));
      record_results.resize(records.size(), CUDA_ERROR_UNKNOWN);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  if (result == CUDA_SUCCESS && !records.empty()) {
    std::vector<std::thread> workers;
    auto build_record = [&records, &record_results](size_t ordinal) {
      record_results[ordinal] =
          lupine_build_device_snapshot_record(ordinal, &records[ordinal]);
    };
    size_t next_ordinal = 1;
    try {
      workers.reserve(records.size() - 1);
      for (; next_ordinal < records.size(); ++next_ordinal) {
        workers.emplace_back(build_record, next_ordinal);
      }
    } catch (...) {
      // Any unlaunched devices fall back to this RPC thread below.
    }

    build_record(0);
    for (size_t ordinal = next_ordinal; ordinal < records.size(); ++ordinal) {
      build_record(ordinal);
    }
    for (auto &worker : workers) {
      worker.join();
    }
    for (CUresult record_result : record_results) {
      if (record_result != CUDA_SUCCESS) {
        result = record_result;
        break;
      }
    }
  }

  uint32_t devices = static_cast<uint32_t>(records.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0) {
    return -1;
  }
  if (result != CUDA_SUCCESS) {
    return rpc_write_end(conn) < 0 ? -1 : 0;
  }
  if (rpc_write(conn, &devices, sizeof(devices)) < 0) {
    return -1;
  }
  for (const auto &record : records) {
    if (rpc_write(conn, record.name, sizeof(record.name)) < 0 ||
        rpc_write(conn, &record.uuid, sizeof(record.uuid)) < 0 ||
        rpc_write(conn, &record.total_mem, sizeof(record.total_mem)) < 0 ||
        rpc_write(conn, &record.pair_count, sizeof(record.pair_count)) < 0 ||
        (record.pair_count != 0 &&
         rpc_write(conn, record.pairs.data(),
                   record.pairs.size() * sizeof(int32_t)) < 0)) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
}

int handle_manual_cuMemcpyAtoH_v2(conn_t *conn) {
  CUarray srcArray = nullptr;
  size_t srcOffset = 0;
  size_t byteCount = 0;
  int request_id = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned char> dstHost;

  if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t staging_size =
      std::min(byteCount, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
  if (staging_size != 0) {
    try {
      dstHost.resize(staging_size);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &result, sizeof(result)) < 0 ||
          rpc_write_end(conn) < 0) {
        return -1;
      }
      return 0;
    }
  }

  size_t offset = 0;
  do {
    size_t chunk = std::min(byteCount - offset, staging_size);
    void *chunk_dst = chunk == 0 ? nullptr : dstHost.data();
    result = cuMemcpyAtoH_v2(chunk_dst, srcArray, srcOffset + offset, chunk);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS && chunk != 0 &&
         rpc_write(conn, dstHost.data(), chunk) < 0) ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      return 0;
    }
    offset += chunk;
  } while (offset < byteCount);

  return 0;
}

int handle_manual_cuMemcpyDtoHAsync_v2(conn_t *conn) {
  void *dstHost = nullptr;
  CUdeviceptr srcDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  if (rpc_read_end(conn) < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }

  void *host = nullptr;
  CUresult alloc_result = CUDA_ERROR_INVALID_VALUE;
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto *resources = lupine_get_stream_resources(stream);
    host = lupine_alloc_capture_scratch(resources, byteCount);
    if (host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS) {
        resources->add_dtoh_copy({dstHost, host, byteCount});
      }
      host = nullptr;
    }
  } else {
    alloc_result = cuMemAllocHost(&host, byteCount);
    if (alloc_result != CUDA_SUCCESS) {
      host = byteCount == 0 ? nullptr : malloc(byteCount);
    }
    if (byteCount != 0 && host == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS && byteCount != 0) {
        lupine_pending_dtoh_copy copy{stream, dstHost, host, byteCount,
                                      alloc_result == CUDA_SUCCESS};
        lupine_pending_dtoh_copies().upsert(
            conn,
            [stream, &copy](lupine_pending_dtoh_streams &streams,
                            libcuckoo::UpsertContext) {
              streams[stream].push_back(copy);
            },
            lupine_pending_dtoh_streams{});
        host = nullptr;
      }
    }
  }

  // A fire-and-forget copy drops an immediate validation error, matching launch
  // semantics: an execution failure poisons the context and the driver reports
  // it from the client's next synchronize.
  if (alloc_result == CUDA_SUCCESS && host != nullptr) {
    cuMemFreeHost(host);
  } else if (host != nullptr) {
    free(host);
  }
  return 0;
}

// Resolve the device alias here so the client does not need a second round
// trip for it. A mapped allocation whose alias cannot be resolved still
// succeeds; the 0 tells the client to query it on first use instead.
int handle_manual_cuMemHostAlloc(conn_t *conn) {
  void *pp = nullptr;
  size_t bytesize = 0;
  unsigned int flags = 0;
  if (rpc_read(conn, &pp, sizeof(pp)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(bytesize)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUdeviceptr device_ptr = 0;
  CUresult result = cuMemHostAlloc(&pp, bytesize, flags);
  if (result == CUDA_SUCCESS && (flags & CU_MEMHOSTALLOC_DEVICEMAP) != 0 &&
      cuMemHostGetDevicePointer(&device_ptr, pp, 0) != CUDA_SUCCESS) {
    device_ptr = 0;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pp, sizeof(pp)) < 0 ||
      rpc_write(conn, &device_ptr, sizeof(device_ptr)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemHostGetFlags(conn_t *conn) {
  unsigned int flags = 0;
  void *p = nullptr;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &p, sizeof(p)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuMemHostGetFlags(&flags, p);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuTensorMapEncodeTiled(conn_t *conn) {
  lupine_tensormap_tiled_request request{};
  if (rpc_read(conn, &request, sizeof(request)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUtensorMap tensor_map{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (request.rank >= 1 && request.rank <= LUPINE_TENSOR_MAP_MAX_RANK) {
    result = cuTensorMapEncodeTiled(
        &tensor_map, static_cast<CUtensorMapDataType>(request.data_type),
        request.rank, reinterpret_cast<void *>(request.global_address),
        request.global_dim,
        request.rank > 1 ? request.global_strides : nullptr, request.box_dim,
        request.element_strides,
        static_cast<CUtensorMapInterleave>(request.interleave),
        static_cast<CUtensorMapSwizzle>(request.swizzle),
        static_cast<CUtensorMapL2promotion>(request.l2_promotion),
        static_cast<CUtensorMapFloatOOBfill>(request.oob_fill));
  }

  // Always send the fixed-size descriptor. The client only exposes it when
  // CUDA reports success, while fixed framing keeps failure responses robust.
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &tensor_map, sizeof(tensor_map)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuCtxSynchronize(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  CUresult result = cuCtxSynchronize();
  lupine_finish_stdout_capture(&capture);
  uint32_t copy_count = 0;
  uint64_t stdout_size = 0;
  auto pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_pending_dtoh_copies(&copy_count, conn, pending) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
  return 0;
}

int handle_manual_cuStreamSynchronize(conn_t *conn) {
  CUstream stream = nullptr;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  CUresult result = cuStreamSynchronize(stream);
  lupine_finish_stdout_capture(&capture);
  lupine_graph_resources *resources = nullptr;
  uint32_t copy_count = 0;
  lupine_stream_capture_resource_map().find(stream, resources);
  std::vector<lupine_graph_host_copy> graph_copies =
      resources == nullptr ? std::vector<lupine_graph_host_copy>()
                           : resources->dtoh_copy_snapshot();
  uint32_t graph_copy_count = static_cast<uint32_t>(graph_copies.size());
  bool all_pending_streams = stream == nullptr;
  auto pending =
      lupine_detach_pending_dtoh_copies(conn, stream, all_pending_streams);
  uint32_t pending_copy_count = static_cast<uint32_t>(pending.size());
  copy_count = graph_copy_count + pending_copy_count;
  uint64_t stdout_size = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &copy_count, sizeof(copy_count)) < 0 ||
      std::any_of(
          graph_copies.begin(), graph_copies.end(),
          [&](const lupine_graph_host_copy &copy) {
            return rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) <
                       0 ||
                   rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
                   (copy.bytes != 0 &&
                    rpc_write_payload(conn, copy.server_src, copy.bytes) < 0);
          }) ||
      lupine_write_pending_dtoh_copies(nullptr, conn, pending) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
  return 0;
}

int handle_manual_cuGraphLaunch(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUstream stream = nullptr;
  uint8_t want_response = 0;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &want_response, sizeof(want_response)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphLaunch(exec, stream);
  lupine_graph_resources *resources = nullptr;
  if (result == CUDA_SUCCESS &&
      lupine_graph_exec_resource_map().find(exec, resources)) {
    lupine_stream_capture_resource_map().insert_or_assign(stream, resources);
  }
  if (want_response == 0) {
    return 0;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventSynchronize(conn_t *conn) {
  CUevent event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  CUresult result = cuEventSynchronize(event);
  lupine_finish_stdout_capture(&capture);
  uint32_t copy_count = 0;
  uint64_t stdout_size = 0;
  auto pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_pending_dtoh_copies(&copy_count, conn, pending) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
  return 0;
}

int handle_manual_cuOccupancyMaxPotentialBlockSize(conn_t *conn,
                                                   bool with_flags) {
  CUfunction func = nullptr;
  size_t dynamicSMemSize = 0;
  int blockSizeLimit = 0;
  unsigned int flags = 0;
  int request_id;
  int minGridSize = 0;
  int blockSize = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &func, sizeof(func)) < 0 ||
      rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
      rpc_read(conn, &blockSizeLimit, sizeof(blockSizeLimit)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (with_flags) {
    result = cuOccupancyMaxPotentialBlockSizeWithFlags(
        &minGridSize, &blockSize, func, nullptr, dynamicSMemSize,
        blockSizeLimit, flags);
  } else {
    result = cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func,
                                              nullptr, dynamicSMemSize,
                                              blockSizeLimit);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &minGridSize, sizeof(minGridSize)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(blockSize)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// The generated marshaller cannot receive a string of unknown length, and the
// driver hands back a static pointer rather than filling a caller buffer, so
// these two forward the answer as an explicit length plus bytes.
static int lupine_handle_error_string(conn_t *conn,
                                      CUresult (*lookup)(CUresult,
                                                         const char **)) {
  CUresult error;
  if (rpc_read(conn, &error, sizeof(error)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  const char *text = nullptr;
  CUresult result = lookup(error, &text);
  uint32_t length = (result == CUDA_SUCCESS && text != nullptr)
                        ? static_cast<uint32_t>(strlen(text))
                        : 0;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_write(conn, text, length) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGetErrorName(conn_t *conn) {
  return lupine_handle_error_string(conn, cuGetErrorName);
}

int handle_manual_cuGetErrorString(conn_t *conn) {
  return lupine_handle_error_string(conn, cuGetErrorString);
}
