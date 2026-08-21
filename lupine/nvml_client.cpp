#include <arpa/inet.h>
#include <cuda.h>
#include <dlfcn.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <nvml.h>
#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "codegen/gen_api.h"
#include "config.h"
#include "lupine_log.h"
#include "rpc.h"

// CUDA <= 12.6 ships NVML API 12, which does not define the versioned
// temperature struct. Keep the wrapper ABI-compatible with newer nvidia-smi.
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) ||                        \
    (defined(NVML_API_VERSION) && NVML_API_VERSION >= 13)
using lupine_nvmlTemperature_t = nvmlTemperature_t;
#else
typedef struct {
  unsigned int version;
  nvmlTemperatureSensors_t sensorType;
  int temperature;
} lupine_nvmlTemperature_t;
#endif

namespace {

constexpr const char *DEFAULT_PORT = "14833";

pthread_mutex_t conn_mutex = PTHREAD_MUTEX_INITIALIZER;
conn_t conns[16] = {};
int nconns = 0;
bool connected = false;

struct lupine_nvml_remote_device {
  bool local = false;
  nvmlDevice_t local_device = nullptr;
  unsigned int conn_index = 0;
  unsigned int remote_index = 0;
  nvmlDevice_t remote_device = nullptr;
  std::string server_label;
};

std::vector<lupine_nvml_remote_device> devices;
std::vector<std::string> conn_labels;
bool devices_ready = false;

void *local_nvml_handle() {
  static void *handle = []() -> void * {
    const char *disabled = getenv("LUPINE_DISABLE_LOCAL");
    if (disabled != nullptr && strcmp(disabled, "0") != 0 &&
        strcasecmp(disabled, "false") != 0 &&
        strcasecmp(disabled, "no") != 0) {
      return nullptr;
    }
    const char *override_path = getenv("LUPINE_REAL_NVML");
    const char *paths[] = {
        override_path,
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        nullptr,
    };
    for (const char *path : paths) {
      if (path == nullptr || path[0] == '\0') {
        continue;
      }
      void *candidate = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (candidate != nullptr) {
        return candidate;
      }
    }
    return nullptr;
  }();
  return handle;
}

template <typename Fn> Fn local_nvml_function(const char *name) {
  void *handle = local_nvml_handle();
  return handle == nullptr ? nullptr : reinterpret_cast<Fn>(dlsym(handle, name));
}

// Real NVML reference counts init/shutdown; this shim connects lazily, so
// without a counter it could never report UNINITIALIZED.
std::atomic<int> init_refcount{0};

thread_local bool init_gate_bypass = false;

bool nvml_initialized() {
  return init_gate_bypass || init_refcount.load(std::memory_order_acquire) > 0;
}

struct uninitialized_entry_point {
  uninitialized_entry_point() { init_gate_bypass = true; }
  ~uninitialized_entry_point() { init_gate_bypass = false; }
};

nvmlReturn_t rpc_error() {
  return nvml_initialized() ? NVML_ERROR_UNKNOWN : NVML_ERROR_UNINITIALIZED;
}

void *rpc_client_dispatch_thread(void *p) {
  conn_t *connection = static_cast<conn_t *>(p);
  while (!connection->closed) {
    int op = rpc_dispatch(connection, 1);
    if (op < 0 || connection->closed) {
      break;
    }
    if (rpc_read_end(connection) < 0) {
      break;
    }
  }
  return nullptr;
}

int open_connection() {
  if (pthread_mutex_lock(&conn_mutex) < 0) {
    return -1;
  }
  if (connected) {
    pthread_mutex_unlock(&conn_mutex);
    return 0;
  }

  std::string server_configuration = lupine_server_configuration();
  if (server_configuration.empty()) {
    LUPINE_LOG_ERROR(
        "No remote endpoints configured (set LUPINE_SERVER or create "
        "/etc/rgpu/endpoints)");
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  char *servers = strdup(server_configuration.c_str());
  if (servers == nullptr) {
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  char *cursor = servers;
  char *token = nullptr;
  while ((token = strsep(&cursor, ",")) != nullptr) {
    if (token[0] == '\0') {
      continue;
    }

    bool tls = false;
    if (strncmp(token, "https://", 8) == 0) {
      tls = true;
      token += 8;
    } else if (strncmp(token, "http://", 7) == 0) {
      token += 7;
    } else if (strstr(token, "://") != nullptr ||
               strncmp(token, "http:", 5) == 0 ||
               strncmp(token, "https:", 6) == 0) {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER URL scheme: " << token);
      continue;
    }

    char *host = token;
    char *port = const_cast<char *>(tls ? "443" : DEFAULT_PORT);
    char *colon = strchr(token, ':');
    if (colon != nullptr) {
      *colon = '\0';
      port = colon + 1;
    }
    if (host[0] == '\0' || port[0] == '\0') {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER endpoint");
      continue;
    }

    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
      continue;
    }

    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd >= 0) {
      lupine_socket_apply_transport_options(sockfd);
      if (connect(sockfd, res->ai_addr, res->ai_addrlen) == 0) {
        if (nconns >= static_cast<int>(sizeof(conns) / sizeof(conns[0]))) {
          close(sockfd);
          freeaddrinfo(res);
          break;
        }
        std::string server_label(host);
        if (strcmp(port, DEFAULT_PORT) != 0) {
          server_label += ":";
          server_label += port;
        }

        conn_t *c = &conns[nconns];
        *c = {};
        c->connfd = sockfd;
        c->request_id = 0;
        c->local_request_parity = c->request_id & 1;
        if (tls) {
#ifdef LUPINE_TLS_OPENSSL
          static SSL_CTX *tls_ctx = []() {
            SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
            if (ctx != nullptr) {
              SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
              SSL_CTX_set_default_verify_paths(ctx);
              SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
            }
            return ctx;
          }();
          SSL *ssl = tls_ctx != nullptr ? SSL_new(tls_ctx) : nullptr;
          if (ssl == nullptr || SSL_set_tlsext_host_name(ssl, host) != 1 ||
              SSL_set1_host(ssl, host) != 1 || SSL_set_fd(ssl, sockfd) != 1 ||
              SSL_connect(ssl) != 1) {
            if (ssl != nullptr) {
              SSL_free(ssl);
            }
            LUPINE_LOG_ERROR("TLS handshake with " << host << " failed");
            close(sockfd);
            freeaddrinfo(res);
            continue;
          }
          c->tls_session = ssl;
#else
          LUPINE_LOG_ERROR("LUPINE_SERVER entry "
                           << host << ":" << port
                           << " uses https:// but this client was built "
                              "without TLS support");
          close(sockfd);
          freeaddrinfo(res);
          continue;
#endif
        }
        if (pthread_mutex_init(&c->read_mutex, nullptr) < 0 ||
            pthread_mutex_init(&c->write_mutex, nullptr) < 0 ||
            pthread_mutex_init(&c->call_mutex, nullptr) < 0 ||
            pthread_cond_init(&c->read_cond, nullptr) < 0 ||
            rpc_http2_client_init(c) < 0 ||
            pthread_create(&c->read_thread, nullptr, rpc_client_dispatch_thread,
                           c) < 0) {
#ifdef LUPINE_TLS_OPENSSL
          if (c->tls_session != nullptr) {
            SSL_free(static_cast<SSL *>(c->tls_session));
            c->tls_session = nullptr;
          }
#endif
          close(sockfd);
          freeaddrinfo(res);
          continue;
        }
        conn_labels.push_back(server_label);
        ++nconns;
        freeaddrinfo(res);
        continue;
      }
      close(sockfd);
    }
    freeaddrinfo(res);
  }
  free(servers);

  if (nconns == 0) {
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  connected = true;
  pthread_mutex_unlock(&conn_mutex);
  return 0;
}

conn_t *connection(unsigned int index = 0) {
  if (!nvml_initialized() || open_connection() < 0) {
    return nullptr;
  }
  if (index >= static_cast<unsigned int>(nconns)) {
    return nullptr;
  }
  return &conns[index];
}

void close_connections() {
  if (pthread_mutex_lock(&conn_mutex) != 0) {
    return;
  }
  int count = nconns;
  for (int i = 0; i < count; ++i) {
    conn_t *c = &conns[i];
    if (!c->closed) {
      c->closed = 1;
      shutdown(c->connfd, SHUT_RDWR);
      close(c->connfd);
    }
    pthread_mutex_lock(&c->read_mutex);
    pthread_cond_broadcast(&c->read_cond);
    pthread_mutex_unlock(&c->read_mutex);
  }
  pthread_mutex_unlock(&conn_mutex);

  for (int i = 0; i < count; ++i) {
    conn_t *c = &conns[i];
    if (c->read_thread != 0) {
      pthread_join(c->read_thread, nullptr);
      c->read_thread = 0;
    }
    if (c->rpc_thread != 0) {
      pthread_join(c->rpc_thread, nullptr);
      c->rpc_thread = 0;
    }
#ifdef LUPINE_TLS_OPENSSL
    if (c->tls_session != nullptr) {
      SSL_free(static_cast<SSL *>(c->tls_session));
      c->tls_session = nullptr;
    }
#endif
    rpc_conn_destroy(c);
  }

  if (pthread_mutex_lock(&conn_mutex) == 0) {
    nconns = 0;
    connected = false;
    devices_ready = false;
    devices.clear();
    conn_labels.clear();
    pthread_mutex_unlock(&conn_mutex);
  }
}

nvmlReturn_t ensure_devices();

lupine_nvml_remote_device *mapped_device(nvmlDevice_t device) {
  if (device == nullptr || devices.empty()) {
    return nullptr;
  }
  uintptr_t begin = reinterpret_cast<uintptr_t>(devices.data());
  uintptr_t end = begin + devices.size() * sizeof(lupine_nvml_remote_device);
  uintptr_t value = reinterpret_cast<uintptr_t>(device);
  if (value < begin || value >= end ||
      (value - begin) % sizeof(lupine_nvml_remote_device) != 0) {
    return nullptr;
  }
  return &devices[(value - begin) / sizeof(lupine_nvml_remote_device)];
}

bool translate_local_device(nvmlDevice_t *device) {
  if (device == nullptr || ensure_devices() != NVML_SUCCESS) {
    return false;
  }
  auto *mapped = mapped_device(*device);
  if (mapped == nullptr || !mapped->local) {
    return false;
  }
  *device = mapped->local_device;
  return true;
}

nvmlReturn_t map_local_device(nvmlDevice_t local, nvmlDevice_t *device) {
  if (device == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  auto mapped = std::find_if(devices.begin(), devices.end(),
                             [&](const auto &candidate) {
                               return candidate.local &&
                                      candidate.local_device == local;
                             });
  if (mapped == devices.end()) {
    return NVML_ERROR_NOT_FOUND;
  }
  *device = reinterpret_cast<nvmlDevice_t>(&*mapped);
  return NVML_SUCCESS;
}

void virtualize_remote_pci_identity(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  if (pci == nullptr) {
    return;
  }
  auto *mapped = mapped_device(device);
  if (mapped == nullptr || mapped->local) {
    return;
  }
  const unsigned int virtual_domain = 0x1000u + mapped->conn_index;
  const char *separator = strchr(pci->busId, ':');
  char physical_suffix[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
  if (separator != nullptr) {
    snprintf(physical_suffix, sizeof(physical_suffix), "%s", separator);
  } else {
    snprintf(physical_suffix, sizeof(physical_suffix), ":%02x:%02x.0",
             pci->bus, pci->device);
  }
  pci->domain = virtual_domain;
  snprintf(pci->busId, sizeof(pci->busId), "%08x%s", virtual_domain,
           physical_suffix);
}

nvmlReturn_t call_no_args_on(conn_t *c, int op) {
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_uint_out_on(conn_t *c, int op, unsigned int *value) {
  nvmlReturn_t result = rpc_error();
  unsigned int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

nvmlReturn_t call_device_from_index_on(conn_t *c, int op, unsigned int index,
                                       nvmlDevice_t *device) {
  nvmlReturn_t result = rpc_error();
  nvmlDevice_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &index, sizeof(index)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

nvmlReturn_t ensure_devices() {
  if (!nvml_initialized() || open_connection() < 0) {
    return rpc_error();
  }
  if (devices_ready) {
    return NVML_SUCCESS;
  }

  devices.clear();
  using local_count_fn = nvmlReturn_t (*)(unsigned int *);
  using local_get_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
  auto local_count =
      local_nvml_function<local_count_fn>("nvmlDeviceGetCount_v2");
  auto local_get = local_nvml_function<local_get_fn>(
      "nvmlDeviceGetHandleByIndex_v2");
  if (local_count != nullptr && local_get != nullptr) {
    unsigned int count = 0;
    if (local_count(&count) == NVML_SUCCESS) {
      for (unsigned int ordinal = 0; ordinal < count; ++ordinal) {
        nvmlDevice_t handle = nullptr;
        if (local_get(ordinal, &handle) == NVML_SUCCESS) {
          lupine_nvml_remote_device entry;
          entry.local = true;
          entry.local_device = handle;
          devices.push_back(std::move(entry));
        }
      }
    }
  }
  for (int i = 0; i < nconns; ++i) {
    unsigned int count = 0;
    nvmlReturn_t result =
        call_uint_out_on(&conns[i], RPC_nvmlDeviceGetCount_v2, &count);
    if (result != NVML_SUCCESS) {
      devices.clear();
      return result;
    }
    for (unsigned int ordinal = 0; ordinal < count; ++ordinal) {
      nvmlDevice_t remote = nullptr;
      result = call_device_from_index_on(
          &conns[i], RPC_nvmlDeviceGetHandleByIndex_v2, ordinal, &remote);
      if (result != NVML_SUCCESS) {
        devices.clear();
        return result;
      }
      const std::string &server_label =
          i < static_cast<int>(conn_labels.size()) ? conn_labels[i] : "";
      lupine_nvml_remote_device entry;
      entry.conn_index = static_cast<unsigned int>(i);
      entry.remote_index = ordinal;
      entry.remote_device = remote;
      entry.server_label = server_label;
      devices.push_back(std::move(entry));
    }
  }
  devices_ready = true;
  return NVML_SUCCESS;
}

template <typename Lookup>
nvmlReturn_t lookup_device_on_all_connections(nvmlDevice_t *device,
                                              Lookup &&lookup) {
  if (device == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }

  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }

  nvmlReturn_t first_error = NVML_ERROR_NOT_FOUND;
  for (int i = 0; i < nconns; ++i) {
    nvmlDevice_t remote = nullptr;
    result = lookup(&conns[i], &remote);
    if (result != NVML_SUCCESS) {
      if (first_error == NVML_ERROR_NOT_FOUND &&
          result != NVML_ERROR_NOT_FOUND) {
        first_error = result;
      }
      continue;
    }

    auto mapped = std::find_if(
        devices.begin(), devices.end(), [&](const auto &candidate) {
          return candidate.conn_index == static_cast<unsigned int>(i) &&
                 candidate.remote_device == remote;
        });
    if (mapped == devices.end()) {
      // The server returned a handle that was not part of the device table
      // built from nvmlDeviceGetCount/GetHandleByIndex. Never expose that raw
      // process-local pointer to the caller.
      if (first_error == NVML_ERROR_NOT_FOUND) {
        first_error = rpc_error();
      }
      continue;
    }

    *device = reinterpret_cast<nvmlDevice_t>(&*mapped);
    return NVML_SUCCESS;
  }
  return first_error;
}

conn_t *connection_for_device(nvmlDevice_t *device) {
  if (device == nullptr || ensure_devices() != NVML_SUCCESS) {
    return nullptr;
  }
  if (devices.empty()) {
    return nullptr;
  }
  auto *mapped = mapped_device(*device);
  if (mapped == nullptr) {
    return connection();
  }
  *device = mapped->remote_device;
  return connection(mapped->conn_index);
}

nvmlReturn_t call_no_args(int op) { return call_no_args_on(connection(), op); }

nvmlReturn_t call_device_string(int op, nvmlDevice_t device, char *value,
                                unsigned int length) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &length, sizeof(length)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      (length != 0 && rpc_read(c, value, length) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_processes(int op, nvmlDevice_t device,
                            unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  nvmlDevice_t local_device = device;
  if (translate_local_device(&local_device)) {
    const char *name = nullptr;
    switch (op) {
    case RPC_nvmlDeviceGetComputeRunningProcesses:
      name = "nvmlDeviceGetComputeRunningProcesses";
      break;
    case RPC_nvmlDeviceGetComputeRunningProcesses_v2:
      name = "nvmlDeviceGetComputeRunningProcesses_v2";
      break;
    case RPC_nvmlDeviceGetGraphicsRunningProcesses:
      name = "nvmlDeviceGetGraphicsRunningProcesses";
      break;
    case RPC_nvmlDeviceGetGraphicsRunningProcesses_v2:
      name = "nvmlDeviceGetGraphicsRunningProcesses_v2";
      break;
    case RPC_nvmlDeviceGetMPSComputeRunningProcesses:
      name = "nvmlDeviceGetMPSComputeRunningProcesses";
      break;
    case RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2:
      name = "nvmlDeviceGetMPSComputeRunningProcesses_v2";
      break;
    default:
      return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    using local_fn_t = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *,
                                        nvmlProcessInfo_t *);
    auto fn = local_nvml_function<local_fn_t>(name);
    return fn == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                         : fn(local_device, infoCount, infos);
  }
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  unsigned int requested_count = infoCount == nullptr ? 0 : *infoCount;
  int has_infos = infos == nullptr ? 0 : 1;
  unsigned int returned_count = 0;
  unsigned int copied_count = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &requested_count, sizeof(requested_count)) < 0 ||
      rpc_write(c, &has_infos, sizeof(has_infos)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &returned_count, sizeof(returned_count)) < 0 ||
      rpc_read(c, &copied_count, sizeof(copied_count)) < 0 ||
      (copied_count != 0 &&
       rpc_read(c, infos, copied_count * sizeof(infos[0])) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (infoCount != nullptr) {
    *infoCount = returned_count;
  }
  return result;
}

nvmlReturn_t call_event_set_create(nvmlEventSet_t *set) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlEventSet_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetCreate) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (set != nullptr) {
    *set = temp;
  }
  return result;
}

nvmlReturn_t call_event_set_free(nvmlEventSet_t set) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetFree) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_event_set_wait(nvmlEventSet_t set, nvmlEventData_t *data,
                                 unsigned int timeoutms) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlEventData_t temp = {};
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetWait_v2) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 ||
      rpc_write(c, &timeoutms, sizeof(timeoutms)) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (data != nullptr) {
    *data = temp;
  }
  return result;
}

nvmlReturn_t call_device_register_events(nvmlDevice_t device,
                                         unsigned long long eventTypes,
                                         nvmlEventSet_t set) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  if (c == nullptr ||
      rpc_write_start_request(c, RPC_nvmlDeviceRegisterEvents) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &eventTypes, sizeof(eventTypes)) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

} // namespace

#ifdef nvmlInit
#undef nvmlInit
#endif
#ifdef nvmlDeviceGetCount
#undef nvmlDeviceGetCount
#endif
#ifdef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByIndex
#endif
#ifdef nvmlDeviceGetHandleByPciBusId
#undef nvmlDeviceGetHandleByPciBusId
#endif
#ifdef nvmlDeviceGetPciInfo
#undef nvmlDeviceGetPciInfo
#endif
#ifdef nvmlDeviceGetComputeRunningProcesses
#undef nvmlDeviceGetComputeRunningProcesses
#endif
#ifdef nvmlDeviceGetGraphicsRunningProcesses
#undef nvmlDeviceGetGraphicsRunningProcesses
#endif
#ifdef nvmlDeviceGetMPSComputeRunningProcesses
#undef nvmlDeviceGetMPSComputeRunningProcesses
#endif
#ifdef nvmlEventSetWait
#undef nvmlEventSetWait
#endif

// Real NVML answers this one without nvmlInit, so rename the generated entry
// point and re-export it below with the gate lifted.
#define nvmlSystemGetNVMLVersion lupine_nvmlSystemGetNVMLVersion_gated

#include "codegen/gen_nvml_client.inc"

#undef nvmlSystemGetNVMLVersion

extern "C" nvmlReturn_t nvmlSystemGetNVMLVersion(char *version,
                                                 unsigned int length) {
  uninitialized_entry_point guard;
  return lupine_nvmlSystemGetNVMLVersion_gated(version, length);
}

extern "C" nvmlReturn_t nvmlInit_v2(void) {
  init_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (open_connection() < 0) {
    init_refcount.fetch_sub(1, std::memory_order_acq_rel);
    return NVML_ERROR_UNKNOWN;
  }
  nvmlReturn_t first_error = NVML_SUCCESS;
  using init_fn = nvmlReturn_t (*)();
  auto local_init = local_nvml_function<init_fn>("nvmlInit_v2");
  if (local_init != nullptr) {
    nvmlReturn_t result = local_init();
    if (result != NVML_SUCCESS) {
      first_error = result;
    }
  }
  for (int i = 0; i < nconns; ++i) {
    nvmlReturn_t result = call_no_args_on(&conns[i], RPC_nvmlInit_v2);
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
  init_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (open_connection() < 0) {
    init_refcount.fetch_sub(1, std::memory_order_acq_rel);
    return NVML_ERROR_UNKNOWN;
  }
  nvmlReturn_t first_error = NVML_SUCCESS;
  using init_flags_fn = nvmlReturn_t (*)(unsigned int);
  auto local_init =
      local_nvml_function<init_flags_fn>("nvmlInitWithFlags");
  if (local_init != nullptr) {
    nvmlReturn_t result = local_init(flags);
    if (result != NVML_SUCCESS) {
      first_error = result;
    }
  }
  for (int i = 0; i < nconns; ++i) {
    conn_t *c = &conns[i];
    nvmlReturn_t result = rpc_error();
    if (rpc_write_start_request(c, RPC_nvmlInitWithFlags) < 0 ||
        rpc_write(c, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(c) < 0 ||
        rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
      result = rpc_error();
    }
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" nvmlReturn_t nvmlShutdown(void) {
  // Claim a reference up front so an unmatched shutdown fails and concurrent
  // callers cannot tear the connections down twice.
  int previous = init_refcount.load(std::memory_order_acquire);
  do {
    if (previous <= 0) {
      return NVML_ERROR_UNINITIALIZED;
    }
  } while (!init_refcount.compare_exchange_weak(previous, previous - 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire));
  bool last = previous == 1;

  if (pthread_mutex_lock(&conn_mutex) != 0) {
    return NVML_ERROR_UNKNOWN;
  }
  if (!connected) {
    pthread_mutex_unlock(&conn_mutex);
    return NVML_SUCCESS;
  }
  int count = nconns;
  pthread_mutex_unlock(&conn_mutex);

  // The server refcounts too, so forward every shutdown, not just the last.
  nvmlReturn_t first_error = NVML_SUCCESS;
  using shutdown_fn = nvmlReturn_t (*)();
  auto local_shutdown = local_nvml_function<shutdown_fn>("nvmlShutdown");
  if (local_shutdown != nullptr) {
    nvmlReturn_t result = local_shutdown();
    if (result != NVML_SUCCESS) {
      first_error = result;
    }
  }
  for (int i = 0; i < count; ++i) {
    nvmlReturn_t result = call_no_args_on(&conns[i], RPC_nvmlShutdown);
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  if (last) {
    close_connections();
  }
  return first_error;
}

extern "C" const char *nvmlErrorString(nvmlReturn_t result) {
  switch (result) {
  case NVML_SUCCESS:
    return "Success";
  case NVML_ERROR_UNINITIALIZED:
    return "Uninitialized";
  case NVML_ERROR_INVALID_ARGUMENT:
    return "Invalid Argument";
  case NVML_ERROR_NOT_SUPPORTED:
    return "Not Supported";
  case NVML_ERROR_NO_PERMISSION:
    return "Insufficient Permissions";
  case NVML_ERROR_ALREADY_INITIALIZED:
    return "Already Initialized";
  case NVML_ERROR_NOT_FOUND:
    return "Not Found";
  case NVML_ERROR_INSUFFICIENT_SIZE:
    return "Insufficient Size";
  case NVML_ERROR_INSUFFICIENT_POWER:
    return "Insufficient External Power";
  case NVML_ERROR_DRIVER_NOT_LOADED:
    return "Driver Not Loaded";
  case NVML_ERROR_TIMEOUT:
    return "Timeout";
  case NVML_ERROR_IRQ_ISSUE:
    return "IRQ Issue";
  case NVML_ERROR_LIBRARY_NOT_FOUND:
    return "NVML Shared Library Not Found";
  case NVML_ERROR_FUNCTION_NOT_FOUND:
    return "Function Not Found";
  case NVML_ERROR_CORRUPTED_INFOROM:
    return "Corrupted InfoROM";
  case NVML_ERROR_GPU_IS_LOST:
    return "GPU is lost";
  case NVML_ERROR_RESET_REQUIRED:
    return "GPU requires reset";
  case NVML_ERROR_OPERATING_SYSTEM:
    return "The operating system has blocked the request";
  case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
    return "RM has detected an NVML/RM version mismatch";
  case NVML_ERROR_IN_USE:
    return "GPU is currently in use";
  case NVML_ERROR_MEMORY:
    return "Insufficient memory";
  case NVML_ERROR_NO_DATA:
    return "No data";
  case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED:
    return "VGPU ECC not supported";
  case NVML_ERROR_INSUFFICIENT_RESOURCES:
    return "Insufficient resources";
  default:
    return "Unknown Error";
  }
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(
    nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(
    nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(
    nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);

static void install_process_export_entries(std::uintptr_t *table,
                                           std::size_t words) {
  // R610's 0x948-byte table uses these three slots for the compute, graphics,
  // and MPS running-process variants. Redirect them through the public shim so
  // synthetic remote handles are translated instead of reaching physical NVML.
  if (words <= 215 || table[0] != 0x948) {
    return;
  }
  table[213] = reinterpret_cast<std::uintptr_t>(
      nvmlDeviceGetComputeRunningProcesses_v2);
  table[214] = reinterpret_cast<std::uintptr_t>(
      nvmlDeviceGetGraphicsRunningProcesses_v2);
  table[215] = reinterpret_cast<std::uintptr_t>(
      nvmlDeviceGetMPSComputeRunningProcesses_v2);
}

extern "C" nvmlReturn_t nvmlInternalGetExportTable(const void **ppExportTable,
                                                   const void *exportTableId) {
  if (ppExportTable == nullptr || exportTableId == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  // nvidia-smi uses this undocumented table for process-display helpers. On a
  // mixed host those helpers are already available in the physical NVML and
  // are process-local utilities, so preserve the driver-matched table instead
  // of pretending that a blank table implements it.
  using local_fn_t = nvmlReturn_t (*)(const void **, const void *);
  auto local_fn =
      local_nvml_function<local_fn_t>("nvmlInternalGetExportTable");
  if (local_fn != nullptr) {
    nvmlReturn_t result = local_fn(ppExportTable, exportTableId);
    if (result == NVML_SUCCESS && *ppExportTable != nullptr) {
      const auto *physical =
          static_cast<const std::uintptr_t *>(*ppExportTable);
      static std::uintptr_t mixed_table[0x948 / sizeof(std::uintptr_t)];
      if (physical[0] == sizeof(mixed_table)) {
        std::memcpy(mixed_table, physical, sizeof(mixed_table));
        install_process_export_entries(
            mixed_table, sizeof(mixed_table) / sizeof(mixed_table[0]));
        *ppExportTable = mixed_table;
      }
    }
    return result;
  }
  // The one table ID nvidia-smi asks for; this shim proxies none of the
  // internal entry points it holds in remote-only mode.
  static const unsigned char known_export_table_id[16] = {
      0xc4, 0xfe, 0x3e, 0x6c, 0xc9, 0x8f, 0x6c, 0x4e,
      0xa3, 0x27, 0xee, 0x69, 0x6e, 0x12, 0xf7, 0xc4};
  if (memcmp(exportTableId, known_export_table_id,
             sizeof(known_export_table_id)) != 0) {
    *ppExportTable = nullptr;
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  // The first word is the byte size of this driver-generation-specific table.
  // Returning an all-zero blob makes nvidia-smi interpret the table as an
  // uninitialized provider instead of falling back to public NVML APIs.
  static std::uintptr_t empty_table[0x948 / sizeof(std::uintptr_t)] = {0x948};
  install_process_export_entries(
      empty_table, sizeof(empty_table) / sizeof(empty_table[0]));
  *ppExportTable = empty_table;
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
  return call_event_set_create(set);
}

extern "C" nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
  return call_event_set_free(set);
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set,
                                            nvmlEventData_t *data,
                                            unsigned int timeoutms) {
  return call_event_set_wait(set, data, timeoutms);
}

extern "C" nvmlReturn_t nvmlEventSetWait(nvmlEventSet_t set,
                                         nvmlEventData_t *data,
                                         unsigned int timeoutms) {
  return nvmlEventSetWait_v2(set, data, timeoutms);
}

extern "C" nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device,
                                                 unsigned long long eventTypes,
                                                 nvmlEventSet_t set) {
  return call_device_register_events(device, eventTypes, set);
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (deviceCount == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *deviceCount = static_cast<unsigned int>(devices.size());
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
  return nvmlDeviceGetCount_v2(deviceCount);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                                      nvmlDevice_t *device) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (device == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  if (index >= devices.size()) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *device = reinterpret_cast<nvmlDevice_t>(&devices[index]);
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index,
                                                   nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                                      nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                                          unsigned int length) {
  nvmlDevice_t local_device = device;
  if (translate_local_device(&local_device)) {
    using local_fn_t = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    auto fn = local_nvml_function<local_fn_t>("nvmlDeviceGetName");
    return fn == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                         : fn(local_device, name, length);
  }
  nvmlReturn_t result =
      call_device_string(RPC_nvmlDeviceGetName, device, name, length);
  if (result != NVML_SUCCESS || name == nullptr || length == 0) {
    return result;
  }

  auto *mapped = mapped_device(device);
  if (mapped == nullptr || mapped->server_label.empty()) {
    return result;
  }

  size_t used = strnlen(name, length);
  if (used >= length) {
    name[length - 1] = '\0';
    used = length - 1;
  }
  if (used + 1 < length) {
    snprintf(name + used, length - used, " (via lupine %s)",
             mapped->server_label.c_str());
  }
  return result;
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device,
                                           unsigned int *index) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (devices.empty() || index == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  // Deliberate divergence: the fan-out index is fabricated client-side, so it
  // can only be recovered from a handle this shim minted.
  auto *mapped = mapped_device(device);
  if (mapped == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *index = static_cast<unsigned int>(mapped - devices.data());
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetCudaComputeCapability(
    nvmlDevice_t device, int *major, int *minor) {
  if (major == nullptr || minor == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  auto *mapped = mapped_device(device);
  if (mapped == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }

  // Some toolkit NVML headers lag the driver and omit this entry point from
  // LUPINE's generated RPC surface. Query the same virtual ordinal through
  // the already-interposed CUDA Driver API instead of fabricating an
  // architecture. This remains correct for heterogeneous and multi-host
  // pools because both shims use the same stable host/ordinal ordering.
  using init_fn = CUresult (*)(unsigned int);
  using get_fn = CUresult (*)(CUdevice *, int);
  using attr_fn = CUresult (*)(int *, CUdevice_attribute, CUdevice);
  void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  auto init = cuda == nullptr
                  ? nullptr
                  : reinterpret_cast<init_fn>(dlsym(cuda, "cuInit"));
  auto get = cuda == nullptr
                 ? nullptr
                 : reinterpret_cast<get_fn>(dlsym(cuda, "cuDeviceGet"));
  auto attr = cuda == nullptr
                  ? nullptr
                  : reinterpret_cast<attr_fn>(dlsym(cuda,
                                                    "cuDeviceGetAttribute"));
  unsigned int ordinal = static_cast<unsigned int>(mapped - devices.data());
  CUdevice cuda_device = 0;
  if (init == nullptr || get == nullptr || attr == nullptr ||
      init(0) != CUDA_SUCCESS ||
      get(&cuda_device, static_cast<int>(ordinal)) != CUDA_SUCCESS ||
      attr(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_device) !=
          CUDA_SUCCESS ||
      attr(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_device) !=
          CUDA_SUCCESS) {
    return NVML_ERROR_UNKNOWN;
  }
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetP2PStatus(
    nvmlDevice_t device1, nvmlDevice_t device2,
    nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus) {
  if (p2pStatus == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  auto *first = mapped_device(device1);
  auto *second = mapped_device(device2);
  if (first == nullptr || second == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  if (p2pIndex != NVML_P2P_CAPS_INDEX_READ &&
      p2pIndex != NVML_P2P_CAPS_INDEX_WRITE) {
    return NVML_ERROR_NOT_SUPPORTED;
  }
  // A device always has access to itself. Cross-server peer access is not
  // physically possible in a strict GPU-only RPC pool, so report that
  // honestly; same-server multi-GPU topology will get a dedicated remote
  // NVML query when a multi-GPU validation host is available.
  *p2pStatus = first == second ? NVML_P2P_STATUS_OK
                               : NVML_P2P_STATUS_NOT_SUPPORTED;
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device,
                                                nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device,
                                             nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000) ||                        \
    (defined(NVML_API_VERSION) && NVML_API_VERSION >= 13)
extern "C" nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device,
                                                nvmlPciInfoExt_t *pci) {
  if (pci == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  nvmlDevice_t local_device = device;
  if (translate_local_device(&local_device)) {
    using local_fn_t = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfoExt_t *);
    auto fn = local_nvml_function<local_fn_t>("nvmlDeviceGetPciInfoExt");
    return fn == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                         : fn(local_device, pci);
  }

  // Older server-side NVML headers expose v3 but not the newer Ext wrapper.
  // Ext is a strict superset for the fields nvidia-smi and topology discovery
  // consume, so translate the stable PCI identity rather than claiming the
  // symbol is unavailable.
  nvmlPciInfo_t legacy = {};
  nvmlReturn_t result = nvmlDeviceGetPciInfo_v3(device, &legacy);
  if (result != NVML_SUCCESS) {
    return result;
  }
  unsigned int version = pci->version;
  memset(pci, 0, sizeof(*pci));
  pci->version = version;
  pci->domain = legacy.domain;
  pci->bus = legacy.bus;
  pci->device = legacy.device;
  pci->pciDeviceId = legacy.pciDeviceId;
  pci->pciSubSystemId = legacy.pciSubSystemId;
  snprintf(pci->busId, sizeof(pci->busId), "%s", legacy.busId);
  return NVML_SUCCESS;
}
#endif

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetComputeRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetComputeRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetGraphicsRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetGraphicsRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return nvmlDeviceGetGraphicsRunningProcesses_v2(device, infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetMPSComputeRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return nvmlDeviceGetMPSComputeRunningProcesses_v2(device, infoCount, infos);
}
