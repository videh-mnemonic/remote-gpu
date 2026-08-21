#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

typedef void *ncclComm_t;
typedef void *cudaStream_t;
typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef int ncclScalarResidence_t;
typedef int ncclCommMemStat_t;
typedef void *ncclWindow_t;
typedef struct ncclSimInfo_v22900 ncclSimInfo_t;
typedef struct { char internal[128]; } ncclUniqueId;
typedef struct ncclConfig_v22800 ncclConfig_t;

enum {
  LUPINE_NCCL_GET_UNIQUE_ID = 12,
  LUPINE_NCCL_INIT_RANK = 1,
  LUPINE_NCCL_GROUP_START = 2,
  LUPINE_NCCL_GROUP_END = 3,
  LUPINE_NCCL_ALL_REDUCE = 4,
  LUPINE_NCCL_ALL_GATHER = 5,
  LUPINE_NCCL_REDUCE_SCATTER = 6,
  LUPINE_NCCL_BROADCAST = 7,
  LUPINE_NCCL_ASYNC_ERROR = 8,
  LUPINE_NCCL_FINALIZE = 9,
  LUPINE_NCCL_DESTROY = 10,
  LUPINE_NCCL_ABORT = 11,
  LUPINE_NCCL_REDUCE = 13,
  LUPINE_NCCL_BCAST = 14,
  LUPINE_NCCL_ALL_TO_ALL = 15,
  LUPINE_NCCL_GATHER = 16,
  LUPINE_NCCL_SCATTER = 17,
  LUPINE_NCCL_SEND = 18,
  LUPINE_NCCL_RECV = 19,
  LUPINE_NCCL_COMM_COUNT = 20,
  LUPINE_NCCL_COMM_CUDA_DEVICE = 21,
  LUPINE_NCCL_COMM_USER_RANK = 22,
  LUPINE_NCCL_COMM_REGISTER = 23,
  LUPINE_NCCL_COMM_DEREGISTER = 24,
  LUPINE_NCCL_COMM_SPLIT = 25,
  LUPINE_NCCL_LAST_ERROR = 26,
  LUPINE_NCCL_MEM_ALLOC = 27,
  LUPINE_NCCL_MEM_FREE = 28,
  LUPINE_NCCL_COMM_SUSPEND = 29,
  LUPINE_NCCL_COMM_RESUME = 30,
  LUPINE_NCCL_COMM_MEM_STATS = 31,
  LUPINE_NCCL_WINDOW_REGISTER = 32,
  LUPINE_NCCL_WINDOW_DEREGISTER = 33,
  LUPINE_NCCL_WINDOW_USER_POINTER = 34,
  LUPINE_NCCL_COMM_GET_UNIQUE_ID = 35,
};

struct lupine_nccl_request {
  uint32_t opcode;
  int32_t ranks, rank, datatype, reduction, root;
  uint64_t count, send_buffer, receive_buffer, communicator, stream;
  unsigned char unique_id[128];
};
struct lupine_nccl_response {
  int32_t result, async_error;
  uint64_t communicator;
  unsigned char unique_id[128];
  int32_t value;
  uint64_t handle;
  char error_string[256];
};

static int remote_enabled(void) {
  const char *value = getenv("RGPU_NCCL_REMOTE");
  if (value != NULL) return strcmp(value, "1") == 0;
  typedef int (*route_query)(void);
  static route_query query;
  static int resolved;
  if (!resolved) {
    query = (route_query)dlsym(RTLD_DEFAULT,
                               "lupine_cuda_current_route_is_remote");
    resolved = 1;
  }
  return query != NULL && query() != 0;
}

static int remote_configured(void) {
  const char *server = getenv("LUPINE_SERVER");
  return (server != NULL && server[0] != '\0') || remote_enabled();
}

/* Communicator calls are legal even after an application changes its current
   CUDA device. Remember remote handles so they can never fall through to the
   workstation's real NCCL library under that common usage pattern. */
enum { MAX_REMOTE_COMMS = 4096 };
static pthread_mutex_t remote_comms_mutex = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t remote_comms[MAX_REMOTE_COMMS];
static int remote_comm_routes[MAX_REMOTE_COMMS];
static int remote_comm_devices[MAX_REMOTE_COMMS];

static int current_route_id(void) {
  typedef int (*route_query)(void);
  static route_query query;
  static int resolved;
  if (!resolved) {
    query = (route_query)dlsym(RTLD_DEFAULT, "lupine_cuda_current_route_id");
    resolved = 1;
  }
  return query == NULL ? -2 : query();
}

static int current_virtual_device(void) {
  typedef int (*get_device_fn)(int *);
  get_device_fn get_device =
      (get_device_fn)dlsym(RTLD_DEFAULT, "cudaGetDevice");
  int device = -1;
  return get_device != NULL && get_device(&device) == 0 ? device : -1;
}

static void remember_remote_comm_owner(ncclComm_t comm, int route_id,
                                       int device) {
  uintptr_t value = (uintptr_t)comm;
  if (value == 0) return;
  pthread_mutex_lock(&remote_comms_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_comms[i] == value) break;
    if (remote_comms[i] == 0) {
      remote_comms[i] = value;
      remote_comm_routes[i] = route_id;
      remote_comm_devices[i] = device;
      break;
    }
  }
  pthread_mutex_unlock(&remote_comms_mutex);
}

static void remember_remote_comm(ncclComm_t comm) {
  remember_remote_comm_owner(comm, current_route_id(), current_virtual_device());
}

static void forget_remote_comm(ncclComm_t comm) {
  uintptr_t value = (uintptr_t)comm;
  pthread_mutex_lock(&remote_comms_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_comms[i] == value) {
      remote_comms[i] = 0;
      remote_comm_routes[i] = -2;
      remote_comm_devices[i] = -1;
      break;
    }
  }
  pthread_mutex_unlock(&remote_comms_mutex);
}

static int remote_comm_route_id(ncclComm_t comm) {
  uintptr_t value = (uintptr_t)comm;
  int route_id = -2;
  pthread_mutex_lock(&remote_comms_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_comms[i] == value && value != 0) {
      route_id = remote_comm_routes[i];
      break;
    }
  }
  pthread_mutex_unlock(&remote_comms_mutex);
  return route_id;
}

static int remote_comm_device(ncclComm_t comm) {
  uintptr_t value = (uintptr_t)comm;
  int device = -1;
  pthread_mutex_lock(&remote_comms_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_comms[i] == value && value != 0) {
      device = remote_comm_devices[i];
      break;
    }
  }
  pthread_mutex_unlock(&remote_comms_mutex);
  return device;
}

static int remote_comm(ncclComm_t comm) {
  uintptr_t value = (uintptr_t)comm;
  int found = 0;
  pthread_mutex_lock(&remote_comms_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_comms[i] == value && value != 0) {
      found = 1;
      break;
    }
  }
  pthread_mutex_unlock(&remote_comms_mutex);
  return found || remote_enabled();
}

static pthread_mutex_t remote_allocations_mutex = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t remote_allocations[MAX_REMOTE_COMMS];

static void remember_remote_allocation(void *pointer) {
  uintptr_t value = (uintptr_t)pointer;
  if (value == 0) return;
  pthread_mutex_lock(&remote_allocations_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_allocations[i] == value) break;
    if (remote_allocations[i] == 0) {
      remote_allocations[i] = value;
      break;
    }
  }
  pthread_mutex_unlock(&remote_allocations_mutex);
}

static int forget_remote_allocation(void *pointer) {
  uintptr_t value = (uintptr_t)pointer;
  int found = 0;
  pthread_mutex_lock(&remote_allocations_mutex);
  for (size_t i = 0; i < MAX_REMOTE_COMMS; ++i) {
    if (remote_allocations[i] == value && value != 0) {
      remote_allocations[i] = 0;
      found = 1;
      break;
    }
  }
  pthread_mutex_unlock(&remote_allocations_mutex);
  return found;
}

static ncclResult_t remote_call(struct lupine_nccl_request *request,
                                struct lupine_nccl_response *response) {
  typedef int (*function)(const struct lupine_nccl_request *,
                          struct lupine_nccl_response *);
  typedef int (*route_function)(int, const struct lupine_nccl_request *,
                                struct lupine_nccl_response *);
  static function call;
  static route_function call_on_route;
  if (call == NULL)
    call = (function)dlsym(RTLD_DEFAULT, "lupine_nccl_call");
  if (call_on_route == NULL)
    call_on_route =
        (route_function)dlsym(RTLD_DEFAULT, "lupine_nccl_call_on_route");
  memset(response, 0, sizeof(*response));
  response->result = 2;
  if (call == NULL) {
    if (getenv("RGPU_NCCL_TRACE") != NULL)
      fprintf(stderr, "RGPU_NCCL_REMOTE missing lupine_nccl_call op=%u\n",
              request->opcode);
    return 2;
  }
  int route_id = remote_comm_route_id(
      (ncclComm_t)(uintptr_t)request->communicator);
  int transport_result = route_id >= 0 && call_on_route != NULL
                             ? call_on_route(route_id, request, response)
                             : call(request, response);
  if (getenv("RGPU_NCCL_TRACE") != NULL)
    fprintf(stderr, "RGPU_NCCL_REMOTE op=%u transport=%d result=%d\n",
            request->opcode, transport_result, response->result);
  if (transport_result != 0) return 2;
  return response->result;
}

#define DECLARE(result, name, signature) extern result p##name signature
#define TRACE(name)                                                           \
  do {                                                                        \
    if (getenv("RGPU_NCCL_TRACE") != NULL)                                   \
      fprintf(stderr, "RGPU_NCCL_FORWARD %s\n", #name);                     \
  } while (0)
#define FORWARD(result, name, signature, arguments)                           \
  DECLARE(result, name, signature);                                           \
  result name signature {                                                     \
    TRACE(name);                                                              \
    return p##name arguments;                                                 \
  }

FORWARD(ncclResult_t, ncclGetVersion, (int *version), (version))
DECLARE(void, ncclResetDebugInit, (void));
void ncclResetDebugInit(void) {
  TRACE(ncclResetDebugInit);
  pncclResetDebugInit();
}
DECLARE(ncclResult_t, ncclGetUniqueId, (ncclUniqueId *));
ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  TRACE(ncclGetUniqueId);
  if (!remote_enabled()) return pncclGetUniqueId(id);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_GET_UNIQUE_ID;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) memcpy(id->internal, response.unique_id, sizeof(id->internal));
  return result;
}
FORWARD(const char *, ncclGetErrorString, (ncclResult_t result), (result))
DECLARE(const char *, ncclGetLastError, (const ncclComm_t));
const char *ncclGetLastError(const ncclComm_t comm) {
  TRACE(ncclGetLastError);
  if (!remote_comm(comm)) return pncclGetLastError(comm);
  static _Thread_local char message[256];
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_LAST_ERROR;
  request.communicator = (uint64_t)(uintptr_t)comm;
  if (remote_call(&request, &response) != 0) return "remote NCCL RPC failed";
  memcpy(message, response.error_string, sizeof(message));
  message[sizeof(message) - 1] = '\0';
  return message;
}
static ncclResult_t remote_init_rank(ncclComm_t *comm, int ranks,
                                     ncclUniqueId id, int rank) {
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_INIT_RANK;
  request.ranks = ranks;
  request.rank = rank;
  memcpy(request.unique_id, id.internal, sizeof(request.unique_id));
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) {
    *comm = (ncclComm_t)(uintptr_t)response.communicator;
    remember_remote_comm(*comm);
  }
  return result;
}

DECLARE(ncclResult_t, ncclCommInitRank,
        (ncclComm_t *, int, ncclUniqueId, int));
ncclResult_t ncclCommInitRank(ncclComm_t *comm, int ranks, ncclUniqueId id,
                              int rank) {
  TRACE(ncclCommInitRank);
  return remote_enabled() ? remote_init_rank(comm, ranks, id, rank)
                          : pncclCommInitRank(comm, ranks, id, rank);
}
DECLARE(ncclResult_t, ncclCommInitRankConfig,
        (ncclComm_t *, int, ncclUniqueId, int, ncclConfig_t *));
ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t *config) {
  TRACE(ncclCommInitRankConfig);
  if (!remote_enabled())
    return pncclCommInitRankConfig(comm, ranks, id, rank, config);
  return remote_init_rank(comm, ranks, id, rank);
}
DECLARE(ncclResult_t, ncclCommInitRankScalable,
        (ncclComm_t *, int, int, int, ncclUniqueId *, ncclConfig_t *));
ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int ranks, int rank,
                                      int id_count, ncclUniqueId *ids,
                                      ncclConfig_t *config) {
  TRACE(ncclCommInitRankScalable);
  /* A scalable init carries a variable-length ID array, which protocol v5
     cannot encode.  Never pass remote handles into the local NCCL library. */
  if (remote_enabled()) return 5; /* ncclInvalidUsage */
  return pncclCommInitRankScalable(comm, ranks, rank, id_count, ids, config);
}

struct init_all_rank {
  ncclComm_t *comm;
  ncclUniqueId id;
  int ranks;
  int rank;
  int device;
  int route_id;
  ncclResult_t result;
};

static void *init_all_rank_main(void *opaque) {
  struct init_all_rank *rank = (struct init_all_rank *)opaque;
  typedef int (*set_device_fn)(int);
  set_device_fn set_device = (set_device_fn)dlsym(RTLD_DEFAULT, "cudaSetDevice");
  if (set_device == NULL || set_device(rank->device) != 0) {
    rank->result = 1; /* ncclUnhandledCudaError */
  } else if (rank->route_id >= 0) {
    rank->result = remote_init_rank(rank->comm, rank->ranks, rank->id,
                                    rank->rank);
  } else {
    rank->result = pncclCommInitRank(rank->comm, rank->ranks, rank->id,
                                     rank->rank);
  }
  return NULL;
}

DECLARE(ncclResult_t, ncclCommInitAll, (ncclComm_t *, int, const int *));
ncclResult_t ncclCommInitAll(ncclComm_t *comms, int devices,
                             const int *device_list) {
  TRACE(ncclCommInitAll);
  if (!remote_configured()) return pncclCommInitAll(comms, devices, device_list);
  if (comms == NULL || devices <= 0 || devices > 64) return 4;

  typedef int (*device_route_fn)(int);
  typedef int (*get_device_fn)(int *);
  typedef int (*set_device_fn)(int);
  device_route_fn device_route =
      (device_route_fn)dlsym(RTLD_DEFAULT, "lupine_cuda_device_route_id");
  get_device_fn get_device =
      (get_device_fn)dlsym(RTLD_DEFAULT, "cudaGetDevice");
  set_device_fn set_device =
      (set_device_fn)dlsym(RTLD_DEFAULT, "cudaSetDevice");
  if (device_route == NULL || get_device == NULL || set_device == NULL) return 5;

  struct init_all_rank ranks[64] = {0};
  int all_local = 1;
  for (int rank = 0; rank < devices; ++rank) {
    int device = device_list == NULL ? rank : device_list[rank];
    int route_id = device_route(device);
    if (route_id < -1) return 4;
    for (int previous = 0; route_id >= 0 && previous < rank; ++previous) {
      /* Multiple devices behind one server need a grouped server-side init,
         which is not representable in protocol v5. */
      if (ranks[previous].route_id == route_id) return 5;
    }
    ranks[rank].comm = &comms[rank];
    ranks[rank].ranks = devices;
    ranks[rank].rank = rank;
    ranks[rank].device = device;
    ranks[rank].route_id = route_id;
    all_local = all_local && route_id == -1;
  }
  if (all_local) return pncclCommInitAll(comms, devices, device_list);

  ncclUniqueId id;
  ncclResult_t result = pncclGetUniqueId(&id);
  if (result != 0) return result;
  int original_device = 0;
  int have_original_device = get_device(&original_device) == 0;
  pthread_t threads[64];
  int started = 0;
  for (int rank = 0; rank < devices; ++rank) {
    ranks[rank].id = id;
    ranks[rank].result = 2;
    if (pthread_create(&threads[rank], NULL, init_all_rank_main,
                       &ranks[rank]) != 0) {
      result = 2;
      break;
    }
    ++started;
  }
  for (int rank = 0; rank < started; ++rank) {
    pthread_join(threads[rank], NULL);
    if (result == 0 && ranks[rank].result != 0) result = ranks[rank].result;
  }
  if (have_original_device) (void)set_device(original_device);
  return result;
}
static ncclResult_t remote_simple(uint32_t opcode, ncclComm_t comm) {
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = opcode;
  request.communicator = (uint64_t)(uintptr_t)comm;
  return remote_call(&request, &response);
}

#define REMOTE_GROUP(name, opcode)                                             \
  DECLARE(ncclResult_t, name, (void));                                         \
  ncclResult_t name(void) {                                                    \
    TRACE(name);                                                              \
    return remote_enabled() ? remote_simple(opcode, NULL) : p##name();        \
  }
REMOTE_GROUP(ncclGroupStart, LUPINE_NCCL_GROUP_START)
REMOTE_GROUP(ncclGroupEnd, LUPINE_NCCL_GROUP_END)

static ncclResult_t remote_collective(
    uint32_t opcode, const void *send, void *receive, size_t count,
    ncclDataType_t datatype, ncclRedOp_t reduction, int root, ncclComm_t comm,
    cudaStream_t stream) {
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = opcode;
  request.datatype = datatype;
  request.reduction = reduction;
  request.root = root;
  request.count = count;
  request.send_buffer = (uint64_t)(uintptr_t)send;
  request.receive_buffer = (uint64_t)(uintptr_t)receive;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.stream = (uint64_t)(uintptr_t)stream;
  return remote_call(&request, &response);
}

#define REMOTE_REDUCTION(name, opcode)                                         \
  DECLARE(ncclResult_t, name,                                                  \
          (const void *, void *, size_t, ncclDataType_t, ncclRedOp_t,         \
           ncclComm_t, cudaStream_t));                                         \
  ncclResult_t name(const void *send, void *receive, size_t count,             \
                    ncclDataType_t datatype, ncclRedOp_t reduction,            \
                    ncclComm_t comm, cudaStream_t stream) {                    \
    TRACE(name);                                                              \
    return remote_comm(comm)                                                 \
               ? remote_collective(opcode, send, receive, count, datatype,    \
                                   reduction, 0, comm, stream)                \
               : p##name(send, receive, count, datatype, reduction, comm,     \
                          stream);                                             \
  }
REMOTE_REDUCTION(ncclAllReduce, LUPINE_NCCL_ALL_REDUCE)
REMOTE_REDUCTION(ncclReduceScatter, LUPINE_NCCL_REDUCE_SCATTER)

DECLARE(ncclResult_t, ncclReduce,
        (const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, int,
         ncclComm_t, cudaStream_t));
ncclResult_t ncclReduce(const void *send, void *receive, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t reduction,
                        int root, ncclComm_t comm, cudaStream_t stream) {
  TRACE(ncclReduce);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_REDUCE, send, receive, count,
                                 datatype, reduction, root, comm, stream)
             : pncclReduce(send, receive, count, datatype, reduction, root,
                           comm, stream);
}

DECLARE(ncclResult_t, ncclAllGather,
        (const void *, void *, size_t, ncclDataType_t, ncclComm_t,
         cudaStream_t));
ncclResult_t ncclAllGather(const void *send, void *receive, size_t count,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream) {
  TRACE(ncclAllGather);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_ALL_GATHER, send, receive, count,
                                 datatype, 0, 0, comm, stream)
             : pncclAllGather(send, receive, count, datatype, comm, stream);
}

DECLARE(ncclResult_t, ncclBroadcast,
        (const void *, void *, size_t, ncclDataType_t, int, ncclComm_t,
         cudaStream_t));
ncclResult_t ncclBroadcast(const void *send, void *receive, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  TRACE(ncclBroadcast);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_BROADCAST, send, receive, count,
                                 datatype, 0, root, comm, stream)
             : pncclBroadcast(send, receive, count, datatype, root, comm,
                               stream);
}

DECLARE(ncclResult_t, ncclBcast,
        (void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t));
ncclResult_t ncclBcast(void *buffer, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, cudaStream_t stream) {
  TRACE(ncclBcast);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_BCAST, buffer, buffer, count,
                                 datatype, 0, root, comm, stream)
             : pncclBcast(buffer, count, datatype, root, comm, stream);
}

#define REMOTE_ROOTED(name, opcode)                                           \
  DECLARE(ncclResult_t, name,                                                 \
          (const void *, void *, size_t, ncclDataType_t, int, ncclComm_t,    \
           cudaStream_t));                                                    \
  ncclResult_t name(const void *send, void *receive, size_t count,            \
                    ncclDataType_t datatype, int root, ncclComm_t comm,       \
                    cudaStream_t stream) {                                    \
    TRACE(name);                                                              \
    return remote_comm(comm)                                                 \
               ? remote_collective(opcode, send, receive, count, datatype,   \
                                   0, root, comm, stream)                     \
               : p##name(send, receive, count, datatype, root, comm, stream);\
  }
REMOTE_ROOTED(ncclGather, LUPINE_NCCL_GATHER)
REMOTE_ROOTED(ncclScatter, LUPINE_NCCL_SCATTER)

DECLARE(ncclResult_t, ncclAlltoAll,
        (const void *, void *, size_t, ncclDataType_t, ncclComm_t,
         cudaStream_t));
ncclResult_t ncclAlltoAll(const void *send, void *receive, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t stream) {
  TRACE(ncclAlltoAll);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_ALL_TO_ALL, send, receive, count,
                                 datatype, 0, 0, comm, stream)
             : pncclAlltoAll(send, receive, count, datatype, comm, stream);
}

DECLARE(ncclResult_t, ncclSend,
        (const void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t));
ncclResult_t ncclSend(const void *send, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  TRACE(ncclSend);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_SEND, send, NULL, count, datatype,
                                 0, peer, comm, stream)
             : pncclSend(send, count, datatype, peer, comm, stream);
}
DECLARE(ncclResult_t, ncclRecv,
        (void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t));
ncclResult_t ncclRecv(void *receive, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  TRACE(ncclRecv);
  return remote_comm(comm)
             ? remote_collective(LUPINE_NCCL_RECV, NULL, receive, count,
                                 datatype, 0, peer, comm, stream)
             : pncclRecv(receive, count, datatype, peer, comm, stream);
}

static ncclResult_t remote_query(uint32_t opcode, ncclComm_t comm, int *value) {
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = opcode;
  request.communicator = (uint64_t)(uintptr_t)comm;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) *value = response.value;
  return result;
}
#define REMOTE_QUERY(name, opcode)                                            \
  DECLARE(ncclResult_t, name, (const ncclComm_t, int *));                    \
  ncclResult_t name(const ncclComm_t comm, int *value) {                     \
    TRACE(name);                                                              \
    return remote_comm(comm) ? remote_query(opcode, comm, value)             \
                            : p##name(comm, value);                           \
  }
REMOTE_QUERY(ncclCommCount, LUPINE_NCCL_COMM_COUNT)
REMOTE_QUERY(ncclCommUserRank, LUPINE_NCCL_COMM_USER_RANK)

DECLARE(ncclResult_t, ncclCommCuDevice, (const ncclComm_t, int *));
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  TRACE(ncclCommCuDevice);
  if (!remote_comm(comm)) return pncclCommCuDevice(comm, device);
  int virtual_device = remote_comm_device(comm);
  if (virtual_device >= 0) {
    *device = virtual_device;
    return 0;
  }
  return remote_query(LUPINE_NCCL_COMM_CUDA_DEVICE, comm, device);
}

DECLARE(ncclResult_t, ncclCommRegister,
        (const ncclComm_t, void *, size_t, void **));
ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buffer, size_t size,
                              void **handle) {
  TRACE(ncclCommRegister);
  if (!remote_comm(comm)) return pncclCommRegister(comm, buffer, size, handle);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_COMM_REGISTER;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.send_buffer = (uint64_t)(uintptr_t)buffer;
  request.count = size;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) *handle = (void *)(uintptr_t)response.handle;
  return result;
}
DECLARE(ncclResult_t, ncclCommDeregister, (const ncclComm_t, void *));
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  TRACE(ncclCommDeregister);
  if (!remote_comm(comm)) return pncclCommDeregister(comm, handle);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_COMM_DEREGISTER;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.send_buffer = (uint64_t)(uintptr_t)handle;
  return remote_call(&request, &response);
}

DECLARE(ncclResult_t, ncclCommSplit,
        (ncclComm_t, int, int, ncclComm_t *, ncclConfig_t *));
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm, ncclConfig_t *config) {
  TRACE(ncclCommSplit);
  if (!remote_comm(comm)) return pncclCommSplit(comm, color, key, newcomm, config);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_COMM_SPLIT;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.rank = color;
  request.root = key;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) {
    *newcomm = (ncclComm_t)(uintptr_t)response.communicator;
    remember_remote_comm_owner(*newcomm, remote_comm_route_id(comm),
                               remote_comm_device(comm));
  }
  return result;
}

DECLARE(ncclResult_t, ncclCommGetAsyncError,
        (ncclComm_t, ncclResult_t *));
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *error) {
  if (!remote_comm(comm)) return pncclCommGetAsyncError(comm, error);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_ASYNC_ERROR;
  request.communicator = (uint64_t)(uintptr_t)comm;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) *error = response.async_error;
  return result;
}

#define REMOTE_COMM(name, opcode)                                             \
  DECLARE(ncclResult_t, name, (ncclComm_t));                                 \
  ncclResult_t name(ncclComm_t comm) {                                       \
    TRACE(name);                                                             \
    return remote_comm(comm) ? remote_simple(opcode, comm) : p##name(comm);  \
  }
REMOTE_COMM(ncclCommFinalize, LUPINE_NCCL_FINALIZE)

#define REMOTE_COMM_RELEASE(name, opcode)                                     \
  DECLARE(ncclResult_t, name, (ncclComm_t));                                 \
  ncclResult_t name(ncclComm_t comm) {                                       \
    TRACE(name);                                                             \
    if (!remote_comm(comm)) return p##name(comm);                            \
    ncclResult_t result = remote_simple(opcode, comm);                       \
    forget_remote_comm(comm);                                                \
    return result;                                                           \
  }
REMOTE_COMM_RELEASE(ncclCommDestroy, LUPINE_NCCL_DESTROY)
REMOTE_COMM_RELEASE(ncclCommAbort, LUPINE_NCCL_ABORT)

DECLARE(ncclResult_t, ncclMemAlloc, (void **, size_t));
ncclResult_t ncclMemAlloc(void **pointer, size_t size) {
  TRACE(ncclMemAlloc);
  if (!remote_enabled()) return pncclMemAlloc(pointer, size);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_MEM_ALLOC;
  request.count = size;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) {
    *pointer = (void *)(uintptr_t)response.handle;
    remember_remote_allocation(*pointer);
  }
  return result;
}

DECLARE(ncclResult_t, ncclMemFree, (void *));
ncclResult_t ncclMemFree(void *pointer) {
  TRACE(ncclMemFree);
  int was_remote = forget_remote_allocation(pointer);
  if (!was_remote && !remote_enabled()) return pncclMemFree(pointer);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_MEM_FREE;
  request.send_buffer = (uint64_t)(uintptr_t)pointer;
  return remote_call(&request, &response);
}

DECLARE(ncclResult_t, ncclCommGetUniqueId, (ncclComm_t, ncclUniqueId *));
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *id) {
  TRACE(ncclCommGetUniqueId);
  if (!remote_comm(comm)) return pncclCommGetUniqueId(comm, id);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_COMM_GET_UNIQUE_ID;
  request.communicator = (uint64_t)(uintptr_t)comm;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) memcpy(id->internal, response.unique_id, sizeof(id->internal));
  return result;
}

#define REMOTE_COMM_FLAGS(name, rpc_opcode)                                   \
  DECLARE(ncclResult_t, name, (ncclComm_t, int));                            \
  ncclResult_t name(ncclComm_t comm, int flags) {                            \
    TRACE(name);                                                             \
    if (!remote_comm(comm)) return p##name(comm, flags);                     \
    struct lupine_nccl_request request = {0};                                \
    struct lupine_nccl_response response;                                    \
    request.opcode = rpc_opcode;                                             \
    request.communicator = (uint64_t)(uintptr_t)comm;                        \
    request.root = flags;                                                    \
    return remote_call(&request, &response);                                 \
  }
REMOTE_COMM_FLAGS(ncclCommSuspend, LUPINE_NCCL_COMM_SUSPEND)
REMOTE_COMM(ncclCommResume, LUPINE_NCCL_COMM_RESUME)

DECLARE(ncclResult_t, ncclCommMemStats,
        (ncclComm_t, ncclCommMemStat_t, uint64_t *));
ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat,
                              uint64_t *value) {
  TRACE(ncclCommMemStats);
  if (!remote_comm(comm)) return pncclCommMemStats(comm, stat, value);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_COMM_MEM_STATS;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.root = stat;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) *value = response.handle;
  return result;
}

DECLARE(ncclResult_t, ncclCommWindowRegister,
        (ncclComm_t, void *, size_t, ncclWindow_t *, int));
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buffer, size_t size,
                                    ncclWindow_t *window, int flags) {
  TRACE(ncclCommWindowRegister);
  if (!remote_comm(comm))
    return pncclCommWindowRegister(comm, buffer, size, window, flags);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_WINDOW_REGISTER;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.send_buffer = (uint64_t)(uintptr_t)buffer;
  request.count = size;
  request.root = flags;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) *window = (ncclWindow_t)(uintptr_t)response.handle;
  return result;
}

DECLARE(ncclResult_t, ncclCommWindowDeregister, (ncclComm_t, ncclWindow_t));
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t window) {
  TRACE(ncclCommWindowDeregister);
  if (!remote_comm(comm)) return pncclCommWindowDeregister(comm, window);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_WINDOW_DEREGISTER;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.send_buffer = (uint64_t)(uintptr_t)window;
  return remote_call(&request, &response);
}

DECLARE(ncclResult_t, ncclWinGetUserPtr, (ncclComm_t, ncclWindow_t, void **));
ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t window,
                               void **pointer) {
  TRACE(ncclWinGetUserPtr);
  if (!remote_comm(comm)) return pncclWinGetUserPtr(comm, window, pointer);
  struct lupine_nccl_request request = {0};
  struct lupine_nccl_response response;
  request.opcode = LUPINE_NCCL_WINDOW_USER_POINTER;
  request.communicator = (uint64_t)(uintptr_t)comm;
  request.send_buffer = (uint64_t)(uintptr_t)window;
  ncclResult_t result = remote_call(&request, &response);
  if (result == 0) *pointer = (void *)(uintptr_t)response.handle;
  return result;
}

/* Remaining extended APIs must not resolve through libnccl_real for a remote
   communicator. Protocol v5 does not encode their arguments yet, so return a
   documented NCCL error instead of risking local dereferences of remote
   pointers or handles. */
#define REMOTE_UNSUPPORTED(name, signature, arguments, predicate)             \
  DECLARE(ncclResult_t, name, signature);                                    \
  ncclResult_t name signature {                                              \
    TRACE(name);                                                             \
    return (predicate) ? 5 : p##name arguments; /* ncclInvalidUsage */       \
  }

REMOTE_UNSUPPORTED(ncclCommRevoke, (ncclComm_t comm, int flags),
                   (comm, flags), remote_comm(comm))
REMOTE_UNSUPPORTED(ncclCommShrink,
                   (ncclComm_t comm, int *excluded, int excluded_count,
                    ncclComm_t *newcomm, ncclConfig_t *config, int flags),
                   (comm, excluded, excluded_count, newcomm, config, flags),
                   remote_comm(comm))
REMOTE_UNSUPPORTED(ncclCommGrow,
                   (ncclComm_t comm, int ranks, const ncclUniqueId *id,
                    int rank, ncclComm_t *newcomm, ncclConfig_t *config),
                   (comm, ranks, id, rank, newcomm, config), remote_comm(comm))
REMOTE_UNSUPPORTED(ncclRedOpCreatePreMulSum,
                   (ncclRedOp_t *operation, void *scalar,
                    ncclDataType_t datatype, ncclScalarResidence_t residence,
                    ncclComm_t comm),
                   (operation, scalar, datatype, residence, comm),
                   remote_comm(comm))
REMOTE_UNSUPPORTED(ncclRedOpDestroy,
                   (ncclRedOp_t operation, ncclComm_t comm), (operation, comm),
                   remote_comm(comm))
REMOTE_UNSUPPORTED(ncclGroupSimulateEnd, (ncclSimInfo_t *info), (info),
                   remote_enabled())
REMOTE_UNSUPPORTED(ncclCommQueryProperties,
                   (ncclComm_t comm, void *properties), (comm, properties),
                   remote_comm(comm))
