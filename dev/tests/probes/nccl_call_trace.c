#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef void *ncclComm_t;
typedef void *cudaStream_t;
typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef struct { char internal[128]; } ncclUniqueId;
typedef struct ncclConfig_v22800 ncclConfig_t;

#define FORWARD(name, signature, arguments)                                      \
  ncclResult_t name signature {                                                  \
    static __typeof__(&name) next;                                               \
    if (next == NULL) next = (__typeof__(&name))dlsym(RTLD_NEXT, #name);         \
    fprintf(stderr, "NCCL_CALL %s\n", #name);                                  \
    return next arguments;                                                       \
  }

FORWARD(ncclGetUniqueId, (ncclUniqueId *id), (id))
FORWARD(ncclCommInitRank,
        (ncclComm_t *comm, int ranks, ncclUniqueId id, int rank),
        (comm, ranks, id, rank))
FORWARD(ncclCommInitRankConfig,
        (ncclComm_t *comm, int ranks, ncclUniqueId id, int rank,
         ncclConfig_t *config),
        (comm, ranks, id, rank, config))
FORWARD(ncclCommInitRankScalable,
        (ncclComm_t *comm, int ranks, int rank, int id_count,
         ncclUniqueId *ids, ncclConfig_t *config),
        (comm, ranks, rank, id_count, ids, config))
FORWARD(ncclGroupStart, (void), ())
FORWARD(ncclGroupEnd, (void), ())
ncclResult_t ncclAllReduce(const void *send, void *receive, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t operation,
                           ncclComm_t comm, cudaStream_t stream) {
  if (getenv("NCCL_TRACE_FAIL_ALLREDUCE") != NULL) {
    fprintf(stderr, "NCCL_CALL ncclAllReduce forced failure\n");
    return 1;
  }
  static __typeof__(&ncclAllReduce) next;
  if (next == NULL) next = (__typeof__(&ncclAllReduce))dlsym(RTLD_NEXT, "ncclAllReduce");
  fprintf(stderr, "NCCL_CALL ncclAllReduce\n");
  return next(send, receive, count, datatype, operation, comm, stream);
}
FORWARD(ncclAllGather,
        (const void *send, void *receive, size_t count, ncclDataType_t datatype,
         ncclComm_t comm, cudaStream_t stream),
        (send, receive, count, datatype, comm, stream))
FORWARD(ncclBroadcast,
        (const void *send, void *receive, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream),
        (send, receive, count, datatype, root, comm, stream))
FORWARD(ncclCommGetAsyncError, (ncclComm_t comm, ncclResult_t *error),
        (comm, error))
FORWARD(ncclCommDestroy, (ncclComm_t comm), (comm))
FORWARD(ncclCommAbort, (ncclComm_t comm), (comm))

FORWARD(pncclCommInitRankScalable,
        (ncclComm_t *comm, int ranks, int rank, int id_count,
         ncclUniqueId *ids, ncclConfig_t *config),
        (comm, ranks, rank, id_count, ids, config))
FORWARD(pncclGroupStart, (void), ())
FORWARD(pncclGroupEnd, (void), ())
FORWARD(pncclAllReduce,
        (const void *send, void *receive, size_t count, ncclDataType_t datatype,
         ncclRedOp_t operation, ncclComm_t comm, cudaStream_t stream),
        (send, receive, count, datatype, operation, comm, stream))
FORWARD(pncclAllGather,
        (const void *send, void *receive, size_t count, ncclDataType_t datatype,
         ncclComm_t comm, cudaStream_t stream),
        (send, receive, count, datatype, comm, stream))
FORWARD(pncclBroadcast,
        (const void *send, void *receive, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream),
        (send, receive, count, datatype, root, comm, stream))
