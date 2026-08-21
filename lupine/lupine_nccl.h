#pragma once

#include <stdint.h>

enum lupine_nccl_opcode : uint32_t {
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
  uint32_t opcode = 0;
  int32_t ranks = 0;
  int32_t rank = 0;
  int32_t datatype = 0;
  int32_t reduction = 0;
  int32_t root = 0;
  uint64_t count = 0;
  uint64_t send_buffer = 0;
  uint64_t receive_buffer = 0;
  uint64_t communicator = 0;
  uint64_t stream = 0;
  unsigned char unique_id[128] = {};
};

struct lupine_nccl_response {
  int32_t result = 2;
  int32_t async_error = 0;
  uint64_t communicator = 0;
  unsigned char unique_id[128] = {};
  int32_t value = 0;
  uint64_t handle = 0;
  char error_string[256] = {};
};

extern "C" int lupine_nccl_call(const lupine_nccl_request *request,
                                 lupine_nccl_response *response);
