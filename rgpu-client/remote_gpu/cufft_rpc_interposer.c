#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  FFT_CREATE = 51,
  FFT_DESTROY,
  FFT_SET_AUTO_ALLOCATION,
  FFT_SET_STREAM,
  FFT_SET_WORK_AREA,
  FFT_XT_MAKE_PLAN_MANY,
  FFT_XT_EXEC,
};

struct request {
  uint32_t opcode;
  uint64_t handle, stream, workspace, workspace_size, a, b, c, d;
  int32_t transa, transb, m, n, k, lda, ldb, ldc, value;
  float alpha, beta;
  int32_t a_type, b_type, c_type, compute_type, algorithm;
  uint32_t scalar_size;
  uint8_t alpha_data[16], beta_data[16];
  uint64_t descriptor, a_descriptor, b_descriptor, c_descriptor;
  uint64_t d_descriptor, preference, rows, columns;
  int64_t leading_dimension;
  int32_t attribute, requested_algorithms;
  uint32_t payload_size;
  uint8_t payload[128];
  int64_t stride_a, stride_b, stride_c;
  int32_t batch_count;
  uint32_t handle_state_mask;
  int32_t math_mode, pointer_mode;
  uint32_t asynchronous;
  uint64_t context;
};
struct response {
  int32_t status, value;
  uint64_t handle;
  int32_t returned_algorithms;
  uint32_t payload_size;
  uint8_t payload[768];
};
typedef int (*rpc_function)(int, const struct request *, struct response *);
typedef int (*route_function)(void);

struct plan_entry {
  int synthetic;
  int remote;
  int route;
};
static struct plan_entry plans[256];
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void *real_library;
static pthread_once_t real_once = PTHREAD_ONCE_INIT;
static void open_real_library(void) {
  real_library = dlopen(
      "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcufft.so.12",
      RTLD_NOW | RTLD_LOCAL);
  if (real_library == 0)
    real_library = dlopen("libcufft.so.12", RTLD_NOW | RTLD_LOCAL);
}
static void *real_symbol(const char *name) {
  pthread_once(&real_once, open_real_library);
  if (real_library == 0) return 0;
  void *symbol = dlvsym(real_library, name, "libcufft.so.12");
  return symbol == 0 ? dlsym(real_library, name) : symbol;
}
static rpc_function rpc(void) {
  static rpc_function call;
  if (call == 0)
    call = (rpc_function)dlsym(RTLD_DEFAULT, "lupine_cublas_call_on_route");
  return call;
}
static int current_route(void) {
  static route_function route;
  if (route == 0)
    route = (route_function)dlsym(RTLD_DEFAULT, "lupine_cuda_current_route_id");
  return route == 0 ? -2 : route();
}
static struct plan_entry *find_plan(int plan) {
  for (size_t i = 0; i < 256; ++i)
    if (plans[i].synthetic == plan) return &plans[i];
  return 0;
}
static int remote_call(const struct plan_entry *entry, struct request *request,
                       struct response *response) {
  typedef int (*get_context_function)(void **);
  static get_context_function get_context;
  if (get_context == 0)
    get_context = (get_context_function)dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
  void *context = 0;
  if (get_context != 0 && get_context(&context) == 0)
    request->context = (uint64_t)(uintptr_t)context;
  memset(response, 0, sizeof(*response));
  response->status = 5;
  return entry == 0 || rpc() == 0 ||
                 rpc()(entry->route, request, response) != 0
             ? 5
             : response->status;
}

int cufftCreate(int *plan) {
  typedef int (*real_function)(int *);
  int route = current_route();
  if (route < 0) {
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cufftCreate");
    return real == 0 ? 5 : real(plan);
  }
  struct plan_entry route_entry = {0, 0, route};
  struct request request = {0};
  struct response response;
  request.opcode = FFT_CREATE;
  int status = remote_call(&route_entry, &request, &response);
  if (status != 0) return status;
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < 256; ++i) {
    if (plans[i].synthetic == 0) {
      int synthetic = 0x60000000 + (int)i + 1;
      plans[i] = (struct plan_entry){synthetic, response.value, route};
      *plan = synthetic;
      pthread_mutex_unlock(&mutex);
      return 0;
    }
  }
  pthread_mutex_unlock(&mutex);
  return 2;
}

int cufftDestroy(int plan) {
  typedef int (*real_function)(int);
  pthread_mutex_lock(&mutex);
  struct plan_entry *found = find_plan(plan);
  if (found == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cufftDestroy");
    return real == 0 ? 5 : real(plan);
  }
  struct plan_entry copy = *found;
  found->synthetic = 0;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = FFT_DESTROY;
  request.handle = (uint64_t)(uint32_t)copy.remote;
  return remote_call(&copy, &request, &response);
}

static int plan_control(int plan, uint32_t opcode, uint64_t argument,
                        const char *symbol_name) {
  pthread_mutex_lock(&mutex);
  struct plan_entry *found = find_plan(plan);
  if (found == 0) {
    pthread_mutex_unlock(&mutex);
    if (opcode == FFT_SET_AUTO_ALLOCATION) {
      typedef int (*function)(int, int);
      function real = (function)real_symbol(symbol_name);
      return real == 0 ? 5 : real(plan, (int)argument);
    }
    typedef int (*function)(int, void *);
    function real = (function)real_symbol(symbol_name);
    return real == 0 ? 5 : real(plan, (void *)(uintptr_t)argument);
  }
  struct plan_entry copy = *found;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = opcode;
  request.handle = (uint64_t)(uint32_t)copy.remote;
  if (opcode == FFT_SET_AUTO_ALLOCATION)
    request.value = (int32_t)argument;
  else if (opcode == FFT_SET_STREAM)
    request.d = argument;
  else
    request.workspace = argument;
  return remote_call(&copy, &request, &response);
}

int cufftSetAutoAllocation(int plan, int auto_allocate) {
  return plan_control(plan, FFT_SET_AUTO_ALLOCATION,
                      (uint64_t)(uint32_t)auto_allocate,
                      "cufftSetAutoAllocation");
}
int cufftSetStream(int plan, void *stream) {
  return plan_control(plan, FFT_SET_STREAM, (uint64_t)(uintptr_t)stream,
                      "cufftSetStream");
}
int cufftSetWorkArea(int plan, void *workspace) {
  return plan_control(plan, FFT_SET_WORK_AREA, (uint64_t)(uintptr_t)workspace,
                      "cufftSetWorkArea");
}

int cufftXtMakePlanMany(int plan, int rank, int64_t *n, int64_t *inembed,
                        int64_t istride, int64_t idist, int input_type,
                        int64_t *onembed, int64_t ostride, int64_t odist,
                        int output_type, int64_t batch, size_t *work_size,
                        int execution_type) {
  typedef int (*real_function)(int, int, int64_t *, int64_t *, int64_t,
                               int64_t, int, int64_t *, int64_t, int64_t, int,
                               int64_t, size_t *, int);
  pthread_mutex_lock(&mutex);
  struct plan_entry *found = find_plan(plan);
  if (found == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cufftXtMakePlanMany");
    return real == 0 ? 5
                     : real(plan, rank, n, inembed, istride, idist, input_type,
                            onembed, ostride, odist, output_type, batch,
                            work_size, execution_type);
  }
  struct plan_entry copy = *found;
  pthread_mutex_unlock(&mutex);
  if (rank < 1 || rank > 5 || n == 0) return 4;
  struct request request = {0};
  struct response response;
  request.opcode = FFT_XT_MAKE_PLAN_MANY;
  request.handle = (uint64_t)(uint32_t)copy.remote;
  request.m = rank; request.stride_a = istride; request.stride_b = idist;
  request.a_type = input_type; request.stride_c = ostride;
  request.leading_dimension = odist; request.b_type = output_type;
  request.rows = (uint64_t)batch; request.compute_type = execution_type;
  request.attribute = (inembed != 0 ? 1 : 0) | (onembed != 0 ? 2 : 0);
  int64_t *payload = (int64_t *)request.payload;
  for (int i = 0; i < rank; ++i) {
    payload[i] = n[i];
    payload[rank + i] = inembed == 0 ? 0 : inembed[i];
    payload[2 * rank + i] = onembed == 0 ? 0 : onembed[i];
  }
  request.payload_size = (uint32_t)(3 * rank * sizeof(int64_t));
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && work_size != 0) *work_size = (size_t)response.handle;
  return status;
}

int cufftXtExec(int plan, void *input, void *output, int direction) {
  typedef int (*real_function)(int, void *, void *, int);
  pthread_mutex_lock(&mutex);
  struct plan_entry *found = find_plan(plan);
  if (found == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cufftXtExec");
    return real == 0 ? 5 : real(plan, input, output, direction);
  }
  struct plan_entry copy = *found;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = FFT_XT_EXEC;
  request.handle = (uint64_t)(uint32_t)copy.remote;
  request.a = (uint64_t)(uintptr_t)input;
  request.b = (uint64_t)(uintptr_t)output;
  request.value = direction;
  return remote_call(&copy, &request, &response);
}
