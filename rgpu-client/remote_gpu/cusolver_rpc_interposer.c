#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void *cusolver_handle;

enum {
  SOLVER_CREATE = 22,
  SOLVER_DESTROY,
  SOLVER_SET_STREAM,
  SOLVER_SGETRF_BUFFER_SIZE,
  SOLVER_SGETRF,
  SOLVER_SGETRS,
  SOLVER_CREATE_GESVDJ_INFO,
  SOLVER_DESTROY_GESVDJ_INFO,
  SOLVER_GESVDJ_SET_TOLERANCE,
  SOLVER_GESVDJ_SET_MAX_SWEEPS,
  SOLVER_SGESVDJ_BATCHED_BUFFER_SIZE,
  SOLVER_SGESVDJ_BATCHED,
  SOLVER_GESVDJ_SET_SORT_EIG,
  SOLVER_CREATE_PARAMS = 37,
  SOLVER_DESTROY_PARAMS,
  SOLVER_SET_ADV_OPTIONS,
  SOLVER_XPOTRF_BUFFER_SIZE,
  SOLVER_XPOTRF,
  SOLVER_XGEQRF_BUFFER_SIZE = 43,
  SOLVER_XGEQRF,
  SOLVER_SORGQR_BUFFER_SIZE,
  SOLVER_SORGQR,
  SOLVER_XSYEV_BATCHED_BUFFER_SIZE,
  SOLVER_XSYEV_BATCHED,
  SOLVER_SORMQR_BUFFER_SIZE,
  SOLVER_SORMQR,
  SOLVER_SSYTRF_BUFFER_SIZE = 59,
  SOLVER_SSYTRF,
  SOLVER_XSYTRS_BUFFER_SIZE,
  SOLVER_XSYTRS,
  SOLVER_XGEEV_BUFFER_SIZE,
  SOLVER_XGEEV,
  SOLVER_CGETRF_BUFFER_SIZE = 66,
  SOLVER_CGETRF,
  SOLVER_CGETRS,
  SOLVER_XPOTRS = 70,
  SOLVER_ZGESVDJ_BUFFER_SIZE = 73,
  SOLVER_ZGESVDJ = 74,
  SOLVER_ZGESVDJ_BATCHED_BUFFER_SIZE = 75,
  SOLVER_ZGESVDJ_BATCHED = 76,
  SOLVER_ZUNMQR_BUFFER_SIZE = 78,
  SOLVER_ZUNMQR = 79,
  SOLVER_ZGETRF_BUFFER_SIZE = 80,
  SOLVER_ZGETRF = 81,
  SOLVER_ZGETRS = 82,
  SOLVER_ZUNGQR_BUFFER_SIZE = 84,
  SOLVER_ZUNGQR = 85,
  SOLVER_CGESVDJ_BUFFER_SIZE = 91,
  SOLVER_CGESVDJ = 92,
  SOLVER_CGESVDJ_BATCHED_BUFFER_SIZE = 93,
  SOLVER_CGESVDJ_BATCHED = 94,
  SOLVER_CUNMQR_BUFFER_SIZE = 96,
  SOLVER_CUNMQR = 97,
  SOLVER_CUNGQR_BUFFER_SIZE = 98,
  SOLVER_CUNGQR = 99,
  SOLVER_DGETRF_BUFFER_SIZE = 101,
  SOLVER_DGETRF = 102,
  SOLVER_DGETRS = 103,
  SOLVER_DGESVDJ_BUFFER_SIZE = 104,
  SOLVER_DGESVDJ = 105,
  SOLVER_DGESVDJ_BATCHED_BUFFER_SIZE = 106,
  SOLVER_DGESVDJ_BATCHED = 107,
  SOLVER_DORGQR_BUFFER_SIZE = 108,
  SOLVER_DORGQR = 109,
  SOLVER_DORMQR_BUFFER_SIZE = 110,
  SOLVER_DORMQR = 111,
  SOLVER_DPOTRF_BUFFER_SIZE = 118,
  SOLVER_DPOTRF = 119,
  SOLVER_DPOTRS = 120,
  SOLVER_DGEQRF_BUFFER_SIZE = 121,
  SOLVER_DGEQRF = 122,
  SOLVER_DGESVD_BUFFER_SIZE = 123,
  SOLVER_DGESVD = 124,
  SOLVER_DSYEVD_BUFFER_SIZE = 125,
  SOLVER_DSYEVD = 126,
  SOLVER_DPOTRF_BATCHED = 127,
  SOLVER_DPOTRS_BATCHED = 128,
  SOLVER_DSYEVJ_BATCHED_BUFFER_SIZE = 129,
  SOLVER_DSYEVJ_BATCHED = 130,
  SOLVER_ZSYTRF_BUFFER_SIZE = 141,
  SOLVER_ZSYTRF = 142,
  SOLVER_CSYTRF_BUFFER_SIZE = 143,
  SOLVER_CSYTRF = 144,
  SOLVER_DSYTRF_BUFFER_SIZE = 145,
  SOLVER_DSYTRF = 146,
  SOLVER_SPOTRF_BATCHED = 157,
  SOLVER_CPOTRF_BATCHED = 158,
  SOLVER_ZPOTRF_BATCHED = 159,
};

/* Keep this wire layout byte-identical to lupine_cublas_request. */
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

struct handle_entry {
  cusolver_handle handle;
  int route;
};
static struct handle_entry handles[1024];
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static int trace_enabled(void) {
  const char *value = getenv("RGPU_CUSOLVER_TRACE");
  return value != 0 && value[0] != '\0' && strcmp(value, "0") != 0;
}
static void trace_signal_handler(int signal_number) {
  void *frames[64];
  int count = backtrace(frames, 64);
  static const char heading[] = "rgpu-cusolver: fatal signal backtrace\n";
  ssize_t ignored = write(STDERR_FILENO, heading, sizeof(heading) - 1);
  (void)ignored;
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  signal(signal_number, SIG_DFL);
  raise(signal_number);
}
static void install_trace_signal_handler_once(void) {
  if (trace_enabled()) {
    signal(SIGSEGV, trace_signal_handler);
    signal(SIGBUS, trace_signal_handler);
  }
}
static void install_trace_signal_handler(void) {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, install_trace_signal_handler_once);
}
__attribute__((constructor)) static void initialize_trace_handler(void) {
  install_trace_signal_handler();
}
#define TRACE_CALL(name, handle)                                                \
  do {                                                                          \
    install_trace_signal_handler();                                              \
    if (trace_enabled())                                                         \
      fprintf(stderr, "rgpu-cusolver: %s handle=%p route=%d\n", name,          \
              (void *)(handle), current_route());                               \
  } while (0)

static void *real_library;
static pthread_once_t real_once = PTHREAD_ONCE_INIT;
static void open_real_library(void) {
  real_library = dlopen(
      "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/"
      "libcusolver.so.12",
      RTLD_NOW | RTLD_LOCAL);
  if (real_library == 0)
    real_library = dlopen("libcusolver.so.12", RTLD_NOW | RTLD_LOCAL);
}
static void *real_symbol(const char *name) {
  pthread_once(&real_once, open_real_library);
  if (real_library == 0) return 0;
  void *symbol = dlvsym(real_library, name, "libcusolver.so.12");
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
static struct handle_entry *find(cusolver_handle handle) {
  for (size_t i = 0; i < 1024; ++i)
    if (handles[i].handle == handle) return &handles[i];
  return 0;
}
static int remote_call(const struct handle_entry *entry, struct request *request,
                       struct response *response) {
  typedef int (*get_context_function)(void **);
  static get_context_function get_context;
  if (get_context == 0)
    get_context = (get_context_function)dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
  void *context = 0;
  if (get_context != 0 && get_context(&context) == 0)
    request->context = (uint64_t)context;
  memset(response, 0, sizeof(*response));
  response->status = 7;
  if (entry == 0 || rpc() == 0) return 7;
  int rpc_status = rpc()(entry->route, request, response);
  if (trace_enabled())
    fprintf(stderr, "rgpu-cusolver: rpc opcode=%u transport=%d status=%d\n",
            request->opcode, rpc_status, response->status);
  return rpc_status != 0 ? 7 : response->status;
}

int cusolverDnCreate(cusolver_handle *handle) {
  typedef int (*real_function)(cusolver_handle *);
  int route = current_route();
  TRACE_CALL("cusolverDnCreate", 0);
  if (route < 0) {
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCreate");
    return real == 0 ? 7 : real(handle);
  }
  struct request request = {0};
  struct response response;
  struct handle_entry route_entry = {0, route};
  request.opcode = SOLVER_CREATE;
  int status = remote_call(&route_entry, &request, &response);
  if (status != 0) return status;
  *handle = (cusolver_handle)(uintptr_t)response.handle;
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < 1024; ++i) {
    if (handles[i].handle == 0) {
      handles[i] = (struct handle_entry){*handle, route};
      pthread_mutex_unlock(&mutex);
      return 0;
    }
  }
  pthread_mutex_unlock(&mutex);
  return 3;
}

int cusolverDnDestroy(cusolver_handle handle) {
  typedef int (*real_function)(cusolver_handle);
  TRACE_CALL("cusolverDnDestroy", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDestroy");
    return real == 0 ? 7 : real(handle);
  }
  struct handle_entry copy = *entry;
  entry->handle = 0;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_DESTROY;
  request.handle = (uint64_t)(uintptr_t)handle;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSetStream(cusolver_handle handle, void *stream) {
  typedef int (*real_function)(cusolver_handle, void *);
  TRACE_CALL("cusolverDnSetStream", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSetStream");
    return real == 0 ? 7 : real(handle, stream);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SET_STREAM;
  request.handle = (uint64_t)(uintptr_t)handle;
  /* Keep control RPCs on the route's ordered control lane.  request.stream is
   * reserved by the transport for selecting a stream lane; the server still
   * needs the opaque remote stream handle, so carry it in an otherwise unused
   * pointer slot. */
  request.d = (uint64_t)(uintptr_t)stream;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSgetrf_bufferSize(cusolver_handle handle, int m, int n, float *a,
                                int lda, int *workspace_elements) {
  typedef int (*real_function)(cusolver_handle, int, int, float *, int, int *);
  TRACE_CALL("cusolverDnSgetrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnSgetrf_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, a, lda, workspace_elements);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SGETRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && workspace_elements != 0)
    *workspace_elements = response.value;
  return status;
}

int cusolverDnSgetrf(cusolver_handle handle, int m, int n, float *a, int lda,
                     float *workspace, int *pivots, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, float *, int, float *,
                               int *, int *);
  TRACE_CALL("cusolverDnSgetrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSgetrf");
    return real == 0 ? 7 : real(handle, m, n, a, lda, workspace, pivots, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SGETRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda; request.workspace = (uint64_t)(uintptr_t)workspace;
  request.b = (uint64_t)(uintptr_t)pivots;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSgetrs(cusolver_handle handle, int transpose, int n, int nrhs,
                     const float *a, int lda, const int *pivots, float *b,
                     int ldb, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const float *,
                               int, const int *, float *, int, int *);
  TRACE_CALL("cusolverDnSgetrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSgetrs");
    return real == 0 ? 7
                     : real(handle, transpose, n, nrhs, a, lda, pivots, b, ldb,
                            info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SGETRS;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivots;
  request.c = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCgetrf_bufferSize(cusolver_handle handle, int m, int n, void *a,
                                int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, void *, int, int *);
  TRACE_CALL("cusolverDnCgetrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCgetrf_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CGETRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnCgetrf(cusolver_handle handle, int m, int n, void *a, int lda,
                     void *work, int *ipiv, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, void *, int, void *,
                               int *, int *);
  TRACE_CALL("cusolverDnCgetrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCgetrf");
    return real == 0 ? 7 : real(handle, m, n, a, lda, work, ipiv, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CGETRF; request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda; request.workspace = (uint64_t)(uintptr_t)work;
  request.b = (uint64_t)(uintptr_t)ipiv; request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCgetrs(cusolver_handle handle, int transpose, int n, int nrhs,
                     const void *a, int lda, const int *ipiv, void *b, int ldb,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const void *,
                               int, const int *, void *, int, int *);
  TRACE_CALL("cusolverDnCgetrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCgetrs");
    return real == 0 ? 7
                     : real(handle, transpose, n, nrhs, a, lda, ipiv, b, ldb,
                            info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CGETRS; request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)ipiv; request.c = (uint64_t)(uintptr_t)b;
  request.ldb = ldb; request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZgetrf_bufferSize(cusolver_handle handle, int m, int n, void *a,
                                int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, void *, int, int *);
  TRACE_CALL("cusolverDnZgetrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZgetrf_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZGETRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnZgetrf(cusolver_handle handle, int m, int n, void *a, int lda,
                     void *work, int *ipiv, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, void *, int, void *,
                               int *, int *);
  TRACE_CALL("cusolverDnZgetrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnZgetrf");
    return real == 0 ? 7 : real(handle, m, n, a, lda, work, ipiv, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZGETRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda; request.workspace = (uint64_t)(uintptr_t)work;
  request.b = (uint64_t)(uintptr_t)ipiv; request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZgetrs(cusolver_handle handle, int transpose, int n, int nrhs,
                     const void *a, int lda, const int *ipiv, void *b, int ldb,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const void *,
                               int, const int *, void *, int, int *);
  TRACE_CALL("cusolverDnZgetrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnZgetrs");
    return real == 0 ? 7
                     : real(handle, transpose, n, nrhs, a, lda, ipiv, b, ldb,
                            info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZGETRS;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)ipiv; request.c = (uint64_t)(uintptr_t)b;
  request.ldb = ldb; request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDgetrf_bufferSize(cusolver_handle handle, int m, int n,
                                double *a, int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, double *, int,
                               int *);
  TRACE_CALL("cusolverDnDgetrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDgetrf_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGETRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDgetrf(cusolver_handle handle, int m, int n, double *a,
                     int lda, double *work, int *ipiv, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, double *, int,
                               double *, int *, int *);
  TRACE_CALL("cusolverDnDgetrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDgetrf");
    return real == 0 ? 7 : real(handle, m, n, a, lda, work, ipiv, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGETRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda; request.workspace = (uint64_t)(uintptr_t)work;
  request.b = (uint64_t)(uintptr_t)ipiv; request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDgetrs(cusolver_handle handle, int transpose, int n, int nrhs,
                     const double *a, int lda, const int *ipiv, double *b,
                     int ldb, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int,
                               const double *, int, const int *, double *, int,
                               int *);
  TRACE_CALL("cusolverDnDgetrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDgetrs");
    return real == 0 ? 7
                     : real(handle, transpose, n, nrhs, a, lda, ipiv, b, ldb,
                            info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGETRS;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)ipiv; request.c = (uint64_t)(uintptr_t)b;
  request.ldb = ldb; request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCreateGesvdjInfo(cusolver_handle *info) {
  typedef int (*real_function)(cusolver_handle *);
  int route = current_route();
  TRACE_CALL("cusolverDnCreateGesvdjInfo", 0);
  if (route < 0) {
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCreateGesvdjInfo");
    return real == 0 ? 7 : real(info);
  }
  struct request request = {0};
  struct response response;
  struct handle_entry route_entry = {0, route};
  request.opcode = SOLVER_CREATE_GESVDJ_INFO;
  int status = remote_call(&route_entry, &request, &response);
  if (status != 0) return status;
  *info = (cusolver_handle)(uintptr_t)response.handle;
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < 1024; ++i) {
    if (handles[i].handle == 0) {
      handles[i] = (struct handle_entry){*info, route};
      pthread_mutex_unlock(&mutex);
      return 0;
    }
  }
  pthread_mutex_unlock(&mutex);
  return 3;
}

int cusolverDnDestroyGesvdjInfo(cusolver_handle info) {
  typedef int (*real_function)(cusolver_handle);
  TRACE_CALL("cusolverDnDestroyGesvdjInfo", info);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(info);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDestroyGesvdjInfo");
    return real == 0 ? 7 : real(info);
  }
  struct handle_entry copy = *entry;
  entry->handle = 0;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_DESTROY_GESVDJ_INFO;
  request.handle = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXgesvdjSetTolerance(cusolver_handle info, double tolerance) {
  typedef int (*real_function)(cusolver_handle, double);
  TRACE_CALL("cusolverDnXgesvdjSetTolerance", info);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(info);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXgesvdjSetTolerance");
    return real == 0 ? 7 : real(info, tolerance);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_GESVDJ_SET_TOLERANCE;
  request.handle = (uint64_t)(uintptr_t)info;
  memcpy(request.payload, &tolerance, sizeof(tolerance));
  request.payload_size = sizeof(tolerance);
  return remote_call(&copy, &request, &response);
}

int cusolverDnXgesvdjSetMaxSweeps(cusolver_handle info, int max_sweeps) {
  typedef int (*real_function)(cusolver_handle, int);
  TRACE_CALL("cusolverDnXgesvdjSetMaxSweeps", info);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(info);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXgesvdjSetMaxSweeps");
    return real == 0 ? 7 : real(info, max_sweeps);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_GESVDJ_SET_MAX_SWEEPS;
  request.handle = (uint64_t)(uintptr_t)info;
  request.value = max_sweeps;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXgesvdjSetSortEig(cusolver_handle info, int sort_eig) {
  typedef int (*real_function)(cusolver_handle, int);
  TRACE_CALL("cusolverDnXgesvdjSetSortEig", info);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(info);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXgesvdjSetSortEig");
    return real == 0 ? 7 : real(info, sort_eig);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_GESVDJ_SET_SORT_EIG;
  request.handle = (uint64_t)(uintptr_t)info;
  request.value = sort_eig;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSgesvdjBatched_bufferSize(
    cusolver_handle handle, int jobz, int m, int n, const float *a, int lda,
    const float *s, const float *u, int ldu, const float *v, int ldv,
    int *lwork, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const float *,
                               int, const float *, const float *, int,
                               const float *, int, int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnSgesvdjBatched_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnSgesvdjBatched_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            lwork, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SGESVDJ_BATCHED_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnSgesvdjBatched(
    cusolver_handle handle, int jobz, int m, int n, float *a, int lda,
    float *s, float *u, int ldu, float *v, int ldv, float *work, int lwork,
    int *info, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, float *, int,
                               float *, float *, int, float *, int, float *,
                               int, int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnSgesvdjBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnSgesvdjBatched");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            work, lwork, info, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SGESVDJ_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDgesvdj_bufferSize(
    cusolver_handle handle, int jobz, int econ, int m, int n, const double *a,
    int lda, const double *s, const double *u, int ldu, const double *v,
    int ldv, int *lwork, cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int,
                               const double *, int, const double *,
                               const double *, int, const double *, int, int *,
                               cusolver_handle);
  TRACE_CALL("cusolverDnDgesvdj_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDgesvdj_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, econ, m, n, a, lda, s, u, ldu, v,
                            ldv, lwork, params);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGESVDJ_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = econ; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDgesvdj(
    cusolver_handle handle, int jobz, int econ, int m, int n, double *a,
    int lda, double *s, double *u, int ldu, double *v, int ldv, double *work,
    int lwork, int *info, cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, double *,
                               int, double *, double *, int, double *, int,
                               double *, int, int *, cusolver_handle);
  TRACE_CALL("cusolverDnDgesvdj", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDgesvdj");
    return real == 0 ? 7
                     : real(handle, jobz, econ, m, n, a, lda, s, u, ldu, v,
                            ldv, work, lwork, info, params);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGESVDJ;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = econ; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDgesvdjBatched_bufferSize(
    cusolver_handle handle, int jobz, int m, int n, const double *a, int lda,
    const double *s, const double *u, int ldu, const double *v, int ldv,
    int *lwork, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const double *,
                               int, const double *, const double *, int,
                               const double *, int, int *, cusolver_handle,
                               int);
  TRACE_CALL("cusolverDnDgesvdjBatched_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDgesvdjBatched_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            lwork, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGESVDJ_BATCHED_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDgesvdjBatched(
    cusolver_handle handle, int jobz, int m, int n, double *a, int lda,
    double *s, double *u, int ldu, double *v, int ldv, double *work,
    int lwork, int *info, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, double *, int,
                               double *, double *, int, double *, int,
                               double *, int, int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnDgesvdjBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDgesvdjBatched");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            work, lwork, info, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGESVDJ_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCgesvdj_bufferSize(
    cusolver_handle handle, int jobz, int econ, int m, int n, const void *a,
    int lda, const float *s, const void *u, int ldu, const void *v, int ldv,
    int *lwork, cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int,
                               const void *, int, const float *, const void *,
                               int, const void *, int, int *, cusolver_handle);
  TRACE_CALL("cusolverDnCgesvdj_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCgesvdj_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, econ, m, n, a, lda, s, u, ldu, v,
                            ldv, lwork, params);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_CGESVDJ_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = econ;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnCgesvdj(
    cusolver_handle handle, int jobz, int econ, int m, int n, void *a,
    int lda, float *s, void *u, int ldu, void *v, int ldv, void *work,
    int lwork, int *info, cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, void *,
                               int, float *, void *, int, void *, int, void *,
                               int, int *, cusolver_handle);
  TRACE_CALL("cusolverDnCgesvdj", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCgesvdj");
    return real == 0 ? 7
                     : real(handle, jobz, econ, m, n, a, lda, s, u, ldu, v,
                            ldv, work, lwork, info, params);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_CGESVDJ;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = econ;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCgesvdjBatched_bufferSize(
    cusolver_handle handle, int jobz, int m, int n, const void *a, int lda,
    const float *s, const void *u, int ldu, const void *v, int ldv,
    int *lwork, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const void *,
                               int, const float *, const void *, int,
                               const void *, int, int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnCgesvdjBatched_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCgesvdjBatched_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            lwork, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_CGESVDJ_BATCHED_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnCgesvdjBatched(
    cusolver_handle handle, int jobz, int m, int n, void *a, int lda,
    float *s, void *u, int ldu, void *v, int ldv, void *work, int lwork,
    int *info, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, void *, int,
                               float *, void *, int, void *, int, void *, int,
                               int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnCgesvdjBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCgesvdjBatched");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            work, lwork, info, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_CGESVDJ_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZgesvdj_bufferSize(
    cusolver_handle handle, int jobz, int econ, int m, int n, const void *a,
    int lda, const double *s, const void *u, int ldu, const void *v, int ldv,
    int *lwork, cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int,
                               const void *, int, const double *, const void *,
                               int, const void *, int, int *, cusolver_handle);
  TRACE_CALL("cusolverDnZgesvdj_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZgesvdj_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, econ, m, n, a, lda, s, u, ldu, v,
                            ldv, lwork, params);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_ZGESVDJ_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = econ;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnZgesvdj(
    cusolver_handle handle, int jobz, int econ, int m, int n, void *a,
    int lda, double *s, void *u, int ldu, void *v, int ldv, void *work,
    int lwork, int *info, cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, void *,
                               int, double *, void *, int, void *, int, void *,
                               int, int *, cusolver_handle);
  TRACE_CALL("cusolverDnZgesvdj", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnZgesvdj");
    return real == 0 ? 7
                     : real(handle, jobz, econ, m, n, a, lda, s, u, ldu, v,
                            ldv, work, lwork, info, params);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_ZGESVDJ;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = econ;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZgesvdjBatched_bufferSize(
    cusolver_handle handle, int jobz, int m, int n, const void *a, int lda,
    const double *s, const void *u, int ldu, const void *v, int ldv,
    int *lwork, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const void *,
                               int, const double *, const void *, int,
                               const void *, int, int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnZgesvdjBatched_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZgesvdjBatched_bufferSize");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            lwork, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_ZGESVDJ_BATCHED_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnZgesvdjBatched(
    cusolver_handle handle, int jobz, int m, int n, void *a, int lda,
    double *s, void *u, int ldu, void *v, int ldv, void *work, int lwork,
    int *info, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, void *, int,
                               double *, void *, int, void *, int, void *, int,
                               int *, cusolver_handle, int);
  TRACE_CALL("cusolverDnZgesvdjBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZgesvdjBatched");
    return real == 0 ? 7
                     : real(handle, jobz, m, n, a, lda, s, u, ldu, v, ldv,
                            work, lwork, info, params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_ZGESVDJ_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)v; request.ldc = ldv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.preference = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCreateParams(cusolver_handle *params) {
  typedef int (*real_function)(cusolver_handle *);
  int route = current_route();
  TRACE_CALL("cusolverDnCreateParams", 0);
  if (route < 0) {
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCreateParams");
    return real == 0 ? 7 : real(params);
  }
  struct request request = {0};
  struct response response;
  struct handle_entry route_entry = {0, route};
  request.opcode = SOLVER_CREATE_PARAMS;
  int status = remote_call(&route_entry, &request, &response);
  if (status != 0) return status;
  *params = (cusolver_handle)(uintptr_t)response.handle;
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < 1024; ++i) {
    if (handles[i].handle == 0) {
      handles[i] = (struct handle_entry){*params, route};
      pthread_mutex_unlock(&mutex);
      return 0;
    }
  }
  pthread_mutex_unlock(&mutex);
  return 3;
}

int cusolverDnDestroyParams(cusolver_handle params) {
  typedef int (*real_function)(cusolver_handle);
  TRACE_CALL("cusolverDnDestroyParams", params);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(params);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDestroyParams");
    return real == 0 ? 7 : real(params);
  }
  struct handle_entry copy = *entry;
  entry->handle = 0;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_DESTROY_PARAMS;
  request.handle = (uint64_t)(uintptr_t)params;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSetAdvOptions(cusolver_handle params, int function, int algo) {
  typedef int (*real_function)(cusolver_handle, int, int);
  TRACE_CALL("cusolverDnSetAdvOptions", params);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(params);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSetAdvOptions");
    return real == 0 ? 7 : real(params, function, algo);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SET_ADV_OPTIONS;
  request.handle = (uint64_t)(uintptr_t)params;
  request.attribute = function;
  request.algorithm = algo;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXpotrf_bufferSize(
    cusolver_handle handle, cusolver_handle params, int uplo, int64_t n,
    int data_type_a, const void *a, int64_t lda, int compute_type,
    size_t *device_bytes, size_t *host_bytes) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int64_t,
                               int, const void *, int64_t, int, size_t *,
                               size_t *);
  TRACE_CALL("cusolverDnXpotrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXpotrf_bufferSize");
    return real == 0 ? 7
                     : real(handle, params, uplo, n, data_type_a, a, lda,
                            compute_type, device_bytes, host_bytes);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XPOTRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = uplo; request.rows = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda; request.compute_type = compute_type;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && response.payload_size >= 2 * sizeof(uint64_t)) {
    uint64_t sizes[2];
    memcpy(sizes, response.payload, sizeof(sizes));
    if (device_bytes != 0) *device_bytes = (size_t)sizes[0];
    if (host_bytes != 0) *host_bytes = (size_t)sizes[1];
  }
  return status;
}

int cusolverDnXpotrf(
    cusolver_handle handle, cusolver_handle params, int uplo, int64_t n,
    int data_type_a, void *a, int64_t lda, int compute_type,
    void *device_buffer, size_t device_bytes, void *host_buffer,
    size_t host_bytes, int *info) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int64_t,
                               int, void *, int64_t, int, void *, size_t,
                               void *, size_t, int *);
  TRACE_CALL("cusolverDnXpotrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXpotrf");
    return real == 0 ? 7
                     : real(handle, params, uplo, n, data_type_a, a, lda,
                            compute_type, device_buffer, device_bytes,
                            host_buffer, host_bytes, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XPOTRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = uplo; request.rows = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda; request.compute_type = compute_type;
  request.workspace = (uint64_t)(uintptr_t)device_buffer;
  request.workspace_size = (uint64_t)device_bytes;
  request.columns = (uint64_t)host_bytes;
  request.b = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXpotrs(cusolver_handle handle, cusolver_handle params, int uplo,
                     int64_t n, int64_t nrhs, int data_type_a, const void *a,
                     int64_t lda, int data_type_b, void *b, int64_t ldb,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int64_t,
                               int64_t, int, const void *, int64_t, int, void *,
                               int64_t, int *);
  TRACE_CALL("cusolverDnXpotrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXpotrs");
    return real == 0 ? 7
                     : real(handle, params, uplo, n, nrhs, data_type_a, a, lda,
                            data_type_b, b, ldb, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_XPOTRS;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = uplo; request.rows = (uint64_t)n;
  request.columns = (uint64_t)nrhs;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_b; request.b = (uint64_t)(uintptr_t)b;
  request.stride_a = ldb; request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXgeqrf_bufferSize(
    cusolver_handle handle, cusolver_handle params, int64_t m, int64_t n,
    int data_type_a, const void *a, int64_t lda, int data_type_tau,
    const void *tau, int compute_type, size_t *device_bytes,
    size_t *host_bytes) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int64_t,
                               int64_t, int, const void *, int64_t, int,
                               const void *, int, size_t *, size_t *);
  TRACE_CALL("cusolverDnXgeqrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXgeqrf_bufferSize");
    return real == 0 ? 7
                     : real(handle, params, m, n, data_type_a, a, lda,
                            data_type_tau, tau, compute_type, device_bytes,
                            host_bytes);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XGEQRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.rows = (uint64_t)m; request.columns = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_tau; request.b = (uint64_t)(uintptr_t)tau;
  request.compute_type = compute_type;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && response.payload_size >= 2 * sizeof(uint64_t)) {
    uint64_t sizes[2];
    memcpy(sizes, response.payload, sizeof(sizes));
    if (device_bytes != 0) *device_bytes = (size_t)sizes[0];
    if (host_bytes != 0) *host_bytes = (size_t)sizes[1];
  }
  return status;
}

int cusolverDnXgeqrf(
    cusolver_handle handle, cusolver_handle params, int64_t m, int64_t n,
    int data_type_a, void *a, int64_t lda, int data_type_tau, void *tau,
    int compute_type, void *device_buffer, size_t device_bytes,
    void *host_buffer, size_t host_bytes, int *info) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int64_t,
                               int64_t, int, void *, int64_t, int, void *, int,
                               void *, size_t, void *, size_t, int *);
  TRACE_CALL("cusolverDnXgeqrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXgeqrf");
    return real == 0 ? 7
                     : real(handle, params, m, n, data_type_a, a, lda,
                            data_type_tau, tau, compute_type, device_buffer,
                            device_bytes, host_buffer, host_bytes, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XGEQRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.rows = (uint64_t)m; request.columns = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_tau; request.b = (uint64_t)(uintptr_t)tau;
  request.compute_type = compute_type;
  request.workspace = (uint64_t)(uintptr_t)device_buffer;
  request.workspace_size = (uint64_t)device_bytes;
  request.stride_a = (int64_t)host_bytes;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSsytrf_bufferSize(cusolver_handle handle, int n, float *a,
                                int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, float *, int, int *);
  TRACE_CALL("cusolverDnSsytrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnSsytrf_bufferSize");
    return real == 0 ? 7 : real(handle, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SSYTRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnSsytrf(cusolver_handle handle, int uplo, int n, float *a,
                     int lda, int *ipiv, float *work, int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, float *, int, int *,
                               float *, int, int *);
  TRACE_CALL("cusolverDnSsytrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSsytrf");
    return real == 0 ? 7
                     : real(handle, uplo, n, a, lda, ipiv, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SSYTRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)ipiv;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZsytrf_bufferSize(cusolver_handle handle, int n, void *a,
                                int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, void *, int, int *);
  TRACE_CALL("cusolverDnZsytrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZsytrf_bufferSize");
    return real == 0 ? 7 : real(handle, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_ZSYTRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnZsytrf(cusolver_handle handle, int uplo, int n, void *a,
                     int lda, int *ipiv, void *work, int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, void *, int, int *,
                               void *, int, int *);
  TRACE_CALL("cusolverDnZsytrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnZsytrf");
    return real == 0
               ? 7
               : real(handle, uplo, n, a, lda, ipiv, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_ZSYTRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  request.b = (uint64_t)(uintptr_t)ipiv;
  request.workspace = (uint64_t)(uintptr_t)work;
  request.value = lwork;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

#define DEFINE_SYTRF_RPC(PREFIX, TYPE, BUFFER_OPCODE, COMPUTE_OPCODE)           \
  int cusolverDn##PREFIX##sytrf_bufferSize(cusolver_handle handle, int n,      \
                                            TYPE *a, int lda, int *lwork) {     \
    typedef int (*real_function)(cusolver_handle, int, TYPE *, int, int *);    \
    TRACE_CALL("cusolverDn" #PREFIX "sytrf_bufferSize", handle);              \
    pthread_mutex_lock(&mutex);                                                 \
    struct handle_entry *entry = find(handle);                                  \
    if (entry == 0) {                                                           \
      pthread_mutex_unlock(&mutex);                                             \
      static real_function real;                                                \
      if (real == 0)                                                            \
        real = (real_function)real_symbol(                                      \
            "cusolverDn" #PREFIX "sytrf_bufferSize");                         \
      return real == 0 ? 7 : real(handle, n, a, lda, lwork);                   \
    }                                                                           \
    struct handle_entry copy = *entry;                                          \
    pthread_mutex_unlock(&mutex);                                               \
    struct request request = {0};                                               \
    struct response response;                                                   \
    request.opcode = BUFFER_OPCODE;                                             \
    request.handle = (uint64_t)(uintptr_t)handle;                               \
    request.n = n;                                                              \
    request.a = (uint64_t)(uintptr_t)a;                                         \
    request.lda = lda;                                                          \
    int status = remote_call(&copy, &request, &response);                       \
    if (status == 0 && lwork != 0) *lwork = response.value;                    \
    return status;                                                              \
  }                                                                             \
  int cusolverDn##PREFIX##sytrf(cusolver_handle handle, int uplo, int n,       \
                                 TYPE *a, int lda, int *ipiv, TYPE *work,       \
                                 int lwork, int *info) {                        \
    typedef int (*real_function)(cusolver_handle, int, int, TYPE *, int, int *,\
                                 TYPE *, int, int *);                           \
    TRACE_CALL("cusolverDn" #PREFIX "sytrf", handle);                         \
    pthread_mutex_lock(&mutex);                                                 \
    struct handle_entry *entry = find(handle);                                  \
    if (entry == 0) {                                                           \
      pthread_mutex_unlock(&mutex);                                             \
      static real_function real;                                                \
      if (real == 0)                                                            \
        real = (real_function)real_symbol("cusolverDn" #PREFIX "sytrf");      \
      return real == 0                                                          \
                 ? 7                                                           \
                 : real(handle, uplo, n, a, lda, ipiv, work, lwork, info);     \
    }                                                                           \
    struct handle_entry copy = *entry;                                          \
    pthread_mutex_unlock(&mutex);                                               \
    struct request request = {0};                                               \
    struct response response;                                                   \
    request.opcode = COMPUTE_OPCODE;                                            \
    request.handle = (uint64_t)(uintptr_t)handle;                               \
    request.transa = uplo;                                                      \
    request.n = n;                                                              \
    request.a = (uint64_t)(uintptr_t)a;                                         \
    request.lda = lda;                                                          \
    request.b = (uint64_t)(uintptr_t)ipiv;                                      \
    request.workspace = (uint64_t)(uintptr_t)work;                              \
    request.value = lwork;                                                      \
    request.d = (uint64_t)(uintptr_t)info;                                      \
    return remote_call(&copy, &request, &response);                             \
  }

DEFINE_SYTRF_RPC(C, void, SOLVER_CSYTRF_BUFFER_SIZE, SOLVER_CSYTRF)
DEFINE_SYTRF_RPC(D, double, SOLVER_DSYTRF_BUFFER_SIZE, SOLVER_DSYTRF)

#undef DEFINE_SYTRF_RPC

int cusolverDnXsytrs_bufferSize(
    cusolver_handle handle, int uplo, int64_t n, int64_t nrhs,
    int data_type_a, const void *a, int64_t lda, const int64_t *ipiv,
    int data_type_b, void *b, int64_t ldb, size_t *device_bytes,
    size_t *host_bytes) {
  typedef int (*real_function)(cusolver_handle, int, int64_t, int64_t, int,
                               const void *, int64_t, const int64_t *, int,
                               void *, int64_t, size_t *, size_t *);
  TRACE_CALL("cusolverDnXsytrs_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXsytrs_bufferSize");
    return real == 0 ? 7
                     : real(handle, uplo, n, nrhs, data_type_a, a, lda, ipiv,
                            data_type_b, b, ldb, device_bytes, host_bytes);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XSYTRS_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.rows = (uint64_t)n;
  request.columns = (uint64_t)nrhs; request.a_type = data_type_a;
  request.a = (uint64_t)(uintptr_t)a; request.leading_dimension = lda;
  request.b = (uint64_t)(uintptr_t)ipiv; request.b_type = data_type_b;
  request.c = (uint64_t)(uintptr_t)b; request.stride_a = ldb;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && response.payload_size >= 2 * sizeof(uint64_t)) {
    uint64_t sizes[2];
    memcpy(sizes, response.payload, sizeof(sizes));
    if (device_bytes != 0) *device_bytes = (size_t)sizes[0];
    if (host_bytes != 0) *host_bytes = (size_t)sizes[1];
  }
  return status;
}

int cusolverDnXsytrs(
    cusolver_handle handle, int uplo, int64_t n, int64_t nrhs,
    int data_type_a, const void *a, int64_t lda, const int64_t *ipiv,
    int data_type_b, void *b, int64_t ldb, void *device_buffer,
    size_t device_bytes, void *host_buffer, size_t host_bytes, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int64_t, int64_t, int,
                               const void *, int64_t, const int64_t *, int,
                               void *, int64_t, void *, size_t, void *, size_t,
                               int *);
  TRACE_CALL("cusolverDnXsytrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXsytrs");
    return real == 0 ? 7
                     : real(handle, uplo, n, nrhs, data_type_a, a, lda, ipiv,
                            data_type_b, b, ldb, device_buffer, device_bytes,
                            host_buffer, host_bytes, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XSYTRS;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.rows = (uint64_t)n;
  request.columns = (uint64_t)nrhs; request.a_type = data_type_a;
  request.a = (uint64_t)(uintptr_t)a; request.leading_dimension = lda;
  request.b = (uint64_t)(uintptr_t)ipiv; request.b_type = data_type_b;
  request.c = (uint64_t)(uintptr_t)b; request.stride_a = ldb;
  request.workspace = (uint64_t)(uintptr_t)device_buffer;
  request.workspace_size = (uint64_t)device_bytes;
  request.descriptor = (uint64_t)host_bytes;
  request.preference = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXgeev_bufferSize(
    cusolver_handle handle, cusolver_handle params, int jobvl, int jobvr,
    int64_t n, int data_type_a, const void *a, int64_t lda, int data_type_w,
    const void *w, int data_type_vl, const void *vl, int64_t ldvl,
    int data_type_vr, const void *vr, int64_t ldvr, int compute_type,
    size_t *device_bytes, size_t *host_bytes) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int,
                               int64_t, int, const void *, int64_t, int,
                               const void *, int, const void *, int64_t, int,
                               const void *, int64_t, int, size_t *, size_t *);
  TRACE_CALL("cusolverDnXgeev_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXgeev_bufferSize");
    return real == 0 ? 7
                     : real(handle, params, jobvl, jobvr, n, data_type_a, a,
                            lda, data_type_w, w, data_type_vl, vl, ldvl,
                            data_type_vr, vr, ldvr, compute_type, device_bytes,
                            host_bytes);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XGEEV_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = jobvl; request.transb = jobvr; request.rows = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_w; request.b = (uint64_t)(uintptr_t)w;
  request.c_type = data_type_vl; request.c = (uint64_t)(uintptr_t)vl;
  request.stride_a = ldvl;
  request.attribute = data_type_vr; request.d = (uint64_t)(uintptr_t)vr;
  request.stride_b = ldvr; request.compute_type = compute_type;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && response.payload_size >= 2 * sizeof(uint64_t)) {
    uint64_t sizes[2];
    memcpy(sizes, response.payload, sizeof(sizes));
    if (device_bytes != 0) *device_bytes = (size_t)sizes[0];
    if (host_bytes != 0) *host_bytes = (size_t)sizes[1];
  }
  return status;
}

int cusolverDnXgeev(
    cusolver_handle handle, cusolver_handle params, int jobvl, int jobvr,
    int64_t n, int data_type_a, void *a, int64_t lda, int data_type_w, void *w,
    int data_type_vl, void *vl, int64_t ldvl, int data_type_vr, void *vr,
    int64_t ldvr, int compute_type, void *device_buffer, size_t device_bytes,
    void *host_buffer, size_t host_bytes, int *info) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int,
                               int64_t, int, void *, int64_t, int, void *, int,
                               void *, int64_t, int, void *, int64_t, int,
                               void *, size_t, void *, size_t, int *);
  TRACE_CALL("cusolverDnXgeev", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXgeev");
    return real == 0 ? 7
                     : real(handle, params, jobvl, jobvr, n, data_type_a, a,
                            lda, data_type_w, w, data_type_vl, vl, ldvl,
                            data_type_vr, vr, ldvr, compute_type, device_buffer,
                            device_bytes, host_buffer, host_bytes, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XGEEV;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = jobvl; request.transb = jobvr; request.rows = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_w; request.b = (uint64_t)(uintptr_t)w;
  request.c_type = data_type_vl; request.c = (uint64_t)(uintptr_t)vl;
  request.stride_a = ldvl;
  request.attribute = data_type_vr; request.d = (uint64_t)(uintptr_t)vr;
  request.stride_b = ldvr; request.compute_type = compute_type;
  request.workspace = (uint64_t)(uintptr_t)device_buffer;
  request.workspace_size = (uint64_t)device_bytes;
  request.columns = (uint64_t)host_bytes;
  request.preference = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDorgqr_bufferSize(cusolver_handle handle, int m, int n, int k,
                                const double *a, int lda, const double *tau,
                                int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const double *,
                               int, const double *, int *);
  TRACE_CALL("cusolverDnDorgqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDorgqr_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, k, a, lda, tau, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DORGQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDorgqr(cusolver_handle handle, int m, int n, int k, double *a,
                     int lda, const double *tau, double *work, int lwork,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, double *, int,
                               const double *, double *, int, int *);
  TRACE_CALL("cusolverDnDorgqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDorgqr");
    return real == 0 ? 7
                     : real(handle, m, n, k, a, lda, tau, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DORGQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSorgqr_bufferSize(cusolver_handle handle, int m, int n, int k,
                                const float *a, int lda, const float *tau,
                                int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const float *,
                               int, const float *, int *);
  TRACE_CALL("cusolverDnSorgqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnSorgqr_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, k, a, lda, tau, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SORGQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnSorgqr(cusolver_handle handle, int m, int n, int k, float *a,
                     int lda, const float *tau, float *work, int lwork,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, float *, int,
                               const float *, float *, int, int *);
  TRACE_CALL("cusolverDnSorgqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSorgqr");
    return real == 0 ? 7
                     : real(handle, m, n, k, a, lda, tau, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SORGQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCungqr_bufferSize(cusolver_handle handle, int m, int n, int k,
                                const void *a, int lda, const void *tau,
                                int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const void *,
                               int, const void *, int *);
  TRACE_CALL("cusolverDnCungqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCungqr_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, k, a, lda, tau, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CUNGQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnCungqr(cusolver_handle handle, int m, int n, int k, void *a,
                     int lda, const void *tau, void *work, int lwork,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, void *, int,
                               const void *, void *, int, int *);
  TRACE_CALL("cusolverDnCungqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCungqr");
    return real == 0 ? 7
                     : real(handle, m, n, k, a, lda, tau, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CUNGQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZungqr_bufferSize(cusolver_handle handle, int m, int n, int k,
                                const void *a, int lda, const void *tau,
                                int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, const void *,
                               int, const void *, int *);
  TRACE_CALL("cusolverDnZungqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZungqr_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, k, a, lda, tau, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZUNGQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnZungqr(cusolver_handle handle, int m, int n, int k, void *a,
                     int lda, const void *tau, void *work, int lwork,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, void *, int,
                               const void *, void *, int, int *);
  TRACE_CALL("cusolverDnZungqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnZungqr");
    return real == 0 ? 7
                     : real(handle, m, n, k, a, lda, tau, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZUNGQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnXsyevBatched_bufferSize(
    cusolver_handle handle, cusolver_handle params, int jobz, int uplo,
    int64_t n, int data_type_a, const void *a, int64_t lda, int data_type_w,
    const void *w, int compute_type, size_t *device_bytes, size_t *host_bytes,
    int64_t batch_size) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int,
                               int64_t, int, const void *, int64_t, int,
                               const void *, int, size_t *, size_t *, int64_t);
  TRACE_CALL("cusolverDnXsyevBatched_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnXsyevBatched_bufferSize");
    return real == 0 ? 7
                     : real(handle, params, jobz, uplo, n, data_type_a, a, lda,
                            data_type_w, w, compute_type, device_bytes,
                            host_bytes, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XSYEV_BATCHED_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = jobz; request.transb = uplo; request.rows = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_w; request.b = (uint64_t)(uintptr_t)w;
  request.compute_type = compute_type; request.stride_b = batch_size;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && response.payload_size >= 2 * sizeof(uint64_t)) {
    uint64_t sizes[2];
    memcpy(sizes, response.payload, sizeof(sizes));
    if (device_bytes != 0) *device_bytes = (size_t)sizes[0];
    if (host_bytes != 0) *host_bytes = (size_t)sizes[1];
  }
  return status;
}

int cusolverDnXsyevBatched(
    cusolver_handle handle, cusolver_handle params, int jobz, int uplo,
    int64_t n, int data_type_a, void *a, int64_t lda, int data_type_w, void *w,
    int compute_type, void *device_buffer, size_t device_bytes,
    void *host_buffer, size_t host_bytes, int *info, int64_t batch_size) {
  typedef int (*real_function)(cusolver_handle, cusolver_handle, int, int,
                               int64_t, int, void *, int64_t, int, void *, int,
                               void *, size_t, void *, size_t, int *, int64_t);
  TRACE_CALL("cusolverDnXsyevBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnXsyevBatched");
    return real == 0 ? 7
                     : real(handle, params, jobz, uplo, n, data_type_a, a, lda,
                            data_type_w, w, compute_type, device_buffer,
                            device_bytes, host_buffer, host_bytes, info,
                            batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_XSYEV_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.transa = jobz; request.transb = uplo; request.rows = (uint64_t)n;
  request.a_type = data_type_a; request.a = (uint64_t)(uintptr_t)a;
  request.leading_dimension = lda;
  request.b_type = data_type_w; request.b = (uint64_t)(uintptr_t)w;
  request.compute_type = compute_type;
  request.workspace = (uint64_t)(uintptr_t)device_buffer;
  request.workspace_size = (uint64_t)device_bytes;
  request.stride_a = (int64_t)host_bytes;
  request.c = (uint64_t)(uintptr_t)info;
  request.stride_b = batch_size;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDormqr_bufferSize(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const double *a, int lda, const double *tau, const double *c, int ldc,
    int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const double *, int, const double *,
                               const double *, int, int *);
  TRACE_CALL("cusolverDnDormqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDormqr_bufferSize");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DORMQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDormqr(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const double *a, int lda, const double *tau, double *c, int ldc,
    double *work, int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const double *, int, const double *, double *,
                               int, double *, int, int *);
  TRACE_CALL("cusolverDnDormqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDormqr");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DORMQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnSormqr_bufferSize(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const float *a, int lda, const float *tau, const float *c, int ldc,
    int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const float *, int, const float *, const float *,
                               int, int *);
  TRACE_CALL("cusolverDnSormqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnSormqr_bufferSize");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SORMQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnSormqr(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const float *a, int lda, const float *tau, float *c, int ldc, float *work,
    int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const float *, int, const float *, float *, int,
                               float *, int, int *);
  TRACE_CALL("cusolverDnSormqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnSormqr");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SOLVER_SORMQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnCunmqr_bufferSize(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const void *a, int lda, const void *tau, const void *c, int ldc,
    int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const void *, int, const void *, const void *,
                               int, int *);
  TRACE_CALL("cusolverDnCunmqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnCunmqr_bufferSize");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CUNMQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnCunmqr(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const void *a, int lda, const void *tau, void *c, int ldc, void *work,
    int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const void *, int, const void *, void *, int,
                               void *, int, int *);
  TRACE_CALL("cusolverDnCunmqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnCunmqr");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_CUNMQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnZunmqr_bufferSize(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const void *a, int lda, const void *tau, const void *c, int ldc,
    int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const void *, int, const void *, const void *,
                               int, int *);
  TRACE_CALL("cusolverDnZunmqr_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnZunmqr_bufferSize");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZUNMQR_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnZunmqr(
    cusolver_handle handle, int side, int transpose, int m, int n, int k,
    const void *a, int lda, const void *tau, void *c, int ldc, void *work,
    int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, int, int,
                               const void *, int, const void *, void *, int,
                               void *, int, int *);
  TRACE_CALL("cusolverDnZunmqr", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnZunmqr");
    return real == 0
               ? 7
               : real(handle, side, transpose, m, n, k, a, lda, tau, c, ldc,
                      work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_ZUNMQR;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.transa = transpose;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.d = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDpotrf_bufferSize(cusolver_handle handle, int uplo, int n,
                                double *a, int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, double *, int,
                               int *);
  TRACE_CALL("cusolverDnDpotrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDpotrf_bufferSize");
    return real == 0 ? 7 : real(handle, uplo, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DPOTRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDpotrf(cusolver_handle handle, int uplo, int n, double *a,
                     int lda, double *work, int lwork, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, double *, int,
                               double *, int, int *);
  TRACE_CALL("cusolverDnDpotrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDpotrf");
    return real == 0 ? 7 : real(handle, uplo, n, a, lda, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DPOTRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.b = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDpotrs(cusolver_handle handle, int uplo, int n, int nrhs,
                     const double *a, int lda, double *b, int ldb, int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int,
                               const double *, int, double *, int, int *);
  TRACE_CALL("cusolverDnDpotrs", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDpotrs");
    return real == 0 ? 7 : real(handle, uplo, n, nrhs, a, lda, b, ldb, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DPOTRS;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDgeqrf_bufferSize(cusolver_handle handle, int m, int n,
                                double *a, int lda, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, double *, int,
                               int *);
  TRACE_CALL("cusolverDnDgeqrf_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDgeqrf_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, a, lda, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGEQRF_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDgeqrf(cusolver_handle handle, int m, int n, double *a,
                     int lda, double *tau, double *work, int lwork,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, double *, int,
                               double *, double *, int, int *);
  TRACE_CALL("cusolverDnDgeqrf", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDgeqrf");
    return real == 0 ? 7
                     : real(handle, m, n, a, lda, tau, work, lwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGEQRF;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)tau;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDgesvd_bufferSize(cusolver_handle handle, int m, int n,
                                int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int *);
  TRACE_CALL("cusolverDnDgesvd_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDgesvd_bufferSize");
    return real == 0 ? 7 : real(handle, m, n, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGESVD_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.m = m; request.n = n;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDgesvd(cusolver_handle handle, signed char jobu,
                     signed char jobvt, int m, int n, double *a, int lda,
                     double *s, double *u, int ldu, double *vt, int ldvt,
                     double *work, int lwork, double *rwork, int *info) {
  typedef int (*real_function)(cusolver_handle, signed char, signed char, int,
                               int, double *, int, double *, double *, int,
                               double *, int, double *, int, double *, int *);
  TRACE_CALL("cusolverDnDgesvd", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDgesvd");
    return real == 0 ? 7
                     : real(handle, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                            ldvt, work, lwork, rwork, info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DGESVD;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobu; request.transb = jobvt;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)s;
  request.c = (uint64_t)(uintptr_t)u; request.ldb = ldu;
  request.d = (uint64_t)(uintptr_t)vt; request.ldc = ldvt;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.a_descriptor = (uint64_t)(uintptr_t)rwork;
  request.b_descriptor = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDsyevd_bufferSize(cusolver_handle handle, int jobz, int uplo,
                                int n, const double *a, int lda,
                                const double *w, int *lwork) {
  typedef int (*real_function)(cusolver_handle, int, int, int,
                               const double *, int, const double *, int *);
  TRACE_CALL("cusolverDnDsyevd_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDsyevd_bufferSize");
    return real == 0 ? 7 : real(handle, jobz, uplo, n, a, lda, w, lwork);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DSYEVD_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)w;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDsyevd(cusolver_handle handle, int jobz, int uplo, int n,
                     double *a, int lda, double *w, double *work, int lwork,
                     int *info) {
  typedef int (*real_function)(cusolver_handle, int, int, int, double *, int,
                               double *, double *, int, int *);
  TRACE_CALL("cusolverDnDsyevd", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cusolverDnDsyevd");
    return real == 0 ? 7
                     : real(handle, jobz, uplo, n, a, lda, w, work, lwork,
                            info);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DSYEVD;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)w;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDpotrfBatched(cusolver_handle handle, int uplo, int n,
                            double *a_array[], int lda, int *info_array,
                            int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, double *[], int,
                               int *, int);
  TRACE_CALL("cusolverDnDpotrfBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDpotrfBatched");
    return real == 0
               ? 7
               : real(handle, uplo, n, a_array, lda, info_array, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DPOTRF_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)info_array;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}

#define DEFINE_POTRF_BATCHED(NAME, TYPE, OPCODE)                              \
  int NAME(cusolver_handle handle, int uplo, int n, TYPE *a_array[], int lda, \
           int *info_array, int batch_size) {                                 \
    typedef int (*real_function)(cusolver_handle, int, int, TYPE *[], int,     \
                                 int *, int);                                  \
    TRACE_CALL(#NAME, handle);                                                 \
    pthread_mutex_lock(&mutex);                                                \
    struct handle_entry *entry = find(handle);                                 \
    if (entry == 0) {                                                          \
      pthread_mutex_unlock(&mutex);                                            \
      static real_function real;                                               \
      if (real == 0) real = (real_function)real_symbol(#NAME);                 \
      return real == 0                                                         \
                 ? 7                                                          \
                 : real(handle, uplo, n, a_array, lda, info_array,             \
                        batch_size);                                           \
    }                                                                          \
    struct handle_entry copy = *entry;                                         \
    pthread_mutex_unlock(&mutex);                                              \
    struct request request = {0};                                              \
    struct response response;                                                  \
    request.opcode = OPCODE;                                                   \
    request.handle = (uint64_t)(uintptr_t)handle;                              \
    request.transa = uplo;                                                     \
    request.n = n;                                                             \
    request.a = (uint64_t)(uintptr_t)a_array;                                  \
    request.lda = lda;                                                         \
    request.b = (uint64_t)(uintptr_t)info_array;                               \
    request.batch_count = batch_size;                                          \
    return remote_call(&copy, &request, &response);                            \
  }

DEFINE_POTRF_BATCHED(cusolverDnSpotrfBatched, float, SOLVER_SPOTRF_BATCHED)
DEFINE_POTRF_BATCHED(cusolverDnCpotrfBatched, void, SOLVER_CPOTRF_BATCHED)
DEFINE_POTRF_BATCHED(cusolverDnZpotrfBatched, void, SOLVER_ZPOTRF_BATCHED)

#undef DEFINE_POTRF_BATCHED

int cusolverDnDpotrsBatched(cusolver_handle handle, int uplo, int n, int nrhs,
                            const double *a_array[], int lda,
                            double *b_array[], int ldb, int *info,
                            int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int,
                               const double *[], int, double *[], int, int *,
                               int);
  TRACE_CALL("cusolverDnDpotrsBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDpotrsBatched");
    return real == 0 ? 7
                     : real(handle, uplo, n, nrhs, a_array, lda, b_array, ldb,
                            info, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DPOTRS_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = uplo; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.c = (uint64_t)(uintptr_t)info;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}

int cusolverDnDsyevjBatched_bufferSize(
    cusolver_handle handle, int jobz, int uplo, int n, const double *a, int lda,
    const double *w, int *lwork, cusolver_handle params, int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int,
                               const double *, int, const double *, int *,
                               cusolver_handle, int);
  TRACE_CALL("cusolverDnDsyevjBatched_bufferSize", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDsyevjBatched_bufferSize");
    return real == 0
               ? 7
               : real(handle, jobz, uplo, n, a, lda, w, lwork, params,
                      batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DSYEVJ_BATCHED_BUFFER_SIZE;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)w;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  int status = remote_call(&copy, &request, &response);
  if (status == 0 && lwork != 0) *lwork = response.value;
  return status;
}

int cusolverDnDsyevjBatched(cusolver_handle handle, int jobz, int uplo, int n,
                            double *a, int lda, double *w, double *work,
                            int lwork, int *info, cusolver_handle params,
                            int batch_size) {
  typedef int (*real_function)(cusolver_handle, int, int, int, double *, int,
                               double *, double *, int, int *, cusolver_handle,
                               int);
  TRACE_CALL("cusolverDnDsyevjBatched", handle);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cusolverDnDsyevjBatched");
    return real == 0
               ? 7
               : real(handle, jobz, uplo, n, a, lda, w, work, lwork, info,
                      params, batch_size);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = SOLVER_DSYEVJ_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = jobz; request.transb = uplo; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)w;
  request.workspace = (uint64_t)(uintptr_t)work; request.value = lwork;
  request.c = (uint64_t)(uintptr_t)info;
  request.descriptor = (uint64_t)(uintptr_t)params;
  request.batch_count = batch_size;
  return remote_call(&copy, &request, &response);
}
