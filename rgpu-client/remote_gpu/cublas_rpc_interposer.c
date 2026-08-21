#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

typedef void *cublas_handle;

enum { CREATE = 1, DESTROY, SET_STREAM, SET_WORKSPACE, SET_MATH_MODE,
       SET_POINTER_MODE, SGEMM, GEMM_EX, SGEMM_EX,
       LT_MATMUL_DESC_CREATE, LT_MATMUL_DESC_DESTROY,
       LT_MATMUL_DESC_SET_ATTRIBUTE, LT_MATRIX_LAYOUT_CREATE,
       LT_MATRIX_LAYOUT_DESTROY, LT_MATRIX_LAYOUT_SET_ATTRIBUTE,
       LT_PREFERENCE_CREATE, LT_PREFERENCE_DESTROY,
       LT_PREFERENCE_SET_ATTRIBUTE, LT_HEURISTIC, LT_MATMUL };
enum { GEMM_STRIDED_BATCHED_EX = 21 };
enum { SGETRS_BATCHED = 35 };
enum { SDOT = 36 };
enum { STRSM_BATCHED = 42 };
enum { SGEMV = 58 };
enum { CGEMM = 65 };
enum { CGETRS_BATCHED = 69 };
enum { DOT_EX = 71 };
enum { ZGEMM = 72 };
enum { ZTRSM_BATCHED = 77 };
enum { ZGETRS_BATCHED = 83 };
enum { ZGEMM_STRIDED_BATCHED = 86 };
enum { CDOTU = 87, CDOTC = 88, ZDOTU = 89, ZDOTC = 90 };
enum { CTRSM_BATCHED = 95 };
enum { CGETRF_BATCHED = 100 };
enum { DTRSM_BATCHED = 112, DGETRS_BATCHED = 113, DGETRF_BATCHED = 114 };
enum { DGEMM_BATCHED = 115, DGEMV = 116 };
enum { DTRSM = 117 };
enum { DAXPY = 131, DCOPY, DSCAL, DNRM2, DASUM, DSWAP, IDAMAX, IDAMIN };
enum { CGEMM_STRIDED_BATCHED = 139 };
enum { SGETRF_BATCHED = 140 };
enum { CGEMV = 147, ZGEMV = 148 };
enum { SGELS_BATCHED = 149, DGELS_BATCHED, CGELS_BATCHED, ZGELS_BATCHED };
enum { SGEQRF_BATCHED = 153, DGEQRF_BATCHED, CGEQRF_BATCHED, ZGEQRF_BATCHED };
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
struct response { int32_t status, value; uint64_t handle;
  int32_t returned_algorithms; uint32_t payload_size; uint8_t payload[768]; };
typedef int (*rpc_function)(int, const struct request *, struct response *);
typedef int (*route_function)(void);

struct handle_entry {
  cublas_handle handle;
  int route, math_mode, pointer_mode;
  uint64_t stream, workspace, workspace_size;
  uint32_t dirty_state;
};
static struct handle_entry handles[4096];
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
enum { OPCODE_STATS_CAPACITY = 256 };
static uint64_t opcode_counts[OPCODE_STATS_CAPACITY];
static pthread_once_t stats_once = PTHREAD_ONCE_INIT;

static void trace_signal_handler(int signal_number) {
  void *frames[64];
  int count = backtrace(frames, 64);
  static const char heading[] = "rgpu-cublas: fatal signal backtrace\n";
  ssize_t ignored = write(STDERR_FILENO, heading, sizeof(heading) - 1);
  (void)ignored;
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  signal(signal_number, SIG_DFL);
  raise(signal_number);
}

__attribute__((constructor)) static void install_trace_signal_handler(void) {
  if (getenv("RGPU_CUBLAS_RPC_DEBUG") != 0) {
    signal(SIGSEGV, trace_signal_handler);
    signal(SIGBUS, trace_signal_handler);
  }
}

static void dump_opcode_counts(void) {
  const char *path = getenv("RGPU_CUBLAS_STATS");
  if (path == 0 || *path == '\0') return;
  FILE *output = fopen(path, "w");
  if (output == 0) return;
  for (size_t opcode = 0; opcode < OPCODE_STATS_CAPACITY; ++opcode) {
    uint64_t count = __atomic_load_n(&opcode_counts[opcode], __ATOMIC_RELAXED);
    if (count != 0) fprintf(output, "%zu\t%llu\n", opcode,
                            (unsigned long long)count);
  }
  fclose(output);
}

static void enable_opcode_stats(void) {
  if (getenv("RGPU_CUBLAS_STATS") != 0) atexit(dump_opcode_counts);
}

#define TRACE(format, ...)                                                    \
  do {                                                                        \
    if (getenv("RGPU_CUBLAS_RPC_DEBUG") != 0)                                \
      fprintf(stderr, "RGPU_CUBLAS_RPC " format "\n", ##__VA_ARGS__);       \
  } while (0)

static void *real_library;
static pthread_once_t real_library_once = PTHREAD_ONCE_INIT;
static void open_real_library(void) {
    real_library = dlopen(
        "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcublas.so.13",
        RTLD_NOW | RTLD_LOCAL);
    if (real_library == 0)
      real_library = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
}
static void *real_symbol(const char *name) {
  pthread_once(&real_library_once, open_real_library);
  return real_library == 0
             ? 0
             : dlvsym(real_library, name, "libcublas.so.13");
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
static int env_enabled(const char *name) {
  const char *value = getenv(name);
  return value != 0 && *value != '\0' && strcmp(value, "0") != 0 &&
         strcasecmp(value, "false") != 0 && strcasecmp(value, "no") != 0 &&
         strcasecmp(value, "off") != 0;
}
static int valid_transpose(int value) {
  return value == 0 || value == 1 || value == 2;
}
static int safe_to_defer(const struct request *request) {
  if (!valid_transpose(request->transa) ||
      !valid_transpose(request->transb) || request->m < 0 || request->n < 0 ||
      request->k < 0 || request->batch_count < 0)
    return 0;
  const int a_rows = request->transa == 0 ? request->m : request->k;
  const int b_rows = request->transb == 0 ? request->k : request->n;
  if (request->lda < (a_rows > 1 ? a_rows : 1) ||
      request->ldb < (b_rows > 1 ? b_rows : 1) ||
      request->ldc < (request->m > 1 ? request->m : 1))
    return 0;
  /* A synchronous no-op retains native cuBLAS's immediate status behavior. */
  return request->m != 0 && request->n != 0 && request->k != 0 &&
         (request->opcode != GEMM_STRIDED_BATCHED_EX ||
          request->batch_count != 0);
}
static struct handle_entry *find(cublas_handle handle) {
  for (size_t i = 0; i < 4096; ++i)
    if (handles[i].handle == handle) return &handles[i];
  return 0;
}
static int remote_call_response(struct handle_entry *entry,
                                struct request *request,
                                struct response *response) {
  rpc_function call = rpc();
  typedef int (*get_context_function)(void **);
  static get_context_function get_context;
  if (get_context == 0)
    get_context = (get_context_function)dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
  void *context = 0;
  pthread_once(&stats_once, enable_opcode_stats);
  if (request->opcode < OPCODE_STATS_CAPACITY)
    __atomic_add_fetch(&opcode_counts[request->opcode], 1, __ATOMIC_RELAXED);
  request->handle_state_mask = entry == 0 ? 0 : entry->dirty_state;
  const int compute = request->opcode == SGEMM || request->opcode == GEMM_EX ||
                      request->opcode == SGEMM_EX ||
                      request->opcode == GEMM_STRIDED_BATCHED_EX;
  request->asynchronous = compute && env_enabled("RGPU_CUBLAS_ASYNC") &&
                                  safe_to_defer(request)
                              ? 1u
                              : 0u;
  if (entry != 0) {
    request->stream = entry->stream;
    request->workspace = entry->workspace;
    request->workspace_size = entry->workspace_size;
    request->math_mode = entry->math_mode;
    request->pointer_mode = entry->pointer_mode;
  }
  if (get_context != 0 && get_context(&context) == 0)
    request->context = (uint64_t)context;
  memset(response, 0, sizeof(*response));
  response->status = 13;
  if (call == 0 || entry == 0) {
    TRACE("rpc unavailable opcode=%u call=%p entry=%p", request->opcode,
          (void *)call, (void *)entry);
    return 13;
  }
  int rpc_status = call(entry->route, request, response);
  TRACE("rpc opcode=%u transport=%d response=%d stream=%p context=%p mask=%u",
        request->opcode, rpc_status, response->status,
        (void *)(uintptr_t)request->stream,
        (void *)(uintptr_t)request->context, request->handle_state_mask);
  if (rpc_status != 0) return 13;
  return response->status;
}
static int remote_call(struct handle_entry *entry, struct request *request) {
  struct response response;
  return remote_call_response(entry, request, &response);
}

int cublasCreate_v2(cublas_handle *handle) {
  typedef int (*real_function)(cublas_handle *);
  typedef int (*get_context_function)(void **);
  int route = current_route();
  TRACE("create route=%d", route);
  if (route < 0) {
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasCreate_v2");
    int status = real == 0 ? 13 : real(handle);
    TRACE("create local symbol=%p status=%d handle=%p", (void *)real, status,
          status == 0 ? *handle : 0);
    return status;
  }
  struct request request = {0}; request.opcode = CREATE;
  static get_context_function get_context;
  if (get_context == 0)
    get_context = (get_context_function)dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
  void *context = 0;
  if (get_context != 0 && get_context(&context) == 0)
    request.context = (uint64_t)(uintptr_t)context;
  struct response response = {13, 0, 0};
  if (rpc() == 0 || rpc()(route, &request, &response) != 0 || response.status != 0) {
    TRACE("create remote route=%d status=%d", route, response.status);
    return response.status;
  }
  TRACE("create remote route=%d status=0 handle=%p", route,
        (void *)(uintptr_t)response.handle);
  *handle = (cublas_handle)response.handle;
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < 4096; ++i) if (handles[i].handle == 0) {
    handles[i] = (struct handle_entry){*handle, route, 0, 0, 0, 0, 0, 0};
    pthread_mutex_unlock(&mutex);
    return 0;
  }
  pthread_mutex_unlock(&mutex);
  return 3;
}

int cublasDestroy_v2(cublas_handle handle) {
  typedef int (*real_function)(cublas_handle);
  pthread_mutex_lock(&mutex); struct handle_entry *entry = find(handle);
  if (entry == 0) { pthread_mutex_unlock(&mutex); static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDestroy_v2");
    return real == 0 ? 13 : real(handle); }
  struct handle_entry copy = *entry; entry->handle = 0; pthread_mutex_unlock(&mutex);
  struct request request = {0}; request.opcode = DESTROY; request.handle = (uint64_t)handle;
  return remote_call(&copy, &request);
}

int cublasSetStream_v2(cublas_handle handle, void *stream) {
  typedef int (*real_function)(cublas_handle, void *);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasSetStream_v2");
    return real == 0 ? 13 : real(handle, stream);
  }
  uint64_t value = (uint64_t)stream;
  /* cuBLAS resets a custom workspace whenever SetStream is called. */
  if (entry->stream == value && entry->workspace == 0) {
    pthread_mutex_unlock(&mutex);
    return 0;
  }
  entry->stream = value;
  entry->workspace = 0;
  entry->workspace_size = 0;
  entry->dirty_state = (entry->dirty_state & ~2u) | 1u;
  pthread_mutex_unlock(&mutex);
  return 0;
}

int cublasSetWorkspace_v2(cublas_handle handle, void *workspace, size_t size) {
  typedef int (*real_function)(cublas_handle, void *, size_t);
  pthread_mutex_lock(&mutex); struct handle_entry *entry = find(handle);
  if (entry == 0) { pthread_mutex_unlock(&mutex); static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasSetWorkspace_v2");
    return real == 0 ? 13 : real(handle, workspace, size); }
  if (entry->workspace == (uint64_t)workspace && entry->workspace_size == size) {
    pthread_mutex_unlock(&mutex);
    return 0;
  }
  entry->workspace = (uint64_t)workspace;
  entry->workspace_size = size;
  entry->dirty_state |= 2u;
  pthread_mutex_unlock(&mutex);
  return 0;
}

static int set_integer(cublas_handle handle, int value, int math) {
  pthread_mutex_lock(&mutex); struct handle_entry *entry = find(handle);
  if (entry == 0) { pthread_mutex_unlock(&mutex); return -1; }
  if ((math ? entry->math_mode : entry->pointer_mode) == value) {
    pthread_mutex_unlock(&mutex);
    return 0;
  }
  if (math) {
    entry->math_mode = value;
    entry->dirty_state |= 4u;
  } else {
    entry->pointer_mode = value;
    entry->dirty_state |= 8u;
  }
  pthread_mutex_unlock(&mutex);
  return 0;
}
int cublasSetMathMode(cublas_handle h,int v) { int s=set_integer(h,v,1);
  if(s>=0)return s; typedef int(*f)(cublas_handle,int); static f r;if(!r)r=(f)real_symbol("cublasSetMathMode");return r?r(h,v):13; }
int cublasSetPointerMode_v2(cublas_handle h,int v) { int s=set_integer(h,v,0);
  if(s>=0)return s; typedef int(*f)(cublas_handle,int); static f r;if(!r)r=(f)real_symbol("cublasSetPointerMode_v2");return r?r(h,v):13; }
int cublasGetMathMode(cublas_handle h,int *v) { pthread_mutex_lock(&mutex);struct handle_entry*e=find(h);if(e){*v=e->math_mode;pthread_mutex_unlock(&mutex);return 0;}pthread_mutex_unlock(&mutex);typedef int(*f)(cublas_handle,int*);static f r;if(!r)r=(f)real_symbol("cublasGetMathMode");return r?r(h,v):13; }
int cublasGetPointerMode_v2(cublas_handle h,int *v) { pthread_mutex_lock(&mutex);struct handle_entry*e=find(h);if(e){*v=e->pointer_mode;pthread_mutex_unlock(&mutex);return 0;}pthread_mutex_unlock(&mutex);typedef int(*f)(cublas_handle,int*);static f r;if(!r)r=(f)real_symbol("cublasGetPointerMode_v2");return r?r(h,v):13; }

int cublasSgemm_v2(cublas_handle h,int ta,int tb,int m,int n,int k,const float*alpha,const float*a,int lda,const float*b,int ldb,const float*beta,float*c,int ldc) {
  typedef int(*f)(cublas_handle,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
  pthread_mutex_lock(&mutex);struct handle_entry*e=find(h);if(!e){pthread_mutex_unlock(&mutex);static f r;if(!r)r=(f)real_symbol("cublasSgemm_v2");return r?r(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc):13;}struct handle_entry copy=*e;pthread_mutex_unlock(&mutex);
  if(copy.pointer_mode!=0)return 15;struct request q={0};q.opcode=SGEMM;q.handle=(uint64_t)h;q.transa=ta;q.transb=tb;q.m=m;q.n=n;q.k=k;q.alpha=*alpha;q.a=(uint64_t)a;q.lda=lda;q.b=(uint64_t)b;q.ldb=ldb;q.beta=*beta;q.c=(uint64_t)c;q.ldc=ldc;return remote_call(&copy,&q);
}

int cublasCgemm_v2(cublas_handle handle, int transa, int transb, int m, int n,
                    int k, const void *alpha, const void *a, int lda,
                    const void *b, int ldb, const void *beta, void *c,
                    int ldc) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const void *, const void *, int, const void *,
                               int, const void *, void *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasCgemm_v2");
    return real == 0 ? 13
                     : real(handle, transa, transb, m, n, k, alpha, a, lda, b,
                            ldb, beta, c, ldc);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = CGEMM; request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transa; request.transb = transb;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, 2 * sizeof(float));
    memcpy(request.beta_data, beta, 2 * sizeof(float));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  TRACE("cgemm route=%d m=%d n=%d k=%d", copy.route, m, n, k);
  return remote_call(&copy, &request);
}

int cublasZgemm_v2(cublas_handle handle, int transa, int transb, int m, int n,
                    int k, const void *alpha, const void *a, int lda,
                    const void *b, int ldb, const void *beta, void *c,
                    int ldc) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const void *, const void *, int, const void *,
                               int, const void *, void *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasZgemm_v2");
    return real == 0 ? 13
                     : real(handle, transa, transb, m, n, k, alpha, a, lda, b,
                            ldb, beta, c, ldc);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = copy.pointer_mode == 0 ? GEMM_EX : ZGEMM;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transa; request.transb = transb;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.a_type = 5; request.b_type = 5; request.c_type = 5;
  request.compute_type = 70; request.algorithm = -1;
  request.scalar_size = 2 * sizeof(double);
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, 2 * sizeof(double));
    memcpy(request.beta_data, beta, 2 * sizeof(double));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  TRACE("zgemm route=%d trans=%d/%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d "
        "A=%p B=%p C=%p pointer_mode=%d alpha=%p beta=%p",
        copy.route, transa, transb, m, n, k, lda, ldb, ldc, a, b, c,
        copy.pointer_mode, alpha, beta);
  return remote_call(&copy, &request);
}

static int complex_dot(cublas_handle handle, int n, const void *x, int incx,
                       const void *y, int incy, void *result, int opcode,
                       const char *symbol_name, uint32_t scalar_size) {
  typedef int (*real_function)(cublas_handle, int, const void *, int,
                               const void *, int, void *);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    real_function real = (real_function)real_symbol(symbol_name);
    return real == 0 ? 13 : real(handle, n, x, incx, y, incy, result);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = (uint32_t)opcode;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)x;
  request.lda = incx;
  request.b = (uint64_t)(uintptr_t)y;
  request.ldb = incy;
  request.c = (uint64_t)(uintptr_t)result;
  request.scalar_size = scalar_size;
  struct response response;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && copy.pointer_mode == 0) {
    if (response.payload_size != scalar_size) return 13;
    memcpy(result, response.payload, scalar_size);
  }
  TRACE("complex_dot opcode=%d route=%d n=%d status=%d", opcode, copy.route,
        n, status);
  return status;
}

int cublasCdotu_v2(cublas_handle handle, int n, const void *x, int incx,
                    const void *y, int incy, void *result) {
  return complex_dot(handle, n, x, incx, y, incy, result, CDOTU,
                     "cublasCdotu_v2", 2 * sizeof(float));
}

int cublasCdotc_v2(cublas_handle handle, int n, const void *x, int incx,
                    const void *y, int incy, void *result) {
  return complex_dot(handle, n, x, incx, y, incy, result, CDOTC,
                     "cublasCdotc_v2", 2 * sizeof(float));
}

int cublasZdotu_v2(cublas_handle handle, int n, const void *x, int incx,
                    const void *y, int incy, void *result) {
  return complex_dot(handle, n, x, incx, y, incy, result, ZDOTU,
                     "cublasZdotu_v2", 2 * sizeof(double));
}

int cublasZdotc_v2(cublas_handle handle, int n, const void *x, int incx,
                    const void *y, int incy, void *result) {
  return complex_dot(handle, n, x, incx, y, incy, result, ZDOTC,
                     "cublasZdotc_v2", 2 * sizeof(double));
}

int cublasSgemv_v2(cublas_handle handle, int transpose, int m, int n,
                    const float *alpha, const float *a, int lda,
                    const float *x, int incx, const float *beta, float *y,
                    int incy) {
  typedef int (*real_function)(cublas_handle, int, int, int, const float *,
                               const float *, int, const float *, int,
                               const float *, float *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasSgemv_v2");
    return real == 0 ? 13
                     : real(handle, transpose, m, n, alpha, a, lda, x, incx,
                            beta, y, incy);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = SGEMV;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose;
  request.m = m;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)a;
  request.lda = lda;
  request.b = (uint64_t)(uintptr_t)x;
  request.ldb = incx;
  request.c = (uint64_t)(uintptr_t)y;
  request.ldc = incy;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, sizeof(float));
    memcpy(request.beta_data, beta, sizeof(float));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  TRACE("sgemv route=%d m=%d n=%d", copy.route, m, n);
  return remote_call(&copy, &request);
}

int cublasDgemv_v2(cublas_handle handle, int transpose, int m, int n,
                    const double *alpha, const double *a, int lda,
                    const double *x, int incx, const double *beta, double *y,
                    int incy) {
  typedef int (*real_function)(cublas_handle, int, int, int, const double *,
                               const double *, int, const double *, int,
                               const double *, double *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDgemv_v2");
    return real == 0 ? 13
                     : real(handle, transpose, m, n, alpha, a, lda, x, incx,
                            beta, y, incy);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DGEMV;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)x; request.ldb = incx;
  request.c = (uint64_t)(uintptr_t)y; request.ldc = incy;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, sizeof(double));
    memcpy(request.beta_data, beta, sizeof(double));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  return remote_call(&copy, &request);
}

#define DEFINE_COMPLEX_GEMV_RPC(PREFIX, OPCODE, SCALAR_BYTES)                  \
  int cublas##PREFIX##gemv_v2(                                                 \
      cublas_handle handle, int transpose, int m, int n, const void *alpha,   \
      const void *a, int lda, const void *x, int incx, const void *beta,      \
      void *y, int incy) {                                                     \
    typedef int (*real_function)(cublas_handle, int, int, int, const void *,  \
                                 const void *, int, const void *, int,         \
                                 const void *, void *, int);                   \
    pthread_mutex_lock(&mutex);                                                \
    struct handle_entry *entry = find(handle);                                 \
    if (entry == 0) {                                                          \
      pthread_mutex_unlock(&mutex);                                            \
      static real_function real;                                               \
      if (real == 0)                                                           \
        real = (real_function)real_symbol("cublas" #PREFIX "gemv_v2");       \
      return real == 0 ? 13                                                    \
                        : real(handle, transpose, m, n, alpha, a, lda, x,     \
                               incx, beta, y, incy);                           \
    }                                                                          \
    struct handle_entry copy = *entry;                                         \
    pthread_mutex_unlock(&mutex);                                              \
    struct request request = {0};                                              \
    request.opcode = OPCODE;                                                   \
    request.handle = (uint64_t)(uintptr_t)handle;                              \
    request.transa = transpose;                                                \
    request.m = m;                                                             \
    request.n = n;                                                             \
    request.a = (uint64_t)(uintptr_t)a;                                        \
    request.lda = lda;                                                         \
    request.b = (uint64_t)(uintptr_t)x;                                        \
    request.ldb = incx;                                                        \
    request.c = (uint64_t)(uintptr_t)y;                                        \
    request.ldc = incy;                                                        \
    request.scalar_size = SCALAR_BYTES;                                        \
    if (copy.pointer_mode == 0) {                                              \
      if (alpha == 0 || beta == 0) return 7;                                  \
      memcpy(request.alpha_data, alpha, SCALAR_BYTES);                         \
      memcpy(request.beta_data, beta, SCALAR_BYTES);                           \
    } else {                                                                   \
      request.d = (uint64_t)(uintptr_t)alpha;                                  \
      request.workspace = (uint64_t)(uintptr_t)beta;                           \
    }                                                                          \
    return remote_call(&copy, &request);                                       \
  }

DEFINE_COMPLEX_GEMV_RPC(C, CGEMV, 8)
DEFINE_COMPLEX_GEMV_RPC(Z, ZGEMV, 16)

#undef DEFINE_COMPLEX_GEMV_RPC

static size_t scalar_size_for_compute_type(int compute_type) {
  switch (compute_type) {
  case 64: case 65: return 2; /* CUBLAS_COMPUTE_16F[_PEDANTIC] */
  case 70: case 71: return 8; /* CUBLAS_COMPUTE_64F[_PEDANTIC] */
  default: return 4;
  }
}

int cublasGemmEx(cublas_handle h, int ta, int tb, int m, int n, int k,
                 const void *alpha, const void *a, int a_type, int lda,
                 const void *b, int b_type, int ldb, const void *beta,
                 void *c, int c_type, int ldc, int compute_type,
                 int algorithm) {
  typedef int (*f)(cublas_handle, int, int, int, int, int, const void *,
                   const void *, int, int, const void *, int, int,
                   const void *, void *, int, int, int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *e = find(h);
  if (e == 0) {
    pthread_mutex_unlock(&mutex);
    static f real;
    if (real == 0) real = (f)real_symbol("cublasGemmEx");
    return real == 0 ? 13 : real(h, ta, tb, m, n, k, alpha, a, a_type, lda,
                                 b, b_type, ldb, beta, c, c_type, ldc,
                                 compute_type, algorithm);
  }
  struct handle_entry copy = *e;
  pthread_mutex_unlock(&mutex);
  if (copy.pointer_mode != 0) return 15;
  struct request q = {0};
  q.opcode = GEMM_EX; q.handle = (uint64_t)h; q.transa = ta; q.transb = tb;
  q.m = m; q.n = n; q.k = k; q.a = (uint64_t)a; q.lda = lda;
  q.b = (uint64_t)b; q.ldb = ldb; q.c = (uint64_t)c; q.ldc = ldc;
  q.a_type = a_type; q.b_type = b_type; q.c_type = c_type;
  q.compute_type = compute_type; q.algorithm = algorithm;
  q.scalar_size = scalar_size_for_compute_type(compute_type);
  memcpy(q.alpha_data, alpha, q.scalar_size);
  memcpy(q.beta_data, beta, q.scalar_size);
  TRACE("gemm_ex route=%d types=%d/%d/%d compute=%d", copy.route, a_type,
        b_type, c_type, compute_type);
  return remote_call(&copy, &q);
}

int cublasDgemm_v2(cublas_handle handle, int transa, int transb, int m, int n,
                    int k, const double *alpha, const double *a, int lda,
                    const double *b, int ldb, const double *beta, double *c,
                    int ldc) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const double *, const double *, int,
                               const double *, int, const double *, double *,
                               int);
  pthread_mutex_lock(&mutex);
  const int remote = find(handle) != 0;
  pthread_mutex_unlock(&mutex);
  if (!remote) {
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDgemm_v2");
    return real == 0 ? 13
                     : real(handle, transa, transb, m, n, k, alpha, a, lda, b,
                            ldb, beta, c, ldc);
  }
  /* CUDA_R_64F=1, CUBLAS_COMPUTE_64F=70, CUBLAS_GEMM_DEFAULT=-1. */
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, a, 1, lda, b,
                      1, ldb, beta, c, 1, ldc, 70, -1);
}

int cublasDgemmBatched(
    cublas_handle handle, int transa, int transb, int m, int n, int k,
    const double *alpha, const double *const a_array[], int lda,
    const double *const b_array[], int ldb, const double *beta,
    double *const c_array[], int ldc, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const double *, const double *const[], int,
                               const double *const[], int, const double *,
                               double *const[], int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDgemmBatched");
    return real == 0
               ? 13
               : real(handle, transa, transb, m, n, k, alpha, a_array, lda,
                      b_array, ldb, beta, c_array, ldc, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DGEMM_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transa; request.transb = transb;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.c = (uint64_t)(uintptr_t)c_array; request.ldc = ldc;
  request.batch_count = batch_count;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, sizeof(double));
    memcpy(request.beta_data, beta, sizeof(double));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  return remote_call(&copy, &request);
}

int cublasSgemmEx(cublas_handle h, int ta, int tb, int m, int n, int k,
                  const float *alpha, const void *a, int a_type, int lda,
                  const void *b, int b_type, int ldb, const float *beta,
                  void *c, int c_type, int ldc) {
  typedef int (*f)(cublas_handle, int, int, int, int, int, const float *,
                   const void *, int, int, const void *, int, int,
                   const float *, void *, int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *e = find(h);
  if (e == 0) {
    pthread_mutex_unlock(&mutex);
    static f real;
    if (real == 0) real = (f)real_symbol("cublasSgemmEx");
    return real == 0 ? 13 : real(h, ta, tb, m, n, k, alpha, a, a_type, lda,
                                 b, b_type, ldb, beta, c, c_type, ldc);
  }
  struct handle_entry copy = *e;
  pthread_mutex_unlock(&mutex);
  if (copy.pointer_mode != 0) return 15;
  struct request q = {0};
  q.opcode = SGEMM_EX; q.handle = (uint64_t)h; q.transa = ta; q.transb = tb;
  q.m = m; q.n = n; q.k = k; q.a = (uint64_t)a; q.lda = lda;
  q.b = (uint64_t)b; q.ldb = ldb; q.c = (uint64_t)c; q.ldc = ldc;
  q.a_type = a_type; q.b_type = b_type; q.c_type = c_type;
  q.alpha = *alpha; q.beta = *beta;
  TRACE("sgemm_ex route=%d types=%d/%d/%d", copy.route, a_type, b_type,
        c_type);
  return remote_call(&copy, &q);
}

int cublasGemmStridedBatchedEx(
    cublas_handle h, int ta, int tb, int m, int n, int k,
    const void *alpha, const void *a, int a_type, int lda, int64_t stride_a,
    const void *b, int b_type, int ldb, int64_t stride_b, const void *beta,
    void *c, int c_type, int ldc, int64_t stride_c, int batch_count,
    int compute_type, int algorithm) {
  TRACE("gemm_strided_batched_ex entry handle=%p batches=%d", h,
        batch_count);
  typedef int (*f)(cublas_handle, int, int, int, int, int, const void *,
                   const void *, int, int, int64_t, const void *, int, int,
                   int64_t, const void *, void *, int, int, int64_t, int, int,
                   int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *e = find(h);
  if (e == 0) {
    pthread_mutex_unlock(&mutex);
    static f real;
    if (real == 0) real = (f)real_symbol("cublasGemmStridedBatchedEx");
    return real == 0
               ? 13
               : real(h, ta, tb, m, n, k, alpha, a, a_type, lda, stride_a,
                      b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c,
                      batch_count, compute_type, algorithm);
  }
  struct handle_entry copy = *e;
  pthread_mutex_unlock(&mutex);
  if (copy.pointer_mode != 0) return 15;
  struct request q = {0};
  q.opcode = GEMM_STRIDED_BATCHED_EX;
  q.handle = (uint64_t)h; q.transa = ta; q.transb = tb;
  q.m = m; q.n = n; q.k = k;
  q.a = (uint64_t)a; q.a_type = a_type; q.lda = lda; q.stride_a = stride_a;
  q.b = (uint64_t)b; q.b_type = b_type; q.ldb = ldb; q.stride_b = stride_b;
  q.c = (uint64_t)c; q.c_type = c_type; q.ldc = ldc; q.stride_c = stride_c;
  q.batch_count = batch_count; q.compute_type = compute_type;
  q.algorithm = algorithm;
  q.scalar_size = scalar_size_for_compute_type(compute_type);
  memcpy(q.alpha_data, alpha, q.scalar_size);
  memcpy(q.beta_data, beta, q.scalar_size);
  TRACE("gemm_strided_batched_ex route=%d batches=%d", copy.route,
        batch_count);
  return remote_call(&copy, &q);
}

int cublasDgemmStridedBatched(
    cublas_handle handle, int transa, int transb, int m, int n, int k,
    const double *alpha, const double *a, int lda, int64_t stride_a,
    const double *b, int ldb, int64_t stride_b, const double *beta, double *c,
    int ldc, int64_t stride_c, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const double *, const double *, int, int64_t,
                               const double *, int, int64_t, const double *,
                               double *, int, int64_t, int);
  pthread_mutex_lock(&mutex);
  const int remote = find(handle) != 0;
  pthread_mutex_unlock(&mutex);
  if (!remote) {
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cublasDgemmStridedBatched");
    return real == 0
               ? 13
               : real(handle, transa, transb, m, n, k, alpha, a, lda,
                      stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                      batch_count);
  }
  return cublasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, alpha, a, 1, lda, stride_a, b, 1, ldb,
      stride_b, beta, c, 1, ldc, stride_c, batch_count, 70, -1);
}

int cublasSgemmStridedBatched(
    cublas_handle h, int ta, int tb, int m, int n, int k,
    const float *alpha, const float *a, int lda, long long stride_a,
    const float *b, int ldb, long long stride_b, const float *beta,
    float *c, int ldc, long long stride_c, int batch_count) {
  typedef int (*f)(cublas_handle, int, int, int, int, int, const float *,
                   const float *, int, long long, const float *, int,
                   long long, const float *, float *, int, long long, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *e = find(h);
  if (e == 0) {
    pthread_mutex_unlock(&mutex);
    static f real;
    if (real == 0) real = (f)real_symbol("cublasSgemmStridedBatched");
    return real == 0
               ? 13
               : real(h, ta, tb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                      stride_b, beta, c, ldc, stride_c, batch_count);
  }
  struct handle_entry copy = *e;
  pthread_mutex_unlock(&mutex);
  if (copy.pointer_mode != 0) return 15;
  struct request q = {0};
  q.opcode = GEMM_STRIDED_BATCHED_EX;
  q.handle = (uint64_t)h; q.transa = ta; q.transb = tb;
  q.m = m; q.n = n; q.k = k;
  q.a = (uint64_t)a; q.a_type = 0; q.lda = lda; q.stride_a = stride_a;
  q.b = (uint64_t)b; q.b_type = 0; q.ldb = ldb; q.stride_b = stride_b;
  q.c = (uint64_t)c; q.c_type = 0; q.ldc = ldc; q.stride_c = stride_c;
  q.batch_count = batch_count; q.compute_type = 68; q.algorithm = -1;
  q.scalar_size = sizeof(float);
  memcpy(q.alpha_data, alpha, sizeof(float));
  memcpy(q.beta_data, beta, sizeof(float));
  TRACE("sgemm_strided_batched route=%d batches=%d", copy.route,
        batch_count);
  return remote_call(&copy, &q);
}

int cublasZgemmStridedBatched(
    cublas_handle handle, int transa, int transb, int m, int n, int k,
    const void *alpha, const void *a, int lda, int64_t stride_a,
    const void *b, int ldb, int64_t stride_b, const void *beta, void *c,
    int ldc, int64_t stride_c, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const void *, const void *, int, int64_t,
                               const void *, int, int64_t, const void *, void *,
                               int, int64_t, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cublasZgemmStridedBatched");
    return real == 0
               ? 13
               : real(handle, transa, transb, m, n, k, alpha, a, lda,
                      stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                      batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = copy.pointer_mode == 0
                       ? GEMM_STRIDED_BATCHED_EX
                       : ZGEMM_STRIDED_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transa; request.transb = transb;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.stride_a = stride_a;
  request.b = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  request.stride_b = stride_b;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.stride_c = stride_c; request.batch_count = batch_count;
  request.a_type = 5; request.b_type = 5; request.c_type = 5;
  request.compute_type = 70; request.algorithm = -1;
  request.scalar_size = 2 * sizeof(double);
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, 2 * sizeof(double));
    memcpy(request.beta_data, beta, 2 * sizeof(double));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  return remote_call(&copy, &request);
}

int cublasCgemmStridedBatched(
    cublas_handle handle, int transa, int transb, int m, int n, int k,
    const void *alpha, const void *a, int lda, int64_t stride_a,
    const void *b, int ldb, int64_t stride_b, const void *beta, void *c,
    int ldc, int64_t stride_c, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int,
                               const void *, const void *, int, int64_t,
                               const void *, int, int64_t, const void *, void *,
                               int, int64_t, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0)
      real = (real_function)real_symbol("cublasCgemmStridedBatched");
    return real == 0
               ? 13
               : real(handle, transa, transb, m, n, k, alpha, a, lda,
                      stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                      batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = CGEMM_STRIDED_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transa; request.transb = transb;
  request.m = m; request.n = n; request.k = k;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.stride_a = stride_a;
  request.b = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  request.stride_b = stride_b;
  request.c = (uint64_t)(uintptr_t)c; request.ldc = ldc;
  request.stride_c = stride_c; request.batch_count = batch_count;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, 2 * sizeof(float));
    memcpy(request.beta_data, beta, 2 * sizeof(float));
  } else {
    request.d = (uint64_t)(uintptr_t)alpha;
    request.workspace = (uint64_t)(uintptr_t)beta;
  }
  return remote_call(&copy, &request);
}

int cublasSgetrsBatched(cublas_handle handle, int transpose, int n, int nrhs,
                        const float *const a_array[], int lda,
                        const int *pivots, float *const b_array[], int ldb,
                        int *info, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int,
                               const float *const[], int, const int *,
                               float *const[], int, int *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasSgetrsBatched");
    return real == 0 ? 13
                     : real(handle, transpose, n, nrhs, a_array, lda, pivots,
                            b_array, ldb, info, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  TRACE("sgetrs_batched route=%d handle=%p A=%p pivots=%p B=%p info=%p "
        "n=%d nrhs=%d lda=%d ldb=%d batch=%d stream=%p dirty=%u",
        copy.route, handle, (const void *)a_array, (const void *)pivots,
        (const void *)b_array, (void *)info, n, nrhs, lda, ldb, batch_count,
        (void *)(uintptr_t)copy.stream, copy.dirty_state);
  struct request request = {0};
  request.opcode = SGETRS_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose;
  request.n = n;
  request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a_array;
  request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivots;
  request.c = (uint64_t)(uintptr_t)b_array;
  request.ldb = ldb;
  request.d = (uint64_t)(uintptr_t)info;
  request.batch_count = batch_count;
  struct response response;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && info != 0) *info = response.value;
  TRACE("sgetrs_batched status=%d", status);
  return status;
}

int cublasCgetrsBatched(cublas_handle handle, int transpose, int n, int nrhs,
                        const void *const a_array[], int lda,
                        const int *pivots, void *const b_array[], int ldb,
                        int *info, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int,
                               const void *const[], int, const int *,
                               void *const[], int, int *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasCgetrsBatched");
    return real == 0 ? 13
                     : real(handle, transpose, n, nrhs, a_array, lda, pivots,
                            b_array, ldb, info, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = CGETRS_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivots;
  request.c = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && info != 0) *info = response.value;
  return status;
}

int cublasZgetrsBatched(cublas_handle handle, int transpose, int n, int nrhs,
                        const void *const a_array[], int lda,
                        const int *pivots, void *const b_array[], int ldb,
                        int *info, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int,
                               const void *const[], int, const int *,
                               void *const[], int, int *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasZgetrsBatched");
    return real == 0 ? 13
                     : real(handle, transpose, n, nrhs, a_array, lda, pivots,
                            b_array, ldb, info, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = ZGETRS_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivots;
  request.c = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && info != 0) *info = response.value;
  return status;
}

int cublasDgetrsBatched(cublas_handle handle, int transpose, int n, int nrhs,
                        const double *const a_array[], int lda,
                        const int *pivots, double *const b_array[], int ldb,
                        int *info, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int,
                               const double *const[], int, const int *,
                               double *const[], int, int *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDgetrsBatched");
    return real == 0 ? 13
                     : real(handle, transpose, n, nrhs, a_array, lda, pivots,
                            b_array, ldb, info, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = DGETRS_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transa = transpose; request.n = n; request.k = nrhs;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivots;
  request.c = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && info != 0) *info = response.value;
  return status;
}

int cublasSdot_v2(cublas_handle handle, int n, const float *x, int incx,
                  const float *y, int incy, float *result) {
  typedef int (*real_function)(cublas_handle, int, const float *, int,
                               const float *, int, float *);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasSdot_v2");
    return real == 0 ? 13 : real(handle, n, x, incx, y, incy, result);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = SDOT;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)x;
  request.lda = incx;
  request.b = (uint64_t)(uintptr_t)y;
  request.ldb = incy;
  request.c = (uint64_t)(uintptr_t)result;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && copy.pointer_mode == 0 && result != 0)
    memcpy(result, response.payload, sizeof(*result));
  return status;
}

static uint32_t cuda_data_type_size(int type) {
  switch (type) {
  case 0: case 10: case 12: return 4;  /* R32F, R32I, R32U */
  case 1: case 24: case 26: return 8;  /* R64F, R64I, R64U */
  case 2: case 14: case 20: case 22: return 2; /* R16F/BF/I/U */
  case 4: case 11: case 13: return 8;  /* C32F/I/U */
  case 5: case 25: case 27: return 16; /* C64F/I/U */
  case 6: case 15: case 21: case 23: return 4; /* C16F/BF/I/U */
  case 3: case 7: case 8: case 9: case 28: case 29: case 30: return 1;
  default: return 0;
  }
}

int cublasDotEx(cublas_handle handle, int n, const void *x, int x_type,
                int incx, const void *y, int y_type, int incy, void *result,
                int result_type, int execution_type) {
  typedef int (*real_function)(cublas_handle, int, const void *, int, int,
                               const void *, int, int, void *, int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDotEx");
    return real == 0 ? 13
                     : real(handle, n, x, x_type, incx, y, y_type, incy,
                            result, result_type, execution_type);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  struct response response;
  request.opcode = DOT_EX;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)x;
  request.a_type = x_type;
  request.lda = incx;
  request.b = (uint64_t)(uintptr_t)y;
  request.b_type = y_type;
  request.ldb = incy;
  request.c = (uint64_t)(uintptr_t)result;
  request.c_type = result_type;
  request.compute_type = execution_type;
  request.scalar_size = cuda_data_type_size(result_type);
  if (request.scalar_size == 0) return 15;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && copy.pointer_mode == 0 && result != 0) {
    if (response.payload_size < request.scalar_size) return 13;
    memcpy(result, response.payload, request.scalar_size);
  }
  return status;
}

int cublasDdot_v2(cublas_handle handle, int n, const double *x, int incx,
                  const double *y, int incy, double *result) {
  typedef int (*real_function)(cublas_handle, int, const double *, int,
                               const double *, int, double *);
  pthread_mutex_lock(&mutex);
  const int remote = find(handle) != 0;
  pthread_mutex_unlock(&mutex);
  if (!remote) {
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDdot_v2");
    return real == 0 ? 13 : real(handle, n, x, incx, y, incy, result);
  }
  return cublasDotEx(handle, n, x, 1, incx, y, 1, incy, result, 1, 1);
}

int cublasStrsmBatched(cublas_handle handle, int side, int uplo,
                       int transpose, int diagonal, int m, int n,
                       const float *alpha, const float *const a_array[],
                       int lda, float *const b_array[], int ldb,
                       int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int, int,
                               const float *, const float *const[], int,
                               float *const[], int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasStrsmBatched");
    return real == 0
               ? 13
               : real(handle, side, uplo, transpose, diagonal, m, n, alpha,
                      a_array, lda, b_array, ldb, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = STRSM_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.attribute = uplo;
  request.transa = transpose; request.value = diagonal;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  if (copy.pointer_mode == 0)
    memcpy(request.alpha_data, alpha, sizeof(float));
  else
    request.c = (uint64_t)(uintptr_t)alpha;
  return remote_call(&copy, &request);
}

int cublasCgetrfBatched(cublas_handle handle, int n, void *a_array[], int lda,
                        int *pivot_array, int *info_array, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, void *[], int, int *, int *,
                               int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasCgetrfBatched");
    return real == 0
               ? 13
               : real(handle, n, a_array, lda, pivot_array, info_array,
                      batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = CGETRF_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivot_array;
  request.c = (uint64_t)(uintptr_t)info_array;
  request.batch_count = batch_count;
  return remote_call(&copy, &request);
}

int cublasDgetrfBatched(cublas_handle handle, int n, double *a_array[],
                        int lda, int *pivot_array, int *info_array,
                        int batch_count) {
  typedef int (*real_function)(cublas_handle, int, double *[], int, int *,
                               int *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDgetrfBatched");
    return real == 0
               ? 13
               : real(handle, n, a_array, lda, pivot_array, info_array,
                      batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DGETRF_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivot_array;
  request.c = (uint64_t)(uintptr_t)info_array;
  request.batch_count = batch_count;
  return remote_call(&copy, &request);
}

int cublasSgetrfBatched(cublas_handle handle, int n, float *a_array[],
                        int lda, int *pivot_array, int *info_array,
                        int batch_count) {
  typedef int (*real_function)(cublas_handle, int, float *[], int, int *,
                               int *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasSgetrfBatched");
    return real == 0
               ? 13
               : real(handle, n, a_array, lda, pivot_array, info_array,
                      batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = SGETRF_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)pivot_array;
  request.c = (uint64_t)(uintptr_t)info_array;
  request.batch_count = batch_count;
  return remote_call(&copy, &request);
}

#define DEFINE_GELS_BATCHED_RPC(PREFIX, TYPE, OPCODE)                         \
int cublas##PREFIX##gelsBatched(cublas_handle handle, int transpose, int m,   \
                                int n, int nrhs, TYPE *const a_array[],       \
                                int lda, TYPE *const c_array[], int ldc,      \
                                int *info, int *dev_info_array,               \
                                int batch_count) {                            \
  typedef int (*real_function)(cublas_handle, int, int, int, int,             \
                               TYPE *const[], int, TYPE *const[], int, int *,  \
                               int *, int);                                   \
  pthread_mutex_lock(&mutex);                                                 \
  struct handle_entry *entry = find(handle);                                  \
  if (entry == 0) {                                                           \
    pthread_mutex_unlock(&mutex);                                             \
    static real_function real;                                                \
    if (real == 0)                                                            \
      real = (real_function)real_symbol("cublas" #PREFIX "gelsBatched");   \
    return real == 0 ? 13 : real(handle, transpose, m, n, nrhs, a_array,      \
                                  lda, c_array, ldc, info, dev_info_array,    \
                                  batch_count);                               \
  }                                                                           \
  struct handle_entry copy = *entry;                                          \
  pthread_mutex_unlock(&mutex);                                               \
  struct request request = {0};                                               \
  request.opcode = OPCODE; request.handle = (uint64_t)(uintptr_t)handle;       \
  request.transa = transpose; request.m = m; request.n = n; request.k = nrhs; \
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;                \
  request.b = (uint64_t)(uintptr_t)c_array; request.ldb = ldc;                \
  request.c = (uint64_t)(uintptr_t)dev_info_array;                            \
  request.batch_count = batch_count;                                          \
  struct response response = {0};                                             \
  int status = remote_call_response(&copy, &request, &response);              \
  if (info != 0) *info = response.value;                                      \
  return status;                                                              \
}

DEFINE_GELS_BATCHED_RPC(S, float, SGELS_BATCHED)
DEFINE_GELS_BATCHED_RPC(D, double, DGELS_BATCHED)
DEFINE_GELS_BATCHED_RPC(C, void, CGELS_BATCHED)
DEFINE_GELS_BATCHED_RPC(Z, void, ZGELS_BATCHED)

#define DEFINE_GEQRF_BATCHED_RPC(PREFIX, TYPE, OPCODE)                       \
int cublas##PREFIX##geqrfBatched(cublas_handle handle, int m, int n,          \
                                 TYPE *const a_array[], int lda,              \
                                 TYPE *const tau_array[], int *info,          \
                                 int batch_count) {                           \
  typedef int (*real_function)(cublas_handle, int, int, TYPE *const[], int,   \
                               TYPE *const[], int *, int);                    \
  pthread_mutex_lock(&mutex);                                                 \
  struct handle_entry *entry = find(handle);                                  \
  if (entry == 0) {                                                           \
    pthread_mutex_unlock(&mutex);                                             \
    static real_function real;                                                \
    if (real == 0)                                                            \
      real = (real_function)real_symbol("cublas" #PREFIX "geqrfBatched");  \
    return real == 0 ? 13 : real(handle, m, n, a_array, lda, tau_array,      \
                                  info, batch_count);                         \
  }                                                                           \
  struct handle_entry copy = *entry;                                          \
  pthread_mutex_unlock(&mutex);                                               \
  struct request request = {0};                                               \
  request.opcode = OPCODE; request.handle = (uint64_t)(uintptr_t)handle;       \
  request.m = m; request.n = n;                                               \
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;                \
  request.b = (uint64_t)(uintptr_t)tau_array;                                 \
  request.batch_count = batch_count;                                          \
  struct response response = {0};                                             \
  int status = remote_call_response(&copy, &request, &response);              \
  if (info != 0) *info = response.value;                                      \
  return status;                                                              \
}

DEFINE_GEQRF_BATCHED_RPC(S, float, SGEQRF_BATCHED)
DEFINE_GEQRF_BATCHED_RPC(D, double, DGEQRF_BATCHED)
DEFINE_GEQRF_BATCHED_RPC(C, void, CGEQRF_BATCHED)
DEFINE_GEQRF_BATCHED_RPC(Z, void, ZGEQRF_BATCHED)

int cublasCtrsmBatched(cublas_handle handle, int side, int uplo,
                       int transpose, int diagonal, int m, int n,
                       const void *alpha, const void *const a_array[], int lda,
                       void *const b_array[], int ldb, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int, int,
                               const void *, const void *const[], int,
                               void *const[], int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasCtrsmBatched");
    return real == 0
               ? 13
               : real(handle, side, uplo, transpose, diagonal, m, n, alpha,
                      a_array, lda, b_array, ldb, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = CTRSM_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.attribute = uplo;
  request.transa = transpose; request.value = diagonal;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, 2 * sizeof(float));
  } else {
    request.c = (uint64_t)(uintptr_t)alpha;
  }
  return remote_call(&copy, &request);
}

int cublasDtrsmBatched(cublas_handle handle, int side, int uplo,
                       int transpose, int diagonal, int m, int n,
                       const double *alpha,
                       const double *const a_array[], int lda,
                       double *const b_array[], int ldb, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int, int,
                               const double *, const double *const[], int,
                               double *const[], int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDtrsmBatched");
    return real == 0
               ? 13
               : real(handle, side, uplo, transpose, diagonal, m, n, alpha,
                      a_array, lda, b_array, ldb, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DTRSM_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.attribute = uplo;
  request.transa = transpose; request.value = diagonal;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  if (copy.pointer_mode == 0)
    memcpy(request.alpha_data, alpha, sizeof(double));
  else
    request.c = (uint64_t)(uintptr_t)alpha;
  return remote_call(&copy, &request);
}

int cublasDtrsm_v2(cublas_handle handle, int side, int uplo, int transpose,
                    int diagonal, int m, int n, const double *alpha,
                    const double *a, int lda, double *b, int ldb) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int, int,
                               const double *, const double *, int, double *,
                               int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDtrsm_v2");
    return real == 0
               ? 13
               : real(handle, side, uplo, transpose, diagonal, m, n, alpha, a,
                      lda, b, ldb);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DTRSM;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.attribute = uplo;
  request.transa = transpose; request.value = diagonal;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b; request.ldb = ldb;
  if (copy.pointer_mode == 0)
    memcpy(request.alpha_data, alpha, sizeof(double));
  else
    request.c = (uint64_t)(uintptr_t)alpha;
  return remote_call(&copy, &request);
}

int cublasDaxpy_v2(cublas_handle handle, int n, const double *alpha,
                    const double *x, int incx, double *y, int incy) {
  typedef int (*real_function)(cublas_handle, int, const double *,
                               const double *, int, double *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDaxpy_v2");
    return real == 0 ? 13 : real(handle, n, alpha, x, incx, y, incy);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DAXPY; request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)x; request.lda = incx;
  request.b = (uint64_t)(uintptr_t)y; request.ldb = incy;
  if (copy.pointer_mode == 0)
    memcpy(request.alpha_data, alpha, sizeof(*alpha));
  else
    request.c = (uint64_t)(uintptr_t)alpha;
  return remote_call(&copy, &request);
}

int cublasDcopy_v2(cublas_handle handle, int n, const double *x, int incx,
                    double *y, int incy) {
  typedef int (*real_function)(cublas_handle, int, const double *, int,
                               double *, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDcopy_v2");
    return real == 0 ? 13 : real(handle, n, x, incx, y, incy);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DCOPY; request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)x; request.lda = incx;
  request.b = (uint64_t)(uintptr_t)y; request.ldb = incy;
  return remote_call(&copy, &request);
}

int cublasDscal_v2(cublas_handle handle, int n, const double *alpha,
                    double *x, int incx) {
  typedef int (*real_function)(cublas_handle, int, const double *, double *,
                               int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDscal_v2");
    return real == 0 ? 13 : real(handle, n, alpha, x, incx);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DSCAL; request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)x; request.lda = incx;
  if (copy.pointer_mode == 0)
    memcpy(request.alpha_data, alpha, sizeof(*alpha));
  else
    request.c = (uint64_t)(uintptr_t)alpha;
  return remote_call(&copy, &request);
}

static int double_reduction_rpc(cublas_handle handle, uint32_t opcode, int n,
                                const double *x, int incx, void *result,
                                size_t result_size) {
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    return -1;
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0}; struct response response;
  request.opcode = opcode; request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)x; request.lda = incx;
  request.c = (uint64_t)(uintptr_t)result;
  int status = remote_call_response(&copy, &request, &response);
  if (status == 0 && copy.pointer_mode == 0 && result != 0) {
    if (response.payload_size < result_size) return 13;
    memcpy(result, response.payload, result_size);
  }
  return status;
}

int cublasDnrm2_v2(cublas_handle handle, int n, const double *x, int incx,
                    double *result) {
  typedef int (*real_function)(cublas_handle, int, const double *, int,
                               double *);
  int status = double_reduction_rpc(handle, DNRM2, n, x, incx, result,
                                    sizeof(*result));
  if (status != -1) return status;
  static real_function real;
  if (real == 0) real = (real_function)real_symbol("cublasDnrm2_v2");
  return real == 0 ? 13 : real(handle, n, x, incx, result);
}

int cublasDasum_v2(cublas_handle handle, int n, const double *x, int incx,
                    double *result) {
  typedef int (*real_function)(cublas_handle, int, const double *, int,
                               double *);
  int status = double_reduction_rpc(handle, DASUM, n, x, incx, result,
                                    sizeof(*result));
  if (status != -1) return status;
  static real_function real;
  if (real == 0) real = (real_function)real_symbol("cublasDasum_v2");
  return real == 0 ? 13 : real(handle, n, x, incx, result);
}

int cublasDswap_v2(cublas_handle handle, int n, double *x, int incx,
                    double *y, int incy) {
  typedef int (*real_function)(cublas_handle, int, double *, int, double *,
                               int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasDswap_v2");
    return real == 0 ? 13 : real(handle, n, x, incx, y, incy);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = DSWAP; request.handle = (uint64_t)(uintptr_t)handle;
  request.n = n; request.a = (uint64_t)(uintptr_t)x; request.lda = incx;
  request.b = (uint64_t)(uintptr_t)y; request.ldb = incy;
  return remote_call(&copy, &request);
}

int cublasIdamax_v2(cublas_handle handle, int n, const double *x, int incx,
                     int *result) {
  typedef int (*real_function)(cublas_handle, int, const double *, int, int *);
  int status = double_reduction_rpc(handle, IDAMAX, n, x, incx, result,
                                    sizeof(*result));
  if (status != -1) return status;
  static real_function real;
  if (real == 0) real = (real_function)real_symbol("cublasIdamax_v2");
  return real == 0 ? 13 : real(handle, n, x, incx, result);
}

int cublasIdamin_v2(cublas_handle handle, int n, const double *x, int incx,
                     int *result) {
  typedef int (*real_function)(cublas_handle, int, const double *, int, int *);
  int status = double_reduction_rpc(handle, IDAMIN, n, x, incx, result,
                                    sizeof(*result));
  if (status != -1) return status;
  static real_function real;
  if (real == 0) real = (real_function)real_symbol("cublasIdamin_v2");
  return real == 0 ? 13 : real(handle, n, x, incx, result);
}

int cublasZtrsmBatched(cublas_handle handle, int side, int uplo,
                       int transpose, int diagonal, int m, int n,
                       const void *alpha, const void *const a_array[], int lda,
                       void *const b_array[], int ldb, int batch_count) {
  typedef int (*real_function)(cublas_handle, int, int, int, int, int, int,
                               const void *, const void *const[], int,
                               void *const[], int, int);
  pthread_mutex_lock(&mutex);
  struct handle_entry *entry = find(handle);
  if (entry == 0) {
    pthread_mutex_unlock(&mutex);
    static real_function real;
    if (real == 0) real = (real_function)real_symbol("cublasZtrsmBatched");
    return real == 0
               ? 13
               : real(handle, side, uplo, transpose, diagonal, m, n, alpha,
                      a_array, lda, b_array, ldb, batch_count);
  }
  struct handle_entry copy = *entry;
  pthread_mutex_unlock(&mutex);
  struct request request = {0};
  request.opcode = ZTRSM_BATCHED;
  request.handle = (uint64_t)(uintptr_t)handle;
  request.transb = side; request.attribute = uplo;
  request.transa = transpose; request.value = diagonal;
  request.m = m; request.n = n;
  request.a = (uint64_t)(uintptr_t)a_array; request.lda = lda;
  request.b = (uint64_t)(uintptr_t)b_array; request.ldb = ldb;
  request.batch_count = batch_count;
  if (copy.pointer_mode == 0) {
    memcpy(request.alpha_data, alpha, 2 * sizeof(double));
  } else {
    request.c = (uint64_t)(uintptr_t)alpha;
  }
  return remote_call(&copy, &request);
}

static void *lt_library;
static pthread_once_t lt_library_once = PTHREAD_ONCE_INIT;
static void open_lt_library(void) {
  lt_library = dlopen(
      "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/"
      "libcublasLt.so.13", RTLD_NOW | RTLD_LOCAL);
  if (lt_library == 0)
    lt_library = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_LOCAL);
}
static void *lt_symbol(const char *name) {
  pthread_once(&lt_library_once, open_lt_library);
  return lt_library == 0 ? 0 : dlvsym(lt_library, name, "libcublasLt.so.13");
}

struct object_entry { void *object; int route, kind, compute_type, pointer_mode; };
static struct object_entry objects[8192];
static struct object_entry *find_object(void *object) {
  for (size_t i = 0; i < 8192; ++i)
    if (objects[i].object == object) return &objects[i];
  return 0;
}
static int remember_object(void *object, int route, int kind,
                           int compute_type) {
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < 8192; ++i) if (objects[i].object == 0) {
    objects[i] = (struct object_entry){object, route, kind, compute_type, 0};
    pthread_mutex_unlock(&mutex); return 0;
  }
  pthread_mutex_unlock(&mutex); return 3;
}
static int take_object(void *object, struct object_entry *copy) {
  int found = 0; pthread_mutex_lock(&mutex);
  struct object_entry *entry = find_object(object);
  if (entry != 0) { *copy = *entry; entry->object = 0; found = 1; }
  pthread_mutex_unlock(&mutex); return found;
}
static int call_route(int route, struct request *request,
                      struct response *response) {
  rpc_function call = rpc();
  pthread_once(&stats_once, enable_opcode_stats);
  if (request->opcode < 64)
    __atomic_add_fetch(&opcode_counts[request->opcode], 1, __ATOMIC_RELAXED);
  typedef int (*get_context_function)(void **);
  static get_context_function get_context;
  if (get_context == 0)
    get_context = (get_context_function)dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
  void *context = 0;
  if (get_context != 0 && get_context(&context) == 0)
    request->context = (uint64_t)context;
  memset(response, 0, sizeof(*response)); response->status = 13;
  return call == 0 || call(route, request, response) != 0 ? 13
                                                          : response->status;
}

int cublasLtMatmulDescCreate(void **descriptor, int compute_type,
                             int scale_type) {
  typedef int (*f)(void **, int, int); int route = current_route();
  TRACE("lt_desc_create route=%d compute=%d scale=%d", route, compute_type,
        scale_type);
  if (route < 0) { static f real; if (!real) real=(f)lt_symbol("cublasLtMatmulDescCreate");
    return real ? real(descriptor,compute_type,scale_type) : 13; }
  struct request q={0}; struct response r; q.opcode=LT_MATMUL_DESC_CREATE;
  q.compute_type=compute_type; q.c_type=scale_type;
  int status=call_route(route,&q,&r); if(status)return status;
  *descriptor=(void *)(uintptr_t)r.handle;
  return remember_object(*descriptor,route,1,compute_type);
}

#define LT_DESTROY(name, opcode_value, symbol_name)                          \
int name(void *object) {                                                     \
  typedef int(*f)(void *); struct object_entry entry;                        \
  if (!take_object(object,&entry)) { static f real;                          \
    if(!real)real=(f)lt_symbol(symbol_name); return real?real(object):13; }   \
  struct request q={0};struct response r;q.opcode=opcode_value;              \
  q.descriptor=(uint64_t)object;return call_route(entry.route,&q,&r);        \
}
LT_DESTROY(cublasLtMatmulDescDestroy,LT_MATMUL_DESC_DESTROY,
           "cublasLtMatmulDescDestroy")
LT_DESTROY(cublasLtMatrixLayoutDestroy,LT_MATRIX_LAYOUT_DESTROY,
           "cublasLtMatrixLayoutDestroy")
LT_DESTROY(cublasLtMatmulPreferenceDestroy,LT_PREFERENCE_DESTROY,
           "cublasLtMatmulPreferenceDestroy")

static int set_lt_attribute(void *object,int attribute,const void *buffer,
                            size_t size,int opcode,const char *symbol_name) {
  typedef int(*f)(void *,int,const void *,size_t);
  pthread_mutex_lock(&mutex);struct object_entry *found=find_object(object);
  if(!found){pthread_mutex_unlock(&mutex);f real=(f)lt_symbol(symbol_name);
    return real?real(object,attribute,buffer,size):13;}
  struct object_entry entry=*found;pthread_mutex_unlock(&mutex);
  if(size>128)return 7;struct request q={0};struct response r;q.opcode=opcode;
  q.descriptor=(uint64_t)object;q.attribute=attribute;q.payload_size=size;
  memcpy(q.payload,buffer,size);int status=call_route(entry.route,&q,&r);
  if(status==0&&opcode==LT_MATMUL_DESC_SET_ATTRIBUTE&&attribute==2&&
     size==sizeof(int)){
    int pointer_mode=0;memcpy(&pointer_mode,buffer,sizeof(pointer_mode));
    pthread_mutex_lock(&mutex);found=find_object(object);
    if(found)found->pointer_mode=pointer_mode;pthread_mutex_unlock(&mutex);
  }
  return status;
}
int cublasLtMatmulDescSetAttribute(void *o,int a,const void*b,size_t s){
  return set_lt_attribute(o,a,b,s,LT_MATMUL_DESC_SET_ATTRIBUTE,
                          "cublasLtMatmulDescSetAttribute");}
int cublasLtMatrixLayoutSetAttribute(void *o,int a,const void*b,size_t s){
  return set_lt_attribute(o,a,b,s,LT_MATRIX_LAYOUT_SET_ATTRIBUTE,
                          "cublasLtMatrixLayoutSetAttribute");}
int cublasLtMatmulPreferenceSetAttribute(void *o,int a,const void*b,size_t s){
  return set_lt_attribute(o,a,b,s,LT_PREFERENCE_SET_ATTRIBUTE,
                          "cublasLtMatmulPreferenceSetAttribute");}

int cublasLtMatrixLayoutCreate(void **layout,int type,uint64_t rows,
                               uint64_t columns,int64_t ld) {
  typedef int(*f)(void **,int,uint64_t,uint64_t,int64_t);int route=current_route();
  if(route<0){static f real;if(!real)real=(f)lt_symbol("cublasLtMatrixLayoutCreate");
    return real?real(layout,type,rows,columns,ld):13;}
  struct request q={0};struct response r;q.opcode=LT_MATRIX_LAYOUT_CREATE;
  q.a_type=type;q.rows=rows;q.columns=columns;q.leading_dimension=ld;
  int status=call_route(route,&q,&r);if(status)return status;
  *layout=(void *)(uintptr_t)r.handle;return remember_object(*layout,route,2,0);
}
int cublasLtMatmulPreferenceCreate(void **preference) {
  typedef int(*f)(void **);int route=current_route();
  if(route<0){static f real;if(!real)real=(f)lt_symbol("cublasLtMatmulPreferenceCreate");
    return real?real(preference):13;}
  struct request q={0};struct response r;q.opcode=LT_PREFERENCE_CREATE;
  int status=call_route(route,&q,&r);if(status)return status;
  *preference=(void *)(uintptr_t)r.handle;
  return remember_object(*preference,route,3,0);
}

int cublasLtMatmulAlgoGetHeuristic(void *handle,void *operation,void *al,
                                   void *bl,void *cl,void *dl,void *preference,
                                   int requested,void *results,int *returned) {
  typedef int(*f)(void*,void*,void*,void*,void*,void*,void*,int,void*,int*);
  pthread_mutex_lock(&mutex);struct object_entry *found=find_object(operation);
  if(!found){pthread_mutex_unlock(&mutex);static f real;
    if(!real)real=(f)lt_symbol("cublasLtMatmulAlgoGetHeuristic");
    return real?real(handle,operation,al,bl,cl,dl,preference,requested,results,returned):13;}
  struct object_entry entry=*found;pthread_mutex_unlock(&mutex);
  if(requested<=0||requested>8)return 7;struct request q={0};struct response r;
  q.opcode=LT_HEURISTIC;q.handle=(uint64_t)handle;q.descriptor=(uint64_t)operation;
  q.a_descriptor=(uint64_t)al;q.b_descriptor=(uint64_t)bl;
  q.c_descriptor=(uint64_t)cl;q.d_descriptor=(uint64_t)dl;
  q.preference=(uint64_t)preference;q.requested_algorithms=requested;
  int status=call_route(entry.route,&q,&r);if(status)return status;
  if(r.payload_size>768)return 13;memcpy(results,r.payload,r.payload_size);
  *returned=r.returned_algorithms;return 0;
}

int cublasLtMatmul(void *handle,void *operation,const void *alpha,const void *a,
                   void *al,const void *b,void *bl,const void *beta,const void*c,
                   void *cl,void*d,void*dl,const void*algorithm,void*workspace,
                   size_t workspace_size,void*stream) {
  typedef int(*f)(void*,void*,const void*,const void*,void*,const void*,void*,
                  const void*,const void*,void*,void*,void*,const void*,void*,
                  size_t,void*);
  pthread_mutex_lock(&mutex);struct object_entry *found=find_object(operation);
  if(!found){pthread_mutex_unlock(&mutex);static f real;
    if(!real)real=(f)lt_symbol("cublasLtMatmul");
    return real?real(handle,operation,alpha,a,al,b,bl,beta,c,cl,d,dl,algorithm,
                     workspace,workspace_size,stream):13;}
  struct object_entry entry=*found;pthread_mutex_unlock(&mutex);
  size_t scalar=scalar_size_for_compute_type(entry.compute_type);
  struct request q={0};struct response r;q.opcode=LT_MATMUL;
  q.handle=(uint64_t)handle;q.descriptor=(uint64_t)operation;
  q.a=(uint64_t)a;q.b=(uint64_t)b;q.c=(uint64_t)c;q.workspace=(uint64_t)workspace;
  q.workspace_size=workspace_size;q.stream=(uint64_t)stream;
  q.a_descriptor=(uint64_t)al;q.b_descriptor=(uint64_t)bl;
  q.c_descriptor=(uint64_t)cl;q.d_descriptor=(uint64_t)dl;q.d=(uint64_t)d;
  q.scalar_size=scalar;q.pointer_mode=entry.pointer_mode;
  q.value=0x100; /* Presence marker; zero remains legacy "both present". */
  if(entry.pointer_mode==0){
    if(alpha){q.value|=1;memcpy(q.alpha_data,alpha,scalar);}
  }else{
    q.preference=(uint64_t)(uintptr_t)alpha;
  }
  if(entry.pointer_mode==0||entry.pointer_mode==4){
    if(beta){q.value|=2;memcpy(q.beta_data,beta,scalar);}
  }else{
    q.rows=(uint64_t)(uintptr_t)beta;
  }
  if(algorithm){q.payload_size=64;memcpy(q.payload,algorithm,64);}
  TRACE("lt_matmul route=%d",entry.route);return call_route(entry.route,&q,&r);
}
