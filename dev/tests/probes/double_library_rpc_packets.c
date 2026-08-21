#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void *handle_t;

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

static uint32_t expected_opcode;
static unsigned calls;

int lupine_cuda_current_route_id(void) { return 0; }
int cuCtxGetCurrent(void **context) {
  *context = (void *)(uintptr_t)0xabc;
  return 0;
}

int lupine_cublas_call_on_route(int route, const struct request *request,
                                struct response *response) {
  assert(route == 0);
  assert(request->opcode == expected_opcode);
  memset(response, 0, sizeof(*response));
  if (request->opcode == 1 || request->opcode == 10 || request->opcode == 22)
    response->handle = 0x1000 + request->opcode;
  if (request->opcode == 71) {
    const double result = 7.0;
    memcpy(response->payload, &result, sizeof(result));
    response->payload_size = sizeof(result);
  }
  if (request->opcode == 134 || request->opcode == 135) {
    const double result = 7.0;
    memcpy(response->payload, &result, sizeof(result));
    response->payload_size = sizeof(result);
  }
  if (request->opcode == 137 || request->opcode == 138) {
    const int result = 2;
    memcpy(response->payload, &result, sizeof(result));
    response->payload_size = sizeof(result);
  }
  if (request->opcode == 101 || request->opcode == 104 ||
      request->opcode == 106 || request->opcode == 108 ||
      request->opcode == 110 || request->opcode == 118 ||
      request->opcode == 121 || request->opcode == 123 ||
      request->opcode == 125 || request->opcode == 129 ||
      request->opcode == 141 || request->opcode == 143 ||
      request->opcode == 145)
    response->value = 64;
  if (request->opcode == 8) {
    double alpha, beta;
    memcpy(&alpha, request->alpha_data, sizeof(alpha));
    memcpy(&beta, request->beta_data, sizeof(beta));
    assert(request->a_type == 1 && request->b_type == 1);
    assert(request->c_type == 1 && request->compute_type == 70);
    assert(request->scalar_size == sizeof(double));
    assert(alpha == 2.0 && beta == 3.0);
  }
  if (request->opcode == 112 || request->opcode == 115 ||
      request->opcode == 117 ||
      request->opcode == 116) {
    double alpha;
    memcpy(&alpha, request->alpha_data, sizeof(alpha));
    assert(alpha == 2.0);
    if (request->opcode != 112 && request->opcode != 117) {
      double beta;
      memcpy(&beta, request->beta_data, sizeof(beta));
      assert(beta == 3.0);
    }
  }
  if (request->opcode == 131 || request->opcode == 133) {
    double alpha;
    memcpy(&alpha, request->alpha_data, sizeof(alpha));
    assert(alpha == 2.0);
  }
  if (request->opcode == 147 || request->opcode == 148) {
    assert(request->scalar_size == (request->opcode == 147 ? 8u : 16u));
  }
  if (request->opcode == 20) {
    assert(request->scalar_size == sizeof(float));
    assert(request->value == 0x100);
    if (request->pointer_mode == 3) {
      assert(request->preference == 0x6000);
      assert(request->rows == 0);
    } else {
      assert(request->pointer_mode == 0);
    }
  }
  ++calls;
  return 0;
}

int cublasCreate_v2(handle_t *);
int cublasDestroy_v2(handle_t);
int cublasSetPointerMode_v2(handle_t, int);
int cublasDgemm_v2(handle_t, int, int, int, int, int, const double *,
                   const double *, int, const double *, int, const double *,
                   double *, int);
int cublasDgemmBatched(handle_t, int, int, int, int, int, const double *,
                       const double *const[], int, const double *const[], int,
                       const double *, double *const[], int, int);
int cublasDgemv_v2(handle_t, int, int, int, const double *, const double *,
                   int, const double *, int, const double *, double *, int);
int cublasDdot_v2(handle_t, int, const double *, int, const double *, int,
                  double *);
int cublasDtrsmBatched(handle_t, int, int, int, int, int, int,
                       const double *, const double *const[], int,
                       double *const[], int, int);
int cublasDtrsm_v2(handle_t, int, int, int, int, int, int, const double *,
                    const double *, int, double *, int);
int cublasDgetrsBatched(handle_t, int, int, int, const double *const[], int,
                        const int *, double *const[], int, int *, int);
int cublasDgetrfBatched(handle_t, int, double *[], int, int *, int *, int);
int cublasSgetrfBatched(handle_t, int, float *[], int, int *, int *, int);
int cublasSgelsBatched(handle_t, int, int, int, int, float *const[], int,
                       float *const[], int, int *, int *, int);
int cublasDgelsBatched(handle_t, int, int, int, int, double *const[], int,
                       double *const[], int, int *, int *, int);
int cublasCgelsBatched(handle_t, int, int, int, int, void *const[], int,
                       void *const[], int, int *, int *, int);
int cublasZgelsBatched(handle_t, int, int, int, int, void *const[], int,
                       void *const[], int, int *, int *, int);
int cublasSgeqrfBatched(handle_t, int, int, float *const[], int,
                        float *const[], int *, int);
int cublasDgeqrfBatched(handle_t, int, int, double *const[], int,
                        double *const[], int *, int);
int cublasCgeqrfBatched(handle_t, int, int, void *const[], int,
                        void *const[], int *, int);
int cublasZgeqrfBatched(handle_t, int, int, void *const[], int,
                        void *const[], int *, int);
int cublasDaxpy_v2(handle_t, int, const double *, const double *, int, double *,
                    int);
int cublasDcopy_v2(handle_t, int, const double *, int, double *, int);
int cublasDscal_v2(handle_t, int, const double *, double *, int);
int cublasDnrm2_v2(handle_t, int, const double *, int, double *);
int cublasDasum_v2(handle_t, int, const double *, int, double *);
int cublasDswap_v2(handle_t, int, double *, int, double *, int);
int cublasIdamax_v2(handle_t, int, const double *, int, int *);
int cublasIdamin_v2(handle_t, int, const double *, int, int *);
int cublasCgemmStridedBatched(handle_t, int, int, int, int, int,
                              const void *, const void *, int, int64_t,
                              const void *, int, int64_t, const void *, void *,
                              int, int64_t, int);
int cublasCgemv_v2(handle_t, int, int, int, const void *, const void *, int,
                    const void *, int, const void *, void *, int);
int cublasZgemv_v2(handle_t, int, int, int, const void *, const void *, int,
                    const void *, int, const void *, void *, int);
int cublasLtMatmulDescCreate(void **, int, int);
int cublasLtMatmulDescDestroy(void *);
int cublasLtMatmulDescSetAttribute(void *, int, const void *, size_t);
int cublasLtMatmul(void *, void *, const void *, const void *, void *,
                   const void *, void *, const void *, const void *, void *,
                   void *, void *, const void *, void *, size_t, void *);

int cusolverDnCreate(handle_t *);
int cusolverDnDestroy(handle_t);
int cusolverDnDgetrf_bufferSize(handle_t, int, int, double *, int, int *);
int cusolverDnDgetrf(handle_t, int, int, double *, int, double *, int *, int *);
int cusolverDnDgetrs(handle_t, int, int, int, const double *, int,
                     const int *, double *, int, int *);
int cusolverDnDgesvdj_bufferSize(handle_t, int, int, int, int,
                                 const double *, int, const double *,
                                 const double *, int, const double *, int,
                                 int *, handle_t);
int cusolverDnDgesvdj(handle_t, int, int, int, int, double *, int, double *,
                      double *, int, double *, int, double *, int, int *,
                      handle_t);
int cusolverDnDgesvdjBatched_bufferSize(handle_t, int, int, int,
                                        const double *, int, const double *,
                                        const double *, int, const double *,
                                        int, int *, handle_t, int);
int cusolverDnDgesvdjBatched(handle_t, int, int, int, double *, int, double *,
                             double *, int, double *, int, double *, int,
                             int *, handle_t, int);
int cusolverDnDorgqr_bufferSize(handle_t, int, int, int, const double *, int,
                                const double *, int *);
int cusolverDnDorgqr(handle_t, int, int, int, double *, int, const double *,
                     double *, int, int *);
int cusolverDnDormqr_bufferSize(handle_t, int, int, int, int, int,
                                const double *, int, const double *,
                                const double *, int, int *);
int cusolverDnDormqr(handle_t, int, int, int, int, int, const double *, int,
                     const double *, double *, int, double *, int, int *);
int cusolverDnDpotrf_bufferSize(handle_t, int, int, double *, int, int *);
int cusolverDnDpotrf(handle_t, int, int, double *, int, double *, int, int *);
int cusolverDnDpotrs(handle_t, int, int, int, const double *, int, double *,
                     int, int *);
int cusolverDnDgeqrf_bufferSize(handle_t, int, int, double *, int, int *);
int cusolverDnDgeqrf(handle_t, int, int, double *, int, double *, double *, int,
                     int *);
int cusolverDnDgesvd_bufferSize(handle_t, int, int, int *);
int cusolverDnDgesvd(handle_t, signed char, signed char, int, int, double *, int,
                     double *, double *, int, double *, int, double *, int,
                     double *, int *);
int cusolverDnDsyevd_bufferSize(handle_t, int, int, int, const double *, int,
                                const double *, int *);
int cusolverDnDsyevd(handle_t, int, int, int, double *, int, double *, double *,
                     int, int *);
int cusolverDnDpotrfBatched(handle_t, int, int, double *[], int, int *, int);
int cusolverDnSpotrfBatched(handle_t, int, int, float *[], int, int *, int);
int cusolverDnCpotrfBatched(handle_t, int, int, void *[], int, int *, int);
int cusolverDnZpotrfBatched(handle_t, int, int, void *[], int, int *, int);
int cusolverDnDpotrsBatched(handle_t, int, int, int, const double *[], int,
                            double *[], int, int *, int);
int cusolverDnDsyevjBatched_bufferSize(handle_t, int, int, int,
                                       const double *, int, const double *,
                                       int *, handle_t, int);
int cusolverDnDsyevjBatched(handle_t, int, int, int, double *, int, double *,
                            double *, int, int *, handle_t, int);
int cusolverDnZsytrf_bufferSize(handle_t, int, void *, int, int *);
int cusolverDnZsytrf(handle_t, int, int, void *, int, int *, void *, int,
                     int *);
int cusolverDnCsytrf_bufferSize(handle_t, int, void *, int, int *);
int cusolverDnCsytrf(handle_t, int, int, void *, int, int *, void *, int,
                     int *);
int cusolverDnDsytrf_bufferSize(handle_t, int, double *, int, int *);
int cusolverDnDsytrf(handle_t, int, int, double *, int, int *, double *, int,
                     int *);

#define EXPECT(opcode, expression) do { expected_opcode = (opcode); \
  assert((expression) == 0); } while (0)

int main(void) {
  handle_t blas = 0, solver = 0, params = (void *)(uintptr_t)0x44;
  double alpha = 2.0, beta = 3.0, scalar = 0.0;
  double *a = (double *)(uintptr_t)0x2000;
  double *b = (double *)(uintptr_t)0x3000;
  double *c = (double *)(uintptr_t)0x4000;
  double *workspace = (double *)(uintptr_t)0x5000;
  double *arrays[] = {a};
  float *float_arrays[] = {(float *)(uintptr_t)0x6000};
  void *complex_arrays[] = {a};
  const double *const_arrays[] = {a};
  int pivots = 0, info = 0, lwork = 0;

  EXPECT(1, cublasCreate_v2(&blas));
  EXPECT(8, cublasDgemm_v2(blas, 0, 0, 2, 2, 2, &alpha, a, 2, b, 2,
                            &beta, c, 2));
  EXPECT(115, cublasDgemmBatched(blas, 0, 0, 2, 2, 2, &alpha, const_arrays,
                                  2, const_arrays, 2, &beta, arrays, 2, 1));
  EXPECT(116, cublasDgemv_v2(blas, 0, 2, 2, &alpha, a, 2, b, 1, &beta, c,
                              1));
  EXPECT(71, cublasDdot_v2(blas, 2, a, 1, b, 1, &scalar));
  assert(scalar == 7.0);
  EXPECT(112, cublasDtrsmBatched(blas, 0, 0, 0, 0, 2, 2, &alpha,
                                  const_arrays, 2, arrays, 2, 1));
  EXPECT(117, cublasDtrsm_v2(blas, 0, 0, 0, 0, 2, 2, &alpha, a, 2, b, 2));
  EXPECT(113, cublasDgetrsBatched(blas, 0, 2, 1, const_arrays, 2, &pivots,
                                   arrays, 2, &info, 1));
  EXPECT(114, cublasDgetrfBatched(blas, 2, arrays, 2, &pivots, &info, 1));
  EXPECT(140, cublasSgetrfBatched(blas, 2, float_arrays, 2, &pivots, &info, 1));
  EXPECT(149, cublasSgelsBatched(blas, 0, 2, 2, 1, float_arrays, 2,
                                 float_arrays, 2, &info, &info, 1));
  EXPECT(150, cublasDgelsBatched(blas, 0, 2, 2, 1, arrays, 2, arrays, 2,
                                 &info, &info, 1));
  EXPECT(151, cublasCgelsBatched(blas, 0, 2, 2, 1, complex_arrays, 2,
                                 complex_arrays, 2, &info, &info, 1));
  EXPECT(152, cublasZgelsBatched(blas, 0, 2, 2, 1, complex_arrays, 2,
                                 complex_arrays, 2, &info, &info, 1));
  EXPECT(153, cublasSgeqrfBatched(blas, 2, 2, float_arrays, 2, float_arrays,
                                  &info, 1));
  EXPECT(154, cublasDgeqrfBatched(blas, 2, 2, arrays, 2, arrays, &info, 1));
  EXPECT(155, cublasCgeqrfBatched(blas, 2, 2, complex_arrays, 2,
                                  complex_arrays, &info, 1));
  EXPECT(156, cublasZgeqrfBatched(blas, 2, 2, complex_arrays, 2,
                                  complex_arrays, &info, 1));
  {
    void *descriptor = 0;
    EXPECT(10, cublasLtMatmulDescCreate(&descriptor, 68, 0));
    EXPECT(20, cublasLtMatmul(blas, descriptor, 0, a, 0, b, 0, 0, c, 0, c,
                              0, 0, 0, 0, 0));
    int pointer_mode = 3;
    EXPECT(12, cublasLtMatmulDescSetAttribute(
                   descriptor, 2, &pointer_mode, sizeof(pointer_mode)));
    EXPECT(20, cublasLtMatmul(blas, descriptor,
                              (void *)(uintptr_t)0x6000, a, 0, b, 0, 0, c, 0,
                              c, 0, 0, 0, 0, 0));
    EXPECT(11, cublasLtMatmulDescDestroy(descriptor));
  }
  EXPECT(131, cublasDaxpy_v2(blas, 2, &alpha, a, 1, b, 1));
  EXPECT(132, cublasDcopy_v2(blas, 2, a, 1, b, 1));
  EXPECT(133, cublasDscal_v2(blas, 2, &alpha, a, 1));
  EXPECT(134, cublasDnrm2_v2(blas, 2, a, 1, &scalar));
  assert(scalar == 7.0);
  assert(cublasSetPointerMode_v2(blas, 1) == 0);
  scalar = 11.0;
  EXPECT(134, cublasDnrm2_v2(blas, 2, a, 1, &scalar));
  assert(scalar == 11.0);
  assert(cublasSetPointerMode_v2(blas, 0) == 0);
  scalar = 0.0;
  EXPECT(135, cublasDasum_v2(blas, 2, a, 1, &scalar));
  assert(scalar == 7.0);
  EXPECT(136, cublasDswap_v2(blas, 2, a, 1, b, 1));
  info = 0;
  EXPECT(137, cublasIdamax_v2(blas, 2, a, 1, &info));
  assert(info == 2);
  info = 0;
  EXPECT(138, cublasIdamin_v2(blas, 2, a, 1, &info));
  assert(info == 2);
  {
    float complex_alpha[2] = {1.0f, 0.0f};
    float complex_beta[2] = {0.0f, 0.0f};
    EXPECT(139, cublasCgemmStridedBatched(
                    blas, 0, 0, 2, 2, 2, complex_alpha, a, 2, 4, b, 2, 4,
                    complex_beta, c, 2, 4, 1));
    EXPECT(147, cublasCgemv_v2(blas, 0, 2, 2, complex_alpha, a, 2, b, 1,
                                complex_beta, c, 1));
  }
  {
    double complex_alpha[2] = {1.0, 0.0};
    double complex_beta[2] = {0.0, 0.0};
    EXPECT(148, cublasZgemv_v2(blas, 0, 2, 2, complex_alpha, a, 2, b, 1,
                                complex_beta, c, 1));
  }
  EXPECT(2, cublasDestroy_v2(blas));

  EXPECT(22, cusolverDnCreate(&solver));
  EXPECT(101, cusolverDnDgetrf_bufferSize(solver, 2, 2, a, 2, &lwork));
  assert(lwork == 64);
  EXPECT(102, cusolverDnDgetrf(solver, 2, 2, a, 2, workspace, &pivots, &info));
  EXPECT(103, cusolverDnDgetrs(solver, 0, 2, 1, a, 2, &pivots, b, 2, &info));
  EXPECT(104, cusolverDnDgesvdj_bufferSize(solver, 0, 0, 2, 2, a, 2, b, c,
                                            2, a, 2, &lwork, params));
  EXPECT(105, cusolverDnDgesvdj(solver, 0, 0, 2, 2, a, 2, b, c, 2, a, 2,
                                workspace, 64, &info, params));
  EXPECT(106, cusolverDnDgesvdjBatched_bufferSize(
                  solver, 0, 2, 2, a, 2, b, c, 2, a, 2, &lwork, params, 1));
  EXPECT(107, cusolverDnDgesvdjBatched(solver, 0, 2, 2, a, 2, b, c, 2, a, 2,
                                       workspace, 64, &info, params, 1));
  EXPECT(108, cusolverDnDorgqr_bufferSize(solver, 2, 2, 2, a, 2, b, &lwork));
  EXPECT(109, cusolverDnDorgqr(solver, 2, 2, 2, a, 2, b, workspace, 64,
                               &info));
  EXPECT(110, cusolverDnDormqr_bufferSize(solver, 0, 0, 2, 2, 2, a, 2, b, c,
                                           2, &lwork));
  EXPECT(111, cusolverDnDormqr(solver, 0, 0, 2, 2, 2, a, 2, b, c, 2,
                               workspace, 64, &info));
  EXPECT(118, cusolverDnDpotrf_bufferSize(solver, 0, 2, a, 2, &lwork));
  assert(lwork == 64);
  EXPECT(119, cusolverDnDpotrf(solver, 0, 2, a, 2, workspace, 64, &info));
  EXPECT(120, cusolverDnDpotrs(solver, 0, 2, 1, a, 2, b, 2, &info));
  EXPECT(121, cusolverDnDgeqrf_bufferSize(solver, 2, 2, a, 2, &lwork));
  EXPECT(122, cusolverDnDgeqrf(solver, 2, 2, a, 2, b, workspace, 64, &info));
  EXPECT(123, cusolverDnDgesvd_bufferSize(solver, 2, 2, &lwork));
  EXPECT(124, cusolverDnDgesvd(solver, 'A', 'A', 2, 2, a, 2, b, c, 2, a, 2,
                               workspace, 64, b, &info));
  EXPECT(125, cusolverDnDsyevd_bufferSize(solver, 1, 0, 2, a, 2, b, &lwork));
  EXPECT(126, cusolverDnDsyevd(solver, 1, 0, 2, a, 2, b, workspace, 64,
                               &info));
  EXPECT(127, cusolverDnDpotrfBatched(solver, 0, 2, arrays, 2, &info, 1));
  {
    float value = 1.0f;
    float *float_arrays[] = {&value};
    void *complex_arrays[] = {a};
    EXPECT(157, cusolverDnSpotrfBatched(solver, 0, 1, float_arrays, 1, &info,
                                        1));
    EXPECT(158, cusolverDnCpotrfBatched(solver, 0, 1, complex_arrays, 1, &info,
                                        1));
    EXPECT(159, cusolverDnZpotrfBatched(solver, 0, 1, complex_arrays, 1, &info,
                                        1));
  }
  EXPECT(128, cusolverDnDpotrsBatched(solver, 0, 2, 1, const_arrays, 2,
                                      arrays, 2, &info, 1));
  EXPECT(129, cusolverDnDsyevjBatched_bufferSize(solver, 1, 0, 2, a, 2, b,
                                                 &lwork, params, 1));
  EXPECT(130, cusolverDnDsyevjBatched(solver, 1, 0, 2, a, 2, b, workspace, 64,
                                      &info, params, 1));
  EXPECT(141, cusolverDnZsytrf_bufferSize(solver, 2, a, 2, &lwork));
  assert(lwork == 64);
  EXPECT(142, cusolverDnZsytrf(solver, 0, 2, a, 2, &pivots, workspace, 64,
                               &info));
  EXPECT(143, cusolverDnCsytrf_bufferSize(solver, 2, a, 2, &lwork));
  EXPECT(144, cusolverDnCsytrf(solver, 0, 2, a, 2, &pivots, workspace, 64,
                               &info));
  EXPECT(145, cusolverDnDsytrf_bufferSize(solver, 2, a, 2, &lwork));
  EXPECT(146, cusolverDnDsytrf(solver, 0, 2, a, 2, &pivots, workspace, 64,
                               &info));
  EXPECT(23, cusolverDnDestroy(solver));
  assert(calls == 71);
  puts("double-library RPC packet checks passed");
  return 0;
}
