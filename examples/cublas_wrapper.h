#include <cublas_v2.h>
#include <stdio.h>

void cublas_gemm_f64(
    int64_t M,
    int64_t N,
    int64_t K,
    double alpha,
    double *A,
    double *B,
    double beta,
    double *C
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
  cublasDestroy(handle);
}
