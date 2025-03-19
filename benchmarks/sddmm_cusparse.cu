#include <cstdint>
#include <cusparse.h>

cusparseHandle_t handle;
const float alpha = 1.0;
const float beta = 0.0;
cusparseSpMatDescr_t A;
cusparseConstDnMatDescr_t C;
cusparseConstDnMatDescr_t D;
size_t *ext_buffer;

extern "C"
void cusparse_init_handle() {
  cusparseCreate(&handle);
}

extern "C"
int sddmm_init(
  int64_t *A_row_indices, int64_t *A_col_indices, float *A_values,
  float *C_data, float *D_data, int64_t N, int64_t M, int64_t K, int64_t nnz
) {
  cusparseCreateCsr(
    &A, N, M, nnz, A_row_indices, A_col_indices, A_values, CUSPARSE_INDEX_64I,
    CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
  );
  cusparseCreateConstDnMat(&C, N, K, K, C_data, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseCreateConstDnMat(&D, K, M, M, D_data, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  size_t buffer_size;
  cusparseSDDMM_bufferSize(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, C, D, &beta, A, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size
  );
  cudaError_t err = cudaMalloc(&ext_buffer, buffer_size);
  if (err == cudaErrorMemoryAllocation) {
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(C);
    cusparseDestroyDnMat(D);
    return 1;
  }
  cusparseSDDMM_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, C, D, &beta, A, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, ext_buffer
  );
  return 0;
}

extern "C"
void sddmm_deinit() {
  cusparseDestroySpMat(A);
  cusparseDestroyDnMat(C);
  cusparseDestroyDnMat(D);
  cudaFree(ext_buffer);
}

__global__
void custom_inplace_elemwise_mul(float *A, float *B, int64_t nnz) {
  int64_t i = blockIdx.x * 1024 + threadIdx.x;
  if (i < nnz) {
    A[i] *= B[i];
  }
}

extern "C"
int sddmm(float *A_values, float *B_values, int64_t nnz) {
  cusparseSDDMM(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, C, D, &beta, A, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, ext_buffer
  );
  const int tpb = 1024;
  const int blocks = (nnz + tpb - 1) / tpb;
  custom_inplace_elemwise_mul<<<blocks, tpb>>>(A_values, B_values, nnz);
  return 0;
}
