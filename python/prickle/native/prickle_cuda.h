#pragma once

#include <cstdio>

#define prickle_cuda_check_error(expr) \
  do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
      prickle_cuda::error_message = cudaGetErrorString(err); \
      return 1; \
    } \
  } while (0)

namespace prickle_cuda {
  const char *error_message = nullptr;
}

extern "C"
const char *prickle_get_error_message() {
  return prickle_cuda::error_message;
}
