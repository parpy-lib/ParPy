#pragma once

#include <cstdio>

#define parpy_cuda_check_error(expr) \
  do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
      parpy_cuda::error_message = cudaGetErrorString(err); \
      return 1; \
    } \
  } while (0)

namespace parpy_cuda {
  const char *error_message = nullptr;
}

extern "C"
const char *parpy_get_error_message() {
  return parpy_cuda::error_message;
}
