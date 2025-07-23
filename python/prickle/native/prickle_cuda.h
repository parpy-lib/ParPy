#pragma once

#include <cstdio>

#define prickle_cuda_check_error(e) \
  do { \
    if (e != cudaSuccess) { \
      prickle_cuda::error_message = cudaGetErrorString(e); \
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
