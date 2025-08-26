#pragma once

#include <cstdio>

#define parpy_cuda_check_error(expr) \
  do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
      return 1; \
    } \
  } while (0)

// Functions used by the ParPy library when initializing, synchronizing with
// running GPU code, and operating on buffers.
extern "C" void parpy_init(int64_t);
extern "C" int32_t parpy_sync();
extern "C" void *parpy_alloc_buffer(int64_t);
extern "C" int32_t parpy_memcpy(void*, void*, int64_t, int64_t);
extern "C" int32_t parpy_free_buffer(void*);
extern "C" const char *parpy_get_error_message();
