#pragma once

#include <cstdio>
#include <cstdint>

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
extern "C" int32_t parpy_memset(void*, int64_t, int8_t);
extern "C" int32_t parpy_free_buffer(void*);
extern "C" int32_t parpy_ensure_sufficient_shared_memory(int64_t);

namespace parpy_cuda {
  char err_buf[1024];
  const char *error_message = nullptr;

  // The cudaGetDeviceProperties function is rather slow (each call takes
  // around one millisecond). Therefore, we manage a cache of these properties
  // per CUDA device. As the user can swap devices between calls to ParPy
  // functions, we track the properties on a per-device basis in case the GPUs
  // have different capabilities. We assume a hard limit of 16 CUDA devices on
  // one machine and that they are assigned indices in order, starting from
  // zero.
  bool device_cached[16] = {false};
  cudaDeviceProp device_properties[16];
  cudaDeviceProp *get_device_properties(int dev) {
    if (dev >= 16) {
      snprintf(parpy_cuda::err_buf, 1024,
          "Internal error fetching CUDA device properties. The device has "
          "index %d. ParPy internally assumes the number of GPUs is at most "
          "16, and that they are numbered in sequence.", dev);
      parpy_cuda::error_message = parpy_cuda::err_buf;
      return nullptr;
    }
    if (!device_cached[dev]) {
      cudaGetDeviceProperties(&device_properties[dev], dev);
      device_cached[dev] = true;
    }
    return &device_properties[dev];
  }

  int32_t check_shared_memory_usage(uint64_t nbytes) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp *prop = get_device_properties(dev);
    if (prop == nullptr) {
      return 1;
    }
    if (nbytes > prop->sharedMemPerBlock) {
      snprintf(parpy_cuda::err_buf, 1024,
          "Insufficient shared memory. Kernels of the function use up to %llu "
          "bytes of shared memory per block, while the current device only "
          "supports up to %llu bytes.", nbytes, prop->sharedMemPerBlock);
      parpy_cuda::error_message = parpy_cuda::err_buf;
      return 1;
    } else {
      return 0;
    }
  }
}

extern "C" const char *parpy_get_error_message() {
  if (parpy_cuda::error_message == nullptr) {
    cudaError_t err = cudaPeekAtLastError();
    return cudaGetErrorString(err);
  } else {
    return parpy_cuda::error_message;
  }
}
