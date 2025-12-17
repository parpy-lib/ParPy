#pragma once

#include <cstdio>
#include <map>
#include <sstream>
#include <string>

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
  std::string error_message;

  // Maintain a cache of the properties per device. This ensures we do not
  // repeatedly load device properties, as this API call seems to consistently
  // take in the order of milliseconds. Also, it avoids potential bugs in code
  // that run ParPy functions on GPUs with different properties.
  std::map<int, cudaDeviceProp> device_properties;
  cudaDeviceProp get_device_properties(int dev) {
    auto it = device_properties.find(dev);
    if (it == device_properties.end()) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, dev);
      device_properties[dev] = prop;
    }
    return it->second;
  }

  int32_t check_shared_memory_usage(uint64_t nbytes) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop = get_device_properties(dev);
    if (nbytes > prop.sharedMemPerBlock) {
      std::ostringstream ss;
      ss << "Insufficient shared memory. ";
      ss << "Kernels of the function use up to " << nbytes;
      ss << " bytes of shared memory per block, while the current device";
      ss << " only supports up to " << prop.sharedMemPerBlock << " bytes.";
      parpy_cuda::error_message = ss.str();
      return 1;
    } else {
      return 0;
    }
  }
}

extern "C" const char *parpy_get_error_message() {
  if (parpy_cuda::error_message.empty()) {
    cudaError_t err = cudaPeekAtLastError();
    return cudaGetErrorString(err);
  } else {
    return parpy_cuda::error_message.c_str();
  }
}
