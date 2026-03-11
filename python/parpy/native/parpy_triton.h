#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string>

#define parpy_triton_check_error(expr) \
  do { \
    CUresult result = (expr); \
    if (result != CUDA_SUCCESS) { \
      cuGetErrorString(result, &parpy_triton::error_message); \
      return 1; \
    } \
  } while (0)

#define parpy_triton_check_non_null(expr) \
  do { \
    if (!expr) { \
      return 1; \
    } \
  } while (0)

namespace parpy_triton {
  const char *error_message = NULL;

  CUfunction load_kernel(const char *cache_path, const char *kernel_id) {
    std::string path = std::string(cache_path) + "/" + kernel_id + ".cubin";
    CUmodule mod;
    CUresult result;
    result = cuModuleLoad(&mod, path.c_str());
    if (result != CUDA_SUCCESS) {
      cuGetErrorString(result, &parpy_triton::error_message);
      return NULL;
    }

    CUfunction kernel;
    result = cuModuleGetFunction(&kernel, mod, kernel_id);
    if (result != CUDA_SUCCESS) {
      cuGetErrorString(result, &parpy_triton::error_message);
      return NULL;
    }
    return kernel;
  }
}

const char *parpy_get_error_message() {
  return parpy_triton::error_message;
}
