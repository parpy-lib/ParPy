#include "parpy_cuda.h"

extern "C" void parpy_init(int64_t _x) {
  return;
}

extern "C" int32_t parpy_sync() {
  return cudaDeviceSynchronize();
}

extern "C" void *parpy_alloc_buffer(int64_t nbytes) {
  void *ptr;
  if (cudaMallocAsync(&ptr, nbytes, 0) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

extern "C" int32_t parpy_memcpy(void *dst, void *src, int64_t nbytes, int64_t k) {
  cudaError_t err;
  if (k == 0) {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToHost, 0);
  } else if (k == 1) {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, 0);
  } else if (k == 2) {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, 0);
  } else {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, 0);
  }
  if (err != cudaSuccess) {
    return 1;
  }
  return 0;
}

extern "C" int32_t parpy_free_buffer(void *p) {
  cudaError_t err = cudaFreeAsync(p, 0);
  if (err != cudaSuccess) {
    return 1;
  }
  return 0;
}

extern "C" const char *parpy_get_error_message() {
  cudaError_t err = cudaGetLastError();
  return cudaGetErrorString(err);
}
